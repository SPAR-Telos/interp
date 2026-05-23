"""Pre-reasoning action-probe steering for held-out size15 non-optimal samples.

This runner injects the linear optimal-action probe's pairwise
optimal-vs-final-wrong direction into the three prompt-suffix activations used
by the pre-reasoning action probe, then regenerates the model's analysis and
final action with MLX. Evaluation is paired by sample/layer/seed against the
same hook path at alpha=0.

Use --clean-autonomous for the one-pass protocol: no second finalization prompt
and no heuristic tail-action fallback; only a naturally generated JSON action
counts as parsed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import struct
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flag_swap_suffix_utils import parse_index_specification, selected_positions_in_pass
from run_seed12_action_probes import ACTION_TO_IDX, ACTIONS, PositionSpec, load_position_activation
from telos_interp.activation_loading import discover_trajectory_folders

DEFAULT_TRAJECTORIES_DIR = Path("data/trajectories_train_single_step/size15")
DEFAULT_ACTIVATIONS_DIR = Path("data/activations/trajectories_train_single_step/size15")
DEFAULT_SOURCE_RUN = Path("results/size15_mistake_aligned_action_probes/20260521T192354Z")
DEFAULT_OUTPUT_ROOT = Path("results/size15_pre_reasoning_action_probe_steering")
DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q4"
DEFAULT_LAYERS = (7, 15, 23)
DEFAULT_ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_MAX_NEW_TOKENS = 8192
PRE_POSITION = PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix")
ACTION_RE = re.compile(r'"action"\s*:\s*"(LEFT|RIGHT|UP|DOWN)"', re.IGNORECASE)
ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass(frozen=True)
class PromptRow:
    """Tokenized prompt data needed by the MLX hook."""

    full_token_ids: list[int]
    prompt_len: int
    n_suffix: int
    generation_prefill_token_ids: list[int]


@dataclass(frozen=True)
class SteeringTarget:
    """One held-out non-optimal steering target."""

    sample: Size15SampleRecord
    original_action: str
    optimal_actions: tuple[str, ...]


@dataclass(frozen=True)
class DirectionBundle:
    """Raw activation-space direction for one sample/probe/layer."""

    w_margin: torch.Tensor
    b_margin: float
    scaler_mean: torch.Tensor
    scaler_std: torch.Tensor
    raw_direction: torch.Tensor
    margin_unit_delta: torch.Tensor
    raw_direction_norm: float


@dataclass(frozen=True)
class Size15TrajectoryRecord:
    """A size15 trajectory JSON paired with its activation folder."""

    name: str
    activation_folder: Path
    trajectory_path: Path
    trajectory: dict[str, Any]


@dataclass(frozen=True)
class Size15SampleRecord:
    """One single-step size15 trajectory sample used by the steering run."""

    name: str
    trajectory_path: Path
    activation_folder: Path
    step: dict[str, Any]
    state: tuple[int, int, bool, bool]
    state_key: str
    state_index: int
    config_type: str
    action: str
    action_idx: int
    astar_actions: tuple[str, ...]
    astar_indices: tuple[int, ...]
    is_optimal_action: bool
    sample_index: int
    grid_cells: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class OracleActionInfo:
    """BFS shortest-path oracle data for one rendered grid."""

    agent_position: tuple[int, int]
    goal_position: tuple[int, int]
    optimal_actions: tuple[str, ...]
    shortest_distance: int | None


@dataclass
class TrainedActionProbe:
    """A loaded linear action probe plus normalization tensors."""

    model: torch.nn.Module
    scaler_mean: torch.Tensor
    scaler_std: torch.Tensor
    metrics: dict[str, Any]
    split: dict[str, Any]
    config: dict[str, Any]


class MlxPromptAddHook:
    """Add a per-position direction to selected prompt positions in one MLX layer."""

    def __init__(self, parent_list: list[Any], idx: int):
        self.parent_list = parent_list
        self.idx = idx
        self.original_layer = parent_list[idx]
        self.mode = "passthrough"
        self.direction = None
        self.scale = 0.0
        self.prompt_length: int | None = None
        self.selected_positions: list[int] = []
        self.prompt_cursor = 0
        self.captured_by_row: dict[int, Any] = {}
        self.fired = False
        self.seq_lens: list[int] = []
        self.touched_positions: list[int] = []
        self.patched_positions: list[int] = []
        self.pre_norm: float | None = None
        self.post_norm: float | None = None
        parent_list[idx] = self

    def __call__(self, x, mask, cache, *args, **kwargs):
        import mlx.core as mx

        out = self.original_layer(x, mask, cache, *args, **kwargs)
        if self.prompt_length is None:
            return out

        seq_len = int(out.shape[1])
        touched = selected_positions_in_pass(
            prompt_cursor=self.prompt_cursor,
            seq_len=seq_len,
            prompt_length=self.prompt_length,
            selected_positions=self.selected_positions,
        )
        if self.prompt_cursor < self.prompt_length:
            self.prompt_cursor += min(seq_len, self.prompt_length - self.prompt_cursor)
        if not touched:
            return out

        self.fired = True
        self.seq_lens.append(seq_len)
        self.touched_positions.extend(abs_pos for _, _, abs_pos in touched)
        if self.mode == "capture":
            for local_idx, row_idx, _ in touched:
                self.captured_by_row[row_idx] = out[0, local_idx, :].astype(mx.float32)
        elif self.mode == "add":
            if self.direction is None:
                raise RuntimeError("direction must be set for add mode")
            expected = (len(self.selected_positions), out.shape[2])
            if self.direction.shape[0] != expected[0] or self.direction.shape[1] != expected[1]:
                raise RuntimeError(f"direction shape {self.direction.shape}, expected {expected}")
            out = self._add_touched(out, touched)
        return out

    def _add_touched(self, out, touched: list[tuple[int, int, int]]):
        import mlx.core as mx

        before = mx.concatenate([out[:, local_idx : local_idx + 1, :] for local_idx, _, _ in touched], axis=1)
        self.pre_norm = float(mx.linalg.norm(before.astype(mx.float32)))
        pieces = []
        cursor = 0
        changed = []
        for local_idx, row_idx, abs_pos in touched:
            if local_idx > cursor:
                pieces.append(out[:, cursor:local_idx, :])
            perturb = (self.direction[row_idx : row_idx + 1, :] * self.scale).astype(out.dtype)
            tail = out[:, local_idx : local_idx + 1, :] + mx.broadcast_to(
                perturb[None, :, :], (out.shape[0], 1, out.shape[2])
            )
            pieces.append(tail)
            changed.append(tail.astype(mx.float32))
            self.patched_positions.append(abs_pos)
            cursor = local_idx + 1
        if cursor < out.shape[1]:
            pieces.append(out[:, cursor:, :])
        after = mx.concatenate(changed, axis=1)
        self.post_norm = float(mx.linalg.norm(after))
        return mx.concatenate(pieces, axis=1)

    def configure_prompt(self, prompt_row: PromptRow, selected_positions: list[int]) -> None:
        self.reset()
        self.prompt_length = prompt_row.prompt_len
        self.selected_positions = list(selected_positions)

    def captured_tensor(self):
        import mlx.core as mx

        missing = [pos for row_idx, pos in enumerate(self.selected_positions) if row_idx not in self.captured_by_row]
        if missing:
            raise RuntimeError(f"layer {self.idx} missing captured prompt positions: {missing}")
        return mx.stack([self.captured_by_row[i] for i in range(len(self.selected_positions))], axis=0)

    def reset(self) -> None:
        self.mode = "passthrough"
        self.direction = None
        self.scale = 0.0
        self.prompt_length = None
        self.selected_positions = []
        self.prompt_cursor = 0
        self.captured_by_row = {}
        self.fired = False
        self.seq_lens = []
        self.touched_positions = []
        self.patched_positions = []
        self.pre_norm = None
        self.post_norm = None

    def log(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "seq_lens": self.seq_lens,
            "mode": self.mode,
            "scale": self.scale,
            "prompt_len": self.prompt_length,
            "selected_prompt_positions": self.selected_positions,
            "touched_prompt_positions": sorted(set(self.touched_positions)),
            "patched_prompt_positions": sorted(set(self.patched_positions)),
            "pre_norm": self.pre_norm,
            "post_norm": self.post_norm,
        }

    def restore(self) -> None:
        self.parent_list[self.idx] = self.original_layer


def parse_int_selection(value: str) -> list[int]:
    """Parse comma-separated integer selections with inclusive ranges."""
    out: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            start_raw, end_raw = part.split(":", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if start <= end else -1
            values = range(start, end + step, step)
        else:
            values = [int(part)]
        for item in values:
            if item not in seen:
                out.append(item)
                seen.add(item)
    if not out:
        raise ValueError("empty integer selection")
    return out


def parse_float_selection(value: str) -> list[float]:
    """Parse comma-separated floats."""
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("empty float selection")
    return out


def parse_rendered_grid_state(grid_state: Any, *, expected_size: int | None = None) -> tuple[tuple[str, ...], ...]:
    """Parse coordinate-labeled rendered grid rows into a rectangular matrix."""
    if not isinstance(grid_state, list):
        raise ValueError("grid_state must be a list of rendered rows")
    rows: list[tuple[str, ...]] = []
    for row_raw in grid_state:
        if not isinstance(row_raw, str):
            continue
        parts = row_raw.split()
        if len(parts) < 2:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        cells = tuple(parts[1:])
        if expected_size is not None and len(cells) != expected_size:
            continue
        rows.append(cells)
    if not rows:
        raise ValueError("No rendered grid rows found")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("Rendered grid rows are not rectangular")
    if expected_size is not None and (len(rows) != expected_size or width != expected_size):
        raise ValueError(f"Expected {expected_size}x{expected_size} grid, got {len(rows)}x{width}")
    return tuple(rows)


def find_symbol(grid_cells: tuple[tuple[str, ...], ...], symbol: str) -> tuple[int, int]:
    """Return the unique coordinate for a rendered grid symbol."""
    positions = [
        (row_idx, col_idx)
        for row_idx, row in enumerate(grid_cells)
        for col_idx, cell in enumerate(row)
        if cell == symbol
    ]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one {symbol!r} cell, found {len(positions)}")
    return positions[0]


def canonical_rendered_grid(grid_cells: tuple[tuple[str, ...], ...]) -> str:
    """Return a stable text representation of a rendered grid."""
    return "\n".join(" ".join(row) for row in grid_cells)


def exact_rendered_state_key(grid_cells: tuple[tuple[str, ...], ...], has_key: bool, door_open: bool) -> str:
    """Return the exact rendered-grid split key used by the size15 experiments."""
    payload = json.dumps(
        {
            "grid": canonical_rendered_grid(grid_cells),
            "carrying_key": bool(has_key),
            "door_open": bool(door_open),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "rendered_sha256=" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_size15_trajectory_records(
    activations_dir: Path,
    trajectories_dir: Path,
    max_trajectories: int | None,
) -> tuple[list[Size15TrajectoryRecord], dict[str, int]]:
    """Load size15 trajectory JSONs paired with activation folders."""
    skipped: Counter[str] = Counter()
    records: list[Size15TrajectoryRecord] = []
    if activations_dir.exists():
        trajectory_names = [folder.name for folder in discover_trajectory_folders(activations_dir)]
    else:
        trajectory_names = [path.stem for path in sorted(trajectories_dir.glob("*.json"))]
        skipped["missing_activations_dir_using_trajectory_jsons"] += 1
    if max_trajectories is not None:
        trajectory_names = trajectory_names[:max_trajectories]

    for trajectory_name in trajectory_names:
        trajectory_path = trajectories_dir / f"{trajectory_name}.json"
        if not trajectory_path.exists():
            skipped["missing_trajectory_json"] += 1
            continue
        try:
            trajectory = json.loads(trajectory_path.read_text())
        except json.JSONDecodeError:
            skipped["invalid_trajectory_json"] += 1
            continue
        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
            skipped["invalid_trajectory_schema"] += 1
            continue
        records.append(
            Size15TrajectoryRecord(
                name=trajectory_name,
                activation_folder=activations_dir / trajectory_name,
                trajectory_path=trajectory_path,
                trajectory=trajectory,
            )
        )
    return records, dict(skipped)


def collect_size15_samples(records: list[Size15TrajectoryRecord], grid_size: int) -> list[Size15SampleRecord]:
    """Collect valid single-step steering samples from size15 records."""
    samples: list[Size15SampleRecord] = []
    skipped: Counter[str] = Counter()
    for record in records:
        if not record.trajectory.get("steps"):
            skipped["missing_steps"] += 1
            continue
        step = dict(record.trajectory["steps"][0])
        step.setdefault("prompt", record.trajectory.get("prompt", {}))

        try:
            grid_cells = parse_rendered_grid_state(step.get("grid_state"), expected_size=grid_size)
            row_idx, col_idx = find_symbol(grid_cells, "A")
        except ValueError:
            skipped["invalid_grid_state"] += 1
            continue

        action = str(step.get("agent_action", "")).upper()
        has_key = bool(step.get("carrying_key"))
        door_open = bool(step.get("door_open"))
        state_key = exact_rendered_state_key(grid_cells, has_key, door_open)
        samples.append(
            Size15SampleRecord(
                name=record.name,
                trajectory_path=record.trajectory_path,
                activation_folder=record.activation_folder,
                step=step,
                state=(row_idx, col_idx, has_key, door_open),
                state_key=state_key,
                state_index=-1,
                config_type="exact_rendered_grid",
                action=action,
                action_idx=ACTION_TO_IDX.get(action, -1),
                astar_actions=(),
                astar_indices=(),
                is_optimal_action=True,
                sample_index=len(samples),
                grid_cells=grid_cells,
            )
        )

    if skipped:
        print(f"Sample collection skipped: {dict(skipped)}")
    state_index_by_key = {
        state_key: idx for idx, state_key in enumerate(sorted({sample.state_key for sample in samples}))
    }
    return [
        Size15SampleRecord(
            name=sample.name,
            trajectory_path=sample.trajectory_path,
            activation_folder=sample.activation_folder,
            step=sample.step,
            state=sample.state,
            state_key=sample.state_key,
            state_index=state_index_by_key[sample.state_key],
            config_type=sample.config_type,
            action=sample.action,
            action_idx=sample.action_idx,
            astar_actions=sample.astar_actions,
            astar_indices=sample.astar_indices,
            is_optimal_action=sample.is_optimal_action,
            sample_index=sample.sample_index,
            grid_cells=sample.grid_cells,
        )
        for sample in samples
    ]


def bfs_distances_to_goal(
    grid_cells: tuple[tuple[str, ...], ...], goal_position: tuple[int, int]
) -> dict[tuple[int, int], int]:
    """Return shortest distances from every reachable cell to the goal."""
    queue = [goal_position]
    distances = {goal_position: 0}
    height = len(grid_cells)
    width = len(grid_cells[0])
    head = 0
    while head < len(queue):
        row_idx, col_idx = queue[head]
        head += 1
        for d_row, d_col in ACTION_DELTAS.values():
            next_pos = (row_idx + d_row, col_idx + d_col)
            nr, nc = next_pos
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if grid_cells[nr][nc] == "#":
                continue
            if next_pos in distances:
                continue
            distances[next_pos] = distances[(row_idx, col_idx)] + 1
            queue.append(next_pos)
    return distances


def compute_oracle_action_info(grid_cells: tuple[tuple[str, ...], ...]) -> OracleActionInfo:
    """Compute BFS optimal next actions from the rendered grid."""
    agent_position = find_symbol(grid_cells, "A")
    goal_position = find_symbol(grid_cells, "G")
    distances = bfs_distances_to_goal(grid_cells, goal_position)
    if agent_position not in distances:
        return OracleActionInfo(agent_position, goal_position, (), None)
    start_distance = distances[agent_position]
    optimal_actions = []
    for action in ACTIONS:
        d_row, d_col = ACTION_DELTAS[action]
        next_pos = (agent_position[0] + d_row, agent_position[1] + d_col)
        if distances.get(next_pos) == start_distance - 1:
            optimal_actions.append(action)
    return OracleActionInfo(agent_position, goal_position, tuple(optimal_actions), start_distance)


def token_ids_from_trace_tokens(tokens: list[dict[str, Any]]) -> list[int]:
    """Extract token ids from stored trajectory token rows."""
    return [int(token["token_id"]) for token in tokens]


def analysis_channel_prefill_token_ids(sample: Size15SampleRecord) -> list[int]:
    """Return the standard generated analysis-channel prefix token ids when present."""
    output_tokens = sample.step.get("output_tokens", [])
    if not isinstance(output_tokens, list) or len(output_tokens) < 3:
        return []
    expected = ["<|channel|>", "analysis", "<|message|>"]
    prefix = output_tokens[:3]
    if [str(token.get("token", "")) for token in prefix] != expected:
        return []
    return token_ids_from_trace_tokens(prefix)


def reconstruct_prompt_row(sample: Size15SampleRecord, *, prefill_analysis_channel: bool = False) -> PromptRow:
    """Reconstruct the exact prompt token ids for a single-step size15 sample."""
    prompt = sample.step.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError(f"sample {sample.name} is missing prompt metadata")
    prefix = prompt.get("prompt_prefix_tokens")
    grid = sample.step.get("grid_state_tokens")
    suffix = sample.step.get("prompt_suffix_tokens", prompt.get("prompt_suffix_tokens"))
    if not isinstance(prefix, list) or not isinstance(grid, list) or not isinstance(suffix, list):
        raise ValueError(f"sample {sample.name} has invalid prompt token metadata")
    prompt_token_ids = token_ids_from_trace_tokens(prefix + grid + suffix)
    generation_prefill_token_ids = analysis_channel_prefill_token_ids(sample) if prefill_analysis_channel else []
    return PromptRow(
        full_token_ids=prompt_token_ids + generation_prefill_token_ids,
        prompt_len=len(prompt_token_ids),
        n_suffix=len(suffix),
        generation_prefill_token_ids=generation_prefill_token_ids,
    )


def resolve_steering_suffix_positions(
    prompt_row: PromptRow, suffix_spec: str = "-3:-1"
) -> tuple[list[int], list[int]]:
    """Resolve suffix-relative steering indices to absolute prompt positions."""
    rel_indices = parse_index_specification(suffix_spec, prompt_row.n_suffix)
    if not rel_indices:
        raise ValueError(f"{suffix_spec!r} selected no suffix positions")
    start = prompt_row.prompt_len - prompt_row.n_suffix
    return rel_indices, [start + rel_idx for rel_idx in rel_indices]


def parse_generated_action(text: str) -> str | None:
    """Parse an action from generated text."""
    match = ACTION_RE.search(text)
    if match:
        return match.group(1).upper()
    tail = text[-300:].upper()
    found = [(tail.rfind(action), action) for action in ACTIONS if tail.rfind(action) >= 0]
    if not found:
        return None
    return max(found)[1]


def parse_json_action(text: str) -> str | None:
    """Parse an action only from a generated JSON action field."""
    match = ACTION_RE.search(text)
    if match:
        return match.group(1).upper()
    return None


def classify_generated_action_text(
    text: str,
    *,
    finalization_text: str = "",
    tail_action_fallback: bool = True,
) -> tuple[str | None, str | None, str]:
    """Classify generated text into a parsed action and parse status."""
    source_text = finalization_text or text
    generated_action = parse_json_action(source_text)
    tail_fallback_action = parse_generated_action(source_text) if tail_action_fallback else None
    if generated_action in ACTION_TO_IDX:
        return generated_action, tail_fallback_action, "json_action"
    if tail_fallback_action in ACTION_TO_IDX:
        return tail_fallback_action, tail_fallback_action, "tail_fallback"
    return None, tail_fallback_action, "parse_fail"


def load_split_by_state(source_run: Path) -> dict[str, str]:
    """Load exact state split mapping from the prior action-probe run."""
    split_manifest = json.loads((source_run / "split_manifest.json").read_text())
    states = split_manifest.get("states")
    if not isinstance(states, dict):
        raise ValueError(f"split manifest missing states: {source_run / 'split_manifest.json'}")
    return {str(key): str(value) for key, value in states.items()}


def select_test_nonoptimal_targets(
    samples: list[Size15SampleRecord],
    split_by_state: dict[str, str],
    max_samples: int | None = None,
) -> list[SteeringTarget]:
    """Select held-out test samples whose original final action is not BFS-optimal."""
    targets: list[SteeringTarget] = []
    for sample in sorted(samples, key=lambda item: item.name):
        if split_by_state.get(sample.state_key) != "test":
            continue
        oracle = compute_oracle_action_info(sample.grid_cells)
        original_action = str(sample.step.get("agent_action", "")).upper()
        if original_action not in ACTION_TO_IDX:
            continue
        if not oracle.optimal_actions or original_action in set(oracle.optimal_actions):
            continue
        targets.append(
            SteeringTarget(sample=sample, original_action=original_action, optimal_actions=oracle.optimal_actions)
        )
        if max_samples is not None and len(targets) >= max_samples:
            break
    return targets


def safe_std(std: torch.Tensor) -> torch.Tensor:
    """Return finite, nonzero normalization std values."""
    return torch.where(torch.isfinite(std) & (std > 1e-8), std, torch.ones_like(std))


def pairwise_raw_direction_from_weights(
    weights: torch.Tensor,
    scaler_std: torch.Tensor,
    optimal_actions: tuple[str, ...],
    wrong_action: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized-space margin vector and raw-space direction."""
    if wrong_action not in ACTION_TO_IDX:
        raise ValueError(f"invalid wrong action: {wrong_action!r}")
    if not optimal_actions:
        raise ValueError("optimal_actions must be non-empty")
    optimal_indices = [ACTION_TO_IDX[action] for action in optimal_actions]
    wrong_idx = ACTION_TO_IDX[wrong_action]
    w_margin = weights[optimal_indices].mean(dim=0) - weights[wrong_idx]
    std = safe_std(scaler_std)
    return w_margin, w_margin / std


def margin_unit_delta(raw_direction: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return a raw-space delta that changes the pairwise probe margin by one unit."""
    denom = float(torch.dot(raw_direction, raw_direction).item()) + eps
    return raw_direction / denom


def pairwise_margin_shift(w_margin: torch.Tensor, scaler_std: torch.Tensor, delta_raw: torch.Tensor) -> float:
    """Return the pairwise normalized probe-margin change caused by a raw-space delta."""
    return float(torch.dot(w_margin, delta_raw / safe_std(scaler_std)).item())


def load_probe_checkpoint(path: Path) -> TrainedActionProbe:
    """Load a saved linear action probe checkpoint."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    config = dict(payload["config"])
    model_type = str(config.get("model_type"))
    if model_type != "linear":
        raise ValueError(f"steering requires a linear probe checkpoint, got model_type={model_type!r}: {path}")
    state_dict = payload["model_state_dict"]
    weight = state_dict["weight"]
    model = torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
    model.load_state_dict(state_dict)
    model.eval()
    return TrainedActionProbe(
        model=model.cpu(),
        scaler_mean=payload["scaler_mean"].detach().cpu().float(),
        scaler_std=payload["scaler_std"].detach().cpu().float(),
        metrics=dict(payload.get("metrics", {})),
        split=dict(payload.get("split", {})),
        config=config,
    )


def build_direction_bundle(
    probe_checkpoint: Path, optimal_actions: tuple[str, ...], wrong_action: str
) -> DirectionBundle:
    """Load a linear probe and build the sample-specific pairwise steering direction."""
    probe = load_probe_checkpoint(probe_checkpoint)
    if str(probe.config.get("model_type")) != "linear":
        raise ValueError(f"steering requires a linear probe: {probe_checkpoint}")
    linear = probe.model
    if not hasattr(linear, "weight") or not hasattr(linear, "bias"):
        raise ValueError(f"probe is not a plain linear model: {probe_checkpoint}")
    weights = linear.weight.detach().cpu().float()
    bias = linear.bias.detach().cpu().float()
    w_margin, raw_direction = pairwise_raw_direction_from_weights(
        weights=weights,
        scaler_std=probe.scaler_std,
        optimal_actions=optimal_actions,
        wrong_action=wrong_action,
    )
    optimal_indices = [ACTION_TO_IDX[action] for action in optimal_actions]
    b_margin = float(bias[optimal_indices].mean().item() - bias[ACTION_TO_IDX[wrong_action]].item())
    unit_delta = margin_unit_delta(raw_direction)
    return DirectionBundle(
        w_margin=w_margin,
        b_margin=b_margin,
        scaler_mean=probe.scaler_mean,
        scaler_std=safe_std(probe.scaler_std),
        raw_direction=raw_direction,
        margin_unit_delta=unit_delta,
        raw_direction_norm=float(torch.linalg.norm(raw_direction).item()),
    )


def probe_pairwise_margin(bundle: DirectionBundle, raw_activation: torch.Tensor) -> float:
    """Compute the optimal-vs-wrong pairwise margin for one raw activation."""
    normalized = (raw_activation.float() - bundle.scaler_mean) / bundle.scaler_std
    return float(torch.dot(bundle.w_margin, normalized).item() + bundle.b_margin)


def get_mlx_layer_list(model: Any) -> list[Any]:
    """Return the mutable MLX transformer layer list."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Could not find MLX model layers")


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    sampler: Any,
    max_tokens: int,
    *,
    stop_on_action: bool = True,
) -> tuple[str, bool]:
    """Generate one completion using MLX-LM.

    Stopping after a generated JSON action does not force the action; it only
    avoids spending tokens after the measured final answer has appeared.
    """
    from mlx_lm import stream_generate

    pieces = []
    for chunk in stream_generate(model, tokenizer, prompt=prompt_token_ids, max_tokens=max_tokens, sampler=sampler):
        pieces.append(chunk.text)
        text = "".join(pieces)
        if stop_on_action and ACTION_RE.search(text):
            return text, True
    return "".join(pieces), False


def final_channel_prefill_token_ids(sample: Size15SampleRecord) -> list[int]:
    """Return the standard assistant final-channel token ids from a stored trajectory."""
    output_tokens = sample.step.get("output_tokens", [])
    if not isinstance(output_tokens, list):
        return []
    expected = ["<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>"]
    for start_idx in range(0, max(0, len(output_tokens) - len(expected) + 1)):
        candidate = output_tokens[start_idx : start_idx + len(expected)]
        if [str(token.get("token", "")) for token in candidate] == expected:
            return token_ids_from_trace_tokens(candidate)
    return []


def build_finalization_prompt_token_ids(
    prompt_row: PromptRow,
    tokenizer: Any,
    generated_cot_text: str,
    final_prefill_token_ids: list[int],
) -> list[int]:
    """Build a prompt for final JSON readout from the generated CoT."""
    cot_token_ids = tokenizer.encode(generated_cot_text)
    return prompt_row.full_token_ids + cot_token_ids + final_prefill_token_ids


def run_calibration(
    model: Any,
    tokenizer: Any,
    targets: list[SteeringTarget],
    layers: list[int],
    source_run: Path,
    prompt_suffix_indices: str,
    prefill_analysis_channel: bool,
    max_samples: int,
) -> dict[str, Any]:
    """Compare live MLX captured prompt-suffix activations with saved PyTorch activations."""
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler

    calibration_targets = targets[: max(0, max_samples)]
    layer_list = get_mlx_layer_list(model)
    rows = []
    by_layer: dict[int, list[dict[str, float]]] = defaultdict(list)
    for layer in layers:
        checkpoint_path = (
            source_run / "checkpoints" / f"probe_optimal_action_pre_reasoning_suffix_layer{layer}_linear.pt"
        )
        hook = MlxPromptAddHook(layer_list, layer)
        try:
            for target in calibration_targets:
                prompt_row = reconstruct_prompt_row(
                    target.sample,
                    prefill_analysis_channel=prefill_analysis_channel,
                )
                _, suffix_positions = resolve_steering_suffix_positions(prompt_row, prompt_suffix_indices)
                hook.configure_prompt(prompt_row, suffix_positions)
                hook.mode = "capture"
                mx.random.seed(0)
                sampler = make_sampler(temp=0.0)
                _, _ = generate_one(
                    model,
                    tokenizer,
                    prompt_row.full_token_ids,
                    sampler,
                    max_tokens=1,
                    stop_on_action=False,
                )
                live = np.array(hook.captured_tensor().astype(mx.float32)).mean(axis=0)
                saved, reason = load_position_activation(
                    trajectory_folder=target.sample.activation_folder,
                    layer_idx=layer,
                    step_idx=int(target.sample.step.get("step_id", 0)),
                    position=PRE_POSITION,
                    output_tokens=target.sample.step.get("output_tokens", []),
                )
                if saved is None:
                    rows.append(
                        {
                            "layer": layer,
                            "trajectory_name": target.sample.name,
                            "status": "missing_saved_activation",
                            "reason": reason,
                        }
                    )
                    continue
                saved_np = saved.detach().cpu().float().numpy()
                live_t = torch.tensor(live, dtype=torch.float32)
                bundle = build_direction_bundle(checkpoint_path, target.optimal_actions, target.original_action)
                saved_margin = probe_pairwise_margin(bundle, saved.float())
                live_margin = probe_pairwise_margin(bundle, live_t)
                cosine = float(np.dot(saved_np, live) / ((np.linalg.norm(saved_np) * np.linalg.norm(live)) + 1e-12))
                norm_ratio = float(np.linalg.norm(live) / (np.linalg.norm(saved_np) + 1e-12))
                row = {
                    "layer": layer,
                    "trajectory_name": target.sample.name,
                    "status": "ok",
                    "cosine_similarity": cosine,
                    "norm_ratio_live_over_saved": norm_ratio,
                    "saved_pairwise_margin": saved_margin,
                    "live_pairwise_margin": live_margin,
                }
                rows.append(row)
                by_layer[layer].append(row)
        finally:
            hook.restore()

    summaries = []
    for layer in layers:
        layer_rows = by_layer.get(layer, [])
        if not layer_rows:
            summaries.append({"layer": layer, "n": 0})
            continue
        saved_margins = np.array([row["saved_pairwise_margin"] for row in layer_rows], dtype=np.float64)
        live_margins = np.array([row["live_pairwise_margin"] for row in layer_rows], dtype=np.float64)
        if len(layer_rows) > 1 and np.std(saved_margins) > 0 and np.std(live_margins) > 0:
            corr = float(np.corrcoef(saved_margins, live_margins)[0, 1])
        else:
            corr = math.nan
        summaries.append(
            {
                "layer": layer,
                "n": len(layer_rows),
                "mean_cosine_similarity": float(np.mean([row["cosine_similarity"] for row in layer_rows])),
                "mean_norm_ratio_live_over_saved": float(
                    np.mean([row["norm_ratio_live_over_saved"] for row in layer_rows])
                ),
                "probe_margin_correlation": corr,
            }
        )
    return {"rows": rows, "summary_by_layer": summaries}


def run_steering_generation(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    """Run the MLX steering generations and return flat row dictionaries."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    records, skipped = load_size15_trajectory_records(args.activations_dir, args.trajectories_dir, None)
    samples = collect_size15_samples(records, args.grid_size)
    split_by_state = load_split_by_state(args.source_run)
    targets = select_test_nonoptimal_targets(samples, split_by_state, args.max_samples)
    if not targets:
        raise RuntimeError("No held-out non-optimal targets selected")

    print(f"Selected {len(targets)} held-out non-optimal targets; skipped records={skipped}", flush=True)
    print(f"Loading MLX model {args.model_id} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model_id)
    print(f"Loaded model in {time.time() - t0:.1f}s", flush=True)

    calibration = {"skipped": True}
    if not args.skip_calibration and args.calibration_samples > 0:
        calibration = run_calibration(
            model=model,
            tokenizer=tokenizer,
            targets=targets,
            layers=args.layers,
            source_run=args.source_run,
            prompt_suffix_indices=args.prompt_suffix_indices,
            prefill_analysis_channel=args.prefill_analysis_channel,
            max_samples=args.calibration_samples,
        )
    write_json(output_dir / "mlx_probe_calibration.json", calibration)

    rows: list[dict[str, Any]] = []
    jsonl_path = output_dir / "steering_rows.jsonl"
    layer_list = get_mlx_layer_list(model)
    with jsonl_path.open("w") as fout:
        for layer_idx, layer in enumerate(args.layers):
            checkpoint_path = (
                args.source_run / "checkpoints" / (f"probe_optimal_action_pre_reasoning_suffix_layer{layer}_linear.pt")
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"missing probe checkpoint: {checkpoint_path}")
            hook = MlxPromptAddHook(layer_list, layer)
            try:
                for sample_idx, target in enumerate(targets):
                    prompt_row = reconstruct_prompt_row(
                        target.sample,
                        prefill_analysis_channel=args.prefill_analysis_channel,
                    )
                    suffix_rel, suffix_positions = resolve_steering_suffix_positions(
                        prompt_row, args.prompt_suffix_indices
                    )
                    direction_bundle = build_direction_bundle(
                        checkpoint_path, target.optimal_actions, target.original_action
                    )
                    unit_delta = direction_bundle.margin_unit_delta.detach().cpu().numpy().astype(np.float32)
                    direction = mx.array(np.stack([unit_delta for _ in suffix_positions], axis=0))
                    for seed_idx in range(args.n_seeds):
                        generation_seed = args.seed + layer_idx * 1_000_000 + sample_idx * 10_000 + seed_idx
                        for alpha in args.alphas:
                            hook.configure_prompt(prompt_row, suffix_positions)
                            hook.mode = "add"
                            hook.direction = direction
                            hook.scale = float(alpha)
                            mx.random.seed(generation_seed)
                            sampler = make_sampler(temp=args.temperature)
                            text, stopped_on_action_regex = generate_one(
                                model,
                                tokenizer,
                                prompt_row.full_token_ids,
                                sampler,
                                args.max_new_tokens,
                                stop_on_action=True,
                            )
                            generated_action = parse_json_action(text)
                            finalization_used = False
                            finalization_text = ""
                            finalization_stopped_on_action_regex = False
                            if generated_action is None and args.finalization_fallback:
                                final_prefill = final_channel_prefill_token_ids(target.sample)
                                final_prompt_ids = build_finalization_prompt_token_ids(
                                    prompt_row,
                                    tokenizer,
                                    text,
                                    final_prefill,
                                )
                                mx.random.seed(generation_seed + 9_000_000)
                                final_sampler = make_sampler(temp=args.temperature)
                                finalization_text, finalization_stopped_on_action_regex = generate_one(
                                    model,
                                    tokenizer,
                                    final_prompt_ids,
                                    final_sampler,
                                    args.finalization_max_new_tokens,
                                    stop_on_action=True,
                                )
                                generated_action = parse_json_action(finalization_text)
                                finalization_used = True
                            generated_action, tail_fallback_action, parse_status = classify_generated_action_text(
                                text,
                                finalization_text=finalization_text,
                                tail_action_fallback=args.tail_action_fallback,
                            )
                            generated_optimal = generated_action in set(target.optimal_actions)
                            row = {
                                "trajectory_name": target.sample.name,
                                "sample_index": target.sample.sample_index,
                                "state_key": target.sample.state_key,
                                "split": "test",
                                "layer": layer,
                                "alpha": float(alpha),
                                "seed_index": seed_idx,
                                "generation_seed": generation_seed,
                                "original_action": target.original_action,
                                "optimal_actions": list(target.optimal_actions),
                                "generated_action": generated_action,
                                "tail_fallback_action": tail_fallback_action,
                                "generated_action_is_optimal": bool(generated_optimal),
                                "action_changed_from_original": generated_action != target.original_action,
                                "parse_status": parse_status,
                                "generated_text_n_chars": len(text),
                                "stopped_on_action_regex": bool(stopped_on_action_regex),
                                "finalization_fallback_used": bool(finalization_used),
                                "tail_action_fallback_enabled": bool(args.tail_action_fallback),
                                "clean_autonomous": bool(args.clean_autonomous),
                                "generation_protocol": args.generation_protocol,
                                "finalization_text_n_chars": len(finalization_text),
                                "finalization_stopped_on_action_regex": bool(finalization_stopped_on_action_regex),
                                "finalization_text_tail": finalization_text[-500:],
                                "generated_text_tail": text[-1000:],
                                "prompt_suffix_indices": args.prompt_suffix_indices,
                                "prefill_analysis_channel": bool(args.prefill_analysis_channel),
                                "generation_prefill_token_ids": prompt_row.generation_prefill_token_ids,
                                "suffix_relative_indices": suffix_rel,
                                "selected_prompt_positions": suffix_positions,
                                "direction_raw_norm": direction_bundle.raw_direction_norm,
                                "intended_margin_shift": pairwise_margin_shift(
                                    direction_bundle.w_margin,
                                    direction_bundle.scaler_std,
                                    direction_bundle.margin_unit_delta * float(alpha),
                                ),
                                "hook_log": hook.log(),
                            }
                            fout.write(json.dumps(row) + "\n")
                            fout.flush()
                            rows.append(row)
                    if (sample_idx + 1) % 10 == 0:
                        print(
                            f"layer={layer} completed {sample_idx + 1}/{len(targets)} targets rows={len(rows)}",
                            flush=True,
                        )
            finally:
                hook.restore()
    return rows


def json_ready(value: Any) -> Any:
    """Convert nested values to JSON-friendly primitives."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON."""
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat-ish row dictionaries to CSV."""
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def summarize_steering_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize raw steering outcomes per layer/alpha."""
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), float(row["alpha"]))].append(row)
    summaries = []
    for (layer, alpha), group_rows in sorted(grouped.items()):
        n = len(group_rows)
        parse_fail_n = sum(1 for row in group_rows if row["parse_status"] == "parse_fail")
        optimal_n = sum(1 for row in group_rows if bool(row["generated_action_is_optimal"]))
        changed_n = sum(1 for row in group_rows if bool(row["action_changed_from_original"]))
        counts = Counter(str(row["generated_action"] or "PARSE_FAIL") for row in group_rows)
        summaries.append(
            {
                "layer": layer,
                "alpha": alpha,
                "n": n,
                "optimal_n": optimal_n,
                "optimal_rate": optimal_n / n if n else math.nan,
                "action_changed_from_original_n": changed_n,
                "action_changed_from_original_rate": changed_n / n if n else math.nan,
                "parse_fail_n": parse_fail_n,
                "parse_fail_rate": parse_fail_n / n if n else math.nan,
                "per_action_counts": dict(sorted(counts.items())),
            }
        )
    return summaries


def build_paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair every nonzero alpha row with alpha=0 for the same sample/layer/seed."""
    by_key = {
        (
            str(row["trajectory_name"]),
            int(row["layer"]),
            int(row["seed_index"]),
            int(row["generation_seed"]),
            float(row["alpha"]),
        ): row
        for row in rows
    }
    paired = []
    for row in rows:
        alpha = float(row["alpha"])
        if alpha == 0.0:
            continue
        baseline = by_key.get(
            (
                str(row["trajectory_name"]),
                int(row["layer"]),
                int(row["seed_index"]),
                int(row["generation_seed"]),
                0.0,
            )
        )
        if baseline is None:
            continue
        baseline_optimal = bool(baseline["generated_action_is_optimal"])
        steered_optimal = bool(row["generated_action_is_optimal"])
        paired.append(
            {
                "trajectory_name": row["trajectory_name"],
                "sample_index": row["sample_index"],
                "state_key": row["state_key"],
                "layer": int(row["layer"]),
                "alpha": alpha,
                "seed_index": int(row["seed_index"]),
                "generation_seed": int(row["generation_seed"]),
                "original_action": row["original_action"],
                "optimal_actions": row["optimal_actions"],
                "baseline_generated_action": baseline["generated_action"],
                "steered_generated_action": row["generated_action"],
                "baseline_parse_status": baseline["parse_status"],
                "steered_parse_status": row["parse_status"],
                "baseline_generated_action_is_optimal": baseline_optimal,
                "steered_generated_action_is_optimal": steered_optimal,
                "delta_optimal": int(steered_optimal) - int(baseline_optimal),
                "action_changed_vs_baseline": baseline["generated_action"] != row["generated_action"],
            }
        )
    return paired


def bootstrap_ci(values: list[float], n_bootstrap: int = 20_000, seed: int = 12) -> tuple[float, float]:
    """Return percentile bootstrap 95% CI for the mean."""
    if not values:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    indices = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    means = arr[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def binom_tail_ge(k: int, n: int) -> float:
    """Exact binomial tail P[X >= k], X~Binomial(n, 0.5)."""
    if n <= 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def paired_stats_by_layer_alpha(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute paired steering deltas and McNemar-style tests."""
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(int(row["layer"]), float(row["alpha"]))].append(row)
    stats = []
    for (layer, alpha), group_rows in sorted(grouped.items()):
        deltas = [float(row["delta_optimal"]) for row in group_rows]
        ci_seed = 17 + layer * 1000 + int(round((alpha + 1000.0) * 100))
        ci_low, ci_high = bootstrap_ci(deltas, seed=ci_seed)
        gain_n = sum(
            1
            for row in group_rows
            if not bool(row["baseline_generated_action_is_optimal"])
            and bool(row["steered_generated_action_is_optimal"])
        )
        drop_n = sum(
            1
            for row in group_rows
            if bool(row["baseline_generated_action_is_optimal"])
            and not bool(row["steered_generated_action_is_optimal"])
        )
        unchanged_success_n = sum(
            1
            for row in group_rows
            if bool(row["baseline_generated_action_is_optimal"]) and bool(row["steered_generated_action_is_optimal"])
        )
        unchanged_failure_n = sum(
            1
            for row in group_rows
            if not bool(row["baseline_generated_action_is_optimal"])
            and not bool(row["steered_generated_action_is_optimal"])
        )
        discordant_n = gain_n + drop_n
        p_gain = binom_tail_ge(gain_n, discordant_n) if gain_n >= drop_n else 1.0
        p_two = min(1.0, 2.0 * binom_tail_ge(max(gain_n, drop_n), discordant_n)) if discordant_n else 1.0
        n = len(group_rows)
        baseline_optimal_n = sum(1 for row in group_rows if bool(row["baseline_generated_action_is_optimal"]))
        steered_optimal_n = sum(1 for row in group_rows if bool(row["steered_generated_action_is_optimal"]))
        stats.append(
            {
                "layer": layer,
                "alpha": alpha,
                "n": n,
                "baseline_optimal_rate": baseline_optimal_n / n if n else math.nan,
                "steered_optimal_rate": steered_optimal_n / n if n else math.nan,
                "delta_optimal_rate": sum(deltas) / n if n else math.nan,
                "delta_optimal_rate_bootstrap_ci95_low": ci_low,
                "delta_optimal_rate_bootstrap_ci95_high": ci_high,
                "mcnemar_gain_baseline_wrong_to_steered_optimal_n": gain_n,
                "mcnemar_drop_baseline_optimal_to_steered_wrong_n": drop_n,
                "mcnemar_unchanged_success_n": unchanged_success_n,
                "mcnemar_unchanged_failure_n": unchanged_failure_n,
                "mcnemar_discordant_n": discordant_n,
                "mcnemar_one_sided_p_gain": p_gain,
                "mcnemar_two_sided_p": p_two,
            }
        )
    return stats


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB PNG without external dependencies."""
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].astype(np.uint8).tobytes() for row in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def write_simple_line_png(path: Path, summaries: list[dict[str, Any]], y_key: str, layers: list[int]) -> None:
    """Write a simple raster plot fallback."""
    width, height = 1200, 360
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    panel_w = 300
    left = 80
    top = 45
    panel_h = 230
    colors = {7: np.array([0, 114, 178]), 15: np.array([213, 94, 0]), 23: np.array([0, 158, 115])}
    alphas = sorted({float(row["alpha"]) for row in summaries})
    if not alphas:
        write_rgb_png(path, image)
        return
    min_alpha, max_alpha = min(alphas), max(alphas)
    by_key = {(int(row["layer"]), float(row["alpha"])): float(row[y_key]) for row in summaries}
    for idx, layer in enumerate(layers):
        px = left + idx * (panel_w + 65)
        image[top : top + panel_h + 1, px : px + 2] = 30
        image[top + panel_h : top + panel_h + 2, px : px + panel_w] = 30
        points = []
        for alpha in alphas:
            denom = max_alpha - min_alpha if max_alpha > min_alpha else 1.0
            x = int(px + (alpha - min_alpha) / denom * panel_w)
            y_val = by_key.get((layer, alpha), math.nan)
            if not math.isfinite(y_val):
                continue
            if "delta" in y_key:
                y_plot = (y_val + 0.5) / 1.0
            else:
                y_plot = y_val
            y_plot = max(0.0, min(1.0, y_plot))
            y = int(top + panel_h - y_plot * panel_h)
            points.append((x, y))
            image[max(0, y - 4) : min(height, y + 5), max(0, x - 4) : min(width, x + 5)] = colors.get(layer, 0)
        for (x1, y1), (x2, y2) in zip(points, points[1:], strict=False):
            steps = max(abs(x2 - x1), abs(y2 - y1), 1)
            for step in range(steps + 1):
                x = int(round(x1 + (x2 - x1) * step / steps))
                y = int(round(y1 + (y2 - y1) * step / steps))
                image[max(0, y - 2) : min(height, y + 3), max(0, x - 2) : min(width, x + 3)] = colors.get(layer, 0)
    write_rgb_png(path, image)


def write_svg_plot(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    y_key: str,
    title: str,
    y_label: str,
    layers: list[int],
) -> None:
    """Write a paper-style SVG line plot with one panel per layer."""
    width, height = 1500, 520
    left, top, panel_w, panel_h, gap = 90, 105, 360, 300, 70
    alphas = sorted({float(row["alpha"]) for row in summaries})
    by_key = {(int(row["layer"]), float(row["alpha"])): float(row[y_key]) for row in summaries}
    min_alpha, max_alpha = (min(alphas), max(alphas)) if alphas else (-1.0, 1.0)
    y_min, y_max = (-0.5, 0.5) if "delta" in y_key else (0.0, 1.0)

    def sx(px: float, alpha: float) -> float:
        denom = max_alpha - min_alpha if max_alpha > min_alpha else 1.0
        return px + (alpha - min_alpha) / denom * panel_w

    def sy(value: float) -> float:
        return top + panel_h - (value - y_min) / (y_max - y_min) * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000px" height="347px" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="42" font-family="Arial, Helvetica, sans-serif" font-size="24" '
        f'font-weight="700" text-anchor="middle">{title}</text>',
    ]
    for layer_idx, layer in enumerate(layers):
        px = left + layer_idx * (panel_w + gap)
        parts.append(
            f'<text x="{px + panel_w / 2}" y="{top - 20}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="17" font-weight="700" text-anchor="middle">Layer {layer}</text>'
        )
        for tick in np.linspace(y_min, y_max, 6):
            y = sy(float(tick))
            parts.append(f'<line x1="{px}" y1="{y:.2f}" x2="{px + panel_w}" y2="{y:.2f}" stroke="#e8e8e8"/>')
            if layer_idx == 0:
                parts.append(
                    f'<text x="{px - 12}" y="{y + 4:.2f}" font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="12" text-anchor="end">{tick:.1f}</text>'
                )
        if "delta" in y_key:
            y0 = sy(0.0)
            parts.append(f'<line x1="{px}" y1="{y0:.2f}" x2="{px + panel_w}" y2="{y0:.2f}" stroke="#555"/>')
        parts.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{top + panel_h}" stroke="#333"/>')
        parts.append(f'<line x1="{px}" y1="{top + panel_h}" x2="{px + panel_w}" y2="{top + panel_h}" stroke="#333"/>')
        points = []
        for alpha in alphas:
            value = by_key.get((layer, alpha), math.nan)
            if not math.isfinite(value):
                continue
            points.append((sx(px, alpha), sy(value)))
            parts.append(
                f'<text x="{sx(px, alpha):.2f}" y="{top + panel_h + 24}" '
                'font-family="Arial, Helvetica, sans-serif" font-size="11" text-anchor="middle">'
                f"{alpha:g}</text>"
            )
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline fill="none" stroke="#0072B2" stroke-width="4" stroke-linecap="round" '
            f'stroke-linejoin="round" points="{point_text}"/>'
        )
        for x, y in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="white" stroke="#0072B2" stroke-width="2"/>')
    parts.append(
        f'<text x="{width / 2}" y="{height - 48}" font-family="Arial, Helvetica, sans-serif" '
        'font-size="14" text-anchor="middle">alpha; positive steers toward BFS-optimal set over original wrong action</text>'
    )
    parts.append(
        f'<text x="30" y="{top + panel_h / 2}" font-family="Arial, Helvetica, sans-serif" font-size="14" '
        f'text-anchor="middle" transform="rotate(-90 30 {top + panel_h / 2})">{y_label}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def write_plots(
    output_dir: Path, summaries: list[dict[str, Any]], paired_stats: list[dict[str, Any]], layers: list[int]
) -> None:
    """Write SVG and simple PNG steering plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_svg_plot(
        plots_dir / "steered_optimal_rate_by_alpha_layer.svg",
        summaries,
        y_key="optimal_rate",
        title="Steered final-action optimal rate by alpha",
        y_label="Optimal final-action rate",
        layers=layers,
    )
    write_simple_line_png(plots_dir / "steered_optimal_rate_by_alpha_layer.png", summaries, "optimal_rate", layers)
    write_svg_plot(
        plots_dir / "paired_delta_optimal_rate_by_alpha_layer.svg",
        paired_stats,
        y_key="delta_optimal_rate",
        title="Paired delta in optimal final-action rate vs alpha=0",
        y_label="Delta optimal rate",
        layers=layers,
    )
    write_simple_line_png(
        plots_dir / "paired_delta_optimal_rate_by_alpha_layer.png", paired_stats, "delta_optimal_rate", layers
    )


def prepare_output_dir(output_root: Path, output_dir: Path | None, overwrite: bool) -> Path:
    """Create the output directory."""
    if output_dir is None:
        output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    return output_dir


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Write all post-generation summaries and plots."""
    write_csv(output_dir / "steering_rows.csv", rows)
    summaries = summarize_steering_rows(rows)
    paired_rows = build_paired_rows(rows)
    paired_stats = paired_stats_by_layer_alpha(paired_rows)
    write_json(output_dir / "steering_summary_by_layer_alpha.json", summaries)
    write_csv(output_dir / "steering_summary_by_layer_alpha.csv", summaries)
    write_json(output_dir / "paired_steering_rows.json", paired_rows)
    write_csv(output_dir / "paired_steering_rows.csv", paired_rows)
    write_json(output_dir / "paired_steering_stats_by_layer_alpha.json", paired_stats)
    write_csv(output_dir / "paired_steering_stats_by_layer_alpha.csv", paired_stats)
    write_plots(output_dir, summaries, paired_stats, args.layers)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "source_run": str(args.source_run),
        "trajectories_dir": str(args.trajectories_dir),
        "activations_dir": str(args.activations_dir),
        "layers": args.layers,
        "alphas": args.alphas,
        "n_seeds": args.n_seeds,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "finalization_fallback": bool(args.finalization_fallback),
        "finalization_max_new_tokens": args.finalization_max_new_tokens,
        "tail_action_fallback": bool(args.tail_action_fallback),
        "clean_autonomous": bool(args.clean_autonomous),
        "generation_protocol": args.generation_protocol,
        "max_samples": args.max_samples,
        "row_count": len(rows),
        "paired_row_count": len(paired_rows),
        "direction_formula": "delta = alpha * ((mean(W_opt)-W_wrong)/std) / ||(mean(W_opt)-W_wrong)/std||^2",
        "prompt_suffix_indices": args.prompt_suffix_indices,
        "prefill_analysis_channel": bool(args.prefill_analysis_channel),
        "outputs": {
            "steering_rows_jsonl": str(output_dir / "steering_rows.jsonl"),
            "steering_rows_csv": str(output_dir / "steering_rows.csv"),
            "paired_steering_rows_json": str(output_dir / "paired_steering_rows.json"),
            "paired_steering_stats_json": str(output_dir / "paired_steering_stats_by_layer_alpha.json"),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", type=Path, default=DEFAULT_TRAJECTORIES_DIR)
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--layers", type=str, default=",".join(str(layer) for layer in DEFAULT_LAYERS))
    parser.add_argument("--alphas", type=str, default=",".join(f"{alpha:g}" for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--clean-autonomous",
        action="store_true",
        help=(
            "Run the clean one-pass protocol: patch the prompt pass, let the model naturally continue until it "
            "emits a JSON action or hits --max-new-tokens, and disable both finalization and tail-action fallbacks."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--prompt-suffix-indices", type=str, default="-3:-1")
    parser.add_argument(
        "--no-prefill-analysis-channel",
        action="store_false",
        dest="prefill_analysis_channel",
        help=(
            "Do not prefill the standard <|channel|>analysis<|message|> tokens. "
            "By default these tokens are prefixed because every stored trajectory begins with them, "
            "and the hook still patches only the original prompt suffix positions."
        ),
    )
    parser.set_defaults(prefill_analysis_channel=True)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument(
        "--no-finalization-fallback",
        action="store_false",
        dest="finalization_fallback",
        help=(
            "Disable the fallback that asks for the final JSON action from the steered generated CoT "
            "when the first generation does not emit JSON by the analysis token budget."
        ),
    )
    parser.add_argument("--finalization-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--no-tail-action-fallback",
        action="store_false",
        dest="tail_action_fallback",
        help=(
            "Disable heuristic parsing of a bare action name near the generated text tail. "
            "With this flag, only a natural JSON action field counts as parsed."
        ),
    )
    parser.set_defaults(finalization_fallback=True)
    parser.set_defaults(tail_action_fallback=True)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize parser strings and derived generation-protocol options."""
    args.layers = parse_int_selection(args.layers) if isinstance(args.layers, str) else list(args.layers)
    args.alphas = parse_float_selection(args.alphas) if isinstance(args.alphas, str) else list(args.alphas)
    if args.clean_autonomous:
        args.finalization_fallback = False
        args.tail_action_fallback = False
    if 0.0 not in set(args.alphas):
        raise SystemExit("--alphas must include 0 for paired same-seed baseline")
    if args.n_seeds <= 0:
        raise SystemExit("--n-seeds must be positive")
    if args.temperature <= 0.0:
        raise SystemExit("temperature must be positive; use 0.7 for this experiment")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    args.generation_protocol = (
        "clean_autonomous"
        if args.clean_autonomous
        else ("two_stage_finalization_fallback" if args.finalization_fallback else "single_pass_with_tail_fallback")
    )
    return args


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = normalize_args(parser.parse_args())
    output_dir = prepare_output_dir(args.output_root, args.output_dir, args.overwrite)
    rows = run_steering_generation(args, output_dir)
    write_outputs(output_dir, rows, args)
    print(f"Wrote steering outputs to {output_dir}")


if __name__ == "__main__":
    main()
