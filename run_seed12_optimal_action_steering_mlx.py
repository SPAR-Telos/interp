"""Run Seed12 optimal-action probe-direction steering with MLX.

This experiment uses real Seed12 trajectory-step prompts and the step-level
linear `optimal_action` probes trained on `pre_reasoning_suffix` activations.
The steering intervention adds one activation-space unit direction to each of
the final three prompt-suffix positions at a selected layer:

    alpha * class_gap * target_direction

Because the probe feature was the mean of these three positions, adding the
same vector to all three positions shifts the probed mean by that vector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flag_swap_suffix_utils import prompt_len, resolve_prompt_suffix_indices, selected_positions_in_pass
from run_seed12_action_probes import (
    ACTION_TO_IDX,
    ACTIONS,
    IDX_TO_ACTION,
    PositionSpec,
    build_state_key,
    build_step_action_probe_dataset,
    load_trajectory_records,
    parse_optimal_action_label,
    parse_true_action_label,
)
from telos_interp.commands.gather_activations.trajectory_activations import reconstruct_text_from_tokens

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q4"
DEFAULT_ACTIVATIONS_DIR = Path("data/activations/Seed12_T0_state_sweep")
DEFAULT_TRAJECTORIES_DIR = Path("data/trajectories/Seed12_T0_state_sweep")
DEFAULT_PROBE_DIR = Path("data/probes/Seed12_T0_state_sweep_action")
DEFAULT_OUT_DIR = Path("data/steering/Seed12_T0_state_sweep_optimal_action")
DEFAULT_REPORT_DIR = Path("reports/seed12_optimal_action_steering")
DEFAULT_LAYERS = (7, 15, 23)
DEFAULT_ALPHAS = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0)
DEFAULT_PROMPT_SUFFIX_INDICES = "-3:-1"
ACTION_LABELS = [*ACTIONS, "PARSE_FAIL"]
ACTION_RE = re.compile(r'"action"\s*:\s*"(LEFT|RIGHT|UP|DOWN)"', re.IGNORECASE)

mx = None
load = None
stream_generate = None
make_sampler = None


@dataclass(frozen=True)
class EligibleStep:
    """A real Seed12 trajectory step eligible for steering."""

    ordinal: int
    trajectory_name: str
    trajectory_index: int
    step_index: int
    step_id: int
    trajectory: dict[str, Any]
    step: dict[str, Any]
    prompt_row: dict[str, Any]
    astar_actions: list[str]
    target_direction_actions: list[str]
    true_action: str | None
    true_action_idx: int | None
    is_original_true_optimal: bool
    state_key: str | None
    state_info: dict[str, Any] | None


def _load_mlx_runtime() -> None:
    """Import MLX dependencies only when the steering mode is used."""
    global mx, load, stream_generate, make_sampler  # noqa: PLW0603
    if mx is not None:
        return

    import mlx.core as mx_mod
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate as mlx_stream_generate
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler

    mx = mx_mod
    load = mlx_load
    stream_generate = mlx_stream_generate
    make_sampler = mlx_make_sampler


def _json_ready(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n")


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_action(text: str) -> str | None:
    """Parse a movement action from a generated completion."""
    match = ACTION_RE.search(text)
    if match:
        return match.group(1).upper()

    tail = text[-300:].upper()
    found = []
    for action in ACTIONS:
        idx = tail.rfind(action)
        if idx >= 0:
            found.append((idx, action))
    if not found:
        return None
    return sorted(found, reverse=True)[0][1]


def parse_layer_selection(raw_layers: list[int] | None) -> list[int]:
    return list(raw_layers) if raw_layers else list(DEFAULT_LAYERS)


def token_ids(tokens: list[dict[str, Any]]) -> list[int]:
    return [int(token["token_id"]) for token in tokens]


def build_step_prompt_row(
    trajectory: dict[str, Any],
    step: dict[str, Any],
    *,
    trajectory_name: str,
    step_index: int,
) -> dict[str, Any]:
    """Build a runtime prompt row from prefix, current grid, and step-specific suffix tokens."""
    prompt = trajectory["prompt"]
    prefix_tokens = prompt["prompt_prefix_tokens"]
    grid_tokens = step["grid_state_tokens"]
    suffix_tokens = step.get("prompt_suffix_tokens", prompt["prompt_suffix_tokens"])

    full_tokens = prefix_tokens + grid_tokens + suffix_tokens
    full_token_ids = token_ids(full_tokens)
    n_prefix = len(prefix_tokens)
    n_grid = len(grid_tokens)
    n_suffix = len(suffix_tokens)
    prompt_text = reconstruct_text_from_tokens(full_tokens)

    return {
        "trajectory_name": trajectory_name,
        "step_index": step_index,
        "step_id": int(step.get("step_id", step_index)),
        "n_prefix": n_prefix,
        "n_grid": n_grid,
        "n_suffix": n_suffix,
        "prompt_len": len(full_token_ids),
        "full_token_ids": full_token_ids,
        "prompt_text": prompt_text,
        "prompt_suffix_tokens": suffix_tokens,
        "prompt_suffix_token_ids": token_ids(suffix_tokens),
        "grid_state_tokens": grid_tokens,
        "grid_state_token_ids": token_ids(grid_tokens),
    }


def resolve_last_suffix_positions(prompt_row: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Resolve the final three step-specific suffix tokens to absolute prompt positions."""
    return resolve_prompt_suffix_indices(prompt_row, DEFAULT_PROMPT_SUFFIX_INDICES)


def normalize_action_list(raw_actions: Any) -> list[str]:
    if not isinstance(raw_actions, list):
        return []
    actions = []
    seen = set()
    for raw_action in raw_actions:
        if not isinstance(raw_action, str):
            continue
        action = raw_action.strip().upper()
        if action in ACTION_TO_IDX and action not in seen:
            actions.append(action)
            seen.add(action)
    return actions


def choose_target_actions(astar_actions: list[str], action_order: list[str] = ACTIONS) -> list[str]:
    """Return tied optimal actions in deterministic action-order order."""
    action_set = set(astar_actions)
    return [action for action in action_order if action in action_set]


def discover_eligible_steps(
    records: list[Any],
    *,
    max_steps: int | None = None,
    max_steps_per_trajectory: int | None = None,
    step_ordinals: set[int] | None = None,
) -> tuple[list[EligibleStep], dict[str, int]]:
    """Collect real Seed12 steps with valid `astar_actions` labels."""
    eligible: list[EligibleStep] = []
    skipped: Counter[str] = Counter()
    eligible_seen = 0

    for trajectory_index, record in enumerate(records):
        steps = record.trajectory.get("steps", [])
        per_trajectory_count = 0
        for step_index, step in enumerate(steps):
            if max_steps is not None and len(eligible) >= max_steps:
                return eligible, dict(skipped)
            if max_steps_per_trajectory is not None and per_trajectory_count >= max_steps_per_trajectory:
                break

            astar_actions = normalize_action_list(step.get("astar_actions"))
            if not astar_actions:
                skipped["missing_or_invalid_astar_actions"] += 1
                continue

            optimal_label = parse_optimal_action_label(step)
            if optimal_label is None:
                skipped["invalid_optimal_action_label"] += 1
                continue

            if "grid_state_tokens" not in step:
                skipped["missing_grid_state_tokens"] += 1
                continue

            prompt_row = build_step_prompt_row(
                record.trajectory,
                step,
                trajectory_name=record.name,
                step_index=step_index,
            )
            try:
                resolve_last_suffix_positions(prompt_row)
            except (KeyError, ValueError):
                skipped["invalid_prompt_suffix_positions"] += 1
                continue

            true_action_idx = parse_true_action_label(step)
            true_action = IDX_TO_ACTION[true_action_idx] if true_action_idx is not None else None
            state_key, state_info, state_reason = build_state_key(step)
            if state_key is None:
                skipped[state_reason or "missing_state_key"] += 1

            current_ordinal = eligible_seen
            eligible_seen += 1
            if step_ordinals is not None and current_ordinal not in step_ordinals:
                continue

            eligible.append(
                EligibleStep(
                    ordinal=current_ordinal,
                    trajectory_name=record.name,
                    trajectory_index=trajectory_index,
                    step_index=step_index,
                    step_id=int(step.get("step_id", step_index)),
                    trajectory=record.trajectory,
                    step=step,
                    prompt_row=prompt_row,
                    astar_actions=astar_actions,
                    target_direction_actions=choose_target_actions(astar_actions),
                    true_action=true_action,
                    true_action_idx=true_action_idx,
                    is_original_true_optimal=true_action in set(astar_actions) if true_action is not None else False,
                    state_key=state_key,
                    state_info=state_info,
                )
            )
            per_trajectory_count += 1

    return eligible, dict(skipped)


def extract_activation_space_weights(checkpoint: dict[str, Any], eps: float = 1e-12) -> torch.Tensor:
    """Convert normalized linear weights into activation-space probe gradients."""
    if checkpoint.get("model_type") != "linear":
        raise ValueError(f"Expected a linear checkpoint, got {checkpoint.get('model_type')!r}")
    weight = checkpoint["model_state_dict"]["weight"].detach().cpu().float()
    scaler_std = checkpoint["scaler_std"].detach().cpu().float()
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D linear weights, got shape {tuple(weight.shape)}")
    if scaler_std.ndim != 1 or scaler_std.shape[0] != weight.shape[1]:
        raise ValueError(f"scaler_std shape {tuple(scaler_std.shape)} does not match weights {tuple(weight.shape)}")
    if not torch.isfinite(weight).all():
        raise ValueError("linear weights contain non-finite values")
    if not torch.isfinite(scaler_std).all() or torch.any(scaler_std <= eps):
        raise ValueError("scaler_std contains non-finite, zero, or negative values")

    return weight / scaler_std.unsqueeze(0)


def unit_normalize_rows(vectors: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a 2D tensor row-wise."""
    norms = vectors.float().norm(dim=1, keepdim=True)
    if torch.any(norms <= eps):
        raise ValueError("one or more vectors has zero norm")
    return vectors.float() / norms


def extract_activation_space_directions(checkpoint: dict[str, Any], eps: float = 1e-12) -> torch.Tensor:
    """Convert normalized linear weights into unit activation-space directions."""
    activation_weights = extract_activation_space_weights(checkpoint, eps=eps)
    return unit_normalize_rows(activation_weights, eps=eps)


def compute_action_mean_diffs(
    activations: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Compute activation mean differences for each optimal-action label."""
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got {tuple(activations.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D multi-hot, got {tuple(labels.shape)}")
    if labels.shape[0] != activations.shape[0]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} incompatible with activations {tuple(activations.shape)}"
        )

    mean_diffs = []
    counts = []
    for action_idx in range(labels.shape[1]):
        positive_mask = labels[:, action_idx].float() > 0.0
        negative_mask = ~positive_mask
        positive_count = int(positive_mask.sum().item())
        negative_count = int(negative_mask.sum().item())
        if positive_count == 0 or negative_count == 0:
            mean_diff = torch.zeros(activations.shape[1], dtype=torch.float32)
        else:
            mean_diff = activations[positive_mask].float().mean(dim=0) - activations[negative_mask].float().mean(dim=0)
        mean_diffs.append(mean_diff)
        counts.append({"positive": positive_count, "negative": negative_count})
    return torch.stack(mean_diffs).float(), counts


def compute_action_positive_means(
    activations: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Compute positive-label activation centroids for each optimal-action label."""
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got {tuple(activations.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D multi-hot, got {tuple(labels.shape)}")
    if labels.shape[0] != activations.shape[0]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} incompatible with activations {tuple(activations.shape)}"
        )

    positive_means = []
    counts = []
    for action_idx in range(labels.shape[1]):
        positive_mask = labels[:, action_idx].float() > 0.0
        positive_count = int(positive_mask.sum().item())
        if positive_count == 0:
            positive_mean = torch.zeros(activations.shape[1], dtype=torch.float32)
        else:
            positive_mean = activations[positive_mask].float().mean(dim=0)
        positive_means.append(positive_mean)
        counts.append({"positive": positive_count})
    return torch.stack(positive_means).float(), counts


def compute_action_class_gaps(
    activations: torch.Tensor,
    labels: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    """Compute positive-minus-negative mean projection gaps for every action."""
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D, got {tuple(activations.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D multi-hot, got {tuple(labels.shape)}")
    if directions.shape != (labels.shape[1], activations.shape[1]):
        raise ValueError(
            f"directions shape {tuple(directions.shape)} incompatible with labels {tuple(labels.shape)} "
            f"and activations {tuple(activations.shape)}"
        )

    gaps = []
    counts = []
    for action_idx in range(labels.shape[1]):
        projections = activations.float() @ directions[action_idx].float()
        positive_mask = labels[:, action_idx].float() > 0.0
        negative_mask = ~positive_mask
        positive_count = int(positive_mask.sum().item())
        negative_count = int(negative_mask.sum().item())
        if positive_count == 0 or negative_count == 0:
            gap = torch.tensor(0.0, dtype=torch.float32)
        else:
            gap = projections[positive_mask].mean() - projections[negative_mask].mean()
        gaps.append(gap)
        counts.append({"positive": positive_count, "negative": negative_count})
    return torch.stack(gaps).float(), counts


def class_gap_for_direction(
    action: str,
    direction: torch.Tensor,
    activation_mean_diffs_by_action: torch.Tensor,
) -> float:
    """Return the positive-minus-negative projection gap for an action and unit direction."""
    action_idx = ACTION_TO_IDX[action]
    return float((activation_mean_diffs_by_action[action_idx].float() @ direction.float()).item())


def build_target_direction(
    optimal_actions: list[str],
    directions_by_action: torch.Tensor,
    class_gaps_by_action: torch.Tensor,
    *,
    action_order: list[str] = ACTIONS,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, list[str]]:
    """Average tied optimal-action directions and class gaps deterministically."""
    ordered_actions = choose_target_actions(optimal_actions, action_order)
    if not ordered_actions:
        raise ValueError("No valid optimal actions to build a steering target")

    indices = [ACTION_TO_IDX[action] for action in ordered_actions]
    target = directions_by_action[indices].float().mean(dim=0)
    norm = target.norm()
    if float(norm) <= eps:
        raise ValueError(f"Target direction for actions {ordered_actions} has zero norm")
    target = target / norm
    class_gap = float(class_gaps_by_action[indices].float().mean().item())
    return target, class_gap, ordered_actions


def build_single_action_direction(
    action: str,
    directions_by_action: torch.Tensor,
    class_gaps_by_action: torch.Tensor,
) -> tuple[torch.Tensor, float, list[str]]:
    """Return the unit direction and class gap for one action label."""
    action_idx = ACTION_TO_IDX[action]
    return directions_by_action[action_idx].float(), float(class_gaps_by_action[action_idx].float().item()), [action]


def build_action_contrast_direction(
    action: str,
    directions_by_action: torch.Tensor,
    class_gaps_by_action: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, list[str]]:
    """Return a one-vs-rest contrast direction for one action label."""
    action_idx = ACTION_TO_IDX[action]
    all_indices = list(range(len(ACTIONS)))
    other_indices = [idx for idx in all_indices if idx != action_idx]
    contrast = directions_by_action[action_idx].float() - directions_by_action[other_indices].float().mean(dim=0)
    norm = contrast.norm()
    if float(norm) <= eps:
        raise ValueError(f"Contrast direction for action {action} has zero norm")
    contrast = contrast / norm
    return contrast, float(class_gaps_by_action[action_idx].float().item()), [action]


def build_action_vs_source_direction(
    action: str,
    source_action: str,
    directions_by_action: torch.Tensor,
    class_gaps_by_action: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, list[str]]:
    """Return a target-minus-source action contrast direction."""
    action_idx = ACTION_TO_IDX[action]
    source_idx = ACTION_TO_IDX[source_action]
    contrast = directions_by_action[action_idx].float() - directions_by_action[source_idx].float()
    norm = contrast.norm()
    if float(norm) <= eps:
        raise ValueError(f"Contrast direction from {source_action} to {action} has zero norm")
    contrast = contrast / norm
    return contrast, float(class_gaps_by_action[action_idx].float().item()), [action]


def build_raw_action_contrast_direction(
    action: str,
    activation_weights_by_action: torch.Tensor,
    activation_mean_diffs_by_action: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, list[str]]:
    """Return a raw probe-logit one-vs-rest contrast direction for one action label."""
    action_idx = ACTION_TO_IDX[action]
    all_indices = list(range(len(ACTIONS)))
    other_indices = [idx for idx in all_indices if idx != action_idx]
    contrast = activation_weights_by_action[action_idx].float() - activation_weights_by_action[
        other_indices
    ].float().mean(dim=0)
    norm = contrast.norm()
    if float(norm) <= eps:
        raise ValueError(f"Raw contrast direction for action {action} has zero norm")
    contrast = contrast / norm
    return contrast, class_gap_for_direction(action, contrast, activation_mean_diffs_by_action), [action]


def build_raw_action_vs_source_direction(
    action: str,
    source_action: str,
    activation_weights_by_action: torch.Tensor,
    activation_mean_diffs_by_action: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, list[str]]:
    """Return a raw probe-logit target-minus-source action contrast direction."""
    action_idx = ACTION_TO_IDX[action]
    source_idx = ACTION_TO_IDX[source_action]
    contrast = activation_weights_by_action[action_idx].float() - activation_weights_by_action[source_idx].float()
    norm = contrast.norm()
    if float(norm) <= eps:
        raise ValueError(f"Raw contrast direction from {source_action} to {action} has zero norm")
    contrast = contrast / norm
    return contrast, class_gap_for_direction(action, contrast, activation_mean_diffs_by_action), [action]


def build_centroid_action_contrast_direction(
    action: str,
    positive_means_by_action: torch.Tensor,
    activation_mean_diffs_by_action: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, list[str]]:
    """Return a centroid one-vs-rest contrast direction for one optimal-action label."""
    action_idx = ACTION_TO_IDX[action]
    all_indices = list(range(len(ACTIONS)))
    other_indices = [idx for idx in all_indices if idx != action_idx]
    contrast = positive_means_by_action[action_idx].float() - positive_means_by_action[other_indices].float().mean(
        dim=0
    )
    norm = contrast.norm()
    if float(norm) <= eps:
        raise ValueError(f"Centroid contrast direction for action {action} has zero norm")
    contrast = contrast / norm
    return contrast, abs(class_gap_for_direction(action, contrast, activation_mean_diffs_by_action)), [action]


def build_centroid_action_vs_source_direction(
    action: str,
    source_action: str,
    positive_means_by_action: torch.Tensor,
    activation_mean_diffs_by_action: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, float, list[str]]:
    """Return a target-minus-source centroid direction for one optimal-action label."""
    action_idx = ACTION_TO_IDX[action]
    source_idx = ACTION_TO_IDX[source_action]
    contrast = positive_means_by_action[action_idx].float() - positive_means_by_action[source_idx].float()
    norm = contrast.norm()
    if float(norm) <= eps:
        raise ValueError(f"Centroid contrast direction from {source_action} to {action} has zero norm")
    contrast = contrast / norm
    return contrast, abs(class_gap_for_direction(action, contrast, activation_mean_diffs_by_action)), [action]


def direction_label(actions: list[str]) -> str:
    return "+".join(actions) if actions else "none"


def probe_checkpoint_path(probe_dir: Path, layer: int) -> Path:
    return probe_dir / f"probe_optimal_action_pre_reasoning_suffix_layer{layer}_linear.pt"


def direction_bundle_path(out_dir: Path, layer: int) -> Path:
    return out_dir / f"directions_layer{layer}.pt"


def load_direction_bundle(out_dir: Path, layer: int) -> dict[str, Any]:
    path = direction_bundle_path(out_dir, layer)
    if not path.exists():
        raise FileNotFoundError(f"Missing direction bundle for layer {layer}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def run_prepare_directions(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layer_selection(args.layers)

    records, skipped_trajectories = load_trajectory_records(
        Path(args.activations_dir),
        Path(args.trajectories_dir),
        args.max_trajectories,
    )
    if not records:
        raise ValueError("No trajectory records with activation folders were found")

    position = PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix")
    manifest: dict[str, Any] = {
        "command": "prepare-directions",
        "activations_dir": str(args.activations_dir),
        "trajectories_dir": str(args.trajectories_dir),
        "probe_dir": str(args.probe_dir),
        "out_dir": str(out_dir),
        "layers": layers,
        "action_order": ACTIONS,
        "action_mapping": ACTION_TO_IDX,
        "skipped_trajectories": skipped_trajectories,
        "directions": {},
    }

    for layer in layers:
        checkpoint_path = probe_checkpoint_path(Path(args.probe_dir), layer)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing probe checkpoint for layer {layer}: {checkpoint_path}")

        output_path = direction_bundle_path(out_dir, layer)
        ensure_can_write(output_path, args.overwrite)

        print(f"Preparing layer {layer} directions from {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        activation_weights = extract_activation_space_weights(checkpoint)
        directions = unit_normalize_rows(activation_weights)

        dataset = build_step_action_probe_dataset(records, "optimal_action", position, layer)
        if int(dataset.activations.shape[0]) == 0:
            raise RuntimeError(f"No pre-suffix optimal-action samples for layer {layer}; skipped={dataset.skipped}")

        activation_mean_diffs, class_gap_counts = compute_action_mean_diffs(dataset.activations, dataset.labels)
        positive_means, positive_mean_counts = compute_action_positive_means(dataset.activations, dataset.labels)
        class_gaps, class_gap_counts = compute_action_class_gaps(dataset.activations, dataset.labels, directions)
        bundle = {
            "layer": layer,
            "probe_path": str(checkpoint_path),
            "probe_config": checkpoint.get("config", {}),
            "probe_metrics": checkpoint.get("metrics", {}),
            "action_order": ACTIONS,
            "action_to_idx": ACTION_TO_IDX,
            "directions_by_action": directions.detach().cpu(),
            "class_gaps_by_action": class_gaps.detach().cpu(),
            "class_gap_counts_by_action": {ACTIONS[idx]: class_gap_counts[idx] for idx in range(len(ACTIONS))},
            "activation_weights_by_action": activation_weights.detach().cpu(),
            "activation_mean_diffs_by_action": activation_mean_diffs.detach().cpu(),
            "positive_means_by_action": positive_means.detach().cpu(),
            "positive_mean_counts_by_action": {ACTIONS[idx]: positive_mean_counts[idx] for idx in range(len(ACTIONS))},
            "scaler_std": checkpoint["scaler_std"].detach().cpu(),
            "linear_weight_normalized": checkpoint["model_state_dict"]["weight"].detach().cpu(),
            "dataset": {
                "variant": "step",
                "label_type": "optimal_action",
                "position": "pre_reasoning_suffix",
                "layer": layer,
                "sample_count": int(dataset.activations.shape[0]),
                "input_dim": int(dataset.activations.shape[1]),
                "label_distribution": dataset.label_distribution,
                "skipped": dataset.skipped,
                "metadata": dataset.metadata,
            },
            "direction_note": (
                "directions_by_action[action] = unit_normalize(linear_weight[action] / scaler_std). "
                "activation_weights_by_action[action] = linear_weight[action] / scaler_std. "
                "activation_mean_diffs_by_action[action] is mean activation for positive labels minus negatives. "
                "positive_means_by_action[action] is the activation centroid for positive labels. "
                "class_gaps_by_action[action] is mean raw-activation projection for positive labels minus negatives."
            ),
        }
        torch.save(bundle, output_path)

        manifest["directions"][str(layer)] = {
            "path": str(output_path),
            "probe_path": str(checkpoint_path),
            "sample_count": int(dataset.activations.shape[0]),
            "skipped": dataset.skipped,
            "class_gaps_by_action": {ACTIONS[idx]: float(class_gaps[idx].item()) for idx in range(len(ACTIONS))},
            "direction_norms_by_action": {
                ACTIONS[idx]: float(directions[idx].norm().item()) for idx in range(len(ACTIONS))
            },
        }
        print(
            f"  saved {output_path}; gaps={manifest['directions'][str(layer)]['class_gaps_by_action']}",
            flush=True,
        )

    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote manifest to {out_dir / 'manifest.json'}")


class AdditiveLayerHook:
    """Add a fixed direction to selected absolute prompt positions in one layer."""

    def __init__(self, parent_list: list[Any], idx: int):
        self.parent_list = parent_list
        self.idx = idx
        self.original_layer = parent_list[idx]
        self.installed = False
        self.mode = "passthrough"
        self.direction = None
        self.scale = 0.0
        self.prompt_length = None
        self.selected_positions: list[int] = []
        self.prompt_cursor = 0
        self.fired = False
        self.seq_lens: list[int] = []
        self.touched_positions: list[int] = []
        self.patched_positions: list[int] = []
        self.pre_norm = None
        self.post_norm = None

    def install(self) -> None:
        if not self.installed:
            self.parent_list[self.idx] = self
            self.installed = True

    def restore(self) -> None:
        if self.installed:
            self.parent_list[self.idx] = self.original_layer
            self.installed = False

    def __call__(self, x, mask, cache, *args, **kwargs):
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
        if self.mode == "add":
            if self.direction is None:
                raise RuntimeError("direction must be set for add mode")
            out = self._add_touched(out, touched)
        return out

    def _add_touched(self, out, touched: list[tuple[int, int, int]]):
        before = mx.concatenate([out[:, local_idx : local_idx + 1, :] for local_idx, _, _ in touched], axis=1)
        self.pre_norm = float(mx.linalg.norm(before.astype(mx.float32)))

        pieces = []
        cursor = 0
        changed = []
        for local_idx, _row_idx, abs_pos in touched:
            if local_idx > cursor:
                pieces.append(out[:, cursor:local_idx, :])
            perturb = (self.direction * self.scale).astype(out.dtype)
            patched = out[:, local_idx : local_idx + 1, :] + mx.broadcast_to(
                perturb[None, None, :], (out.shape[0], 1, out.shape[2])
            )
            pieces.append(patched)
            changed.append(patched.astype(mx.float32))
            self.patched_positions.append(abs_pos)
            cursor = local_idx + 1
        if cursor < out.shape[1]:
            pieces.append(out[:, cursor:, :])

        after = mx.concatenate(changed, axis=1)
        self.post_norm = float(mx.linalg.norm(after))
        return mx.concatenate(pieces, axis=1)

    def configure_prompt(self, prompt_row: dict[str, Any], selected_positions: list[int]) -> None:
        self.reset()
        self.prompt_length = prompt_len(prompt_row)
        self.selected_positions = list(selected_positions)

    def reset(self) -> None:
        self.mode = "passthrough"
        self.direction = None
        self.scale = 0.0
        self.prompt_length = None
        self.selected_positions = []
        self.prompt_cursor = 0
        self.fired = False
        self.seq_lens = []
        self.touched_positions = []
        self.patched_positions = []
        self.pre_norm = None
        self.post_norm = None

    def log(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
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


def generate_one(model, tokenizer, prompt_token_ids: list[int], sampler, max_tokens: int) -> str:
    pieces = []
    for chunk in stream_generate(model, tokenizer, prompt=prompt_token_ids, max_tokens=max_tokens, sampler=sampler):
        pieces.append(chunk.text)
    return "".join(pieces)


def stable_seed(base_seed: int, *parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base_seed).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), byteorder="little") % (2**31 - 1)


def steering_sample_seed(
    base_seed: int,
    *,
    layer: int,
    trajectory_name: str,
    step_id: int,
    sample_idx: int,
) -> int:
    """Return the paired generation seed for a step/layer/sample condition.

    Every alpha for the same step/layer/sample intentionally shares this seed.
    That makes condition comparisons paired on decoding randomness, so measured
    differences are attributable to the hook rather than independent sampling.
    """
    return stable_seed(base_seed, "sample", layer, trajectory_name, step_id, sample_idx)


def no_hook_log(prompt_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "installed": False,
        "fired": False,
        "seq_lens": [],
        "mode": "none",
        "scale": 0.0,
        "prompt_len": prompt_len(prompt_row),
        "selected_prompt_positions": [],
        "touched_prompt_positions": [],
        "patched_prompt_positions": [],
        "pre_norm": None,
        "post_norm": None,
    }


def build_result_row(
    *,
    eligible_step: EligibleStep,
    layer: int,
    condition: str,
    alpha: float | None,
    sample_idx: int,
    seed: int,
    emitted_action: str | None,
    text: str,
    suffix_relative_indices: list[int],
    suffix_positions: list[int],
    class_gap: float | None,
    scale: float,
    target_direction_actions: list[str],
    direction_mode: str,
    hook_log: dict[str, Any] | None,
    source_action: str | None = None,
) -> dict[str, Any]:
    row = {
        "condition": condition,
        "layer": layer,
        "alpha": alpha,
        "class_gap": class_gap,
        "scale": scale,
        "trajectory_name": eligible_step.trajectory_name,
        "trajectory_index": eligible_step.trajectory_index,
        "step_index": eligible_step.step_index,
        "step_id": eligible_step.step_id,
        "step_ordinal": eligible_step.ordinal,
        "state_key": eligible_step.state_key,
        "state_info": eligible_step.state_info,
        "true_action": eligible_step.true_action,
        "astar_actions": eligible_step.astar_actions,
        "target_direction_actions": target_direction_actions,
        "direction_mode": direction_mode,
        "direction_label": direction_label(target_direction_actions),
        "source_action": source_action,
        "is_desired_direction": bool(set(target_direction_actions) & set(eligible_step.astar_actions)),
        "is_original_true_optimal": eligible_step.is_original_true_optimal,
        "carrying_key": bool(eligible_step.step.get("carrying_key")),
        "door_open": bool(eligible_step.step.get("door_open")),
        "prompt_len": prompt_len(eligible_step.prompt_row),
        "prompt_suffix_indices": DEFAULT_PROMPT_SUFFIX_INDICES,
        "suffix_relative_indices": suffix_relative_indices,
        "selected_prompt_positions": suffix_positions,
        "sample_idx": sample_idx,
        "seed": seed,
        "emitted_action": emitted_action,
        "parse_failed": emitted_action is None,
        "emitted_action_label": emitted_action or "PARSE_FAIL",
        "text_tail": text[-500:],
    }
    if hook_log is not None:
        row["_hook_log"] = hook_log
    return row


def run_no_hook_sample(
    model,
    tokenizer,
    prompt_row: dict[str, Any],
    *,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> tuple[str | None, str]:
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature)
    text = generate_one(model, tokenizer, prompt_row["full_token_ids"], sampler, max_new_tokens)
    return parse_action(text), text


def run_hooked_sample(
    model,
    tokenizer,
    hook: AdditiveLayerHook,
    prompt_row: dict[str, Any],
    suffix_positions: list[int],
    direction,
    scale: float,
    *,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> tuple[str | None, str, dict[str, Any]]:
    hook.install()
    hook.configure_prompt(prompt_row, suffix_positions)
    hook.mode = "add"
    hook.direction = direction
    hook.scale = scale
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature)
    text = generate_one(model, tokenizer, prompt_row["full_token_ids"], sampler, max_new_tokens)
    return parse_action(text), text, hook.log()


def run_steering(args: argparse.Namespace) -> None:  # noqa: PLR0912
    if args.temperature <= 0.0:
        raise SystemExit("Steering sweeps use sampled generation; pass a positive --temperature such as 0.7.")

    _load_mlx_runtime()
    layers = parse_layer_selection(args.layers)
    output_path = Path(args.output)
    ensure_can_write(output_path, args.overwrite)

    records, skipped_trajectories = load_trajectory_records(
        Path(args.activations_dir),
        Path(args.trajectories_dir),
        args.max_trajectories,
    )
    eligible_steps, skipped_steps = discover_eligible_steps(
        records,
        max_steps=args.max_steps,
        max_steps_per_trajectory=args.max_steps_per_trajectory,
        step_ordinals=set(args.step_ordinals) if args.step_ordinals else None,
    )
    if not eligible_steps:
        raise ValueError(f"No eligible Seed12 steps found; skipped={skipped_steps}")

    print(f"Loading {args.model_id} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model_id)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    n_layers = len(model.layers)
    for layer in layers:
        if layer >= n_layers:
            raise SystemExit(f"--layer {layer} out of range; model has {n_layers} layers")

    direction_bundles = (
        {} if args.baseline_only else {layer: load_direction_bundle(Path(args.out_dir), layer) for layer in layers}
    )
    manifest = {
        "command": "steer",
        "model_id": args.model_id,
        "activations_dir": str(args.activations_dir),
        "trajectories_dir": str(args.trajectories_dir),
        "out_dir": str(args.out_dir),
        "output": str(output_path),
        "layers": layers,
        "alphas": [float(alpha) for alpha in args.alphas],
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "seed_policy": "paired_by_step_layer_sample_across_no_hook_and_all_alphas",
        "baseline_only": bool(args.baseline_only),
        "direction_set": args.direction_set,
        "step_ordinals": args.step_ordinals,
        "action_order": ACTIONS,
        "action_mapping": ACTION_TO_IDX,
        "eligible_step_count": len(eligible_steps),
        "loaded_trajectory_count": len(records),
        "skipped_trajectories": skipped_trajectories,
        "skipped_steps": skipped_steps,
        "direction_paths": {str(layer): str(direction_bundle_path(Path(args.out_dir), layer)) for layer in layers},
    }
    write_json(Path(args.out_dir) / "steering_manifest.json", manifest)

    n_written = 0
    with output_path.open("w") as fout:
        for layer_index, layer in enumerate(layers):
            hook = AdditiveLayerHook(model.model.layers, layer)
            if args.baseline_only:
                directions_by_action = None
                class_gaps_by_action = None
            else:
                bundle = direction_bundles[layer]
                directions_by_action = bundle["directions_by_action"].float()
                class_gaps_by_action = bundle["class_gaps_by_action"].float()
                activation_weights_by_action = bundle.get("activation_weights_by_action")
                activation_mean_diffs_by_action = bundle.get("activation_mean_diffs_by_action")
                positive_means_by_action = bundle.get("positive_means_by_action")
                if activation_weights_by_action is not None:
                    activation_weights_by_action = activation_weights_by_action.float()
                if activation_mean_diffs_by_action is not None:
                    activation_mean_diffs_by_action = activation_mean_diffs_by_action.float()
                if positive_means_by_action is not None:
                    positive_means_by_action = positive_means_by_action.float()
            try:
                for step in eligible_steps:
                    suffix_relative_indices, suffix_positions = resolve_last_suffix_positions(step.prompt_row)
                    if args.baseline_only:
                        class_gap = None
                        direction_specs = [
                            {
                                "mode": "none",
                                "actions": step.target_direction_actions,
                                "class_gap": None,
                                "direction": None,
                            }
                        ]
                    elif args.direction_set == "optimal":
                        target_direction_t, class_gap, target_actions = build_target_direction(
                            step.astar_actions,
                            directions_by_action,
                            class_gaps_by_action,
                        )
                        direction_specs = [
                            {
                                "mode": "optimal",
                                "actions": target_actions,
                                "class_gap": class_gap,
                                "direction": mx.array(target_direction_t.detach().cpu().numpy().astype(np.float32)),
                            }
                        ]
                    elif args.direction_set == "all-actions":
                        direction_specs = []
                        for action in ACTIONS:
                            action_direction_t, action_class_gap, action_targets = build_single_action_direction(
                                action,
                                directions_by_action,
                                class_gaps_by_action,
                            )
                            direction_specs.append(
                                {
                                    "mode": "single_action",
                                    "actions": action_targets,
                                    "class_gap": action_class_gap,
                                    "direction": mx.array(
                                        action_direction_t.detach().cpu().numpy().astype(np.float32)
                                    ),
                                }
                            )
                    elif args.direction_set == "raw-action-vs-baseline":
                        if activation_weights_by_action is None or activation_mean_diffs_by_action is None:
                            raise RuntimeError(
                                "raw-action-vs-baseline requires direction bundles prepared with "
                                "activation_weights_by_action and activation_mean_diffs_by_action"
                            )
                        direction_specs = []
                    else:
                        direction_specs = []
                        if args.direction_set == "action-contrast":
                            for action in ACTIONS:
                                action_direction_t, action_class_gap, action_targets = build_action_contrast_direction(
                                    action,
                                    directions_by_action,
                                    class_gaps_by_action,
                                )
                                direction_specs.append(
                                    {
                                        "mode": "action_contrast",
                                        "actions": action_targets,
                                        "class_gap": action_class_gap,
                                        "direction": mx.array(
                                            action_direction_t.detach().cpu().numpy().astype(np.float32)
                                        ),
                                    }
                                )
                        elif args.direction_set == "raw-action-contrast":
                            if activation_weights_by_action is None or activation_mean_diffs_by_action is None:
                                raise RuntimeError(
                                    "raw-action-contrast requires direction bundles prepared with "
                                    "activation_weights_by_action and activation_mean_diffs_by_action"
                                )
                            for action in ACTIONS:
                                action_direction_t, action_class_gap, action_targets = (
                                    build_raw_action_contrast_direction(
                                        action,
                                        activation_weights_by_action,
                                        activation_mean_diffs_by_action,
                                    )
                                )
                                direction_specs.append(
                                    {
                                        "mode": "raw_action_contrast",
                                        "actions": action_targets,
                                        "class_gap": action_class_gap,
                                        "direction": mx.array(
                                            action_direction_t.detach().cpu().numpy().astype(np.float32)
                                        ),
                                    }
                                )
                        elif args.direction_set == "centroid-action-contrast":
                            if positive_means_by_action is None or activation_mean_diffs_by_action is None:
                                raise RuntimeError(
                                    "centroid-action-contrast requires direction bundles prepared with "
                                    "positive_means_by_action and activation_mean_diffs_by_action"
                                )
                            for action in ACTIONS:
                                action_direction_t, action_class_gap, action_targets = (
                                    build_centroid_action_contrast_direction(
                                        action,
                                        positive_means_by_action,
                                        activation_mean_diffs_by_action,
                                    )
                                )
                                direction_specs.append(
                                    {
                                        "mode": "centroid_action_contrast",
                                        "actions": action_targets,
                                        "class_gap": action_class_gap,
                                        "direction": mx.array(
                                            action_direction_t.detach().cpu().numpy().astype(np.float32)
                                        ),
                                    }
                                )
                        elif step.true_action in ACTION_TO_IDX:
                            if args.direction_set == "action-vs-true":
                                for action in ACTIONS:
                                    if action == step.true_action:
                                        continue
                                    action_direction_t, action_class_gap, action_targets = (
                                        build_action_vs_source_direction(
                                            action,
                                            step.true_action,
                                            directions_by_action,
                                            class_gaps_by_action,
                                        )
                                    )
                                    direction_specs.append(
                                        {
                                            "mode": "action_vs_true",
                                            "actions": action_targets,
                                            "class_gap": action_class_gap,
                                            "direction": mx.array(
                                                action_direction_t.detach().cpu().numpy().astype(np.float32)
                                            ),
                                        }
                                    )
                            elif args.direction_set == "raw-action-vs-true":
                                if activation_weights_by_action is None or activation_mean_diffs_by_action is None:
                                    raise RuntimeError(
                                        "raw-action-vs-true requires direction bundles prepared with "
                                        "activation_weights_by_action and activation_mean_diffs_by_action"
                                    )
                                for action in ACTIONS:
                                    if action == step.true_action:
                                        continue
                                    action_direction_t, action_class_gap, action_targets = (
                                        build_raw_action_vs_source_direction(
                                            action,
                                            step.true_action,
                                            activation_weights_by_action,
                                            activation_mean_diffs_by_action,
                                        )
                                    )
                                    direction_specs.append(
                                        {
                                            "mode": "raw_action_vs_true",
                                            "actions": action_targets,
                                            "class_gap": action_class_gap,
                                            "direction": mx.array(
                                                action_direction_t.detach().cpu().numpy().astype(np.float32)
                                            ),
                                        }
                                    )
                            else:
                                if positive_means_by_action is None or activation_mean_diffs_by_action is None:
                                    raise RuntimeError(
                                        "centroid-action-vs-true requires direction bundles prepared with "
                                        "positive_means_by_action and activation_mean_diffs_by_action"
                                    )
                                for action in ACTIONS:
                                    if action == step.true_action:
                                        continue
                                    action_direction_t, action_class_gap, action_targets = (
                                        build_centroid_action_vs_source_direction(
                                            action,
                                            step.true_action,
                                            positive_means_by_action,
                                            activation_mean_diffs_by_action,
                                        )
                                    )
                                    direction_specs.append(
                                        {
                                            "mode": "centroid_action_vs_true",
                                            "actions": action_targets,
                                            "class_gap": action_class_gap,
                                            "direction": mx.array(
                                                action_direction_t.detach().cpu().numpy().astype(np.float32)
                                            ),
                                        }
                                    )

                    baseline_actions_by_sample: dict[int, str | None] = {}
                    for sample_idx in range(args.n_samples):
                        seed = steering_sample_seed(
                            args.seed,
                            layer=layer,
                            trajectory_name=step.trajectory_name,
                            step_id=step.step_id,
                            sample_idx=sample_idx,
                        )
                        hook.restore()
                        emitted_action, text = run_no_hook_sample(
                            model,
                            tokenizer,
                            step.prompt_row,
                            seed=seed,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                        )
                        baseline_actions_by_sample[sample_idx] = emitted_action
                        row = build_result_row(
                            eligible_step=step,
                            layer=layer,
                            condition="no_hook",
                            alpha=None,
                            sample_idx=sample_idx,
                            seed=seed,
                            emitted_action=emitted_action,
                            text=text,
                            suffix_relative_indices=suffix_relative_indices,
                            suffix_positions=suffix_positions,
                            class_gap=None,
                            scale=0.0,
                            target_direction_actions=step.target_direction_actions,
                            direction_mode="none",
                            hook_log=no_hook_log(step.prompt_row) if sample_idx == 0 else None,
                        )
                        fout.write(json.dumps(_json_ready(row)) + "\n")
                        n_written += 1

                    if args.baseline_only:
                        continue

                    hook.install()
                    if args.direction_set == "raw-action-vs-baseline":
                        for alpha in args.alphas:
                            counts = Counter()
                            rows_for_alpha = 0
                            t1 = time.time()
                            for sample_idx, source_action in baseline_actions_by_sample.items():
                                if source_action not in ACTION_TO_IDX or source_action in set(step.astar_actions):
                                    continue
                                seed = steering_sample_seed(
                                    args.seed,
                                    layer=layer,
                                    trajectory_name=step.trajectory_name,
                                    step_id=step.step_id,
                                    sample_idx=sample_idx,
                                )
                                for action in ACTIONS:
                                    if action == source_action:
                                        continue
                                    action_direction_t, action_class_gap, action_targets = (
                                        build_raw_action_vs_source_direction(
                                            action,
                                            source_action,
                                            activation_weights_by_action,
                                            activation_mean_diffs_by_action,
                                        )
                                    )
                                    scale = float(alpha) * float(action_class_gap)
                                    emitted_action, text, hook_log = run_hooked_sample(
                                        model,
                                        tokenizer,
                                        hook,
                                        step.prompt_row,
                                        suffix_positions,
                                        mx.array(action_direction_t.detach().cpu().numpy().astype(np.float32)),
                                        scale,
                                        seed=seed,
                                        temperature=args.temperature,
                                        max_new_tokens=args.max_new_tokens,
                                    )
                                    row = build_result_row(
                                        eligible_step=step,
                                        layer=layer,
                                        condition="hooked",
                                        alpha=float(alpha),
                                        sample_idx=sample_idx,
                                        seed=seed,
                                        emitted_action=emitted_action,
                                        text=text,
                                        suffix_relative_indices=suffix_relative_indices,
                                        suffix_positions=suffix_positions,
                                        class_gap=float(action_class_gap),
                                        scale=scale,
                                        target_direction_actions=action_targets,
                                        direction_mode="raw_action_vs_baseline",
                                        hook_log=hook_log if sample_idx == 0 else None,
                                        source_action=source_action,
                                    )
                                    fout.write(json.dumps(_json_ready(row)) + "\n")
                                    fout.flush()
                                    counts[row["emitted_action_label"]] += 1
                                    rows_for_alpha += 1
                                    n_written += 1
                            print(
                                f"layer={layer} step={step.ordinal + 1}/{len(eligible_steps)} "
                                f"dir=raw_action_vs_baseline alpha={float(alpha):g} "
                                f"target={step.target_direction_actions} rows={rows_for_alpha} "
                                f"counts={dict(counts)} elapsed={time.time() - t1:.1f}s",
                                flush=True,
                            )
                        continue

                    for spec in direction_specs:
                        for alpha in args.alphas:
                            class_gap = float(spec["class_gap"])
                            scale = float(alpha) * class_gap
                            counts = Counter()
                            t1 = time.time()
                            for sample_idx in range(args.n_samples):
                                seed = steering_sample_seed(
                                    args.seed,
                                    layer=layer,
                                    trajectory_name=step.trajectory_name,
                                    step_id=step.step_id,
                                    sample_idx=sample_idx,
                                )
                                emitted_action, text, hook_log = run_hooked_sample(
                                    model,
                                    tokenizer,
                                    hook,
                                    step.prompt_row,
                                    suffix_positions,
                                    spec["direction"],
                                    scale,
                                    seed=seed,
                                    temperature=args.temperature,
                                    max_new_tokens=args.max_new_tokens,
                                )
                                row = build_result_row(
                                    eligible_step=step,
                                    layer=layer,
                                    condition="hooked",
                                    alpha=float(alpha),
                                    sample_idx=sample_idx,
                                    seed=seed,
                                    emitted_action=emitted_action,
                                    text=text,
                                    suffix_relative_indices=suffix_relative_indices,
                                    suffix_positions=suffix_positions,
                                    class_gap=class_gap,
                                    scale=scale,
                                    target_direction_actions=spec["actions"],
                                    direction_mode=spec["mode"],
                                    hook_log=hook_log if sample_idx == 0 else None,
                                )
                                fout.write(json.dumps(_json_ready(row)) + "\n")
                                fout.flush()
                                counts[row["emitted_action_label"]] += 1
                                n_written += 1
                            print(
                                f"layer={layer} step={step.ordinal + 1}/{len(eligible_steps)} "
                                f"dir={direction_label(spec['actions'])} alpha={float(alpha):g} "
                                f"target={step.target_direction_actions} counts={dict(counts)} "
                                f"elapsed={time.time() - t1:.1f}s",
                                flush=True,
                            )
            finally:
                hook.restore()
            print(f"Finished layer {layer} ({layer_index + 1}/{len(layers)})", flush=True)

    print(f"Wrote {n_written} rows to {output_path}")


def action_label(row: dict[str, Any]) -> str:
    action = row.get("emitted_action")
    if isinstance(action, str) and action in ACTION_TO_IDX:
        return action
    label = row.get("emitted_action_label")
    if isinstance(label, str) and label in ACTION_LABELS:
        return label
    return "PARSE_FAIL"


def is_optimal_emission(row: dict[str, Any]) -> bool:
    return action_label(row) in set(row.get("astar_actions") or [])


def is_true_action_emission(row: dict[str, Any]) -> bool:
    true_action = row.get("true_action")
    return isinstance(true_action, str) and action_label(row) == true_action


def result_group_key(row: dict[str, Any]) -> tuple[int, str, float | None]:
    alpha = row.get("alpha")
    alpha_value = None if alpha is None else float(alpha)
    return int(row["layer"]), str(row["condition"]), alpha_value


def summarize_result_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty result group")

    n = len(rows)
    counts = Counter(action_label(row) for row in rows)
    optimal_count = sum(1 for row in rows if is_optimal_emission(row))
    true_count = sum(1 for row in rows if is_true_action_emission(row))
    nonoptimal_true_rows = [row for row in rows if not bool(row.get("is_original_true_optimal"))]
    nonoptimal_to_optimal_count = sum(1 for row in nonoptimal_true_rows if is_optimal_emission(row))
    nonoptimal_denominator = len(nonoptimal_true_rows)

    layer, condition, alpha = result_group_key(rows[0])
    summary = {
        "layer": layer,
        "condition": condition,
        "alpha": alpha,
        "n": n,
        "optimal_action_count": optimal_count,
        "optimal_action_rate": optimal_count / n if n else 0.0,
        "true_action_count": true_count,
        "true_action_rate": true_count / n if n else 0.0,
        "nonoptimal_original_n": nonoptimal_denominator,
        "nonoptimal_to_optimal_count": nonoptimal_to_optimal_count,
        "nonoptimal_to_optimal_rate": (
            nonoptimal_to_optimal_count / nonoptimal_denominator if nonoptimal_denominator else 0.0
        ),
    }
    for label in ACTION_LABELS:
        summary[f"{label.lower()}_count"] = counts.get(label, 0)
        summary[f"{label.lower()}_rate"] = counts.get(label, 0) / n if n else 0.0
    return summary


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[result_group_key(row)].append(row)

    summaries = [summarize_result_group(group_rows) for group_rows in grouped.values()]
    by_key = {(row["layer"], row["condition"], row["alpha"]): row for row in summaries}
    for summary in summaries:
        layer = summary["layer"]
        no_hook = by_key.get((layer, "no_hook", None))
        alpha_zero = by_key.get((layer, "hooked", 0.0))
        for metric in ("optimal_action_rate", "true_action_rate", "nonoptimal_to_optimal_rate"):
            summary[f"{metric}_delta_vs_no_hook"] = summary[metric] - no_hook[metric] if no_hook is not None else None
            summary[f"{metric}_delta_vs_alpha0"] = (
                summary[metric] - alpha_zero[metric] if alpha_zero is not None else None
            )
    return sorted(
        summaries,
        key=lambda row: (
            int(row["layer"]),
            0 if row["condition"] == "no_hook" else 1,
            -math.inf if row["alpha"] is None else float(row["alpha"]),
        ),
    )


def build_noop_agreement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare no-hook baselines against alpha 0 hook baselines."""
    paired: dict[tuple[int, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = row.get("condition")
        alpha = row.get("alpha")
        if condition == "no_hook":
            label = "no_hook"
        elif condition == "hooked" and alpha is not None and float(alpha) == 0.0:
            label = "alpha0_hook"
        else:
            continue
        key = (
            int(row["layer"]),
            str(row["trajectory_name"]),
            int(row["step_id"]),
            int(row["sample_idx"]),
        )
        paired[key][label] = row

    by_layer: dict[int, Counter] = defaultdict(Counter)
    for key, pair in paired.items():
        layer = key[0]
        if "no_hook" not in pair or "alpha0_hook" not in pair:
            by_layer[layer]["missing_pairs"] += 1
            continue
        no_hook = pair["no_hook"]
        alpha0_hook = pair["alpha0_hook"]
        by_layer[layer]["pairs"] += 1
        by_layer[layer]["same_seed"] += int(no_hook.get("seed") == alpha0_hook.get("seed"))
        by_layer[layer]["same_action"] += int(action_label(no_hook) == action_label(alpha0_hook))
        by_layer[layer]["same_text_tail"] += int(no_hook.get("text_tail") == alpha0_hook.get("text_tail"))

    rows_out = []
    for layer in sorted(by_layer):
        counts = by_layer[layer]
        pairs = int(counts["pairs"])
        rows_out.append(
            {
                "layer": layer,
                "pairs": pairs,
                "missing_pairs": int(counts["missing_pairs"]),
                "same_seed_count": int(counts["same_seed"]),
                "same_seed_rate": counts["same_seed"] / pairs if pairs else 0.0,
                "same_action_count": int(counts["same_action"]),
                "same_action_rate": counts["same_action"] / pairs if pairs else 0.0,
                "same_text_tail_count": int(counts["same_text_tail"]),
                "same_text_tail_rate": counts["same_text_tail"] / pairs if pairs else 0.0,
            }
        )
    return rows_out


def build_paired_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize hooked conditions as paired interventions against no-hook."""
    baselines: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    hooked: dict[tuple[int, float], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for row in rows:
        if row.get("condition") != "no_hook":
            continue
        key = (
            int(row["layer"]),
            str(row["trajectory_name"]),
            int(row["step_id"]),
            int(row["sample_idx"]),
        )
        baselines[key] = row

    for row in rows:
        if row.get("condition") != "hooked" or row.get("alpha") is None:
            continue
        key = (
            int(row["layer"]),
            str(row["trajectory_name"]),
            int(row["step_id"]),
            int(row["sample_idx"]),
        )
        baseline = baselines.get(key)
        if baseline is None:
            continue
        hooked[(int(row["layer"]), float(row["alpha"]))].append((baseline, row))

    effect_rows = []
    for (layer, alpha), pairs in sorted(hooked.items()):
        counts = Counter()
        n = len(pairs)
        for baseline, row in pairs:
            baseline_action = action_label(baseline)
            hooked_action = action_label(row)
            true_action = row.get("true_action")
            baseline_optimal = is_optimal_emission(baseline)
            hooked_optimal = is_optimal_emission(row)
            baseline_true = isinstance(true_action, str) and baseline_action == true_action
            hooked_true = isinstance(true_action, str) and hooked_action == true_action

            counts["same_seed"] += int(baseline.get("seed") == row.get("seed"))
            counts["changed_action"] += int(baseline_action != hooked_action)
            counts["changed_text_tail"] += int(baseline.get("text_tail") != row.get("text_tail"))
            counts["nonoptimal_to_optimal"] += int((not baseline_optimal) and hooked_optimal)
            counts["optimal_to_nonoptimal"] += int(baseline_optimal and not hooked_optimal)
            counts["true_to_not_true"] += int(baseline_true and not hooked_true)
            counts["not_true_to_true"] += int((not baseline_true) and hooked_true)

        effect_rows.append(
            {
                "layer": layer,
                "alpha": alpha,
                "pairs": n,
                "same_seed_count": int(counts["same_seed"]),
                "same_seed_rate": counts["same_seed"] / n if n else 0.0,
                "changed_action_count": int(counts["changed_action"]),
                "changed_action_rate": counts["changed_action"] / n if n else 0.0,
                "changed_text_tail_count": int(counts["changed_text_tail"]),
                "changed_text_tail_rate": counts["changed_text_tail"] / n if n else 0.0,
                "nonoptimal_to_optimal_count": int(counts["nonoptimal_to_optimal"]),
                "nonoptimal_to_optimal_rate": counts["nonoptimal_to_optimal"] / n if n else 0.0,
                "optimal_to_nonoptimal_count": int(counts["optimal_to_nonoptimal"]),
                "optimal_to_nonoptimal_rate": counts["optimal_to_nonoptimal"] / n if n else 0.0,
                "true_to_not_true_count": int(counts["true_to_not_true"]),
                "true_to_not_true_rate": counts["true_to_not_true"] / n if n else 0.0,
                "not_true_to_true_count": int(counts["not_true_to_true"]),
                "not_true_to_true_rate": counts["not_true_to_true"] / n if n else 0.0,
            }
        )
    return effect_rows


def build_direction_control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize single-action direction controls against paired no-hook baselines."""
    baselines: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    groups: dict[tuple[int, float, str, bool], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for row in rows:
        if row.get("condition") != "no_hook":
            continue
        key = (
            int(row["layer"]),
            str(row["trajectory_name"]),
            int(row["step_id"]),
            int(row["sample_idx"]),
        )
        baselines[key] = row

    for row in rows:
        if (
            row.get("condition") != "hooked"
            or row.get("direction_mode")
            not in {
                "single_action",
                "action_contrast",
                "action_vs_true",
                "raw_action_contrast",
                "raw_action_vs_true",
                "raw_action_vs_baseline",
                "centroid_action_contrast",
                "centroid_action_vs_true",
            }
            or row.get("alpha") is None
        ):
            continue
        key = (
            int(row["layer"]),
            str(row["trajectory_name"]),
            int(row["step_id"]),
            int(row["sample_idx"]),
        )
        baseline = baselines.get(key)
        if baseline is None:
            continue
        groups[
            (
                int(row["layer"]),
                float(row["alpha"]),
                str(row["direction_label"]),
                bool(row.get("is_desired_direction")),
            )
        ].append((baseline, row))

    out_rows = []
    for (layer, alpha, label, is_desired), pairs in sorted(groups.items()):
        n = len(pairs)
        counts = Counter()
        for baseline, row in pairs:
            baseline_action = action_label(baseline)
            hooked_action = action_label(row)
            astar_actions = set(row.get("astar_actions") or [])
            counts["same_seed"] += int(baseline.get("seed") == row.get("seed"))
            counts["changed_action"] += int(baseline_action != hooked_action)
            counts["baseline_steered"] += int(baseline_action == label)
            counts["hooked_steered"] += int(hooked_action == label)
            counts["baseline_desired"] += int(baseline_action in astar_actions)
            counts["hooked_desired"] += int(hooked_action in astar_actions)
            counts["nonoptimal_to_desired"] += int(
                (baseline_action not in astar_actions) and (hooked_action in astar_actions)
            )
            counts["desired_to_nondesired"] += int(
                (baseline_action in astar_actions) and (hooked_action not in astar_actions)
            )

        baseline_steered_rate = counts["baseline_steered"] / n if n else 0.0
        hooked_steered_rate = counts["hooked_steered"] / n if n else 0.0
        baseline_desired_rate = counts["baseline_desired"] / n if n else 0.0
        hooked_desired_rate = counts["hooked_desired"] / n if n else 0.0
        out_rows.append(
            {
                "layer": layer,
                "alpha": alpha,
                "direction_label": label,
                "is_desired_direction": is_desired,
                "pairs": n,
                "same_seed_rate": counts["same_seed"] / n if n else 0.0,
                "changed_action_rate": counts["changed_action"] / n if n else 0.0,
                "baseline_steered_action_rate": baseline_steered_rate,
                "hooked_steered_action_rate": hooked_steered_rate,
                "steered_action_rate_delta": hooked_steered_rate - baseline_steered_rate,
                "baseline_desired_action_rate": baseline_desired_rate,
                "hooked_desired_action_rate": hooked_desired_rate,
                "desired_action_rate_delta": hooked_desired_rate - baseline_desired_rate,
                "nonoptimal_to_desired_rate": counts["nonoptimal_to_desired"] / n if n else 0.0,
                "desired_to_nondesired_rate": counts["desired_to_nondesired"] / n if n else 0.0,
            }
        )
    return out_rows


def build_direction_control_advantage_rows(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare desired directions to wrong directions at the same layer and alpha."""
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in control_rows:
        groups[(int(row["layer"]), float(row["alpha"]))].append(row)

    out_rows = []
    for (layer, alpha), rows in sorted(groups.items()):
        desired_rows = [row for row in rows if row["is_desired_direction"]]
        wrong_rows = [row for row in rows if not row["is_desired_direction"]]
        if not desired_rows or not wrong_rows:
            continue

        best_desired = max(
            desired_rows,
            key=lambda row: (float(row["desired_action_rate_delta"]), float(row["nonoptimal_to_desired_rate"])),
        )
        best_wrong = max(
            wrong_rows,
            key=lambda row: (float(row["desired_action_rate_delta"]), float(row["nonoptimal_to_desired_rate"])),
        )
        desired_delta = float(best_desired["desired_action_rate_delta"])
        wrong_delta = float(best_wrong["desired_action_rate_delta"])
        desired_conversion = float(best_desired["nonoptimal_to_desired_rate"])
        wrong_conversion = float(best_wrong["nonoptimal_to_desired_rate"])
        out_rows.append(
            {
                "layer": layer,
                "alpha": alpha,
                "desired_direction_label": best_desired["direction_label"],
                "desired_pairs": best_desired["pairs"],
                "desired_action_rate_delta": desired_delta,
                "best_wrong_direction_label": best_wrong["direction_label"],
                "best_wrong_pairs": best_wrong["pairs"],
                "best_wrong_desired_action_rate_delta": wrong_delta,
                "desired_advantage_delta": desired_delta - wrong_delta,
                "desired_nonoptimal_to_desired_rate": desired_conversion,
                "best_wrong_nonoptimal_to_desired_rate": wrong_conversion,
                "nonoptimal_to_desired_advantage": desired_conversion - wrong_conversion,
                "supports_desired_direction": (
                    desired_delta > 0.0 and desired_delta > wrong_delta and desired_conversion > wrong_conversion
                ),
            }
        )
    return out_rows


def build_direction_control_overall_rows(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate desired-direction and wrong-direction controls across action labels."""
    grouped: dict[tuple[int, float, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in control_rows:
        grouped[(int(row["layer"]), float(row["alpha"]), bool(row["is_desired_direction"]))].append(row)

    out_rows = []
    for (layer, alpha, is_desired), rows in sorted(grouped.items()):
        pairs = sum(int(row["pairs"]) for row in rows)
        if pairs == 0:
            continue

        baseline_desired = sum(float(row["baseline_desired_action_rate"]) * int(row["pairs"]) for row in rows)
        hooked_desired = sum(float(row["hooked_desired_action_rate"]) * int(row["pairs"]) for row in rows)
        changed_action = sum(float(row["changed_action_rate"]) * int(row["pairs"]) for row in rows)
        nonoptimal_to_desired = sum(float(row["nonoptimal_to_desired_rate"]) * int(row["pairs"]) for row in rows)
        desired_to_nondesired = sum(float(row["desired_to_nondesired_rate"]) * int(row["pairs"]) for row in rows)

        baseline_rate = baseline_desired / pairs
        hooked_rate = hooked_desired / pairs
        out_rows.append(
            {
                "layer": layer,
                "alpha": alpha,
                "is_desired_direction": is_desired,
                "direction_group": "desired" if is_desired else "wrong",
                "direction_labels": ",".join(sorted({str(row["direction_label"]) for row in rows})),
                "pairs": pairs,
                "changed_action_rate": changed_action / pairs,
                "baseline_desired_action_rate": baseline_rate,
                "hooked_desired_action_rate": hooked_rate,
                "desired_action_rate_delta": hooked_rate - baseline_rate,
                "nonoptimal_to_desired_rate": nonoptimal_to_desired / pairs,
                "desired_to_nondesired_rate": desired_to_nondesired / pairs,
            }
        )
    return out_rows


def build_direction_control_overall_advantage_rows(overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare overall desired-direction controls against overall wrong-direction controls."""
    groups: dict[tuple[int, float], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in overall_rows:
        groups[(int(row["layer"]), float(row["alpha"]))][bool(row["is_desired_direction"])] = row

    out_rows = []
    for (layer, alpha), rows_by_kind in sorted(groups.items()):
        desired = rows_by_kind.get(True)
        wrong = rows_by_kind.get(False)
        if desired is None or wrong is None:
            continue
        desired_delta = float(desired["desired_action_rate_delta"])
        wrong_delta = float(wrong["desired_action_rate_delta"])
        desired_conversion = float(desired["nonoptimal_to_desired_rate"])
        wrong_conversion = float(wrong["nonoptimal_to_desired_rate"])
        out_rows.append(
            {
                "layer": layer,
                "alpha": alpha,
                "desired_pairs": desired["pairs"],
                "wrong_pairs": wrong["pairs"],
                "desired_action_rate_delta": desired_delta,
                "wrong_desired_action_rate_delta": wrong_delta,
                "desired_advantage_delta": desired_delta - wrong_delta,
                "desired_nonoptimal_to_desired_rate": desired_conversion,
                "wrong_nonoptimal_to_desired_rate": wrong_conversion,
                "nonoptimal_to_desired_advantage": desired_conversion - wrong_conversion,
                "supports_desired_direction": (
                    desired_delta > 0.0 and desired_delta > wrong_delta and desired_conversion > wrong_conversion
                ),
            }
        )
    return out_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_rate(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def line_plot_svg(
    series_by_label: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    y_label: str,
    path: Path,
    width: int = 760,
    height: int = 420,
) -> None:
    """Write a small dependency-free SVG line plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left = 70
    margin_right = 28
    margin_top = 48
    margin_bottom = 56
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_points = [point for points in series_by_label.values() for point in points]
    if not all_points:
        path.write_text("")
        return

    xs = [point[0] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    min_y, max_y = 0.0, 1.0

    def sx(x: float) -> float:
        return margin_left + ((x - min_x) / (max_x - min_x)) * plot_w

    def sy(y: float) -> float:
        return margin_top + (1.0 - ((y - min_y) / (max_y - min_y))) * plot_h

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" '
        f'y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" font-family="sans-serif" '
        'font-size="13">alpha</text>',
        f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" '
        f'text-anchor="middle" font-family="sans-serif" font-size="13">{y_label}</text>',
    ]

    for tick in range(0, 6):
        y_value = tick / 5
        y = sy(y_value)
        parts.append(
            f'<line x1="{margin_left - 4}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11">{y_value:.1f}</text>'
        )

    x_ticks = sorted(set(xs))
    for x_value in x_ticks:
        x = sx(x_value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{margin_top + plot_h}" x2="{x:.1f}" y2="{margin_top + plot_h + 4}" stroke="#333"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{margin_top + plot_h + 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{x_value:g}</text>'
        )

    legend_y = margin_top + 8
    for idx, (label, points) in enumerate(sorted(series_by_label.items())):
        color = palette[idx % len(palette)]
        points_sorted = sorted(points)
        polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points_sorted)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')
        for x, y in points_sorted:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>')
        lx = width - margin_right - 135
        ly = legend_y + idx * 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{lx + 24}" y="{ly + 4}" font-family="sans-serif" font-size="12">{label}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_report_markdown(  # noqa: PLR0912
    report_path: Path,
    summary_rows: list[dict[str, Any]],
    noop_rows: list[dict[str, Any]],
    paired_effect_rows: list[dict[str, Any]],
    direction_control_rows: list[dict[str, Any]],
    direction_advantage_rows: list[dict[str, Any]],
    direction_overall_rows: list[dict[str, Any]],
    direction_overall_advantage_rows: list[dict[str, Any]],
    results_path: Path,
) -> None:
    hooked_rows = [row for row in summary_rows if row["condition"] == "hooked"]
    layers = sorted({int(row["layer"]) for row in summary_rows})
    alpha_values = sorted({float(row["alpha"]) for row in hooked_rows})

    def metric_or(row: dict[str, Any], metric: str, fallback: float) -> float:
        value = row.get(metric)
        return fallback if value is None else float(value)

    best_by_optimal = max(
        hooked_rows,
        key=lambda row: (
            metric_or(row, "optimal_action_rate_delta_vs_no_hook", -math.inf),
            row["optimal_action_rate"],
        ),
    )
    best_by_true_loss = min(
        hooked_rows,
        key=lambda row: metric_or(row, "true_action_rate_delta_vs_no_hook", math.inf),
    )
    best_changed = max(
        paired_effect_rows,
        key=lambda row: (float(row["changed_action_rate"]), float(abs(row["alpha"]))),
        default=None,
    )
    best_true_change = max(
        paired_effect_rows,
        key=lambda row: (float(row["true_to_not_true_rate"]), float(abs(row["alpha"]))),
        default=None,
    )
    best_nonoptimal_to_optimal = max(
        paired_effect_rows,
        key=lambda row: (float(row["nonoptimal_to_optimal_rate"]), float(abs(row["alpha"]))),
        default=None,
    )

    out = []
    out.append("# Seed12 Optimal-Action Direction Steering\n\n")
    out.append("## Setup\n\n")
    out.append(
        "This report aggregates real Seed12 trajectory-step steering results. The hooked condition adds "
        "`alpha * class_gap * direction` to each of the final three step-specific prompt-suffix tokens. "
        "The no-hook baseline is separate from the alpha 0 hook baseline.\n\n"
    )
    out.append(f"- Results: `{results_path}`\n")
    out.append(f"- Layers: {', '.join(str(layer) for layer in layers)}\n")
    out.append(f"- Hooked alphas: {', '.join(f'{alpha:g}' for alpha in alpha_values)}\n\n")

    out.append("## Main Rates\n\n")
    out.append(
        "| layer | condition | alpha | n | optimal action rate | true action rate | "
        "nonoptimal-to-optimal rate | delta optimal vs no-hook | delta true vs no-hook |\n"
    )
    out.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in summary_rows:
        alpha = "" if row["alpha"] is None else f"{float(row['alpha']):g}"
        out.append(
            f"| {row['layer']} | {row['condition']} | {alpha} | {row['n']} | "
            f"{format_rate(row['optimal_action_rate'])} | {format_rate(row['true_action_rate'])} | "
            f"{format_rate(row['nonoptimal_to_optimal_rate'])} | "
            f"{format_rate(row.get('optimal_action_rate_delta_vs_no_hook'))} | "
            f"{format_rate(row.get('true_action_rate_delta_vs_no_hook'))} |\n"
        )
    out.append("\n")

    out.append("## Alpha 0 No-Op Check\n\n")
    out.append("| layer | pairs | same seed rate | same action rate | same text-tail rate | missing pairs |\n")
    out.append("|---:|---:|---:|---:|---:|---:|\n")
    for row in noop_rows:
        out.append(
            f"| {row['layer']} | {row['pairs']} | {format_rate(row['same_seed_rate'])} | "
            f"{format_rate(row['same_action_rate'])} | {format_rate(row['same_text_tail_rate'])} | "
            f"{row['missing_pairs']} |\n"
        )
    out.append("\n")

    out.append("## Paired Causal Effects\n\n")
    out.append(
        "| layer | alpha | pairs | changed action rate | nonoptimal-to-optimal rate | "
        "optimal-to-nonoptimal rate | true-to-not-true rate |\n"
    )
    out.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in paired_effect_rows:
        out.append(
            f"| {row['layer']} | {float(row['alpha']):g} | {row['pairs']} | "
            f"{format_rate(row['changed_action_rate'])} | "
            f"{format_rate(row['nonoptimal_to_optimal_rate'])} | "
            f"{format_rate(row['optimal_to_nonoptimal_rate'])} | "
            f"{format_rate(row['true_to_not_true_rate'])} |\n"
        )
    out.append("\n")

    if direction_control_rows:
        out.append("## Direction Controls\n\n")
        out.append(
            "| layer | alpha | direction | desired dir | pairs | steered-action delta | "
            "desired-action delta | nonoptimal-to-desired rate |\n"
        )
        out.append("|---:|---:|---|---|---:|---:|---:|---:|\n")
        for row in direction_control_rows:
            out.append(
                f"| {row['layer']} | {float(row['alpha']):g} | {row['direction_label']} | "
                f"{row['is_desired_direction']} | {row['pairs']} | "
                f"{format_rate(row['steered_action_rate_delta'])} | "
                f"{format_rate(row['desired_action_rate_delta'])} | "
                f"{format_rate(row['nonoptimal_to_desired_rate'])} |\n"
            )
        out.append("\n")

    if direction_advantage_rows:
        out.append("## Direction-Control Advantage\n\n")
        out.append(
            "| layer | alpha | desired direction | desired delta | best wrong direction | "
            "best wrong delta | desired advantage | supports desired direction |\n"
        )
        out.append("|---:|---:|---|---:|---|---:|---:|---|\n")
        for row in direction_advantage_rows:
            out.append(
                f"| {row['layer']} | {float(row['alpha']):g} | {row['desired_direction_label']} | "
                f"{format_rate(row['desired_action_rate_delta'])} | {row['best_wrong_direction_label']} | "
                f"{format_rate(row['best_wrong_desired_action_rate_delta'])} | "
                f"{format_rate(row['desired_advantage_delta'])} | "
                f"{row['supports_desired_direction']} |\n"
            )
        out.append("\n")

    if direction_overall_rows:
        out.append("## Overall Direction Controls\n\n")
        out.append(
            "| layer | alpha | direction group | pairs | desired-action delta | "
            "nonoptimal-to-desired rate | desired-to-nondesired rate |\n"
        )
        out.append("|---:|---:|---|---:|---:|---:|---:|\n")
        for row in direction_overall_rows:
            out.append(
                f"| {row['layer']} | {float(row['alpha']):g} | {row['direction_group']} | "
                f"{row['pairs']} | {format_rate(row['desired_action_rate_delta'])} | "
                f"{format_rate(row['nonoptimal_to_desired_rate'])} | "
                f"{format_rate(row['desired_to_nondesired_rate'])} |\n"
            )
        out.append("\n")

    if direction_overall_advantage_rows:
        out.append("## Overall Direction-Control Advantage\n\n")
        out.append(
            "| layer | alpha | desired delta | wrong delta | desired advantage | supports desired direction |\n"
        )
        out.append("|---:|---:|---:|---:|---:|---|\n")
        for row in direction_overall_advantage_rows:
            out.append(
                f"| {row['layer']} | {float(row['alpha']):g} | "
                f"{format_rate(row['desired_action_rate_delta'])} | "
                f"{format_rate(row['wrong_desired_action_rate_delta'])} | "
                f"{format_rate(row['desired_advantage_delta'])} | "
                f"{row['supports_desired_direction']} |\n"
            )
        out.append("\n")

    out.append("## Interpretation\n\n")
    best_optimal_delta = float(best_by_optimal.get("optimal_action_rate_delta_vs_no_hook") or 0.0)
    if best_optimal_delta > 0.0:
        out.append(
            f"The largest optimal-action improvement versus no-hook is layer {best_by_optimal['layer']} at "
            f"alpha {float(best_by_optimal['alpha']):g}: "
            f"optimal-action rate {format_rate(best_by_optimal['optimal_action_rate'])}, "
            f"delta {format_rate(best_by_optimal.get('optimal_action_rate_delta_vs_no_hook'))}.\n\n"
        )
    else:
        out.append(
            "No hooked condition improved optimal-action rate versus no-hook. The least-negative condition is "
            f"layer {best_by_optimal['layer']} at alpha {float(best_by_optimal['alpha']):g}: "
            f"optimal-action rate {format_rate(best_by_optimal['optimal_action_rate'])}, "
            f"delta {format_rate(best_by_optimal.get('optimal_action_rate_delta_vs_no_hook'))}.\n\n"
        )

    best_true_delta = float(best_by_true_loss.get("true_action_rate_delta_vs_no_hook") or 0.0)
    if best_true_delta < 0.0:
        out.append(
            f"The largest true-action retention drop versus no-hook is layer {best_by_true_loss['layer']} at "
            f"alpha {float(best_by_true_loss['alpha']):g}: "
            f"true-action rate {format_rate(best_by_true_loss['true_action_rate'])}, "
            f"delta {format_rate(best_by_true_loss.get('true_action_rate_delta_vs_no_hook'))}.\n\n"
        )
    else:
        out.append(
            "No hooked condition reduced true-action retention versus no-hook. The smallest true-action-rate "
            f"delta is layer {best_by_true_loss['layer']} at alpha {float(best_by_true_loss['alpha']):g}: "
            f"true-action rate {format_rate(best_by_true_loss['true_action_rate'])}, "
            f"delta {format_rate(best_by_true_loss.get('true_action_rate_delta_vs_no_hook'))}.\n\n"
        )
    if best_changed is not None:
        out.append(
            f"The strongest paired causal effect on emitted action is layer {best_changed['layer']} at "
            f"alpha {float(best_changed['alpha']):g}: "
            f"{format_rate(best_changed['changed_action_rate'])} of paired samples changed final action under "
            "the same generation seed.\n\n"
        )
    if best_true_change is not None:
        out.append(
            f"The strongest paired effect away from the recorded true action is layer {best_true_change['layer']} "
            f"at alpha {float(best_true_change['alpha']):g}: "
            f"{format_rate(best_true_change['true_to_not_true_rate'])} of pairs changed from true-action emission "
            "to a different final action.\n\n"
        )
    if best_nonoptimal_to_optimal is not None:
        out.append(
            f"The strongest paired nonoptimal-to-optimal conversion is layer {best_nonoptimal_to_optimal['layer']} "
            f"at alpha {float(best_nonoptimal_to_optimal['alpha']):g}: "
            f"{format_rate(best_nonoptimal_to_optimal['nonoptimal_to_optimal_rate'])} of pairs converted from "
            "a nonoptimal emitted action to an action in `astar_actions`.\n\n"
        )
    overall_supports_desired = (
        any(bool(row["supports_desired_direction"]) for row in direction_overall_advantage_rows)
        if direction_overall_advantage_rows
        else None
    )
    if direction_advantage_rows:
        supported_advantage_rows = [row for row in direction_advantage_rows if bool(row["supports_desired_direction"])]
        best_advantage = max(
            supported_advantage_rows or direction_advantage_rows,
            key=lambda row: (
                float(row["desired_advantage_delta"]),
                float(row["desired_action_rate_delta"]),
                float(row["nonoptimal_to_desired_advantage"]),
            ),
        )
        if bool(best_advantage["supports_desired_direction"]):
            if overall_supports_desired is False:
                out.append(
                    "The per-action direction-control table contains a local positive slice, but the overall "
                    "desired-vs-wrong control fails the strict support criterion. The strongest local slice is layer "
                    f"{best_advantage['layer']} alpha {float(best_advantage['alpha']):g}, desired direction "
                    f"{best_advantage['desired_direction_label']} delta "
                    f"{format_rate(best_advantage['desired_action_rate_delta'])}, best wrong direction "
                    f"{best_advantage['best_wrong_direction_label']} delta "
                    f"{format_rate(best_advantage['best_wrong_desired_action_rate_delta'])}, advantage "
                    f"{format_rate(best_advantage['desired_advantage_delta'])}.\n\n"
                )
            else:
                out.append(
                    "The per-action same-alpha direction-control check supports desired-direction steering at "
                    f"its strongest tested setting: layer {best_advantage['layer']} alpha "
                    f"{float(best_advantage['alpha']):g}, desired direction "
                    f"{best_advantage['desired_direction_label']} delta "
                    f"{format_rate(best_advantage['desired_action_rate_delta'])}, best wrong direction "
                    f"{best_advantage['best_wrong_direction_label']} delta "
                    f"{format_rate(best_advantage['best_wrong_desired_action_rate_delta'])}, advantage "
                    f"{format_rate(best_advantage['desired_advantage_delta'])}.\n\n"
                )
        else:
            out.append(
                "The per-action same-alpha direction-control check does not support desired-direction steering. "
                "Strict support requires both desired-action-rate and nonoptimal-to-desired conversion advantages. "
                f"The best tested desired-vs-wrong advantage is layer {best_advantage['layer']} alpha "
                f"{float(best_advantage['alpha']):g}: desired direction "
                f"{best_advantage['desired_direction_label']} delta "
                f"{format_rate(best_advantage['desired_action_rate_delta'])}, best wrong direction "
                f"{best_advantage['best_wrong_direction_label']} delta "
                f"{format_rate(best_advantage['best_wrong_desired_action_rate_delta'])}, advantage "
                f"{format_rate(best_advantage['desired_advantage_delta'])}.\n\n"
            )
    if direction_overall_advantage_rows:
        supported_overall_rows = [
            row for row in direction_overall_advantage_rows if bool(row["supports_desired_direction"])
        ]
        best_overall_advantage = max(
            supported_overall_rows or direction_overall_advantage_rows,
            key=lambda row: (
                float(row["desired_advantage_delta"]),
                float(row["desired_action_rate_delta"]),
                float(row["nonoptimal_to_desired_advantage"]),
            ),
        )
        if bool(best_overall_advantage["supports_desired_direction"]):
            out.append(
                "The overall desired-vs-wrong control is strictly positive at its strongest tested setting: "
                f"layer {best_overall_advantage['layer']} alpha {float(best_overall_advantage['alpha']):g}, "
                f"desired-direction delta {format_rate(best_overall_advantage['desired_action_rate_delta'])}, "
                f"wrong-direction delta {format_rate(best_overall_advantage['wrong_desired_action_rate_delta'])}, "
                f"advantage {format_rate(best_overall_advantage['desired_advantage_delta'])}, "
                "nonoptimal-to-desired advantage "
                f"{format_rate(best_overall_advantage['nonoptimal_to_desired_advantage'])}.\n\n"
            )
        else:
            out.append(
                "The overall desired-vs-wrong control fails the strict support criterion. The best tested setting is "
                f"layer {best_overall_advantage['layer']} alpha {float(best_overall_advantage['alpha']):g}, "
                f"desired-direction delta {format_rate(best_overall_advantage['desired_action_rate_delta'])}, "
                f"wrong-direction delta {format_rate(best_overall_advantage['wrong_desired_action_rate_delta'])}, "
                f"advantage {format_rate(best_overall_advantage['desired_advantage_delta'])}, "
                "nonoptimal-to-desired advantage "
                f"{format_rate(best_overall_advantage['nonoptimal_to_desired_advantage'])}.\n\n"
            )
    out.append(
        "These are distinct behavioral claims. A higher optimal-action rate means the model more often emits an "
        "action in `astar_actions`. A lower true-action rate means the model moved away from the recorded "
        "`agent_action`. Those can move in opposite directions because many recorded true actions are not in "
        "the optimal set.\n\n"
    )

    out.append("## Artifacts\n\n")
    out.append("- `summary_by_layer_alpha.csv`\n")
    out.append("- `noop_baseline_agreement.csv`\n")
    out.append("- `paired_effects_vs_no_hook.csv`\n")
    if direction_control_rows:
        out.append("- `direction_control_effects.csv`\n")
    if direction_advantage_rows:
        out.append("- `direction_control_advantage.csv`\n")
    if direction_overall_rows:
        out.append("- `direction_control_overall.csv`\n")
    if direction_overall_advantage_rows:
        out.append("- `direction_control_overall_advantage.csv`\n")
    out.append("- `optimal_action_rate_vs_alpha.svg`\n")
    out.append("- `true_action_rate_vs_alpha.svg`\n")
    out.append("- `nonoptimal_to_optimal_rate_vs_alpha.svg`\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(out))


def run_report(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    report_dir = Path(args.report_dir)
    rows = read_jsonl(results_path)
    if not rows:
        raise SystemExit(f"No rows found in {results_path}")
    selected_layers = set(parse_layer_selection(getattr(args, "layers", None)))
    rows = [row for row in rows if int(row["layer"]) in selected_layers]
    if not rows:
        raise SystemExit(f"No rows for selected layers {sorted(selected_layers)} in {results_path}")

    summary_rows = aggregate_results(rows)
    write_csv(report_dir / "summary_by_layer_alpha.csv", summary_rows)
    noop_rows = build_noop_agreement_rows(rows)
    write_csv(report_dir / "noop_baseline_agreement.csv", noop_rows)
    paired_effect_rows = build_paired_effect_rows(rows)
    write_csv(report_dir / "paired_effects_vs_no_hook.csv", paired_effect_rows)
    direction_control_rows = build_direction_control_rows(rows)
    if direction_control_rows:
        write_csv(report_dir / "direction_control_effects.csv", direction_control_rows)
    direction_advantage_rows = build_direction_control_advantage_rows(direction_control_rows)
    if direction_advantage_rows:
        write_csv(report_dir / "direction_control_advantage.csv", direction_advantage_rows)
    direction_overall_rows = build_direction_control_overall_rows(direction_control_rows)
    if direction_overall_rows:
        write_csv(report_dir / "direction_control_overall.csv", direction_overall_rows)
    direction_overall_advantage_rows = build_direction_control_overall_advantage_rows(direction_overall_rows)
    if direction_overall_advantage_rows:
        write_csv(report_dir / "direction_control_overall_advantage.csv", direction_overall_advantage_rows)

    hooked_rows = [row for row in summary_rows if row["condition"] == "hooked" and row["alpha"] is not None]
    for metric, title, filename in (
        ("optimal_action_rate", "Optimal-action rate vs alpha", "optimal_action_rate_vs_alpha.svg"),
        ("true_action_rate", "True-action retention vs alpha", "true_action_rate_vs_alpha.svg"),
        (
            "nonoptimal_to_optimal_rate",
            "Nonoptimal-to-optimal conversion vs alpha",
            "nonoptimal_to_optimal_rate_vs_alpha.svg",
        ),
    ):
        series: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in hooked_rows:
            series[f"layer {row['layer']}"].append((float(row["alpha"]), float(row[metric])))
        line_plot_svg(dict(series), title=title, y_label=metric, path=report_dir / filename)

    for action in ACTION_LABELS:
        series = defaultdict(list)
        metric = f"{action.lower()}_rate"
        for row in hooked_rows:
            series[f"layer {row['layer']}"].append((float(row["alpha"]), float(row[metric])))
        line_plot_svg(
            dict(series),
            title=f"{action} emission rate vs alpha",
            y_label=metric,
            path=report_dir / f"{action.lower()}_rate_vs_alpha.svg",
        )

    write_report_markdown(
        report_dir / "seed12_optimal_action_steering_report.md",
        summary_rows,
        noop_rows,
        paired_effect_rows,
        direction_control_rows,
        direction_advantage_rows,
        direction_overall_rows,
        direction_overall_advantage_rows,
        results_path,
    )
    print(f"Wrote report artifacts to {report_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-directions", help="Extract activation-space action directions.")
    prepare.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    prepare.add_argument("--trajectories-dir", type=Path, default=DEFAULT_TRAJECTORIES_DIR)
    prepare.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    prepare.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    prepare.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    prepare.add_argument("--max-trajectories", type=int, default=None)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=run_prepare_directions)

    steer = sub.add_parser("steer", help="Run sampled MLX steering over real Seed12 steps.")
    steer.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    steer.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    steer.add_argument("--trajectories-dir", type=Path, default=DEFAULT_TRAJECTORIES_DIR)
    steer.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    steer.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "steering_results.jsonl")
    steer.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    steer.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    steer.add_argument(
        "--direction-set",
        choices=(
            "optimal",
            "all-actions",
            "action-contrast",
            "action-vs-true",
            "raw-action-contrast",
            "raw-action-vs-true",
            "raw-action-vs-baseline",
            "centroid-action-contrast",
            "centroid-action-vs-true",
        ),
        default="optimal",
        help=(
            "Which directions to test. 'optimal' averages the astar action directions; "
            "'all-actions' tests each single action direction as desired/wrong controls; "
            "'action-contrast' tests each action direction minus the mean of the other action directions; "
            "'action-vs-true' tests each unit action direction minus the recorded true action direction; "
            "'raw-action-contrast' and 'raw-action-vs-true' use raw activation-space probe-logit gradients; "
            "'raw-action-vs-baseline' uses the paired no-hook emitted action as the source action; "
            "'centroid-action-contrast' and 'centroid-action-vs-true' use optimal-action activation centroids."
        ),
    )
    steer.add_argument("--n-samples", type=int, default=1)
    steer.add_argument("--temperature", type=float, default=0.7)
    steer.add_argument("--max-new-tokens", type=int, default=4096)
    steer.add_argument("--seed", type=int, default=42)
    steer.add_argument("--max-trajectories", type=int, default=None)
    steer.add_argument("--max-steps", type=int, default=None)
    steer.add_argument("--max-steps-per-trajectory", type=int, default=None)
    steer.add_argument(
        "--step-ordinals",
        type=int,
        nargs="+",
        default=None,
        help="Optional eligible-step ordinals to evaluate, using zero-based ordinals from the full eligible set.",
    )
    steer.add_argument("--baseline-only", action="store_true", help="Only run no-hook generations for screening.")
    steer.add_argument("--overwrite", action="store_true")
    steer.set_defaults(func=run_steering)

    report = sub.add_parser("report", help="Aggregate JSONL steering results into report artifacts.")
    report.add_argument("--results", type=Path, default=DEFAULT_OUT_DIR / "steering_results.jsonl")
    report.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    report.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    report.set_defaults(func=run_report)

    return parser


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/interp_matplotlib")
    parser = build_parser()
    args = parser.parse_args()
    args.layers = parse_layer_selection(getattr(args, "layers", None))
    args.func(args)


if __name__ == "__main__":
    main()
