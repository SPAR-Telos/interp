"""Train Seed12 action probes across pre-reasoning, CoT ranks, and post-tail positions.

The expected activation layout is the Seed12 state sweep gathered with:
``prompt_suffix_indices=-3:-1`` and ``output_indices=-16:-14,@analysis/10``.
Each probe uses a single layer activation vector, so the GPT-OSS-20B hidden size
is expected to be 2880 for the current data. The default dataset variant trains
one sample per trajectory step; ``state_average`` first averages activations for
each ``(agent position, has_key, door_open)`` state.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from telos_interp.activation_loading import (
    discover_available_token_indices,
    discover_model_folder,
    discover_trajectory_folders,
    load_activation,
    parse_index_specification,
)
from telos_interp.probe_models import MLPProbe
from telos_interp.training import compute_normalization_params, normalize_activations, resolve_device, set_seed
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DEFAULT_ACTIVATIONS_DIR = Path("data/activations/Seed12_T0_state_sweep")
DEFAULT_TRAJECTORIES_DIR = Path("data/trajectories/Seed12_T0_state_sweep")
DEFAULT_OUTPUT_DIR = Path("data/probes/Seed12_T0_state_sweep_action")
DEFAULT_STATE_AVG_OUTPUT_DIR = Path("data/probes/Seed12_T0_state_sweep_action_state_avg")

ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}

LabelType = Literal["true_action", "optimal_action"]
ModelType = Literal["linear", "mlp"]
PositionKind = Literal["pre_suffix", "cot_rank", "post_tail"]
DatasetVariant = Literal["step", "state_average"]


@dataclass(frozen=True)
class PositionSpec:
    """One activation position to probe."""

    name: str
    kind: PositionKind
    cot_rank: int | None = None


@dataclass(frozen=True)
class TrajectoryRecord:
    """A trajectory JSON paired with its activation folder."""

    name: str
    activation_folder: Path
    trajectory_path: Path
    trajectory: dict[str, Any]


@dataclass
class ActionProbeDataset:
    """In-memory supervised dataset for one label type, position, and layer."""

    activations: torch.Tensor
    labels: torch.Tensor
    split_group_ids: list[str]
    skipped: dict[str, int]
    label_distribution: dict[str, int]
    metadata: dict[str, Any]


def parse_int_selection(spec: str) -> list[int]:
    """Parse a comma-separated inclusive integer selection such as ``7,15,23`` or ``0:29``."""
    indices: list[int] = []
    seen: set[int] = set()

    for part_raw in spec.split(","):
        part = part_raw.strip()
        if not part:
            continue

        if ":" in part:
            start_raw, end_raw = part.split(":", maxsplit=1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            step = 1 if start <= end else -1
            values = range(start, end + step, step)
        else:
            values = [int(part)]

        for value in values:
            if value not in seen:
                indices.append(value)
                seen.add(value)

    return indices


def parse_csv_choices(value: str, valid: set[str], name: str) -> list[str]:
    """Parse and validate a comma-separated choice list."""
    choices = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [choice for choice in choices if choice not in valid]
    if invalid:
        raise ValueError(f"Invalid {name}: {invalid}. Valid choices: {sorted(valid)}")
    if not choices:
        raise ValueError(f"At least one {name} must be selected")
    return choices


def build_position_specs(
    cot_ranks: list[int], include_pre_suffix: bool, include_post_tail: bool
) -> list[PositionSpec]:
    """Build the ordered activation positions for the sweep."""
    positions: list[PositionSpec] = []
    if include_pre_suffix:
        positions.append(PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix"))
    positions.extend(PositionSpec(name=f"cot_rank_{rank}", kind="cot_rank", cot_rank=rank) for rank in cot_ranks)
    if include_post_tail:
        positions.append(PositionSpec(name="post_reasoning_tail", kind="post_tail"))
    return positions


def resolve_post_tail_indices(output_tokens: list[dict[str, Any]]) -> list[int]:
    """Return relative output indices for the post-reasoning tail ``-16:-14``."""
    n_tokens = len(output_tokens)
    indices = [n_tokens + offset for offset in range(-16, -13)]
    if any(idx < 0 or idx >= n_tokens for idx in indices):
        return []
    return indices


def resolve_cot_checkpoint_indices(output_tokens: list[dict[str, Any]]) -> list[int]:
    """Return every 10th output token whose token groups include ``analysis``."""
    analysis_indices = []
    for fallback_idx, token in enumerate(output_tokens):
        groups = token.get("token_groups", [])
        if "analysis" in groups:
            analysis_indices.append(int(token.get("id", fallback_idx)))
    return analysis_indices[::10]


def select_cot_rank_index(output_tokens: list[dict[str, Any]], cot_rank: int) -> tuple[int | None, str | None]:
    """Select one CoT checkpoint index, excluding indices reserved for the post-tail probe."""
    checkpoints = resolve_cot_checkpoint_indices(output_tokens)
    if cot_rank >= len(checkpoints):
        return None, "missing_cot_rank"

    token_idx = checkpoints[cot_rank]
    if token_idx in set(resolve_post_tail_indices(output_tokens)):
        return None, "cot_rank_overlaps_post_tail"
    return token_idx, None


def parse_true_action_label(step: dict[str, Any]) -> int | None:
    """Return a single-class true action label, or None for non-action steps."""
    action = step.get("agent_action")
    if not isinstance(action, str):
        return None
    return ACTION_TO_IDX.get(action.strip().upper())


def parse_optimal_action_label(step: dict[str, Any]) -> torch.Tensor | None:
    """Return a multi-hot optimal action target, preserving ties."""
    raw_actions = step.get("astar_actions")
    if not isinstance(raw_actions, list):
        return None

    label = torch.zeros(len(ACTIONS), dtype=torch.float32)
    for raw_action in raw_actions:
        if not isinstance(raw_action, str):
            continue
        action_idx = ACTION_TO_IDX.get(raw_action.strip().upper())
        if action_idx is not None:
            label[action_idx] = 1.0

    if label.sum().item() == 0:
        return None
    return label


def parse_action_label(step: dict[str, Any], label_type: LabelType) -> int | torch.Tensor | None:
    """Parse the requested action label from a trajectory step."""
    if label_type == "true_action":
        return parse_true_action_label(step)
    if label_type == "optimal_action":
        return parse_optimal_action_label(step)
    raise ValueError(f"Unknown label_type: {label_type}")


def parse_agent_position_from_grid_state(grid_state: Any) -> tuple[int, int] | None:
    """Parse the ``A`` marker coordinate from a rendered grid_state."""
    if not isinstance(grid_state, list):
        return None

    for row_fallback, row_raw in enumerate(grid_state):
        if not isinstance(row_raw, str):
            continue
        cells = row_raw.split()
        if len(cells) < 2:
            continue

        try:
            row_idx = int(cells[0])
            grid_cells = cells[1:]
        except ValueError:
            row_idx = row_fallback
            grid_cells = cells

        for col_idx, cell in enumerate(grid_cells):
            if cell == "A":
                return row_idx, col_idx
    return None


def format_state_key(agent_position: tuple[int, int], has_key: bool, door_open: bool) -> str:
    """Return a stable state key for grouping and splitting."""
    row_idx, col_idx = agent_position
    return f"pos={row_idx},{col_idx}|has_key={int(has_key)}|door_open={int(door_open)}"


def build_state_key(step: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Build the state-average grouping key from a trajectory step."""
    agent_position = parse_agent_position_from_grid_state(step.get("grid_state"))
    if agent_position is None:
        return None, None, "missing_agent_position"

    has_key = bool(step.get("carrying_key"))
    door_open = bool(step.get("door_open"))
    state_key = format_state_key(agent_position, has_key, door_open)
    state_info = {
        "agent_position": agent_position,
        "has_key": has_key,
        "door_open": door_open,
    }
    return state_key, state_info, None


def label_signature(label: int | torch.Tensor) -> int | tuple[int, ...]:
    """Return a comparable label signature for conflict detection."""
    if isinstance(label, torch.Tensor):
        return tuple(int(value) for value in label.detach().cpu().tolist())
    return int(label)


def label_debug_value(label: int | torch.Tensor, label_type: LabelType) -> str:
    """Format a label for diagnostics."""
    if label_type == "true_action":
        return IDX_TO_ACTION[int(label)]
    if isinstance(label, torch.Tensor):
        actions = [ACTIONS[idx] for idx, value in enumerate(label.tolist()) if value > 0.0]
        return ",".join(actions)
    return str(label)


def _load_vector(file_path: Path) -> tuple[torch.Tensor | None, str | None]:
    if not file_path.exists():
        return None, "missing_activation"

    tensor = load_activation(file_path).float()
    if tensor.ndim != 1:
        tensor = tensor.flatten()
    if not torch.isfinite(tensor).all():
        return None, "nonfinite_activation"
    return tensor, None


def _mean_pool_files(file_paths: list[Path]) -> tuple[torch.Tensor | None, str | None]:
    vectors = []
    for file_path in file_paths:
        vector, reason = _load_vector(file_path)
        if vector is None:
            return None, reason
        vectors.append(vector)

    if not vectors:
        return None, "missing_activation"

    first_shape = vectors[0].shape
    if any(vector.shape != first_shape for vector in vectors):
        return None, "activation_dim_mismatch"
    return torch.stack(vectors).mean(dim=0), None


def _load_pre_suffix_activation(step_folder: Path) -> tuple[torch.Tensor | None, str | None]:
    category_folder = step_folder / "prompt_suffix"
    if not category_folder.exists():
        return None, "missing_prompt_suffix"
    selected = parse_index_specification("-3:-1", discover_available_token_indices(category_folder))
    if len(selected) != 3:
        return None, "missing_prompt_suffix_tokens"
    return _mean_pool_files([category_folder / f"{token_idx}.pt" for token_idx in selected])


def _load_post_tail_activation(
    step_folder: Path,
    output_tokens: list[dict[str, Any]],
) -> tuple[torch.Tensor | None, str | None]:
    tail_indices = resolve_post_tail_indices(output_tokens)
    if len(tail_indices) != 3:
        return None, "missing_post_tail_indices"
    category_folder = step_folder / "output"
    return _mean_pool_files([category_folder / f"{token_idx}.pt" for token_idx in tail_indices])


def _load_cot_rank_activation(
    step_folder: Path,
    position: PositionSpec,
    output_tokens: list[dict[str, Any]],
) -> tuple[torch.Tensor | None, str | None]:
    if position.cot_rank is None:
        return None, "missing_cot_rank_config"
    token_idx, reason = select_cot_rank_index(output_tokens, position.cot_rank)
    if token_idx is None:
        return None, reason
    return _load_vector(step_folder / "output" / f"{token_idx}.pt")


def load_position_activation(
    trajectory_folder: Path,
    layer_idx: int,
    step_idx: int,
    position: PositionSpec,
    output_tokens: list[dict[str, Any]],
) -> tuple[torch.Tensor | None, str | None]:
    """Load one activation vector for the requested position."""
    model_folder = discover_model_folder(trajectory_folder)
    if model_folder is None:
        return None, "missing_model_folder"

    step_folder = model_folder / f"layer_{layer_idx}" / f"step_{step_idx}"
    if not step_folder.exists():
        return None, "missing_step_folder"

    if position.kind == "pre_suffix":
        return _load_pre_suffix_activation(step_folder)
    if position.kind == "post_tail":
        return _load_post_tail_activation(step_folder, output_tokens)
    if position.kind == "cot_rank":
        return _load_cot_rank_activation(step_folder, position, output_tokens)

    raise ValueError(f"Unknown position kind: {position.kind}")


def load_trajectory_records(
    activations_dir: Path,
    trajectories_dir: Path,
    max_trajectories: int | None,
) -> tuple[list[TrajectoryRecord], dict[str, int]]:
    """Load trajectory JSONs paired with activation folders."""
    skipped: Counter[str] = Counter()
    records: list[TrajectoryRecord] = []
    trajectory_folders = discover_trajectory_folders(activations_dir)
    if max_trajectories is not None:
        trajectory_folders = trajectory_folders[:max_trajectories]

    for trajectory_folder in trajectory_folders:
        trajectory_path = trajectories_dir / f"{trajectory_folder.name}.json"
        if not trajectory_path.exists():
            skipped["missing_trajectory_json"] += 1
            continue

        try:
            with trajectory_path.open() as f:
                trajectory = json.load(f)
        except json.JSONDecodeError:
            skipped["invalid_trajectory_json"] += 1
            continue

        if not isinstance(trajectory, dict) or not isinstance(trajectory.get("steps"), list):
            skipped["invalid_trajectory_schema"] += 1
            continue

        records.append(
            TrajectoryRecord(
                name=trajectory_folder.name,
                activation_folder=trajectory_folder,
                trajectory_path=trajectory_path,
                trajectory=trajectory,
            )
        )

    return records, dict(skipped)


def summarize_label_distribution(labels: torch.Tensor, label_type: LabelType) -> dict[str, int]:
    """Summarize labels in action-name order."""
    if label_type == "true_action":
        return {IDX_TO_ACTION[idx]: int((labels == idx).sum().item()) for idx in range(len(ACTIONS))}
    if label_type == "optimal_action":
        return {action: int(labels[:, idx].sum().item()) for idx, action in enumerate(ACTIONS)}
    raise ValueError(f"Unknown label_type: {label_type}")


def summarize_group_sizes(group_sizes: list[int]) -> dict[str, Any]:
    """Summarize state-average source group sizes."""
    if not group_sizes:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "distribution": {}}

    sorted_sizes = sorted(group_sizes)
    middle = len(sorted_sizes) // 2
    if len(sorted_sizes) % 2:
        median = float(sorted_sizes[middle])
    else:
        median = float(sorted_sizes[middle - 1] + sorted_sizes[middle]) / 2.0

    distribution = Counter(group_sizes)
    return {
        "min": min(group_sizes),
        "max": max(group_sizes),
        "mean": sum(group_sizes) / len(group_sizes),
        "median": median,
        "distribution": {str(size): count for size, count in sorted(distribution.items())},
    }


def _empty_action_probe_dataset(
    label_type: LabelType,
    skipped: Counter[str],
    metadata: dict[str, Any],
) -> ActionProbeDataset:
    empty_labels = torch.empty((0, len(ACTIONS)), dtype=torch.float32)
    if label_type == "true_action":
        empty_labels = torch.empty(0, dtype=torch.long)
    return ActionProbeDataset(
        activations=torch.empty((0, 0), dtype=torch.float32),
        labels=empty_labels,
        split_group_ids=[],
        skipped=dict(skipped),
        label_distribution={},
        metadata=metadata,
    )


def _tensorize_labels(labels: list[int | torch.Tensor], label_type: LabelType) -> torch.Tensor:
    if label_type == "true_action":
        return torch.tensor(labels, dtype=torch.long)
    return torch.stack([label for label in labels if isinstance(label, torch.Tensor)]).float()


def build_step_action_probe_dataset(
    records: list[TrajectoryRecord],
    label_type: LabelType,
    position: PositionSpec,
    layer_idx: int,
) -> ActionProbeDataset:
    """Build one supervised sample per eligible trajectory step."""
    activations: list[torch.Tensor] = []
    labels: list[int | torch.Tensor] = []
    split_group_ids: list[str] = []
    skipped: Counter[str] = Counter()
    raw_step_count = 0

    desc = f"dataset {label_type}/{position.name}/layer{layer_idx}"
    for record in tqdm(records, desc=desc, leave=False):
        for fallback_step_idx, step in enumerate(record.trajectory["steps"]):
            raw_step_count += 1
            label = parse_action_label(step, label_type)
            if label is None:
                skipped["invalid_label"] += 1
                continue

            output_tokens = step.get("output_tokens", [])
            if not isinstance(output_tokens, list):
                skipped["missing_output_tokens"] += 1
                continue

            step_idx = int(step.get("step_id", fallback_step_idx))
            activation, reason = load_position_activation(
                trajectory_folder=record.activation_folder,
                layer_idx=layer_idx,
                step_idx=step_idx,
                position=position,
                output_tokens=output_tokens,
            )
            if activation is None:
                skipped[reason or "missing_activation"] += 1
                continue

            activations.append(activation)
            labels.append(label)
            split_group_ids.append(record.name)

    metadata = {
        "dataset_variant": "step",
        "split_group_key": "trajectory_name",
        "raw_step_count": raw_step_count,
        "eligible_step_count": len(activations),
        "sample_count": len(activations),
        "group_size_summary": summarize_group_sizes([1 for _ in activations]),
    }

    if not activations:
        return _empty_action_probe_dataset(label_type, skipped, metadata)

    activation_tensor = torch.stack(activations).float()
    label_tensor = _tensorize_labels(labels, label_type)

    return ActionProbeDataset(
        activations=activation_tensor,
        labels=label_tensor,
        split_group_ids=split_group_ids,
        skipped=dict(skipped),
        label_distribution=summarize_label_distribution(label_tensor, label_type),
        metadata=metadata,
    )


def build_state_averaged_action_probe_dataset(
    records: list[TrajectoryRecord],
    label_type: LabelType,
    position: PositionSpec,
    layer_idx: int,
) -> ActionProbeDataset:
    """Build one supervised sample per state by averaging eligible step activations."""
    groups: dict[str, dict[str, Any]] = {}
    skipped: Counter[str] = Counter()
    raw_step_count = 0
    eligible_step_count = 0

    desc = f"dataset {label_type}/{position.name}/layer{layer_idx}/state_average"
    for record in tqdm(records, desc=desc, leave=False):
        for fallback_step_idx, step in enumerate(record.trajectory["steps"]):
            raw_step_count += 1

            label = parse_action_label(step, label_type)
            if label is None:
                skipped["invalid_label"] += 1
                continue

            state_key, state_info, state_reason = build_state_key(step)
            if state_key is None or state_info is None:
                skipped[state_reason or "invalid_state_key"] += 1
                continue

            output_tokens = step.get("output_tokens", [])
            if not isinstance(output_tokens, list):
                skipped["missing_output_tokens"] += 1
                continue

            step_idx = int(step.get("step_id", fallback_step_idx))
            activation, reason = load_position_activation(
                trajectory_folder=record.activation_folder,
                layer_idx=layer_idx,
                step_idx=step_idx,
                position=position,
                output_tokens=output_tokens,
            )
            if activation is None:
                skipped[reason or "missing_activation"] += 1
                continue

            signature = label_signature(label)
            group = groups.get(state_key)
            if group is None:
                groups[state_key] = {
                    "activations": [activation],
                    "label": label,
                    "label_signature": signature,
                    "state_info": state_info,
                    "source_trajectories": {record.name},
                    "source_steps": [(record.name, step_idx)],
                }
            else:
                if group["label_signature"] != signature:
                    existing_label = label_debug_value(group["label"], label_type)
                    new_label = label_debug_value(label, label_type)
                    raise ValueError(
                        f"Conflicting {label_type} labels for state {state_key}: {existing_label} vs {new_label}"
                    )
                group["activations"].append(activation)
                group["source_trajectories"].add(record.name)
                group["source_steps"].append((record.name, step_idx))

            eligible_step_count += 1

    group_sizes = [len(group["activations"]) for group in groups.values()]
    metadata = {
        "dataset_variant": "state_average",
        "split_group_key": "state_key",
        "state_key_schema": "(agent_position_row,agent_position_col,has_key,door_open)",
        "raw_step_count": raw_step_count,
        "eligible_step_count": eligible_step_count,
        "grouped_state_count": len(groups),
        "sample_count": len(groups),
        "group_size_summary": summarize_group_sizes(group_sizes),
        "label_conflict_count": 0,
    }

    if not groups:
        return _empty_action_probe_dataset(label_type, skipped, metadata)

    activations: list[torch.Tensor] = []
    labels: list[int | torch.Tensor] = []
    split_group_ids: list[str] = []

    for state_key in sorted(groups):
        group = groups[state_key]
        activations.append(torch.stack(group["activations"]).mean(dim=0))
        labels.append(group["label"])
        split_group_ids.append(state_key)

    activation_tensor = torch.stack(activations).float()
    label_tensor = _tensorize_labels(labels, label_type)

    return ActionProbeDataset(
        activations=activation_tensor,
        labels=label_tensor,
        split_group_ids=split_group_ids,
        skipped=dict(skipped),
        label_distribution=summarize_label_distribution(label_tensor, label_type),
        metadata=metadata,
    )


def build_action_probe_dataset(
    records: list[TrajectoryRecord],
    label_type: LabelType,
    position: PositionSpec,
    layer_idx: int,
    dataset_variant: DatasetVariant = "step",
) -> ActionProbeDataset:
    """Build the configured supervised dataset variant."""
    if dataset_variant == "step":
        return build_step_action_probe_dataset(records, label_type, position, layer_idx)
    if dataset_variant == "state_average":
        return build_state_averaged_action_probe_dataset(records, label_type, position, layer_idx)
    raise ValueError(f"Unknown dataset_variant: {dataset_variant}")


def split_trajectory_names(
    trajectory_names: list[str],
    eval_split: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Split by trajectory name so steps from one trajectory cannot cross splits."""
    if not 0.0 <= eval_split < 1.0:
        raise ValueError("eval_split must be in [0, 1)")

    unique_names = sorted(set(trajectory_names))
    if len(unique_names) <= 1 or eval_split == 0.0:
        return set(unique_names), set()

    rng = random.Random(seed)
    rng.shuffle(unique_names)
    n_eval = max(1, int(round(len(unique_names) * eval_split)))
    n_eval = min(n_eval, len(unique_names) - 1)

    eval_names = set(unique_names[:n_eval])
    train_names = set(unique_names[n_eval:])
    return train_names, eval_names


def create_action_probe_model(
    model_type: ModelType,
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
) -> nn.Module:
    """Create a 4-way action probe model."""
    if model_type == "linear":
        return nn.Linear(input_dim, len(ACTIONS))
    if model_type == "mlp":
        return MLPProbe(input_dim=input_dim, num_classes=len(ACTIONS), hidden_dims=hidden_dims, dropout=dropout)
    raise ValueError(f"Unknown model_type: {model_type}")


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x_raw, batch_y_raw in dataloader:
        batch_x = batch_x_raw.to(device)
        batch_y = batch_y_raw.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches else 0.0


@torch.no_grad()
def _collect_logits(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    labels_list = []
    for batch_x, batch_y in dataloader:
        logits_list.append(model(batch_x.to(device)).cpu())
        labels_list.append(batch_y.cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def evaluate_true_action_probe(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Compute multiclass true-action metrics."""
    logits, labels = _collect_logits(model, dataloader, device)
    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()

    per_action: dict[str, dict[str, float | int]] = {}
    recalls = []
    for idx, action in enumerate(ACTIONS):
        pred_mask = preds == idx
        label_mask = labels == idx
        tp = int((pred_mask & label_mask).sum().item())
        fp = int((pred_mask & ~label_mask).sum().item())
        fn = int((~pred_mask & label_mask).sum().item())
        support = int(label_mask.sum().item())
        predicted = int(pred_mask.sum().item())

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall)

        per_action[action] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0
    return {
        "top1_accuracy": accuracy,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_action": per_action,
        "n": int(labels.numel()),
    }


def evaluate_optimal_action_probe(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute top-1-in-set and per-action multi-label metrics."""
    logits, labels = _collect_logits(model, dataloader, device)
    top1_preds = logits.argmax(dim=-1)
    row_indices = torch.arange(labels.shape[0])
    top1_in_set = labels[row_indices, top1_preds] > 0.0
    top1_accuracy = top1_in_set.float().mean().item()

    multi_preds = torch.sigmoid(logits) >= threshold
    label_bools = labels > 0.0

    per_action: dict[str, dict[str, float | int]] = {}
    for idx, action in enumerate(ACTIONS):
        pred_mask = multi_preds[:, idx]
        label_mask = label_bools[:, idx]
        tp = int((pred_mask & label_mask).sum().item())
        fp = int((pred_mask & ~label_mask).sum().item())
        fn = int((~pred_mask & label_mask).sum().item())
        support = int(label_mask.sum().item())
        predicted = int(pred_mask.sum().item())

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_action[action] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "top1_in_optimal_set_accuracy": top1_accuracy,
        "top1_accuracy": top1_accuracy,
        "multilabel_threshold": threshold,
        "per_action": per_action,
        "n": int(labels.shape[0]),
    }


def _compute_safe_normalization(train_activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, std = compute_normalization_params(train_activations)
    mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
    std = torch.where(torch.isfinite(std) & (std > 1e-8), std, torch.ones_like(std))
    return mean, std


def train_action_probe(
    dataset: ActionProbeDataset,
    label_type: LabelType,
    model_type: ModelType,
    hidden_dims: list[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_split: float,
    seed: int,
    device: torch.device,
    verbose: bool,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Train one probe and return the model, scaler, metrics, and split metadata."""
    set_seed(seed)
    train_group_ids, eval_group_ids = split_trajectory_names(dataset.split_group_ids, eval_split, seed)
    train_indices = [idx for idx, group_id in enumerate(dataset.split_group_ids) if group_id in train_group_ids]
    eval_indices = [idx for idx, group_id in enumerate(dataset.split_group_ids) if group_id in eval_group_ids]

    if not train_indices:
        raise ValueError("No training samples after grouped split")

    eval_is_train = False
    if not eval_indices:
        eval_indices = train_indices
        eval_is_train = True

    train_x_raw = dataset.activations[train_indices]
    eval_x_raw = dataset.activations[eval_indices]
    train_y = dataset.labels[train_indices]
    eval_y = dataset.labels[eval_indices]

    scaler_mean, scaler_std = _compute_safe_normalization(train_x_raw)
    train_x = normalize_activations(train_x_raw, scaler_mean, scaler_std)
    eval_x = normalize_activations(eval_x_raw, scaler_mean, scaler_std)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=batch_size, shuffle=False)

    model = create_action_probe_model(
        model_type=model_type,
        input_dim=dataset.activations.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    if label_type == "true_action":
        criterion = nn.CrossEntropyLoss()
        primary_metric = "balanced_accuracy"
        evaluate = evaluate_true_action_probe
    elif label_type == "optimal_action":
        criterion = nn.BCEWithLogitsLoss()
        primary_metric = "top1_in_optimal_set_accuracy"
        evaluate = evaluate_optimal_action_probe
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, eval_loader, device)
        score = float(metrics[primary_metric])
        history.append({"epoch": epoch, "train_loss": train_loss, primary_metric: score})

        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if verbose and (epoch in (1, epochs) or epoch % 10 == 0):
            print(f"    epoch {epoch:3d}/{epochs}: loss={train_loss:.4f} {primary_metric}={score:.4f}")

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    final_metrics = evaluate(model, eval_loader, device)
    split_group_key = dataset.metadata.get("split_group_key", "trajectory_name")
    split = {
        "split_group_key": split_group_key,
        "train_group_count": len(train_group_ids),
        "eval_group_count": len(eval_group_ids),
        "train_trajectory_count": len(train_group_ids) if split_group_key == "trajectory_name" else None,
        "eval_trajectory_count": len(eval_group_ids) if split_group_key == "trajectory_name" else None,
        "train_state_count": len(train_group_ids) if split_group_key == "state_key" else None,
        "eval_state_count": len(eval_group_ids) if split_group_key == "state_key" else None,
        "train_sample_count": len(train_indices),
        "eval_sample_count": len(eval_indices),
        "eval_is_train": eval_is_train,
        "eval_split": eval_split,
        "seed": seed,
        "primary_metric": primary_metric,
        "best_primary_metric": best_score,
        "history": history,
    }
    return model, scaler_mean, scaler_std, final_metrics, split


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n")


def save_probe_checkpoint(
    path: Path,
    model: nn.Module,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    config: dict[str, Any],
    metrics: dict[str, Any],
    split: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_type": config["model_type"],
        "dataset_variant": config["dataset_variant"],
        "input_dim": config["input_dim"],
        "num_classes": len(ACTIONS),
        "hidden_dims": config["hidden_dims"],
        "dropout": config["dropout"],
        "label_type": config["label_type"],
        "position": config["position"],
        "layer": config["layer"],
        "action_order": ACTIONS,
        "action_to_idx": ACTION_TO_IDX,
        "scaler_mean": scaler_mean.detach().cpu(),
        "scaler_std": scaler_std.detach().cpu(),
        "normalization": {"enabled": True, "source": "train_split"},
        "config": config,
        "dataset_metadata": dataset_metadata,
        "metrics": metrics,
        "split": split,
    }
    torch.save(checkpoint, path)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_probe_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run the configured probe sweep and return the manifest."""
    activations_dir = Path(args.activations_dir)
    trajectories_dir = Path(args.trajectories_dir)
    dataset_variant = args.dataset_variant
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (DEFAULT_STATE_AVG_OUTPUT_DIR if dataset_variant == "state_average" else DEFAULT_OUTPUT_DIR)
    )

    layers = parse_int_selection(args.layers)
    cot_ranks = parse_int_selection(args.cot_ranks)
    label_types = parse_csv_choices(args.label_types, {"true_action", "optimal_action"}, "label_types")
    model_types = parse_csv_choices(args.model_types, {"linear", "mlp"}, "model_types")
    hidden_dims = parse_int_selection(args.hidden_dims)
    positions = build_position_specs(cot_ranks, args.include_pre_suffix, args.include_post_tail)

    if not layers:
        raise ValueError("At least one layer must be selected")
    if not positions:
        raise ValueError("At least one position must be selected")

    prepare_output_dir(output_dir, args.overwrite)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    records, skipped_trajectories = load_trajectory_records(activations_dir, trajectories_dir, args.max_trajectories)
    if not records:
        raise ValueError("No trajectory records with activation folders were found")
    print(f"Loaded {len(records)} trajectory records")

    aggregate_results: list[dict[str, Any]] = []
    probe_configs: list[dict[str, Any]] = []

    total_datasets = len(label_types) * len(positions) * len(layers)
    dataset_idx = 0
    for label_type_raw in label_types:
        label_type = label_type_raw  # type: ignore[assignment]
        for position in positions:
            for layer_idx in layers:
                dataset_idx += 1
                print(f"\n[{dataset_idx}/{total_datasets}] Building {label_type}/{position.name}/layer{layer_idx}")
                dataset = build_action_probe_dataset(records, label_type, position, layer_idx, dataset_variant)
                sample_count = int(dataset.activations.shape[0])
                if sample_count == 0:
                    raise RuntimeError(
                        f"No samples for {label_type}/{position.name}/layer{layer_idx}. "
                        f"Skipped counts: {dataset.skipped}"
                    )

                print(
                    f"  samples={sample_count} dim={dataset.activations.shape[1]} "
                    f"labels={dataset.label_distribution} skipped={dataset.skipped} "
                    f"split_group_key={dataset.metadata.get('split_group_key')}"
                )

                for model_type_raw in model_types:
                    model_type = model_type_raw  # type: ignore[assignment]
                    print(f"  Training {model_type}")
                    model, scaler_mean, scaler_std, metrics, split = train_action_probe(
                        dataset=dataset,
                        label_type=label_type,
                        model_type=model_type,
                        hidden_dims=hidden_dims,
                        dropout=args.dropout,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                        weight_decay=args.weight_decay,
                        eval_split=args.eval_split,
                        seed=args.seed,
                        device=device,
                        verbose=not args.quiet,
                    )

                    stem = f"{label_type}_{position.name}_layer{layer_idx}_{model_type}"
                    checkpoint_path = output_dir / f"probe_{stem}.pt"
                    metrics_path = output_dir / f"metrics_{stem}.json"
                    config = {
                        "dataset_variant": dataset_variant,
                        "label_type": label_type,
                        "position": position.name,
                        "position_kind": position.kind,
                        "cot_rank": position.cot_rank,
                        "layer": layer_idx,
                        "model_type": model_type,
                        "input_dim": int(dataset.activations.shape[1]),
                        "hidden_dims": hidden_dims,
                        "dropout": args.dropout,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "split_group_key": dataset.metadata.get("split_group_key"),
                    }
                    metrics_record = {
                        "config": config,
                        "metrics": metrics,
                        "split": split,
                        "sample_count": sample_count,
                        "skipped": dataset.skipped,
                        "label_distribution": dataset.label_distribution,
                        "dataset_metadata": dataset.metadata,
                        "action_mapping": ACTION_TO_IDX,
                    }

                    save_probe_checkpoint(
                        checkpoint_path,
                        model=model,
                        scaler_mean=scaler_mean,
                        scaler_std=scaler_std,
                        config=config,
                        metrics=metrics,
                        split=split,
                        dataset_metadata=dataset.metadata,
                    )
                    write_json(metrics_path, metrics_record)

                    primary_metric = split["primary_metric"]
                    aggregate_entry = {
                        "label_type": label_type,
                        "dataset_variant": dataset_variant,
                        "position": position.name,
                        "position_kind": position.kind,
                        "cot_rank": position.cot_rank,
                        "layer": layer_idx,
                        "model_type": model_type,
                        "sample_count": sample_count,
                        "raw_step_count": dataset.metadata.get("raw_step_count"),
                        "eligible_step_count": dataset.metadata.get("eligible_step_count"),
                        "grouped_state_count": dataset.metadata.get("grouped_state_count"),
                        "split_group_key": dataset.metadata.get("split_group_key"),
                        "metric_name": primary_metric,
                        "metric_value": metrics[primary_metric],
                        "checkpoint": checkpoint_path.name,
                        "metrics_file": metrics_path.name,
                    }
                    aggregate_results.append(aggregate_entry)
                    probe_configs.append(
                        {
                            **config,
                            "checkpoint": checkpoint_path.name,
                            "metrics_file": metrics_path.name,
                            "sample_count": sample_count,
                            "skipped": dataset.skipped,
                            "label_distribution": dataset.label_distribution,
                            "dataset_metadata": dataset.metadata,
                        }
                    )
                    print(f"    saved {checkpoint_path.name} ({primary_metric}={metrics[primary_metric]:.4f})")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "activations_dir": activations_dir,
        "trajectories_dir": trajectories_dir,
        "output_dir": output_dir,
        "dataset_variant": dataset_variant,
        "state_key_schema": "(agent_position_row,agent_position_col,has_key,door_open)"
        if dataset_variant == "state_average"
        else None,
        "trajectory_count": len(records),
        "skipped_trajectories": skipped_trajectories,
        "layers": layers,
        "positions": [asdict(position) for position in positions],
        "label_types": label_types,
        "model_types": model_types,
        "expected_probe_count": len(layers) * len(positions) * len(label_types) * len(model_types),
        "completed_probe_count": len(aggregate_results),
        "action_order": ACTIONS,
        "action_mapping": ACTION_TO_IDX,
        "normalization": "train-set z-score saved in each checkpoint",
        "probe_configs": probe_configs,
        "aggregate_results": aggregate_results,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"\nDone. Wrote {len(aggregate_results)} probes and manifest to {output_dir / 'run_manifest.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--trajectories-dir", type=Path, default=DEFAULT_TRAJECTORIES_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-variant", choices=["step", "state_average"], default="step")
    parser.add_argument("--layers", default="7,15,23")
    parser.add_argument("--cot-ranks", default="0:29")
    parser.add_argument("--include-pre-suffix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-post-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-types", default="true_action,optimal_action")
    parser.add_argument("--model-types", default="linear,mlp")
    parser.add_argument("--hidden-dims", default="512,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_probe_sweep(parse_args())
