"""Tests for the Seed12 action probe sweep helpers."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_seed12_action_probes import (  # noqa: E402
    ACTION_TO_IDX,
    PositionSpec,
    TrajectoryRecord,
    build_action_probe_dataset,
    build_state_key,
    format_state_key,
    load_position_activation,
    parse_agent_position_from_grid_state,
    parse_optimal_action_label,
    parse_true_action_label,
    resolve_cot_checkpoint_indices,
    resolve_post_tail_indices,
    select_cot_rank_index,
    split_trajectory_names,
)


def _output_tokens(n_tokens: int, analysis_end: int) -> list[dict]:
    tokens = []
    for idx in range(n_tokens):
        groups = ["output"]
        groups.append("analysis" if idx < analysis_end else "final")
        tokens.append({"id": idx, "token": f"tok_{idx}", "token_id": idx, "token_groups": groups})
    return tokens


def _save_vector(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(values, dtype=torch.float32), path)


def test_cot_rank_selection_uses_every_tenth_analysis_token():
    output_tokens = _output_tokens(n_tokens=40, analysis_end=35)

    assert resolve_cot_checkpoint_indices(output_tokens) == [0, 10, 20, 30]
    assert select_cot_rank_index(output_tokens, 0) == (0, None)
    assert select_cot_rank_index(output_tokens, 3) == (30, None)
    assert select_cot_rank_index(output_tokens, 4) == (None, "missing_cot_rank")


def test_cot_rank_selection_excludes_post_tail_indices_without_shifting_ranks():
    output_tokens = _output_tokens(n_tokens=24, analysis_end=24)

    assert resolve_post_tail_indices(output_tokens) == [8, 9, 10]
    assert resolve_cot_checkpoint_indices(output_tokens) == [0, 10, 20]
    assert select_cot_rank_index(output_tokens, 1) == (None, "cot_rank_overlaps_post_tail")
    assert select_cot_rank_index(output_tokens, 2) == (20, None)


def test_load_position_activation_mean_pools_pre_suffix_and_post_tail(tmp_path):
    trajectory_folder = tmp_path / "traj_0"
    step_folder = trajectory_folder / "org__model" / "layer_7" / "step_0"

    _save_vector(step_folder / "prompt_suffix" / "16.pt", [1.0, 3.0])
    _save_vector(step_folder / "prompt_suffix" / "17.pt", [3.0, 5.0])
    _save_vector(step_folder / "prompt_suffix" / "18.pt", [5.0, 7.0])

    output_tokens = _output_tokens(n_tokens=30, analysis_end=30)
    _save_vector(step_folder / "output" / "14.pt", [2.0, 4.0])
    _save_vector(step_folder / "output" / "15.pt", [4.0, 6.0])
    _save_vector(step_folder / "output" / "16.pt", [6.0, 8.0])

    pre_activation, pre_reason = load_position_activation(
        trajectory_folder,
        layer_idx=7,
        step_idx=0,
        position=PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix"),
        output_tokens=output_tokens,
    )
    post_activation, post_reason = load_position_activation(
        trajectory_folder,
        layer_idx=7,
        step_idx=0,
        position=PositionSpec(name="post_reasoning_tail", kind="post_tail"),
        output_tokens=output_tokens,
    )

    assert pre_reason is None
    assert post_reason is None
    assert torch.allclose(pre_activation, torch.tensor([3.0, 5.0]))
    assert torch.allclose(post_activation, torch.tensor([4.0, 6.0]))


def test_action_label_parsing_true_and_optimal_actions():
    assert parse_true_action_label({"agent_action": "LEFT"}) == ACTION_TO_IDX["LEFT"]
    assert parse_true_action_label({"agent_action": "N/A"}) is None
    assert parse_true_action_label({"agent_action": None}) is None

    optimal = parse_optimal_action_label({"astar_actions": ["RIGHT", "UP"]})
    assert torch.equal(optimal, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert parse_optimal_action_label({"astar_actions": ["WAIT"]}) is None
    assert parse_optimal_action_label({"astar_actions": None}) is None


def test_split_trajectory_names_keeps_groups_disjoint_and_deterministic():
    names = ["traj_a", "traj_a", "traj_b", "traj_c", "traj_d", "traj_d"]

    train_names_1, eval_names_1 = split_trajectory_names(names, eval_split=0.25, seed=7)
    train_names_2, eval_names_2 = split_trajectory_names(names, eval_split=0.25, seed=7)

    assert (train_names_1, eval_names_1) == (train_names_2, eval_names_2)
    assert train_names_1.isdisjoint(eval_names_1)
    assert train_names_1 | eval_names_1 == {"traj_a", "traj_b", "traj_c", "traj_d"}
    assert len(eval_names_1) == 1


def _grid_state(row: int, col: int) -> list[str]:
    rows = ["  0 1 2 "]
    for row_idx in range(3):
        cells = ["_", "_", "_"]
        if row_idx == row:
            cells[col] = "A"
        rows.append(f"{row_idx} " + " ".join(cells))
    return rows


def _state_step(
    step_id: int,
    row: int,
    col: int,
    action: str,
    optimal_actions: list[str] | None = None,
    carrying_key: bool = False,
    door_open: bool = False,
) -> dict:
    return {
        "step_id": step_id,
        "grid_state": _grid_state(row, col),
        "carrying_key": carrying_key,
        "door_open": door_open,
        "agent_action": action,
        "astar_actions": optimal_actions or [action],
        "output_tokens": _output_tokens(n_tokens=30, analysis_end=30),
    }


def _state_record(tmp_path: Path, name: str, steps: list[dict], vectors: list[list[float]]) -> TrajectoryRecord:
    activation_folder = tmp_path / name
    for step, vector in zip(steps, vectors, strict=True):
        step_folder = activation_folder / "org__model" / "layer_7" / f"step_{step['step_id']}"
        for token_idx in (16, 17, 18):
            _save_vector(step_folder / "prompt_suffix" / f"{token_idx}.pt", vector)
    return TrajectoryRecord(
        name=name,
        activation_folder=activation_folder,
        trajectory_path=tmp_path / f"{name}.json",
        trajectory={"steps": steps},
    )


def test_parse_agent_position_and_state_key_from_grid_state():
    step = _state_step(0, row=1, col=2, action="LEFT", carrying_key=True, door_open=False)

    assert parse_agent_position_from_grid_state(step["grid_state"]) == (1, 2)
    state_key, state_info, reason = build_state_key(step)

    assert reason is None
    assert state_key == format_state_key((1, 2), has_key=True, door_open=False)
    assert state_info == {"agent_position": (1, 2), "has_key": True, "door_open": False}


def test_state_average_dataset_averages_activations_by_state(tmp_path):
    steps_a = [_state_step(0, row=1, col=1, action="LEFT")]
    steps_b = [_state_step(0, row=1, col=1, action="LEFT")]
    records = [
        _state_record(tmp_path, "traj_a", steps_a, [[1.0, 3.0]]),
        _state_record(tmp_path, "traj_b", steps_b, [[5.0, 7.0]]),
    ]

    dataset = build_action_probe_dataset(
        records,
        label_type="true_action",
        position=PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix"),
        layer_idx=7,
        dataset_variant="state_average",
    )

    assert dataset.activations.shape == (1, 2)
    assert torch.allclose(dataset.activations[0], torch.tensor([3.0, 5.0]))
    assert dataset.labels.tolist() == [ACTION_TO_IDX["LEFT"]]
    assert dataset.split_group_ids == [format_state_key((1, 1), has_key=False, door_open=False)]
    assert dataset.metadata["raw_step_count"] == 2
    assert dataset.metadata["eligible_step_count"] == 2
    assert dataset.metadata["grouped_state_count"] == 1
    assert dataset.metadata["group_size_summary"]["distribution"] == {"2": 1}


def test_state_average_dataset_preserves_multihot_optimal_label(tmp_path):
    step = _state_step(0, row=2, col=0, action="RIGHT", optimal_actions=["RIGHT", "DOWN"], door_open=True)
    records = [_state_record(tmp_path, "traj_a", [step], [[2.0, 4.0]])]

    dataset = build_action_probe_dataset(
        records,
        label_type="optimal_action",
        position=PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix"),
        layer_idx=7,
        dataset_variant="state_average",
    )

    assert dataset.activations.shape == (1, 2)
    assert torch.equal(dataset.labels[0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert dataset.label_distribution == {"LEFT": 0, "RIGHT": 1, "UP": 0, "DOWN": 1}


def test_state_average_dataset_fails_on_conflicting_labels(tmp_path):
    records = [
        _state_record(tmp_path, "traj_a", [_state_step(0, row=0, col=0, action="LEFT")], [[1.0, 1.0]]),
        _state_record(tmp_path, "traj_b", [_state_step(0, row=0, col=0, action="RIGHT")], [[2.0, 2.0]]),
    ]

    try:
        build_action_probe_dataset(
            records,
            label_type="true_action",
            position=PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix"),
            layer_idx=7,
            dataset_variant="state_average",
        )
    except ValueError as exc:
        assert "Conflicting true_action labels" in str(exc)
    else:
        raise AssertionError("Expected conflicting state labels to raise ValueError")


def test_state_average_split_groups_are_state_keys(tmp_path):
    steps = [
        _state_step(0, row=0, col=0, action="UP"),
        _state_step(1, row=0, col=1, action="DOWN"),
    ]
    records = [_state_record(tmp_path, "traj_a", steps, [[1.0, 1.0], [2.0, 2.0]])]

    dataset = build_action_probe_dataset(
        records,
        label_type="true_action",
        position=PositionSpec(name="pre_reasoning_suffix", kind="pre_suffix"),
        layer_idx=7,
        dataset_variant="state_average",
    )
    train_keys, eval_keys = split_trajectory_names(dataset.split_group_ids, eval_split=0.5, seed=3)

    assert dataset.metadata["split_group_key"] == "state_key"
    assert set(dataset.split_group_ids) == train_keys | eval_keys
    assert train_keys.isdisjoint(eval_keys)
