from pathlib import Path

import pytest
import torch
from run_seed12_action_probes import ACTIONS
from run_size15_pre_reasoning_action_probe_steering_hf import (
    build_arg_parser as build_hf_arg_parser,
)
from run_size15_pre_reasoning_action_probe_steering_hf import (
    normalize_hf_args,
)
from run_size15_pre_reasoning_action_probe_steering_mlx import (
    Size15SampleRecord,
    analysis_channel_prefill_token_ids,
    build_arg_parser,
    build_paired_rows,
    classify_generated_action_text,
    margin_unit_delta,
    normalize_args,
    paired_stats_by_layer_alpha,
    pairwise_margin_shift,
    pairwise_raw_direction_from_weights,
    parse_generated_action,
    reconstruct_prompt_row,
    resolve_steering_suffix_positions,
    select_test_nonoptimal_targets,
    summarize_steering_rows,
)


def _tok(token_id: int) -> dict:
    return {"token_id": token_id, "token": str(token_id)}


def _sample(name: str, state_key: str, action: str, grid_cells: tuple[tuple[str, ...], ...]) -> Size15SampleRecord:
    step = {
        "agent_action": action,
        "grid_state": [" ".join(row) for row in grid_cells],
        "grid_state_tokens": [_tok(3), _tok(4)],
        "prompt_suffix_tokens": [_tok(5), _tok(6), _tok(7)],
        "output_tokens": [
            {"token": "<|channel|>", "token_id": 8},
            {"token": "analysis", "token_id": 9},
            {"token": "<|message|>", "token_id": 10},
        ],
        "step_id": 0,
        "prompt": {
            "prompt_prefix_tokens": [_tok(1), _tok(2)],
            "prompt_suffix_tokens": [_tok(5), _tok(6), _tok(7)],
        },
    }
    return Size15SampleRecord(
        name=name,
        trajectory_path=Path(f"{name}.json"),
        activation_folder=Path(name),
        step=step,
        state=(1, 1, False, False),
        state_key=state_key,
        state_index=0,
        config_type="test",
        action=action,
        action_idx=ACTIONS.index(action),
        astar_actions=(),
        astar_indices=(),
        is_optimal_action=False,
        sample_index=0,
        grid_cells=grid_cells,
    )


def test_select_test_nonoptimal_targets_only_heldout_wrong_samples() -> None:
    grid = (
        ("#", "#", "#", "#", "#"),
        ("#", "A", "_", "G", "#"),
        ("#", "#", "#", "#", "#"),
    )
    test_wrong = _sample("test_wrong", "state_test_wrong", "UP", grid)
    test_correct = _sample("test_correct", "state_test_correct", "RIGHT", grid)
    train_wrong = _sample("train_wrong", "state_train_wrong", "UP", grid)
    split = {
        "state_test_wrong": "test",
        "state_test_correct": "test",
        "state_train_wrong": "train",
    }

    targets = select_test_nonoptimal_targets([train_wrong, test_correct, test_wrong], split)

    assert [target.sample.name for target in targets] == ["test_wrong"]
    assert targets[0].original_action == "UP"
    assert targets[0].optimal_actions == ("RIGHT",)


def test_reconstruct_prompt_row_preserves_prefix_grid_suffix_order() -> None:
    sample = _sample(
        "prompt",
        "state",
        "UP",
        (
            ("#", "#", "#"),
            ("#", "A", "G"),
            ("#", "#", "#"),
        ),
    )

    prompt_row = reconstruct_prompt_row(sample)
    rel_indices, abs_positions = resolve_steering_suffix_positions(prompt_row, "-3:-1")

    assert prompt_row.full_token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert prompt_row.prompt_len == 7
    assert prompt_row.n_suffix == 3
    assert rel_indices == [0, 1, 2]
    assert abs_positions == [4, 5, 6]


def test_reconstruct_prompt_row_can_prefill_analysis_channel_without_moving_patch_positions() -> None:
    sample = _sample(
        "prompt",
        "state",
        "UP",
        (
            ("#", "#", "#"),
            ("#", "A", "G"),
            ("#", "#", "#"),
        ),
    )

    prompt_row = reconstruct_prompt_row(sample, prefill_analysis_channel=True)
    _, abs_positions = resolve_steering_suffix_positions(prompt_row, "-3:-1")

    assert analysis_channel_prefill_token_ids(sample) == [8, 9, 10]
    assert prompt_row.full_token_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert prompt_row.prompt_len == 7
    assert abs_positions == [4, 5, 6]


def test_pairwise_direction_handles_single_and_tied_optimal_sets() -> None:
    weights = torch.tensor(
        [
            [1.0, 2.0],  # LEFT
            [3.0, 4.0],  # RIGHT
            [5.0, 6.0],  # UP
            [7.0, 8.0],  # DOWN
        ]
    )
    scaler_std = torch.tensor([2.0, 4.0])

    w_margin, raw = pairwise_raw_direction_from_weights(weights, scaler_std, ("UP",), "LEFT")
    tied_margin, tied_raw = pairwise_raw_direction_from_weights(weights, scaler_std, ("RIGHT", "UP"), "LEFT")

    assert torch.allclose(w_margin, torch.tensor([4.0, 4.0]))
    assert torch.allclose(raw, torch.tensor([2.0, 1.0]))
    assert torch.allclose(tied_margin, torch.tensor([3.0, 3.0]))
    assert torch.allclose(tied_raw, torch.tensor([1.5, 0.75]))


def test_margin_unit_scaling_changes_pairwise_margin_by_alpha() -> None:
    weights = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    scaler_std = torch.tensor([2.0, 4.0])
    w_margin, raw = pairwise_raw_direction_from_weights(weights, scaler_std, ("UP",), "LEFT")
    unit_delta = margin_unit_delta(raw)

    assert pairwise_margin_shift(w_margin, scaler_std, unit_delta) == pytest.approx(1.0)
    assert pairwise_margin_shift(w_margin, scaler_std, 2.0 * unit_delta) == pytest.approx(2.0)


def test_parse_generated_action_json_and_tail_fallback() -> None:
    assert parse_generated_action('{"action": "DOWN"}') == "DOWN"
    assert parse_generated_action("analysis text ... final answer is LEFT") == "LEFT"
    assert parse_generated_action("no action present") is None


def test_clean_action_classification_requires_natural_json_action() -> None:
    assert classify_generated_action_text('{"action": "DOWN"}', tail_action_fallback=False) == (
        "DOWN",
        None,
        "json_action",
    )
    assert classify_generated_action_text("analysis text ... final answer is LEFT", tail_action_fallback=True) == (
        "LEFT",
        "LEFT",
        "tail_fallback",
    )
    assert classify_generated_action_text("analysis text ... final answer is LEFT", tail_action_fallback=False) == (
        None,
        None,
        "parse_fail",
    )


def test_clean_autonomous_cli_disables_fallbacks_and_uses_large_token_budget() -> None:
    args = normalize_args(build_arg_parser().parse_args(["--clean-autonomous"]))

    assert args.max_new_tokens == 8192
    assert args.finalization_fallback is False
    assert args.tail_action_fallback is False
    assert args.generation_protocol == "clean_autonomous"


def test_hf_cli_is_clean_autonomous_for_runpod() -> None:
    args = normalize_hf_args(build_hf_arg_parser().parse_args(["--max-samples", "3"]))

    assert args.clean_autonomous is True
    assert args.finalization_fallback is False
    assert args.tail_action_fallback is False
    assert args.generation_protocol == "clean_autonomous_hf"
    assert args.max_samples == 3


def test_paired_rows_match_same_sample_layer_seed_alpha_zero() -> None:
    rows = [
        {
            "trajectory_name": "a",
            "sample_index": 0,
            "state_key": "s",
            "layer": 7,
            "alpha": 0.0,
            "seed_index": 0,
            "generation_seed": 12,
            "original_action": "UP",
            "optimal_actions": ["RIGHT"],
            "generated_action": "UP",
            "parse_status": "ok",
            "generated_action_is_optimal": False,
        },
        {
            "trajectory_name": "a",
            "sample_index": 0,
            "state_key": "s",
            "layer": 7,
            "alpha": 1.0,
            "seed_index": 0,
            "generation_seed": 12,
            "original_action": "UP",
            "optimal_actions": ["RIGHT"],
            "generated_action": "RIGHT",
            "parse_status": "ok",
            "generated_action_is_optimal": True,
        },
    ]

    paired = build_paired_rows(rows)

    assert len(paired) == 1
    assert paired[0]["baseline_generated_action"] == "UP"
    assert paired[0]["steered_generated_action"] == "RIGHT"
    assert paired[0]["delta_optimal"] == 1


def test_summary_preserves_separate_layers() -> None:
    rows = [
        {
            "layer": 7,
            "alpha": 0.0,
            "parse_status": "ok",
            "generated_action": "RIGHT",
            "generated_action_is_optimal": True,
            "action_changed_from_original": True,
        },
        {
            "layer": 15,
            "alpha": 0.0,
            "parse_status": "parse_fail",
            "generated_action": None,
            "generated_action_is_optimal": False,
            "action_changed_from_original": True,
        },
    ]

    summaries = summarize_steering_rows(rows)

    assert [row["layer"] for row in summaries] == [7, 15]
    assert summaries[0]["optimal_rate"] == 1.0
    assert summaries[1]["parse_fail_rate"] == 1.0


def test_paired_stats_delta_equals_mean_paired_outcomes() -> None:
    paired_rows = [
        {
            "layer": 7,
            "alpha": 1.0,
            "baseline_generated_action_is_optimal": False,
            "steered_generated_action_is_optimal": True,
            "delta_optimal": 1,
        },
        {
            "layer": 7,
            "alpha": 1.0,
            "baseline_generated_action_is_optimal": True,
            "steered_generated_action_is_optimal": False,
            "delta_optimal": -1,
        },
        {
            "layer": 7,
            "alpha": 1.0,
            "baseline_generated_action_is_optimal": False,
            "steered_generated_action_is_optimal": False,
            "delta_optimal": 0,
        },
        {
            "layer": 15,
            "alpha": 1.0,
            "baseline_generated_action_is_optimal": False,
            "steered_generated_action_is_optimal": True,
            "delta_optimal": 1,
        },
    ]

    stats = paired_stats_by_layer_alpha(paired_rows)

    layer7 = next(row for row in stats if row["layer"] == 7)
    layer15 = next(row for row in stats if row["layer"] == 15)
    assert layer7["delta_optimal_rate"] == pytest.approx(0.0)
    assert layer7["mcnemar_gain_baseline_wrong_to_steered_optimal_n"] == 1
    assert layer7["mcnemar_drop_baseline_optimal_to_steered_wrong_n"] == 1
    assert layer15["delta_optimal_rate"] == pytest.approx(1.0)
