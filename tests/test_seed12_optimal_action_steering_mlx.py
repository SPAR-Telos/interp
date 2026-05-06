"""Tests for Seed12 optimal-action steering helpers."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_seed12_optimal_action_steering_mlx import (  # noqa: E402
    ACTION_TO_IDX,
    aggregate_results,
    build_action_contrast_direction,
    build_action_vs_source_direction,
    build_centroid_action_contrast_direction,
    build_centroid_action_vs_source_direction,
    build_direction_control_advantage_rows,
    build_direction_control_overall_advantage_rows,
    build_direction_control_overall_rows,
    build_direction_control_rows,
    build_noop_agreement_rows,
    build_paired_effect_rows,
    build_raw_action_contrast_direction,
    build_raw_action_vs_source_direction,
    build_step_prompt_row,
    build_target_direction,
    compute_action_mean_diffs,
    compute_action_positive_means,
    extract_activation_space_directions,
    extract_activation_space_weights,
    resolve_last_suffix_positions,
    steering_sample_seed,
)


def _token(token_id: int, token: str | None = None) -> dict:
    return {"token_id": token_id, "token": token if token is not None else str(token_id)}


def test_linear_checkpoint_direction_extraction_divides_by_scaler_std():
    checkpoint = {
        "model_type": "linear",
        "model_state_dict": {
            "weight": torch.tensor(
                [
                    [2.0, 4.0],
                    [0.0, 3.0],
                    [-2.0, 0.0],
                    [1.0, 1.0],
                ]
            )
        },
        "scaler_std": torch.tensor([2.0, 4.0]),
    }

    directions = extract_activation_space_directions(checkpoint)

    expected_left = torch.tensor([1.0, 1.0])
    expected_left = expected_left / expected_left.norm()
    expected_right = torch.tensor([0.0, 0.75])
    expected_right = expected_right / expected_right.norm()

    assert torch.allclose(directions[ACTION_TO_IDX["LEFT"]], expected_left)
    assert torch.allclose(directions[ACTION_TO_IDX["RIGHT"]], expected_right)
    assert torch.allclose(directions.norm(dim=1), torch.ones(4))


def test_linear_checkpoint_weight_extraction_preserves_raw_activation_space_weights():
    checkpoint = {
        "model_type": "linear",
        "model_state_dict": {
            "weight": torch.tensor(
                [
                    [2.0, 4.0],
                    [0.0, 3.0],
                    [-2.0, 0.0],
                    [1.0, 1.0],
                ]
            )
        },
        "scaler_std": torch.tensor([2.0, 4.0]),
    }

    activation_weights = extract_activation_space_weights(checkpoint)

    assert torch.allclose(activation_weights[ACTION_TO_IDX["LEFT"]], torch.tensor([1.0, 1.0]))
    assert torch.allclose(activation_weights[ACTION_TO_IDX["RIGHT"]], torch.tensor([0.0, 0.75]))


def test_tied_optimal_actions_average_directions_and_class_gaps_in_action_order():
    directions_by_action = torch.zeros(4, 2)
    directions_by_action[ACTION_TO_IDX["LEFT"]] = torch.tensor([1.0, 0.0])
    directions_by_action[ACTION_TO_IDX["UP"]] = torch.tensor([0.0, 1.0])
    directions_by_action[ACTION_TO_IDX["RIGHT"]] = torch.tensor([-1.0, 0.0])
    directions_by_action[ACTION_TO_IDX["DOWN"]] = torch.tensor([0.0, -1.0])
    class_gaps = torch.zeros(4)
    class_gaps[ACTION_TO_IDX["LEFT"]] = 2.0
    class_gaps[ACTION_TO_IDX["UP"]] = 4.0

    target, gap, actions = build_target_direction(["UP", "LEFT"], directions_by_action, class_gaps)

    expected = torch.tensor([1.0, 1.0])
    expected = expected / expected.norm()
    assert actions == ["LEFT", "UP"]
    assert gap == 3.0
    assert torch.allclose(target, expected)


def test_action_contrast_direction_subtracts_mean_wrong_actions():
    directions_by_action = torch.zeros(4, 2)
    directions_by_action[ACTION_TO_IDX["LEFT"]] = torch.tensor([1.0, 0.0])
    directions_by_action[ACTION_TO_IDX["RIGHT"]] = torch.tensor([-1.0, 0.0])
    directions_by_action[ACTION_TO_IDX["UP"]] = torch.tensor([0.0, 1.0])
    directions_by_action[ACTION_TO_IDX["DOWN"]] = torch.tensor([0.0, -1.0])
    class_gaps = torch.zeros(4)
    class_gaps[ACTION_TO_IDX["LEFT"]] = 2.5

    contrast, gap, actions = build_action_contrast_direction("LEFT", directions_by_action, class_gaps)

    assert actions == ["LEFT"]
    assert gap == 2.5
    assert torch.allclose(contrast, torch.tensor([1.0, 0.0]))
    assert torch.isclose(contrast.norm(), torch.tensor(1.0))


def test_action_vs_source_direction_subtracts_source_action():
    directions_by_action = torch.zeros(4, 2)
    directions_by_action[ACTION_TO_IDX["LEFT"]] = torch.tensor([1.0, 0.0])
    directions_by_action[ACTION_TO_IDX["RIGHT"]] = torch.tensor([-1.0, 0.0])
    directions_by_action[ACTION_TO_IDX["UP"]] = torch.tensor([0.0, 1.0])
    directions_by_action[ACTION_TO_IDX["DOWN"]] = torch.tensor([0.0, -1.0])
    class_gaps = torch.zeros(4)
    class_gaps[ACTION_TO_IDX["LEFT"]] = 2.5

    contrast, gap, actions = build_action_vs_source_direction("LEFT", "RIGHT", directions_by_action, class_gaps)

    assert actions == ["LEFT"]
    assert gap == 2.5
    assert torch.allclose(contrast, torch.tensor([1.0, 0.0]))
    assert torch.isclose(contrast.norm(), torch.tensor(1.0))


def test_action_mean_diffs_compute_positive_minus_negative_activation_means():
    activations = torch.tensor(
        [
            [2.0, 0.0],
            [4.0, 0.0],
            [0.0, 3.0],
            [0.0, 5.0],
        ]
    )
    labels = torch.zeros(4, 4)
    labels[0, ACTION_TO_IDX["LEFT"]] = 1.0
    labels[1, ACTION_TO_IDX["LEFT"]] = 1.0
    labels[2, ACTION_TO_IDX["RIGHT"]] = 1.0
    labels[3, ACTION_TO_IDX["RIGHT"]] = 1.0

    mean_diffs, counts = compute_action_mean_diffs(activations, labels)

    assert counts[ACTION_TO_IDX["LEFT"]] == {"positive": 2, "negative": 2}
    assert torch.allclose(mean_diffs[ACTION_TO_IDX["LEFT"]], torch.tensor([3.0, -4.0]))
    assert torch.allclose(mean_diffs[ACTION_TO_IDX["RIGHT"]], torch.tensor([-3.0, 4.0]))


def test_action_positive_means_compute_optimal_action_centroids():
    activations = torch.tensor(
        [
            [2.0, 0.0],
            [4.0, 0.0],
            [0.0, 3.0],
            [0.0, 5.0],
        ]
    )
    labels = torch.zeros(4, 4)
    labels[0, ACTION_TO_IDX["LEFT"]] = 1.0
    labels[1, ACTION_TO_IDX["LEFT"]] = 1.0
    labels[2, ACTION_TO_IDX["RIGHT"]] = 1.0
    labels[3, ACTION_TO_IDX["RIGHT"]] = 1.0

    positive_means, counts = compute_action_positive_means(activations, labels)

    assert counts[ACTION_TO_IDX["LEFT"]] == {"positive": 2}
    assert torch.allclose(positive_means[ACTION_TO_IDX["LEFT"]], torch.tensor([3.0, 0.0]))
    assert torch.allclose(positive_means[ACTION_TO_IDX["RIGHT"]], torch.tensor([0.0, 4.0]))


def test_raw_action_contrast_uses_activation_weights_and_gap_along_contrast():
    activation_weights = torch.zeros(4, 2)
    activation_weights[ACTION_TO_IDX["LEFT"]] = torch.tensor([2.0, 0.0])
    activation_weights[ACTION_TO_IDX["RIGHT"]] = torch.tensor([0.0, 1.0])
    activation_weights[ACTION_TO_IDX["UP"]] = torch.tensor([0.0, 1.0])
    activation_weights[ACTION_TO_IDX["DOWN"]] = torch.tensor([0.0, 1.0])
    mean_diffs = torch.zeros(4, 2)
    mean_diffs[ACTION_TO_IDX["LEFT"]] = torch.tensor([3.0, 4.0])

    contrast, gap, actions = build_raw_action_contrast_direction("LEFT", activation_weights, mean_diffs)

    expected = torch.tensor([2.0, -1.0])
    expected = expected / expected.norm()
    assert actions == ["LEFT"]
    assert torch.allclose(contrast, expected)
    assert torch.isclose(torch.tensor(gap), mean_diffs[ACTION_TO_IDX["LEFT"]] @ expected)
    assert torch.isclose(contrast.norm(), torch.tensor(1.0))


def test_centroid_action_contrast_uses_positive_means_and_abs_gap():
    positive_means = torch.zeros(4, 2)
    positive_means[ACTION_TO_IDX["LEFT"]] = torch.tensor([2.0, 0.0])
    positive_means[ACTION_TO_IDX["RIGHT"]] = torch.tensor([0.0, 1.0])
    positive_means[ACTION_TO_IDX["UP"]] = torch.tensor([0.0, 1.0])
    positive_means[ACTION_TO_IDX["DOWN"]] = torch.tensor([0.0, 1.0])
    mean_diffs = torch.zeros(4, 2)
    mean_diffs[ACTION_TO_IDX["LEFT"]] = torch.tensor([-3.0, -4.0])

    contrast, gap, actions = build_centroid_action_contrast_direction("LEFT", positive_means, mean_diffs)

    expected = torch.tensor([2.0, -1.0])
    expected = expected / expected.norm()
    assert actions == ["LEFT"]
    assert torch.allclose(contrast, expected)
    assert torch.isclose(torch.tensor(gap), torch.abs(mean_diffs[ACTION_TO_IDX["LEFT"]] @ expected))
    assert torch.isclose(contrast.norm(), torch.tensor(1.0))


def test_centroid_action_vs_source_uses_positive_means_and_abs_gap():
    positive_means = torch.zeros(4, 2)
    positive_means[ACTION_TO_IDX["LEFT"]] = torch.tensor([2.0, 0.0])
    positive_means[ACTION_TO_IDX["RIGHT"]] = torch.tensor([0.0, 1.0])
    mean_diffs = torch.zeros(4, 2)
    mean_diffs[ACTION_TO_IDX["LEFT"]] = torch.tensor([-3.0, -4.0])

    contrast, gap, actions = build_centroid_action_vs_source_direction("LEFT", "RIGHT", positive_means, mean_diffs)

    expected = torch.tensor([2.0, -1.0])
    expected = expected / expected.norm()
    assert actions == ["LEFT"]
    assert torch.allclose(contrast, expected)
    assert torch.isclose(torch.tensor(gap), torch.abs(mean_diffs[ACTION_TO_IDX["LEFT"]] @ expected))
    assert torch.isclose(contrast.norm(), torch.tensor(1.0))


def test_raw_action_vs_source_uses_activation_weights_and_gap_along_contrast():
    activation_weights = torch.zeros(4, 2)
    activation_weights[ACTION_TO_IDX["LEFT"]] = torch.tensor([2.0, 0.0])
    activation_weights[ACTION_TO_IDX["RIGHT"]] = torch.tensor([0.0, 1.0])
    mean_diffs = torch.zeros(4, 2)
    mean_diffs[ACTION_TO_IDX["LEFT"]] = torch.tensor([3.0, 4.0])

    contrast, gap, actions = build_raw_action_vs_source_direction("LEFT", "RIGHT", activation_weights, mean_diffs)

    expected = torch.tensor([2.0, -1.0])
    expected = expected / expected.norm()
    assert actions == ["LEFT"]
    assert torch.allclose(contrast, expected)
    assert torch.isclose(torch.tensor(gap), mean_diffs[ACTION_TO_IDX["LEFT"]] @ expected)
    assert torch.isclose(contrast.norm(), torch.tensor(1.0))


def test_prompt_reconstruction_uses_step_level_prompt_suffix_tokens():
    trajectory = {
        "prompt": {
            "prompt_prefix_tokens": [_token(10, "P"), _token(11, "R")],
            "prompt_suffix_tokens": [_token(90, "TEMPLATE"), _token(91, "SUFFIX")],
        }
    }
    step = {
        "step_id": 5,
        "grid_state_tokens": [_token(20, "G"), _token(21, "R"), _token(22, "D")],
        "prompt_suffix_tokens": [_token(30, "S"), _token(31, "T"), _token(32, "E"), _token(33, "P")],
    }

    prompt_row = build_step_prompt_row(trajectory, step, trajectory_name="traj", step_index=0)

    assert prompt_row["full_token_ids"] == [10, 11, 20, 21, 22, 30, 31, 32, 33]
    assert prompt_row["prompt_suffix_token_ids"] == [30, 31, 32, 33]
    assert prompt_row["n_suffix"] == 4


def test_last_three_suffix_positions_are_absolute_prompt_positions():
    prompt_row = {
        "full_token_ids": [10, 11, 20, 21, 22, 30, 31, 32, 33],
        "prompt_len": 9,
        "n_suffix": 4,
    }

    relative, absolute = resolve_last_suffix_positions(prompt_row)

    assert relative == [1, 2, 3]
    assert absolute == [6, 7, 8]


def _result_row(
    *,
    layer: int,
    condition: str,
    alpha: float | None,
    emitted_action: str | None,
    true_action: str,
    astar_actions: list[str],
    seed: int = 1,
    text_tail: str = "tail",
) -> dict:
    return {
        "layer": layer,
        "condition": condition,
        "alpha": alpha,
        "emitted_action": emitted_action,
        "emitted_action_label": emitted_action or "PARSE_FAIL",
        "true_action": true_action,
        "astar_actions": astar_actions,
        "is_original_true_optimal": true_action in set(astar_actions),
        "trajectory_name": "traj",
        "step_id": 0,
        "sample_idx": 0,
        "seed": seed,
        "text_tail": text_tail,
    }


def test_result_aggregation_computes_optimal_true_and_conversion_rates():
    rows = [
        _result_row(
            layer=23,
            condition="no_hook",
            alpha=None,
            emitted_action="LEFT",
            true_action="LEFT",
            astar_actions=["RIGHT"],
        ),
        _result_row(
            layer=23,
            condition="no_hook",
            alpha=None,
            emitted_action="UP",
            true_action="UP",
            astar_actions=["UP"],
        ),
        _result_row(
            layer=23,
            condition="hooked",
            alpha=1.0,
            emitted_action="RIGHT",
            true_action="LEFT",
            astar_actions=["RIGHT"],
        ),
        _result_row(
            layer=23,
            condition="hooked",
            alpha=1.0,
            emitted_action=None,
            true_action="UP",
            astar_actions=["UP"],
        ),
    ]

    summaries = aggregate_results(rows)
    by_key = {(row["condition"], row["alpha"]): row for row in summaries}

    no_hook = by_key[("no_hook", None)]
    hooked = by_key[("hooked", 1.0)]

    assert no_hook["optimal_action_rate"] == 0.5
    assert no_hook["true_action_rate"] == 1.0
    assert no_hook["nonoptimal_to_optimal_rate"] == 0.0
    assert hooked["optimal_action_rate"] == 0.5
    assert hooked["true_action_rate"] == 0.0
    assert hooked["nonoptimal_to_optimal_rate"] == 1.0
    assert hooked["parse_fail_count"] == 1
    assert hooked["true_action_rate_delta_vs_no_hook"] == -1.0


def test_all_alpha_conditions_share_generation_seed_for_paired_comparison():
    alpha_zero_seed = steering_sample_seed(
        42,
        layer=23,
        trajectory_name="traj_a",
        step_id=7,
        sample_idx=3,
    )
    alpha_one_seed = steering_sample_seed(
        42,
        layer=23,
        trajectory_name="traj_a",
        step_id=7,
        sample_idx=3,
    )
    next_sample_seed = steering_sample_seed(
        42,
        layer=23,
        trajectory_name="traj_a",
        step_id=7,
        sample_idx=4,
    )

    assert alpha_zero_seed == alpha_one_seed
    assert alpha_zero_seed != next_sample_seed


def test_noop_agreement_rows_compare_no_hook_and_alpha_zero_hook():
    rows = [
        _result_row(
            layer=23,
            condition="no_hook",
            alpha=None,
            emitted_action="LEFT",
            true_action="LEFT",
            astar_actions=["LEFT"],
            seed=7,
            text_tail="same",
        ),
        _result_row(
            layer=23,
            condition="hooked",
            alpha=0.0,
            emitted_action="LEFT",
            true_action="LEFT",
            astar_actions=["LEFT"],
            seed=7,
            text_tail="same",
        ),
        _result_row(
            layer=23,
            condition="hooked",
            alpha=1.0,
            emitted_action="RIGHT",
            true_action="LEFT",
            astar_actions=["LEFT"],
            seed=8,
            text_tail="different",
        ),
    ]

    noop_rows = build_noop_agreement_rows(rows)

    assert noop_rows == [
        {
            "layer": 23,
            "pairs": 1,
            "missing_pairs": 0,
            "same_seed_count": 1,
            "same_seed_rate": 1.0,
            "same_action_count": 1,
            "same_action_rate": 1.0,
            "same_text_tail_count": 1,
            "same_text_tail_rate": 1.0,
        }
    ]


def test_paired_effect_rows_compare_hooked_conditions_to_no_hook():
    rows = [
        _result_row(
            layer=7,
            condition="no_hook",
            alpha=None,
            emitted_action="RIGHT",
            true_action="RIGHT",
            astar_actions=["DOWN"],
            seed=11,
            text_tail="baseline",
        ),
        _result_row(
            layer=7,
            condition="hooked",
            alpha=8.0,
            emitted_action="DOWN",
            true_action="RIGHT",
            astar_actions=["DOWN"],
            seed=11,
            text_tail="hooked",
        ),
        _result_row(
            layer=7,
            condition="no_hook",
            alpha=None,
            emitted_action="DOWN",
            true_action="RIGHT",
            astar_actions=["DOWN"],
            seed=12,
            text_tail="baseline2",
        ),
        _result_row(
            layer=7,
            condition="hooked",
            alpha=8.0,
            emitted_action="LEFT",
            true_action="RIGHT",
            astar_actions=["DOWN"],
            seed=12,
            text_tail="hooked2",
        ),
    ]
    rows[2]["step_id"] = 1
    rows[3]["step_id"] = 1

    paired_rows = build_paired_effect_rows(rows)

    assert paired_rows == [
        {
            "layer": 7,
            "alpha": 8.0,
            "pairs": 2,
            "same_seed_count": 2,
            "same_seed_rate": 1.0,
            "changed_action_count": 2,
            "changed_action_rate": 1.0,
            "changed_text_tail_count": 2,
            "changed_text_tail_rate": 1.0,
            "nonoptimal_to_optimal_count": 1,
            "nonoptimal_to_optimal_rate": 0.5,
            "optimal_to_nonoptimal_count": 1,
            "optimal_to_nonoptimal_rate": 0.5,
            "true_to_not_true_count": 1,
            "true_to_not_true_rate": 0.5,
            "not_true_to_true_count": 0,
            "not_true_to_true_rate": 0.0,
        }
    ]


def test_direction_control_rows_compare_desired_and_wrong_single_action_directions():
    rows = [
        _result_row(
            layer=7,
            condition="no_hook",
            alpha=None,
            emitted_action="RIGHT",
            true_action="RIGHT",
            astar_actions=["DOWN"],
            seed=11,
            text_tail="baseline",
        ),
        _result_row(
            layer=7,
            condition="hooked",
            alpha=8.0,
            emitted_action="DOWN",
            true_action="RIGHT",
            astar_actions=["DOWN"],
            seed=11,
            text_tail="desired",
        ),
        _result_row(
            layer=7,
            condition="hooked",
            alpha=8.0,
            emitted_action="LEFT",
            true_action="RIGHT",
            astar_actions=["DOWN"],
            seed=11,
            text_tail="wrong",
        ),
    ]
    rows[1].update(
        {
            "direction_mode": "single_action",
            "direction_label": "DOWN",
            "is_desired_direction": True,
        }
    )
    rows[2].update(
        {
            "direction_mode": "single_action",
            "direction_label": "LEFT",
            "is_desired_direction": False,
        }
    )

    control_rows = build_direction_control_rows(rows)

    assert control_rows == [
        {
            "layer": 7,
            "alpha": 8.0,
            "direction_label": "DOWN",
            "is_desired_direction": True,
            "pairs": 1,
            "same_seed_rate": 1.0,
            "changed_action_rate": 1.0,
            "baseline_steered_action_rate": 0.0,
            "hooked_steered_action_rate": 1.0,
            "steered_action_rate_delta": 1.0,
            "baseline_desired_action_rate": 0.0,
            "hooked_desired_action_rate": 1.0,
            "desired_action_rate_delta": 1.0,
            "nonoptimal_to_desired_rate": 1.0,
            "desired_to_nondesired_rate": 0.0,
        },
        {
            "layer": 7,
            "alpha": 8.0,
            "direction_label": "LEFT",
            "is_desired_direction": False,
            "pairs": 1,
            "same_seed_rate": 1.0,
            "changed_action_rate": 1.0,
            "baseline_steered_action_rate": 0.0,
            "hooked_steered_action_rate": 1.0,
            "steered_action_rate_delta": 1.0,
            "baseline_desired_action_rate": 0.0,
            "hooked_desired_action_rate": 0.0,
            "desired_action_rate_delta": 0.0,
            "nonoptimal_to_desired_rate": 0.0,
            "desired_to_nondesired_rate": 0.0,
        },
    ]


def test_direction_control_rows_include_raw_action_vs_baseline_controls():
    rows = [
        _result_row(
            layer=15,
            condition="no_hook",
            alpha=None,
            emitted_action="LEFT",
            true_action="LEFT",
            astar_actions=["RIGHT"],
            seed=22,
            text_tail="baseline",
        ),
        _result_row(
            layer=15,
            condition="hooked",
            alpha=2.0,
            emitted_action="RIGHT",
            true_action="LEFT",
            astar_actions=["RIGHT"],
            seed=22,
            text_tail="desired",
        ),
    ]
    rows[1].update(
        {
            "direction_mode": "raw_action_vs_baseline",
            "direction_label": "RIGHT",
            "source_action": "LEFT",
            "is_desired_direction": True,
        }
    )

    control_rows = build_direction_control_rows(rows)

    assert len(control_rows) == 1
    assert control_rows[0]["direction_label"] == "RIGHT"
    assert control_rows[0]["is_desired_direction"] is True
    assert control_rows[0]["desired_action_rate_delta"] == 1.0
    assert control_rows[0]["nonoptimal_to_desired_rate"] == 1.0


def test_direction_control_advantage_rows_compare_controls_at_same_alpha():
    control_rows = [
        {
            "layer": 15,
            "alpha": -32.0,
            "direction_label": "DOWN",
            "is_desired_direction": True,
            "pairs": 4,
            "desired_action_rate_delta": 0.75,
            "nonoptimal_to_desired_rate": 0.75,
        },
        {
            "layer": 15,
            "alpha": -32.0,
            "direction_label": "LEFT",
            "is_desired_direction": False,
            "pairs": 4,
            "desired_action_rate_delta": 0.5,
            "nonoptimal_to_desired_rate": 0.5,
        },
        {
            "layer": 15,
            "alpha": 8.0,
            "direction_label": "DOWN",
            "is_desired_direction": True,
            "pairs": 4,
            "desired_action_rate_delta": 0.25,
            "nonoptimal_to_desired_rate": 0.25,
        },
        {
            "layer": 15,
            "alpha": 8.0,
            "direction_label": "RIGHT",
            "is_desired_direction": False,
            "pairs": 4,
            "desired_action_rate_delta": 0.75,
            "nonoptimal_to_desired_rate": 0.75,
        },
    ]

    advantage_rows = build_direction_control_advantage_rows(control_rows)

    assert advantage_rows == [
        {
            "layer": 15,
            "alpha": -32.0,
            "desired_direction_label": "DOWN",
            "desired_pairs": 4,
            "desired_action_rate_delta": 0.75,
            "best_wrong_direction_label": "LEFT",
            "best_wrong_pairs": 4,
            "best_wrong_desired_action_rate_delta": 0.5,
            "desired_advantage_delta": 0.25,
            "desired_nonoptimal_to_desired_rate": 0.75,
            "best_wrong_nonoptimal_to_desired_rate": 0.5,
            "nonoptimal_to_desired_advantage": 0.25,
            "supports_desired_direction": True,
        },
        {
            "layer": 15,
            "alpha": 8.0,
            "desired_direction_label": "DOWN",
            "desired_pairs": 4,
            "desired_action_rate_delta": 0.25,
            "best_wrong_direction_label": "RIGHT",
            "best_wrong_pairs": 4,
            "best_wrong_desired_action_rate_delta": 0.75,
            "desired_advantage_delta": -0.5,
            "desired_nonoptimal_to_desired_rate": 0.25,
            "best_wrong_nonoptimal_to_desired_rate": 0.75,
            "nonoptimal_to_desired_advantage": -0.5,
            "supports_desired_direction": False,
        },
    ]


def test_direction_control_overall_rows_weight_desired_and_wrong_groups():
    control_rows = [
        {
            "layer": 15,
            "alpha": -32.0,
            "direction_label": "DOWN",
            "is_desired_direction": True,
            "pairs": 10,
            "changed_action_rate": 0.4,
            "baseline_desired_action_rate": 0.2,
            "hooked_desired_action_rate": 0.6,
            "desired_action_rate_delta": 0.4,
            "nonoptimal_to_desired_rate": 0.5,
            "desired_to_nondesired_rate": 0.1,
        },
        {
            "layer": 15,
            "alpha": -32.0,
            "direction_label": "LEFT",
            "is_desired_direction": False,
            "pairs": 10,
            "changed_action_rate": 0.2,
            "baseline_desired_action_rate": 0.2,
            "hooked_desired_action_rate": 0.3,
            "desired_action_rate_delta": 0.1,
            "nonoptimal_to_desired_rate": 0.2,
            "desired_to_nondesired_rate": 0.1,
        },
        {
            "layer": 15,
            "alpha": -32.0,
            "direction_label": "UP",
            "is_desired_direction": False,
            "pairs": 30,
            "changed_action_rate": 0.4,
            "baseline_desired_action_rate": 0.2,
            "hooked_desired_action_rate": 0.1,
            "desired_action_rate_delta": -0.1,
            "nonoptimal_to_desired_rate": 0.0,
            "desired_to_nondesired_rate": 0.1,
        },
    ]

    overall_rows = build_direction_control_overall_rows(control_rows)
    advantage_rows = build_direction_control_overall_advantage_rows(overall_rows)

    assert overall_rows == [
        {
            "layer": 15,
            "alpha": -32.0,
            "is_desired_direction": False,
            "direction_group": "wrong",
            "direction_labels": "LEFT,UP",
            "pairs": 40,
            "changed_action_rate": 0.35,
            "baseline_desired_action_rate": 0.2,
            "hooked_desired_action_rate": 0.15,
            "desired_action_rate_delta": -0.05000000000000002,
            "nonoptimal_to_desired_rate": 0.05,
            "desired_to_nondesired_rate": 0.1,
        },
        {
            "layer": 15,
            "alpha": -32.0,
            "is_desired_direction": True,
            "direction_group": "desired",
            "direction_labels": "DOWN",
            "pairs": 10,
            "changed_action_rate": 0.4,
            "baseline_desired_action_rate": 0.2,
            "hooked_desired_action_rate": 0.6,
            "desired_action_rate_delta": 0.39999999999999997,
            "nonoptimal_to_desired_rate": 0.5,
            "desired_to_nondesired_rate": 0.1,
        },
    ]
    assert advantage_rows == [
        {
            "layer": 15,
            "alpha": -32.0,
            "desired_pairs": 10,
            "wrong_pairs": 40,
            "desired_action_rate_delta": 0.39999999999999997,
            "wrong_desired_action_rate_delta": -0.05000000000000002,
            "desired_advantage_delta": 0.44999999999999996,
            "desired_nonoptimal_to_desired_rate": 0.5,
            "wrong_nonoptimal_to_desired_rate": 0.05,
            "nonoptimal_to_desired_advantage": 0.45,
            "supports_desired_direction": True,
        }
    ]
