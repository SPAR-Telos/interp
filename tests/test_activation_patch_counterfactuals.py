"""Tests for activation patching on counterfactual trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn import (
    PATCH_SITE_PROMPT_BOUNDARY,
    _build_summary_dataframe,
    activation_patch_counterfactuals,
    compute_outcome_flags,
    discover_patch_pairs,
    extract_action_from_text,
    resolve_pre_final_boundary_token_ids,
    resolve_prompt_boundary_token_ids,
)


def _make_output_tokens() -> list[dict]:
    return [
        {"id": 0, "token": "<|channel|>", "token_id": 200005, "token_groups": ["output", "template"]},
        {"id": 1, "token": "analysis", "token_id": 35644, "token_groups": ["output", "template"]},
        {"id": 2, "token": "<|message|>", "token_id": 200008, "token_groups": ["output", "template"]},
        {"id": 3, "token": "Reason", "token_id": 10, "token_groups": ["output", "analysis"]},
        {"id": 4, "token": "<|end|>", "token_id": 200007, "token_groups": ["output", "template"]},
        {"id": 5, "token": "<|start|>", "token_id": 200006, "token_groups": ["output", "template"]},
        {"id": 6, "token": "assistant", "token_id": 173781, "token_groups": ["output", "template"]},
        {"id": 7, "token": "<|channel|>", "token_id": 200005, "token_groups": ["output", "template"]},
        {"id": 8, "token": "final", "token_id": 17196, "token_groups": ["output", "template"]},
        {"id": 9, "token": "<|message|>", "token_id": 200008, "token_groups": ["output", "template"]},
        {"id": 10, "token": "{", "token_id": 90, "token_groups": ["output", "final"]},
        {"id": 11, "token": '"action"', "token_id": 1976, "token_groups": ["output", "final"]},
        {"id": 12, "token": '"DOWN"', "token_id": 49412, "token_groups": ["output", "final", "action"]},
        {"id": 13, "token": "}", "token_id": 91, "token_groups": ["output", "final"]},
    ]


def _make_trajectory(action: str) -> dict:
    return {
        "model_params": {"model_id": "dummy/model"},
        "prompt": {
            "prompt_prefix_tokens": [
                {"id": 0, "token": "prefix", "token_id": 101, "token_groups": ["prompt"]},
                {"id": 1, "token": "grid", "token_id": 102, "token_groups": ["prompt"]},
            ],
            "prompt_suffix_tokens": [
                {"id": 0, "token": "<|end|>", "token_id": 200007, "token_groups": ["prompt", "template"]},
                {"id": 1, "token": "<|start|>", "token_id": 200006, "token_groups": ["prompt", "template"]},
                {"id": 2, "token": "assistant", "token_id": 173781, "token_groups": ["prompt", "template"]},
            ],
        },
        "steps": [
            {
                "step_id": 0,
                "grid_state_tokens": [
                    {"id": 0, "token": "A", "token_id": 32, "token_groups": ["grid_state"]},
                ],
                "output_tokens": _make_output_tokens(),
                "agent_action": action,
            }
        ],
    }


def test_resolve_prompt_boundary_token_ids_valid():
    tokens = _make_trajectory("DOWN")["prompt"]["prompt_suffix_tokens"]
    assert resolve_prompt_boundary_token_ids(tokens) == [0, 1, 2]


def test_resolve_prompt_boundary_token_ids_wrong_tokens():
    tokens = _make_trajectory("DOWN")["prompt"]["prompt_suffix_tokens"]
    tokens[2]["token"] = "user"
    try:
        resolve_prompt_boundary_token_ids(tokens)
    except ValueError as exc:
        assert "expected template boundary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong prompt boundary tokens")


def test_resolve_pre_final_boundary_token_ids_valid():
    assert resolve_pre_final_boundary_token_ids(_make_output_tokens()) == [4, 5, 6]


def test_resolve_pre_final_boundary_token_ids_missing():
    tokens = _make_output_tokens()[:4]
    try:
        resolve_pre_final_boundary_token_ids(tokens)
    except ValueError as exc:
        assert "Could not find" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing pre-final boundary")


def test_resolve_pre_final_boundary_token_ids_wrong_tokens():
    tokens = _make_output_tokens()
    tokens[6]["token"] = "user"
    try:
        resolve_pre_final_boundary_token_ids(tokens)
    except ValueError as exc:
        assert "Could not find" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong pre-final boundary tokens")


def test_resolve_pre_final_boundary_token_ids_ambiguous():
    tokens = _make_output_tokens() + [
        {"id": 20, "token": "<|end|>", "token_id": 200007, "token_groups": ["output", "template"]},
        {"id": 21, "token": "<|start|>", "token_id": 200006, "token_groups": ["output", "template"]},
        {"id": 22, "token": "assistant", "token_id": 173781, "token_groups": ["output", "template"]},
        {"id": 23, "token": "<|channel|>", "token_id": 200005, "token_groups": ["output", "template"]},
        {"id": 24, "token": "final", "token_id": 17196, "token_groups": ["output", "template"]},
        {"id": 25, "token": "<|message|>", "token_id": 200008, "token_groups": ["output", "template"]},
    ]
    try:
        resolve_pre_final_boundary_token_ids(tokens)
    except ValueError as exc:
        assert "multiple candidate" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambiguous pre-final boundary")


def test_extract_action_from_text():
    text = '<|channel|>final<|message|>{"action":"LEFT"}'
    assert extract_action_from_text(text) == "LEFT"


def test_extract_action_from_text_tolerates_case_and_formatting():
    assert extract_action_from_text('{"action": "down"}') == "DOWN"
    assert extract_action_from_text("action = right") == "RIGHT"
    assert extract_action_from_text("noise\n{'action':'up'}") == "UP"


def test_compute_outcome_flags_same_and_disjoint():
    same_flags = compute_outcome_flags(
        direction="original_to_counterfactual",
        relation_to_original="same",
        source_optimal_actions=("DOWN",),
        target_optimal_actions=("DOWN",),
        baseline_target_action="DOWN",
        patched_target_action="DOWN",
    )
    assert same_flags["success_primary"] is True

    disjoint_flags = compute_outcome_flags(
        direction="counterfactual_to_original",
        relation_to_original="disjoint",
        source_optimal_actions=("RIGHT",),
        target_optimal_actions=("DOWN",),
        baseline_target_action="DOWN",
        patched_target_action="RIGHT",
    )
    assert disjoint_flags["success_primary"] is True
    assert disjoint_flags["patched_in_source_optimal_set"] is True


def test_discover_patch_pairs(tmp_path: Path):
    original_root = tmp_path / "data" / "trajectories_test_full"
    original_root.mkdir(parents=True)
    counterfactual_root = tmp_path / "data" / "counterfactual_trajectories"
    counterfactual_root.mkdir(parents=True)
    metadata_root = tmp_path / "data" / "counterfactual_grids" / "size9"
    metadata_root.mkdir(parents=True)

    original_path = original_root / "original.json"
    counterfactual_path = counterfactual_root / "cf.json"
    original_path.write_text(json.dumps(_make_trajectory("DOWN")))
    counterfactual_path.write_text(json.dumps(_make_trajectory("RIGHT")))

    metadata_path = metadata_root / "counterfactuals.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_relative_path": str(original_path.relative_to(tmp_path)),
                "source_step_id": 0,
                "requested_instance_id": 0,
                "actual_instance_id": 1,
                "original_optimal_action_set_names": ["DOWN"],
                "agent_moved_counterfactuals": [
                    {
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "optimal_action_set_names": ["RIGHT"],
                    }
                ],
            }
        )
    )

    (counterfactual_root / "counterfactual_trajectory_batch_summary.json").write_text(
        json.dumps(
            {
                "successes": [
                    {
                        "source_counterfactual_path": str(metadata_path),
                        "output_path": str(counterfactual_path),
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "relation_to_original": "disjoint",
                        "grid_size": 9,
                        "grid_complexity": 0.0,
                        "source_filename": "original.json",
                    }
                ]
            }
        )
    )

    pairs, skipped = discover_patch_pairs(
        counterfactual_metadata_root=str(tmp_path / "data" / "counterfactual_grids"),
        counterfactual_trajectories_dir=str(counterfactual_root),
        original_trajectories_dir=str(original_root),
    )

    assert not skipped
    assert len(pairs) == 1
    assert pairs[0].original_optimal_actions == ("DOWN",)
    assert pairs[0].counterfactual_optimal_actions == ("RIGHT",)


def test_build_summary_dataframe():
    df = _build_summary_dataframe(
        [
            {
                "counterfactual_id": "x",
                "direction": "original_to_counterfactual",
                "layer": 7,
                "patch_site": PATCH_SITE_PROMPT_BOUNDARY,
                "relation_to_original": "disjoint",
                "counterfactual_type": "agent_moved",
                "grid_size": 9,
                "grid_complexity": 0.0,
                "success_primary": True,
                "action_changed_vs_baseline": True,
                "patched_in_source_optimal_set": True,
                "patched_in_target_optimal_set": False,
                "patched_in_expected_optimal_set": True,
                "parse_failure": False,
            }
        ]
    )
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["runs"] == 1
    assert df.iloc[0]["success_rate_primary"] == 1.0


def test_activation_patch_counterfactuals_smoke(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "data"
    original_root = data_root / "trajectories_test_full"
    counterfactual_root = data_root / "counterfactual_trajectories"
    metadata_root = data_root / "counterfactual_grids" / "size9"
    original_root.mkdir(parents=True)
    counterfactual_root.mkdir(parents=True)
    metadata_root.mkdir(parents=True)

    original_path = original_root / "original.json"
    counterfactual_path = counterfactual_root / "cf.json"
    original_path.write_text(json.dumps(_make_trajectory("DOWN")))
    counterfactual_path.write_text(json.dumps(_make_trajectory("RIGHT")))

    metadata_path = metadata_root / "counterfactuals.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_relative_path": str(original_path.relative_to(tmp_path)),
                "source_step_id": 0,
                "requested_instance_id": 0,
                "actual_instance_id": 1,
                "original_optimal_action_set_names": ["DOWN"],
                "agent_moved_counterfactuals": [
                    {
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "optimal_action_set_names": ["RIGHT"],
                    }
                ],
            }
        )
    )
    (counterfactual_root / "counterfactual_trajectory_batch_summary.json").write_text(
        json.dumps(
            {
                "successes": [
                    {
                        "source_counterfactual_path": str(metadata_path),
                        "output_path": str(counterfactual_path),
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "relation_to_original": "disjoint",
                        "grid_size": 9,
                        "grid_complexity": 0.0,
                        "source_filename": "original.json",
                    }
                ]
            }
        )
    )

    calls = {"capture": 0, "generate": 0}

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            return '<|channel|>final<|message|>{"action":"DOWN"}'

    class FakeConfig:
        num_hidden_layers = 1

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.config = FakeConfig()
            self.tokenizer = FakeTokenizer()

    def fake_capture(model, trajectory, step_idx, patch_site, layer):
        calls["capture"] += 1
        return {"activations": torch.ones(3, 4), "absolute_positions": [3, 4, 5], "input_token_count": 6}

    def fake_generate(model, trajectory, step_idx, patch_site, layer=None, donor_activations=None):
        calls["generate"] += 1
        action = "RIGHT" if donor_activations is not None else "DOWN"
        return {
            "output_text": f'<|channel|>final<|message|>{{"action":"{action}"}}',
            "action": action,
            "parse_failure": False,
            "input_token_count": 6,
            "generated_token_count": 8,
            "absolute_positions": [3, 4, 5],
        }

    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn.StandardizedTransformer",
        FakeModel,
    )
    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn._capture_donor_activations",
        fake_capture,
    )
    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn._run_target_generation",
        fake_generate,
    )

    output_dir = tmp_path / "out"
    manifest = activation_patch_counterfactuals(
        counterfactual_metadata_root=str(data_root / "counterfactual_grids"),
        counterfactual_trajectories_dir=str(counterfactual_root),
        original_trajectories_dir=str(original_root),
        patch_sites=PATCH_SITE_PROMPT_BOUNDARY,
        layers="0",
        directions="counterfactual_to_original",
        output_dir=str(output_dir),
        verbose=False,
        overwrite=True,
    )

    assert calls["capture"] == 1
    assert calls["generate"] == 2
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "runs.jsonl").exists()
    assert (output_dir / "summary_by_group.csv").exists()
    assert manifest["run_counts"]["total_runs"] == 1


def test_activation_patch_counterfactuals_resume_skips_completed(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "data"
    original_root = data_root / "trajectories_test_full"
    counterfactual_root = data_root / "counterfactual_trajectories"
    metadata_root = data_root / "counterfactual_grids" / "size9"
    original_root.mkdir(parents=True)
    counterfactual_root.mkdir(parents=True)
    metadata_root.mkdir(parents=True)

    original_path = original_root / "original.json"
    counterfactual_path = counterfactual_root / "cf.json"
    original_path.write_text(json.dumps(_make_trajectory("DOWN")))
    counterfactual_path.write_text(json.dumps(_make_trajectory("RIGHT")))

    metadata_path = metadata_root / "counterfactuals.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_relative_path": str(original_path.relative_to(tmp_path)),
                "source_step_id": 0,
                "requested_instance_id": 0,
                "actual_instance_id": 1,
                "original_optimal_action_set_names": ["DOWN"],
                "agent_moved_counterfactuals": [
                    {
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "optimal_action_set_names": ["RIGHT"],
                    }
                ],
            }
        )
    )
    (counterfactual_root / "counterfactual_trajectory_batch_summary.json").write_text(
        json.dumps(
            {
                "successes": [
                    {
                        "source_counterfactual_path": str(metadata_path),
                        "output_path": str(counterfactual_path),
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "relation_to_original": "disjoint",
                        "grid_size": 9,
                        "grid_complexity": 0.0,
                        "source_filename": "original.json",
                    }
                ]
            }
        )
    )

    class FakeConfig:
        num_hidden_layers = 1

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.config = FakeConfig()

    def fail_capture(*args, **kwargs):
        raise AssertionError("resume path should skip donor capture for completed runs")

    def fail_generate(*args, **kwargs):
        raise AssertionError("resume path should skip target generation for completed runs")

    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn.StandardizedTransformer",
        FakeModel,
    )
    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn._capture_donor_activations",
        fail_capture,
    )
    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn._run_target_generation",
        fail_generate,
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing_record = {
        "run_key": "agent_moved_00|0|counterfactual_to_original|prompt_boundary|0",
        "counterfactual_id": "agent_moved_00",
        "step_idx": 0,
        "direction": "counterfactual_to_original",
        "patch_site": "prompt_boundary",
        "layer": 0,
        "success_primary": True,
        "parse_failure": False,
        "error": None,
        "action_changed_vs_baseline": True,
        "patched_in_source_optimal_set": True,
        "patched_in_target_optimal_set": False,
        "patched_in_expected_optimal_set": True,
    }
    (output_dir / "runs.jsonl").write_text(json.dumps(existing_record) + "\n")

    manifest = activation_patch_counterfactuals(
        counterfactual_metadata_root=str(data_root / "counterfactual_grids"),
        counterfactual_trajectories_dir=str(counterfactual_root),
        original_trajectories_dir=str(original_root),
        patch_sites=PATCH_SITE_PROMPT_BOUNDARY,
        layers="0",
        directions="counterfactual_to_original",
        output_dir=str(output_dir),
        verbose=False,
    )

    assert manifest["progress"]["resumed_run_count"] == 1
    assert manifest["run_counts"]["total_runs"] == 1


def test_activation_patch_counterfactuals_skips_non_optimal_recorded_actions_by_default(
    monkeypatch, tmp_path: Path
):
    data_root = tmp_path / "data"
    original_root = data_root / "trajectories_test_full"
    counterfactual_root = data_root / "counterfactual_trajectories"
    metadata_root = data_root / "counterfactual_grids" / "size9"
    original_root.mkdir(parents=True)
    counterfactual_root.mkdir(parents=True)
    metadata_root.mkdir(parents=True)

    original_path = original_root / "original.json"
    counterfactual_path = counterfactual_root / "cf.json"
    original_path.write_text(json.dumps(_make_trajectory("LEFT")))
    counterfactual_path.write_text(json.dumps(_make_trajectory("UP")))

    metadata_path = metadata_root / "counterfactuals.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_relative_path": str(original_path.relative_to(tmp_path)),
                "source_step_id": 0,
                "requested_instance_id": 0,
                "actual_instance_id": 1,
                "original_optimal_action_set_names": ["DOWN"],
                "agent_moved_counterfactuals": [
                    {
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "optimal_action_set_names": ["RIGHT"],
                    }
                ],
            }
        )
    )
    (counterfactual_root / "counterfactual_trajectory_batch_summary.json").write_text(
        json.dumps(
            {
                "successes": [
                    {
                        "source_counterfactual_path": str(metadata_path),
                        "output_path": str(counterfactual_path),
                        "counterfactual_id": "agent_moved_00",
                        "counterfactual_type": "agent_moved",
                        "relation_to_original": "disjoint",
                        "grid_size": 9,
                        "grid_complexity": 0.0,
                        "source_filename": "original.json",
                    }
                ]
            }
        )
    )

    class FakeConfig:
        num_hidden_layers = 1

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.config = FakeConfig()

    def fail_capture(*args, **kwargs):
        raise AssertionError("filtered runs should not capture donor activations")

    def fail_generate(*args, **kwargs):
        raise AssertionError("filtered runs should not run generation")

    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn.StandardizedTransformer",
        FakeModel,
    )
    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn._capture_donor_activations",
        fail_capture,
    )
    monkeypatch.setattr(
        "telos_interp.commands.activation_patch_counterfactuals.activation_patch_counterfactuals_fn._run_target_generation",
        fail_generate,
    )

    output_dir = tmp_path / "out"
    manifest = activation_patch_counterfactuals(
        counterfactual_metadata_root=str(data_root / "counterfactual_grids"),
        counterfactual_trajectories_dir=str(counterfactual_root),
        original_trajectories_dir=str(original_root),
        patch_sites=PATCH_SITE_PROMPT_BOUNDARY,
        layers="0",
        directions="counterfactual_to_original",
        output_dir=str(output_dir),
        verbose=False,
        overwrite=True,
    )

    assert manifest["run_counts"]["skipped_count"] == 1
    runs = [json.loads(line) for line in (output_dir / "runs.jsonl").read_text().splitlines()]
    assert runs[0]["skipped"] is True
    assert runs[0]["skip_reason"] == "recorded_action_not_in_optimal_set"
