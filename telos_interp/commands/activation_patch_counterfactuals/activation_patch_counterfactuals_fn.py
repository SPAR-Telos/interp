"""Activation patching between original and counterfactual trajectories."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from nnterp import StandardizedTransformer
from tqdm import tqdm

from telos_interp.commands.gather_activations.gather_activations_fn import _resolve_torch_dtype
from telos_interp.commands.gather_activations.gather_activations_utils import (
    parse_index_specification,
)

VALID_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
PATCH_SITE_PROMPT_BOUNDARY = "prompt_boundary"
PATCH_SITE_PRE_FINAL_BOUNDARY = "pre_final_boundary"
PATCH_SITE_MODIFIED_GRID_CELLS = "modified_grid_cells"
EXPECTED_BOUNDARY_TOKENS = ("<|end|>", "<|start|>", "assistant")
EXPECTED_FINAL_PREFIX = ("<|channel|>", "final", "<|message|>")
VALID_PATCH_SITES = (PATCH_SITE_PROMPT_BOUNDARY, PATCH_SITE_PRE_FINAL_BOUNDARY, PATCH_SITE_MODIFIED_GRID_CELLS)
VALID_DIRECTIONS = ("original_to_counterfactual", "counterfactual_to_original")
VALID_RELATIONS = ("same", "disjoint", "overlap")
VALID_COUNTERFACTUAL_TYPES = ("agent_moved", "goal_moved")
VALID_EVALUATION_MODES = ("answer_forcing", "free_generation")


@dataclass(frozen=True)
class PatchPair:
    """Resolved original/counterfactual pair plus metadata."""

    metadata_path: str
    counterfactual_id: str
    counterfactual_type: str
    relation_to_original: str
    grid_size: int
    grid_complexity: float
    source_step_id: int
    source_filename: str
    original_trajectory_path: str
    counterfactual_trajectory_path: str
    original_optimal_actions: tuple[str, ...]
    counterfactual_optimal_actions: tuple[str, ...]
    moved_entity: str | None
    from_position: tuple[int, int] | None
    to_position: tuple[int, int] | None
    requested_instance_id: int | None
    actual_instance_id: int | None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _resolve_repo_relative_path(path_str: str, root_candidates: list[Path]) -> Path:
    """Resolve a repo-relative path against a set of candidate roots."""
    candidate = Path(path_str)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    for root in root_candidates:
        resolved = root / candidate
        if resolved.exists():
            return resolved

    return candidate


def _normalize_action_set(actions: list[str] | None) -> tuple[str, ...]:
    if not actions:
        return ()
    return tuple(str(action).upper() for action in actions)


def _normalize_position(position: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if position is None:
        return None
    if len(position) != 2:
        raise ValueError(f"Expected a 2D grid position, got {position}.")
    return int(position[0]), int(position[1])


def _parse_string_spec(spec: str, valid_values: tuple[str, ...], name: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(valid_values)

    values = []
    valid_set = set(valid_values)
    for item in spec.split(","):
        value = item.strip()
        if not value:
            continue
        if value not in valid_set:
            raise ValueError(f"Invalid {name}: {value}. Expected one of {sorted(valid_set)} or 'all'.")
        values.append(value)

    if not values:
        raise ValueError(f"No {name} values selected.")

    return values


def _parse_int_spec(spec: str) -> set[int] | None:
    if spec.strip().lower() == "all":
        return None
    values = set()
    for item in spec.split(","):
        value = item.strip()
        if value:
            values.add(int(value))
    return values


def _parse_float_spec(spec: str) -> set[float] | None:
    if spec.strip().lower() == "all":
        return None
    values = set()
    for item in spec.split(","):
        value = item.strip()
        if value:
            values.add(float(value))
    return values


def resolve_prompt_boundary_token_ids(prompt_suffix_tokens: list[dict[str, Any]]) -> list[int]:
    """Resolve and validate the prompt-side assistant boundary tokens."""
    if len(prompt_suffix_tokens) != 3:
        raise ValueError(
            f"prompt_boundary requires exactly 3 prompt_suffix tokens, found {len(prompt_suffix_tokens)}."
        )

    token_text = tuple(token["token"] for token in prompt_suffix_tokens)
    if token_text != EXPECTED_BOUNDARY_TOKENS:
        raise ValueError(
            "prompt_boundary tokens do not match the expected template boundary. "
            f"Expected {EXPECTED_BOUNDARY_TOKENS}, got {token_text}."
        )

    token_ids = [int(token["id"]) for token in prompt_suffix_tokens]
    if token_ids != [0, 1, 2]:
        raise ValueError(f"prompt_boundary token ids must be [0, 1, 2]. Got {token_ids}.")

    return token_ids


def resolve_pre_final_boundary_token_ids(output_tokens: list[dict[str, Any]]) -> list[int]:
    """Resolve and validate the output-side boundary before the final answer."""
    matches: list[list[int]] = []
    for start_idx in range(0, len(output_tokens) - 5):
        boundary = tuple(token["token"] for token in output_tokens[start_idx : start_idx + 3])
        final_prefix = tuple(token["token"] for token in output_tokens[start_idx + 3 : start_idx + 6])
        if boundary == EXPECTED_BOUNDARY_TOKENS and final_prefix == EXPECTED_FINAL_PREFIX:
            matches.append([int(output_tokens[start_idx + offset]["id"]) for offset in range(3)])

    if not matches:
        raise ValueError("Could not find a unique pre-final assistant boundary in output_tokens.")
    if len(matches) > 1:
        raise ValueError("Found multiple candidate pre-final assistant boundaries in output_tokens.")

    return matches[0]


def resolve_modified_grid_cell_token_ids(
    grid_state_tokens: list[dict[str, Any]],
    grid_size: int,
    modified_positions: tuple[tuple[int, int], ...],
) -> list[int]:
    """Resolve the grid_tile token ids for the modified source/destination cells."""
    grid_tile_ids = [int(token["id"]) for token in grid_state_tokens if "grid_tile" in token.get("token_groups", [])]
    expected_tile_count = grid_size * grid_size
    if len(grid_tile_ids) != expected_tile_count:
        raise ValueError(
            "modified_grid_cells requires exactly one grid_tile token per grid cell. "
            f"Expected {expected_tile_count}, found {len(grid_tile_ids)}."
        )

    resolved_ids: list[int] = []
    for x, y in modified_positions:
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            raise ValueError(f"Modified grid position {(x, y)} is out of bounds for grid_size={grid_size}.")
        resolved_ids.append(grid_tile_ids[(y * grid_size) + x])

    unique_ids = sorted(set(resolved_ids))
    if len(unique_ids) != len(resolved_ids):
        raise ValueError(f"modified_grid_cells resolved duplicate token ids from positions {modified_positions}.")

    return unique_ids


def _build_input_ids_for_patch_site(
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    grid_size: int | None = None,
    modified_positions: tuple[tuple[int, int], ...] | None = None,
    target_mode: str = "capture",
) -> tuple[list[int], list[int], int]:
    """Build input ids up to the selected site and return absolute patch positions."""
    step = trajectory["steps"][step_idx]

    prefix_ids = [token["token_id"] for token in trajectory["prompt"]["prompt_prefix_tokens"]]
    grid_ids = [token["token_id"] for token in step["grid_state_tokens"]]
    suffix_tokens = trajectory["prompt"]["prompt_suffix_tokens"]
    suffix_ids = [token["token_id"] for token in suffix_tokens]
    output_tokens = step["output_tokens"]
    output_ids = [token["token_id"] for token in output_tokens]

    prompt_offset = len(prefix_ids) + len(grid_ids)
    output_offset = prompt_offset + len(suffix_ids)

    if patch_site == PATCH_SITE_PROMPT_BOUNDARY:
        relative_positions = resolve_prompt_boundary_token_ids(suffix_tokens)
        absolute_positions = [prompt_offset + position for position in relative_positions]
        input_ids = prefix_ids + grid_ids + suffix_ids
        remaining_tokens = len(output_ids)
        return input_ids, absolute_positions, remaining_tokens

    if patch_site == PATCH_SITE_PRE_FINAL_BOUNDARY:
        relative_positions = resolve_pre_final_boundary_token_ids(output_tokens)
        absolute_positions = [output_offset + position for position in relative_positions]
        end_idx = max(relative_positions)
        input_ids = prefix_ids + grid_ids + suffix_ids + output_ids[: end_idx + 1]
        remaining_tokens = len(output_ids) - (end_idx + 1)
        return input_ids, absolute_positions, remaining_tokens

    if patch_site == PATCH_SITE_MODIFIED_GRID_CELLS:
        if grid_size is None or modified_positions is None:
            raise ValueError("modified_grid_cells requires grid_size and modified_positions.")
        relative_positions = resolve_modified_grid_cell_token_ids(
            step["grid_state_tokens"], grid_size, modified_positions
        )
        absolute_positions = [len(prefix_ids) + position for position in relative_positions]
        if target_mode == "capture":
            end_idx = max(relative_positions)
            input_ids = prefix_ids + grid_ids[: end_idx + 1]
        elif target_mode == "free_generation":
            input_ids = prefix_ids + grid_ids + suffix_ids
        else:
            raise ValueError(f"Unsupported target_mode for modified_grid_cells: {target_mode}")
        remaining_tokens = len(output_ids)
        return input_ids, absolute_positions, remaining_tokens

    raise ValueError(f"Unsupported patch_site: {patch_site}")


def _capture_donor_activations(
    model: StandardizedTransformer,
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    layer: int,
    grid_size: int | None = None,
    modified_positions: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    """Capture donor activations at the selected boundary tokens."""
    input_ids_list, absolute_positions, _ = _build_input_ids_for_patch_site(
        trajectory,
        step_idx,
        patch_site,
        grid_size=grid_size,
        modified_positions=modified_positions,
        target_mode="capture",
    )
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    with torch.no_grad():
        with model.trace(input_ids):
            activations = model.layers_output[layer][0, absolute_positions, :].save()

    donor_activations = activations.detach().cpu().clone()
    return {
        "activations": donor_activations,
        "absolute_positions": absolute_positions,
        "input_token_count": len(input_ids_list),
    }


def _resolve_action_token(output_tokens: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    final_matches = [
        (idx, token)
        for idx, token in enumerate(output_tokens)
        if "action" in token.get("token_groups", []) and "final" in token.get("token_groups", [])
    ]
    if len(final_matches) == 1:
        return final_matches[0]
    if len(final_matches) > 1:
        raise ValueError("Found multiple final action tokens in output_tokens.")

    action_matches = [(idx, token) for idx, token in enumerate(output_tokens) if "action" in token.get("token_groups", [])]
    if not action_matches:
        raise ValueError("Could not find an action token in output_tokens.")
    if len(action_matches) == 1:
        return action_matches[0]

    raise ValueError("Found multiple action tokens in output_tokens and could not isolate a final action token.")


def _build_answer_forcing_input(
    model: StandardizedTransformer,
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    grid_size: int | None = None,
    modified_positions: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    step = trajectory["steps"][step_idx]

    prefix_ids = [token["token_id"] for token in trajectory["prompt"]["prompt_prefix_tokens"]]
    grid_ids = [token["token_id"] for token in step["grid_state_tokens"]]
    suffix_tokens = trajectory["prompt"]["prompt_suffix_tokens"]
    suffix_ids = [token["token_id"] for token in suffix_tokens]
    output_tokens = step["output_tokens"]
    output_ids = [token["token_id"] for token in output_tokens]

    prompt_offset = len(prefix_ids) + len(grid_ids)
    output_offset = prompt_offset + len(suffix_ids)
    action_output_idx, action_token = _resolve_action_token(output_tokens)

    if patch_site == PATCH_SITE_PROMPT_BOUNDARY:
        relative_positions = resolve_prompt_boundary_token_ids(suffix_tokens)
        absolute_positions = [prompt_offset + position for position in relative_positions]
    elif patch_site == PATCH_SITE_PRE_FINAL_BOUNDARY:
        relative_positions = resolve_pre_final_boundary_token_ids(output_tokens)
        absolute_positions = [output_offset + position for position in relative_positions]
    elif patch_site == PATCH_SITE_MODIFIED_GRID_CELLS:
        if grid_size is None or modified_positions is None:
            raise ValueError("modified_grid_cells requires grid_size and modified_positions.")
        relative_positions = resolve_modified_grid_cell_token_ids(
            step["grid_state_tokens"], grid_size, modified_positions
        )
        absolute_positions = [len(prefix_ids) + position for position in relative_positions]
    else:
        raise ValueError(f"Unsupported patch_site: {patch_site}")

    input_ids = prefix_ids + grid_ids + suffix_ids + output_ids[:action_output_idx]

    action_token_text = str(action_token["token"])
    candidate_texts = (
        {action: f'"{action}"' for action in VALID_ACTIONS}
        if action_token_text.startswith('"') and action_token_text.endswith('"')
        else {action: action for action in VALID_ACTIONS}
    )

    action_token_ids: dict[str, int] = {}
    for action, token_text in candidate_texts.items():
        token_ids = model.tokenizer.encode(token_text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Answer-forcing requires a single token for action {action!r}; got token ids {token_ids}."
            )
        action_token_ids[action] = int(token_ids[0])

    return {
        "input_ids": input_ids,
        "absolute_positions": absolute_positions,
        "action_token_ids": action_token_ids,
        "recorded_action_token_id": int(action_token["token_id"]),
        "recorded_action_token_text": action_token_text,
    }


def _generate_tokens_with_optional_patch(
    model: StandardizedTransformer,
    input_ids: torch.Tensor,
    absolute_positions: list[int],
    max_new_tokens: int,
    layer: int | None = None,
    donor_activations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run generation, optionally patching hidden states at the selected positions."""
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}

    with torch.no_grad():
        with model.generate(**gen_kwargs) as tracer:
            with tracer.invoke(input_ids):
                if donor_activations is not None:
                    if layer is None:
                        raise ValueError("layer must be provided when donor_activations are supplied.")
                    device = model.layers_output[layer].device
                    model.layers_output[layer][0, absolute_positions, :] = donor_activations.to(device)
                generated_ids = tracer.result.save()

    return generated_ids.detach().cpu()


def _decode_generated_ids(model: StandardizedTransformer, generated_ids: torch.Tensor) -> str:
    """Decode generated token ids into text."""
    if generated_ids.ndim == 1:
        token_ids = generated_ids.tolist()
    else:
        token_ids = generated_ids[0].tolist()
    return model.tokenizer.decode(token_ids, skip_special_tokens=False)


def extract_action_from_text(text: str) -> str | None:
    """Parse the final JSON action from model text."""
    patterns = (
        r'["\']action["\']\s*:\s*["\'](?P<action>up|down|left|right)["\']',
        r"\baction\b\s*[:=]\s*[\"']?(?P<action>up|down|left|right)[\"']?",
    )
    matches: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(match.group("action").upper())

    if not matches:
        return None

    return matches[-1]


def _run_target_generation(
    model: StandardizedTransformer,
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    layer: int | None = None,
    donor_activations: torch.Tensor | None = None,
    grid_size: int | None = None,
    modified_positions: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    """Run baseline or patched generation for a target trajectory."""
    input_ids_list, absolute_positions, remaining_tokens = _build_input_ids_for_patch_site(
        trajectory,
        step_idx,
        patch_site,
        grid_size=grid_size,
        modified_positions=modified_positions,
        target_mode="free_generation",
    )
    max_new_tokens = max(remaining_tokens + 16, 16)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    generated_ids = _generate_tokens_with_optional_patch(
        model=model,
        input_ids=input_ids,
        absolute_positions=absolute_positions,
        max_new_tokens=max_new_tokens,
        layer=layer,
        donor_activations=donor_activations,
    )
    output_text = _decode_generated_ids(model, generated_ids)
    action = extract_action_from_text(output_text)

    generated_length = int(generated_ids.shape[-1] if generated_ids.ndim > 1 else generated_ids.shape[0])
    return {
        "output_text": output_text,
        "action": action,
        "parse_failure": action is None,
        "input_token_count": len(input_ids_list),
        "generated_token_count": generated_length,
        "absolute_positions": absolute_positions,
    }


def _run_target_answer_forcing(
    model: StandardizedTransformer,
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    layer: int | None = None,
    donor_activations: torch.Tensor | None = None,
    grid_size: int | None = None,
    modified_positions: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    forced_input = _build_answer_forcing_input(
        model,
        trajectory,
        step_idx,
        patch_site,
        grid_size=grid_size,
        modified_positions=modified_positions,
    )
    input_ids_list = forced_input["input_ids"]
    absolute_positions = forced_input["absolute_positions"]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    with torch.no_grad():
        with model.trace(input_ids):
            if donor_activations is not None:
                if layer is None:
                    raise ValueError("layer must be provided when donor_activations are supplied.")
                device = model.layers_output[layer].device
                model.layers_output[layer][0, absolute_positions, :] = donor_activations.to(device)
            next_token_logits = model.logits[0, -1, :].save()

    logits = next_token_logits.detach().cpu()
    probabilities = torch.softmax(logits, dim=-1)
    action_probabilities = {
        action: float(probabilities[token_id].item()) for action, token_id in forced_input["action_token_ids"].items()
    }
    predicted_action = max(action_probabilities, key=action_probabilities.get)

    return {
        "output_text": None,
        "action": predicted_action,
        "parse_failure": False,
        "input_token_count": len(input_ids_list),
        "generated_token_count": 0,
        "absolute_positions": absolute_positions,
        "action_probabilities": action_probabilities,
        "action_token_ids": forced_input["action_token_ids"],
        "recorded_action_token_id": forced_input["recorded_action_token_id"],
        "recorded_action_token_text": forced_input["recorded_action_token_text"],
    }


def _run_target_evaluation(
    model: StandardizedTransformer,
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    evaluation_mode: str,
    layer: int | None = None,
    donor_activations: torch.Tensor | None = None,
    grid_size: int | None = None,
    modified_positions: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    if evaluation_mode == "answer_forcing":
        return _run_target_answer_forcing(
            model=model,
            trajectory=trajectory,
            step_idx=step_idx,
            patch_site=patch_site,
            layer=layer,
            donor_activations=donor_activations,
            grid_size=grid_size,
            modified_positions=modified_positions,
        )
    if evaluation_mode == "free_generation":
        return _run_target_generation(
            model=model,
            trajectory=trajectory,
            step_idx=step_idx,
            patch_site=patch_site,
            layer=layer,
            donor_activations=donor_activations,
            grid_size=grid_size,
            modified_positions=modified_positions,
        )
    raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")


def _extract_recorded_action(trajectory: dict[str, Any], step_idx: int) -> str | None:
    action = trajectory["steps"][step_idx].get("agent_action")
    if action is None:
        return None
    action = str(action).upper()
    return action if action in VALID_ACTIONS else None


def compute_outcome_flags(
    direction: str,
    relation_to_original: str,
    source_optimal_actions: tuple[str, ...],
    target_optimal_actions: tuple[str, ...],
    baseline_target_action: str | None,
    patched_target_action: str | None,
) -> dict[str, bool]:
    """Compute boolean success metrics for a single patching run."""
    patched_in_source_optimal = patched_target_action in source_optimal_actions if patched_target_action else False
    patched_in_target_optimal = patched_target_action in target_optimal_actions if patched_target_action else False

    if relation_to_original == "same":
        expected_optimal_actions = target_optimal_actions
        success_primary = (
            patched_target_action is not None
            and baseline_target_action is not None
            and patched_target_action == baseline_target_action
        )
    elif relation_to_original == "disjoint":
        expected_optimal_actions = source_optimal_actions
        success_primary = patched_in_source_optimal
    else:
        expected_optimal_actions = target_optimal_actions
        success_primary = patched_in_target_optimal

    return {
        "patched_in_source_optimal_set": patched_in_source_optimal,
        "patched_in_target_optimal_set": patched_in_target_optimal,
        "patched_in_expected_optimal_set": (
            patched_target_action in expected_optimal_actions if patched_target_action else False
        ),
        "action_changed_vs_baseline": (
            patched_target_action is not None
            and baseline_target_action is not None
            and patched_target_action != baseline_target_action
        ),
        "success_primary": success_primary,
        "expected_source_dominance": direction in VALID_DIRECTIONS and relation_to_original == "disjoint",
    }


def _load_counterfactual_metadata(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    entries: dict[str, dict[str, Any]] = {}
    for key in ("agent_moved_counterfactuals", "goal_moved_counterfactuals"):
        for item in data.get(key, []):
            entries[item["counterfactual_id"]] = item
    data["_counterfactual_entries"] = entries
    return data


def discover_patch_pairs(
    counterfactual_metadata_root: str,
    counterfactual_trajectories_dir: str,
    original_trajectories_dir: str,
    counterfactual_types: str = "all",
    relations: str = "all",
    grid_sizes: str = "all",
    complexities: str = "all",
    max_examples: int | None = None,
) -> tuple[list[PatchPair], list[dict[str, Any]]]:
    """Discover valid original/counterfactual patch pairs from metadata and trajectory summary."""
    selected_counterfactual_types = set(
        _parse_string_spec(counterfactual_types, VALID_COUNTERFACTUAL_TYPES, "counterfactual_type")
    )
    selected_relations = set(_parse_string_spec(relations, VALID_RELATIONS, "relation"))
    selected_sizes = _parse_int_spec(grid_sizes)
    selected_complexities = _parse_float_spec(complexities)

    summary_path = Path(counterfactual_trajectories_dir) / "counterfactual_trajectory_batch_summary.json"
    summary = _load_json(summary_path)

    metadata_cache: dict[Path, dict[str, Any]] = {}
    pairs: list[PatchPair] = []
    skipped: list[dict[str, Any]] = []

    for success in summary.get("successes", []):
        metadata_path = Path(success["source_counterfactual_path"])
        if metadata_path not in metadata_cache:
            metadata_cache[metadata_path] = _load_counterfactual_metadata(metadata_path)
        metadata = metadata_cache[metadata_path]

        counterfactual_id = success["counterfactual_id"]
        entry = metadata["_counterfactual_entries"].get(counterfactual_id)
        if entry is None:
            skipped.append(
                {
                    "metadata_path": str(metadata_path),
                    "counterfactual_id": counterfactual_id,
                    "reason": "counterfactual_id_not_found_in_metadata",
                }
            )
            continue

        relation = str(success["relation_to_original"])
        counterfactual_type = str(success["counterfactual_type"])
        grid_size = int(success["grid_size"])
        grid_complexity = float(success["grid_complexity"])

        if counterfactual_type not in selected_counterfactual_types:
            continue
        if relation not in selected_relations:
            continue
        if selected_sizes is not None and grid_size not in selected_sizes:
            continue
        if selected_complexities is not None and grid_complexity not in selected_complexities:
            continue

        original_path = _resolve_repo_relative_path(
            metadata["source_relative_path"],
            [
                Path(counterfactual_metadata_root).resolve().parent.parent,
                Path(original_trajectories_dir).resolve().parent.parent,
            ],
        )
        counterfactual_path = _resolve_repo_relative_path(
            success["output_path"],
            [
                Path(counterfactual_trajectories_dir).resolve().parent.parent,
                Path(counterfactual_metadata_root).resolve().parent.parent,
            ],
        )

        if not original_path.exists():
            skipped.append(
                {
                    "counterfactual_id": counterfactual_id,
                    "reason": "missing_original_trajectory",
                    "path": str(original_path),
                }
            )
            continue

        if not counterfactual_path.exists():
            skipped.append(
                {
                    "counterfactual_id": counterfactual_id,
                    "reason": "missing_counterfactual_trajectory",
                    "path": str(counterfactual_path),
                }
            )
            continue

        pair = PatchPair(
            metadata_path=str(metadata_path),
            counterfactual_id=counterfactual_id,
            counterfactual_type=counterfactual_type,
            relation_to_original=relation,
            grid_size=grid_size,
            grid_complexity=grid_complexity,
            source_step_id=int(metadata.get("source_step_id", 0)),
            source_filename=str(success.get("source_filename", metadata.get("source_filename", ""))),
            original_trajectory_path=str(original_path),
            counterfactual_trajectory_path=str(counterfactual_path),
            original_optimal_actions=_normalize_action_set(metadata.get("original_optimal_action_set_names")),
            counterfactual_optimal_actions=_normalize_action_set(entry.get("optimal_action_set_names")),
            moved_entity=entry.get("moved_entity"),
            from_position=_normalize_position(entry.get("from_position")),
            to_position=_normalize_position(entry.get("to_position")),
            requested_instance_id=metadata.get("requested_instance_id"),
            actual_instance_id=metadata.get("actual_instance_id"),
        )
        pairs.append(pair)

    if max_examples is not None:
        pairs = pairs[:max_examples]

    return pairs, skipped


def _get_git_sha() -> str | None:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output or None


def _build_summary_dataframe(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not run_records:
        return pd.DataFrame(
            columns=[
                "direction",
                "layer",
                "patch_site",
                "relation_to_original",
                "counterfactual_type",
                "grid_size",
                "grid_complexity",
                "runs",
                "executed_runs",
                "skipped_runs",
                "successes_primary",
                "success_rate_primary",
                "action_changed_rate",
                "patched_in_source_optimal_rate",
                "patched_in_target_optimal_rate",
                "patched_in_expected_optimal_rate",
                "parse_failure_rate",
            ]
        )

    df = pd.DataFrame(run_records)
    df["skipped"] = df["skipped"].fillna(False)
    group_cols = [
        "direction",
        "layer",
        "patch_site",
        "relation_to_original",
        "counterfactual_type",
        "grid_size",
        "grid_complexity",
    ]

    base_summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("counterfactual_id", "count"),
            skipped_runs=("skipped", "sum"),
        )
        .reset_index()
    )
    base_summary["executed_runs"] = base_summary["runs"] - base_summary["skipped_runs"]

    executed_df = df[df["skipped"] != True]  # noqa: E712
    if executed_df.empty:
        summary = base_summary.copy()
        summary["successes_primary"] = 0
        summary["success_rate_primary"] = None
        summary["action_changed_rate"] = None
        summary["patched_in_source_optimal_rate"] = None
        summary["patched_in_target_optimal_rate"] = None
        summary["patched_in_expected_optimal_rate"] = None
        summary["parse_failure_rate"] = None
        return summary

    executed_summary = (
        executed_df.groupby(group_cols, dropna=False)
        .agg(
            successes_primary=("success_primary", "sum"),
            success_rate_primary=("success_primary", "mean"),
            action_changed_rate=("action_changed_vs_baseline", "mean"),
            patched_in_source_optimal_rate=("patched_in_source_optimal_set", "mean"),
            patched_in_target_optimal_rate=("patched_in_target_optimal_set", "mean"),
            patched_in_expected_optimal_rate=("patched_in_expected_optimal_set", "mean"),
            parse_failure_rate=("parse_failure", "mean"),
        )
        .reset_index()
    )

    summary = base_summary.merge(executed_summary, on=group_cols, how="left")
    summary["successes_primary"] = summary["successes_primary"].fillna(0).astype(int)

    return summary


def _mean_numeric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_timing_summary(run_records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    return {
        "avg_run_wall_time_seconds": _mean_numeric(run_records, "run_wall_time_seconds"),
        "avg_baseline_generation_seconds": _mean_numeric(run_records, "baseline_generation_seconds"),
        "avg_donor_capture_seconds": _mean_numeric(run_records, "donor_capture_seconds"),
        "avg_patched_generation_seconds": _mean_numeric(run_records, "patched_generation_seconds"),
        "baseline_cache_miss_count": int(
            sum(1 for record in run_records if (record.get("baseline_generation_seconds") or 0.0) > 0)
        ),
        "donor_cache_miss_count": int(
            sum(1 for record in run_records if (record.get("donor_capture_seconds") or 0.0) > 0)
        ),
    }


def _format_tqdm_postfix(run_records: list[dict[str, Any]], expected_total_runs: int) -> dict[str, str]:
    timing = _build_timing_summary(run_records)
    completed_runs = len(run_records)
    error_count = int(sum(record["error"] is not None for record in run_records))

    def _fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.1f}s"

    return {
        "runs": f"{completed_runs}/{expected_total_runs}",
        "run": _fmt(timing["avg_run_wall_time_seconds"]),
        "base": _fmt(timing["avg_baseline_generation_seconds"]),
        "donor": _fmt(timing["avg_donor_capture_seconds"]),
        "patch": _fmt(timing["avg_patched_generation_seconds"]),
        "err": str(error_count),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _make_run_key(
    pair: PatchPair,
    direction: str,
    patch_site: str,
    layer: int,
) -> str:
    return "|".join(
        [
            pair.counterfactual_id,
            str(pair.source_step_id),
            direction,
            patch_site,
            str(layer),
        ]
    )


def _load_existing_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    runs: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            runs.append(json.loads(stripped))
    return runs


def _failed_generation_result(
    *,
    absolute_positions: list[int] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "output_text": "",
        "action": None,
        "parse_failure": True,
        "input_token_count": None,
        "generated_token_count": None,
        "absolute_positions": absolute_positions,
        "error": error,
    }


def _get_modified_positions_for_pair(pair: PatchPair, patch_site: str) -> tuple[tuple[int, int], ...] | None:
    if patch_site != PATCH_SITE_MODIFIED_GRID_CELLS:
        return None
    if pair.from_position is None or pair.to_position is None:
        raise ValueError(f"modified_grid_cells requires from/to positions in metadata for {pair.counterfactual_id}.")
    return (pair.from_position, pair.to_position)


def _build_manifest(
    *,
    model_id: str,
    counterfactual_metadata_root: str,
    counterfactual_trajectories_dir: str,
    original_trajectories_dir: str,
    selected_patch_sites: list[str],
    layer_indices: list[int],
    selected_directions: list[str],
    counterfactual_types: str,
    relations: str,
    grid_sizes: str,
    complexities: str,
    max_examples: int | None,
    output_dir: str,
    device_map: str,
    torch_dtype: str,
    evaluation_mode: str,
    require_recorded_actions_in_optimal_set: bool,
    overwrite: bool,
    verbose: bool,
    pairs: list[PatchPair],
    discovery_skips: list[dict[str, Any]],
    manifest_path: Path,
    runs_path: Path,
    summary_path: Path,
    failures_path: Path,
    runs: list[dict[str, Any]],
    expected_total_runs: int,
    resumed_run_count: int,
) -> dict[str, Any]:
    return {
        "command": "activation_patch_counterfactuals",
        "model_id": model_id,
        "git_sha": _get_git_sha(),
        "args": {
            "counterfactual_metadata_root": counterfactual_metadata_root,
            "counterfactual_trajectories_dir": counterfactual_trajectories_dir,
            "original_trajectories_dir": original_trajectories_dir,
            "patch_sites": selected_patch_sites,
            "layers": layer_indices,
            "directions": selected_directions,
            "counterfactual_types": counterfactual_types,
            "relations": relations,
            "grid_sizes": grid_sizes,
            "complexities": complexities,
            "max_examples": max_examples,
            "output_dir": output_dir,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "evaluation_mode": evaluation_mode,
            "require_recorded_actions_in_optimal_set": require_recorded_actions_in_optimal_set,
            "overwrite": overwrite,
            "verbose": verbose,
        },
        "discovery": {
            "pair_count": len(pairs),
            "skipped_pairs": discovery_skips,
        },
        "artifacts": {
            "manifest_path": str(manifest_path),
            "runs_path": str(runs_path),
            "summary_path": str(summary_path),
            "parse_failures_path": str(failures_path),
        },
        "progress": {
            "expected_total_runs": expected_total_runs,
            "completed_run_count": len(runs),
            "remaining_run_count": max(expected_total_runs - len(runs), 0),
            "resumed_run_count": resumed_run_count,
        },
        "run_counts": {
            "total_runs": len(runs),
            "skipped_count": int(sum(bool(record.get("skipped")) for record in runs)),
            "success_primary_count": int(sum(record["success_primary"] for record in runs)),
            "parse_failure_count": int(sum(record["parse_failure"] for record in runs)),
            "error_count": int(sum(record["error"] is not None for record in runs)),
        },
        "timing": _build_timing_summary(runs),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


def activation_patch_counterfactuals(  # noqa: PLR0912
    counterfactual_metadata_root: str = "data/counterfactual_grids",
    counterfactual_trajectories_dir: str = "data/counterfactual_trajectories",
    original_trajectories_dir: str = "data/trajectories_test_full",
    patch_sites: str = "prompt_boundary,pre_final_boundary",
    layers: str = "all",
    directions: str = "original_to_counterfactual,counterfactual_to_original",
    counterfactual_types: str = "all",
    relations: str = "all",
    grid_sizes: str = "all",
    complexities: str = "all",
    max_examples: int | None = None,
    output_dir: str = "data/activation_patching",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    evaluation_mode: str = "answer_forcing",
    require_recorded_actions_in_optimal_set: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run online activation patching between originals and counterfactuals."""
    selected_patch_sites = _parse_string_spec(patch_sites, VALID_PATCH_SITES, "patch_site")
    selected_directions = _parse_string_spec(directions, VALID_DIRECTIONS, "direction")
    if evaluation_mode not in VALID_EVALUATION_MODES:
        raise ValueError(
            f"Invalid evaluation_mode: {evaluation_mode}. Expected one of {sorted(VALID_EVALUATION_MODES)}."
        )

    pairs, discovery_skips = discover_patch_pairs(
        counterfactual_metadata_root=counterfactual_metadata_root,
        counterfactual_trajectories_dir=counterfactual_trajectories_dir,
        original_trajectories_dir=original_trajectories_dir,
        counterfactual_types=counterfactual_types,
        relations=relations,
        grid_sizes=grid_sizes,
        complexities=complexities,
        max_examples=max_examples,
    )

    if not pairs:
        raise ValueError("No patch pairs were discovered for the selected filters.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir_path / "manifest.json"
    runs_path = output_dir_path / "runs.jsonl"
    summary_path = output_dir_path / "summary_by_group.csv"
    failures_path = output_dir_path / "parse_failures.jsonl"
    if overwrite:
        for path in (manifest_path, runs_path, summary_path, failures_path):
            if path.exists():
                path.unlink()
        existing_runs: list[dict[str, Any]] = []
    elif runs_path.exists():
        existing_runs = _load_existing_runs(runs_path)
    else:
        existing_runs = []
        if manifest_path.exists() or summary_path.exists() or failures_path.exists():
            raise FileExistsError(
                f"Found partial outputs in {output_dir_path} without runs.jsonl. Pass overwrite=True to replace them."
            )

    first_original = _load_json(Path(pairs[0].original_trajectory_path))
    model_id = first_original["model_params"]["model_id"]
    resolved_dtype = _resolve_torch_dtype(torch_dtype, model_id)
    model = StandardizedTransformer(model_id, device_map=device_map, torch_dtype=resolved_dtype)

    try:
        layer_indices = parse_index_specification(layers, model.config.num_hidden_layers)
        expected_total_runs = len(pairs) * len(selected_directions) * len(selected_patch_sites) * len(layer_indices)
        completed_run_keys = {
            record.get("run_key")
            or "|".join(
                [
                    str(record["counterfactual_id"]),
                    str(record["step_idx"]),
                    str(record["direction"]),
                    str(record["patch_site"]),
                    str(record["layer"]),
                ]
            )
            for record in existing_runs
        }
        if verbose:
            print(f"Discovered {len(pairs)} patch pairs")
            print(f"Selected layers: {layer_indices}")
            print(f"Selected patch sites: {selected_patch_sites}")
            if existing_runs:
                print(f"Resuming from {len(existing_runs)} existing runs")

        runs: list[dict[str, Any]] = list(existing_runs)
        donor_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        baseline_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

        manifest = _build_manifest(
            model_id=model_id,
            counterfactual_metadata_root=counterfactual_metadata_root,
            counterfactual_trajectories_dir=counterfactual_trajectories_dir,
            original_trajectories_dir=original_trajectories_dir,
            selected_patch_sites=selected_patch_sites,
            layer_indices=layer_indices,
            selected_directions=selected_directions,
            counterfactual_types=counterfactual_types,
            relations=relations,
            grid_sizes=grid_sizes,
            complexities=complexities,
            max_examples=max_examples,
            output_dir=output_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            evaluation_mode=evaluation_mode,
            require_recorded_actions_in_optimal_set=require_recorded_actions_in_optimal_set,
            overwrite=overwrite,
            verbose=verbose,
            pairs=pairs,
            discovery_skips=discovery_skips,
            manifest_path=manifest_path,
            runs_path=runs_path,
            summary_path=summary_path,
            failures_path=failures_path,
            runs=runs,
            expected_total_runs=expected_total_runs,
            resumed_run_count=len(existing_runs),
        )
        _write_manifest(manifest_path, manifest)

        pair_iterator = tqdm(pairs, desc="Patch pairs") if verbose else pairs
        for pair in pair_iterator:
            original_trajectory = _load_json(Path(pair.original_trajectory_path))
            counterfactual_trajectory = _load_json(Path(pair.counterfactual_trajectory_path))

            if original_trajectory["model_params"]["model_id"] != model_id:
                raise ValueError(
                    "All trajectories must use the same model_id. "
                    f"Expected {model_id}, got {original_trajectory['model_params']['model_id']}."
                )
            if counterfactual_trajectory["model_params"]["model_id"] != model_id:
                raise ValueError(
                    "All trajectories must use the same model_id. "
                    f"Expected {model_id}, got {counterfactual_trajectory['model_params']['model_id']}."
                )

            trajectory_by_label = {
                "original": original_trajectory,
                "counterfactual": counterfactual_trajectory,
            }

            for direction in selected_directions:
                source_label = "original" if direction == "original_to_counterfactual" else "counterfactual"
                target_label = "counterfactual" if source_label == "original" else "original"

                source_trajectory = trajectory_by_label[source_label]
                target_trajectory = trajectory_by_label[target_label]
                source_path = (
                    pair.original_trajectory_path
                    if source_label == "original"
                    else pair.counterfactual_trajectory_path
                )
                target_path = (
                    pair.counterfactual_trajectory_path
                    if target_label == "counterfactual"
                    else pair.original_trajectory_path
                )

                source_optimal_actions = (
                    pair.original_optimal_actions
                    if source_label == "original"
                    else pair.counterfactual_optimal_actions
                )
                target_optimal_actions = (
                    pair.counterfactual_optimal_actions
                    if target_label == "counterfactual"
                    else pair.original_optimal_actions
                )

                source_recorded_action = _extract_recorded_action(source_trajectory, pair.source_step_id)
                target_recorded_action = _extract_recorded_action(target_trajectory, pair.source_step_id)
                source_recorded_action_in_optimal_set = (
                    source_recorded_action in source_optimal_actions if source_recorded_action is not None else False
                )
                target_recorded_action_in_optimal_set = (
                    target_recorded_action in target_optimal_actions if target_recorded_action is not None else False
                )

                if require_recorded_actions_in_optimal_set and (
                    not source_recorded_action_in_optimal_set or not target_recorded_action_in_optimal_set
                ):
                    skip_reason = "recorded_action_not_in_optimal_set"
                    for patch_site in selected_patch_sites:
                        for layer in layer_indices:
                            run_key = _make_run_key(pair, direction, patch_site, layer)
                            if run_key in completed_run_keys:
                                continue
                            run_record = {
                                "metadata_path": pair.metadata_path,
                                "counterfactual_id": pair.counterfactual_id,
                                "counterfactual_type": pair.counterfactual_type,
                                "relation_to_original": pair.relation_to_original,
                                "grid_size": pair.grid_size,
                                "grid_complexity": pair.grid_complexity,
                                "requested_instance_id": pair.requested_instance_id,
                                "actual_instance_id": pair.actual_instance_id,
                                "source_filename": pair.source_filename,
                                "run_key": run_key,
                                "direction": direction,
                                "source_label": source_label,
                                "target_label": target_label,
                                "patch_site": patch_site,
                                "layer": layer,
                                "step_idx": pair.source_step_id,
                                "source_trajectory_path": source_path,
                                "target_trajectory_path": target_path,
                                "source_recorded_action": source_recorded_action,
                                "target_recorded_action": target_recorded_action,
                                "source_recorded_action_in_optimal_set": source_recorded_action_in_optimal_set,
                                "target_recorded_action_in_optimal_set": target_recorded_action_in_optimal_set,
                                "baseline_target_action": None,
                                "patched_target_action": None,
                                "source_optimal_actions": list(source_optimal_actions),
                                "target_optimal_actions": list(target_optimal_actions),
                                "baseline_parse_failure": False,
                                "parse_failure": False,
                                "baseline_error": None,
                                "donor_error": None,
                                "patched_error": None,
                                "error": None,
                                "run_wall_time_seconds": 0.0,
                                "baseline_generation_seconds": 0.0,
                                "donor_capture_seconds": 0.0,
                                "patched_generation_seconds": 0.0,
                                "baseline_input_token_count": None,
                                "patched_input_token_count": None,
                                "baseline_generated_token_count": None,
                                "patched_generated_token_count": None,
                                "patched_boundary_positions": None,
                                "donor_boundary_positions": None,
                                "donor_input_token_count": None,
                                "patched_in_source_optimal_set": None,
                                "patched_in_target_optimal_set": None,
                                "patched_in_expected_optimal_set": None,
                                "action_changed_vs_baseline": None,
                                "success_primary": False,
                                "expected_source_dominance": direction in VALID_DIRECTIONS
                                and pair.relation_to_original == "disjoint",
                                "skipped": True,
                                "skip_reason": skip_reason,
                            }
                            runs.append(run_record)
                            completed_run_keys.add(run_key)
                            _append_jsonl_record(runs_path, run_record)
                            manifest = _build_manifest(
                                model_id=model_id,
                                counterfactual_metadata_root=counterfactual_metadata_root,
                                counterfactual_trajectories_dir=counterfactual_trajectories_dir,
                                original_trajectories_dir=original_trajectories_dir,
                                selected_patch_sites=selected_patch_sites,
                                layer_indices=layer_indices,
                                selected_directions=selected_directions,
                                counterfactual_types=counterfactual_types,
                                relations=relations,
                                grid_sizes=grid_sizes,
                                complexities=complexities,
                                max_examples=max_examples,
                                output_dir=output_dir,
                                device_map=device_map,
                                torch_dtype=torch_dtype,
                                evaluation_mode=evaluation_mode,
                                require_recorded_actions_in_optimal_set=require_recorded_actions_in_optimal_set,
                                overwrite=overwrite,
                                verbose=verbose,
                                pairs=pairs,
                                discovery_skips=discovery_skips,
                                manifest_path=manifest_path,
                                runs_path=runs_path,
                                summary_path=summary_path,
                                failures_path=failures_path,
                                runs=runs,
                                expected_total_runs=expected_total_runs,
                                resumed_run_count=len(existing_runs),
                            )
                            _write_manifest(manifest_path, manifest)
                    continue

                for patch_site in selected_patch_sites:
                    modified_positions = _get_modified_positions_for_pair(pair, patch_site)
                    baseline_key = (target_path, patch_site, pair.source_step_id)
                    baseline_cache_missed = baseline_key not in baseline_cache
                    if baseline_key not in baseline_cache:
                        try:
                            baseline_started_at = perf_counter()
                            baseline_cache[baseline_key] = _run_target_evaluation(
                                model=model,
                                trajectory=target_trajectory,
                                step_idx=pair.source_step_id,
                                patch_site=patch_site,
                                evaluation_mode=evaluation_mode,
                                grid_size=pair.grid_size,
                                modified_positions=modified_positions,
                            ) | {
                                "error": None,
                                "baseline_generation_seconds": perf_counter() - baseline_started_at,
                            }
                        except Exception as exc:  # noqa: BLE001
                            baseline_cache[baseline_key] = _failed_generation_result(error=str(exc)) | {
                                "baseline_generation_seconds": perf_counter() - baseline_started_at,
                            }

                    baseline_result = baseline_cache[baseline_key]
                    baseline_generation_seconds = (
                        float(baseline_result.get("baseline_generation_seconds") or 0.0)
                        if baseline_cache_missed
                        else 0.0
                    )

                    for layer in layer_indices:
                        run_started_at = perf_counter()
                        run_key = _make_run_key(pair, direction, patch_site, layer)
                        if run_key in completed_run_keys:
                            continue

                        donor_key = (source_path, patch_site, pair.source_step_id, layer)
                        error_messages: list[str] = []
                        donor_capture_seconds = 0.0
                        patched_generation_seconds = 0.0

                        if baseline_result["error"] is not None:
                            donor_result = {
                                "absolute_positions": None,
                                "input_token_count": None,
                                "error": None,
                                "donor_capture_seconds": 0.0,
                            }
                            patched_result = _failed_generation_result(error=baseline_result["error"]) | {
                                "patched_generation_seconds": 0.0,
                            }
                            error_messages.append(f"baseline_generation_failed: {baseline_result['error']}")
                        else:
                            donor_cache_missed = donor_key not in donor_cache
                            if donor_key not in donor_cache:
                                try:
                                    donor_started_at = perf_counter()
                                    donor_cache[donor_key] = _capture_donor_activations(
                                        model=model,
                                        trajectory=source_trajectory,
                                        step_idx=pair.source_step_id,
                                        patch_site=patch_site,
                                        layer=layer,
                                        grid_size=pair.grid_size,
                                        modified_positions=modified_positions,
                                    ) | {
                                        "error": None,
                                        "donor_capture_seconds": perf_counter() - donor_started_at,
                                    }
                                except Exception as exc:  # noqa: BLE001
                                    donor_cache[donor_key] = {
                                        "activations": None,
                                        "absolute_positions": None,
                                        "input_token_count": None,
                                        "error": str(exc),
                                        "donor_capture_seconds": perf_counter() - donor_started_at,
                                    }

                            donor_result = donor_cache[donor_key]
                            donor_capture_seconds = (
                                float(donor_result.get("donor_capture_seconds") or 0.0) if donor_cache_missed else 0.0
                            )
                            if donor_result["error"] is not None:
                                patched_result = _failed_generation_result(error=donor_result["error"]) | {
                                    "patched_generation_seconds": 0.0,
                                }
                                error_messages.append(f"donor_capture_failed: {donor_result['error']}")
                            else:
                                try:
                                    patched_started_at = perf_counter()
                                    patched_result = _run_target_evaluation(
                                        model=model,
                                        trajectory=target_trajectory,
                                        step_idx=pair.source_step_id,
                                        patch_site=patch_site,
                                        evaluation_mode=evaluation_mode,
                                        layer=layer,
                                        donor_activations=donor_result["activations"],
                                        grid_size=pair.grid_size,
                                        modified_positions=modified_positions,
                                    ) | {
                                        "error": None,
                                        "patched_generation_seconds": perf_counter() - patched_started_at,
                                    }
                                except Exception as exc:  # noqa: BLE001
                                    patched_result = _failed_generation_result(
                                        absolute_positions=donor_result["absolute_positions"],
                                        error=str(exc),
                                    ) | {"patched_generation_seconds": perf_counter() - patched_started_at}
                                    error_messages.append(f"patched_generation_failed: {exc}")
                                patched_generation_seconds = float(
                                    patched_result.get("patched_generation_seconds") or 0.0
                                )

                        outcome_flags = compute_outcome_flags(
                            direction=direction,
                            relation_to_original=pair.relation_to_original,
                            source_optimal_actions=source_optimal_actions,
                            target_optimal_actions=target_optimal_actions,
                            baseline_target_action=baseline_result["action"],
                            patched_target_action=patched_result["action"],
                        )

                        run_record = {
                            "metadata_path": pair.metadata_path,
                            "counterfactual_id": pair.counterfactual_id,
                            "counterfactual_type": pair.counterfactual_type,
                            "relation_to_original": pair.relation_to_original,
                            "grid_size": pair.grid_size,
                            "grid_complexity": pair.grid_complexity,
                            "requested_instance_id": pair.requested_instance_id,
                            "actual_instance_id": pair.actual_instance_id,
                            "source_filename": pair.source_filename,
                            "run_key": run_key,
                            "direction": direction,
                            "source_label": source_label,
                            "target_label": target_label,
                            "patch_site": patch_site,
                            "layer": layer,
                            "step_idx": pair.source_step_id,
                            "source_trajectory_path": source_path,
                            "target_trajectory_path": target_path,
                            "source_recorded_action": source_recorded_action,
                            "target_recorded_action": target_recorded_action,
                            "source_recorded_action_in_optimal_set": source_recorded_action_in_optimal_set,
                            "target_recorded_action_in_optimal_set": target_recorded_action_in_optimal_set,
                            "baseline_target_action": baseline_result["action"],
                            "patched_target_action": patched_result["action"],
                            "baseline_action_probabilities": baseline_result.get("action_probabilities"),
                            "patched_action_probabilities": patched_result.get("action_probabilities"),
                            "action_token_ids": baseline_result.get("action_token_ids")
                            or patched_result.get("action_token_ids"),
                            "recorded_action_token_id": (
                                baseline_result.get("recorded_action_token_id")
                                or patched_result.get("recorded_action_token_id")
                            ),
                            "recorded_action_token_text": (
                                baseline_result.get("recorded_action_token_text")
                                or patched_result.get("recorded_action_token_text")
                            ),
                            "source_optimal_actions": list(source_optimal_actions),
                            "target_optimal_actions": list(target_optimal_actions),
                            "baseline_parse_failure": bool(baseline_result["parse_failure"]),
                            "parse_failure": bool(patched_result["parse_failure"]),
                            "baseline_error": baseline_result["error"],
                            "donor_error": donor_result.get("error"),
                            "patched_error": patched_result["error"],
                            "error": "; ".join(error_messages) or None,
                            "run_wall_time_seconds": perf_counter() - run_started_at,
                            "baseline_generation_seconds": baseline_generation_seconds,
                            "donor_capture_seconds": donor_capture_seconds,
                            "patched_generation_seconds": patched_generation_seconds,
                            "baseline_input_token_count": baseline_result["input_token_count"],
                            "patched_input_token_count": patched_result["input_token_count"],
                            "baseline_generated_token_count": baseline_result["generated_token_count"],
                            "patched_generated_token_count": patched_result["generated_token_count"],
                            "patched_boundary_positions": patched_result["absolute_positions"],
                            "donor_boundary_positions": donor_result["absolute_positions"],
                            "donor_input_token_count": donor_result["input_token_count"],
                            "skipped": False,
                            "skip_reason": None,
                            **outcome_flags,
                        }
                        runs.append(run_record)
                        completed_run_keys.add(run_key)
                        _append_jsonl_record(runs_path, run_record)
                        if baseline_result["parse_failure"] or patched_result["parse_failure"]:
                            _append_jsonl_record(
                                failures_path,
                                {
                                    "run_key": run_key,
                                    "counterfactual_id": pair.counterfactual_id,
                                    "direction": direction,
                                    "patch_site": patch_site,
                                    "layer": layer,
                                    "step_idx": pair.source_step_id,
                                    "baseline_parse_failure": bool(baseline_result["parse_failure"]),
                                    "parse_failure": bool(patched_result["parse_failure"]),
                                    "baseline_error": baseline_result["error"],
                                    "patched_error": patched_result["error"],
                                    "baseline_output_text": (
                                        baseline_result["output_text"] if baseline_result["parse_failure"] else None
                                    ),
                                    "patched_output_text": (
                                        patched_result["output_text"] if patched_result["parse_failure"] else None
                                    ),
                                },
                            )
                        if verbose and hasattr(pair_iterator, "set_postfix") and len(runs) % 5 == 0:
                            pair_iterator.set_postfix(_format_tqdm_postfix(runs, expected_total_runs))
                        manifest = _build_manifest(
                            model_id=model_id,
                            counterfactual_metadata_root=counterfactual_metadata_root,
                            counterfactual_trajectories_dir=counterfactual_trajectories_dir,
                            original_trajectories_dir=original_trajectories_dir,
                            selected_patch_sites=selected_patch_sites,
                            layer_indices=layer_indices,
                            selected_directions=selected_directions,
                            counterfactual_types=counterfactual_types,
                            relations=relations,
                            grid_sizes=grid_sizes,
                            complexities=complexities,
                            max_examples=max_examples,
                            output_dir=output_dir,
                            device_map=device_map,
                            torch_dtype=torch_dtype,
                            evaluation_mode=evaluation_mode,
                            require_recorded_actions_in_optimal_set=require_recorded_actions_in_optimal_set,
                            overwrite=overwrite,
                            verbose=verbose,
                            pairs=pairs,
                            discovery_skips=discovery_skips,
                            manifest_path=manifest_path,
                            runs_path=runs_path,
                            summary_path=summary_path,
                            failures_path=failures_path,
                            runs=runs,
                            expected_total_runs=expected_total_runs,
                            resumed_run_count=len(existing_runs),
                        )
                        _write_manifest(manifest_path, manifest)

            if verbose and hasattr(pair_iterator, "set_postfix"):
                pair_iterator.set_postfix(_format_tqdm_postfix(runs, expected_total_runs))

        summary_df = _build_summary_dataframe(runs)
        summary_df.to_csv(summary_path, index=False)
        manifest = _build_manifest(
            model_id=model_id,
            counterfactual_metadata_root=counterfactual_metadata_root,
            counterfactual_trajectories_dir=counterfactual_trajectories_dir,
            original_trajectories_dir=original_trajectories_dir,
            selected_patch_sites=selected_patch_sites,
            layer_indices=layer_indices,
            selected_directions=selected_directions,
            counterfactual_types=counterfactual_types,
            relations=relations,
            grid_sizes=grid_sizes,
            complexities=complexities,
            max_examples=max_examples,
            output_dir=output_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            evaluation_mode=evaluation_mode,
            require_recorded_actions_in_optimal_set=require_recorded_actions_in_optimal_set,
            overwrite=overwrite,
            verbose=verbose,
            pairs=pairs,
            discovery_skips=discovery_skips,
            manifest_path=manifest_path,
            runs_path=runs_path,
            summary_path=summary_path,
            failures_path=failures_path,
            runs=runs,
            expected_total_runs=expected_total_runs,
            resumed_run_count=len(existing_runs),
        )
        _write_manifest(manifest_path, manifest)

        if verbose:
            print(f"Manifest saved to {manifest_path}")
            print(f"Run records saved to {runs_path}")
            print(f"Summary saved to {summary_path}")

        return manifest
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
