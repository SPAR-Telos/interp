"""Activation patching between original and counterfactual trajectories."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
EXPECTED_BOUNDARY_TOKENS = ("<|end|>", "<|start|>", "assistant")
EXPECTED_FINAL_PREFIX = ("<|channel|>", "final", "<|message|>")
VALID_PATCH_SITES = (PATCH_SITE_PROMPT_BOUNDARY, PATCH_SITE_PRE_FINAL_BOUNDARY)
VALID_DIRECTIONS = ("original_to_counterfactual", "counterfactual_to_original")
VALID_RELATIONS = ("same", "disjoint", "overlap")
VALID_COUNTERFACTUAL_TYPES = ("agent_moved", "goal_moved")


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
            "prompt_boundary requires exactly 3 prompt_suffix tokens, "
            f"found {len(prompt_suffix_tokens)}."
        )

    token_text = tuple(token["token"] for token in prompt_suffix_tokens)
    if token_text != EXPECTED_BOUNDARY_TOKENS:
        raise ValueError(
            "prompt_boundary tokens do not match the expected template boundary. "
            f"Expected {EXPECTED_BOUNDARY_TOKENS}, got {token_text}."
        )

    token_ids = [int(token["id"]) for token in prompt_suffix_tokens]
    if token_ids != [0, 1, 2]:
        raise ValueError(
            "prompt_boundary token ids must be [0, 1, 2]. "
            f"Got {token_ids}."
        )

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


def _build_input_ids_for_patch_site(
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
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

    raise ValueError(f"Unsupported patch_site: {patch_site}")


def _capture_donor_activations(
    model: StandardizedTransformer,
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    layer: int,
) -> dict[str, Any]:
    """Capture donor activations at the selected boundary tokens."""
    input_ids_list, absolute_positions, _ = _build_input_ids_for_patch_site(trajectory, step_idx, patch_site)
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
                generated_ids = model.generator.output.save()

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
    match = re.search(r'"action"\s*:\s*"(?P<action>UP|DOWN|LEFT|RIGHT)"', text)
    if match is None:
        return None
    return match.group("action")


def _run_target_generation(
    model: StandardizedTransformer,
    trajectory: dict[str, Any],
    step_idx: int,
    patch_site: str,
    layer: int | None = None,
    donor_activations: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run baseline or patched generation for a target trajectory."""
    input_ids_list, absolute_positions, remaining_tokens = _build_input_ids_for_patch_site(
        trajectory, step_idx, patch_site
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
    group_cols = [
        "direction",
        "layer",
        "patch_site",
        "relation_to_original",
        "counterfactual_type",
        "grid_size",
        "grid_complexity",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("counterfactual_id", "count"),
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

    return summary


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def activation_patch_counterfactuals(
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
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run online activation patching between originals and counterfactuals."""
    selected_patch_sites = _parse_string_spec(patch_sites, VALID_PATCH_SITES, "patch_site")
    selected_directions = _parse_string_spec(directions, VALID_DIRECTIONS, "direction")

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
    if not overwrite and (manifest_path.exists() or runs_path.exists() or summary_path.exists()):
        raise FileExistsError(f"Output files already exist in {output_dir_path}. Pass overwrite=True to replace them.")

    first_original = _load_json(Path(pairs[0].original_trajectory_path))
    model_id = first_original["model_params"]["model_id"]
    resolved_dtype = _resolve_torch_dtype(torch_dtype, model_id)
    model = StandardizedTransformer(model_id, device_map=device_map, torch_dtype=resolved_dtype)

    try:
        layer_indices = parse_index_specification(layers, model.config.num_hidden_layers)
        if verbose:
            print(f"Discovered {len(pairs)} patch pairs")
            print(f"Selected layers: {layer_indices}")
            print(f"Selected patch sites: {selected_patch_sites}")

        runs: list[dict[str, Any]] = []
        donor_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        baseline_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

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

                source_optimal_actions = (
                    pair.original_optimal_actions if source_label == "original" else pair.counterfactual_optimal_actions
                )
                target_optimal_actions = (
                    pair.counterfactual_optimal_actions if target_label == "counterfactual" else pair.original_optimal_actions
                )

                source_recorded_action = _extract_recorded_action(source_trajectory, pair.source_step_id)
                target_recorded_action = _extract_recorded_action(target_trajectory, pair.source_step_id)

                for patch_site in selected_patch_sites:
                    baseline_key = (target_label == "original" and pair.original_trajectory_path or pair.counterfactual_trajectory_path, patch_site, pair.source_step_id)
                    if baseline_key not in baseline_cache:
                        baseline_cache[baseline_key] = _run_target_generation(
                            model=model,
                            trajectory=target_trajectory,
                            step_idx=pair.source_step_id,
                            patch_site=patch_site,
                        )

                    baseline_result = baseline_cache[baseline_key]

                    for layer in layer_indices:
                        donor_key = (
                            source_label == "original" and pair.original_trajectory_path or pair.counterfactual_trajectory_path,
                            patch_site,
                            pair.source_step_id,
                            layer,
                        )
                        if donor_key not in donor_cache:
                            donor_cache[donor_key] = _capture_donor_activations(
                                model=model,
                                trajectory=source_trajectory,
                                step_idx=pair.source_step_id,
                                patch_site=patch_site,
                                layer=layer,
                            )

                        donor_result = donor_cache[donor_key]
                        try:
                            patched_result = _run_target_generation(
                                model=model,
                                trajectory=target_trajectory,
                                step_idx=pair.source_step_id,
                                patch_site=patch_site,
                                layer=layer,
                                donor_activations=donor_result["activations"],
                            )
                            error_message = None
                        except Exception as exc:  # noqa: BLE001
                            patched_result = {
                                "output_text": "",
                                "action": None,
                                "parse_failure": True,
                                "input_token_count": None,
                                "generated_token_count": None,
                                "absolute_positions": donor_result["absolute_positions"],
                            }
                            error_message = str(exc)

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
                            "direction": direction,
                            "source_label": source_label,
                            "target_label": target_label,
                            "patch_site": patch_site,
                            "layer": layer,
                            "step_idx": pair.source_step_id,
                            "source_trajectory_path": pair.original_trajectory_path
                            if source_label == "original"
                            else pair.counterfactual_trajectory_path,
                            "target_trajectory_path": pair.counterfactual_trajectory_path
                            if target_label == "counterfactual"
                            else pair.original_trajectory_path,
                            "source_recorded_action": source_recorded_action,
                            "target_recorded_action": target_recorded_action,
                            "baseline_target_action": baseline_result["action"],
                            "patched_target_action": patched_result["action"],
                            "source_optimal_actions": list(source_optimal_actions),
                            "target_optimal_actions": list(target_optimal_actions),
                            "baseline_parse_failure": bool(baseline_result["parse_failure"]),
                            "parse_failure": bool(patched_result["parse_failure"]),
                            "error": error_message,
                            "baseline_input_token_count": baseline_result["input_token_count"],
                            "patched_input_token_count": patched_result["input_token_count"],
                            "baseline_generated_token_count": baseline_result["generated_token_count"],
                            "patched_generated_token_count": patched_result["generated_token_count"],
                            "patched_boundary_positions": patched_result["absolute_positions"],
                            "donor_boundary_positions": donor_result["absolute_positions"],
                            "donor_input_token_count": donor_result["input_token_count"],
                            **outcome_flags,
                        }
                        runs.append(run_record)

        summary_df = _build_summary_dataframe(runs)
        _write_jsonl(runs_path, runs)
        summary_df.to_csv(summary_path, index=False)

        manifest = {
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
            },
            "run_counts": {
                "total_runs": len(runs),
                "success_primary_count": int(sum(record["success_primary"] for record in runs)),
                "parse_failure_count": int(sum(record["parse_failure"] for record in runs)),
                "error_count": int(sum(record["error"] is not None for record in runs)),
            },
        }

        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

        if verbose:
            print(f"Manifest saved to {manifest_path}")
            print(f"Run records saved to {runs_path}")
            print(f"Summary saved to {summary_path}")

        return manifest
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
