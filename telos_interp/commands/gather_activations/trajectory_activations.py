"""Small helpers for trajectory activation tests and file layout utilities."""

from pathlib import Path
from shutil import copy2

from .gather_activations_utils import save_activations_to_files as _save_activations_to_files


def save_activations_to_files(*args, **kwargs):
    """Compatibility wrapper for activation file saving."""
    return _save_activations_to_files(*args, **kwargs)


def reconstruct_text_from_tokens(tokens: list[dict]) -> str:
    """Reconstruct text by concatenating token strings."""
    return "".join(token.get("token", "") for token in tokens)


def reconstruct_input_text(trajectory: dict, step_idx: int) -> str:
    """Reconstruct the prompt text for a single trajectory step."""
    step = trajectory["steps"][step_idx]
    prefix_tokens = trajectory["prompt"]["prompt_prefix_tokens"]
    grid_tokens = step["grid_state_tokens"]
    suffix_tokens = step.get("prompt_suffix_tokens", trajectory["prompt"]["prompt_suffix_tokens"])
    return reconstruct_text_from_tokens(prefix_tokens + grid_tokens + suffix_tokens)


def copy_activations_to_step(
    source_base: Path,
    output_base: Path,
    step_idx: int,
    category: str,
    layer_indices: list[int],
) -> None:
    """Copy step-independent activations into a step-specific directory layout."""
    for layer_idx in layer_indices:
        source_dir = source_base / f"layer_{layer_idx}" / category
        if not source_dir.exists():
            continue

        target_dir = output_base / f"layer_{layer_idx}" / f"step_{step_idx}" / category
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path in source_dir.glob("*.pt"):
            copy2(source_path, target_dir / source_path.name)
