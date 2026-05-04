"""Prompt suffix index helpers for flag-swap MLX intervention scripts."""

from __future__ import annotations

from typing import Any


def parse_index_specification(spec: str, max_index: int) -> list[int]:
    """Parse the repository's inclusive index/range syntax."""
    if spec.strip().lower() == "all":
        return list(range(max_index))

    indices = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            start_str, end_str = part.split(":", maxsplit=1)
            start_idx = int(start_str.strip())
            end_idx = int(end_str.strip())
            if start_idx < 0:
                start_idx = max_index + start_idx
            if end_idx < 0:
                end_idx = max_index + end_idx
            if start_idx < 0 or start_idx >= max_index:
                raise ValueError(f"Start index {start_str} out of bounds [0, {max_index})")
            if end_idx < 0 or end_idx >= max_index:
                raise ValueError(f"End index {end_str} out of bounds [0, {max_index})")
            lo, hi = sorted((start_idx, end_idx))
            indices.update(range(lo, hi + 1))
        else:
            idx = int(part)
            if idx < 0:
                idx = max_index + idx
            if idx < 0 or idx >= max_index:
                raise ValueError(f"Index {part} out of bounds [0, {max_index})")
            indices.add(idx)
    return sorted(indices)


def prompt_len(prompt_row: dict[str, Any]) -> int:
    """Return the prompt length recorded in a prompt row."""
    return int(prompt_row.get("prompt_len", len(prompt_row["full_token_ids"])))


def resolve_prompt_suffix_indices(prompt_row: dict[str, Any], spec: str) -> tuple[list[int], list[int]]:
    """Resolve suffix-relative indices to absolute prompt positions.

    The project index syntax uses inclusive ranges, so ``-3:-1`` resolves to the
    final three suffix tokens.
    """
    n_suffix = int(prompt_row["n_suffix"])
    rel_indices = parse_index_specification(spec, n_suffix)
    if not rel_indices:
        raise ValueError(f"{spec!r} selected no prompt-suffix positions")

    start = prompt_len(prompt_row) - n_suffix
    abs_positions = [start + idx for idx in rel_indices]
    return rel_indices, abs_positions


def resolve_prompt_positions(
    prompt_row: dict[str, Any],
    *,
    patch_scope: str,
    prompt_suffix_indices: str,
) -> tuple[list[int], list[int] | None]:
    """Resolve selected prompt positions for full-prompt or suffix-only patching."""
    if patch_scope == "full":
        return list(range(prompt_len(prompt_row))), None
    if patch_scope != "suffix":
        raise ValueError(f"unknown patch_scope: {patch_scope}")
    rel_indices, abs_positions = resolve_prompt_suffix_indices(prompt_row, prompt_suffix_indices)
    return abs_positions, rel_indices


def selected_positions_in_pass(
    *,
    prompt_cursor: int,
    seq_len: int,
    prompt_length: int,
    selected_positions: list[int],
) -> list[tuple[int, int, int]]:
    """Return ``(local_idx, selected_row_idx, absolute_pos)`` touched by this model call."""
    if prompt_cursor >= prompt_length:
        return []

    prompt_seq_len = min(seq_len, prompt_length - prompt_cursor)
    row_by_position = {pos: i for i, pos in enumerate(selected_positions)}
    touched = []
    for local_idx in range(prompt_seq_len):
        abs_pos = prompt_cursor + local_idx
        row_idx = row_by_position.get(abs_pos)
        if row_idx is not None:
            touched.append((local_idx, row_idx, abs_pos))
    return touched
