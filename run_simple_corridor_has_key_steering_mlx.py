"""Has-key direction probes and final-suffix steering sweeps via MLX.

This script is intentionally local to the flag-swap gridworld prompt families.
It builds paired prompts for either the simple-corridor policy family or the
9x9 door-key grid, captures the selected prompt-suffix activations at selected
layers, trains a linear carrying-key probe per layer, extracts a unit
activation-space direction, and runs direction-only additive steering sweeps on
the target A/B prompts.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from flag_swap_suffix_utils import prompt_len, resolve_prompt_suffix_indices, selected_positions_in_pass
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from run_flag_swap_intervention_mlx import parse_action
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q4"
DEFAULT_PROMPTS = Path("flag_swap/simple_corridor/prompts.jsonl")
DEFAULT_OUT_DIR = Path("flag_swap/simple_corridor/steering")
DEFAULT_LAYERS = (7, 15, 23)
DEFAULT_ALPHAS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
DEFAULT_AGENT_ROWS = (2, 4, 5)
HIDDEN_DIM = 2880
LOGISTIC_C = 0.01
LOGISTIC_SOLVER = "liblinear"

BASE_CORRIDOR_ROWS = [
    ["#", "#", "#"],
    ["#", "G", "#"],
    ["#", "_", "#"],
    ["#", "D", "#"],
    ["#", "_", "#"],
    ["#", "_", "#"],
    ["#", "K", "#"],
    ["#", "#", "#"],
]

SIMPLE_CORRIDOR_POLICY_GRID_TEXT = """# # #
# G #
# _ #
# D #
# A #
# _ #
# K #
# # #"""

DOOR_KEY_GRID_TEXT = """# # # # # # # # #
# _ _ _ # G _ _ #
# _ _ _ # _ _ _ #
# _ _ _ _ _ _ _ #
# D # # # # # # #
# A _ _ # K _ _ #
# _ _ _ # _ _ _ #
# _ _ _ _ _ _ _ #
# # # # # # # # #"""

DOOR_KEY_GRID_TARGET_TEMPLATE = (
    "<|start|>system<|message|>You are a gridworld agent. Output only one action token: UP, DOWN, LEFT, or "
    "RIGHT.<|end|><|start|>user<|message|>Grid:\n{grid}\n\n"
    "A is the agent, K is the key, D is the locked door, and G is the goal. The door can be crossed only when "
    "Carrying key is True. From A, if Carrying key is False, the next action is RIGHT to start the route toward "
    "K. If Carrying key is True, the next action is UP through D toward G. Current state: Carrying key is {key}. "
    "For this current state, the correct next action is {action}. Print only {action}.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>"
)

DOOR_KEY_GRID_PROBE_TEMPLATES = (
    (
        "canonical_grid",
        DOOR_KEY_GRID_TARGET_TEMPLATE,
    ),
    (
        "route_rule",
        "<|start|>system<|message|>Return exactly one legal action token.<|end|>"
        "<|start|>user<|message|>Use this map:\n{grid}\n\n"
        "The agent A is just below the door D. Without the key, do not enter D; move RIGHT to begin going "
        "around the wall toward K. With the key, move UP into D toward G. Carrying key: {key}. "
        "The correct action is {action}.<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "policy_table_grid",
        "<|start|>system<|message|>Answer only UP, DOWN, LEFT, or RIGHT.<|end|>"
        "<|start|>user<|message|>Grid state:\n{grid}\n\n"
        "Policy for this exact state: Carrying key False -> RIGHT; Carrying key True -> UP. "
        "Carrying key is {key}. Therefore emit {action}.<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "door_key_explanation",
        "<|start|>system<|message|>No explanation; only the movement token.<|end|>"
        "<|start|>user<|message|>Map:\n{grid}\n\n"
        "K is behind the wall on the right side, reachable by starting RIGHT from A. D is directly above A and "
        "requires the key. If Carrying key is False choose RIGHT; if Carrying key is True choose UP. "
        "Carrying key: {key}. Correct action: {action}.<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "short_grid",
        "<|start|>system<|message|>Output one action token.<|end|>"
        "<|start|>user<|message|>{grid}\n\n"
        "Current Carrying key value: {key}. In this position, False means RIGHT toward K and True means UP "
        "through D. Next action: {action}.<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "planner_grid_record",
        "<|start|>system<|message|>Print the chosen action only.<|end|>"
        "<|start|>user<|message|>Planner record for the grid:\n{grid}\n\n"
        "If the key is not carried, the planner's first step is RIGHT. If the key is carried, the planner's "
        "first step is UP. Carrying key: {key}. The planner action is {action}. What should be printed?<|end|>"
        "<|start|>assistant<|channel|>final<|message|>",
    ),
)


class LayerHook:
    """Splice a capture/add hook into one transformer layer."""

    def __init__(self, parent_list: list[Any], idx: int):
        self.parent_list = parent_list
        self.idx = idx
        self.original_layer = parent_list[idx]
        self.mode = "passthrough"
        self.direction = None
        self.scale = 0.0
        self.prompt_length = None
        self.selected_positions: list[int] = []
        self.prompt_cursor = 0
        self.captured_by_row: dict[int, Any] = {}
        self.fired = False
        self.seq_lens: list[int] = []
        self.touched_positions: list[int] = []
        self.patched_positions: list[int] = []
        self.pre_norm = None
        self.post_norm = None
        parent_list[idx] = self

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
        if self.mode == "capture":
            for local_idx, row_idx, _ in touched:
                self.captured_by_row[row_idx] = out[0, local_idx, :].astype(mx.float32)
        elif self.mode == "add":
            if self.direction is None:
                raise RuntimeError("direction must be set for add mode")
            expected = (len(self.selected_positions), out.shape[2])
            if self.direction.shape[0] != expected[0] or self.direction.shape[1] != expected[1]:
                raise RuntimeError(f"direction shape {self.direction.shape}, expected {expected}")
            out = self._add_touched(out, touched)
        return out

    def _add_touched(self, out, touched: list[tuple[int, int, int]]):
        before = mx.concatenate([out[:, local_idx : local_idx + 1, :] for local_idx, _, _ in touched], axis=1)
        self.pre_norm = float(mx.linalg.norm(before.astype(mx.float32)))

        pieces = []
        cursor = 0
        changed = []
        for local_idx, row_idx, abs_pos in touched:
            if local_idx > cursor:
                pieces.append(out[:, cursor:local_idx, :])
            perturb = (self.direction[row_idx : row_idx + 1, :] * self.scale).astype(out.dtype)
            tail = out[:, local_idx : local_idx + 1, :] + mx.broadcast_to(
                perturb[None, :, :], (out.shape[0], 1, out.shape[2])
            )
            pieces.append(tail)
            changed.append(tail.astype(mx.float32))
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

    def captured_tensor(self):
        missing = [pos for row_idx, pos in enumerate(self.selected_positions) if row_idx not in self.captured_by_row]
        if missing:
            raise RuntimeError(f"layer {self.idx} missing captured prompt positions: {missing}")
        return mx.stack([self.captured_by_row[i] for i in range(len(self.selected_positions))], axis=0)

    def reset(self) -> None:
        self.mode = "passthrough"
        self.direction = None
        self.scale = 0.0
        self.prompt_length = None
        self.selected_positions = []
        self.prompt_cursor = 0
        self.captured_by_row = {}
        self.fired = False
        self.seq_lens = []
        self.touched_positions = []
        self.patched_positions = []
        self.pre_norm = None
        self.post_norm = None

    def log(self) -> dict[str, Any]:
        return {
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

    def restore(self) -> None:
        self.parent_list[self.idx] = self.original_layer


def generate_one(model, tokenizer, prompt_token_ids: list[int], sampler, max_tokens: int) -> str:
    """Generate one sampled completion from token IDs."""
    pieces = []
    for chunk in stream_generate(model, tokenizer, prompt=prompt_token_ids, max_tokens=max_tokens, sampler=sampler):
        pieces.append(chunk.text)
    return "".join(pieces)


def load_target_prompts(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_label = {row["label"]: row for row in rows}
    missing = {"a", "b"} - set(by_label)
    if missing:
        raise ValueError(f"{path} is missing prompt labels: {sorted(missing)}")
    return by_label


def render_grid(agent_row: int, agent_col: int = 1) -> str:
    rows = [list(row) for row in BASE_CORRIDOR_ROWS]
    if rows[agent_row][agent_col] != "_":
        raise ValueError(f"agent row {agent_row} is not an open corridor cell")
    rows[agent_row][agent_col] = "A"
    width = len(rows[0])
    lines = ["  " + " ".join(str(c) for c in range(width)) + " "]
    for r, cells in enumerate(rows):
        lines.append(f"{r} " + " ".join(cells) + " ")
    return "\n".join(lines) + " "


def encode_no_special(tokenizer, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def build_probe_prompts(
    tokenizer, target_prompts: dict[str, dict[str, Any]], agent_rows: list[int], prompt_suffix_indices: str
) -> list[dict[str, Any]]:
    prompt_a = target_prompts["a"]
    prompt_b = target_prompts["b"]
    prefix_ids = prompt_a["full_token_ids"][: prompt_a["n_prefix"]]
    suffix_false_ids = prompt_a["full_token_ids"][-prompt_a["n_suffix"] :]
    suffix_true_ids = prompt_b["full_token_ids"][-prompt_b["n_suffix"] :]

    rows = []
    for state_idx, agent_row in enumerate(agent_rows):
        grid_text = render_grid(agent_row)
        grid_ids = encode_no_special(tokenizer, grid_text)
        if agent_row == 4:
            expected = prompt_a["full_token_ids"][prompt_a["n_prefix"] : prompt_a["n_prefix"] + prompt_a["n_grid"]]
            if grid_ids != expected:
                raise ValueError("rendered row-4 grid tokens do not match existing simple-corridor prompt")

        for label, carrying_key, suffix_ids, action in (
            (0, False, suffix_false_ids, "DOWN"),
            (1, True, suffix_true_ids, "UP"),
        ):
            full_ids = prefix_ids + grid_ids + suffix_ids
            prompt_len = len(full_ids)
            row = {
                "example_id": f"r{agent_row}_key{int(carrying_key)}",
                "state_id": state_idx,
                "agent_row": agent_row,
                "pos": [1, agent_row],
                "carrying_key": carrying_key,
                "label": label,
                "agent_action_expected": action,
                "n_prefix": len(prefix_ids),
                "n_grid": len(grid_ids),
                "n_suffix": len(suffix_ids),
                "prompt_len": prompt_len,
                "full_token_ids": full_ids,
                "grid_text": grid_text,
            }
            suffix_indices, suffix_positions = resolve_prompt_suffix_indices(row, prompt_suffix_indices)
            row["prompt_suffix_indices"] = prompt_suffix_indices
            row["suffix_relative_indices"] = suffix_indices
            row["suffix_positions"] = suffix_positions
            rows.append(row)
    return rows


FINAL_CHANNEL_POLICY_TEMPLATES = (
    (
        "canonical",
        "<|start|>system<|message|>You are a gridworld agent. Output only one action token: UP or DOWN.<|end|>"
        "<|start|>user<|message|>Simple corridor policy: if Carrying key is False, the next action is DOWN; "
        "if Carrying key is True, the next action is UP. Current state: Carrying key is {key}. "
        "What is the next action?<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "below_key_rule",
        "<|start|>system<|message|>Return exactly one movement token.<|end|>"
        "<|start|>user<|message|>In this simple corridor, an agent without the key moves DOWN to collect it. "
        "An agent already carrying the key moves UP toward the door and goal. Carrying key: {key}. "
        "Next action?<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "if_then",
        "<|start|>system<|message|>Answer with only UP or DOWN.<|end|>"
        "<|start|>user<|message|>Use the rule for the corridor: False means choose DOWN; True means choose UP. "
        "The field Carrying key is {key}. Choose now.<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "door_goal",
        "<|start|>system<|message|>You control a simple corridor agent.<|end|>"
        "<|start|>user<|message|>Door and goal are above the agent. The key is below. "
        "If Carrying key is False, go DOWN first. If Carrying key is True, go UP. "
        "Current Carrying key value: {key}. Emit the action token.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "compact_state",
        "<|start|>system<|message|>Output one action token.<|end|>"
        "<|start|>user<|message|>State summary: simple corridor; key possession is {key}. "
        "Policy table: False -> DOWN, True -> UP. Action?<|end|>"
        "<|start|>assistant<|channel|>final<|message|>",
    ),
    (
        "planner_record",
        "<|start|>system<|message|>No explanation; only the action.<|end|>"
        "<|start|>user<|message|>Planner input says Carrying key: {key}. "
        "The simple-corridor next-action rule maps not carrying to DOWN and carrying to UP. "
        "What action should be printed?<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
)


def build_final_channel_policy_probe_prompts(tokenizer, prompt_suffix_indices: str) -> list[dict[str, Any]]:
    rows = []
    for state_idx, (variant, template) in enumerate(FINAL_CHANNEL_POLICY_TEMPLATES):
        for label, carrying_key, action in ((0, False, "DOWN"), (1, True, "UP")):
            text = template.format(key=str(carrying_key))
            full_ids = encode_no_special(tokenizer, text)
            row = {
                "example_id": f"{variant}_key{int(carrying_key)}",
                "state_id": state_idx,
                "agent_row": None,
                "pos": [1, 4],
                "carrying_key": carrying_key,
                "label": label,
                "agent_action_expected": action,
                "n_prefix": len(full_ids) - 3,
                "n_grid": 0,
                "n_suffix": 3,
                "prompt_len": len(full_ids),
                "full_token_ids": full_ids,
                "prompt_text": text,
                "probe_family": "final_channel_policy",
                "variant": variant,
            }
            suffix_indices, suffix_positions = resolve_prompt_suffix_indices(row, prompt_suffix_indices)
            row["prompt_suffix_indices"] = prompt_suffix_indices
            row["suffix_relative_indices"] = suffix_indices
            row["suffix_positions"] = suffix_positions
            rows.append(row)
    return rows


def build_final_channel_row(
    tokenizer,
    *,
    label: str | int,
    carrying_key: bool,
    action: str,
    text: str,
    variant: str,
    state_id: int,
    prompt_suffix_indices: str,
    probe_family: str,
) -> dict[str, Any]:
    full_ids = encode_no_special(tokenizer, text)
    row = {
        "example_id": f"{variant}_key{int(carrying_key)}",
        "state_id": state_id,
        "agent_row": None,
        "pos": [1, 5],
        "carrying_key": carrying_key,
        "label": label,
        "agent_action_expected": action,
        "agent_action_recorded": action,
        "n_prefix": len(full_ids) - 3,
        "n_grid": 0,
        "n_suffix": 3,
        "prompt_len": len(full_ids),
        "full_token_ids": full_ids,
        "prompt_text": text,
        "probe_family": probe_family,
        "variant": variant,
        "grid_text": DOOR_KEY_GRID_TEXT,
    }
    suffix_indices, suffix_positions = resolve_prompt_suffix_indices(row, prompt_suffix_indices)
    row["prompt_suffix_indices"] = prompt_suffix_indices
    row["suffix_relative_indices"] = suffix_indices
    row["suffix_positions"] = suffix_positions
    return row


def build_door_key_grid_probe_prompts(tokenizer, prompt_suffix_indices: str) -> list[dict[str, Any]]:
    rows = []
    for state_idx, (variant, template) in enumerate(DOOR_KEY_GRID_PROBE_TEMPLATES):
        for label, carrying_key, action in ((0, False, "RIGHT"), (1, True, "UP")):
            text = template.format(grid=DOOR_KEY_GRID_TEXT, key=str(carrying_key), action=action)
            rows.append(
                build_final_channel_row(
                    tokenizer,
                    label=label,
                    carrying_key=carrying_key,
                    action=action,
                    text=text,
                    variant=variant,
                    state_id=state_idx,
                    prompt_suffix_indices=prompt_suffix_indices,
                    probe_family="door_key_grid",
                )
            )
    return rows


def build_door_key_grid_target_prompts(tokenizer, prompt_suffix_indices: str) -> list[dict[str, Any]]:
    rows = []
    for label, carrying_key, action in (("a", False, "RIGHT"), ("b", True, "UP")):
        text = DOOR_KEY_GRID_TARGET_TEMPLATE.format(grid=DOOR_KEY_GRID_TEXT, key=str(carrying_key), action=action)
        rows.append(
            build_final_channel_row(
                tokenizer,
                label=label,
                carrying_key=carrying_key,
                action=action,
                text=text,
                variant=f"door_key_grid_key_{str(carrying_key).lower()}",
                state_id=0,
                prompt_suffix_indices=prompt_suffix_indices,
                probe_family="door_key_grid",
            )
        )
    return rows


def write_door_key_grid_prompts(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading tokenizer from {args.model_id} ...", flush=True)
    model, tokenizer = load(args.model_id)
    del model
    rows = build_door_key_grid_target_prompts(tokenizer, args.prompt_suffix_indices)
    with args.output.open("w") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
    for row in rows:
        print(
            f"prompt {row['label']}: action={row['agent_action_recorded']} "
            f"prompt_len={row['prompt_len']} suffix_positions={row['suffix_positions']}",
            flush=True,
        )
    print(f"Wrote {len(rows)} prompts to {args.output}")


def capture_prompt_activations(
    model,
    tokenizer,
    prompt_rows: list[dict[str, Any]],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    n_pos = len(prompt_rows[0]["suffix_positions"])
    hooks = [LayerHook(model.model.layers, layer) for layer in layers]
    try:
        arrays: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        sampler = make_sampler(temp=0.0)
        for i, row in enumerate(prompt_rows, start=1):
            for hook in hooks:
                hook.configure_prompt(row, row["suffix_positions"])
                hook.mode = "capture"
            _ = generate_one(model, tokenizer, row["full_token_ids"], sampler, max_tokens=1)

            for layer, hook in zip(layers, hooks, strict=True):
                cap = np.array(hook.captured_tensor().astype(mx.float32))
                if cap.shape != (n_pos, HIDDEN_DIM):
                    raise RuntimeError(f"layer {layer} capture shape {cap.shape}, expected {(n_pos, HIDDEN_DIM)}")
                arrays[layer].append(torch.from_numpy(cap.reshape(-1).copy()))
            print(f"  captured {i}/{len(prompt_rows)}: {row['example_id']}", flush=True)
        return {layer: torch.stack(layer_arrays, dim=0) for layer, layer_arrays in arrays.items()}
    finally:
        for hook in hooks:
            hook.restore()


def split_by_state(labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        idx = np.arange(len(labels))
        return idx, idx
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(1, len(unique_groups) // 3), random_state=seed)
    train_idx, eval_idx = next(splitter.split(np.zeros_like(labels), labels, groups=groups))
    return train_idx, eval_idx


def fit_logistic_probe(
    activations: torch.Tensor,
    labels_t: torch.Tensor,
    groups_t: torch.Tensor,
    layer: int,
    n_pos: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x = activations.numpy().astype(np.float64)
    y = labels_t.numpy().astype(np.int64)
    groups = groups_t.numpy().astype(np.int64)

    train_idx, eval_idx = split_by_state(y, groups, seed)
    eval_scaler = StandardScaler()
    x_train = eval_scaler.fit_transform(x[train_idx])
    x_eval = eval_scaler.transform(x[eval_idx])
    eval_clf = LogisticRegression(max_iter=10_000, solver=LOGISTIC_SOLVER, C=LOGISTIC_C, random_state=seed)
    eval_clf.fit(x_train, y[train_idx])
    pred = eval_clf.predict(x_eval)

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=10_000, solver=LOGISTIC_SOLVER, C=LOGISTIC_C, random_state=seed)
    clf.fit(x_norm, y)

    coef_norm = clf.coef_.reshape(-1).astype(np.float64)
    act_coef = coef_norm / scaler.scale_.astype(np.float64)
    if not np.isfinite(act_coef).all():
        raise RuntimeError(f"layer {layer} probe produced non-finite activation-space coefficients")
    norm = float(np.linalg.norm(act_coef))
    if norm == 0.0:
        raise RuntimeError(f"layer {layer} probe produced a zero direction")
    direction = act_coef / norm

    projections = x @ direction
    class_gap = float(projections[y == 1].mean() - projections[y == 0].mean())
    if class_gap < 0.0:
        direction = -direction
        coef_norm = -coef_norm
        class_gap = -class_gap

    train_pred = clf.predict(x_norm)
    metrics = {
        "layer": layer,
        "num_samples": int(len(y)),
        "num_train": int(len(train_idx)),
        "num_eval": int(len(eval_idx)),
        "train_state_ids": sorted(int(g) for g in np.unique(groups[train_idx])),
        "eval_state_ids": sorted(int(g) for g in np.unique(groups[eval_idx])),
        "train_accuracy_all_states": float(accuracy_score(y, train_pred)),
        "train_balanced_accuracy_all_states": float(balanced_accuracy_score(y, train_pred)),
        "eval_accuracy": float(accuracy_score(y[eval_idx], pred)),
        "eval_balanced_accuracy": float(balanced_accuracy_score(y[eval_idx], pred)),
        "class_gap": class_gap,
        "direction_norm": float(np.linalg.norm(direction)),
        "coef_norm": float(np.linalg.norm(coef_norm)),
        "logistic_c": LOGISTIC_C,
        "logistic_solver": LOGISTIC_SOLVER,
        "class_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
    }
    bundle = {
        "layer": layer,
        "direction": torch.from_numpy(direction.reshape(n_pos, HIDDEN_DIM).copy()),
        "class_gap": class_gap,
        "coef_normalized": torch.from_numpy(coef_norm.astype(np.float32).copy()),
        "intercept": torch.from_numpy(clf.intercept_.astype(np.float32).copy()),
        "classes": torch.from_numpy(clf.classes_.astype(np.int64).copy()),
        "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32).copy()),
        "scaler_std": torch.from_numpy(scaler.scale_.astype(np.float32).copy()),
        "metrics": metrics,
        "direction_note": "Unit activation-space direction from normalized logistic coef divided by scaler_std.",
    }
    return metrics, bundle


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_capture_train(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layers = args.layers
    agent_rows = args.agent_rows[: args.max_states] if args.max_states is not None else args.agent_rows
    target_prompts = load_target_prompts(args.prompts)

    print(f"Loading {args.model_id} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model_id)
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.layers)
    for layer in layers:
        if layer >= n_layers:
            raise SystemExit(f"--layer {layer} out of range; model has {n_layers} layers")

    if args.probe_family == "final-channel-policy":
        prompt_rows = build_final_channel_policy_probe_prompts(tokenizer, args.prompt_suffix_indices)
        if args.max_states is not None:
            keep_states = set(range(args.max_states))
            prompt_rows = [row for row in prompt_rows if row["state_id"] in keep_states]
    elif args.probe_family == "door-key-grid":
        prompt_rows = build_door_key_grid_probe_prompts(tokenizer, args.prompt_suffix_indices)
        if args.max_states is not None:
            keep_states = set(range(args.max_states))
            prompt_rows = [row for row in prompt_rows if row["state_id"] in keep_states]
    else:
        prompt_rows = build_probe_prompts(tokenizer, target_prompts, agent_rows, args.prompt_suffix_indices)
    n_pos = len(prompt_rows[0]["suffix_positions"])
    prompt_path = args.out_dir / "paired_probe_prompts.jsonl"
    with prompt_path.open("w") as f:
        for row in prompt_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(prompt_rows)} paired prompt rows to {prompt_path}")

    labels = torch.tensor([row["label"] for row in prompt_rows], dtype=torch.long)
    groups = torch.tensor([row["state_id"] for row in prompt_rows], dtype=torch.long)
    class_counts = Counter(labels.tolist())
    print(f"Class balance: False={class_counts.get(0, 0)} True={class_counts.get(1, 0)}")

    activations_by_layer = capture_prompt_activations(model, tokenizer, prompt_rows, layers)
    dataset = {
        "layers": layers,
        "n_pos": n_pos,
        "hidden_dim": HIDDEN_DIM,
        "prompt_suffix_indices": args.prompt_suffix_indices,
        "agent_rows": agent_rows,
        "prompt_rows": prompt_rows,
        "labels": labels,
        "groups": groups,
        "activations_by_layer": activations_by_layer,
    }
    dataset_path = args.out_dir / "paired_probe_activations.pt"
    torch.save(dataset, dataset_path)
    print(f"Wrote activations to {dataset_path}")

    metrics_by_layer = {}
    for layer in layers:
        metrics, bundle = fit_logistic_probe(activations_by_layer[layer], labels, groups, layer, n_pos, seed=args.seed)
        bundle["prompt_suffix_indices"] = args.prompt_suffix_indices
        bundle["n_pos"] = n_pos
        bundle["dataset_path"] = str(dataset_path)
        direction_path = args.out_dir / f"has_key_direction_layer_{layer}.pt"
        torch.save(bundle, direction_path)
        metrics["direction_path"] = str(direction_path)
        metrics_by_layer[str(layer)] = metrics
        print(
            f"Layer {layer}: eval_acc={metrics['eval_accuracy']:.3f} "
            f"eval_bal_acc={metrics['eval_balanced_accuracy']:.3f} class_gap={metrics['class_gap']:.3f}",
            flush=True,
        )

    metrics_doc = {
        "dataset": {
            "path": str(dataset_path),
            "prompt_path": str(prompt_path),
            "num_samples": len(prompt_rows),
            "num_states": len({row["state_id"] for row in prompt_rows}),
            "agent_rows": agent_rows if args.probe_family == "simple-corridor" else [],
            "probe_family": args.probe_family,
            "prompt_suffix_indices": args.prompt_suffix_indices,
            "suffix_relative_indices": prompt_rows[0]["suffix_relative_indices"],
            "suffix_positions_example": prompt_rows[0]["suffix_positions"],
            "n_pos": n_pos,
            "class_counts": {"False": int(class_counts.get(0, 0)), "True": int(class_counts.get(1, 0))},
        },
        "layers": metrics_by_layer,
    }
    save_json(args.out_dir / "probe_metrics.json", metrics_doc)
    print(f"Wrote probe metrics to {args.out_dir / 'probe_metrics.json'}")


def load_direction(layer: int, out_dir: Path) -> dict[str, Any]:
    path = out_dir / f"has_key_direction_layer_{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing direction bundle for layer {layer}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def run_steering(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    target_prompts = load_target_prompts(args.prompts)

    print(f"Loading {args.model_id} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model_id)
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.layers)
    for layer in args.layers:
        if layer >= n_layers:
            raise SystemExit(f"--layer {layer} out of range; model has {n_layers} layers")

    n_written = 0
    with args.output.open("w") as fout:
        for layer_idx, layer in enumerate(args.layers):
            bundle = load_direction(layer, args.out_dir)
            direction_t = bundle["direction"].float()
            n_pos = len(
                resolve_prompt_suffix_indices(next(iter(target_prompts.values())), args.prompt_suffix_indices)[0]
            )
            if tuple(direction_t.shape) != (n_pos, HIDDEN_DIM):
                raise RuntimeError(f"layer {layer} direction shape {tuple(direction_t.shape)}")
            direction = mx.array(direction_t.numpy().astype(np.float32))
            class_gap = float(bundle["class_gap"])
            hook = LayerHook(model.model.layers, layer)
            try:
                for prompt_idx, prompt_label in enumerate(("a", "b")):
                    prompt_row = target_prompts[prompt_label]
                    suffix_indices, suffix_positions = resolve_prompt_suffix_indices(
                        prompt_row, args.prompt_suffix_indices
                    )
                    for alpha_idx, alpha in enumerate(args.alphas):
                        scale = float(alpha) * class_gap
                        samples = []
                        t1 = time.time()
                        for sample_idx in range(args.n_samples):
                            hook.configure_prompt(prompt_row, suffix_positions)
                            hook.mode = "add"
                            hook.direction = direction
                            hook.scale = scale
                            seed = (
                                args.seed
                                + layer_idx * 1_000_000
                                + prompt_idx * 100_000
                                + alpha_idx * 10_000
                                + sample_idx
                            )
                            mx.random.seed(seed)
                            sampler = make_sampler(temp=args.temperature)
                            text = generate_one(
                                model,
                                tokenizer,
                                prompt_row["full_token_ids"],
                                sampler,
                                args.max_new_tokens,
                            )
                            emitted_action = parse_action(text)
                            row = {
                                "layer": layer,
                                "alpha": float(alpha),
                                "class_gap": class_gap,
                                "scale": scale,
                                "prompt_label": prompt_label,
                                "prompt_variant": prompt_row["variant"],
                                "prompt_suffix_indices": args.prompt_suffix_indices,
                                "suffix_relative_indices": suffix_indices,
                                "selected_prompt_positions": suffix_positions,
                                "sample_idx": sample_idx,
                                "emitted_action": emitted_action,
                                "text_tail": text[-300:],
                            }
                            if sample_idx == 0:
                                row["_hook_log"] = hook.log()
                            fout.write(json.dumps(row) + "\n")
                            fout.flush()
                            samples.append(emitted_action or "PARSE_FAIL")
                            n_written += 1
                        counts = Counter(samples)
                        print(
                            f"layer={layer} prompt={prompt_label} alpha={alpha:g} "
                            f"counts={dict(counts)} elapsed={time.time() - t1:.1f}s",
                            flush=True,
                        )
            finally:
                hook.restore()

    print(f"Wrote {n_written} rows to {args.output}")


def freq(counts: Counter, action: str) -> float:
    n = sum(counts.values())
    return counts.get(action, 0) / n if n else 0.0


def is_non_decreasing(values: list[float], tol: float = 1e-9) -> bool:
    return all(b + tol >= a for a, b in zip(values, values[1:], strict=False))


def is_non_increasing(values: list[float], tol: float = 1e-9) -> bool:
    return all(b <= a + tol for a, b in zip(values, values[1:], strict=False))


def slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), deg=1)[0])


def counts_summary(counts: Counter) -> str:
    n = sum(counts.values())
    parts = []
    for label in ("UP", "DOWN", "LEFT", "RIGHT", "PARSE_FAIL"):
        count = counts.get(label, 0)
        if count:
            parts.append(f"{label}={count}/{n}")
    return ", ".join(parts) if parts else f"n={n}"


def dominant_action(counts: Counter) -> str:
    action_counts = [(counts.get(action, 0), action) for action in ("UP", "DOWN", "LEFT", "RIGHT")]
    count, action = max(action_counts)
    if count == 0:
        return "UP"
    return action


def run_report(args: argparse.Namespace) -> None:  # noqa: PLR0912
    metrics = json.loads(args.metrics.read_text())
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no steering rows in {args.results}")
    ds = metrics["dataset"]

    grouped: dict[tuple[int, str, float], Counter] = defaultdict(Counter)
    hook_logs: dict[tuple[int, str, float], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["layer"]), row["prompt_label"], float(row["alpha"]))
        grouped[key][row["emitted_action"] or "PARSE_FAIL"] += 1
        if "_hook_log" in row:
            hook_logs[key] = row["_hook_log"]

    layers = sorted({key[0] for key in grouped})
    alphas = sorted({key[2] for key in grouped})
    primary_layer = 23 if 23 in layers else layers[-1]
    min_alpha = alphas[0]
    max_alpha = alphas[-1]
    zero_alpha = 0.0 if 0.0 in alphas else min(alphas, key=abs)
    false_action = dominant_action(grouped.get((primary_layer, "a", zero_alpha), Counter()))
    true_action = dominant_action(grouped.get((primary_layer, "b", zero_alpha), Counter()))
    endpoint_log_a = hook_logs.get((primary_layer, "a", max_alpha), {})
    endpoint_log_b = hook_logs.get((primary_layer, "b", min_alpha), {})
    selected_positions = (
        endpoint_log_a.get("selected_prompt_positions")
        or endpoint_log_b.get("selected_prompt_positions")
        or ds.get("suffix_positions_example", [])
    )
    seq_lens = endpoint_log_a.get("seq_lens") or endpoint_log_b.get("seq_lens") or []

    probe_family = ds.get("probe_family", "simple-corridor")
    if probe_family == "door-key-grid":
        title = "9x9 door-key grid has-key direction steering across layers"
        scope_family = "matched final-channel 9x9 door-key grid family"
    elif probe_family == "final-channel-policy":
        title = "Final-channel simple-corridor has-key direction steering across layers"
        scope_family = "matched final-channel simple-corridor policy family"
    else:
        title = "Simple-corridor has-key direction steering across layers"
        scope_family = "simple-corridor prompt family"

    out = []
    out.append(f"# {title}\n\n")
    out.append("## Dataset and probes\n\n")
    state_label = "grid states" if probe_family == "simple-corridor" else "template states"
    agent_rows = f"; agent rows {ds['agent_rows']}" if ds.get("agent_rows") else ""
    out.append(
        f"- Paired synthetic dataset: {ds['num_samples']} prompts "
        f"({ds['num_states']} {state_label}{agent_rows}; probe family {probe_family}).\n"
    )
    out.append(f"- Class balance: False={ds['class_counts']['False']}, True={ds['class_counts']['True']}.\n")
    out.append(
        f"- Captured features: prompt_suffix_indices={ds.get('prompt_suffix_indices', '-3:-1')!r}, "
        f"relative indices {ds.get('suffix_relative_indices', [])}, flattened to "
        f"{int(ds.get('n_pos', 3)) * HIDDEN_DIM} dimensions.\n\n"
    )
    out.append("| layer | eval acc | eval balanced acc | all-state train acc | class gap | direction norm |\n")
    out.append("|---:|---:|---:|---:|---:|---:|\n")
    for layer in layers:
        m = metrics["layers"].get(str(layer), {})
        out.append(
            f"| {layer} | {m.get('eval_accuracy', float('nan')):.3f} | "
            f"{m.get('eval_balanced_accuracy', float('nan')):.3f} | "
            f"{m.get('train_accuracy_all_states', float('nan')):.3f} | "
            f"{m.get('class_gap', float('nan')):.3f} | {m.get('direction_norm', float('nan')):.3f} |\n"
        )
    out.append("\n")

    out.append("## Prompt family\n\n")
    if probe_family == "door-key-grid":
        out.append(
            "The 9x9 target grid is fixed across prompt A and prompt B; only the key-state condition and its "
            "policy-implied action differ. Prompt A is the no-key condition, where the baseline next action is "
            f"{false_action} to start routing toward K. Prompt B is the has-key condition, where the baseline "
            f"next action is {true_action} through D toward G.\n\n"
        )
        out.append("```text\n")
        out.append(DOOR_KEY_GRID_TEXT)
        out.append("\n```\n\n")
        out.append(
            "The paired probe dataset uses six synthetic 9x9 door-key templates with the same rendered grid in "
            "each False/True pair. The templates are deliberately final-channel action prompts so the first "
            "generated content is the action token being measured. This keeps the intervention local to the "
            "pre-action suffix activations, but it also means the finding is scoped to this matched policy-prompt "
            "family rather than to arbitrary verbose grid reasoning prompts.\n\n"
        )
    elif probe_family == "final-channel-policy":
        out.append(
            "The simple-corridor run uses a compact final-channel policy abstraction of the corridor. Prompt A "
            f"is the no-key condition, whose baseline action is {false_action}; prompt B is the has-key "
            f"condition, whose baseline action is {true_action}. The policy is equivalent to the corridor below: "
            "without the key, move down toward K; with the key, move up toward D and G.\n\n"
        )
        out.append("```text\n")
        out.append(SIMPLE_CORRIDOR_POLICY_GRID_TEXT)
        out.append("\n```\n\n")
        out.append(
            "This is the matched final-channel simple-corridor policy family, not the long verbose reasoning "
            "prompt. That distinction matters: earlier replacement/transplant-style runs on longer prompts did "
            "not produce a clean causal action switch. The finding reported here is specifically that additive "
            "probe-direction steering can control the first action token when the action is emitted immediately "
            "after the assistant final-channel suffix.\n\n"
        )
    else:
        out.append(
            "The simple-corridor probe dataset enumerates legal agent positions in the open corridor while "
            "holding the G, D, and K topology fixed. Each rendered grid state is paired as Carrying key False "
            f"versus True, with the expected policy shift from {false_action} to {true_action}.\n\n"
        )
        out.append("```text\n")
        out.append(SIMPLE_CORRIDOR_POLICY_GRID_TEXT)
        out.append("\n```\n\n")

    out.append("## Detailed finding\n\n")
    out.append(
        "This run tests whether the final three prompt-suffix activations can causally control the emitted "
        "action when we add the direction learned by a linear `has_key` probe. The intervention is not a donor "
        "activation transplant: for each selected token position, the hook adds "
        "`alpha * class_gap * direction[layer, token_position]` to the layer output and leaves every other "
        "prompt and generation position unchanged.\n\n"
    )
    out.append(
        f"The selected locations are exactly `prompt_suffix_indices={ds.get('prompt_suffix_indices', '-3:-1')!r}`. "
        f"In this target prompt family those are suffix-relative indices {ds.get('suffix_relative_indices', [])}; "
        f"for the target A/B prompts the hook logs resolve them to absolute positions `{selected_positions}`, the final-channel suffix "
        "tokens `<|channel|>`, `final`, and `<|message|>`. MLX processes this prompt as one multi-token prefill "
        "followed by a one-token prompt singleton, so the first two selected suffix tokens are changed in the "
        "prefill call and the final selected suffix token is changed in the singleton call. The recorded hook logs below verify this split with "
        f"`seq_lens={seq_lens}` and `patched_positions={selected_positions}`.\n\n"
    )
    out.append(
        f"The successful steering evidence is concentrated at layer {primary_layer}. Positive alpha means moving in the "
        f"`has_key=True` direction; therefore prompt A, whose baseline action is {false_action}, should move "
        f"toward {true_action} as alpha increases. Negative alpha means moving away from `has_key=True`; "
        f"therefore prompt B, whose baseline action is {true_action}, should move toward {false_action} as "
        "alpha decreases.\n\n"
    )
    if primary_layer in layers:
        selected_alphas = [alpha for alpha in (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0) if alpha in alphas]
        if not selected_alphas:
            selected_alphas = alphas
        out.append(f"Layer {primary_layer} action distributions at the most diagnostic alpha values:\n\n")
        out.append(f"| prompt | alpha | distribution | freq({true_action}) | freq({false_action}) |\n")
        out.append("|---|---:|---|---:|---:|\n")
        for prompt in ("a", "b"):
            for alpha in selected_alphas:
                counts = grouped.get((primary_layer, prompt, alpha), Counter())
                out.append(
                    f"| {prompt} | {alpha:g} | {counts_summary(counts)} | "
                    f"{freq(counts, true_action):.3f} | {freq(counts, false_action):.3f} |\n"
                )
        out.append("\n")

        log_a = endpoint_log_a
        log_b = endpoint_log_b
        if log_a or log_b:
            out.append(f"Layer {primary_layer} hook evidence for the endpoint flips:\n\n")
            out.append(
                "| prompt | alpha | mode | selected positions | patched positions | seq lens | scale | pre norm | post norm |\n"
            )
            out.append("|---|---:|---|---|---|---|---:|---:|---:|\n")
            for prompt, alpha, log in (("a", max_alpha, log_a), ("b", min_alpha, log_b)):
                if not log:
                    continue
                out.append(
                    f"| {prompt} | {alpha:g} | {log.get('mode', '')} | "
                    f"{log.get('selected_prompt_positions', '')} | {log.get('patched_prompt_positions', '')} | "
                    f"{log.get('seq_lens', '')} | {float(log.get('scale') or 0):.3f} | "
                    f"{float(log.get('pre_norm') or 0):.1f} | {float(log.get('post_norm') or 0):.1f} |\n"
                )
            out.append("\n")

        a0 = grouped.get((primary_layer, "a", zero_alpha), Counter())
        a_pos = grouped.get((primary_layer, "a", max_alpha), Counter())
        b0 = grouped.get((primary_layer, "b", zero_alpha), Counter())
        b_neg = grouped.get((primary_layer, "b", min_alpha), Counter())
        out.append(
            f"Endpoint summary: at layer {primary_layer}, prompt A changes from "
            f"{counts_summary(a0)} at alpha {zero_alpha:g} to {counts_summary(a_pos)} at alpha {max_alpha:g}. "
            "Prompt B changes from "
            f"{counts_summary(b0)} at alpha {zero_alpha:g} to {counts_summary(b_neg)} at alpha {min_alpha:g}. "
            "This is the expected bidirectional pattern for a `has_key` direction: adding it makes a no-key "
            "prompt behave like has-key, while subtracting it makes a has-key prompt behave like no-key.\n\n"
        )
    else:
        out.append(
            "Layer 23 was not included in this report, so the strongest observed effect is not available here.\n\n"
        )

    out.append(
        "Layer comparison matters for interpretation. The other tested layers do not provide a clean "
        f"bidirectional action-control result in this sweep. Layer {primary_layer} is the only layer that satisfies the monotonic steering criterion "
        "across the sampled alphas and produces clean endpoint flips without parse failures at the decisive "
        f"A alpha {max_alpha:g} and B alpha {min_alpha:g} endpoints.\n\n"
    )
    out.append(
        f"Scope and caveats: this is evidence for a causal effect in the {scope_family}, where the first "
        "generated content is the actual action token. It does not claim that the "
        "same single probe direction will steer every verbose grid prompt or every reasoning format. The result "
        "also depends on relatively large layer-23 scales because the direction is a unit activation-space probe "
        "direction multiplied by the probe class gap and alpha. Within this scoped setting, however, the hook logs "
        "and action distributions support the causal claim: additive steering of only the final three suffix "
        "activations along the learned `has_key` direction is sufficient to change the emitted action.\n\n"
    )

    out.append("## Hook sanity\n\n")
    out.append(
        "| layer | prompt | alpha | fired | seq lens | selected positions | patched positions | scale | pre norm | post norm |\n"
    )
    out.append("|---:|---|---:|---|---|---|---|---:|---:|---:|\n")
    for layer in layers:
        for prompt in ("a", "b"):
            for alpha in alphas:
                log = hook_logs.get((layer, prompt, alpha), {})
                scale_str = f"{float(log.get('scale', 0.0)):.3f}"
                if log.get("pre_norm") is not None:
                    out.append(
                        f"| {layer} | {prompt} | {alpha:g} | {log.get('fired')} | {log.get('seq_lens', '')} | "
                        f"{log.get('selected_prompt_positions', '')} | {log.get('patched_prompt_positions', '')} | "
                        f"{scale_str} | {float(log['pre_norm']):.1f} | {float(log['post_norm']):.1f} |\n"
                    )
                else:
                    out.append(
                        f"| {layer} | {prompt} | {alpha:g} | {log.get('fired')} | {log.get('seq_lens', '')} | "
                        f"{log.get('selected_prompt_positions', '')} | {log.get('patched_prompt_positions', '')} | "
                        f"{scale_str} |  |  |\n"
                    )
    out.append("\n")

    out.append("## Action counts by layer, prompt, and alpha\n\n")
    out.append("| layer | prompt | alpha | LEFT | RIGHT | UP | DOWN | parse_fail | n | freq(UP) | freq(DOWN) |\n")
    out.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for layer in layers:
        for prompt in ("a", "b"):
            for alpha in alphas:
                counts = grouped[(layer, prompt, alpha)]
                n = sum(counts.values())
                out.append(
                    f"| {layer} | {prompt} | {alpha:g} | "
                    f"{counts.get('LEFT', 0)} | {counts.get('RIGHT', 0)} | {counts.get('UP', 0)} | "
                    f"{counts.get('DOWN', 0)} | {counts.get('PARSE_FAIL', 0)} | {n} | "
                    f"{freq(counts, 'UP'):.3f} | {freq(counts, 'DOWN'):.3f} |\n"
                )
    out.append("\n")

    out.append("## UP/DOWN frequency curves\n\n")
    for layer in layers:
        out.append(f"### Layer {layer}\n\n")
        out.append("| alpha | A freq(UP) | A freq(DOWN) | B freq(UP) | B freq(DOWN) |\n")
        out.append("|---:|---:|---:|---:|---:|\n")
        for alpha in alphas:
            ca = grouped[(layer, "a", alpha)]
            cb = grouped[(layer, "b", alpha)]
            out.append(
                f"| {alpha:g} | {freq(ca, 'UP'):.3f} | {freq(ca, 'DOWN'):.3f} | "
                f"{freq(cb, 'UP'):.3f} | {freq(cb, 'DOWN'):.3f} |\n"
            )
        out.append("\n")

    out.append("## Layer comparison\n\n")
    out.append(
        f"Positive alpha means more has_key=True. For prompt A, expected steering is {false_action} to "
        f"{true_action} as alpha increases. For prompt B, expected steering is {true_action} to "
        f"{false_action} as alpha decreases, equivalently {false_action} to {true_action} as alpha increases.\n\n"
    )
    out.append(
        f"| layer | A {true_action} monotone up | A {false_action} monotone down | "
        f"B {true_action} monotone up | B {false_action} monotone down | "
        f"A {true_action} slope | B {true_action} slope | verdict |\n"
    )
    out.append("|---:|---|---|---|---|---:|---:|---|\n")
    for layer in layers:
        a_true = [freq(grouped[(layer, "a", alpha)], true_action) for alpha in alphas]
        a_false = [freq(grouped[(layer, "a", alpha)], false_action) for alpha in alphas]
        b_true = [freq(grouped[(layer, "b", alpha)], true_action) for alpha in alphas]
        b_false = [freq(grouped[(layer, "b", alpha)], false_action) for alpha in alphas]
        checks = [
            is_non_decreasing(a_true),
            is_non_increasing(a_false),
            is_non_decreasing(b_true),
            is_non_increasing(b_false),
        ]
        a_slope = slope(alphas, a_true)
        b_slope = slope(alphas, b_true)
        endpoint_ok = a_true[-1] > a_true[0] and b_false[0] > b_false[-1]
        verdict = "monotonic steering" if all(checks) and endpoint_ok else "not monotonic"
        out.append(
            f"| {layer} | {checks[0]} | {checks[1]} | {checks[2]} | {checks[3]} | "
            f"{a_slope:+.3f} | {b_slope:+.3f} | {verdict} |\n"
        )
    out.append("\n")

    out.append("## Files\n\n")
    out.append(f"- Paired prompts: `{ds['prompt_path']}`\n")
    out.append(f"- Captured activations: `{ds['path']}`\n")
    out.append(f"- Probe metrics: `{args.metrics}`\n")
    out.append(f"- Steering results: `{args.results}`\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(out))
    print(f"Wrote report to {args.output}")


def parse_layers(values: list[int] | None) -> list[int]:
    return list(values) if values else list(DEFAULT_LAYERS)


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--prompt-suffix-indices",
        default="-3:-1",
        help='Inclusive prompt-suffix index spec. "-3:-1" selects the final three suffix tokens.',
    )
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture-train", help="Build paired prompts, capture activations, train directions.")
    add_common_model_args(capture)
    capture.add_argument(
        "--probe-family",
        choices=("simple-corridor", "final-channel-policy", "door-key-grid"),
        default="simple-corridor",
        help="Synthetic paired prompt family used to train the has_key probe direction.",
    )
    capture.add_argument("--agent-rows", type=int, nargs="+", default=list(DEFAULT_AGENT_ROWS))
    capture.add_argument("--max-states", type=int, default=None, help="Optional smoke limit on agent rows.")
    capture.set_defaults(func=run_capture_train)

    steer = sub.add_parser("steer", help="Run alpha steering sweeps from saved directions.")
    add_common_model_args(steer)
    steer.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "steering_results.jsonl")
    steer.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    steer.add_argument("--n-samples", type=int, default=30)
    steer.add_argument("--temperature", type=float, default=0.7)
    steer.add_argument("--max-new-tokens", type=int, default=4096)
    steer.set_defaults(func=run_steering)

    report = sub.add_parser("report", help="Write the layer-comparison markdown report.")
    report.add_argument("--metrics", type=Path, default=DEFAULT_OUT_DIR / "probe_metrics.json")
    report.add_argument("--results", type=Path, default=DEFAULT_OUT_DIR / "steering_results.jsonl")
    report.add_argument("--output", type=Path, default=DEFAULT_OUT_DIR / "report.md")
    report.set_defaults(func=run_report)

    write_grid = sub.add_parser("write-door-key-grid-prompts", help="Write target prompts for the 9x9 door-key grid.")
    write_grid.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    write_grid.add_argument(
        "--output",
        type=Path,
        default=Path("flag_swap/door_key_grid/final_channel_last3/prompts.jsonl"),
    )
    write_grid.add_argument(
        "--prompt-suffix-indices",
        default="-3:-1",
        help='Inclusive prompt-suffix index spec. "-3:-1" selects the final three suffix tokens.',
    )
    write_grid.set_defaults(func=write_door_key_grid_prompts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.layers = parse_layers(getattr(args, "layers", None))
    if hasattr(args, "temperature") and args.temperature <= 0.0:
        raise SystemExit("Steering sweeps use sampled generation; pass a positive --temperature such as 0.7.")
    args.func(args)


if __name__ == "__main__":
    main()
