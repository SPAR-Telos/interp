"""Prepare the two prompts and the two layer-15 prompt-suffix
activation tensors for the flag-swap experiment (Direction 2c,
clean activation patch on the carrying_key flag).

Picks one K0D0 visit and one K1D0 visit at cell (1, 5) — the cell
directly below the locked door at (1, 4). The K0D0 visit has the
agent without the key (BFS-optimal: head back for the key); the
K1D0 visit has the agent holding the key (BFS-optimal: UP through
the door, which auto-opens).

Outputs:
  flag_swap/prompts.jsonl   — two rows: a (K0D0) and b (K1D0)
  flag_swap/act_a.pt        — (3, 2880) bfloat16: layer-15 last 3
                              prompt-suffix activations from prompt a
  flag_swap/act_b.pt        — same, from prompt b

Mac CPU only.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch


VARIANT_RE = re.compile(r"(K\d+D\d+)")
LAYER = "layer_15"
TOKEN_POS = "prompt_suffix"
N_DONOR_TOKENS = 3
HIDDEN_DIM = 2880
TARGET_CELL = (1, 5)


def parse_pos(grid_state):
    for row in grid_state:
        m = re.match(r"^\s*(\d+)\s+", row)
        if not m:
            continue
        r = int(m.group(1))
        cells = row[m.end():].split()
        for c, x in enumerate(cells):
            if x == "A":
                return (c, r)
    return None


def variant_of(stem):
    m = VARIANT_RE.search(stem)
    return m.group(1) if m else None


def load_step_activations(act_dir: Path, traj_stem: str, step_id: int):
    folder = act_dir / traj_stem / "openai__gpt-oss-20b" / LAYER / f"step_{step_id}" / TOKEN_POS
    if not folder.exists():
        return None
    files = sorted(folder.iterdir(), key=lambda p: int(p.stem))
    if len(files) < N_DONOR_TOKENS:
        return None
    vecs = [
        torch.load(p, map_location="cpu", weights_only=True).float()
        for p in files[:N_DONOR_TOKENS]
    ]
    return torch.stack(vecs, dim=0)


def find_visit(traj_dir: Path, target_variant: str, target_cell: tuple[int, int]):
    """Find the lexicographically-first (traj_stem, step_id) where the
    target_variant trajectory has the agent at target_cell."""
    candidates: list[tuple[str, int, str]] = []
    for tp in sorted(traj_dir.glob("together_ai*.json")):
        stem = tp.stem
        if variant_of(stem) != target_variant:
            continue
        with open(tp) as f:
            tj = json.load(f)
        for s in tj["steps"]:
            if parse_pos(s["grid_state"]) == target_cell:
                candidates.append((stem, int(s["step_id"]), s.get("agent_action"), tj))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0]


def build_prompt_row(label: str, variant: str, traj_stem: str, step_id: int,
                     agent_action: str, tj: dict) -> dict:
    prefix_tokens = tj["prompt"]["prompt_prefix_tokens"]
    suffix_tokens = tj["prompt"]["prompt_suffix_tokens"]
    n_prefix = len(prefix_tokens)
    n_suffix = len(suffix_tokens)
    s = next(s for s in tj["steps"] if s["step_id"] == step_id)
    grid_tokens = s["grid_state_tokens"]
    n_grid = len(grid_tokens)
    full_token_ids = (
        [t["token_id"] for t in prefix_tokens]
        + [t["token_id"] for t in grid_tokens]
        + [t["token_id"] for t in suffix_tokens]
    )
    prompt_len = n_prefix + n_grid + n_suffix
    assert prompt_len == len(full_token_ids)
    suffix_positions = [prompt_len - 3, prompt_len - 2, prompt_len - 1]
    return {
        "label": label,
        "variant": variant,
        "traj": traj_stem,
        "step_id": int(step_id),
        "pos": list(TARGET_CELL),
        "agent_action_recorded": agent_action,
        "n_prefix": n_prefix,
        "n_grid": n_grid,
        "n_suffix": n_suffix,
        "prompt_len": prompt_len,
        "suffix_positions": suffix_positions,
        "full_token_ids": full_token_ids,
    }


def diff_suffix_text(row_a: dict, row_b: dict) -> str:
    """Show only the tokens after the grid block, since prefix is
    constant across trajectories. Pretty-print the differing region."""
    out = []
    pa, pb = row_a["full_token_ids"], row_b["full_token_ids"]
    suffix_start_a = row_a["n_prefix"] + row_a["n_grid"]
    suffix_start_b = row_b["n_prefix"] + row_b["n_grid"]
    out.append(f"  prompt a length: {row_a['prompt_len']}  "
               f"(prefix={row_a['n_prefix']} + grid={row_a['n_grid']} + suffix={row_a['n_suffix']})")
    out.append(f"  prompt b length: {row_b['prompt_len']}  "
               f"(prefix={row_b['n_prefix']} + grid={row_b['n_grid']} + suffix={row_b['n_suffix']})")
    suffix_a = pa[suffix_start_a:]
    suffix_b = pb[suffix_start_b:]
    out.append(f"  suffix-region token IDs match: {suffix_a == suffix_b}")
    if suffix_a != suffix_b:
        for i, (ta, tb) in enumerate(zip(suffix_a, suffix_b)):
            if ta != tb:
                out.append(f"    suffix offset {i}: a={ta}  vs  b={tb}")
    # Find token IDs in the *grid* region that differ. Prompt grids may differ
    # in the K cell; suffix should differ at the True/False slot only.
    grid_a = pa[row_a["n_prefix"]: row_a["n_prefix"] + row_a["n_grid"]]
    grid_b = pb[row_b["n_prefix"]: row_b["n_prefix"] + row_b["n_grid"]]
    if len(grid_a) == len(grid_b):
        ndiff = sum(1 for x, y in zip(grid_a, grid_b) if x != y)
        out.append(f"  grid region: same length ({len(grid_a)}), {ndiff} differing token positions")
    else:
        out.append(f"  grid region length differs: a={len(grid_a)}  b={len(grid_b)}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path,
                    default=Path("/Users/wws/interp/data/trajectories/fourroom_episode_sweep_T0"))
    ap.add_argument("--act-dir", type=Path,
                    default=Path("/Users/wws/interp/data/activations/fourroom_episode_sweep_T0"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Users/wws/interp/flag_swap"))
    ap.add_argument("--target-cell", type=int, nargs=2, default=list(TARGET_CELL))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cell = tuple(args.target_cell)

    print(f"Looking for K0D0 and K1D0 visits at cell {cell} ...")
    a = find_visit(args.traj_dir, "K0D0", cell)
    b = find_visit(args.traj_dir, "K1D0", cell)
    if a is None:
        raise SystemExit(f"no K0D0 visit found at {cell}")
    if b is None:
        raise SystemExit(f"no K1D0 visit found at {cell}")
    a_stem, a_step, a_action, a_tj = a
    b_stem, b_step, b_action, b_tj = b
    print(f"  prompt a (K0D0): {a_stem} step {a_step}  recorded action = {a_action}")
    print(f"  prompt b (K1D0): {b_stem} step {b_step}  recorded action = {b_action}")

    row_a = build_prompt_row("a", "K0D0", a_stem, a_step, a_action, a_tj)
    row_b = build_prompt_row("b", "K1D0", b_stem, b_step, b_action, b_tj)

    print("\nPrompt diff (decoded-text-free; token-level):")
    print(diff_suffix_text(row_a, row_b))

    # Activations
    act_a = load_step_activations(args.act_dir, a_stem, a_step)
    act_b = load_step_activations(args.act_dir, b_stem, b_step)
    if act_a is None:
        raise SystemExit(f"missing activations for prompt a: {a_stem}/step_{a_step}")
    if act_b is None:
        raise SystemExit(f"missing activations for prompt b: {b_stem}/step_{b_step}")
    act_a_bf = act_a.to(torch.bfloat16)
    act_b_bf = act_b.to(torch.bfloat16)
    print(f"\nact_a shape={tuple(act_a_bf.shape)} dtype={act_a_bf.dtype} "
          f"norm-per-token={[round(a.float().norm().item(), 2) for a in act_a_bf]}")
    print(f"act_b shape={tuple(act_b_bf.shape)} dtype={act_b_bf.dtype} "
          f"norm-per-token={[round(b.float().norm().item(), 2) for b in act_b_bf]}")
    # Distance between act_a and act_b
    diff = (act_a.float() - act_b.float()).norm(dim=-1)
    print(f"||act_a − act_b|| per token = {[round(d.item(), 2) for d in diff]}  "
          f"(non-zero ⇒ activations carry the variant signal)")

    # Save
    prompts_path = args.out_dir / "prompts.jsonl"
    with open(prompts_path, "w") as f:
        for row in (row_a, row_b):
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {prompts_path}")
    torch.save(act_a_bf, args.out_dir / "act_a.pt")
    torch.save(act_b_bf, args.out_dir / "act_b.pt")
    print(f"Wrote {args.out_dir / 'act_a.pt'}  and  {args.out_dir / 'act_b.pt'}")


if __name__ == "__main__":
    main()
