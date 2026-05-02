"""Prepare targets and donor activations for the variant-transplant
intervention experiment (Direction 2a).

Idea: at runtime on a `K0D0` agent at cell (c, r), we'll replace the
layer-15 prompt-suffix activations with the corresponding `K1D1`
activations (cross-transplant) — and as a control, with K0D0
activations from a different visit (self-transplant). If
cross-transplant flips the action toward the K1D1-optimal direction
while self-transplant doesn't, the variant-encoding at layer-15
prompt-suffix is causally read out by action selection.

This script (Mac CPU only) iterates the fourroom_episode_sweep_T0
trajectories, finds cells where K0D0 / K1D1 optimal actions differ,
builds donor activations (mean or single-visit) for each (cell,
variant), and saves:

  transplant_intervention/donors.pt
    dict[(col, row, variant_str)] -> (3, 2880) bfloat16 tensor

  transplant_intervention/targets.jsonl
    one line per (target_visit, condition) — metadata + full prompt
    token IDs + the 3 absolute positions to intervene at + the
    expected K0D0 / K1D1 optimal action sets at the target cell.

The GPU-side runner (`run_transplant_intervention.py`) reads these
two files, hooks layer 15 of GPT-OSS-20B, and overwrites the
indicated positions with donor[(cell, variant)].

Mac CPU only.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Reuse the upstream's grid + transition + BFS code.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_agent_vs_optimal_extended import (  # type: ignore
    parse_grid, make_step_state, bfs_optimal, parse_agent_pos, ACTIONS, A2I,
)


VARIANT_FLAGS = {
    "K0D0": (False, False),
    "K1D0": (True, False),
    "K1D1": (True, True),
}
VARIANT_RE = re.compile(r"(K\d+D\d+)")
LAYER = "layer_15"
TOKEN_POS = "prompt_suffix"
N_DONOR_TOKENS = 3   # last 3 of the 25 prompt-suffix tokens
HIDDEN_DIM = 2880


def variant_of(stem: str) -> str:
    m = VARIANT_RE.search(stem)
    if not m:
        raise ValueError(f"cannot parse variant from filename: {stem}")
    return m.group(1)


def load_step_activations(act_dir: Path, traj_stem: str, step_id: int):
    """Load the 3 saved per-token activations at layer 15 prompt_suffix.

    Returns a tensor of shape (3, hidden_dim) in float32, or None if
    the directory is missing.
    """
    folder = act_dir / traj_stem / "openai__gpt-oss-20b" / LAYER / f"step_{step_id}" / TOKEN_POS
    if not folder.exists():
        return None
    files = sorted(folder.iterdir(), key=lambda p: int(p.stem))
    if len(files) < N_DONOR_TOKENS:
        return None
    vecs = [torch.load(p, map_location="cpu", weights_only=True).float() for p in files[:N_DONOR_TOKENS]]
    return torch.stack(vecs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path,
                    default=Path("/Users/wws/interp/data/trajectories/fourroom_episode_sweep_T0"))
    ap.add_argument("--act-dir", type=Path,
                    default=Path("/Users/wws/interp/data/activations/fourroom_episode_sweep_T0"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Users/wws/interp/transplant_intervention"))
    ap.add_argument("--target-variant", choices=["K0D0", "K1D0", "K1D1"], default="K0D0",
                    help="Which variant the test agent is.")
    ap.add_argument("--donor-variant", choices=["K0D0", "K1D0", "K1D1"], default="K1D1",
                    help="Which variant the cross-transplant donor comes from.")
    ap.add_argument("--donor-mode", choices=["mean", "single"], default="mean",
                    help="Aggregate donor visits to the cell, or pick a single (deterministic) one.")
    ap.add_argument("--max-targets-per-cell", type=int, default=4,
                    help="Cap target visits per cell to keep the experiment small. 0 = unlimited.")
    args = ap.parse_args()

    if args.target_variant == args.donor_variant:
        raise SystemExit("--target-variant and --donor-variant must differ")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Grid + state-space BFS ─────────────────────────────────────────
    H, W, walls, walkable, GOAL, KEY, DOOR = parse_grid(args.traj_dir)
    step_state = make_step_state(walls, W, H, GOAL, KEY, DOOR)
    dist, optimal_set = bfs_optimal(walkable, GOAL, step_state)
    print(f"Grid {W}x{H}, walkable={len(walkable)}, GOAL={GOAL}, KEY={KEY}, DOOR={DOOR}")

    # ── Walk trajectories: index visits by (cell, variant) ──────────────
    # cell_index[(col, row)][variant] = list of (traj_stem, step_id, agent_action)
    cell_index: dict[tuple[int, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    # For each (traj_stem, step_id) we'll need the full prompt token IDs,
    # the n_grid for this step (since suffix's absolute position depends
    # on it), and the agent action.
    visit_meta: dict[tuple[str, int], dict] = {}

    n_traj = 0
    for tp in sorted(args.traj_dir.glob("together_ai*.json")):
        n_traj += 1
        with open(tp) as f:
            tj = json.load(f)
        stem = tp.stem
        variant = variant_of(stem)
        prefix_tokens = tj["prompt"]["prompt_prefix_tokens"]
        suffix_tokens = tj["prompt"]["prompt_suffix_tokens"]
        n_prefix = len(prefix_tokens)
        n_suffix = len(suffix_tokens)
        for s in tj["steps"]:
            action = s.get("agent_action")
            if action not in A2I:
                continue
            pos = parse_agent_pos(s["grid_state"])
            if pos is None or pos == GOAL:
                continue
            n_grid = len(s["grid_state_tokens"])
            full_token_ids = (
                [t["token_id"] for t in prefix_tokens]
                + [t["token_id"] for t in s["grid_state_tokens"]]
                + [t["token_id"] for t in suffix_tokens]
            )
            # Last 3 prompt-suffix tokens, absolute positions in the full prompt.
            prompt_len = n_prefix + n_grid + n_suffix
            assert prompt_len == len(full_token_ids)
            suffix_positions = [prompt_len - 3, prompt_len - 2, prompt_len - 1]
            cell_index[pos][variant].append((stem, s["step_id"], action))
            visit_meta[(stem, s["step_id"])] = {
                "variant": variant,
                "pos": list(pos),
                "agent_action": action,
                "full_token_ids": full_token_ids,
                "suffix_positions": suffix_positions,
                "n_prefix": n_prefix,
                "n_grid": n_grid,
                "n_suffix": n_suffix,
                "prompt_len": prompt_len,
            }

    print(f"Loaded {n_traj} trajectories, {len(visit_meta)} valid visits across "
          f"{len(cell_index)} cells")

    # ── Find candidate cells: target+donor variants both present, optimal differs ──
    tgt = args.target_variant
    don = args.donor_variant
    tgt_flags = VARIANT_FLAGS[tgt]
    don_flags = VARIANT_FLAGS[don]

    candidate_cells = []
    for cell, by_variant in sorted(cell_index.items()):
        if tgt not in by_variant or don not in by_variant:
            continue
        if len(by_variant[tgt]) < 2:
            # Need ≥2 K0D0 visits for self-transplant control (donor mean
            # excluding the target visit).
            continue
        opt_tgt = optimal_set.get((cell[0], cell[1], *tgt_flags), [])
        opt_don = optimal_set.get((cell[0], cell[1], *don_flags), [])
        if not opt_tgt or not opt_don:
            continue
        if set(opt_tgt) == set(opt_don):
            # Same optimal — uninformative for the transplant test.
            continue
        candidate_cells.append((cell, opt_tgt, opt_don))

    print(f"\nCandidate cells (target={tgt}, donor={don}, ≥2 target visits, optimal differs): "
          f"{len(candidate_cells)}")
    for cell, opt_tgt, opt_don in candidate_cells:
        print(f"  {cell}: {tgt} optimal={opt_tgt}  {don} optimal={opt_don}  "
              f"({len(cell_index[cell][tgt])} {tgt} visits, {len(cell_index[cell][don])} {don} visits)")

    # ── Build donor activations per (cell, variant) ────────────────────
    # We need:
    #   donors[(cell, donor_variant)] = mean of donor-variant activations at cell
    #   donors[(cell, target_variant, target_visit_id)] = mean of target-variant
    #     activations at cell EXCLUDING the target visit (for self-transplant)
    #
    # For the donor (cross) mean: same across all targets at this cell.
    # For the self-transplant: depends on which visit is held out.
    # To keep donors.pt manageable, we'll save per-cell per-variant means
    # *without* exclusion, and also save per-cell per-variant *sums* and
    # *counts* so the GPU script can subtract the held-out visit on the
    # fly. Cleaner: just precompute per-target self-transplant donors and
    # store them keyed by visit id.
    donors: dict = {}
    sum_act: dict[tuple[tuple[int, int], str], torch.Tensor] = {}
    cnt_act: dict[tuple[tuple[int, int], str], int] = {}
    per_visit_act: dict[tuple[str, int], torch.Tensor] = {}  # for self-transplant exclusion

    print(f"\nBuilding donor activations (mode={args.donor_mode}) ...")
    for cell, _, _ in candidate_cells:
        for variant in (tgt, don):
            visits = cell_index[cell][variant]
            visit_acts = []
            for stem, sid, _ in visits:
                act = load_step_activations(args.act_dir, stem, sid)
                if act is None:
                    print(f"  WARN: no activation for {stem} step {sid}; skipping")
                    continue
                visit_acts.append(act)
                per_visit_act[(stem, sid)] = act
                key_sum = (cell, variant)
                if key_sum in sum_act:
                    sum_act[key_sum] = sum_act[key_sum] + act
                    cnt_act[key_sum] += 1
                else:
                    sum_act[key_sum] = act.clone()
                    cnt_act[key_sum] = 1
            if not visit_acts:
                continue
            if args.donor_mode == "mean":
                donors[(cell[0], cell[1], variant)] = torch.stack(visit_acts, dim=0).mean(dim=0)
            else:
                # Deterministic single-visit donor: sort by (stem, sid) and pick the first.
                ordered = sorted(zip(visits, visit_acts), key=lambda x: (x[0][0], x[0][1]))
                donors[(cell[0], cell[1], variant)] = ordered[0][1]

    # Cast to bfloat16 to match the saved-activation dtype and reduce file size.
    for k in list(donors.keys()):
        donors[k] = donors[k].to(torch.bfloat16)

    print(f"Built {len(donors)} donor entries.")

    donors_path = args.out_dir / "donors.pt"
    torch.save(donors, donors_path)
    print(f"Saved donors to {donors_path}")

    # ── Build target rows ───────────────────────────────────────────────
    targets: list[dict] = []
    n_skipped_no_self_donor = 0
    for cell, opt_tgt, opt_don in candidate_cells:
        target_visits = cell_index[cell][tgt]
        if args.max_targets_per_cell and len(target_visits) > args.max_targets_per_cell:
            target_visits = target_visits[: args.max_targets_per_cell]
        for stem, sid, action in target_visits:
            meta = visit_meta[(stem, sid)]
            # Self-transplant donor: mean of target-variant visits at this cell
            # EXCLUDING the target visit itself.
            self_n = cnt_act.get((cell, tgt), 0)
            if self_n <= 1 or (stem, sid) not in per_visit_act:
                n_skipped_no_self_donor += 1
                continue
            tgt_sum = sum_act[(cell, tgt)] - per_visit_act[(stem, sid)]
            self_donor = (tgt_sum / (self_n - 1)).to(torch.bfloat16)
            self_donor_key = f"self::{stem}::{sid}"
            donors[(cell[0], cell[1], self_donor_key)] = self_donor

            targets.append({
                "target_traj": stem,
                "target_step_id": int(sid),
                "target_pos": list(cell),
                "target_variant": tgt,
                "target_agent_action": action,
                "donor_variant_cross": don,
                "donor_key_cross": [int(cell[0]), int(cell[1]), don],
                "donor_key_self":  [int(cell[0]), int(cell[1]), self_donor_key],
                "opt_target_variant": opt_tgt,
                "opt_donor_variant": opt_don,
                "n_grid_tokens": int(meta["n_grid"]),
                "n_prefix_tokens": int(meta["n_prefix"]),
                "n_suffix_tokens": int(meta["n_suffix"]),
                "prompt_len": int(meta["prompt_len"]),
                "suffix_positions": list(map(int, meta["suffix_positions"])),
                "full_token_ids": list(map(int, meta["full_token_ids"])),
            })

    # Re-save donors (now includes self-transplant per-target donors).
    torch.save(donors, donors_path)
    print(f"Re-saved donors with {len(donors)} entries (including per-target self-transplant donors)")

    targets_path = args.out_dir / "targets.jsonl"
    with open(targets_path, "w") as f:
        for t in targets:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(targets)} targets to {targets_path}  (skipped {n_skipped_no_self_donor} for missing self-transplant donor)")

    # Summary
    by_cell = defaultdict(int)
    for t in targets:
        by_cell[tuple(t["target_pos"])] += 1
    print("\nTarget distribution by cell:")
    for cell, n in sorted(by_cell.items()):
        print(f"  {cell}: {n} targets")


if __name__ == "__main__":
    main()
