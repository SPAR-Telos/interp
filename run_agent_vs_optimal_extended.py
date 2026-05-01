"""Extended Cost-IRL: agent action vs BFS-optimal action.

Runs 8 configurations per dataset:
  model_type ∈ {linear, mlp}  ×  token ∈ {pre, post}  ×  label ∈ {agent, optimal}

For each (model, token) pair we train one Model A (agent label) and one
Model B (optimal label) — 4 pairs per dataset.

Datasets:
  - Seed12               (data/trajectories/Seed12, varied starts, hard 9x9)
  - two_path_no_key_T0   (data/trajectories/two_path_no_key_T0, fixed start, easy 7x8)
"""
from __future__ import annotations
import argparse
import json
import re
import pickle
from collections import defaultdict, deque, Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── hyperparameters (verbatim from cost_updated.ipynb) ─────────────────
BETA, LR, N_EPOCHS, REG, SEED, TEST_FRAC = 1.0, 1e-3, 500, 0.01, 42, 0.2
LAYER = "layer_15"
HIDDEN_DIM = 128
DROPOUT = 0.1


@dataclass
class DatasetCfg:
    name: str
    traj_dir: Path
    act_dir: Path
    out_pkl: Path
    out_json: Path


DATASETS = {
    "seed12": DatasetCfg(
        name="seed12",
        traj_dir=Path("/Users/wws/interp/data/trajectories/Seed12"),
        act_dir=Path("/Users/wws/interp/data/activations/Seed12"),
        out_pkl=Path("/Users/wws/interp/results/seed12_agent_vs_optimal_per_visit.pkl"),
        out_json=Path("/Users/wws/interp/results/seed12_agent_vs_optimal_per_visit.json"),
    ),
    "two_path": DatasetCfg(
        name="two_path",
        traj_dir=Path("/Users/wws/interp/data/trajectories/two_path_no_key_T0"),
        act_dir=Path("/Users/wws/interp/data/activations/two_path_no_key_T0"),
        out_pkl=Path("/Users/wws/interp/results/two_path_agent_vs_optimal_per_visit.pkl"),
        out_json=Path("/Users/wws/interp/results/two_path_agent_vs_optimal_per_visit.json"),
    ),
}

ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
A2I = {a: i for i, a in enumerate(ACTIONS)}
DELTAS = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}


def parse_grid(traj_dir):
    with open(traj_dir / "grid_layout.json") as f:
        gl = json.load(f)
    grid_text_lines = gl["grid_text"].rstrip("\n").split("\n")
    grid_layout = []
    for row_str in grid_text_lines[1:]:
        m = re.match(r"^\s*(\d+)\s+", row_str)
        if not m:
            continue
        cells = row_str[m.end():].split()
        grid_layout.append(cells)
    H = len(grid_layout)
    W = len(grid_layout[0])
    walls = {(c, r) for r in range(H) for c in range(W) if grid_layout[r][c] == "#"}
    walkable = {(c, r) for r in range(H) for c in range(W) if grid_layout[r][c] != "#"}
    GOAL = tuple(gl["goal_pos"])
    return H, W, walls, walkable, GOAL


def make_step(walls, walkable, W, H, GOAL):
    def step(pos, action):
        if pos == GOAL:
            return pos
        c, r = pos
        dc, dr = DELTAS[action]
        nc, nr = c + dc, r + dr
        if (nc, nr) in walls or not (0 <= nc < W and 0 <= nr < H):
            return pos
        return (nc, nr)
    return step


def bfs_optimal(walkable, GOAL, step):
    dist = {GOAL: 0}
    q = deque([GOAL])
    while q:
        cur = q.popleft()
        c, r = cur
        for dc, dr in DELTAS.values():
            nxt = (c + dc, r + dr)
            if nxt in walkable and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    optimal_set = {}
    for pos in walkable:
        if pos == GOAL or pos not in dist:
            continue
        best = []
        for a in ACTIONS:
            nxt = step(pos, a)
            if nxt != pos and dist.get(nxt) == dist[pos] - 1:
                best.append(a)
        optimal_set[pos] = best
    return dist, optimal_set


def parse_agent_pos(grid_state):
    for row_str in grid_state:
        m = re.match(r"^\s*(\d+)\s+", row_str)
        if not m:
            continue
        row_idx = int(m.group(1))
        cells = row_str[m.end():].split()
        for col_idx, cell in enumerate(cells):
            if cell == "A":
                return (col_idx, row_idx)
    return None


def load_activation(act_dir, traj_stem, step_id, token_pos):
    folder = act_dir / traj_stem / "openai__gpt-oss-20b" / LAYER / f"step_{step_id}" / token_pos
    if not folder.exists():
        return None
    files = sorted(folder.iterdir(), key=lambda p: int(p.stem))
    if len(files) < 3:
        return None
    vecs = [torch.load(p, map_location="cpu").float() for p in files[:3]]
    return torch.cat(vecs, dim=0).numpy()


def collect_records(cfg, walkable, GOAL, optimal_set, step, token_pos):
    """Walk trajectories and build per-visit records.

    Each (trajectory, step_id, position) visit becomes its own record with
    its own raw 8640-dim phi. *No averaging across visits.*
    """
    records = []
    n_total = skipped_no_phi = skipped_no_pos = skipped_at_goal = skipped_no_optimal = 0
    states_seen = set()

    for tp in sorted(cfg.traj_dir.glob("together_ai*.json")):
        with open(tp) as f:
            tj = json.load(f)
        stem = tp.stem
        for s in tj["steps"]:
            n_total += 1
            sid = s["step_id"]
            action = s.get("agent_action")
            if action not in A2I:
                continue
            pos = parse_agent_pos(s["grid_state"])
            if pos is None:
                skipped_no_pos += 1; continue
            if pos == GOAL:
                skipped_at_goal += 1; continue
            if pos not in optimal_set or not optimal_set[pos]:
                skipped_no_optimal += 1; continue
            phi = load_activation(cfg.act_dir, stem, sid, token_pos)
            if phi is None:
                skipped_no_phi += 1; continue
            state = (pos[0], pos[1], False, False)
            states_seen.add(state)
            opt = optimal_set[pos]
            opt_label = action if action in opt else opt[0]
            records.append({"state": state, "phi": phi, "agent_action": action,
                            "opt_label": opt_label, "opt_set": opt})
    return records, states_seen, {
        "n_total_steps": n_total, "skipped_no_phi": skipped_no_phi,
        "skipped_no_pos": skipped_no_pos, "skipped_at_goal": skipped_at_goal,
        "skipped_no_optimal": skipped_no_optimal,
    }


def build_tensors(records, states_seen, step):
    """Per-visit phi tensors. No state-level averaging anywhere.

    phi_visit[i] is the i-th record's own raw 8640-dim activation at its
    visit (no averaging). The cost model is reformulated to use this
    per-visit feature directly with a *per-action* head — see
    LinearCostIRL_PerVisit / MLPCostIRL_PerVisit below — instead of the
    next-state-cost framing of the original cost_updated.ipynb (which
    inherently requires phi(next_state) for 3 counterfactual actions and
    therefore *some* state-level summary).

    Per-action cost:
        C_θ(s, a) = (θ_a)ᵀ · phi(s)
        P(a | s) = softmax_a(−β · C_θ(s, a))

    Each visit ⇒ one training row with phi = per-visit activation, label
    = the visit's action label.
    """
    state_list = sorted(states_seen)
    s2i = {s: i for i, s in enumerate(state_list)}
    PHI_DIM = records[0]["phi"].shape[0]

    # Drop records whose state has any neighbouring next-state outside the
    # observed state set (kept for parity with the averaged version, although
    # in this per-visit formulation we don't actually look up next-state
    # features; we still drop these records to keep the comparison apples-to-
    # apples with the averaged runs).
    kept = []
    dropped = 0
    for rec in records:
        c, r, _, _ = rec["state"]
        ok = True
        for a in ACTIONS:
            nxt_pos = step((c, r), a)
            if (nxt_pos[0], nxt_pos[1], False, False) not in s2i:
                ok = False; break
        if not ok:
            dropped += 1; continue
        kept.append(rec)

    phi_visit = torch.tensor(np.stack([rec["phi"] for rec in kept]),
                             dtype=torch.float32)
    # Z-normalise using stats from the per-visit matrix only.
    mean = phi_visit.mean(0, keepdim=True)
    std = phi_visit.std(0, keepdim=True) + 1e-8
    phi_visit_norm = (phi_visit - mean) / std

    agent_labels = torch.tensor([A2I[rec["agent_action"]] for rec in kept],
                                dtype=torch.long)
    opt_labels = torch.tensor([A2I[rec["opt_label"]] for rec in kept],
                              dtype=torch.long)
    opt_sets = [[A2I[a] for a in rec["opt_set"]] for rec in kept]

    return (phi_visit_norm, agent_labels, opt_labels, opt_sets,
            PHI_DIM, len(state_list), dropped)


# ── models (per-visit, per-action head) ────────────────────────────────
class LinearCostIRL(nn.Module):
    """Per-visit linear cost-IRL.

    C_θ(s, a) = (θ_a)ᵀ · phi(s)
    P(a | s) = softmax(−β · C_θ(s, a))

    Equivalent to multinomial logistic regression with weight matrix
    W = −β · θ ∈ ℝ^{4 × phi_dim}; framed as cost-IRL for continuity with
    the averaged version of the experiment.
    """
    def __init__(self, phi_dim, n_actions=4, beta=1.0):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_actions, phi_dim))
        self.beta = beta

    def forward(self, phi):                         # phi: (N, phi_dim)
        costs = phi @ self.theta.T                  # (N, n_actions)
        return torch.log_softmax(-self.beta * costs, dim=-1)


class MLPCostIRL(nn.Module):
    """Per-visit MLP cost-IRL — 8640 → 128 → 128 → 4 logits."""
    def __init__(self, phi_dim, hidden_dim=128, n_actions=4, beta=1.0, dropout=0.1):
        super().__init__()
        self.beta = beta
        self.net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions),
        )
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.1)
        nn.init.zeros_(self.net[6].weight)

    def forward(self, phi):
        costs = self.net(phi)                       # (N, n_actions)
        return torch.log_softmax(-self.beta * costs, dim=-1)


def train_one(model_type, phi_dim, phi_visit, labels, train_idx, test_idx, name):
    torch.manual_seed(SEED)
    if model_type == "linear":
        model = LinearCostIRL(phi_dim, beta=BETA)
        opt = optim.Adam([model.theta], lr=LR)
    else:
        model = MLPCostIRL(phi_dim, hidden_dim=HIDDEN_DIM, beta=BETA, dropout=DROPOUT)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=REG)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    phi_tr, y_tr = phi_visit[train_idx], labels[train_idx]
    phi_te, y_te = phi_visit[test_idx], labels[test_idx]
    best_te_ll = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(N_EPOCHS):
        model.train()
        opt.zero_grad()
        logp = model(phi_tr)
        loss = nn.functional.nll_loss(logp, y_tr)
        if model_type == "linear":
            loss = loss + REG * (model.theta ** 2).sum()
        loss.backward()
        if model_type == "mlp":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sch.step()
        with torch.no_grad():
            logp_te = model(phi_te)
            te_ll = -nn.functional.nll_loss(logp_te, y_te).item()
        if te_ll > best_te_ll:
            best_te_ll = te_ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def eval_model(model, labels, phi_visit, opt_sets, idx):
    model.eval()
    with torch.no_grad():
        logp = model(phi_visit[idx])
        preds = logp.argmax(-1)
        ll = -nn.functional.nll_loss(logp, labels[idx]).item()
        acc = (preds == labels[idx]).float().mean().item()
        opt_sets_idx = [opt_sets[int(i)] for i in idx.tolist()]
        in_opt = sum(int(int(p) in s) for p, s in zip(preds, opt_sets_idx))
    per_action = {}
    for a, ai in A2I.items():
        mask = labels[idx] == ai
        n = int(mask.sum())
        pa = (preds[mask] == ai).float().mean().item() if n else 0.0
        per_action[a] = (pa, n)
    return {"ll": ll, "acc": acc, "in_opt_acc": in_opt / len(idx),
            "per_action": per_action}


def run_dataset(dataset_key, datasets):
    cfg = datasets[dataset_key]
    print(f"\n{'='*70}\n{cfg.name.upper()}\n{'='*70}")
    H, W, walls, walkable, GOAL = parse_grid(cfg.traj_dir)
    step = make_step(walls, walkable, W, H, GOAL)
    dist, optimal_set = bfs_optimal(walkable, GOAL, step)
    print(f"Grid {W}x{H}, walkable={len(walkable)}, goal={GOAL}, "
          f"BFS-reached={len(dist)}/{len(walkable)}")

    all_results = {}
    # For each token category we re-collect activations (different files).
    for token_pos in ["prompt_suffix", "output"]:
        token_label = "pre" if token_pos == "prompt_suffix" else "post"
        print(f"\n--- token = {token_label} ({token_pos}) ---")
        torch.manual_seed(SEED); np.random.seed(SEED)
        records, states_seen, drop_stats = collect_records(
            cfg, walkable, GOAL, optimal_set, step, token_pos)
        phi_visit, agent_labels, opt_labels, opt_sets, PHI_DIM, n_states, drop_t = \
            build_tensors(records, states_seen, step)
        N = len(agent_labels)
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(N)
        n_test = int(N * TEST_FRAC)
        test_idx = torch.tensor(perm[:n_test], dtype=torch.long)
        train_idx = torch.tensor(perm[n_test:], dtype=torch.long)
        agent_is_optimal = sum(int(a.item() in s) for a, s in zip(agent_labels, opt_sets))
        agent_dist = Counter(int(a) for a in agent_labels.tolist())
        opt_dist = Counter(int(a) for a in opt_labels.tolist())
        meta = {
            "n_records": int(N), "n_unique_states": int(n_states),
            "phi_dim": int(PHI_DIM), "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "drop_filter": drop_stats, "dropped_no_next_phi": int(drop_t),
            "agent_action_dist": {ACTIONS[i]: int(agent_dist[i]) for i in range(4)},
            "optimal_action_dist": {ACTIONS[i]: int(opt_dist[i]) for i in range(4)},
            "agent_is_optimal_frac": agent_is_optimal / N,
            "majority_baseline_agent": max(agent_dist.values()) / N,
            "majority_baseline_opt": max(opt_dist.values()) / N,
        }
        print(f"  N={N} records, {n_states} states, phi_dim={PHI_DIM}, "
              f"agent-optimal={agent_is_optimal/N:.3f}")

        for model_type in ["linear", "mlp"]:
            for label_kind, labels in [("agent", agent_labels), ("optimal", opt_labels)]:
                key = f"{model_type}__{token_label}__{label_kind}"
                print(f"  training [{key}] ...", end=" ", flush=True)
                model = train_one(model_type, PHI_DIM, phi_visit, labels,
                                  train_idx, test_idx, key)
                tr = eval_model(model, labels, phi_visit, opt_sets, train_idx)
                te = eval_model(model, labels, phi_visit, opt_sets, test_idx)
                # cross: model trained on `labels` scored against the OTHER labels
                other = opt_labels if label_kind == "agent" else agent_labels
                cross = eval_model(model, other, phi_visit, opt_sets, test_idx)
                all_results[key] = {"train": tr, "test": te, "cross": cross}
                print(f"test_acc={te['acc']:.3f}  in_opt={te['in_opt_acc']:.3f}  "
                      f"ll={te['ll']:.4f}")
        all_results[f"meta__{token_label}"] = meta

    cfg.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_pkl, "wb") as f:
        pickle.dump(all_results, f)
    with open(cfg.out_json, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved {cfg.out_pkl}")
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["seed12", "two_path", "both"], default="both")
    args = ap.parse_args()
    keys = ["seed12", "two_path"] if args.dataset == "both" else [args.dataset]
    for k in keys:
        run_dataset(k, DATASETS)


if __name__ == "__main__":
    main()
