"""Cost-IRL: agent action vs BFS-optimal action.

Algorithm (next-state averaged-φ form, ported from `cost_updated.ipynb`):

  - φ(s) = mean of activations across all visits to state s (one φ per state)
  - Single θ ∈ ℝ^{phi_dim};  cost(s) = θᵀ · φ(s)
  - Policy:  P(a | s) = softmax(−β · cost(f(s, a)))
  - For each (s_t, a_t) transition we look up φ(f(s, a)) for all 4 actions;
    transitions whose next state was never observed are dropped.

Runs 8 configurations per dataset:
  model_type ∈ {linear, mlp}  ×  token ∈ {pre, post}  ×  label ∈ {agent, optimal}

Outputs `<dataset>_agent_vs_optimal_extended.{pkl,json}` under results/.

Datasets:
  - Seed12               (data/trajectories/Seed12)
  - two_path_no_key_T0   (data/trajectories/two_path_no_key_T0)

Both datasets have constant (carrying_key, door_open) across all steps,
so the transition function is plain 4-direction movement that stops at
walls and the goal.
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
        out_pkl=Path("/Users/wws/interp/results/seed12_agent_vs_optimal_extended.pkl"),
        out_json=Path("/Users/wws/interp/results/seed12_agent_vs_optimal_extended.json"),
    ),
    "two_path": DatasetCfg(
        name="two_path",
        traj_dir=Path("/Users/wws/interp/data/trajectories/two_path_no_key_T0"),
        act_dir=Path("/Users/wws/interp/data/activations/two_path_no_key_T0"),
        out_pkl=Path("/Users/wws/interp/results/two_path_agent_vs_optimal_extended.pkl"),
        out_json=Path("/Users/wws/interp/results/two_path_agent_vs_optimal_extended.json"),
    ),
}

ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
A2I = {a: i for i, a in enumerate(ACTIONS)}
DELTAS = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}


def parse_grid(traj_dir: Path):
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
    KEY = next(((c, r) for r in range(H) for c in range(W) if grid_layout[r][c] == "K"), None)
    DOOR = next(((c, r) for r in range(H) for c in range(W) if grid_layout[r][c] == "D"), None)
    return H, W, walls, walkable, GOAL, KEY, DOOR


def make_step(walls, W, H, GOAL):
    """Position-only step, used for BFS / optimal-action-set computation."""
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


def make_step_state(walls, W, H, GOAL, KEY, DOOR):
    """Full-state transition: (col, row, has_key, door_open) → next state.

    - Walls/out-of-bounds: stay.
    - Door cell is treated as walkable (matches `grid_text` parsing and
      Seed12's env: agents in `door_open` variants traverse it freely;
      agents in `standard`/`has_key` variants visibly cannot, but the
      action set is movement-only — no `toggle` — so we never *update*
      `door_open` here either way).
    - Key pickup: walking onto the key cell with `has_key=False` flips
      `has_key` to True. This is the only flag transition observable
      in Seed12.
    - The goal is terminal.
    """
    def f(state, action):
        col, row, has_key, door_open = state
        if (col, row) == GOAL:
            return state
        dc, dr = DELTAS[action]
        nc, nr = col + dc, row + dr
        if not (0 <= nc < W and 0 <= nr < H):
            return state
        if (nc, nr) in walls:
            return state
        # Locked door blocks unless the agent has the key (auto-opens).
        new_door_open = door_open
        if DOOR is not None and (nc, nr) == DOOR and not door_open:
            if not has_key:
                return state
            new_door_open = True
        new_has_key = has_key
        if KEY is not None and (nc, nr) == KEY and not has_key:
            new_has_key = True
        return (nc, nr, new_has_key, new_door_open)
    return f


def bfs_optimal(walkable, GOAL, step_state):
    """BFS in the full ``(col, row, has_key, door_open)`` state space.

    Distances respect the env dynamics encoded in ``step_state``: an
    agent without the key on the wrong side of a locked door must
    detour through the key cell first, so its distance to the goal is
    larger than the same position with the key already collected.
    Position-only BFS would mistakenly give them the same distance.
    """
    states = []
    for (c, r) in walkable:
        for has_key in (False, True):
            for door_open in (False, True):
                states.append((c, r, has_key, door_open))

    incoming = defaultdict(list)
    for s in states:
        for a in ACTIONS:
            ns = step_state(s, a)
            if ns != s:
                incoming[ns].append(s)

    dist = {}
    q = deque()
    for s in states:
        if (s[0], s[1]) == GOAL:
            dist[s] = 0
            q.append(s)
    while q:
        cur = q.popleft()
        for s in incoming[cur]:
            if s not in dist:
                dist[s] = dist[cur] + 1
                q.append(s)

    optimal_set = {}
    for s in states:
        if (s[0], s[1]) == GOAL or s not in dist:
            continue
        best = []
        for a in ACTIONS:
            ns = step_state(s, a)
            if ns != s and dist.get(ns) == dist[s] - 1:
                best.append(a)
        optimal_set[s] = best
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


def collect_records(cfg, GOAL, optimal_set, token_pos):
    """Walk trajectories. Each (traj, step_id) becomes one record.

    state = (col, row, carrying_key, door_open), read verbatim from
    each trajectory step. The optimal-action set comes from BFS in the
    full state space (`optimal_set` is keyed by the 4-tuple, not by
    position).
    """
    records = []
    skipped = Counter()
    n_total = 0
    for tp in sorted(cfg.traj_dir.glob("together_ai*.json")):
        with open(tp) as f:
            tj = json.load(f)
        stem = tp.stem
        for s in tj["steps"]:
            n_total += 1
            action = s.get("agent_action")
            if action not in A2I:
                skipped["bad_action"] += 1; continue
            pos = parse_agent_pos(s["grid_state"])
            if pos is None:
                skipped["no_pos"] += 1; continue
            if pos == GOAL:
                skipped["at_goal"] += 1; continue
            has_key = bool(s.get("carrying_key", False))
            door_open = bool(s.get("door_open", False))
            state = (pos[0], pos[1], has_key, door_open)
            if state not in optimal_set or not optimal_set[state]:
                skipped["no_optimal"] += 1; continue
            phi = load_activation(cfg.act_dir, stem, s["step_id"], token_pos)
            if phi is None:
                skipped["no_phi"] += 1; continue
            opt = optimal_set[state]
            opt_label = action if action in opt else opt[0]
            records.append({
                "state": state, "phi": phi,
                "agent_action": action, "opt_label": opt_label,
                "opt_set": opt,
            })
    return records, dict(skipped), n_total


def build_phi_table(records):
    """Average φ across all visits to the same state."""
    state_phis = defaultdict(list)
    for rec in records:
        state_phis[rec["state"]].append(rec["phi"])
    return {s: np.stack(phis).mean(axis=0) for s, phis in state_phis.items()}


def build_dataset(records, phi_table, step_state):
    """Build the next-state-indexed training set.

    Uses the full-state transition `step_state` (key pickup +
    locked-door auto-open with key) for all 4 actions per record. For
    the action the agent took, we override the rule-based flags with
    the trajectory's recorded step-(t+1) flags — the rules can diverge
    from reality in some Seed12 variants where the door cell is
    rendered `_` rather than `D`. Counterfactual actions only have
    the rule-based prediction.
    """
    state_list = sorted(phi_table.keys())
    s2i = {s: i for i, s in enumerate(state_list)}
    phi_matrix = torch.tensor(np.stack([phi_table[s] for s in state_list]),
                              dtype=torch.float32)
    phi_mean = phi_matrix.mean(0, keepdim=True)
    phi_std = phi_matrix.std(0, keepdim=True) + 1e-8
    phi_matrix_norm = (phi_matrix - phi_mean) / phi_std

    next_idx, agent_labels, opt_labels, opt_sets = [], [], [], []
    dropped = 0
    for rec in records:
        cur = rec["state"]   # (col, row, has_key, door_open)
        rows, valid = [], True
        for a in ACTIONS:
            ns = step_state(cur, a)
            if ns not in s2i:
                valid = False; break
            rows.append(s2i[ns])
        if not valid:
            dropped += 1; continue
        next_idx.append(rows)
        agent_labels.append(A2I[rec["agent_action"]])
        opt_labels.append(A2I[rec["opt_label"]])
        opt_sets.append([A2I[a] for a in rec["opt_set"]])
    return (
        torch.tensor(next_idx, dtype=torch.long),
        torch.tensor(agent_labels, dtype=torch.long),
        torch.tensor(opt_labels, dtype=torch.long),
        opt_sets, phi_matrix_norm, state_list, dropped,
    )


# ── models: single θ over next-state φ ─────────────────────────────────
class LinearCostIRL(nn.Module):
    """cost(s) = θᵀ · φ(s);  P(a|s) = softmax(−β · cost(f(s, a)))."""
    def __init__(self, phi_dim, beta=1.0):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(phi_dim))
        self.beta = beta

    def forward(self, phi_next):                       # (N, 4, phi_dim)
        costs = torch.einsum("naf,f->na", phi_next, self.theta)
        return torch.log_softmax(-self.beta * costs, dim=-1)


class MLPCostIRL(nn.Module):
    """cost(s) = MLP(φ(s));  P(a|s) = softmax(−β · cost(f(s, a)))."""
    def __init__(self, phi_dim, hidden_dim=128, beta=1.0, dropout=0.1):
        super().__init__()
        self.beta = beta
        self.net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.1)
        nn.init.zeros_(self.net[6].weight)

    def forward(self, phi_next):                       # (N, 4, phi_dim)
        costs = self.net(phi_next).squeeze(-1)         # (N, 4)
        return torch.log_softmax(-self.beta * costs, dim=-1)


def train_one(model_type, phi_next, labels, train_idx, test_idx):
    torch.manual_seed(SEED)
    phi_dim = phi_next.shape[-1]
    if model_type == "linear":
        model = LinearCostIRL(phi_dim, beta=BETA)
        opt = optim.Adam([model.theta], lr=LR)
    else:
        model = MLPCostIRL(phi_dim, hidden_dim=HIDDEN_DIM, beta=BETA, dropout=DROPOUT)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=REG)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    phi_tr, y_tr = phi_next[train_idx], labels[train_idx]
    phi_te, y_te = phi_next[test_idx], labels[test_idx]
    best_te_ll = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(N_EPOCHS):
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
            te_ll = -nn.functional.nll_loss(model(phi_te), y_te).item()
        if te_ll > best_te_ll:
            best_te_ll = te_ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def eval_model(model, labels, phi_next, opt_sets, idx):
    model.eval()
    with torch.no_grad():
        logp = model(phi_next[idx])
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
    H, W, walls, walkable, GOAL, KEY, DOOR = parse_grid(cfg.traj_dir)
    step = make_step(walls, W, H, GOAL)
    step_state = make_step_state(walls, W, H, GOAL, KEY, DOOR)
    dist, optimal_set = bfs_optimal(walkable, GOAL, step_state)
    print(f"Grid {W}x{H}, walkable={len(walkable)}, goal={GOAL}, "
          f"BFS-reached={len(dist)}/{len(walkable)}")

    all_results = {}
    for token_pos in ["prompt_suffix", "output"]:
        token_label = "pre" if token_pos == "prompt_suffix" else "post"
        print(f"\n--- token = {token_label} ({token_pos}) ---")
        torch.manual_seed(SEED); np.random.seed(SEED)
        records, drop_stats, n_total = collect_records(cfg, GOAL, optimal_set, token_pos)
        phi_table = build_phi_table(records)
        next_idx, agent_labels, opt_labels, opt_sets, phi_mat_norm, state_list, dropped = \
            build_dataset(records, phi_table, step_state)
        N = len(agent_labels)

        # Materialise φ-of-next-state once: (N, 4, phi_dim)
        phi_next = phi_mat_norm[next_idx]

        rng = np.random.default_rng(SEED)
        perm = rng.permutation(N)
        n_test = int(N * TEST_FRAC)
        test_idx = torch.tensor(perm[:n_test], dtype=torch.long)
        train_idx = torch.tensor(perm[n_test:], dtype=torch.long)

        agent_is_optimal = sum(int(a.item() in s) for a, s in zip(agent_labels, opt_sets))
        agent_dist = Counter(int(a) for a in agent_labels.tolist())
        opt_dist = Counter(int(a) for a in opt_labels.tolist())
        meta = {
            "n_records": int(N), "n_unique_states": len(state_list),
            "phi_dim": int(phi_mat_norm.shape[1]),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "drop_filter": drop_stats, "dropped_no_next_phi": int(dropped),
            "n_total_steps": int(n_total),
            "agent_action_dist": {ACTIONS[i]: int(agent_dist[i]) for i in range(4)},
            "optimal_action_dist": {ACTIONS[i]: int(opt_dist[i]) for i in range(4)},
            "agent_is_optimal_frac": agent_is_optimal / N,
            "majority_baseline_agent": max(agent_dist.values()) / N,
            "majority_baseline_opt": max(opt_dist.values()) / N,
        }
        print(f"  N={N} records, {len(state_list)} states, phi_dim={phi_mat_norm.shape[1]}, "
              f"agent-optimal={agent_is_optimal/N:.3f}")

        for model_type in ["linear", "mlp"]:
            for label_kind, labels in [("agent", agent_labels), ("optimal", opt_labels)]:
                key = f"{model_type}__{token_label}__{label_kind}"
                print(f"  training [{key}] ...", end=" ", flush=True)
                model = train_one(model_type, phi_next, labels, train_idx, test_idx)
                tr = eval_model(model, labels, phi_next, opt_sets, train_idx)
                te = eval_model(model, labels, phi_next, opt_sets, test_idx)
                other = opt_labels if label_kind == "agent" else agent_labels
                cross = eval_model(model, other, phi_next, opt_sets, test_idx)
                all_results[key] = {"train": tr, "test": te, "cross": cross}
                print(f"test_acc={te['acc']:.3f}  in_opt={te['in_opt_acc']:.3f}  ll={te['ll']:.4f}")
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
