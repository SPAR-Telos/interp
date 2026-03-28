"""One-step Maximum Entropy IRL for the key-door environment.

Learns cost function C_θ(s) = θ^T φ(s) from demonstrated trajectories.

Feature vector φ(s) — 7 dimensions:
  0  dist_goal  BFS distance from agent to goal
  1  dist_key   BFS distance to key; 0 if already carrying
  2  dist_door  BFS distance to door; 0 if not carrying key or door already open
  3  has_key    1 if agent is carrying the key, else 0
  4  door_open  1 if door is absent from grid (opened), else 0
  5  p_key      P(carrying_key | h(s)) from key_collected_probe_mlp
  6  p_door     P(door_open | h(s)) from door_open_probe_mlp

Algorithm (one-step MaxEnt IRL):
  Policy:   π_θ(a|s) ∝ exp(-C_θ(T(s, a)))
  Gradient: ∇_θ L = φ(T(s, a_observed)) − Σ_a π_θ(a|s) φ(T(s, a))
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from telos_interp.activation_loading import discover_trajectory_folders
from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_utils import (
    load_activations_for_trajectory,
)
from telos_interp.commands.train_key_collected_probe import KeyCollectedProbe
from telos_interp.grid_utils import CELL_SYMBOL_TO_ID, parse_grid_state

ACTIVATIONS_DIR = Path("data/activations/activations_key_door_env_100")
TRAJECTORIES_DIR = Path("data/trajectories/trajectories_key_door_100/trajectories_key_door")
SINGLE_STEP_DIR = Path("data/trajectories/trajectories_train_single_step/size9")
GRID_SIZE = 9

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

WALL_ID = CELL_SYMBOL_TO_ID["#"]
AGENT_ID = CELL_SYMBOL_TO_ID["A"]
GOAL_ID = CELL_SYMBOL_TO_ID["G"]
KEY_ID = CELL_SYMBOL_TO_ID["K"]
DOOR_ID = CELL_SYMBOL_TO_ID["D"]

FEATURE_NAMES = ["dist_goal", "dist_key", "dist_door", "has_key", "door_open", "p_key", "p_door"]
N_FEATURES = len(FEATURE_NAMES)


# ── Trajectory JSON lookup ──────────────────────────────────────────────────

def find_trajectory_json(folder_name: str) -> Path | None:
    exact = TRAJECTORIES_DIR / f"{folder_name}.json"
    if exact.exists():
        return exact
    parts = folder_name.rsplit("_doorkey_", 1)
    if len(parts) == 2:
        candidate = TRAJECTORIES_DIR / f"{parts[0]}_doorkey_keepdoor_{parts[1]}.json"
        if candidate.exists():
            return candidate
    return None


# ── State parsing and transitions ──────────────────────────────────────────

def parse_state(step: dict) -> dict | None:
    """Extract state components from a trajectory step dict."""
    triples = parse_grid_state(step["grid_state"])

    walls: set[tuple[int, int]] = set()
    agent_pos = None
    goal_pos = None
    key_pos = None
    door_pos = None

    for row, col, cell_id in triples:
        if cell_id == WALL_ID:
            walls.add((row, col))
        elif cell_id == AGENT_ID:
            agent_pos = (row, col)
        elif cell_id == GOAL_ID:
            goal_pos = (row, col)
        elif cell_id == KEY_ID:
            key_pos = (row, col)
        elif cell_id == DOOR_ID:
            door_pos = (row, col)

    if agent_pos is None or goal_pos is None:
        return None

    has_key = bool(step.get("carrying_key", False))
    door_open = door_pos is None  # D absent → door open

    # Infer grid size from the outermost border wall row index
    grid_size = max((r for r, c in walls), default=GRID_SIZE - 1) + 1

    return {
        "agent_pos": agent_pos,
        "walls": walls,
        "goal_pos": goal_pos,
        "key_pos": key_pos,
        "door_pos": door_pos,
        "has_key": has_key,
        "door_open": door_open,
        "grid_size": grid_size,
    }


def apply_action(state: dict, action: str) -> dict:
    """Return the next state after applying action (no mutation)."""
    new = dict(state)  # shallow copy; walls/goal_pos are immutable in this context

    if action not in DELTA:
        return new  # unknown action: stay

    dr, dc = DELTA[action]
    r, c = state["agent_pos"]
    nr, nc = r + dr, c + dc

    # Wall or out-of-bounds → stay
    gs = state["grid_size"]
    if (nr, nc) in state["walls"] or not (0 <= nr < gs and 0 <= nc < gs):
        return new

    new["agent_pos"] = (nr, nc)

    # Auto-collect key when stepping onto it
    if state["key_pos"] == (nr, nc) and not state["has_key"]:
        new["has_key"] = True
        new["key_pos"] = None

    # Auto-open door when stepping onto it while carrying key
    if state["door_pos"] == (nr, nc) and state["has_key"] and not state["door_open"]:
        new["door_open"] = True
        new["door_pos"] = None

    return new


# ── BFS distance ────────────────────────────────────────────────────────────

def bfs_distance(
    start: tuple[int, int],
    goal: tuple[int, int] | None,
    walls: set,
    grid_size: int = GRID_SIZE,
) -> float:
    """BFS shortest path distance normalised to [0, 1].

    Returns dist / (grid_size ** 2) so the value is always in (0, 1] regardless
    of grid size. Unreachable cells return 1.0 (the maximum sentinel).
    """
    if goal is None or start == goal:
        return 0.0

    visited = {start}
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])

    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited or (nr, nc) in walls:
                continue
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                continue
            if (nr, nc) == goal:
                return float(dist + 1) / (grid_size * grid_size)
            visited.add((nr, nc))
            queue.append(((nr, nc), dist + 1))

    return 1.0  # unreachable sentinel


# ── Feature computation ─────────────────────────────────────────────────────

def compute_features(
    state: dict,
    p_key: float | None = None,
    p_door: float | None = None,
) -> np.ndarray:
    """Compute 7-dim feature vector for a state.

    p_key / p_door: probe output probabilities. Falls back to binary proxy if None.
    """
    pos = state["agent_pos"]
    walls = state["walls"]
    grid_size = state.get("grid_size", GRID_SIZE)

    f0 = bfs_distance(pos, state["goal_pos"], walls, grid_size)
    f1 = 0.0 if state["has_key"] else bfs_distance(pos, state["key_pos"], walls, grid_size)
    f2 = 0.0 if (not state["has_key"] or state["door_open"]) else bfs_distance(pos, state["door_pos"], walls, grid_size)
    f3 = float(state["has_key"])
    f4 = float(state["door_open"])
    f5 = p_key if p_key is not None else f3
    f6 = p_door if p_door is not None else f4

    return np.array([f0, f1, f2, f3, f4, f5, f6], dtype=np.float64)


# ── Softmax ─────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── Probe loading and inference ─────────────────────────────────────────────

def load_probes() -> tuple[KeyCollectedProbe, KeyCollectedProbe]:
    key_probe = KeyCollectedProbe.load(ACTIVATIONS_DIR / "key_collected_probe_mlp.pt")
    door_probe = KeyCollectedProbe.load(ACTIVATIONS_DIR / "door_open_probe_mlp.pt")
    return key_probe, door_probe


def get_probe_probs(
    traj_folder: Path,
    step_idx: int,
    key_probe: KeyCollectedProbe,
    door_probe: KeyCollectedProbe,
) -> tuple[float | None, float | None]:
    """Load activation at step_idx and return (p_has_key, p_door_open)."""
    activation = load_activations_for_trajectory(
        trajectory_folder=traj_folder,
        layers="all",
        steps=str(step_idx),
        prompt_prefix_indices=None,
        grid_state_indices=None,
        prompt_suffix_indices=None,
        output_indices="-1",
    )
    if activation is None:
        return None, None

    act = activation.unsqueeze(0)  # (1, D)
    p_key = key_probe.predict_proba(act)[0, 1].item()   # P(carrying_key=True)
    p_door = door_probe.predict_proba(act)[0, 1].item()  # P(door_open=True)
    return p_key, p_door


# ── Dataset collection ───────────────────────────────────────────────────────

def collect_dataset(
    key_probe: KeyCollectedProbe,
    door_probe: KeyCollectedProbe,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build list of (phi_actual, phi_counterfactual) pairs.

    phi_actual:          features of the actually-visited next state (with probe probs).
    phi_counterfactual:  (4, 7) array of features for all 4 possible next states
                         (probe probs approximated with binary proxies).
    """
    trajectory_folders = discover_trajectory_folders(ACTIVATIONS_DIR)

    dataset: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0

    for traj_folder in tqdm(trajectory_folders, desc="Collecting dataset"):
        json_path = find_trajectory_json(traj_folder.name)
        if json_path is None:
            skipped += 1
            continue

        with open(json_path) as f:
            traj_data = json.load(f)
        steps = traj_data.get("steps", [])

        for t in range(len(steps) - 1):
            action = steps[t].get("agent_action")
            if action not in DELTA:
                continue

            s_t = parse_state(steps[t])
            if s_t is None:
                continue

            # Actual next state
            s_next = apply_action(s_t, action)

            # Probe probabilities for actual next state (from step t+1 activations)
            p_key, p_door = get_probe_probs(traj_folder, t + 1, key_probe, door_probe)

            phi_actual = compute_features(s_next, p_key, p_door)

            # Features for all 4 possible next states (binary proxy for probe features)
            phi_counterfactual = np.array([
                compute_features(apply_action(s_t, a))
                for a in ACTIONS
            ])  # (4, 7)

            action_idx = ACTIONS.index(action)
            dataset.append((phi_actual, phi_counterfactual, action_idx))

    print(f"  Collected {len(dataset)} samples ({skipped} trajectories skipped)")
    return dataset


# ── Single-step dataset collection ─────────────────────────────────────────

def collect_dataset_single_step(json_dir: Path) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Build dataset from single-step trajectory JSONs (no activations needed).

    These trajectories have no key/door, so probe features fall back to binary proxies.
    Compatible with the same (phi_actual, phi_cf, action_idx) tuple format.
    """
    dataset: list[tuple[np.ndarray, np.ndarray, int]] = []
    json_files = sorted(json_dir.glob("*.json"))

    for json_path in tqdm(json_files, desc=f"Collecting {json_dir.name}"):
        with open(json_path) as f:
            traj_data = json.load(f)
        steps = traj_data.get("steps", [])
        if not steps:
            continue

        step = steps[0]
        action = step.get("agent_action")
        if action not in DELTA:
            continue

        s = parse_state(step)
        if s is None:
            continue

        # No activations available — binary proxies used for p_key / p_door
        phi_actual = compute_features(apply_action(s, action))
        phi_cf = np.array([compute_features(apply_action(s, a)) for a in ACTIONS])
        dataset.append((phi_actual, phi_cf, ACTIONS.index(action)))

    print(f"  Collected {len(dataset)} samples from {json_dir.name}")
    return dataset


# ── Training ────────────────────────────────────────────────────────────────

def train_maxent_irl(
    dataset: list[tuple[np.ndarray, np.ndarray]],
    learning_rate: float = 0.01,
    num_epochs: int = 10,
) -> np.ndarray:
    """Run one-step MaxEnt IRL gradient ascent."""
    theta = np.zeros(N_FEATURES)

    for epoch in range(num_epochs):
        total_grad_norm = 0.0

        for phi_actual, phi_cf, _ in dataset:
            # phi_cf: (4, 7)  — counterfactual next-state features
            costs = phi_cf @ theta           # (4,)
            pi = softmax(-costs)             # (4,)
            phi_expected = pi @ phi_cf       # (7,)

            grad = phi_actual - phi_expected
            theta += learning_rate * grad
            total_grad_norm += float(np.linalg.norm(grad))

        avg_grad = total_grad_norm / max(len(dataset), 1)
        print(f"  Epoch {epoch + 1}/{num_epochs}: avg grad norm = {avg_grad:.4f}")

    return theta


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(dataset: list[tuple[np.ndarray, np.ndarray, int]], theta: np.ndarray) -> dict:
    """Compute three quality metrics for the learned reward weights θ.

    Returns:
        log_likelihood:     mean log π_θ(a_demo | s) over all steps
        random_baseline_ll: log(1/4) — uniform random policy baseline
        accuracy:           fraction of steps where argmax_a R(T(s,a)) == a_demo
        random_baseline_acc: 0.25
        feature_gap:        (N_FEATURES,) array of (E_demo[φ] - E_π[φ]) / n
    """
    log_liks: list[float] = []
    correct = 0
    phi_demo_sum = np.zeros(N_FEATURES)
    phi_policy_sum = np.zeros(N_FEATURES)

    for phi_actual, phi_cf, action_idx in dataset:
        rewards = phi_cf @ theta          # R(T(s,a)) for each of 4 actions
        pi = softmax(rewards)             # π_θ(a|s)

        log_liks.append(float(np.log(pi[action_idx] + 1e-12)))
        correct += int(np.argmax(rewards) == action_idx)
        phi_demo_sum += phi_actual
        phi_policy_sum += pi @ phi_cf

    n = len(dataset)
    return {
        "log_likelihood": float(np.mean(log_liks)),
        "random_baseline_ll": float(np.log(0.25)),
        "accuracy": correct / n,
        "random_baseline_acc": 0.25,
        "feature_gap": (phi_demo_sum - phi_policy_sum) / n,
    }


def print_eval(metrics: dict) -> None:
    ll = metrics["log_likelihood"]
    ll_base = metrics["random_baseline_ll"]
    acc = metrics["accuracy"]
    acc_base = metrics["random_baseline_acc"]
    gap = metrics["feature_gap"]

    print("\n" + "=" * 55)
    print("EVALUATION")
    print("=" * 55)
    print(f"  Log-likelihood:   {ll:>8.4f}  (random baseline: {ll_base:.4f})")
    print(f"  Action accuracy:  {acc:>7.1%}  (random baseline: {acc_base:.1%})")
    print()
    print(f"  Feature expectation gap  (demo − policy) / n:")
    print(f"  {'Feature':<12} {'Gap':>10}  {'OK?' }")
    print(f"  {'-'*35}")
    for name, g in zip(FEATURE_NAMES, gap):
        ok = "✓" if abs(g) < 0.5 else "✗ large gap"
        print(f"  {name:<12} {g:>10.4f}  {ok}")
    print("=" * 55)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="One-step MaxEnt IRL for the key-door environment.")
    parser.add_argument("--eval", action="store_true", help="Evaluate learned θ after training")
    args = parser.parse_args()

    print("Loading probes...")
    key_probe, door_probe = load_probes()

    print("\n=== Phase 1a: Collecting key-door dataset ===")
    dataset_kd = collect_dataset(key_probe, door_probe)

    print("\n=== Phase 1b: Collecting single-step dataset ===")
    dataset_ss = collect_dataset_single_step(SINGLE_STEP_DIR)

    dataset = dataset_kd + dataset_ss
    if not dataset:
        raise RuntimeError("No samples collected — check data paths.")
    print(f"\nCombined: {len(dataset)} samples total ({len(dataset_kd)} key-door + {len(dataset_ss)} single-step)")

    print(f"\n=== Phase 2: Training (n={len(dataset)} samples) ===")
    theta = train_maxent_irl(dataset, learning_rate=0.01, num_epochs=10)

    print("\n" + "=" * 50)
    print("LEARNED REWARD WEIGHTS θ  (R(s) = θ^T φ(s))")
    print("=" * 50)
    print(f"{'Feature':<12} {'Weight':>10}  Interpretation")
    print("-" * 50)
    for name, w in zip(FEATURE_NAMES, theta):
        if abs(w) < 0.01:
            interp = "~ neutral"
        elif w > 0:
            interp = "✓ agent seeks high values"
        else:
            interp = "↓ agent avoids high values"
        print(f"{name:<12} {w:>10.4f}  {interp}")
    print("=" * 50)
    print("\nNote: R(s) = θ^T φ(s) is the inferred reward function.")
    print("      Cost function: C(s) = -R(s) = (-θ)^T φ(s).")
    print("      Positive weight → agent seeks states with high feature value.")
    print("      Negative weight → agent avoids states with high feature value.")

    np.save(str(ACTIVATIONS_DIR / "maxent_irl_theta.npy"), theta)
    print(f"\nSaved θ → {ACTIVATIONS_DIR / 'maxent_irl_theta.npy'}")

    if args.eval:
        print("\n=== Phase 3: Evaluation ===")
        metrics = evaluate(dataset, theta)
        print_eval(metrics)


if __name__ == "__main__":
    main()
