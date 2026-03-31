"""n-Step Maximum Entropy IRL on a single fixed key-door grid.

Unlike the one-step approximation in run_maxent_irl.py (which treats each
transition i.i.d.), this script uses the full MaxEnt IRL algorithm:

  Backward pass: truncated soft value iteration over n steps
      Q(s, a) = R(s) + V(T(s, a))
      V(s) = log Σ_a exp(Q(s, a))      (soft Bellman)

  Forward pass: propagate state visitation for n steps
      μ_0 = empirical start-state distribution
      μ_{t+1}(s') = Σ_{s,a} μ_t(s) π(a|s) 1[T(s,a)=s']

  Gradient: θ += α * (φ_demo − φ_policy)
      φ_demo   = mean φ(s_t) over demonstration states
      φ_policy = μ-weighted mean φ(s) over all states

Requires all 100 trajectories on the same fixed grid.

Usage:
    # Step 0: gather activations (requires GPU + model access)
    interp-cli gather_activations \\
        --trajectory-paths "data/trajectories/together_ai_openai_gpt-oss-20b_rooms3_doorkey_grid0_traj*.json" \\
        --output-dir data/activations \\
        --layers "7,15,23" --steps all \\
        --prompt-suffix-indices "-3:-1" --output-indices "-16:-14"

    # Step 1: run IRL
    uv run python run_fixed_grid_maxent_irl.py [--n 50] [--epochs 50] [--lr 0.1]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

# Re-use shared utilities from the one-step script
from run_maxent_irl import (
    ACTIONS,
    DELTA,
    FEATURE_NAMES,
    N_FEATURES,
    compute_features,
    parse_state,
    softmax,
)
from telos_interp.activation_loading import discover_trajectory_folders
from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_utils import (
    load_activations_for_trajectory,
)
from telos_interp.commands.train_key_collected_probe import KeyCollectedProbe
from tqdm import tqdm

TRAJ_DIR = Path("data/trajectories")
ACTIVATION_DIR = Path("data/activations/activations_fixed_key_door_grid")

# State tuple: (row, col, has_key: bool, door_open: bool)
StateTuple = tuple[int, int, bool, bool]


# ── Corrected transition model ──────────────────────────────────────────────


def apply_action_fixed(state: dict, action: str) -> dict:
    """Deterministic transition with correct door mechanics.

    Differences from run_maxent_irl.apply_action:
    - A locked door blocks movement (treated as impassable wall).
    - The door opens automatically when the agent becomes adjacent
      (Manhattan distance 1) to the door while carrying the key.
    """
    new = dict(state)
    if action not in DELTA:
        return new

    dr, dc = DELTA[action]
    r, c = state["agent_pos"]
    nr, nc = r + dr, c + dc
    gs = state["grid_size"]

    # Wall or out-of-bounds → stay
    if (nr, nc) in state["walls"] or not (0 <= nr < gs and 0 <= nc < gs):
        return new

    # Locked door blocks movement
    door_pos = state["door_pos"]
    door_locked = door_pos is not None and not state["door_open"]
    if door_locked and (nr, nc) == door_pos:
        return new

    new["agent_pos"] = (nr, nc)

    # Auto-collect key when stepping onto it
    if state["key_pos"] == (nr, nc) and not state["has_key"]:
        new["has_key"] = True
        new["key_pos"] = None

    # Auto-open door when adjacent (Manhattan dist 1) and carrying key
    if door_locked and new["has_key"]:
        ar, ac = new["agent_pos"]
        if abs(ar - door_pos[0]) + abs(ac - door_pos[1]) == 1:
            new["door_open"] = True
            new["door_pos"] = None

    return new


# ── State conversion helpers ─────────────────────────────────────────────────


def state_tuple_to_dict(s: StateTuple, grid: dict) -> dict:
    r, c, has_key, door_open = s
    return {
        "agent_pos": (r, c),
        "walls": grid["walls"],
        "goal_pos": grid["goal_pos"],
        "key_pos": None if has_key else grid["key_start_pos"],
        "door_pos": None if door_open else grid["door_start_pos"],
        "has_key": has_key,
        "door_open": door_open,
        "grid_size": grid["grid_size"],
    }


def state_dict_to_tuple(s: dict) -> StateTuple:
    r, c = s["agent_pos"]
    return (r, c, s["has_key"], s["door_open"])


# ── Grid loading ─────────────────────────────────────────────────────────────


def load_fixed_grid(traj_dir: Path) -> dict:
    """Parse the fixed grid layout from step 0 of the first trajectory."""
    traj_files = sorted(f for f in traj_dir.glob("*.json") if f.name != "grid_layout.json")
    if not traj_files:
        raise FileNotFoundError(f"No trajectory JSONs found in {traj_dir}")

    with open(traj_files[0]) as f:
        traj_data = json.load(f)

    step0 = traj_data["steps"][0]
    state = parse_state(step0)
    if state is None:
        raise ValueError("Failed to parse state from first trajectory step")

    return {
        "walls": state["walls"],
        "goal_pos": state["goal_pos"],
        "key_start_pos": state["key_pos"],
        "door_start_pos": state["door_pos"],
        "grid_size": state["grid_size"],
    }


# ── State space ──────────────────────────────────────────────────────────────


def enumerate_states(grid: dict) -> list[StateTuple]:
    """Return all reachable (r, c, has_key, door_open) non-wall states.

    The combination (has_key=False, door_open=True) is excluded: in this
    environment the door can only be opened while carrying the key, so the
    agent always has the key whenever the door is open.
    """
    gs = grid["grid_size"]
    walls = grid["walls"]
    states: list[StateTuple] = []
    for r in range(gs):
        for c in range(gs):
            if (r, c) in walls:
                continue
            for has_key in (False, True):
                for door_open in (False, True):
                    if door_open and not has_key:
                        continue  # unreachable
                    states.append((r, c, has_key, door_open))
    return states


# ── Transition model ─────────────────────────────────────────────────────────


def build_transition_model(states: list[StateTuple], grid: dict) -> np.ndarray:
    """Build deterministic transition table.

    Returns:
        T: int array of shape (n_states, n_actions).
           T[i, a] is the index of the next state when taking ACTIONS[a] in
           states[i]. Goal states are absorbing (T[i, :] = i).
    """
    state_idx = {s: i for i, s in enumerate(states)}
    goal_pos = grid["goal_pos"]
    T = np.empty((len(states), len(ACTIONS)), dtype=np.int32)

    for i, s in enumerate(states):
        r, c, _, _ = s
        if (r, c) == goal_pos:
            T[i, :] = i  # absorbing
            continue

        s_dict = state_tuple_to_dict(s, grid)
        for j, action in enumerate(ACTIONS):
            s_next = state_dict_to_tuple(apply_action_fixed(s_dict, action))
            T[i, j] = state_idx.get(s_next, i)  # fallback: stay

    return T


# ── Demonstrations ───────────────────────────────────────────────────────────


def load_demonstrations(traj_dir: Path) -> list[list[tuple[StateTuple, int]]]:
    """Load all trajectories as lists of (state_tuple, action_idx) pairs."""
    traj_files = sorted(f for f in traj_dir.glob("*.json") if f.name != "grid_layout.json")
    demos: list[list[tuple[StateTuple, int]]] = []

    for traj_file in traj_files:
        with open(traj_file) as f:
            traj_data = json.load(f)

        steps = traj_data.get("steps", [])
        traj: list[tuple[StateTuple, int]] = []

        for step in steps:
            action = step.get("agent_action")
            if action not in DELTA:
                continue
            s = parse_state(step)
            if s is None:
                continue
            traj.append((state_dict_to_tuple(s), ACTIONS.index(action)))

        if traj:
            demos.append(traj)

    n_transitions = sum(len(d) for d in demos)
    print(f"  Loaded {len(demos)} demonstrations, {n_transitions} transitions total")
    return demos


# ── Probe lookup ─────────────────────────────────────────────────────────────


TokenCategory = Literal["pre", "post"]


def build_probe_lookup(
    activation_dir: Path,
    traj_dir: Path,
    key_probe: KeyCollectedProbe,
    door_probe: KeyCollectedProbe,
    token_category: TokenCategory = "post",
) -> dict[StateTuple, tuple[float, float]]:
    """Build state → (p_key, p_door) from all demonstration activations.

    When the same logical state is visited across multiple trajectories the
    probe outputs are averaged.  States with no activation data are absent from
    the returned dict; callers fall back to binary proxies for those.

    Args:
        token_category: Which activation to use — 'pre' (last token of prompt
            suffix, before the model generates output) or 'post' (last output
            token, after the model has committed to an action).
    """
    if not activation_dir.exists():
        print(f"  WARNING: {activation_dir} not found — probe features will use binary proxies.")
        print("  Run the gather-activations command first to enable probe-based features.")
        return {}

    accum: dict[StateTuple, list[tuple[float, float]]] = defaultdict(list)

    for traj_folder in tqdm(discover_trajectory_folders(activation_dir), desc="Probe lookup"):
        json_path = traj_dir / f"{traj_folder.name}.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            steps = json.load(f).get("steps", [])

        for step_idx, step in enumerate(steps):
            s = parse_state(step)
            if s is None:
                continue

            activation = load_activations_for_trajectory(
                trajectory_folder=traj_folder,
                layers="all",
                steps=str(step_idx),
                prompt_prefix_indices=None,
                grid_state_indices=None,
                prompt_suffix_indices="-1" if token_category == "pre" else None,
                output_indices="-1" if token_category == "post" else None,
            )
            if activation is None:
                continue

            act = activation.unsqueeze(0)  # (1, D)
            p_key = key_probe.predict_proba(act)[0, 1].item()
            p_door = door_probe.predict_proba(act)[0, 1].item()
            accum[state_dict_to_tuple(s)].append((p_key, p_door))

    lookup = {
        s: (
            float(np.mean([v[0] for v in vals])),
            float(np.mean([v[1] for v in vals])),
        )
        for s, vals in accum.items()
    }
    print(f"  Probe lookup: {len(lookup)} unique states with activation data")
    return lookup


# ── Feature table ────────────────────────────────────────────────────────────


def compute_phi_table(
    states: list[StateTuple],
    grid: dict,
    probe_lookup: dict[StateTuple, tuple[float, float]],
) -> np.ndarray:
    """Pre-compute feature vectors for all states.

    Returns:
        phi: float64 array of shape (n_states, N_FEATURES)
    """
    phi = np.zeros((len(states), N_FEATURES), dtype=np.float64)
    for i, s in enumerate(states):
        p_key, p_door = probe_lookup.get(s, (None, None))
        phi[i] = compute_features(state_tuple_to_dict(s, grid), p_key, p_door)
    return phi


# ── n-Step MaxEnt IRL ────────────────────────────────────────────────────────


def run_maxent_irl(
    states: list[StateTuple],
    T: np.ndarray,
    phi: np.ndarray,
    demonstrations: list[list[tuple[StateTuple, int]]],
    n: int = 50,
    learning_rate: float = 0.1,
    num_epochs: int = 50,
) -> np.ndarray:
    """Full n-step MaxEnt IRL.

    Reward model: R(s) = θ^T φ(s)   (state-based, convention 1)
    Soft Bellman: Q(s, a) = R(s) + V(T(s, a))
                  V(s) = logsumexp_a Q(s, a)
    Policy:       π(a|s) = softmax_a Q(s, a)

    Args:
        states:         List of all reachable state tuples.
        T:              Transition index array, shape (n_states, n_actions).
        phi:            Feature matrix, shape (n_states, N_FEATURES).
        demonstrations: List of trajectories; each is [(state_tuple, action_idx)].
        n:              Horizon for backward value iteration and forward propagation.
        learning_rate:  Gradient ascent step size.
        num_epochs:     Number of full gradient passes.

    Returns:
        theta: Learned reward weights, shape (N_FEATURES,).
    """
    state_idx = {s: i for i, s in enumerate(states)}
    n_states, n_actions = T.shape

    # ── Empirical feature expectation (average φ(s_t) over demo transitions) ──
    phi_demo = np.zeros(N_FEATURES)
    total_steps = 0
    for traj in demonstrations:
        for s_tuple, _ in traj:
            if s_tuple in state_idx:
                phi_demo += phi[state_idx[s_tuple]]
                total_steps += 1
    phi_demo /= max(total_steps, 1)

    # ── Initial state distribution from demonstrations ─────────────────────
    mu0 = np.zeros(n_states)
    for traj in demonstrations:
        if traj:
            s0, _ = traj[0]
            if s0 in state_idx:
                mu0[state_idx[s0]] += 1.0
    mu0 /= mu0.sum() if mu0.sum() > 0 else 1.0

    theta = np.zeros(N_FEATURES)

    for epoch in range(num_epochs):
        # ── Backward pass: soft value iteration ────────────────────────────
        R = phi @ theta  # (n_states,)  — reward for each state
        V = np.zeros(n_states)

        for _ in range(n):
            Q = R[:, np.newaxis] + V[T]  # (n_states, n_actions)
            # Numerically stable logsumexp over actions
            Q_max = Q.max(axis=1, keepdims=True)
            V = Q_max[:, 0] + np.log(np.exp(Q - Q_max).sum(axis=1))

        # Final policy from converged Q
        Q = R[:, np.newaxis] + V[T]
        Q_shifted = Q - Q.max(axis=1, keepdims=True)
        pi = np.exp(Q_shifted) / np.exp(Q_shifted).sum(axis=1, keepdims=True)  # (n_states, n_actions)

        # ── Forward pass: state visitation ─────────────────────────────────
        mu = np.zeros(n_states)
        mu_t = mu0.copy()
        for _ in range(n):
            mu += mu_t
            mu_next = np.zeros(n_states)
            weight = mu_t[:, np.newaxis] * pi  # (n_states, n_actions)
            for a in range(n_actions):
                np.add.at(mu_next, T[:, a], weight[:, a])
            mu_t = mu_next

        mu /= mu.sum() if mu.sum() > 0 else 1.0

        # ── Gradient update ─────────────────────────────────────────────────
        phi_policy = mu @ phi  # (N_FEATURES,)
        grad = phi_demo - phi_policy
        theta += learning_rate * grad

        if (epoch + 1) % 10 == 0:
            grad_norm = float(np.linalg.norm(grad))
            max_gap = float(np.abs(grad).max())
            print(f"  Epoch {epoch + 1:3d}/{num_epochs}: ‖Δφ‖={grad_norm:.6f}  max_gap={max_gap:.6f}")

    return theta


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate(
    theta: np.ndarray,
    demonstrations: list[list[tuple[StateTuple, int]]],
    state_idx: dict[StateTuple, int],
    T: np.ndarray,
    phi: np.ndarray,
    n: int,
) -> dict:
    """Evaluate θ: action accuracy, log-likelihood, feature expectation gap."""
    n_states = len(state_idx)

    # Re-run backward pass to get the policy under learned θ
    R = phi @ theta
    V = np.zeros(n_states)
    for _ in range(n):
        Q = R[:, np.newaxis] + V[T]
        Q_max = Q.max(axis=1, keepdims=True)
        V = Q_max[:, 0] + np.log(np.exp(Q - Q_max).sum(axis=1))

    Q_final = R[:, np.newaxis] + V[T]  # (n_states, n_actions)

    log_liks: list[float] = []
    correct = 0
    phi_demo_sum = np.zeros(N_FEATURES)
    phi_policy_sum = np.zeros(N_FEATURES)
    total = 0

    for traj in demonstrations:
        for s_tuple, a_demo in traj:
            if s_tuple not in state_idx:
                continue
            i = state_idx[s_tuple]
            pi = softmax(Q_final[i])

            log_liks.append(float(np.log(pi[a_demo] + 1e-12)))
            correct += int(np.argmax(Q_final[i]) == a_demo)
            phi_demo_sum += phi[i]
            phi_policy_sum += pi @ phi[T[i]]
            total += 1

    return {
        "log_likelihood": float(np.mean(log_liks)),
        "random_baseline_ll": float(np.log(0.25)),
        "accuracy": correct / max(total, 1),
        "random_baseline_acc": 0.25,
        "feature_gap": (phi_demo_sum - phi_policy_sum) / max(total, 1),
        "n_transitions": total,
    }


def print_results(theta: np.ndarray, metrics: dict) -> None:
    total_abs = max(np.abs(theta).sum(), 1e-9)
    print("\n" + "=" * 55)
    print("LEARNED REWARD WEIGHTS θ  (R(s) = θ^T φ(s))")
    print("=" * 55)
    for name, w in zip(FEATURE_NAMES, theta, strict=False):
        pct = 100 * abs(w) / total_abs
        sign = "+" if w >= 0 else ""
        print(f"  {name:<12} {sign}{w:9.4f}   ({pct:4.1f}%)")

    ll = metrics["log_likelihood"]
    ll_base = metrics["random_baseline_ll"]
    acc = metrics["accuracy"]
    acc_base = metrics["random_baseline_acc"]
    gap = metrics["feature_gap"]

    print("\n" + "=" * 55)
    print("EVALUATION")
    print("=" * 55)
    print(f"  Transitions: {metrics['n_transitions']}")
    print(f"  Log-likelihood:  {ll:>8.4f}  (random: {ll_base:.4f})")
    print(f"  Action accuracy: {acc:>7.1%}  (random: {acc_base:.1%})")
    print()
    print("  Feature expectation gap  (demo − policy):")
    print(f"  {'Feature':<12} {'Gap':>12}")
    print(f"  {'-' * 26}")
    for name, g in zip(FEATURE_NAMES, gap, strict=False):
        ok = "✓" if abs(g) < 0.01 else ("~" if abs(g) < 0.05 else "✗ large")
        print(f"  {name:<12} {g:>12.6f}  {ok}")
    print("=" * 55)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="n-Step MaxEnt IRL on a single fixed key-door grid.")
    parser.add_argument(
        "--n", type=int, default=50, help="Backward/forward horizon (default: 50; covers the full task)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument(
        "--token-category",
        choices=["pre", "post"],
        default="post",
        help="Activation category for probe features: 'pre' (prompt suffix last token) "
        "or 'post' (output last token). Default: post",
    )
    parser.add_argument(
        "--probe-dir",
        type=str,
        default=None,
        help="Directory containing key_collected_probe_mlp.pt and "
        "door_open_probe_mlp.pt. If omitted or probes not found, "
        "binary proxies (has_key, door_open) are used for p_key/p_door.",
    )
    args = parser.parse_args()

    print(f"n-Step MaxEnt IRL  (n={args.n}, epochs={args.epochs}, lr={args.lr})")

    print("\n=== Phase 1: Loading fixed grid ===")
    grid = load_fixed_grid(TRAJ_DIR)
    print(f"  Grid size:  {grid['grid_size']}x{grid['grid_size']}")
    print(f"  Key start:  {grid['key_start_pos']}")
    print(f"  Door start: {grid['door_start_pos']}")
    print(f"  Goal:       {grid['goal_pos']}")

    print("\n=== Phase 2: Enumerating state space ===")
    states = enumerate_states(grid)
    state_idx = {s: i for i, s in enumerate(states)}
    print(f"  {len(states)} reachable states")

    print("\n=== Phase 3: Building transition model ===")
    T = build_transition_model(states, grid)
    print(f"  Transition table: {T.shape}")

    print("\n=== Phase 4: Loading demonstrations ===")
    demonstrations = load_demonstrations(TRAJ_DIR)

    print("\n=== Phase 5: Loading probes ===")
    probe_lookup: dict[StateTuple, tuple[float, float]] = {}
    probe_dir = Path(args.probe_dir) if args.probe_dir else None
    if probe_dir is not None:
        key_path = probe_dir / "key_collected_probe_mlp.pt"
        door_path = probe_dir / "door_open_probe_mlp.pt"
        if key_path.exists() and door_path.exists():
            key_probe = KeyCollectedProbe.load(key_path)
            door_probe = KeyCollectedProbe.load(door_path)
            print("  Probes loaded.")

            print("\n=== Phase 6: Building probe lookup ===")
            probe_lookup = build_probe_lookup(
                ACTIVATION_DIR,
                TRAJ_DIR,
                key_probe,
                door_probe,
                token_category=args.token_category,
            )
        else:
            print(f"  Probe files not found in {probe_dir}")
            print("  Using binary proxies for p_key / p_door.")
    else:
        print("  No --probe-dir specified; using binary proxies for p_key / p_door.")

    print("\n=== Phase 7: Pre-computing feature table ===")
    phi = compute_phi_table(states, grid, probe_lookup)
    print(f"  Feature matrix: {phi.shape}")

    print(f"\n=== Phase 8: Training (n={args.n}, epochs={args.epochs}, lr={args.lr}) ===")
    theta = run_maxent_irl(
        states=states,
        T=T,
        phi=phi,
        demonstrations=demonstrations,
        n=args.n,
        learning_rate=args.lr,
        num_epochs=args.epochs,
    )

    print("\n=== Phase 9: Evaluation ===")
    metrics = evaluate(theta, demonstrations, state_idx, T, phi, args.n)
    print_results(theta, metrics)

    out_path = ACTIVATION_DIR / "fixed_grid_maxent_irl_theta.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), theta)
    print(f"\nSaved θ → {out_path}")


if __name__ == "__main__":
    main()
