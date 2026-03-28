"""Strategy 2: Behavioral cloning from activations (action prediction).

Learns a policy π(a|s) = softmax(W h(s_t)) directly from the agent's activation vectors.
This is equivalent to MaxEnt IRL with an action-conditional reward R(s,a) = W_a^T h(s),
which avoids the counterfactual next-state problem of strategy 1.

Trains on:
  - Key-door trajectories (activations_key_door_env_100)
  - Single-step trajectories (activations_train_single_step, all grid sizes)

Usage:
    uv run python run_action_probe.py
"""

import json
import math
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from telos_interp.activation_loading import discover_trajectory_folders
from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_utils import (
    load_activations_for_trajectory,
)
from telos_interp.probe_models import create_classification_model
from telos_interp.training import (
    compute_normalization_params,
    normalize_activations,
    resolve_device,
    set_seed,
    train_epoch,
)

KEYDOOR_ACTIVATIONS_DIR = Path("data/activations/activations_key_door_env_100")
KEYDOOR_TRAJECTORIES_DIR = Path("data/trajectories/trajectories_key_door_100/trajectories_key_door")
SINGLE_STEP_ACTIVATIONS_DIR = Path("data/activations/activations_train_single_step/activations_train_single_step")
SINGLE_STEP_TRAJECTORIES_DIR = Path("data/trajectories/trajectories_train_single_step")

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

TokenCategory = Literal["pre", "post"]

# Strategy 1 reference results (from run_maxent_irl.py --eval output)
STRATEGY1_ACCURACY = 0.797
STRATEGY1_LOG_LIK = -3.153


# ── Key-door dataset collection ─────────────────────────────────────────────

def _find_keydoor_json(folder_name: str) -> Path | None:
    exact = KEYDOOR_TRAJECTORIES_DIR / f"{folder_name}.json"
    if exact.exists():
        return exact
    parts = folder_name.rsplit("_doorkey_", 1)
    if len(parts) == 2:
        candidate = KEYDOOR_TRAJECTORIES_DIR / f"{parts[0]}_doorkey_keepdoor_{parts[1]}.json"
        if candidate.exists():
            return candidate
    return None


def collect_dataset_keydoor(token_category: TokenCategory = "post") -> list[tuple[torch.Tensor, int]]:
    """Load (h(s_t), action_idx) pairs from key-door activation folders."""
    dataset: list[tuple[torch.Tensor, int]] = []
    skipped = 0

    for traj_folder in tqdm(discover_trajectory_folders(KEYDOOR_ACTIVATIONS_DIR), desc="Key-door"):
        json_path = _find_keydoor_json(traj_folder.name)
        if json_path is None:
            skipped += 1
            continue

        with open(json_path) as f:
            steps = json.load(f).get("steps", [])

        for t, step in enumerate(steps):
            action = step.get("agent_action")
            if action not in ACTION_TO_IDX:
                continue

            activation = load_activations_for_trajectory(
                trajectory_folder=traj_folder,
                layers="all",
                steps=str(t),
                prompt_prefix_indices=None,
                grid_state_indices=None,
                prompt_suffix_indices="-1" if token_category == "pre" else None,
                output_indices="-1" if token_category == "post" else None,
            )
            if activation is None:
                continue

            dataset.append((activation.float(), ACTION_TO_IDX[action]))

    print(f"  Key-door ({token_category}): {len(dataset)} samples ({skipped} folders skipped)")
    return dataset


# ── Single-step dataset collection ──────────────────────────────────────────

def collect_dataset_single_step(token_category: TokenCategory = "post") -> list[tuple[torch.Tensor, int]]:
    """Load (h(s_0), action_idx) pairs from single-step activation folders (all sizes)."""
    if not SINGLE_STEP_ACTIVATIONS_DIR.exists():
        print(f"  WARNING: {SINGLE_STEP_ACTIVATIONS_DIR} not found — skipping single-step data.")
        print("  Run: cd data/activations/activations_train_single_step && tar -xf activations_train_single_step.tar")
        return []

    dataset: list[tuple[torch.Tensor, int]] = []
    skipped = 0

    # Iterate over size*/trajectory_folder subdirectories
    size_dirs = sorted(SINGLE_STEP_ACTIVATIONS_DIR.glob("size*"))
    for size_dir in size_dirs:
        traj_folders = sorted(d for d in size_dir.iterdir() if d.is_dir())
        size_name = size_dir.name  # e.g. "size9"

        for traj_folder in tqdm(traj_folders, desc=f"Single-step {size_name}", leave=False):
            # Find matching trajectory JSON
            json_path = SINGLE_STEP_TRAJECTORIES_DIR / size_name / f"{traj_folder.name}.json"
            if not json_path.exists():
                skipped += 1
                continue

            with open(json_path) as f:
                steps = json.load(f).get("steps", [])
            if not steps:
                skipped += 1
                continue

            action = steps[0].get("agent_action")
            if action not in ACTION_TO_IDX:
                skipped += 1
                continue

            activation = load_activations_for_trajectory(
                trajectory_folder=traj_folder,
                layers="all",
                steps="0",
                prompt_prefix_indices=None,
                grid_state_indices=None,
                prompt_suffix_indices="-1" if token_category == "pre" else None,
                output_indices="-1" if token_category == "post" else None,
            )
            if activation is None:
                skipped += 1
                continue

            dataset.append((activation.float(), ACTION_TO_IDX[action]))

    print(f"  Single-step ({token_category}): {len(dataset)} samples ({skipped} skipped)")
    return dataset


# ── Training ────────────────────────────────────────────────────────────────

def train_probe(
    dataset: list[tuple[torch.Tensor, int]],
    model_type: str = "lr",
    hidden_dims: list[int] | None = None,
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    num_epochs: int = 50,
    batch_size: int = 256,
    eval_split: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train action prediction probe. Returns dict with model, scaler, and metrics."""
    set_seed(seed)
    device = resolve_device(None)

    activations = torch.stack([x for x, _ in dataset])   # (N, 8640)
    labels = torch.tensor([y for _, y in dataset], dtype=torch.long)  # (N,)

    input_dim = activations.shape[1]
    n_train = int(len(dataset) * (1 - eval_split))
    n_eval = len(dataset) - n_train

    full_dataset = TensorDataset(activations, labels)
    train_ds, eval_ds = random_split(full_dataset, [n_train, n_eval],
                                      generator=torch.Generator().manual_seed(seed))

    # Normalization from training split only
    train_acts = activations[list(train_ds.indices)]
    scaler_mean, scaler_std = compute_normalization_params(train_acts)
    scaler_mean, scaler_std = scaler_mean.to(device), scaler_std.to(device)

    def make_loader(ds: TensorDataset | torch.utils.data.Subset, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_ds, shuffle=True)
    eval_loader = make_loader(eval_ds, shuffle=False)

    model = create_classification_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=4,
        hidden_dims=hidden_dims or [],
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Wrap loaders to apply normalization on-the-fly
    class NormLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for x, y in self.loader:
                yield normalize_activations(x.to(device), scaler_mean, scaler_std), y.to(device)
        def __len__(self):
            return len(self.loader)

    norm_train = NormLoader(train_loader)
    norm_eval = NormLoader(eval_loader)

    best_bal_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        train_epoch(model, norm_train, criterion, optimizer, device)

        metrics = _evaluate(model, norm_eval, device)
        if metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = metrics["balanced_accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: acc={metrics['accuracy']:.3f}  bal_acc={metrics['balanced_accuracy']:.3f}  ll={metrics['log_likelihood']:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_metrics = _evaluate(model, norm_eval, device)
    return {
        "model": model,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "metrics": final_metrics,
        "best_balanced_accuracy": best_bal_acc,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels, all_log_probs = [], [], []

    for x, y in loader:
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        all_log_probs.append(log_probs.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    log_probs = torch.cat(all_log_probs)

    # Accuracy
    accuracy = (preds == labels).float().mean().item()

    # Log-likelihood of true actions
    n = len(labels)
    ll = log_probs[torch.arange(n), labels].mean().item()

    # Per-class recall and balanced accuracy
    per_class_recall = {}
    for cls in range(4):
        mask = labels == cls
        if mask.sum() == 0:
            per_class_recall[cls] = float("nan")
        else:
            per_class_recall[cls] = (preds[mask] == cls).float().mean().item()

    valid_recalls = [v for v in per_class_recall.values() if not math.isnan(v)]
    balanced_accuracy = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "log_likelihood": ll,
        "per_class_recall": per_class_recall,
        "n": n,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    model_configs = [
        ("LR",  "lr",  [],    0.0),
        ("MLP", "mlp", [128], 0.1),
    ]

    results: dict[str, dict] = {}
    dataset_sizes: dict[str, tuple[int, int]] = {}

    for category in ("pre", "post"):
        print(f"\n=== Phase 1 ({category}-reasoning): Collecting dataset ===")
        dataset_kd = collect_dataset_keydoor(category)
        dataset_ss = collect_dataset_single_step(category)
        dataset = dataset_kd + dataset_ss
        if not dataset:
            raise RuntimeError(f"No samples collected for {category}-reasoning.")
        dataset_sizes[category] = (len(dataset_kd), len(dataset_ss))
        print(f"  Combined: {len(dataset)} samples ({len(dataset_kd)} key-door + {len(dataset_ss)} single-step)")

        print(f"\n=== Phase 2 ({category}-reasoning): Training ===")
        for name, model_type, hidden_dims, dropout in model_configs:
            label = f"{category}-{name}"
            print(f"\n--- {label} ---")
            results[label] = train_probe(
                dataset,
                model_type=model_type,
                hidden_dims=hidden_dims,
                dropout=dropout,
                learning_rate=0.001,
                weight_decay=0.01,
                num_epochs=50,
                batch_size=256,
                eval_split=0.2,
                seed=42,
                verbose=True,
            )

    # ── Summary table ──────────────────────────────────────────────────────
    n_kd_post, n_ss_post = dataset_sizes["post"]
    print("\n" + "=" * 68)
    print("STRATEGY 2: PRE- vs POST-REASONING ACTIVATIONS")
    print(f"Samples: {n_kd_post + n_ss_post}  ({n_kd_post} key-door + {n_ss_post} single-step)")
    print("=" * 68)
    print(f"{'Model':<22} {'Accuracy':>10} {'Log-Lik':>10} {'Bal.Acc':>10}")
    print("-" * 56)
    print(f"{'Random baseline':<22} {'25.0%':>10} {'-1.386':>10} {'25.0%':>10}")
    print(f"{'Strategy 1 (IRL)':<22} {STRATEGY1_ACCURACY:>9.1%} {STRATEGY1_LOG_LIK:>10.3f} {'N/A':>10}")
    for label, result in results.items():
        m = result["metrics"]
        category, model_name = label.split("-", 1)
        row_label = f"{category}-reasoning {model_name}"
        print(
            f"{row_label:<22}"
            f" {m['accuracy']:>9.1%}"
            f" {m['log_likelihood']:>10.3f}"
            f" {m['balanced_accuracy']:>9.1%}"
        )
    print("=" * 68)

    # Per-action recall for both LR variants
    for category in ("pre", "post"):
        label = f"{category}-LR"
        if label in results:
            m = results[label]["metrics"]
            print(f"\nPer-action recall ({category}-reasoning LR):")
            for cls, action_name in enumerate(ACTIONS):
                recall = m["per_class_recall"].get(cls, float("nan"))
                print(f"  {action_name:<6}  {recall:>6.1%}")


if __name__ == "__main__":
    main()
