"""Run key-collected binary probes (LR + MLP) for pre- and post-reasoning activations.

Builds datasets by collecting one sample per (trajectory, step) pair across all steps,
then trains and evaluates probes. Prints a final accuracy table.
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm

from telos_interp.activation_loading import discover_trajectory_folders
from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_utils import (
    load_activations_for_trajectory,
)
from telos_interp.commands.train_key_collected_probe import train_key_collected_probe

ACTIVATIONS_DIR = Path("data/activations/activations_key_door_env_100")
TRAJECTORIES_DIR = Path("data/trajectories/trajectories_key_door_100/trajectories_key_door")


def find_trajectory_json(folder_name: str) -> Path | None:
    """Map an activation folder name to its trajectory JSON file.

    Activation folders: ...doorkey_N
    JSON files:         ...doorkey_keepdoor_N.json
    """
    # Try exact match first (in case naming is consistent)
    exact = TRAJECTORIES_DIR / f"{folder_name}.json"
    if exact.exists():
        return exact

    # Insert 'keepdoor' before the trailing numeric segment
    parts = folder_name.rsplit("_doorkey_", 1)
    if len(parts) == 2:
        candidate = TRAJECTORIES_DIR / f"{parts[0]}_doorkey_keepdoor_{parts[1]}.json"
        if candidate.exists():
            return candidate

    return None


def build_dataset(category_kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (activations, labels) dataset across all trajectories and steps.

    Args:
        category_kwargs: Keyword args for load_activations_for_trajectory that
            specify which token category to load, e.g.
            {"prompt_suffix_indices": "-1"} or {"output_indices": "-1"}.

    Returns:
        Tuple of (activations tensor, labels tensor) where each row is one
        (trajectory, step) sample and labels are 0/1 for carrying_key.
    """
    trajectory_folders = discover_trajectory_folders(ACTIVATIONS_DIR)

    all_activations = []
    all_labels = []
    skipped_no_json = 0
    skipped_no_act = 0
    skipped_no_label = 0

    for traj_folder in tqdm(trajectory_folders, desc="Building dataset"):
        json_path = find_trajectory_json(traj_folder.name)
        if json_path is None:
            skipped_no_json += 1
            continue

        with open(json_path) as f:
            traj_data = json.load(f)

        steps = traj_data.get("steps", [])
        if not steps:
            skipped_no_json += 1
            continue

        for step_idx, step in enumerate(steps):
            # Get label
            label = step.get("carrying_key")
            if label is None:
                label = step.get("key_collected")
            if label is None:
                skipped_no_label += 1
                continue

            # Load activation for this specific step across all available layers
            activation = load_activations_for_trajectory(
                trajectory_folder=traj_folder,
                layers="all",
                steps=str(step_idx),
                prompt_prefix_indices=None,
                grid_state_indices=None,
                **category_kwargs,
            )

            if activation is None:
                skipped_no_act += 1
                continue

            all_activations.append(activation)
            all_labels.append(int(label))

    print(
        f"  Collected {len(all_activations)} samples "
        f"(skipped: {skipped_no_json} missing JSON, "
        f"{skipped_no_act} missing activation, "
        f"{skipped_no_label} missing label)"
    )

    if not all_activations:
        raise RuntimeError("No samples collected — check paths and data structure.")

    labels = torch.tensor(all_labels, dtype=torch.int64)
    unique, counts = torch.unique(labels, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist(), strict=False):
        name = "carrying_key=True " if u == 1 else "carrying_key=False"
        print(f"  {name}: {c} samples ({100 * c / len(labels):.1f}%)")

    return torch.stack(all_activations), labels


def main():
    # ── Build datasets ──────────────────────────────────────────────────────
    print("\n=== Building pre-reasoning dataset (prompt_suffix last token) ===")
    pre_acts, pre_labels = build_dataset({"prompt_suffix_indices": "-1", "output_indices": None})

    print("\n=== Building post-reasoning dataset (output last token) ===")
    post_acts, post_labels = build_dataset({"prompt_suffix_indices": None, "output_indices": "-1"})

    # Save to disk so train_key_collected_probe can load them
    pre_path = ACTIVATIONS_DIR / "pre_reasoning.pt"
    post_path = ACTIVATIONS_DIR / "post_reasoning.pt"

    torch.save({"activations": pre_acts, "labels": pre_labels, "probe_type": "key_collected"}, pre_path)
    torch.save({"activations": post_acts, "labels": post_labels, "probe_type": "key_collected"}, post_path)

    print(f"\nSaved pre-reasoning dataset  → {pre_path}")
    print(f"Saved post-reasoning dataset → {post_path}")

    # ── Train probes ─────────────────────────────────────────────────────────
    results = {}
    configs = [
        ("pre-reasoning", str(pre_path)),
        ("post-reasoning", str(post_path)),
    ]

    for condition, data_path in configs:
        results[condition] = {}
        for model_type in ["lr", "mlp"]:
            print(f"\n=== Training {model_type.upper()} probe on {condition} ===")
            probe = train_key_collected_probe(
                train_data_path=data_path,
                model_type=model_type,
                num_epochs=200,
                learning_rate=0.001,
                batch_size=256,
                hidden_dims="128",
                dropout=0.3,
                normalize=True,
                balance_classes=True,
                eval_split=0.2,
                seed=42,
                verbose=True,
            )
            results[condition][model_type] = {
                "accuracy": probe.results["final_accuracy"],
                "balanced_accuracy": probe.results["final_balanced_accuracy"],
                "best_accuracy": probe.results["best_eval_accuracy"],
                "best_balanced_accuracy": probe.results["best_balanced_accuracy"],
            }

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY-COLLECTED PROBE RESULTS")
    print("=" * 70)
    print(f"{'Condition':<20} {'Model':<6} {'Accuracy':>10} {'Bal. Acc':>10} {'Best Acc':>10} {'Best Bal':>10}")
    print("-" * 70)
    for condition, model_results in results.items():
        for model_type, metrics in model_results.items():
            print(
                f"{condition:<20} {model_type.upper():<6} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['balanced_accuracy']:>10.4f} "
                f"{metrics['best_accuracy']:>10.4f} "
                f"{metrics['best_balanced_accuracy']:>10.4f}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
