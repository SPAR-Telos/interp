"""Run door-open binary probes (LR + MLP) for pre- and post-reasoning activations.

Label derivation: 'D' present in grid_state → door closed (0), absent → door open (1).
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
from telos_interp.grid_utils import CELL_SYMBOL_TO_ID, parse_grid_state

ACTIVATIONS_DIR = Path("data/activations/activations_key_door_env_100")
TRAJECTORIES_DIR = Path("data/trajectories/trajectories_key_door_100/trajectories_key_door")

DOOR_ID = CELL_SYMBOL_TO_ID["D"]  # = 4

_LABEL_NAMES = {0: "door_closed", 1: "door_open"}


def find_trajectory_json(folder_name: str) -> Path | None:
    """Map an activation folder name to its trajectory JSON file."""
    exact = TRAJECTORIES_DIR / f"{folder_name}.json"
    if exact.exists():
        return exact
    parts = folder_name.rsplit("_doorkey_", 1)
    if len(parts) == 2:
        candidate = TRAJECTORIES_DIR / f"{parts[0]}_doorkey_keepdoor_{parts[1]}.json"
        if candidate.exists():
            return candidate
    return None


def get_door_open_label(step: dict) -> int | None:
    """Return 0 if door is closed (D present in grid), 1 if door is open (D absent)."""
    grid_state = step.get("grid_state")
    if not grid_state:
        return None
    triples = parse_grid_state(grid_state)
    door_present = any(triple[2] == DOOR_ID for triple in triples)
    return 0 if door_present else 1


def build_dataset(category_kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (activations, labels) dataset across all trajectories and steps."""
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
            label = get_door_open_label(step)
            if label is None:
                skipped_no_label += 1
                continue

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
            all_labels.append(label)

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
        print(f"  {_LABEL_NAMES[u]}: {c} samples ({100 * c / len(labels):.1f}%)")

    return torch.stack(all_activations), labels


def main():
    # ── Build datasets ──────────────────────────────────────────────────────
    print("\n=== Building pre-reasoning dataset (prompt_suffix last token) ===")
    pre_acts, pre_labels = build_dataset({"prompt_suffix_indices": "-1", "output_indices": None})

    print("\n=== Building post-reasoning dataset (output last token) ===")
    post_acts, post_labels = build_dataset({"prompt_suffix_indices": None, "output_indices": "-1"})

    pre_path = ACTIVATIONS_DIR / "door_pre_reasoning.pt"
    post_path = ACTIVATIONS_DIR / "door_post_reasoning.pt"

    torch.save({"activations": pre_acts, "labels": pre_labels, "probe_type": "key_collected"}, pre_path)
    torch.save({"activations": post_acts, "labels": post_labels, "probe_type": "key_collected"}, post_path)

    print(f"\nSaved pre-reasoning dataset  → {pre_path}")
    print(f"Saved post-reasoning dataset → {post_path}")

    # ── Train probes ─────────────────────────────────────────────────────────
    results = {}
    configs = [("pre-reasoning", str(pre_path)), ("post-reasoning", str(post_path))]

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
                output_path=str(ACTIVATIONS_DIR / f"door_open_probe_{model_type}.pt"),
                verbose=True,
            )
            results[condition][model_type] = probe.results

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DOOR-OPEN PROBE RESULTS")
    print("=" * 70)
    print(f"{'Condition':<20} {'Model':<6} {'Accuracy':>10} {'Bal. Acc':>10}")
    print("-" * 50)
    for condition, mr in results.items():
        for model_type, r in mr.items():
            print(
                f"{condition:<20} {model_type.upper():<6} "
                f"{r['final_accuracy']:>10.4f} "
                f"{r['final_balanced_accuracy']:>10.4f}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
