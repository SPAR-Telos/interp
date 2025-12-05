#!/usr/bin/env python3
"""
Visualize decoder probe predictions vs ground truth with distance-to-goal tracking.

Creates visualizations showing:
1. Distance to goal at each action step (predicted vs ground truth)
2. Trajectory comparison plots
3. Summary statistics across test set
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telos_interp.decoder_probe import (
    EOS_TOKEN_ID,
    ActionDecoderProbe,
    load_activations_from_hf,
    load_activations_from_local,
)


def parse_position(pos_str):
    """Parse position string like '(1, 2)' or '(1,2)' to tuple."""
    if pd.isna(pos_str) or pos_str == "":
        return None
    try:
        # Try to parse as tuple string
        if isinstance(pos_str, str):
            # Remove parentheses and split
            pos_str = pos_str.strip()
            if pos_str.startswith("(") and pos_str.endswith(")"):
                pos_str = pos_str[1:-1]
            parts = pos_str.split(",")
            if len(parts) == 2:
                return (int(parts[0].strip()), int(parts[1].strip()))
        # Try ast.literal_eval as fallback
        return ast.literal_eval(str(pos_str))
    except:
        return None


def apply_action(pos, action):
    """Apply action to position. Returns new position.
    
    Actions: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    """
    x, y = pos
    if action == 0:  # LEFT
        return (x - 1, y)
    elif action == 1:  # RIGHT
        return (x + 1, y)
    elif action == 2:  # UP
        return (x, y - 1)
    elif action == 3:  # DOWN
        return (x, y + 1)
    else:
        return pos  # Invalid action, stay in place


def euclidean_distance(pos1, pos2):
    """Calculate euclidean distance between two positions."""
    if pos1 is None or pos2 is None:
        return None
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def simulate_trajectory(start_pos, actions, goal_pos):
    """Simulate agent trajectory and calculate distance to goal at each step.
    
    Returns:
        positions: List of (x, y) positions at each step
        distances: List of euclidean distances to goal at each step
    """
    if start_pos is None:
        return [], []
    
    positions = [start_pos]
    distances = [euclidean_distance(start_pos, goal_pos)]
    
    current_pos = start_pos
    for action in actions:
        if action == EOS_TOKEN_ID or action < 0 or action > 3:
            break
        current_pos = apply_action(current_pos, action)
        positions.append(current_pos)
        distances.append(euclidean_distance(current_pos, goal_pos))
    
    return positions, distances


def load_model(model_path, device):
    """Load trained decoder probe model."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        activation_dim = checkpoint.get("activation_dim", 2880)
        vocab_size = checkpoint.get("vocab_size", 5)
        n_layer = checkpoint.get("n_layer", 4)
        n_head = checkpoint.get("n_head", 4)
        n_embd = checkpoint.get("n_embd", 256)
        max_seq_len = checkpoint.get("max_seq_len", 100)
    else:
        state_dict = checkpoint
        if "activation_projection.weight" in state_dict:
            n_embd, activation_dim = state_dict["activation_projection.weight"].shape
        else:
            activation_dim = 2880
            n_embd = 256
        
        if "action_embeddings.weight" in state_dict:
            vocab_size, _ = state_dict["action_embeddings.weight"].shape
        else:
            vocab_size = 5
        
        n_layer = 4
        n_head = 4
        max_seq_len = 100
    
    model = ActionDecoderProbe(
        activation_dim=activation_dim,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_seq_len=max_seq_len,
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print(f"Model loaded: vocab_size={vocab_size}, activation_dim={activation_dim}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Visualize decoder probe predictions")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to test CSV file with grid information",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer number for activations",
    )
    parser.add_argument(
        "--activations-dir",
        type=str,
        required=True,
        help="Subdirectory path for activations (e.g., '7by7testgrids/full_prompt_activations')",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace repository ID (e.g., 'project-telos/decoder'). If None, uses local filesystem.",
    )
    parser.add_argument(
        "--env-indices",
        type=int,
        nargs="+",
        default=None,
        help="List of env_idx values to visualize. If None, uses all in CSV.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save visualization plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--disable-heuristic-stopping",
        action="store_true",
        help="Disable heuristic stopping during generation",
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Load test CSV
    print(f"\nLoading test data from: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    
    # Filter to trajectory_step == 0 (initial state)
    if "trajectory_step" in df.columns:
        df = df[df["trajectory_step"] == 0]
    
    # Get env_indices
    if args.env_indices is None:
        args.env_indices = sorted(df["env_idx"].unique().tolist())
        print(f"Testing all {len(args.env_indices)} grids in CSV")
    else:
        available = set(df["env_idx"].unique())
        args.env_indices = [idx for idx in args.env_indices if idx in available]
        print(f"Testing {len(args.env_indices)} specified grids")
    
    # Load activations
    print(f"\nLoading activations...")
    if args.hf_repo_id:
        activations = load_activations_from_hf(
            repo_id=args.hf_repo_id,
            path_in_repo=f"activations/{args.activations_dir}",
            layer=args.layer,
            grid_indices=args.env_indices,
        )
    else:
        activations = load_activations_from_local(
            activations_dir=args.activations_dir,
            layer=args.layer,
            grid_indices=args.env_indices,
        )
    
    print(f"Loaded {len(activations)} activations")
    
    # Process each grid
    results = []
    
    for env_idx in sorted(args.env_indices):
        if env_idx not in activations:
            print(f"Warning: No activation for env_idx {env_idx}, skipping...")
            continue
        
        # Get grid info from CSV
        row = df[df["env_idx"] == env_idx]
        if len(row) == 0:
            print(f"Warning: No CSV data for env_idx {env_idx}, skipping...")
            continue
        
        grid_info = row.iloc[0]
        
        # Parse positions
        start_pos = parse_position(grid_info.get("start_pos"))
        goal_pos = parse_position(grid_info.get("goal_pos"))
        
        if start_pos is None or goal_pos is None:
            print(f"Warning: Missing positions for env_idx {env_idx}, skipping...")
            continue
        
        # Parse ground truth actions
        true_actions = []
        if "action_sequence" in grid_info and pd.notna(grid_info["action_sequence"]):
            try:
                true_actions = json.loads(grid_info["action_sequence"])
                # Remove EOS token
                true_actions = [a for a in true_actions if a != EOS_TOKEN_ID and 0 <= a <= 3]
            except:
                pass
        
        # Generate prediction
        activation = activations[env_idx]
        if activation.ndim > 1:
            activation = activation[-1]
        activation = activation.unsqueeze(0).to(args.device).to(torch.float32)
        
        with torch.no_grad():
            pred_actions_tensor = model.generate(
                activation,
                max_length=args.max_length,
                disable_heuristic_stopping=args.disable_heuristic_stopping,
            )
        
        pred_actions = pred_actions_tensor[0].cpu().tolist()
        pred_actions = [a for a in pred_actions if a != -100 and a != EOS_TOKEN_ID and 0 <= a <= 3]
        
        # Simulate trajectories
        true_positions, true_distances = simulate_trajectory(start_pos, true_actions, goal_pos)
        pred_positions, pred_distances = simulate_trajectory(start_pos, pred_actions, goal_pos)
        
        # Store results
        results.append({
            "env_idx": env_idx,
            "start_pos": start_pos,
            "goal_pos": goal_pos,
            "true_actions": true_actions,
            "pred_actions": pred_actions,
            "true_distances": true_distances,
            "pred_distances": pred_distances,
            "true_final_distance": true_distances[-1] if true_distances else None,
            "pred_final_distance": pred_distances[-1] if pred_distances else None,
            "true_reached_goal": true_distances[-1] == 0.0 if true_distances else False,
            "pred_reached_goal": pred_distances[-1] == 0.0 if pred_distances else False,
        })
        
        print(f"Processed env_idx {env_idx}: true_len={len(true_actions)}, pred_len={len(pred_actions)}")
    
    print(f"\nProcessed {len(results)} grids")
    
    # Create visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter results for meaningful comparisons
    # Only show grids where:
    # 1. Predictions exist (not empty)
    # 2. If ground truth has > 2 steps, predictions should have > 1 step
    filtered_results = []
    for r in results:
        true_len = len(r["true_distances"])
        pred_len = len(r["pred_distances"])
        
        # Skip if no predictions
        if pred_len == 0:
            continue
        
        # Skip if ground truth has > 2 steps but prediction only has 1 step
        if true_len > 2 and pred_len <= 1:
            continue
        
        filtered_results.append(r)
    
    print(f"\nFiltered to {len(filtered_results)} grids with meaningful predictions (from {len(results)} total)")
    
    # 1. Distance-to-goal over steps (individual plots for first N grids)
    n_individual_plots = min(10, len(filtered_results))
    
    if n_individual_plots == 0:
        print("Warning: No grids meet the filtering criteria. Skipping individual plots.")
    else:
        # Calculate grid layout
        n_cols = 5
        n_rows = (n_individual_plots + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_individual_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_individual_plots):
            r = filtered_results[i]
            ax = axes[i]
            
            true_steps = list(range(len(r["true_distances"])))
            pred_steps = list(range(len(r["pred_distances"])))
            
            ax.plot(true_steps, r["true_distances"], "o-", label="Ground Truth", linewidth=2, markersize=6, color="blue")
            ax.plot(pred_steps, r["pred_distances"], "s-", label="Prediction", linewidth=2, markersize=6, color="orange")
            ax.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Goal")
            ax.set_xlabel("Action Step")
            ax.set_ylabel("Distance to Goal")
            ax.set_title(f"Grid {r['env_idx']}\n(True: {len(r['true_actions'])} actions, Pred: {len(r['pred_actions'])} actions)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_individual_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "distance_to_goal_individual.png"), dpi=150)
        print(f"Saved: {args.output_dir}/distance_to_goal_individual.png")
        plt.close()
    
    # 2. Aggregate distance-to-goal comparison (use filtered results for better visualization)
    # But also show all results for aggregate stats
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average distance over steps (use all results for aggregate)
    max_steps = max(max(len(r["true_distances"]), len(r["pred_distances"])) for r in results)
    
    true_avg_distances = []
    pred_avg_distances = []
    true_std_distances = []
    pred_std_distances = []
    
    for step in range(max_steps):
        true_dists_at_step = [r["true_distances"][step] for r in results if step < len(r["true_distances"])]
        pred_dists_at_step = [r["pred_distances"][step] for r in results if step < len(r["pred_distances"])]
        
        if true_dists_at_step:
            true_avg_distances.append(np.mean(true_dists_at_step))
            true_std_distances.append(np.std(true_dists_at_step))
        else:
            true_avg_distances.append(np.nan)
            true_std_distances.append(0)
        
        if pred_dists_at_step:
            pred_avg_distances.append(np.mean(pred_dists_at_step))
            pred_std_distances.append(np.std(pred_dists_at_step))
        else:
            pred_avg_distances.append(np.nan)
            pred_std_distances.append(0)
    
    steps = list(range(max_steps))
    ax1.plot(steps, true_avg_distances, "o-", label="Ground Truth (avg)", linewidth=2)
    ax1.fill_between(steps, 
                     [t - s for t, s in zip(true_avg_distances, true_std_distances)],
                     [t + s for t, s in zip(true_avg_distances, true_std_distances)],
                     alpha=0.2)
    ax1.plot(steps, pred_avg_distances, "s-", label="Prediction (avg)", linewidth=2)
    ax1.fill_between(steps,
                     [p - s for p, s in zip(pred_avg_distances, pred_std_distances)],
                     [p + s for p, s in zip(pred_avg_distances, pred_std_distances)],
                     alpha=0.2)
    ax1.axhline(y=0, color="green", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Action Step")
    ax1.set_ylabel("Average Distance to Goal")
    ax1.set_title("Average Distance to Goal Over Steps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final distance distribution
    true_final_dists = [r["true_final_distance"] for r in results if r["true_final_distance"] is not None]
    pred_final_dists = [r["pred_final_distance"] for r in results if r["pred_final_distance"] is not None]
    
    ax2.hist(true_final_dists, bins=20, alpha=0.6, label="Ground Truth", color="blue")
    ax2.hist(pred_final_dists, bins=20, alpha=0.6, label="Prediction", color="orange")
    ax2.set_xlabel("Final Distance to Goal")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Final Distance Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "distance_to_goal_aggregate.png"), dpi=150)
    print(f"Saved: {args.output_dir}/distance_to_goal_aggregate.png")
    plt.close()
    
    # 3. Summary statistics
    true_reached = sum(1 for r in results if r["true_reached_goal"])
    pred_reached = sum(1 for r in results if r["pred_reached_goal"])
    
    true_final_dists = [r["true_final_distance"] for r in results if r["true_final_distance"] is not None]
    pred_final_dists = [r["pred_final_distance"] for r in results if r["pred_final_distance"] is not None]
    
    true_avg_final = np.mean(true_final_dists) if true_final_dists else None
    pred_avg_final = np.mean(pred_final_dists) if pred_final_dists else None
    
    true_avg_len = np.mean([len(r["true_actions"]) for r in results])
    pred_avg_len = np.mean([len(r["pred_actions"]) for r in results])
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total grids tested: {len(results)}")
    print(f"Grids with meaningful predictions (shown in individual plots): {len(filtered_results)}")
    print(f"\nGoal Reached:")
    print(f"  Ground Truth: {true_reached}/{len(results)} ({true_reached/len(results)*100:.1f}%)")
    print(f"  Prediction:   {pred_reached}/{len(results)} ({pred_reached/len(results)*100:.1f}%)")
    print(f"\nAverage Final Distance to Goal:")
    print(f"  Ground Truth: {true_avg_final:.2f}")
    print(f"  Prediction:   {pred_avg_final:.2f}")
    print(f"\nAverage Sequence Length:")
    print(f"  Ground Truth: {true_avg_len:.1f}")
    print(f"  Prediction:   {pred_avg_len:.1f}")
    print("=" * 60)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(args.output_dir, "prediction_summary.csv"), index=False)
    print(f"\nSaved detailed results to: {args.output_dir}/prediction_summary.csv")
    
    print(f"\n✅ Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

