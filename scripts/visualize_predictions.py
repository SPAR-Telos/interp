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
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telos_interp.decoder_probe import (
    EOS_TOKEN_ID,
    ActionDecoderProbe,
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
        vocab_size = checkpoint.get("vocab_size", 7)
        n_layer = checkpoint.get("n_layer", 2)
        n_head = checkpoint.get("n_head", 4)
        num_memory_tokens = checkpoint.get("num_memory_tokens", 8)
        max_seq_len = checkpoint.get("max_seq_len", 15)
    else:
        state_dict = checkpoint
        # Infer parameters from state dict
        if "action_embeddings.weight" in state_dict:
            vocab_size, memory_token_dim = state_dict["action_embeddings.weight"].shape
        else:
            vocab_size = 7
            memory_token_dim = 360  # Default: 2880 / 8
        
        # Infer activation_dim and num_memory_tokens
        # Check decoder_layers to see the expected input dimension
        if "decoder_layers.layers.0.self_attn.in_proj_weight" in state_dict:
            # in_proj_weight shape is (3 * d_model, d_model) for query, key, value
            in_proj_dim = state_dict["decoder_layers.layers.0.self_attn.in_proj_weight"].shape[0] // 3
            memory_token_dim = in_proj_dim
        
        # Try to infer num_memory_tokens from activation_dim
        # Common values: 2880 / 8 = 360, 2880 / 4 = 720, etc.
        activation_dim = 2880  # Default
        num_memory_tokens = 8  # Default
        
        # Try to infer from position_embeddings (max_seq_len)
        if "position_embeddings.weight" in state_dict:
            max_seq_len, pos_dim = state_dict["position_embeddings.weight"].shape
            if pos_dim != memory_token_dim:
                memory_token_dim = pos_dim
        else:
            max_seq_len = 15  # Default from training
        
        # Infer num_memory_tokens: activation_dim must be divisible by num_memory_tokens
        # Try common values
        for num_tokens in [8, 4, 6, 10, 12, 16, 20]:
            if activation_dim % num_tokens == 0:
                inferred_dim = activation_dim // num_tokens
                if inferred_dim == memory_token_dim:
                    num_memory_tokens = num_tokens
                    break
        
        # Infer n_layer from decoder_layers
        n_layer = 2  # Default
        if "decoder_layers.layers.1.self_attn.in_proj_weight" in state_dict:
            n_layer = 2
        elif "decoder_layers.layers.3.self_attn.in_proj_weight" in state_dict:
            n_layer = 4
        else:
            n_layer = 2  # Default from training
        
        n_head = 4  # Default
    
    # Create model with inferred parameters
    model = ActionDecoderProbe(
        activation_dim=activation_dim,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        num_memory_tokens=num_memory_tokens,
        max_seq_len=max_seq_len,
    )
    
    # Try to load state dict, with fallback to CPU if device fails
    try:
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)
        # Test if device actually works
        if device == "cuda":
            test_tensor = torch.tensor([1.0]).to(device)
    except RuntimeError as e:
        if "cuda" in str(e).lower() or "CUDA" in str(e):
            print(f"⚠️  Warning: CUDA initialization failed: {e}")
            print("   Falling back to CPU for visualization.")
            device = "cpu"
            model = model.to(device)
        else:
            raise
    
    print(f"Model loaded: vocab_size={vocab_size}, activation_dim={activation_dim}, "
          f"n_layer={n_layer}, n_head={n_head}, num_memory_tokens={num_memory_tokens}, "
          f"max_seq_len={max_seq_len}, device={device}")
    
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
        default=None,
        help="Device to run on (auto-detects if not specified, falls back to CPU if CUDA fails)",
    )
    parser.add_argument(
        "--disable-heuristic-stopping",
        action="store_true",
        help="Disable heuristic stopping during generation",
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified (with fallback to CPU)
    if args.device is None:
        if torch.cuda.is_available():
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                args.device = "cuda"
            except (AssertionError, RuntimeError):
                args.device = "cpu"
                print("⚠️  CUDA not available, using CPU")
        else:
            args.device = "cpu"
    
    # Load model (will handle device fallback internally)
    model = load_model(args.model_path, args.device)
    
    # Use the actual device the model is on (may have fallen back to CPU)
    actual_device = next(model.parameters()).device
    if actual_device.type != args.device:
        print(f"⚠️  Model is on {actual_device}, not {args.device}")
        args.device = str(actual_device)
    
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
    
    # Load activations (load_activations_from_local handles both local and HF)
    print(f"\nLoading activations...")
    activations = load_activations_from_local(
        activations_dir=args.activations_dir,
        layer=args.layer,
        grid_indices=args.env_indices,
        hf_repo_id=args.hf_repo_id,  # Pass None for local, or repo_id for HF
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
        # Use the actual device the model is on
        actual_device = next(model.parameters()).device
        activation = activation.unsqueeze(0).to(actual_device).to(torch.float32)
        
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
    
    # 3. Prefix Accuracy by Sequence Position (TWO PLOTS: Sequence-level and Token-level)
    print("\nComputing prefix accuracy (sequence-level and token-level)...")
    
    # Find maximum sequence length from true actions (what we're trying to predict)
    max_seq_len = max(len(r["true_actions"]) for r in results if len(r["true_actions"]) > 0)
    max_seq_len = min(max_seq_len, args.max_length)
    
    # SEQUENCE-LEVEL: First X actions all correct
    prefix_seq_accuracies = []
    prefix_seq_counts = []
    baseline_seq_accuracies = []
    
    # TOKEN-LEVEL: Xth action correct
    token_accuracies = []
    token_counts = []
    baseline_token_accuracies = []
    
    # Compute both metrics for each position
    for pos in range(1, max_seq_len + 1):
        # SEQUENCE-LEVEL: Are the first X actions all correct?
        seq_correct_count = 0
        seq_total_count = 0
        
        # TOKEN-LEVEL: Is the Xth action correct?
        token_correct_count = 0
        token_total_count = 0
        
        for r in results:
            true_len = len(r["true_actions"])
            pred_len = len(r["pred_actions"])
            
            # SEQUENCE-LEVEL: Only consider sequences where true sequence has at least pos actions
            if true_len >= pos:
                seq_total_count += 1
                
                # If prediction is shorter than pos, it's automatically incorrect
                if pred_len < pos:
                    seq_correct_count += 0  # Incorrect (can't match first X if prediction is too short)
                else:
                    # Check if first pos actions match exactly
                    true_prefix = r["true_actions"][:pos]
                    pred_prefix = r["pred_actions"][:pos]
                    if true_prefix == pred_prefix:
                        seq_correct_count += 1
            
            # TOKEN-LEVEL: Is the pos-th action (0-indexed: pos-1) correct?
            if true_len >= pos:
                token_total_count += 1
                
                # If prediction is shorter than pos, the pos-th action doesn't exist (incorrect)
                if pred_len < pos:
                    token_correct_count += 0  # Incorrect (prediction too short)
                else:
                    # Check if the pos-th action (index pos-1) matches
                    if r["true_actions"][pos - 1] == r["pred_actions"][pos - 1]:
                        token_correct_count += 1
        
        # Store sequence-level results
        if seq_total_count > 0:
            seq_accuracy = seq_correct_count / seq_total_count
            prefix_seq_accuracies.append(seq_accuracy)
            prefix_seq_counts.append(seq_total_count)
        else:
            prefix_seq_accuracies.append(0.0)
            prefix_seq_counts.append(0)
        
        # Store token-level results
        if token_total_count > 0:
            token_accuracy = token_correct_count / token_total_count
            token_accuracies.append(token_accuracy)
            token_counts.append(token_total_count)
        else:
            token_accuracies.append(0.0)
            token_counts.append(0)
        
        # Compute baselines
        # Sequence-level: probability of getting first X actions all correct = (0.25)^X
        baseline_seq = (0.25) ** pos
        baseline_seq_accuracies.append(baseline_seq)
        
        # Token-level: probability of getting Xth action correct = 0.25 (always 25% for any position)
        baseline_token = 0.25
        baseline_token_accuracies.append(baseline_token)
    
    # Create TWO plots: Sequence-level and Token-level
    
    # ===== PLOT 1: SEQUENCE-LEVEL ACCURACY (First X actions all correct) =====
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Filter to only plot points with data (n > 0)
    x_values_seq = []
    model_accs_seq = []
    baseline_accs_seq = []
    counts_seq = []
    improvements_seq = []
    
    for i, (x, acc, count, baseline) in enumerate(zip(range(1, max_seq_len + 1), 
                                                      prefix_seq_accuracies, 
                                                      prefix_seq_counts, 
                                                      baseline_seq_accuracies)):
        if count > 0:  # Only include points with data
            x_values_seq.append(x)
            model_accs_seq.append(acc)
            baseline_accs_seq.append(baseline)
            counts_seq.append(count)
            improvements_seq.append(acc - baseline)
    
    # Fill area between model and baseline to show improvement
    ax.fill_between(x_values_seq, baseline_accs_seq, model_accs_seq, 
                    where=[m > b for m, b in zip(model_accs_seq, baseline_accs_seq)],
                    alpha=0.3, color='green', label='Above Baseline', zorder=1)
    ax.fill_between(x_values_seq, baseline_accs_seq, model_accs_seq, 
                    where=[m <= b for m, b in zip(model_accs_seq, baseline_accs_seq)],
                    alpha=0.2, color='red', label='Below Baseline', zorder=1)
    
    # Plot baseline (random chance: 25%^X for X actions)
    ax.plot(x_values_seq, baseline_accs_seq, "--", label=f"Random Baseline (25%^X)", 
            linewidth=2.5, color="#666666", alpha=0.8, zorder=2)
    
    # Plot model accuracy with better styling
    ax.plot(x_values_seq, model_accs_seq, "o-", label="Model Accuracy", 
            linewidth=3, markersize=12, color="#8B4CBF", zorder=4, 
            markerfacecolor="#8B4CBF", markeredgecolor="white", markeredgewidth=2,
            alpha=0.9)
    
    # Add percentage labels on points (for key points)
    for i, (x, acc, count) in enumerate(zip(x_values_seq, model_accs_seq, counts_seq)):
        # Only label every other point or key points to avoid clutter
        if i % 2 == 0 or acc > 0.3 or count < 50:  # Label sparse points or high accuracy
            ax.annotate(f"{acc:.1%}", xy=(x, acc), xytext=(x, acc + 0.08),
                       fontsize=9, ha='center', fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#8B4CBF', alpha=0.9, linewidth=1.5))
    
    # Add sample size annotations (smaller, less intrusive)
    for x, acc, count in zip(x_values_seq, model_accs_seq, counts_seq):
        y_offset = -0.08 if acc > 0.5 else 0.05
        ax.annotate(f"n={count}", xy=(x, acc), xytext=(x, acc + y_offset),
                   fontsize=8, ha='center', alpha=0.6, style='italic',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', 
                            edgecolor='none', alpha=0.5))
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    ax.set_xlabel("Minimum Sequence Length (first X actions)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
    ax.set_title("Prefix Sequence Accuracy\n(Are the first X actions ALL correct?)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    if len(x_values_seq) > 0:
        ax.set_xlim([min(x_values_seq) - 0.5, max(x_values_seq) + 0.5])
        ax.set_xticks(x_values_seq)
    
    # Better grid
    ax.grid(True, alpha=0.2, linestyle='-', axis='both', linewidth=0.5)
    ax.grid(True, alpha=0.4, linestyle='--', axis='y', which='major', linewidth=1)
    
    # Better legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, shadow=True, 
             fancybox=True, borderpad=1)
    
    # Add horizontal line at 25% for reference
    ax.axhline(y=0.25, color="#FF6B6B", linestyle=":", alpha=0.6, linewidth=2, 
               label="25% Reference", zorder=0)
    
    # Add summary statistics box
    if len(x_values_seq) > 0:
        max_improvement = max(improvements_seq)
        max_improvement_idx = improvements_seq.index(max_improvement)
        avg_improvement = sum(improvements_seq) / len(improvements_seq)
        
        stats_text = (f"Max Improvement: {max_improvement:.1%} at X={x_values_seq[max_improvement_idx]}\n"
                     f"Avg Improvement: {avg_improvement:.1%}\n"
                     f"Best Accuracy: {max(model_accs_seq):.1%} at X={x_values_seq[model_accs_seq.index(max(model_accs_seq))]}")
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                        edgecolor='#8B4CBF', alpha=0.9, linewidth=2))
        
        explanation = ("SEQUENCE-LEVEL: For each X, shows % of sequences where\n"
                      "the first X predicted actions exactly match the first X true actions.")
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', alpha=0.8,
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF9E6', 
                        edgecolor='#8B4CBF', alpha=0.9, linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "prefix_sequence_accuracy.png"), dpi=150)
    print(f"Saved: {args.output_dir}/prefix_sequence_accuracy.png")
    plt.close()
    
    # ===== PLOT 2: TOKEN-LEVEL ACCURACY (Xth action correct) =====
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Filter to only plot points with data (n > 0)
    x_values_token = []
    model_accs_token = []
    baseline_accs_token = []
    counts_token = []
    improvements_token = []
    
    for i, (x, acc, count, baseline) in enumerate(zip(range(1, max_seq_len + 1), 
                                                      token_accuracies, 
                                                      token_counts, 
                                                      baseline_token_accuracies)):
        if count > 0:  # Only include points with data
            x_values_token.append(x)
            model_accs_token.append(acc)
            baseline_accs_token.append(baseline)
            counts_token.append(count)
            improvements_token.append(acc - baseline)
    
    # Fill area between model and baseline to show improvement
    ax.fill_between(x_values_token, baseline_accs_token, model_accs_token, 
                    where=[m > b for m, b in zip(model_accs_token, baseline_accs_token)],
                    alpha=0.3, color='green', label='Above Baseline', zorder=1)
    ax.fill_between(x_values_token, baseline_accs_token, model_accs_token, 
                    where=[m <= b for m, b in zip(model_accs_token, baseline_accs_token)],
                    alpha=0.2, color='red', label='Below Baseline', zorder=1)
    
    # Plot baseline (always 25% for any position)
    ax.axhline(y=0.25, color="#666666", linestyle="--", linewidth=2.5, 
               alpha=0.8, label="Random Baseline (25%)", zorder=2)
    
    # Plot model accuracy with better styling
    ax.plot(x_values_token, model_accs_token, "o-", label="Model Accuracy", 
            linewidth=3, markersize=12, color="#2E86AB", zorder=4, 
            markerfacecolor="#2E86AB", markeredgecolor="white", markeredgewidth=2,
            alpha=0.9)
    
    # Add percentage labels on points (for key points)
    for i, (x, acc, count) in enumerate(zip(x_values_token, model_accs_token, counts_token)):
        # Label every point or key points
        if i % 2 == 0 or acc > 0.35 or count < 50:  # Label sparse points or high accuracy
            ax.annotate(f"{acc:.1%}", xy=(x, acc), xytext=(x, acc + 0.08),
                       fontsize=9, ha='center', fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#2E86AB', alpha=0.9, linewidth=1.5))
    
    # Add sample size annotations (smaller, less intrusive)
    for x, acc, count in zip(x_values_token, model_accs_token, counts_token):
        y_offset = -0.08 if acc > 0.5 else 0.05
        ax.annotate(f"n={count}", xy=(x, acc), xytext=(x, acc + y_offset),
                   fontsize=8, ha='center', alpha=0.6, style='italic',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', 
                            edgecolor='none', alpha=0.5))
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    ax.set_xlabel("Action Position (Xth action)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=14, fontweight='bold')
    ax.set_title("Token-Level Accuracy by Position\n(Is the Xth action correct?)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    if len(x_values_token) > 0:
        ax.set_xlim([min(x_values_token) - 0.5, max(x_values_token) + 0.5])
        ax.set_xticks(x_values_token)
    
    # Better grid
    ax.grid(True, alpha=0.2, linestyle='-', axis='both', linewidth=0.5)
    ax.grid(True, alpha=0.4, linestyle='--', axis='y', which='major', linewidth=1)
    
    # Better legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, shadow=True, 
             fancybox=True, borderpad=1)
    
    # Add summary statistics box
    if len(x_values_token) > 0:
        max_improvement = max(improvements_token)
        max_improvement_idx = improvements_token.index(max_improvement)
        avg_improvement = sum(improvements_token) / len(improvements_token)
        positions_above_baseline = sum(1 for imp in improvements_token if imp > 0)
        
        stats_text = (f"Max Improvement: {max_improvement:.1%} at pos {x_values_token[max_improvement_idx]}\n"
                     f"Avg Improvement: {avg_improvement:.1%}\n"
                     f"Positions Above Baseline: {positions_above_baseline}/{len(x_values_token)}\n"
                     f"Best Accuracy: {max(model_accs_token):.1%} at pos {x_values_token[model_accs_token.index(max(model_accs_token))]}")
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                        edgecolor='#2E86AB', alpha=0.9, linewidth=2))
        
        explanation = ("TOKEN-LEVEL: For each position X, shows % of sequences where\n"
                      "the Xth predicted action matches the Xth true action.")
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', alpha=0.8,
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#E6F3FF', 
                        edgecolor='#2E86AB', alpha=0.9, linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "prefix_token_accuracy.png"), dpi=150)
    print(f"Saved: {args.output_dir}/prefix_token_accuracy.png")
    plt.close()
    
    # Print summary statistics for both
    print("\n" + "=" * 60)
    print("SEQUENCE-LEVEL Prefix Accuracy Summary:")
    print("(Are the first X actions ALL correct?)")
    print("=" * 60)
    for i, (prefix_len, acc, count, baseline) in enumerate(zip(x_values_seq, model_accs_seq, counts_seq, baseline_accs_seq)):
        if count > 0:
            improvement = acc - baseline
            print(f"First {prefix_len:2d} actions: Accuracy={acc:.3f} (n={count:3d}), "
                  f"Baseline={baseline:.3f}, Improvement={improvement:+.3f}")
    
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL Accuracy Summary:")
    print("(Is the Xth action correct?)")
    print("=" * 60)
    for i, (pos, acc, count, baseline) in enumerate(zip(x_values_token, model_accs_token, counts_token, baseline_accs_token)):
        if count > 0:
            improvement = acc - baseline
            print(f"Position {pos:2d}: Accuracy={acc:.3f} (n={count:3d}), "
                  f"Baseline={baseline:.3f}, Improvement={improvement:+.3f}")
    print("=" * 60)
    
    # 4. Summary statistics
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

