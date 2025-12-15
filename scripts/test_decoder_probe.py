#!/usr/bin/env python3
"""
Test a trained decoder probe model by providing activations and getting action sequences.

Usage:
    python scripts/test_decoder_probe.py \
        --model-path models/decoder_probe_layer_20.pt \
        --activation-path data/activations/gpt-oss-20b/7by7testgrids/full_prompt_activations/env_0_step_0/layer_20/activations.pt \
        --max-length 50

Or test on multiple grids:
    python scripts/test_decoder_probe.py \
        --model-path models/decoder_probe_layer_20.pt \
        --activations-dir data/activations/gpt-oss-20b/7by7testgrids/full_prompt_activations \
        --layer 20 \
        --env-indices 0 1 2 3 4 \
        --max-length 50
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telos_interp.decoder_probe import (
    EOS_TOKEN_ID,
    ActionDecoderProbe,
    load_activations,
    load_activations_from_local,
    load_activations_from_hf,
)


def main():
    parser = argparse.ArgumentParser(description="Test decoder probe model with activations")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--activation-path",
        type=str,
        default=None,
        help="Path to single activation file (.pt file)",
    )
    parser.add_argument(
        "--activations-dir",
        type=str,
        default=None,
        help="Directory containing activation files (for testing multiple grids) or subdirectory path for HF",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer number (required if using activations-dir or hf-repo-id)",
    )
    parser.add_argument(
        "--env-indices",
        type=int,
        nargs="+",
        default=None,
        help="List of env_idx values to test (e.g., 0 1 2 3). If not provided and using --test-csv, tests all grids in CSV.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to test CSV file with grid information (e.g., telos_interp/grids/7by7testgrids.csv)",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace repository ID for loading activations (e.g., 'project-telos/decoder'). If provided, loads from HF instead of local filesystem.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum length of generated action sequence (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (default: 1.0, use 0.0 for greedy)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence (probability) to continue generation (default: 0.0, disabled). Lower values allow more uncertain predictions.",
    )
    parser.add_argument(
        "--max-repetition",
        type=int,
        default=10,
        help="Maximum consecutive identical actions before stopping (default: 10, higher = less aggressive)",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=2.0,
        help="Maximum entropy (uncertainty) before stopping (default: 2.0, higher = less aggressive). For vocab_size=5, max entropy ≈ 1.609",
    )
    parser.add_argument(
        "--disable-heuristic-stopping",
        action="store_true",
        help="Disable heuristic stopping (confidence, entropy, repetition). Only stop on EOS token or max_length.",
    )
    parser.add_argument(
        "--max-length-override",
        type=int,
        default=10,
        help="Hard limit on sequence length based on training data stats (default: 10, ~90th percentile)",
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=None,
        help="Number of decoder layers (inferred from checkpoint if not provided)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Max sequence length for position embeddings (inferred from checkpoint if not provided)",
    )
    parser.add_argument(
        "--num-memory-tokens",
        type=int,
        default=None,
        help="Number of memory tokens (inferred from checkpoint if not provided)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Testing Decoder Probe")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Max length: {args.max_length}")
    print("=" * 60)

    # Load model
    print("\n📥 Loading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Extract model parameters from checkpoint
    # The checkpoint might be just state_dict or full model
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # Try to get activation_dim from checkpoint metadata
        activation_dim = checkpoint.get("activation_dim", 2880)  # Default for gpt-oss-20b
        vocab_size = checkpoint.get("vocab_size", 7)
        n_layer = checkpoint.get("n_layer", 4)
        n_head = checkpoint.get("n_head", 4)
        num_memory_tokens = checkpoint.get("num_memory_tokens", 8)
        max_seq_len = checkpoint.get("max_seq_len", 100)
    else:
        # It's a state_dict directly
        state_dict = checkpoint
        
        # Default values
        activation_dim = 2880  # Default for gpt-oss-20b
        vocab_size = 7
        n_head = 4
        
        # action_embeddings.weight shape is (vocab_size, memory_token_dim)
        # memory_token_dim = activation_dim // num_memory_tokens
        if "action_embeddings.weight" in state_dict:
            vocab_size, memory_token_dim = state_dict["action_embeddings.weight"].shape
        else:
            memory_token_dim = 360  # Default: 2880 / 8
        
        # Infer num_memory_tokens from memory_token_dim
        # num_memory_tokens = activation_dim // memory_token_dim
        num_memory_tokens = activation_dim // memory_token_dim
        
        # Infer n_layer from state_dict (count decoder layers)
        layer_indices = set()
        for key in state_dict.keys():
            if "decoder_layers.layers." in key:
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        n_layer = len(layer_indices) if layer_indices else 4
        
        # Infer max_seq_len from position_embeddings.weight shape
        if "position_embeddings.weight" in state_dict:
            max_seq_len, _ = state_dict["position_embeddings.weight"].shape
        else:
            max_seq_len = 100

    # Override with command-line arguments if provided
    if args.n_layer is not None:
        n_layer = args.n_layer
    if args.max_seq_len is not None:
        max_seq_len = args.max_seq_len
    if args.num_memory_tokens is not None:
        num_memory_tokens = args.num_memory_tokens
    
    memory_token_dim = activation_dim // num_memory_tokens
    
    print(f"Model parameters:")
    print(f"  Activation dim: {activation_dim}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Layers: {n_layer}")
    print(f"  Heads: {n_head}")
    print(f"  Memory tokens: {num_memory_tokens} x {memory_token_dim} dims")
    print(f"  Max seq len: {max_seq_len}")

    # Create model
    model = ActionDecoderProbe(
        activation_dim=activation_dim,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        num_memory_tokens=num_memory_tokens,
        max_seq_len=max_seq_len,
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(args.device)
    print("✅ Model loaded successfully")

    # Load activations
    if args.activation_path:
        # Single activation file
        print(f"\n📥 Loading activation from: {args.activation_path}")
        activation = load_activations(args.activation_path)
        if activation.ndim > 1:
            activation = activation[-1]  # Take last token if multi-token
        activation = activation.unsqueeze(0).to(args.device).to(torch.float32)  # Add batch dimension
        
        print(f"Activation shape: {activation.shape}")
        
        # Try to extract env_idx from path and load corresponding grid from CSV
        import re
        
        # Extract env_idx from path like "env_0_step_0/layer_20/activations.pt"
        env_match = re.search(r'env_(\d+)', args.activation_path)
        step_match = re.search(r'step_(\d+)', args.activation_path)
        
        if env_match:
            env_idx = int(env_match.group(1))
            trajectory_step = int(step_match.group(1)) if step_match else 0
            
            # Try to find CSV file (check common locations)
            csv_paths = [
                "outputs/7by7testgrids.csv",
                "outputs/7by7traingrids.csv",
                "outputs/7by7valgrids.csv",
            ]
            
            grid_info = None
            for csv_path in csv_paths:
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        row = df[(df["env_idx"] == env_idx) & (df["trajectory_step"] == trajectory_step)]
                        if len(row) > 0:
                            grid_info = row.iloc[0]
                            break
                    except Exception as e:
                        continue
            
            if grid_info is not None:
                print(f"\n📊 Grid Information (env_idx={env_idx}, step={trajectory_step}):")
                print("=" * 60)
                
                # Print grid observation
                if "fo_observation" in grid_info:
                    print("Grid:")
                    print(grid_info["fo_observation"])
                
                # Print start and goal positions
                if "start_pos" in grid_info and pd.notna(grid_info["start_pos"]):
                    print(f"\nStart position: {grid_info['start_pos']}")
                if "goal_pos" in grid_info and pd.notna(grid_info["goal_pos"]):
                    print(f"Goal position: {grid_info['goal_pos']}")
                
                # Print optimal trajectory length
                if "optimal_trajectory_length" in grid_info and pd.notna(grid_info["optimal_trajectory_length"]):
                    print(f"Optimal trajectory length: {int(grid_info['optimal_trajectory_length'])}")
                
                # Print ground truth action sequence if available
                if "action_sequence" in grid_info and pd.notna(grid_info["action_sequence"]):
                    import json
                    try:
                        true_actions = json.loads(grid_info["action_sequence"])
                        action_names = ["LEFT", "RIGHT", "UP", "DOWN"]  # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
                        true_seq_str = " → ".join([action_names[a] for a in true_actions])
                        print(f"\nGround truth actions ({len(true_actions)}): {true_actions}")
                        print(f"Ground truth sequence: {true_seq_str}")
                    except:
                        pass
                
                print("=" * 60)
        
        # Generate actions
        print(f"\n🚀 Generating action sequence...")
        if not args.disable_heuristic_stopping:
            print(f"Early stopping: min_confidence={args.min_confidence}, max_repetition={args.max_repetition}, entropy_threshold={args.entropy_threshold}")
        else:
            print("Heuristic stopping disabled - only stopping on EOS token or max_length")
        with torch.no_grad():
            actions = model.generate(
                activation,
                max_length=min(args.max_length, max_seq_len - 1),
                temperature=args.temperature,
                min_confidence=args.min_confidence,
                max_repetition=args.max_repetition,
                entropy_threshold=args.entropy_threshold,
                max_length_override=args.max_length_override,
                disable_heuristic_stopping=args.disable_heuristic_stopping,
            )
        
        actions_list = actions[0].cpu().tolist()
        print(f"\n✅ Generated {len(actions_list)} actions:")
        print(f"Actions: {actions_list}")
        print(f"\nAction sequence (as directions):")
        action_names = ["LEFT", "RIGHT", "UP", "DOWN"]  # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
        action_str = " → ".join([action_names[a] for a in actions_list])
        print(f"  {action_str}")
        
    elif (args.activations_dir and args.layer is not None) or (args.test_csv and args.layer is not None):
        # Multiple activations from directory or CSV
        # Load CSV for grid information
        if args.test_csv:
            if not os.path.exists(args.test_csv):
                print(f"Error: Test CSV file not found: {args.test_csv}")
                sys.exit(1)
            print(f"\n📥 Loading grid information from: {args.test_csv}")
            df_grids = pd.read_csv(args.test_csv)
            
            # Infer activations_dir from CSV filename if not provided
            if not args.activations_dir:
                # Extract directory name from CSV filename (e.g., "7by7testgrids.csv" -> "7by7testgrids/full_prompt_activations")
                csv_basename = os.path.basename(args.test_csv)
                if "test" in csv_basename.lower():
                    args.activations_dir = "7by7testgrids/full_prompt_activations"
                elif "val" in csv_basename.lower():
                    args.activations_dir = "7by7valgrids/full_prompt_activations"
                elif "train" in csv_basename.lower():
                    args.activations_dir = "7by7traingrids/full_prompt_activations"
                else:
                    print("Warning: Could not infer activations_dir from CSV filename. Please provide --activations-dir")
                    if not args.hf_repo_id:
                        print("Error: --activations-dir required when not using --hf-repo-id")
                        sys.exit(1)
            
            # Get env_indices from CSV if not provided
            if args.env_indices is None:
                args.env_indices = sorted(df_grids["env_idx"].unique().tolist())
                print(f"Found {len(args.env_indices)} grids in CSV. Testing all of them.")
            else:
                # Filter to only grids that exist in CSV
                available_indices = set(df_grids["env_idx"].unique())
                args.env_indices = [idx for idx in args.env_indices if idx in available_indices]
                if len(args.env_indices) == 0:
                    print("Error: None of the specified env_indices found in CSV")
                    sys.exit(1)
        else:
            # Try to find CSV file (check common locations)
            csv_paths = [
                "telos_interp/grids/7by7testgrids.csv",
                "telos_interp/grids/7by7traingrids.csv",
                "telos_interp/grids/7by7valgrids.csv",
                "outputs/7by7testgrids.csv",
                "outputs/7by7traingrids.csv",
                "outputs/7by7valgrids.csv",
            ]
            df_grids = None
            for csv_path in csv_paths:
                if os.path.exists(csv_path):
                    try:
                        df_grids = pd.read_csv(csv_path)
                        print(f"Found grid CSV: {csv_path}")
                        break
                    except:
                        continue
            
            if args.env_indices is None:
                print("Error: --env-indices required when using --activations-dir without --test-csv")
                sys.exit(1)
        
        # Load activations
        if args.hf_repo_id:
            print(f"\n📥 Loading activations from HuggingFace: {args.hf_repo_id}")
            print(f"  Path: {args.activations_dir}")
            print(f"  Layer: {args.layer}")
            activations = load_activations_from_hf(
                repo_id=args.hf_repo_id,
                path_in_repo=args.activations_dir,  # Don't add extra "activations/" prefix
                layer=args.layer,
                grid_indices=args.env_indices,
            )
        else:
            print(f"\n📥 Loading activations from: {args.activations_dir}")
            activations = load_activations_from_local(
                activations_dir=args.activations_dir,
                layer=args.layer,
                grid_indices=args.env_indices,
            )
        
        print(f"Loaded {len(activations)} activations")
        
        # Process each activation
        for env_idx in sorted(activations.keys()):
            if env_idx not in activations:
                print(f"Warning: Activation not found for env_idx {env_idx}, skipping...")
                continue
                
            activation = activations[env_idx]
            if activation.ndim > 1:
                activation = activation[-1]  # Take last token if multi-token
            activation = activation.unsqueeze(0).to(args.device).to(torch.float32)
            
            print(f"\n{'='*60}")
            print(f"Environment {env_idx}:")
            print(f"{'='*60}")
            
            # Load grid information from CSV
            grid_info = None
            if df_grids is not None:
                try:
                    # Try to get row with trajectory_step == 0 first, then any row
                    row = df_grids[(df_grids["env_idx"] == env_idx) & (df_grids.get("trajectory_step", 0) == 0)]
                    if len(row) == 0:
                        row = df_grids[df_grids["env_idx"] == env_idx]
                    if len(row) > 0:
                        grid_info = row.iloc[0]
                except Exception as e:
                    print(f"Warning: Could not load grid info for env_idx {env_idx}: {e}")
            
            # Display grid
            if grid_info is not None:
                # Print grid observation
                if "fo_observation" in grid_info and pd.notna(grid_info["fo_observation"]):
                    print("\n📊 Grid:")
                    print(grid_info["fo_observation"])
                
                # Print start and goal positions
                if "start_pos" in grid_info and pd.notna(grid_info["start_pos"]):
                    print(f"\n📍 Start position: {grid_info['start_pos']}")
                if "goal_pos" in grid_info and pd.notna(grid_info["goal_pos"]):
                    print(f"🎯 Goal position: {grid_info['goal_pos']}")
                
                # Print optimal trajectory length
                if "optimal_trajectory_length" in grid_info and pd.notna(grid_info["optimal_trajectory_length"]):
                    print(f"📏 Optimal trajectory length: {int(grid_info['optimal_trajectory_length'])}")
                
                # Print ground truth action sequence if available
                true_actions = None
                if "action_sequence" in grid_info and pd.notna(grid_info["action_sequence"]):
                    import json
                    try:
                        true_actions = json.loads(grid_info["action_sequence"])
                        # Remove EOS token if present
                        if len(true_actions) > 0 and true_actions[-1] == EOS_TOKEN_ID:
                            true_actions = true_actions[:-1]
                        
                        action_names = ["LEFT", "RIGHT", "UP", "DOWN"]  # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
                        true_seq_str = " → ".join([action_names[a] for a in true_actions])
                        print(f"\n✅ Ground truth ({len(true_actions)} actions): {true_actions}")
                        print(f"   {true_seq_str}")
                    except Exception as e:
                        print(f"Warning: Could not parse action_sequence: {e}")
            else:
                print("⚠️  No grid information available in CSV")
            
            print("-" * 60)
            
            # Generate actions
            print(f"\n🚀 Generating prediction...")
            with torch.no_grad():
                actions = model.generate(
                    activation,
                    max_length=min(args.max_length, max_seq_len - 1),
                    temperature=args.temperature,
                    min_confidence=args.min_confidence,
                    max_repetition=args.max_repetition,
                    entropy_threshold=args.entropy_threshold,
                    max_length_override=args.max_length_override,
                    disable_heuristic_stopping=args.disable_heuristic_stopping,
                )
            
            # Remove padding from predictions
            actions_list = actions[0].cpu().tolist()
            actions_list = [a for a in actions_list if a != -100]  # Remove padding
            
            action_names = ["LEFT", "RIGHT", "UP", "DOWN", "SOS", "EOS", "PAD"]  # 0-3=actions, 4=SOS, 5=EOS, 6=PAD
            # Filter to only valid actions (0-3) for display
            valid_actions = [a for a in actions_list if 0 <= a <= 3]
            pred_seq_str = " → ".join([action_names[a] for a in valid_actions]) if valid_actions else "(no valid actions)"
            
            # Filter to only valid actions for display
            valid_actions = [a for a in actions_list if 0 <= a <= 3]
            print(f"🤖 Prediction ({len(valid_actions)} actions): {valid_actions}")
            print(f"   {pred_seq_str}")
            
            # Compare with ground truth if available
            if true_actions is not None:
                # Compare sequences using valid_actions (filtered to 0-3 only)
                min_len = min(len(valid_actions), len(true_actions))
                matches = sum(1 for i in range(min_len) if valid_actions[i] == true_actions[i])
                accuracy = matches / len(true_actions) if len(true_actions) > 0 else 0.0
                exact_match = (len(valid_actions) == len(true_actions) and 
                              all(valid_actions[i] == true_actions[i] for i in range(len(true_actions))))
                
                print(f"\n📊 Comparison:")
                print(f"   Token accuracy: {matches}/{len(true_actions)} ({accuracy*100:.1f}%)")
                print(f"   Exact match: {'✅ YES' if exact_match else '❌ NO'}")
                if not exact_match and len(valid_actions) != len(true_actions):
                    print(f"   Length: predicted={len(valid_actions)}, ground_truth={len(true_actions)}")
    else:
        print("Error: Must provide one of:")
        print("  1. --activation-path (single activation file)")
        print("  2. --activations-dir and --layer (local directory)")
        print("  3. --test-csv and --layer (test from CSV, optionally with --hf-repo-id for HuggingFace)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

