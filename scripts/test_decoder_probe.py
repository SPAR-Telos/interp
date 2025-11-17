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

from telos_interp.decoder_probe import ActionDecoderProbe, load_activations, load_activations_from_local


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
        help="Directory containing activation files (for testing multiple grids)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer number (required if using activations-dir)",
    )
    parser.add_argument(
        "--env-indices",
        type=int,
        nargs="+",
        default=None,
        help="List of env_idx values to test (e.g., 0 1 2 3)",
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
        default=0.5,
        help="Minimum confidence (probability) to continue generation (default: 0.5)",
    )
    parser.add_argument(
        "--max-repetition",
        type=int,
        default=3,
        help="Maximum consecutive identical actions before stopping (default: 3)",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.9,
        help="Maximum entropy (uncertainty) before stopping (default: 0.9)",
    )
    parser.add_argument(
        "--max-length-override",
        type=int,
        default=10,
        help="Hard limit on sequence length based on training data stats (default: 10, ~90th percentile)",
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
        vocab_size = checkpoint.get("vocab_size", 4)
        n_layer = checkpoint.get("n_layer", 4)
        n_head = checkpoint.get("n_head", 4)
        n_embd = checkpoint.get("n_embd", 256)
        max_seq_len = checkpoint.get("max_seq_len", 100)
    elif isinstance(checkpoint, dict) and any(k.startswith("activation_projection") for k in checkpoint.keys()):
        # It's a state_dict directly
        state_dict = checkpoint
        # Try to infer from state_dict
        # activation_projection.weight shape is (n_embd, activation_dim)
        if "activation_projection.weight" in state_dict:
            n_embd, activation_dim = state_dict["activation_projection.weight"].shape
        else:
            activation_dim = 2880  # Default for gpt-oss-20b
            n_embd = 256
        
        # action_embeddings.weight shape is (vocab_size, n_embd)
        if "action_embeddings.weight" in state_dict:
            vocab_size, _ = state_dict["action_embeddings.weight"].shape
        else:
            vocab_size = 4
        
        # Try to infer other params
        n_layer = 4
        n_head = 4
        max_seq_len = 100
    else:
        raise ValueError("Could not parse model checkpoint. Expected state_dict or dict with 'model_state_dict'")

    print(f"Model parameters:")
    print(f"  Activation dim: {activation_dim}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Layers: {n_layer}")
    print(f"  Heads: {n_head}")
    print(f"  Embedding dim: {n_embd}")
    print(f"  Max seq len: {max_seq_len}")

    # Create model
    model = ActionDecoderProbe(
        activation_dim=activation_dim,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
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
        print(f"Early stopping: min_confidence={args.min_confidence}, max_repetition={args.max_repetition}, entropy_threshold={args.entropy_threshold}")
        with torch.no_grad():
            actions = model.generate(
                activation,
                max_length=min(args.max_length, max_seq_len - 1),
                temperature=args.temperature,
                min_confidence=args.min_confidence,
                max_repetition=args.max_repetition,
                entropy_threshold=args.entropy_threshold,
                max_length_override=args.max_length_override,
            )
        
        actions_list = actions[0].cpu().tolist()
        print(f"\n✅ Generated {len(actions_list)} actions:")
        print(f"Actions: {actions_list}")
        print(f"\nAction sequence (as directions):")
        action_names = ["LEFT", "RIGHT", "UP", "DOWN"]  # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
        action_str = " → ".join([action_names[a] for a in actions_list])
        print(f"  {action_str}")
        
    elif args.activations_dir and args.layer is not None:
        # Multiple activations from directory
        if args.env_indices is None:
            print("Error: --env-indices required when using --activations-dir")
            sys.exit(1)
        
        print(f"\n📥 Loading activations from: {args.activations_dir}")
        activations = load_activations_from_local(
            activations_dir=args.activations_dir,
            layer=args.layer,
            grid_indices=args.env_indices,
        )
        
        print(f"Loaded {len(activations)} activations")
        
        # Load CSV for grid information
        csv_paths = [
            "outputs/7by7testgrids.csv",
            "outputs/7by7traingrids.csv",
            "outputs/7by7valgrids.csv",
        ]
        df_grids = None
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                try:
                    df_grids = pd.read_csv(csv_path)
                    break
                except:
                    continue
        
        # Process each activation
        for env_idx in sorted(activations.keys()):
            activation = activations[env_idx]
            if activation.ndim > 1:
                activation = activation[-1]  # Take last token if multi-token
            activation = activation.unsqueeze(0).to(args.device).to(torch.float32)
            
            print(f"\n{'='*60}")
            print(f"Environment {env_idx}:")
            print(f"{'='*60}")
            
            # Try to load grid information from CSV
            if df_grids is not None:
                try:
                    row = df_grids[(df_grids["env_idx"] == env_idx) & (df_grids["trajectory_step"] == 0)]
                    if len(row) > 0:
                        grid_info = row.iloc[0]
                        
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
                        
                        print("-" * 60)
                except:
                    pass
            
            # Generate actions
            with torch.no_grad():
                actions = model.generate(
                    activation,
                    max_length=min(args.max_length, max_seq_len - 1),
                    temperature=args.temperature,
                    min_confidence=args.min_confidence,
                    max_repetition=args.max_repetition,
                    entropy_threshold=args.entropy_threshold,
                    max_length_override=args.max_length_override,
                )
            
            actions_list = actions[0].cpu().tolist()
            print(f"Generated {len(actions_list)} actions: {actions_list}")
            action_names = ["LEFT", "RIGHT", "UP", "DOWN"]  # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
            action_str = " → ".join([action_names[a] for a in actions_list])
            print(f"Sequence: {action_str}")
    else:
        print("Error: Must provide either --activation-path or (--activations-dir and --layer)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

