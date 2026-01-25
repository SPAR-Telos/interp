"""Training script for OneShotPrefixProbe using HuggingFace datasets.

This script loads activations and action sequences from HuggingFace,
matches them by trajectory name, and trains the decoder probe.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, HfFileSystem
from tqdm import tqdm

# Add parent directory to path to import telos_interp
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_interp.decoder_probe import OneShotPrefixProbe, train_decoder_probe, PAD_TOKEN_ID, EOS_TOKEN_ID


# Action mapping
ACTION_MAP = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}


def extract_action_sequence_from_json(trajectory_data: dict) -> list[int]:
    """Extract action sequence from trajectory JSON.
    
    Args:
        trajectory_data: Full trajectory JSON object
    
    Returns:
        List of action integers (0=LEFT, 1=RIGHT, 2=UP, 3=DOWN)
    """
    steps = trajectory_data.get("steps", [])
    actions = []
    
    for step in steps:
        agent_action = step.get("agent_action")
        if agent_action is None:
            continue
        
        action_int = ACTION_MAP.get(agent_action)
        if action_int is not None:
            actions.append(action_int)
    
    return actions


def load_trajectory_data(
    activations_repo: str,
    trajectories_repo: str,
    split: str,
    layer: int,
    max_trajectories: int | None = None,
    max_seq_len: int = 20,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """Load activations and action sequences for a split.
    
    Args:
        activations_repo: HF repo with activations (e.g., 'project-telos/train_and_val_activations')
        trajectories_repo: HF repo with trajectory JSONs (e.g., 'project-telos/train_and_val_full_trajectories')
        split: 'train' or 'val'
        layer: Layer number to load activations from
        max_trajectories: Maximum number of trajectories to load (for testing)
        max_seq_len: Maximum sequence length for padding
    
    Returns:
        Tuple of:
        - List of (activations, actions) tuples where:
            - activations: (3, activation_dim) tensor
            - actions: (max_seq_len,) tensor with padding
        - Statistics dictionary
    """
    split_display = split.upper() if split else "ROOT"
    print(f"\n{'='*80}")
    print(f"Loading {split_display} data")
    print(f"  Activations: {activations_repo}/{split if split else '(root)'}")
    print(f"  Trajectories: {trajectories_repo}/{split if split else '(root)'}")
    print(f"  Layer: {layer}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"{'='*80}\n")
    
    # List trajectory JSON files to get trajectory names
    fs = HfFileSystem()
    # Handle both split subdirectories (train/val) and root-level files (test)
    if split:
        traj_path = f"datasets/{trajectories_repo}/{split}"
    else:
        traj_path = f"datasets/{trajectories_repo}"
    print(f"Listing trajectory files from {traj_path}...")
    traj_files = fs.ls(traj_path, detail=False)
    json_files = [f for f in traj_files if f.endswith('.json')]
    
    if max_trajectories is not None:
        json_files = json_files[:max_trajectories]
    
    print(f"Found {len(json_files)} trajectory files")
    
    # Process each trajectory
    dataset = []
    stats = {
        'total_files': len(json_files),
        'loaded': 0,
        'skipped_no_activations': 0,
        'skipped_parse_error': 0,
        'skipped_seq_too_long': 0,
        'sequence_lengths': [],
    }
    
    for json_file_path in tqdm(json_files, desc=f"Loading {split} data"):
        # Extract trajectory name from path
        # e.g., "datasets/project-telos/train_and_val_full_trajectories/train/together_ai_openai_gpt-oss-20b_size11_comp0.0_0.json"
        # -> "together_ai_openai_gpt-oss-20b_size11_comp0.0_0"
        traj_name = Path(json_file_path).stem
        
        try:
            # 1. Load action sequence from JSON
            with fs.open(json_file_path, 'r') as f:
                content = f.read()
            
            if not content or content.strip() == '':
                stats['skipped_parse_error'] += 1
                continue
            
            traj_data = json.loads(content)
            actions = extract_action_sequence_from_json(traj_data)
            
            if len(actions) == 0:
                stats['skipped_parse_error'] += 1
                continue
            
            # Skip if too long (need room for EOS token)
            # If actions has length N, after adding EOS it becomes N+1
            # So we need N+1 <= max_seq_len, which means N < max_seq_len
            if len(actions) >= max_seq_len:
                stats['skipped_seq_too_long'] += 1
                continue
            
            # 2. Load all activations from step_0/output folder
            # Path pattern: {split}/{traj_name}/openai__gpt-oss-20b/layer_{layer}/step_0/output/*.pt
            # Note: activation folder may have suffix (e.g., _from_missing1), so we try exact match first,
            # then look for folders starting with traj_name
            activation_tensors = []
            try:
                # Construct paths based on whether split is provided
                if split:
                    split_prefix = f"{split}/"
                else:
                    split_prefix = ""
                
                # Try exact match first
                output_folder = f"datasets/{activations_repo}/{split_prefix}{traj_name}/openai__gpt-oss-20b/layer_{layer}/step_0/output"
                output_files = None
                
                try:
                    output_files = fs.ls(output_folder, detail=False)
                except Exception as e:
                    # Exact match failed, try prefix match
                    # List all folders in split directory and find one starting with traj_name
                    split_path = f"datasets/{activations_repo}/{split}" if split else f"datasets/{activations_repo}"
                    try:
                        all_folders = fs.ls(split_path, detail=False)
                        matching_folders = [
                            f for f in all_folders 
                            if f.split('/')[-1].startswith(traj_name)
                        ]
                        
                        if len(matching_folders) == 0:
                            stats['skipped_no_activations'] += 1
                            continue
                        
                        # Use the first matching folder (prefer exact match if multiple)
                        # Sort to prioritize exact matches (shorter names)
                        matching_folders = sorted(matching_folders, key=lambda x: len(x))
                        matched_folder = matching_folders[0].split('/')[-1]
                        
                        output_folder = f"datasets/{activations_repo}/{split_prefix}{matched_folder}/openai__gpt-oss-20b/layer_{layer}/step_0/output"
                        output_files = fs.ls(output_folder, detail=False)
                    except Exception as e2:
                        stats['skipped_no_activations'] += 1
                        continue
                
                if output_files is None or len(output_files) == 0:
                    stats['skipped_no_activations'] += 1
                    continue
                # Filter for .pt files and sort by token position
                pt_files = [f for f in output_files if f.endswith('.pt')]
                
                if len(pt_files) == 0:
                    stats['skipped_no_activations'] += 1
                    continue
                
                pt_files = sorted(pt_files, key=lambda x: int(Path(x).stem))
                
                # Load each activation file
                for pt_file in pt_files:
                    # Extract the relative path from the repo
                    # pt_file is like: "datasets/{repo_id}/{split}/{traj}/..."
                    # We need: "{split}/{traj}/openai__gpt-oss-20b/layer_{layer}/step_0/output/{num}.pt"
                    parts = pt_file.split('/')
                    # Find where the split starts (after repo name)
                    repo_parts = activations_repo.split('/')
                    split_idx = len(['datasets'] + repo_parts)  # datasets/project-telos/train_and_val_activations = 3 parts before split
                    filename = '/'.join(parts[split_idx:])  # Everything after the repo name
                    
                    act_path = hf_hub_download(
                        repo_id=activations_repo,
                        filename=filename,
                        repo_type='dataset',
                    )
                    act_tensor = torch.load(act_path, weights_only=True)
                    
                    # Ensure it's 1D
                    if act_tensor.ndim > 1:
                        act_tensor = act_tensor.squeeze()
                    
                    activation_tensors.append(act_tensor)
                
                # Stack activations into (num_tokens, activation_dim)
                activations = torch.stack(activation_tensors, dim=0)
                
                # Validate we have the expected number of tokens
                if activations.shape[0] != 3:
                    # Skip if not exactly 3 tokens
                    stats['skipped_no_activations'] += 1
                    continue
                
            except Exception as e:
                # Activation file not found or error loading
                stats['skipped_no_activations'] += 1
                continue
            
            # 3. Add EOS token after action sequence and pad to max_seq_len
            action_tensor = torch.tensor(actions, dtype=torch.long)
            
            # Add EOS token (4) after the action sequence
            eos_tensor = torch.tensor([EOS_TOKEN_ID], dtype=torch.long)
            action_tensor = torch.cat([action_tensor, eos_tensor])
            
            # Pad to max_seq_len if needed
            if len(action_tensor) < max_seq_len:
                padding = torch.full((max_seq_len - len(action_tensor),), PAD_TOKEN_ID, dtype=torch.long)
                action_tensor = torch.cat([action_tensor, padding])
            
            # Add to dataset
            dataset.append((activations, action_tensor))
            stats['loaded'] += 1
            stats['sequence_lengths'].append(len(actions))
            
        except json.JSONDecodeError as e:
            stats['skipped_parse_error'] += 1
            continue
        except Exception as e:
            print(f"\nWarning: Error processing {traj_name}: {e}")
            stats['skipped_parse_error'] += 1
            continue
    
    # Compute sequence length statistics
    if stats['sequence_lengths']:
        stats['min_length'] = min(stats['sequence_lengths'])
        stats['max_length'] = max(stats['sequence_lengths'])
        stats['mean_length'] = sum(stats['sequence_lengths']) / len(stats['sequence_lengths'])
        stats['median_length'] = sorted(stats['sequence_lengths'])[len(stats['sequence_lengths']) // 2]
    
    print(f"\n{'='*80}")
    print(f"Loading complete for {split.upper()}")
    print(f"  Successfully loaded: {stats['loaded']}/{stats['total_files']}")
    print(f"  Skipped (no activations): {stats['skipped_no_activations']}")
    print(f"  Skipped (parse error): {stats['skipped_parse_error']}")
    print(f"  Skipped (seq too long): {stats['skipped_seq_too_long']}")
    if stats['sequence_lengths']:
        print(f"  Sequence lengths: min={stats['min_length']}, max={stats['max_length']}, "
              f"mean={stats['mean_length']:.1f}, median={stats['median_length']}")
    print(f"{'='*80}\n")
    
    return dataset, stats


def main():
    parser = argparse.ArgumentParser(description="Train OneShotPrefixProbe from HuggingFace datasets")
    parser.add_argument(
        "--activations-repo",
        type=str,
        default="project-telos/train_and_val_activations",
        help="HuggingFace repository with activations",
    )
    parser.add_argument(
        "--trajectories-repo",
        type=str,
        default="project-telos/train_and_val_full_trajectories",
        help="HuggingFace repository with trajectory JSONs",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=23,
        help="Layer number to use for activations",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=10,
        help="Maximum sequence length INCLUDING EOS token (sequences longer than max_seq_len-1 will be skipped)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to load per split (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1024,
        help="Hidden dimension for the probe",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer decoder layers",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--position-loss-decay",
        type=float,
        default=0.95,
        help="Position loss decay factor for long-tail handling",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/decoder_probe_layer23.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--test-activations-repo",
        type=str,
        default="project-telos/activations_test_full",
        help="HuggingFace repository with test activations",
    )
    parser.add_argument(
        "--test-trajectories-repo",
        type=str,
        default="project-telos/trajectories_test_full",
        help="HuggingFace repository with test trajectory JSONs",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip test evaluation after training",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("OneShotPrefixProbe Training Configuration")
    print(f"{'='*80}")
    print(f"  Activations repo: {args.activations_repo}")
    print(f"  Trajectories repo: {args.trajectories_repo}")
    print(f"  Layer: {args.layer}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  Max trajectories: {args.max_trajectories or 'All'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Transformer layers: {args.n_layers}")
    print(f"  Attention heads: {args.n_heads}")
    print(f"  Position loss decay: {args.position_loss_decay}")
    print(f"  Device: {args.device}")
    print(f"  Save path: {args.save_path}")
    print(f"{'='*80}\n")
    
    # Load training data
    train_dataset, train_stats = load_trajectory_data(
        activations_repo=args.activations_repo,
        trajectories_repo=args.trajectories_repo,
        split='train',
        layer=args.layer,
        max_trajectories=args.max_trajectories,
        max_seq_len=args.max_seq_len,
    )
    
    # Load validation data
    val_dataset, val_stats = load_trajectory_data(
        activations_repo=args.activations_repo,
        trajectories_repo=args.trajectories_repo,
        split='val',
        layer=args.layer,
        max_trajectories=args.max_trajectories,
        max_seq_len=args.max_seq_len,
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training data loaded!")
        return
    
    # Get activation dimension from first sample
    activation_dim = train_dataset[0][0].shape[1]
    print(f"Detected activation dimension: {activation_dim}")
    
    # Initialize model
    model = OneShotPrefixProbe(
        activation_dim=activation_dim,
        hidden_dim=args.hidden_dim,
        num_actions=5,  # 0-3=directions, 4=GOAL/EOS
        num_memory_tokens=3,
        max_path_len=args.max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=0.1,
        position_loss_decay=args.position_loss_decay,
    )
    
    print(f"\nModel architecture:")
    print(f"  Activation dim: {activation_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Max path len: {args.max_seq_len}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train model
    results = train_decoder_probe(
        model=model,
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        eval_split=0.0,  # We have a separate val set
        val_dataset=val_dataset,
        save_path=args.save_path,
    )
    
    # Save training statistics
    stats_path = args.save_path.replace('.pt', '_stats.json')
    os.makedirs(os.path.dirname(stats_path) if os.path.dirname(stats_path) else ".", exist_ok=True)
    
    # Evaluate on test set if not skipped
    test_stats = None
    test_results = None
    if not args.skip_test_eval:
        print(f"\n{'='*80}")
        print("Evaluating on Test Set")
        print(f"{'='*80}\n")
        
        # Load test data
        test_dataset, test_stats = load_trajectory_data(
            activations_repo=args.test_activations_repo,
            trajectories_repo=args.test_trajectories_repo,
            split='',  # Test repos don't have train/val splits, just root level
            layer=args.layer,
            max_trajectories=args.max_trajectories,
            max_seq_len=args.max_seq_len,
        )
        
        if len(test_dataset) > 0:
            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            test_correct_tokens = 0
            test_correct_sequences = 0
            test_total_tokens = 0
            test_total_sequences = 0
            
            with torch.no_grad():
                for i in range(0, len(test_dataset), args.batch_size):
                    batch = test_dataset[i:i + args.batch_size]
                    activations = torch.stack([item[0] for item in batch]).to(args.device)
                    targets = torch.stack([item[1] for item in batch]).to(args.device)
                    
                    # Forward pass
                    logits = model(activations, targets)
                    
                    # Compute loss (only on non-padded tokens)
                    mask = targets != PAD_TOKEN_ID
                    loss = torch.nn.functional.cross_entropy(
                        logits[mask],
                        targets[mask],
                        reduction='mean'
                    )
                    test_loss += loss.item()
                    
                    # Compute accuracy
                    predictions = logits.argmax(dim=-1)
                    test_correct_tokens += (predictions[mask] == targets[mask]).sum().item()
                    test_total_tokens += mask.sum().item()
                    
                    # Sequence accuracy (all non-padded tokens correct)
                    for pred_seq, target_seq in zip(predictions, targets):
                        seq_mask = target_seq != PAD_TOKEN_ID
                        if seq_mask.sum() > 0:
                            test_correct_sequences += (pred_seq[seq_mask] == target_seq[seq_mask]).all().item()
                            test_total_sequences += 1
            
            test_loss /= (len(test_dataset) / args.batch_size)
            test_token_acc = test_correct_tokens / test_total_tokens if test_total_tokens > 0 else 0
            test_seq_acc = test_correct_sequences / test_total_sequences if test_total_sequences > 0 else 0
            
            test_results = {
                'test_loss': test_loss,
                'test_token_accuracy': test_token_acc,
                'test_sequence_accuracy': test_seq_acc,
                'test_samples': len(test_dataset),
            }
            
            print(f"\n{'='*80}")
            print("Test Set Results")
            print(f"{'='*80}")
            print(f"  Test samples: {len(test_dataset)}")
            print(f"  Test loss: {test_loss:.4f}")
            print(f"  Test token accuracy: {test_token_acc:.2%}")
            print(f"  Test sequence accuracy: {test_seq_acc:.2%}")
            print(f"{'='*80}\n")
        else:
            print("WARNING: No test data loaded, skipping test evaluation.\n")
    
    # Save all statistics including test results
    with open(stats_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'train_stats': train_stats,
            'val_stats': val_stats,
            'test_stats': test_stats,
            'results': {
                k: v for k, v in results.items() 
                if not isinstance(v, list)  # Skip lists for JSON serialization
            },
            'test_results': test_results,
        }, f, indent=2)
    print(f"Training statistics saved to {stats_path}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"  Final train loss: {results['final_train_loss']:.4f}")
    print(f"  Final val loss: {results['final_eval_loss']:.4f}")
    print(f"  Final token accuracy: {results['final_token_accuracy']:.4f}")
    print(f"  Final sequence accuracy: {results['final_sequence_accuracy']:.4f}")
    if test_results:
        print(f"  Test loss: {test_results['test_loss']:.4f}")
        print(f"  Test token accuracy: {test_results['test_token_accuracy']:.2%}")
        print(f"  Test sequence accuracy: {test_results['test_sequence_accuracy']:.2%}")
    print(f"  Model saved to: {args.save_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
