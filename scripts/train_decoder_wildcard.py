"""Training script for OneShotPrefixProbe using wildcard folder iteration.

This script:
1. Iterates through ALL folders in train/val (no naming convention required)
2. Loads activations from layer_15/step_0/output/*.pt files
3. Loads action sequences from a separate trajectories repo OR from metadata in activation folders
4. Trains the decoder probe

Usage:
    # If you have action sequences in a separate repo:
    python scripts/train_decoder_wildcard.py --layer 15 --trajectories-repo project-telos/train_and_val_full_trajectories
    
    # If action sequences are embedded in activation metadata:
    python scripts/train_decoder_wildcard.py --layer 15 --no-trajectories-repo
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, HfFileSystem
from tqdm import tqdm

# Add parent directory to path to import telos_interp
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_interp.decoder_probe import OneShotPrefixProbe, train_decoder_probe, PAD_TOKEN_ID


# Action mapping
ACTION_MAP = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if an exception is a rate limit error from HuggingFace."""
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()
    
    # Check for common rate limit indicators
    rate_limit_indicators = [
        'rate limit',
        'too many requests',
        '429',
        'quota exceeded',
        'throttled',
    ]
    
    return any(indicator in error_str or indicator in error_type for indicator in rate_limit_indicators)


def retry_with_backoff(func, *args, max_retries=10, initial_delay=1.0, max_delay=60.0, max_wait_time=300.0, **kwargs):
    """Retry a function with exponential backoff on rate limit errors.
    
    Args:
        func: Function to call
        *args: Arguments to pass to func
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        max_wait_time: Maximum total time to spend retrying in seconds (default: 300.0 = 5 minutes)
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Result of func, or None if all retries failed
    """
    start_time = time.time()
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            elapsed_time = time.time() - start_time
            
            # Check if we've exceeded max wait time
            if elapsed_time >= max_wait_time:
                print(f"  ⏱ Max wait time ({max_wait_time}s) exceeded. Giving up.")
                return None
            
            # Only retry on rate limit errors
            if not is_rate_limit_error(e):
                # Not a rate limit error, don't retry
                return None
            
            # This is a rate limit error - retry with backoff
            if attempt < max_retries - 1:
                # Calculate sleep time, ensuring we don't exceed max_wait_time
                sleep_time = min(delay, max_wait_time - elapsed_time)
                
                if sleep_time > 0:
                    print(f"  🔄 Rate limit hit. Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    
                    # Exponential backoff with jitter
                    delay = min(delay * 2, max_delay)
                else:
                    # No time left to wait
                    return None
            else:
                print(f"  ❌ Rate limit persists after {max_retries} attempts. Skipping.")
                return None
    
    return None


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


def load_activations_from_folder(
    fs: HfFileSystem,
    activations_repo: str,
    folder_path: str,
    layer: int,
) -> torch.Tensor | None:
    """Load activations from a specific folder with retry logic for rate limits.
    
    Args:
        fs: HuggingFace filesystem
        activations_repo: Repository ID
        folder_path: Path to the trajectory folder (relative to repo)
        layer: Layer number
    
    Returns:
        Tensor of shape (3, activation_dim) or None if not found
    """
    # Build path to output folder
    output_folder = f"{folder_path}/openai__gpt-oss-20b/layer_{layer}/step_0/output"
    
    try:
        # List all files in the output folder (with retry on rate limit)
        def list_files():
            return fs.ls(output_folder, detail=False)
        
        output_files = retry_with_backoff(list_files)
        
        if output_files is None:
            # Failed after retries or non-rate-limit error
            return None
        
        pt_files = [f for f in output_files if f.endswith('.pt')]
        
        if len(pt_files) == 0:
            return None
        
        # Sort by filename to ensure consistent ordering
        # Even though names may vary, we want consistent loading order
        pt_files = sorted(pt_files)
        
        # We expect 3 activation files
        if len(pt_files) != 3:
            # Don't print warning for every folder to avoid spam
            # Adjust: take first 3 or pad if less than 3
            if len(pt_files) < 3:
                return None  # Skip if less than 3
            pt_files = pt_files[:3]  # Take first 3 if more
        
        # Load each activation file
        activation_tensors = []
        for pt_file in pt_files:
            # Extract relative path from the full path
            # pt_file is like: "datasets/repo_id/split/folder/openai__gpt-oss-20b/layer_15/step_0/output/file.pt"
            parts = pt_file.split('/')
            repo_parts = activations_repo.split('/')
            split_idx = len(['datasets'] + repo_parts)
            filename = '/'.join(parts[split_idx:])
            
            # Download the file (with retry on rate limit)
            def download_file():
                return hf_hub_download(
                    repo_id=activations_repo,
                    filename=filename,
                    repo_type='dataset',
                )
            
            act_path = retry_with_backoff(download_file)
            
            if act_path is None:
                # Failed to download after retries
                return None
            
            act_tensor = torch.load(act_path, weights_only=True)
            
            # Ensure it's 1D
            if act_tensor.ndim > 1:
                act_tensor = act_tensor.squeeze()
            
            activation_tensors.append(act_tensor)
        
        # Stack into (3, activation_dim)
        activations = torch.stack(activation_tensors, dim=0)
        return activations
        
    except Exception as e:
        # Only print non-rate-limit errors
        if not is_rate_limit_error(e):
            pass  # Silently skip to avoid spam
        return None


def load_action_sequence_from_trajectories_repo(
    fs: HfFileSystem,
    trajectories_repo: str,
    split: str,
    folder_name: str,
) -> list[int] | None:
    """Load action sequence from trajectories repo with retry logic for rate limits.
    
    The pairing works as follows:
    - Activation folder: train_and_val_activations/train/together_ai_openai_gpt-oss-20b_size11_comp0.0_0/
    - Trajectory JSON:   train_and_val_full_trajectories/train/together_ai_openai_gpt-oss-20b_size11_comp0.0_0.json
    
    Args:
        fs: HuggingFace filesystem
        trajectories_repo: Repository ID for trajectories
        split: 'train' or 'val'
        folder_name: Name of the activation folder (without path)
    
    Returns:
        List of action integers or None if not found
    """
    # Match folder name to JSON filename: folder_name -> folder_name.json
    json_path = f"datasets/{trajectories_repo}/{split}/{folder_name}.json"
    
    try:
        # Open and read JSON file (with retry on rate limit)
        def read_json():
            with fs.open(json_path, 'r') as f:
                return f.read()
        
        content = retry_with_backoff(read_json)
        
        if content is None:
            # Failed after retries or non-rate-limit error
            return None
        
        if not content or content.strip() == '':
            return None
        
        traj_data = json.loads(content)
        actions = extract_action_sequence_from_json(traj_data)
        return actions if len(actions) > 0 else None
        
    except FileNotFoundError:
        # No matching JSON file found for this activation folder (not a rate limit issue)
        return None
    except Exception as e:
        # Other errors - only print if not a rate limit error
        if not is_rate_limit_error(e):
            pass  # Silently skip to avoid spam
        return None


def load_action_sequence_from_metadata(
    fs: HfFileSystem,
    activations_repo: str,
    folder_path: str,
) -> list[int] | None:
    """Try to load action sequence from metadata in activation folder with retry logic.
    
    Args:
        fs: HuggingFace filesystem
        activations_repo: Repository ID
        folder_path: Path to the trajectory folder
    
    Returns:
        List of action integers or None if not found
    """
    # Look for metadata files in the folder (with retry on rate limit)
    try:
        def list_files():
            return fs.ls(folder_path, detail=False)
        
        files = retry_with_backoff(list_files)
        
        if files is None:
            return None
        
        # Look for JSON files that might contain action sequences
        for file in files:
            if file.endswith('.json') or file.endswith('metadata.json'):
                try:
                    def read_metadata():
                        with fs.open(file, 'r') as f:
                            return f.read()
                    
                    content = retry_with_backoff(read_metadata)
                    
                    if content is None or not content or content.strip() == '':
                        continue
                    
                    metadata = json.loads(content)
                    
                    # Try to extract actions from metadata
                    # This depends on the metadata format - adjust as needed
                    if 'actions' in metadata:
                        return metadata['actions']
                    elif 'steps' in metadata:
                        actions = extract_action_sequence_from_json(metadata)
                        if len(actions) > 0:
                            return actions
                        
                except Exception:
                    continue
        
        return None
        
    except Exception:
        return None


def load_dataset_wildcard(
    activations_repo: str,
    trajectories_repo: str | None,
    split: str,
    layer: int,
    max_seq_len: int = 20,
    max_folders: int | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """Load activations and action sequences by iterating through all folders.
    
    Matching logic:
    - Iterates through ALL folders in activations_repo/{split}/
    - For each folder, extracts the folder name (e.g., "together_ai_openai_gpt-oss-20b_size11_comp0.0_0")
    - Looks for matching JSON file in trajectories_repo/{split}/{folder_name}.json
    - Loads activations from folder/openai__gpt-oss-20b/layer_{layer}/step_0/output/*.pt
    - Pairs them together for training
    
    Args:
        activations_repo: HF repo with activations (e.g., 'project-telos/train_and_val_activations')
        trajectories_repo: HF repo with trajectory JSONs (e.g., 'project-telos/train_and_val_full_trajectories')
                          or None to look for metadata in activation folders
        split: 'train' or 'val'
        layer: Layer number to load
        max_seq_len: Maximum sequence length (longer sequences are skipped)
        max_folders: Maximum number of folders to process (for testing)
    
    Returns:
        Tuple of:
        - List of (activations, actions) tuples
        - Statistics dictionary
    """
    print(f"\n{'='*80}")
    print(f"Loading {split.upper()} data with WILDCARD iteration")
    print(f"  Activations repo: {activations_repo}")
    print(f"  Trajectories repo: {trajectories_repo or 'None (looking in activation folders)'}")
    print(f"  Matching strategy: folder_name <-> folder_name.json")
    print(f"  Layer: {layer}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"{'='*80}\n")
    
    # Initialize filesystem
    fs = HfFileSystem()
    
    # List all folders in the split directory (with retry on rate limit)
    print(f"Listing folders from {activations_repo}/{split}...")
    split_path = f"datasets/{activations_repo}/{split}"
    
    def list_folders():
        all_items = fs.ls(split_path, detail=True)
        # Filter for directories only
        return [item['name'] for item in all_items if item['type'] == 'directory']
    
    folders = retry_with_backoff(list_folders)
    
    if folders is None:
        print(f"❌ Error listing folders (rate limit or other error). Cannot proceed.")
        return [], {}
    
    print(f"Found {len(folders)} folders\n")
    
    # Limit folders if requested
    if max_folders is not None:
        folders = folders[:max_folders]
        print(f"Processing first {len(folders)} folders (max_folders={max_folders})\n")
    
    # Process each folder
    dataset = []
    stats = {
        'total_folders': len(folders),
        'loaded': 0,
        'skipped_no_activations': 0,
        'skipped_no_actions': 0,
        'skipped_seq_too_long': 0,
        'skipped_parse_error': 0,
        'skipped_rate_limit': 0,  # Failed after rate limit retries
        'sequence_lengths': [],
        'matched_pairs': 0,  # Folders with both activations and actions
    }
    
    for folder_path in tqdm(folders, desc=f"Processing {split}"):
        # Extract folder name (last component of path)
        folder_name = folder_path.split('/')[-1]
        
        try:
            # 1. Load activations from layer_{layer}/step_0/output/*.pt
            activations = load_activations_from_folder(fs, activations_repo, folder_path, layer)
            
            if activations is None:
                stats['skipped_no_activations'] += 1
                continue
            
            # 2. Load action sequences
            actions = None
            
            # Try trajectories repo first (if provided)
            if trajectories_repo is not None:
                actions = load_action_sequence_from_trajectories_repo(
                    fs, trajectories_repo, split, folder_name
                )
            
            # If not found in trajectories repo, try metadata in activation folder
            if actions is None:
                actions = load_action_sequence_from_metadata(fs, activations_repo, folder_path)
            
            # Skip if no actions found
            if actions is None or len(actions) == 0:
                stats['skipped_no_actions'] += 1
                continue
            
            # We successfully matched activations + actions!
            stats['matched_pairs'] += 1
            
            # Skip if sequence too long
            if len(actions) > max_seq_len:
                stats['skipped_seq_too_long'] += 1
                continue
            
            # 3. Pad action sequence to max_seq_len
            action_tensor = torch.tensor(actions, dtype=torch.long)
            if len(action_tensor) < max_seq_len:
                padding = torch.full((max_seq_len - len(action_tensor),), PAD_TOKEN_ID, dtype=torch.long)
                action_tensor = torch.cat([action_tensor, padding])
            
            # Add to dataset
            dataset.append((activations, action_tensor))
            stats['loaded'] += 1
            stats['sequence_lengths'].append(len(actions))
            
        except Exception as e:
            # Catch any unexpected errors
            stats['skipped_parse_error'] += 1
            if stats['skipped_parse_error'] <= 5:  # Only print first 5 errors
                print(f"\nError processing {folder_name}: {e}")
    
    # Compute sequence length statistics
    if stats['sequence_lengths']:
        stats['min_length'] = min(stats['sequence_lengths'])
        stats['max_length'] = max(stats['sequence_lengths'])
        stats['mean_length'] = sum(stats['sequence_lengths']) / len(stats['sequence_lengths'])
        stats['median_length'] = sorted(stats['sequence_lengths'])[len(stats['sequence_lengths']) // 2]
    
    print(f"\n{'='*80}")
    print(f"Loading complete for {split.upper()}")
    print(f"  Total folders scanned: {stats['total_folders']}")
    print(f"  Matched pairs (activations + actions): {stats['matched_pairs']}")
    print(f"  Successfully loaded (after filtering): {stats['loaded']}/{stats['total_folders']}")
    print(f"\n  Skipped reasons:")
    print(f"    - No activations found: {stats['skipped_no_activations']}")
    print(f"    - No action sequences found: {stats['skipped_no_actions']}")
    print(f"    - Sequence too long (>{max_seq_len}): {stats['skipped_seq_too_long']}")
    print(f"    - Rate limit failures: {stats['skipped_rate_limit']}")
    print(f"    - Parse/other errors: {stats['skipped_parse_error']}")
    if stats['sequence_lengths']:
        print(f"\n  Sequence length statistics:")
        print(f"    - Min: {stats['min_length']}, Max: {stats['max_length']}")
        print(f"    - Mean: {stats['mean_length']:.1f}, Median: {stats['median_length']}")
    print(f"{'='*80}\n")
    
    return dataset, stats


def main():
    parser = argparse.ArgumentParser(description="Train OneShotPrefixProbe with wildcard folder iteration")
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
        help="HuggingFace repository with trajectory JSONs (set to None to look in activation folders)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer number to use for activations",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=20,
        help="Maximum sequence length (sequences longer than this will be skipped)",
    )
    parser.add_argument(
        "--max-folders",
        type=int,
        default=None,
        help="Maximum number of folders to process per split (for testing)",
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
        default="models/decoder_probe_wildcard.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("OneShotPrefixProbe Training Configuration (WILDCARD MODE)")
    print(f"{'='*80}")
    print(f"  Activations repo: {args.activations_repo}")
    print(f"  Trajectories repo: {args.trajectories_repo or 'None (looking in activation folders)'}")
    print(f"  Layer: {args.layer}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  Max folders: {args.max_folders or 'All'}")
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
    train_dataset, train_stats = load_dataset_wildcard(
        activations_repo=args.activations_repo,
        trajectories_repo=args.trajectories_repo,
        split='train',
        layer=args.layer,
        max_seq_len=args.max_seq_len,
        max_folders=args.max_folders,
    )
    
    # Load validation data
    val_dataset, val_stats = load_dataset_wildcard(
        activations_repo=args.activations_repo,
        trajectories_repo=args.trajectories_repo,
        split='val',
        layer=args.layer,
        max_seq_len=args.max_seq_len,
        max_folders=args.max_folders,
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training data loaded!")
        print("\nPossible issues:")
        print("1. No activations found in the expected folder structure")
        print("2. No action sequences found (need --trajectories-repo or metadata in activation folders)")
        print("3. All sequences are longer than --max-seq-len")
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
    with open(stats_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'train_stats': train_stats,
            'val_stats': val_stats,
            'results': {
                k: v for k, v in results.items() 
                if not isinstance(v, list)  # Skip lists for JSON serialization
            },
        }, f, indent=2)
    print(f"\nTraining statistics saved to {stats_path}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"  Final train loss: {results['final_train_loss']:.4f}")
    print(f"  Final val loss: {results['final_eval_loss']:.4f}")
    print(f"  Final token accuracy: {results['final_token_accuracy']:.4f}")
    print(f"  Final sequence accuracy: {results['final_sequence_accuracy']:.4f}")
    print(f"  Model saved to: {args.save_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
