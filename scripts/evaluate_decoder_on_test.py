"""Evaluate trained OneShotPrefixProbe on test set from HuggingFace.

This script:
1. Loads a trained decoder probe model
2. Extracts test activations and trajectories from HuggingFace
3. Evaluates model performance by grid size (7, 9, 11, 13, 15)
4. Reports detailed metrics

Test data structure:
- Activations: project-telos/activations_test_full
  - Organized by grid size: size7/, size9/, size11/, size13/, size15/
  - Path: size{N}/{folder}/openai__gpt-oss-20b/layer_{layer}/step_0/output/*.pt
  
- Trajectories: project-telos/trajectories_test_full
  - Organized by grid size: size7/, size9/, size11/, size13/, size15/
  - Path: size{N}/{folder}.json

Usage:
    python scripts/evaluate_decoder_on_test.py \
      --model-path models/decoder_probe_layer15_full.pt \
      --layer 15 \
      --grid-sizes 7 9 11 13 15
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_interp.decoder_probe import OneShotPrefixProbe, PAD_TOKEN_ID


# Action mapping
ACTION_MAP = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}


def extract_action_sequence_from_json(trajectory_data: dict) -> list[int]:
    """Extract action sequence from trajectory JSON."""
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


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if an exception is a rate limit error from HuggingFace."""
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()
    
    rate_limit_indicators = [
        'rate limit',
        'too many requests',
        '429',
        'quota exceeded',
        'throttled',
    ]
    
    return any(indicator in error_str or indicator in error_type for indicator in rate_limit_indicators)


def retry_with_backoff(func, *args, max_retries=10, initial_delay=1.0, max_delay=60.0, max_wait_time=300.0, **kwargs):
    """Retry a function with exponential backoff on rate limit errors."""
    start_time = time.time()
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= max_wait_time:
                return None
            
            if not is_rate_limit_error(e):
                return None
            
            if attempt < max_retries - 1:
                sleep_time = min(delay, max_wait_time - elapsed_time)
                
                if sleep_time > 0:
                    print(f"  🔄 Rate limit hit. Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    delay = min(delay * 2, max_delay)
                else:
                    return None
            else:
                return None
    
    return None


def load_activations_from_local_folder(
    folder_path: str,
    layer: int,
) -> torch.Tensor | None:
    """Load activations from a local extracted folder."""
    import os
    output_folder = os.path.join(folder_path, "openai__gpt-oss-20b", f"layer_{layer}", "step_0", "output")
    
    if not os.path.exists(output_folder):
        return None
    
    try:
        pt_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.pt')])
        
        if len(pt_files) == 0:
            return None
        
        if len(pt_files) != 3:
            if len(pt_files) < 3:
                return None
            pt_files = pt_files[:3]
        
        activation_tensors = []
        for pt_file in pt_files:
            pt_path = os.path.join(output_folder, pt_file)
            act_tensor = torch.load(pt_path, weights_only=True)
            
            if act_tensor.ndim > 1:
                act_tensor = act_tensor.squeeze()
            
            activation_tensors.append(act_tensor)
        
        activations = torch.stack(activation_tensors, dim=0)
        return activations
        
    except Exception:
        return None


def load_activations_from_folder(
    fs: HfFileSystem,
    activations_repo: str,
    folder_path: str,
    layer: int,
) -> torch.Tensor | None:
    """Load activations from a specific folder with retry logic."""
    output_folder = f"{folder_path}/openai__gpt-oss-20b/layer_{layer}/step_0/output"
    
    try:
        def list_files():
            return fs.ls(output_folder, detail=False)
        
        output_files = retry_with_backoff(list_files)
        
        if output_files is None:
            return None
        
        pt_files = [f for f in output_files if f.endswith('.pt')]
        
        if len(pt_files) == 0:
            return None
        
        pt_files = sorted(pt_files)
        
        if len(pt_files) != 3:
            if len(pt_files) < 3:
                return None
            pt_files = pt_files[:3]
        
        activation_tensors = []
        for pt_file in pt_files:
            parts = pt_file.split('/')
            repo_parts = activations_repo.split('/')
            split_idx = len(['datasets'] + repo_parts)
            filename = '/'.join(parts[split_idx:])
            
            def download_file():
                return hf_hub_download(
                    repo_id=activations_repo,
                    filename=filename,
                    repo_type='dataset',
                )
            
            act_path = retry_with_backoff(download_file)
            
            if act_path is None:
                return None
            
            act_tensor = torch.load(act_path, weights_only=True)
            
            if act_tensor.ndim > 1:
                act_tensor = act_tensor.squeeze()
            
            activation_tensors.append(act_tensor)
        
        activations = torch.stack(activation_tensors, dim=0)
        return activations
        
    except Exception:
        return None


def load_test_data_for_grid_size(
    activations_repo: str,
    trajectories_repo: str,
    grid_size: int,
    layer: int,
    max_seq_len: int = 50,
    local_activations_dir: str = None,
) -> tuple[list[tuple[torch.Tensor, list[int], str]], dict]:
    """Load test data for a specific grid size.
    
    Since folder names and JSON names don't match, we:
    1. Load all activation folders (sorted)
    2. Load all trajectory JSON files (sorted)
    3. Match them by index (folder[i] pairs with json[i])
    
    Returns:
        Tuple of:
        - List of (activations, actions, folder_name) tuples
        - Statistics dictionary
    """
    fs = HfFileSystem()
    
    # List all activation folders
    if local_activations_dir:
        # Use local extracted directory
        import os
        size_path = os.path.join(local_activations_dir, f"size{grid_size}")
        if not os.path.exists(size_path):
            print(f"  ❌ Local directory not found: {size_path}")
            return [], {}
        
        try:
            folders = []
            for item in sorted(os.listdir(size_path)):
                item_path = os.path.join(size_path, item)
                if os.path.isdir(item_path):
                    folders.append(item_path)
        except Exception as e:
            print(f"  ❌ Error listing local directory {size_path}: {e}")
            return [], {}
    else:
        # Use HuggingFace repo
        size_path = f"datasets/{activations_repo}/size{grid_size}"
        
        try:
            def list_folders():
                all_items = fs.ls(size_path, detail=True)
                return sorted([item['name'] for item in all_items if item['type'] == 'directory'])
            
            folders = retry_with_backoff(list_folders)
            
            if folders is None:
                print(f"  ❌ Could not list folders for size{grid_size}")
                return [], {}
        
        except Exception as e:
            print(f"  ❌ Error accessing size{grid_size}: {e}")
            return [], {}
    
    # List all trajectory JSON files (sorted)
    traj_path = f"datasets/{trajectories_repo}/size{grid_size}"
    
    try:
        def list_jsons():
            all_items = fs.ls(traj_path, detail=False)
            return sorted([f for f in all_items if f.endswith('.json')])
        
        json_files = retry_with_backoff(list_jsons)
        
        if json_files is None:
            print(f"  ❌ Could not list JSON files for size{grid_size}")
            return [], {}
    
    except Exception as e:
        print(f"  ❌ Error accessing trajectory JSONs for size{grid_size}: {e}")
        return [], {}
    
    # Match by index: folders[i] <-> json_files[i]
    num_pairs = min(len(folders), len(json_files))
    
    if num_pairs == 0:
        print(f"  ⚠️  No matching pairs found (folders={len(folders)}, jsons={len(json_files)})")
        return [], {}
    
    dataset = []
    stats = {
        'total_pairs': num_pairs,
        'loaded': 0,
        'skipped_no_activations': 0,
        'skipped_no_actions': 0,
        'skipped_seq_too_long': 0,
        'sequence_lengths': [],
    }
    
    for idx in tqdm(range(num_pairs), desc=f"  Loading size{grid_size}", leave=False):
        folder_path = folders[idx]
        json_file = json_files[idx]
        
        try:
            # Load activations
            if local_activations_dir:
                # Load from local directory
                activations = load_activations_from_local_folder(folder_path, layer)
            else:
                # Load from HuggingFace
                activations = load_activations_from_folder(fs, activations_repo, folder_path, layer)
            
            if activations is None:
                stats['skipped_no_activations'] += 1
                continue
            
            # Load action sequences from JSON
            def read_json():
                with fs.open(json_file, 'r') as f:
                    return f.read()
            
            content = retry_with_backoff(read_json)
            
            if content is None or not content or content.strip() == '':
                stats['skipped_no_actions'] += 1
                continue
            
            traj_data = json.loads(content)
            actions = extract_action_sequence_from_json(traj_data)
            
            if actions is None or len(actions) == 0:
                stats['skipped_no_actions'] += 1
                continue
            
            # Skip if sequence too long
            if len(actions) > max_seq_len:
                stats['skipped_seq_too_long'] += 1
                continue
            
            # Add to dataset (use index as identifier since names don't match)
            identifier = f"size{grid_size}_idx{idx}"
            dataset.append((activations, actions, identifier))
            stats['loaded'] += 1
            stats['sequence_lengths'].append(len(actions))
            
        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"  ⚠️  Error at index {idx}: {e}")
            continue
    
    # Compute statistics
    if stats['sequence_lengths']:
        stats['min_length'] = min(stats['sequence_lengths'])
        stats['max_length'] = max(stats['sequence_lengths'])
        stats['mean_length'] = sum(stats['sequence_lengths']) / len(stats['sequence_lengths'])
        stats['median_length'] = sorted(stats['sequence_lengths'])[len(stats['sequence_lengths']) // 2]
    
    return dataset, stats


@torch.no_grad()
def evaluate_model(
    model: OneShotPrefixProbe,
    dataset: list[tuple[torch.Tensor, list[int], str]],
    device: str,
    max_path_len: int,
) -> dict:
    """Evaluate model on a dataset.
    
    Returns metrics including token accuracy, sequence accuracy, and prefix accuracies.
    """
    model.eval()
    model.to(device)
    
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = len(dataset)
    
    # Track prefix accuracies (first N steps correct)
    prefix_correct = defaultdict(int)
    prefix_total = defaultdict(int)
    
    all_predictions = []
    all_ground_truth = []
    
    for activations, actions, folder_name in tqdm(dataset, desc="  Evaluating", leave=False):
        # Prepare input
        activations_batch = activations.unsqueeze(0).to(device)  # (1, 3, activation_dim)
        
        # Pad actions to max_path_len
        action_tensor = torch.tensor(actions, dtype=torch.long)
        true_length = len(actions)
        
        if len(action_tensor) < max_path_len:
            padding = torch.full((max_path_len - len(action_tensor),), PAD_TOKEN_ID, dtype=torch.long)
            action_tensor = torch.cat([action_tensor, padding])
        else:
            action_tensor = action_tensor[:max_path_len]
            true_length = max_path_len
        
        # Get predictions
        predictions = model.generate(activations_batch)  # (1, max_path_len)
        predictions = predictions[0].cpu()  # (max_path_len,)
        
        # Compare only valid positions
        valid_actions = action_tensor[:true_length]
        valid_predictions = predictions[:true_length]
        
        # Token-level accuracy
        correct_mask = (valid_predictions == valid_actions)
        correct_tokens += correct_mask.sum().item()
        total_tokens += true_length
        
        # Sequence-level accuracy
        if correct_mask.all():
            correct_sequences += 1
        
        # Prefix accuracies
        for n in range(1, true_length + 1):
            prefix_total[n] += 1
            if correct_mask[:n].all():
                prefix_correct[n] += 1
        
        all_predictions.append(valid_predictions.tolist())
        all_ground_truth.append(valid_actions.tolist())
    
    # Compute metrics
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    prefix_accuracies = {}
    for n in sorted(prefix_total.keys()):
        if prefix_total[n] > 0:
            prefix_accuracies[n] = prefix_correct[n] / prefix_total[n]
    
    return {
        'token_accuracy': token_accuracy,
        'sequence_accuracy': sequence_accuracy,
        'prefix_accuracies': prefix_accuracies,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens,
        'total_sequences': total_sequences,
        'correct_sequences': correct_sequences,
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
    }


def extract_tar_if_needed(tar_path: str) -> str:
    """Extract tar file if not already extracted.
    
    The tar file contains a top-level 'activations_test_full/' directory.
    
    Returns:
        Path to extracted directory (the one containing size7/, size9/, etc.)
    """
    import tarfile
    import os
    
    # Extract to the same directory as the tar file
    extract_base_dir = os.path.dirname(tar_path)
    
    # Check if already extracted (look for activations_test_full directory)
    expected_dir = os.path.join(extract_base_dir, "activations_test_full")
    
    if os.path.exists(expected_dir) and os.path.isdir(expected_dir):
        # Verify it has the expected structure (size7/, size9/, etc.)
        if os.path.exists(os.path.join(expected_dir, "size7")):
            print(f"  ✓ Using existing extracted directory: {expected_dir}")
            return expected_dir
    
    print(f"  📦 Extracting {os.path.basename(tar_path)}...")
    print(f"     This may take a few minutes...")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_base_dir)
        
        # The tar file should have created activations_test_full/ directory
        if os.path.exists(expected_dir):
            print(f"  ✓ Extraction complete: {expected_dir}")
            return expected_dir
        else:
            # Try to find the actual extracted directory
            for item in os.listdir(extract_base_dir):
                item_path = os.path.join(extract_base_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "size7")):
                    print(f"  ✓ Extraction complete: {item_path}")
                    return item_path
            
            print(f"  ❌ Could not find extracted directory with size7/, size9/, etc.")
            return None
            
    except Exception as e:
        print(f"  ❌ Error extracting tar file: {e}")
        return None


def download_tar_if_needed(repo_id: str, tar_filename: str) -> str:
    """Download tar file from HuggingFace if not in cache.
    
    Returns:
        Path to downloaded tar file
    """
    print(f"  📥 Downloading {tar_filename} from {repo_id}...")
    print(f"     This may take several minutes (566 MB)...")
    
    try:
        tar_path = hf_hub_download(
            repo_id=repo_id,
            filename=tar_filename,
            repo_type='dataset',
        )
        print(f"  ✓ Download complete: {tar_path}")
        return tar_path
    except Exception as e:
        print(f"  ❌ Error downloading tar file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate OneShotPrefixProbe on test set")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.pt file)",
    )
    parser.add_argument(
        "--activations-repo",
        type=str,
        default="project-telos/activations_test_full",
        help="HuggingFace repository with test activations",
    )
    parser.add_argument(
        "--trajectories-repo",
        type=str,
        default="project-telos/trajectories_test_full",
        help="HuggingFace repository with test trajectories",
    )
    parser.add_argument(
        "--local-activations-dir",
        type=str,
        default=None,
        help="Path to local extracted activations directory (if already extracted)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer number (must match training)",
    )
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs='+',
        default=[7, 9, 11, 13, 15],
        help="Grid sizes to evaluate on (e.g., 7 9 11 13 15)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=50,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--activation-dim",
        type=int,
        default=2880,
        help="Activation dimension (must match training)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1024,
        help="Hidden dimension (must match training)",
    )
    parser.add_argument(
        "--max-path-len",
        type=int,
        default=20,
        help="Max path length (must match training)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers (must match training)",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Number of attention heads (must match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save detailed results (JSON)",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("OneShotPrefixProbe Test Evaluation")
    print(f"{'='*80}")
    print(f"  Model: {args.model_path}")
    print(f"  Activations repo: {args.activations_repo}")
    print(f"  Trajectories repo: {args.trajectories_repo}")
    print(f"  Layer: {args.layer}")
    print(f"  Grid sizes: {args.grid_sizes}")
    print(f"  Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    model = OneShotPrefixProbe(
        activation_dim=args.activation_dim,
        hidden_dim=args.hidden_dim,
        num_actions=5,
        num_memory_tokens=3,
        max_path_len=args.max_path_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    print(f"✓ Model loaded successfully\n")
    
    # Handle activations extraction
    local_activations_dir = args.local_activations_dir
    
    if local_activations_dir is None:
        # Need to download and extract tar file
        print(f"\n{'='*80}")
        print("Preparing Activations")
        print(f"{'='*80}")
        
        # Download tar file
        tar_path = download_tar_if_needed(
            repo_id=args.activations_repo,
            tar_filename="activations_test_full.tar"
        )
        
        if tar_path is None:
            print("❌ Failed to download activations. Exiting.")
            return
        
        # Extract tar file
        local_activations_dir = extract_tar_if_needed(tar_path)
        
        if local_activations_dir is None:
            print("❌ Failed to extract activations. Exiting.")
            return
        
        print(f"{'='*80}\n")
    else:
        print(f"Using local activations directory: {local_activations_dir}\n")
    
    # Evaluate on each grid size
    results_by_size = {}
    all_results = {
        'grid_sizes': {},
        'overall': {},
    }
    
    for grid_size in args.grid_sizes:
        print(f"\n{'='*80}")
        print(f"Evaluating on Grid Size {grid_size}")
        print(f"{'='*80}")
        
        # Load test data
        print(f"Loading test data for size {grid_size}...")
        dataset, stats = load_test_data_for_grid_size(
            activations_repo=args.activations_repo,
            trajectories_repo=args.trajectories_repo,
            grid_size=grid_size,
            layer=args.layer,
            max_seq_len=args.max_seq_len,
            local_activations_dir=local_activations_dir,
        )
        
        if len(dataset) == 0:
            print(f"  ⚠️  No test data loaded for size {grid_size}")
            continue
        
        print(f"  Loaded: {stats['loaded']}/{stats.get('total_pairs', stats.get('total_folders', 0))} samples")
        if stats['sequence_lengths']:
            print(f"  Sequence lengths: min={stats['min_length']}, max={stats['max_length']}, "
                  f"mean={stats['mean_length']:.1f}, median={stats['median_length']}")
        if stats.get('skipped_no_activations', 0) > 0 or stats.get('skipped_no_actions', 0) > 0:
            print(f"  Skipped: activations={stats.get('skipped_no_activations', 0)}, "
                  f"actions={stats.get('skipped_no_actions', 0)}, "
                  f"too_long={stats.get('skipped_seq_too_long', 0)}")
        
        # Evaluate
        print(f"Evaluating...")
        metrics = evaluate_model(model, dataset, args.device, args.max_path_len)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS FOR GRID SIZE {grid_size}")
        print(f"{'='*80}")
        print(f"  Token Accuracy:    {metrics['token_accuracy']:.4f} ({metrics['correct_tokens']}/{metrics['total_tokens']})")
        print(f"  Sequence Accuracy: {metrics['sequence_accuracy']:.4f} ({metrics['correct_sequences']}/{metrics['total_sequences']})")
        
        print(f"\n  Prefix Accuracies (first N steps correct):")
        for n in sorted(metrics['prefix_accuracies'].keys())[:15]:
            acc = metrics['prefix_accuracies'][n]
            print(f"    N={n:2d}: {acc:.1%}")
        
        results_by_size[grid_size] = {
            'stats': stats,
            'metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'ground_truth']},
        }
        
        all_results['grid_sizes'][grid_size] = results_by_size[grid_size]
    
    # Compute overall results
    print(f"\n{'='*80}")
    print("OVERALL TEST SET RESULTS")
    print(f"{'='*80}")
    
    total_tokens = sum(r['metrics']['total_tokens'] for r in results_by_size.values())
    total_correct_tokens = sum(r['metrics']['correct_tokens'] for r in results_by_size.values())
    total_sequences = sum(r['metrics']['total_sequences'] for r in results_by_size.values())
    total_correct_sequences = sum(r['metrics']['correct_sequences'] for r in results_by_size.values())
    
    overall_token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    overall_seq_acc = total_correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    print(f"  Token Accuracy:    {overall_token_acc:.4f} ({total_correct_tokens}/{total_tokens})")
    print(f"  Sequence Accuracy: {overall_seq_acc:.4f} ({total_correct_sequences}/{total_sequences})")
    print(f"\n  Breakdown by grid size:")
    for grid_size in sorted(results_by_size.keys()):
        metrics = results_by_size[grid_size]['metrics']
        print(f"    Size {grid_size:2d}: Token={metrics['token_accuracy']:.4f}, Seq={metrics['sequence_accuracy']:.4f}")
    
    all_results['overall'] = {
        'token_accuracy': overall_token_acc,
        'sequence_accuracy': overall_seq_acc,
        'total_tokens': total_tokens,
        'total_correct_tokens': total_correct_tokens,
        'total_sequences': total_sequences,
        'total_correct_sequences': total_correct_sequences,
    }
    
    # Save results
    if args.output_file:
        output_path = args.output_file
    else:
        model_name = Path(args.model_path).stem
        output_path = f"results/{model_name}_test_results.json"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
