"""Evaluation script for OneShotPrefixProbe using HuggingFace test datasets.

This script loads a trained decoder probe and evaluates it on the test set.
"""

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, HfFileSystem
from tqdm import tqdm

# Add parent directory to path to import telos_interp
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_interp.decoder_probe import OneShotPrefixProbe, PAD_TOKEN_ID, EOS_TOKEN_ID


# Action mapping
ACTION_MAP = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}


def download_and_extract_tar(repo_id: str, tar_filename: str, extract_dir: Path) -> Path:
    """Download and extract a tar file from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID
        tar_filename: Name of the tar file (e.g., 'trajectories_test_full.tar')
        extract_dir: Directory to extract to
        
    Returns:
        Path to extracted directory
    """
    # Download tar file
    print(f"  Downloading {tar_filename} from {repo_id}...")
    tar_path = hf_hub_download(
        repo_id=repo_id,
        filename=tar_filename,
        repo_type='dataset',
    )
    
    # Extract tar file
    extract_path = extract_dir / tar_filename.replace('.tar', '')
    if not extract_path.exists():
        print(f"  Extracting {tar_filename}...")
        extract_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_path)
        print(f"  ✓ Extracted to {extract_path}")
    else:
        print(f"  ✓ Already extracted at {extract_path}")
    
    # The tar file extracts to a nested directory with the same name
    # Check if there's a nested directory
    nested_path = extract_path / tar_filename.replace('.tar', '')
    if nested_path.exists():
        return nested_path
    
    return extract_path


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


def load_test_data(
    activations_repo: str,
    trajectories_repo: str,
    layer: int,
    max_trajectories: int | None = None,
    max_seq_len: int = 20,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict]:
    """Load test activations and action sequences from tar archives.
    
    Args:
        activations_repo: HF repo with activations tar
        trajectories_repo: HF repo with trajectory JSONs tar
        layer: Layer number to load activations from
        max_trajectories: Maximum number of trajectories to load (for testing)
        max_seq_len: Maximum sequence length for padding
    
    Returns:
        Tuple of (dataset, statistics)
    """
    print(f"\n{'='*80}")
    print(f"Loading TEST data")
    print(f"  Activations: {activations_repo}")
    print(f"  Trajectories: {trajectories_repo}")
    print(f"  Layer: {layer}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"{'='*80}\n")
    
    # Extract tar archives to temp directory
    print("Downloading and extracting tar archives...")
    extract_dir = Path(tempfile.gettempdir()) / "hf_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract trajectory tar
    traj_tar_name = Path(trajectories_repo).name + '.tar'
    traj_extracted = download_and_extract_tar(trajectories_repo, traj_tar_name, extract_dir)
    
    # Extract activations tar
    act_tar_name = Path(activations_repo).name + '.tar'
    act_extracted = download_and_extract_tar(activations_repo, act_tar_name, extract_dir)
    
    # List all JSON files from all size folders
    json_files = []
    for size_dir in traj_extracted.glob('size*'):
        json_files.extend(list(size_dir.glob('*.json')))
    json_files = [str(f) for f in json_files]
    print(f"Found {len(json_files)} trajectory files in extracted archives")
    
    if max_trajectories is not None:
        json_files = json_files[:max_trajectories]
        print(f"Limited to {len(json_files)} trajectories")
    
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
    
    for json_file_path in tqdm(json_files, desc="Loading test data"):
        traj_name = Path(json_file_path).stem
        
        try:
            # 1. Load action sequence from JSON
            with open(json_file_path, 'r') as f:
                content = f.read()
            
            if not content or content.strip() == '':
                stats['skipped_parse_error'] += 1
                continue
            
            traj_data = json.loads(content)
            actions = extract_action_sequence_from_json(traj_data)
            
            if len(actions) == 0:
                stats['skipped_parse_error'] += 1
                continue
            
            # Skip if too long
            if len(actions) >= max_seq_len:
                stats['skipped_seq_too_long'] += 1
                continue
            
            # 2. Load activations
            json_path = Path(json_file_path)
            size_folder = json_path.parent.name
            
            act_traj_dir = act_extracted / size_folder / traj_name / "openai__gpt-oss-20b" / f"layer_{layer}" / "step_0" / "output"
            
            if not act_traj_dir.exists():
                # Try with suffix
                parent_dir = act_extracted / size_folder
                matching_dirs = list(parent_dir.glob(f"{traj_name}*"))
                if not matching_dirs:
                    stats['skipped_no_activations'] += 1
                    continue
                matching_dirs = sorted(matching_dirs, key=lambda x: len(x.name))
                act_traj_dir = matching_dirs[0] / "openai__gpt-oss-20b" / f"layer_{layer}" / "step_0" / "output"
                
                if not act_traj_dir.exists():
                    stats['skipped_no_activations'] += 1
                    continue
            
            # Load all .pt files
            pt_files = sorted(list(act_traj_dir.glob('*.pt')), key=lambda x: int(x.stem))
            
            if len(pt_files) == 0:
                stats['skipped_no_activations'] += 1
                continue
            
            activation_tensors = []
            for pt_file in pt_files:
                act_tensor = torch.load(pt_file, weights_only=True)
                if act_tensor.ndim > 1:
                    act_tensor = act_tensor.squeeze()
                activation_tensors.append(act_tensor)
            
            # Stack activations
            activations = torch.stack(activation_tensors, dim=0)
            
            # Validate we have 3 tokens
            if activations.shape[0] != 3:
                stats['skipped_no_activations'] += 1
                continue
            
            # 3. Add EOS token and pad
            action_tensor = torch.tensor(actions, dtype=torch.long)
            eos_tensor = torch.tensor([EOS_TOKEN_ID], dtype=torch.long)
            action_tensor = torch.cat([action_tensor, eos_tensor])
            
            if len(action_tensor) < max_seq_len:
                padding = torch.full((max_seq_len - len(action_tensor),), PAD_TOKEN_ID, dtype=torch.long)
                action_tensor = torch.cat([action_tensor, padding])
            
            dataset.append((activations, action_tensor))
            stats['loaded'] += 1
            stats['sequence_lengths'].append(len(actions))
            
        except Exception as e:
            print(f"\nWarning: Error processing {traj_name}: {e}")
            stats['skipped_parse_error'] += 1
            continue
    
    # Compute statistics
    if stats['sequence_lengths']:
        stats['min_length'] = min(stats['sequence_lengths'])
        stats['max_length'] = max(stats['sequence_lengths'])
        stats['mean_length'] = sum(stats['sequence_lengths']) / len(stats['sequence_lengths'])
        stats['median_length'] = sorted(stats['sequence_lengths'])[len(stats['sequence_lengths']) // 2]
    
    print(f"\n{'='*80}")
    print(f"Loading complete for TEST")
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
    parser = argparse.ArgumentParser(description="Evaluate OneShotPrefixProbe on test set")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number used for training (must match model)",
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
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length (if not specified, will be inferred from model)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of test trajectories to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to evaluate on (cuda/cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("OneShotPrefixProbe Evaluation Configuration")
    print(f"{'='*80}")
    print(f"  Model path: {args.model_path}")
    print(f"  Layer: {args.layer}")
    print(f"  Test activations: {args.test_activations_repo}")
    print(f"  Test trajectories: {args.test_trajectories_repo}")
    print(f"  Max sequence length: {args.max_seq_len if args.max_seq_len else 'Auto (from model)'}")
    print(f"  Max trajectories: {args.max_trajectories or 'All'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    state_dict = torch.load(args.model_path, map_location=args.device, weights_only=True)
    
    # Infer model architecture from state_dict
    # feature_map.0.weight has shape (hidden_dim, activation_dim)
    hidden_dim = state_dict['feature_map.0.weight'].shape[0]
    activation_dim = state_dict['feature_map.0.weight'].shape[1]
    
    # Count decoder layers by looking for layer indices
    n_layers = 0
    for key in state_dict.keys():
        if key.startswith('decoder.layers.'):
            layer_idx = int(key.split('.')[2])
            n_layers = max(n_layers, layer_idx + 1)
    
    # Infer n_heads from attention weights
    # self_attn.in_proj_weight has shape (3*hidden_dim, hidden_dim)
    # n_heads = hidden_dim / head_dim, typically 8 for hidden_dim=1024
    n_heads = 8  # Standard default
    
    # path_queries has shape (max_path_len, num_memory_tokens, hidden_dim)
    max_path_len = state_dict['path_queries'].shape[0]
    
    print(f"✓ Inferred model architecture:")
    print(f"  Activation dim: {activation_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Max path len: {max_path_len}")
    print(f"  Decoder layers: {n_layers}")
    print(f"  Attention heads: {n_heads}")
    
    # Initialize model with inferred architecture
    model = OneShotPrefixProbe(
        activation_dim=activation_dim,
        hidden_dim=hidden_dim,
        num_actions=5,
        num_memory_tokens=3,
        max_path_len=max_path_len,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.1,
        position_loss_decay=0.95,
    )
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Use inferred max_seq_len if not specified
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else max_path_len
    print(f"Using max sequence length: {max_seq_len}\n")
    
    # Load test data
    test_dataset, test_stats = load_test_data(
        activations_repo=args.test_activations_repo,
        trajectories_repo=args.test_trajectories_repo,
        layer=args.layer,
        max_trajectories=args.max_trajectories,
        max_seq_len=max_seq_len,
    )
    
    if len(test_dataset) == 0:
        print("ERROR: No test data loaded!")
        return
    
    # Evaluate
    print(f"\n{'='*80}")
    print("Evaluating on Test Set")
    print(f"{'='*80}\n")
    
    test_loss = 0.0
    test_correct_tokens = 0
    test_correct_sequences = 0
    test_total_tokens = 0
    test_total_sequences = 0
    
    # Prefix accuracy tracking (first N steps correct)
    prefix_correct = {}
    prefix_total = {}
    
    # Sequence length tracking
    predicted_lengths = []
    ground_truth_lengths = []
    length_exact_matches = 0
    length_differences = []
    
    # Per-position accuracy
    position_correct = [0] * max_seq_len
    position_total = [0] * max_seq_len
    
    # Action distribution
    pred_action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    true_action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), args.batch_size), desc="Evaluating"):
            batch = test_dataset[i:i + args.batch_size]
            activations = torch.stack([item[0] for item in batch]).to(args.device)
            targets = torch.stack([item[1] for item in batch]).to(args.device)
            
            # Forward pass - model returns (loss, logits) when labels are provided
            loss, logits = model(activations, targets)
            test_loss += loss.item()
            
            # Compute accuracy
            mask = targets != PAD_TOKEN_ID
            predictions = logits.argmax(dim=-1)
            test_correct_tokens += (predictions[mask] == targets[mask]).sum().item()
            test_total_tokens += mask.sum().item()
            
            # Detailed per-sequence analysis
            for pred_seq, target_seq in zip(predictions, targets):
                seq_mask = target_seq != PAD_TOKEN_ID
                if seq_mask.sum() == 0:
                    continue
                
                # Get valid sequences (without padding)
                valid_target = target_seq[seq_mask].cpu()
                valid_pred = pred_seq[seq_mask].cpu()
                
                seq_len = len(valid_target)
                test_total_sequences += 1
                
                # Track action distributions
                for action in valid_target:
                    true_action_counts[action.item()] += 1
                for action in valid_pred:
                    pred_action_counts[action.item()] += 1
                
                # Sequence accuracy (all tokens correct)
                if torch.equal(valid_pred, valid_target):
                    test_correct_sequences += 1
                
                # Prefix accuracy (first N steps correct)
                for n in range(1, seq_len + 1):
                    if n not in prefix_total:
                        prefix_total[n] = 0
                        prefix_correct[n] = 0
                    prefix_total[n] += 1
                    if torch.equal(valid_pred[:n], valid_target[:n]):
                        prefix_correct[n] += 1
                
                # Per-position accuracy
                for pos in range(seq_len):
                    position_total[pos] += 1
                    if valid_pred[pos] == valid_target[pos]:
                        position_correct[pos] += 1
                
                # Sequence length analysis
                # Find where EOS (token 4) appears or use sequence length
                pred_eos_pos = None
                for pos, token in enumerate(valid_pred):
                    if token.item() == EOS_TOKEN_ID:
                        pred_eos_pos = pos
                        break
                
                true_eos_pos = None
                for pos, token in enumerate(valid_target):
                    if token.item() == EOS_TOKEN_ID:
                        true_eos_pos = pos
                        break
                
                # If EOS not found, use full sequence length
                pred_len = pred_eos_pos if pred_eos_pos is not None else len(valid_pred)
                true_len = true_eos_pos if true_eos_pos is not None else len(valid_target)
                
                predicted_lengths.append(pred_len)
                ground_truth_lengths.append(true_len)
                length_differences.append(pred_len - true_len)
                
                if pred_len == true_len:
                    length_exact_matches += 1
    
    test_loss /= (len(test_dataset) / args.batch_size)
    test_token_acc = test_correct_tokens / test_total_tokens if test_total_tokens > 0 else 0
    test_seq_acc = test_correct_sequences / test_total_sequences if test_total_sequences > 0 else 0
    
    # Compute prefix accuracies
    prefix_accuracies = {}
    for n in sorted(prefix_total.keys()):
        if prefix_total[n] > 0:
            prefix_accuracies[n] = prefix_correct[n] / prefix_total[n]
    
    # Compute per-position accuracies
    per_position_accuracies = []
    for pos in range(max_seq_len):
        if position_total[pos] > 0:
            per_position_accuracies.append({
                'position': pos,
                'accuracy': position_correct[pos] / position_total[pos],
                'total_samples': position_total[pos]
            })
    
    # Sequence length statistics
    length_acc = length_exact_matches / test_total_sequences if test_total_sequences > 0 else 0
    avg_pred_len = np.mean(predicted_lengths) if predicted_lengths else 0
    avg_true_len = np.mean(ground_truth_lengths) if ground_truth_lengths else 0
    avg_len_diff = np.mean(length_differences) if length_differences else 0
    median_len_diff = np.median(length_differences) if length_differences else 0
    
    results = {
        'model_path': args.model_path,
        'layer': args.layer,
        'test_samples': len(test_dataset),
        'test_loss': test_loss,
        'test_token_accuracy': test_token_acc,
        'test_sequence_accuracy': test_seq_acc,
        'test_stats': test_stats,
        'prefix_accuracies': prefix_accuracies,
        'per_position_accuracies': per_position_accuracies,
        'sequence_length_accuracy': length_acc,
        'avg_predicted_length': float(avg_pred_len),
        'avg_ground_truth_length': float(avg_true_len),
        'avg_length_difference': float(avg_len_diff),
        'median_length_difference': float(median_len_diff),
        'predicted_action_distribution': pred_action_counts,
        'ground_truth_action_distribution': true_action_counts,
    }
    
    print(f"\n{'='*80}")
    print("Test Set Results")
    print(f"{'='*80}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test token accuracy: {test_token_acc:.2%}")
    print(f"  Test sequence accuracy: {test_seq_acc:.2%}")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print("Prefix Accuracies (First N Steps Correct)")
    print(f"{'='*80}")
    for n in sorted(prefix_accuracies.keys())[:15]:  # Show first 15
        acc = prefix_accuracies[n]
        total = prefix_total[n]
        correct = prefix_correct[n]
        print(f"  First {n:2d} steps: {acc:6.2%}  ({correct:4d}/{total:4d} sequences)")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print("Sequence Length Analysis")
    print(f"{'='*80}")
    print(f"  Length exact match: {length_acc:.2%} ({length_exact_matches}/{test_total_sequences})")
    print(f"  Avg predicted length: {avg_pred_len:.2f}")
    print(f"  Avg ground truth length: {avg_true_len:.2f}")
    print(f"  Avg length difference: {avg_len_diff:.2f} (pred - true)")
    print(f"  Median length difference: {median_len_diff:.1f}")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print("Per-Position Accuracy")
    print(f"{'='*80}")
    for pos_data in per_position_accuracies[:15]:  # Show first 15 positions
        pos = pos_data['position']
        acc = pos_data['accuracy']
        total = pos_data['total_samples']
        print(f"  Position {pos:2d}: {acc:6.2%}  ({total:4d} samples)")
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print("Action Distribution")
    print(f"{'='*80}")
    action_names = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "EOS"}
    print(f"  {'Action':<10} {'Ground Truth':<15} {'Predicted':<15} {'Ratio'}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    for action_id in sorted(pred_action_counts.keys()):
        name = action_names[action_id]
        true_count = true_action_counts[action_id]
        pred_count = pred_action_counts[action_id]
        ratio = pred_count / true_count if true_count > 0 else 0
        print(f"  {name:<10} {true_count:<15d} {pred_count:<15d} {ratio:.2f}x")
    print(f"{'='*80}\n")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = args.model_path.replace('.pt', '_test_results.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
