"""Process action sequences from HuggingFace trajectory dataset.

This script:
1. Loads trajectory JSON files from HuggingFace dataset
2. Extracts agent_action from each trajectory step
3. Converts actions to integer codes
4. Computes statistics (min, max, distribution)
5. Saves processed sequences to a file

Requirements:
    pip install huggingface_hub torch tqdm

Usage:
    python scripts/process_action_sequences.py --split train
    python scripts/process_action_sequences.py --split val
    python scripts/process_action_sequences.py --split both
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import HfFileSystem
from tqdm import tqdm


# Action mapping: string -> integer
ACTION_MAP = {
    "LEFT": 0,
    "RIGHT": 1,
    "UP": 2,
    "DOWN": 3,
}


def extract_action_sequence(trajectory_data: dict) -> tuple[list[int], dict]:
    """Extract action sequence from trajectory JSON.
    
    Args:
        trajectory_data: Full trajectory JSON object
    
    Returns:
        Tuple of:
        - List of action integers (0=LEFT, 1=RIGHT, 2=UP, 3=DOWN)
        - Metadata dict with grid_params, etc.
    """
    steps = trajectory_data.get("steps", [])
    
    actions = []
    for step in steps:
        agent_action = step.get("agent_action")
        if agent_action is None:
            continue
        
        # Convert action string to integer
        action_int = ACTION_MAP.get(agent_action)
        if action_int is not None:
            actions.append(action_int)
        else:
            print(f"Warning: Unknown action '{agent_action}' found")
    
    # Extract metadata
    metadata = {
        'grid_params': trajectory_data.get('grid_params', {}),
        'model_params': trajectory_data.get('model_params', {}),
        'n_steps': len(actions),
    }
    
    return actions, metadata


def process_split(
    repo_id: str,
    split: str,
    output_dir: Path,
) -> dict[str, dict]:
    """Process one split (train or val) of the dataset.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        split: Dataset split ("train" or "val")
        output_dir: Directory to save processed sequences
    
    Returns:
        Dictionary with statistics and metadata
    """
    print(f"\n{'='*80}")
    print(f"Processing {split.upper()} split from {repo_id}")
    print(f"{'='*80}")
    
    # List all JSON files in the split directory
    print(f"Listing files from HuggingFace...")
    fs = HfFileSystem()
    split_path = f"datasets/{repo_id}/{split}"
    files = fs.ls(split_path, detail=False)
    json_files = [f for f in files if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files\n")
    
    # Process each trajectory
    action_sequences = {}
    metadata_by_idx = {}
    length_distribution = defaultdict(int)
    failed_trajectories = []  # Trajectories that hit max steps (likely failed)
    skipped_files = []  # Files that couldn't be parsed
    
    for idx, file_path in enumerate(tqdm(json_files, desc=f"Processing {split}")):
        try:
            # Download and load JSON file
            with fs.open(file_path, 'r') as f:
                content = f.read()
                
            # Skip empty files
            if not content or content.strip() == '':
                skipped_files.append((idx, file_path, "Empty file"))
                continue
                
            example = json.loads(content)
            
            # Extract action sequence
            actions, metadata = extract_action_sequence(example)
            
            if len(actions) == 0:
                skipped_files.append((idx, file_path, "No valid actions"))
                continue
            
            # Track if trajectory hit max steps (likely failed/stuck)
            max_steps = metadata.get('model_params', {}).get('max_steps_per_trajectory', 50)
            if len(actions) >= max_steps:
                failed_trajectories.append(idx)
            
            # Store sequence and metadata
            action_sequences[idx] = actions
            metadata_by_idx[idx] = metadata
            length_distribution[len(actions)] += 1
            
        except json.JSONDecodeError as e:
            skipped_files.append((idx, file_path, f"JSON decode error: {str(e)[:50]}"))
            continue
        except Exception as e:
            skipped_files.append((idx, file_path, f"Error: {str(e)[:50]}"))
            continue
    
    # Report skipped files
    if skipped_files:
        print(f"\n⚠️  Skipped {len(skipped_files)} files due to errors:")
        for idx, file_path, reason in skipped_files[:5]:
            filename = file_path.split('/')[-1]
            print(f"  - {filename}: {reason}")
        if len(skipped_files) > 5:
            print(f"  ... and {len(skipped_files) - 5} more")
    
    # Compute statistics
    lengths = [len(seq) for seq in action_sequences.values()]
    
    if not lengths:
        print("ERROR: No valid trajectories found!")
        return {}
    
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = sum(lengths) / len(lengths)
    
    # Find examples with min and max lengths
    min_examples = [idx for idx, seq in action_sequences.items() if len(seq) == min_length]
    max_examples = [idx for idx, seq in action_sequences.items() if len(seq) == max_length]
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"STATISTICS FOR {split.upper()} SPLIT")
    print(f"{'='*80}")
    print(f"Total trajectories: {len(action_sequences)}")
    print(f"Failed trajectories (hit max steps): {len(failed_trajectories)} ({100*len(failed_trajectories)/len(action_sequences):.1f}%)")
    print(f"Min length: {min_length} (from trajectories: {min_examples[:5]}{'...' if len(min_examples) > 5 else ''})")
    print(f"Max length: {max_length} (from trajectories: {max_examples[:5]}{'...' if len(max_examples) > 5 else ''})")
    print(f"Mean length: {mean_length:.2f}")
    print(f"Median length: {sorted(lengths)[len(lengths)//2]}")
    print(f"\nLength distribution:")
    for length in sorted(length_distribution.keys()):
        count = length_distribution[length]
        percentage = 100 * count / len(action_sequences)
        bar = '█' * int(percentage / 2)
        is_max = " (MAX - likely failed)" if length >= 45 else ""
        print(f"  Length {length:2d}: {count:4d} trajectories ({percentage:5.2f}%) {bar}{is_max}")
    
    # Show sample sequences
    print(f"\nSample successful sequences (shorter ones):")
    successful_seqs = [(idx, seq) for idx, seq in action_sequences.items() if len(seq) < 20]
    for i, (idx, seq) in enumerate(list(successful_seqs)[:3]):
        action_names = [list(ACTION_MAP.keys())[list(ACTION_MAP.values()).index(a)] for a in seq]
        print(f"  Trajectory {idx}: {action_names} (length={len(seq)})")
    
    if failed_trajectories:
        print(f"\nSample failed sequences (hit max steps):")
        for i, idx in enumerate(failed_trajectories[:2]):
            seq = action_sequences[idx]
            action_names = [list(ACTION_MAP.keys())[list(ACTION_MAP.values()).index(a)] for a in seq[:10]]
            print(f"  Trajectory {idx}: {action_names}... (length={len(seq)})")
    
    # Save processed sequences
    output_file = output_dir / f"action_sequences_{split}.pt"
    print(f"\nSaving processed sequences to {output_file}")
    
    # Convert to tensors (no padding yet - keep variable length)
    sequences_tensor_dict = {
        idx: torch.tensor(seq, dtype=torch.long)
        for idx, seq in action_sequences.items()
    }
    
    torch.save({
        'sequences': sequences_tensor_dict,
        'stats': {
            'total': len(action_sequences),
            'min_length': min_length,
            'max_length': max_length,
            'mean_length': mean_length,
            'length_distribution': dict(length_distribution),
            'min_examples': min_examples,
            'max_examples': max_examples,
            'failed_trajectories': failed_trajectories,
        },
        'metadata': metadata_by_idx,  # Store metadata for each trajectory
    }, output_file)
    
    # Also save human-readable JSON
    json_file = output_dir / f"action_sequences_{split}_stats.json"
    with open(json_file, 'w') as f:
        json.dump({
            'split': split,
            'total_trajectories': len(action_sequences),
            'failed_trajectories': len(failed_trajectories),
            'failed_trajectory_indices': failed_trajectories[:20],  # First 20
            'min_length': min_length,
            'max_length': max_length,
            'mean_length': mean_length,
            'median_length': sorted(lengths)[len(lengths)//2],
            'length_distribution': dict(length_distribution),
            'min_length_examples': min_examples[:10],
            'max_length_examples': max_examples[:10],
            'action_mapping': ACTION_MAP,
            'action_mapping_reverse': {v: k for k, v in ACTION_MAP.items()},
        }, f, indent=2)
    
    print(f"Saved statistics to {json_file}")
    
    return {
        'total': len(action_sequences),
        'min_length': min_length,
        'max_length': max_length,
        'mean_length': mean_length,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process action sequences from HuggingFace trajectory dataset"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="project-telos/train_and_val_full_trajectories",
        help="HuggingFace dataset repository ID"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Which split to process (train, val, or both)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed sequences"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process requested splits
    results = {}
    
    if args.split in ["train", "both"]:
        results['train'] = process_split(args.repo_id, "train", output_dir)
    
    if args.split in ["val", "both"]:
        results['val'] = process_split(args.repo_id, "val", output_dir)
    
    # Print combined summary if both splits processed
    if args.split == "both" and len(results) == 2:
        print(f"\n{'='*80}")
        print("COMBINED SUMMARY")
        print(f"{'='*80}")
        for split, stats in results.items():
            print(f"{split.upper()}: {stats['total']} trajectories, "
                  f"length range [{stats['min_length']}-{stats['max_length']}], "
                  f"mean={stats['mean_length']:.2f}")
        
        overall_max = max(stats['max_length'] for stats in results.values())
        print(f"\n✓ Recommended max_path_len for model: {overall_max}")
        print(f"  (This ensures no trajectories are truncated)")


if __name__ == "__main__":
    main()
