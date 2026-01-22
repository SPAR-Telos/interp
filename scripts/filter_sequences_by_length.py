"""Filter action sequences by length and create new processed files.

This script loads the processed action sequences and filters them to keep only
sequences within a specified length range.

Usage:
    python scripts/filter_sequences_by_length.py --max-length 36 --min-length 1
"""

import argparse
from pathlib import Path

import torch


def filter_sequences(
    input_file: Path,
    output_file: Path,
    min_length: int = 1,
    max_length: int = 36,
):
    """Filter sequences by length.
    
    Args:
        input_file: Path to input .pt file
        output_file: Path to output .pt file
        min_length: Minimum sequence length to keep
        max_length: Maximum sequence length to keep
    """
    print(f"\nLoading {input_file}...")
    data = torch.load(input_file)
    
    sequences = data['sequences']
    stats = data['stats']
    metadata = data.get('metadata', {})
    
    print(f"Original: {len(sequences)} sequences")
    print(f"Length range: {stats['min_length']}-{stats['max_length']}")
    
    # Filter sequences
    filtered_sequences = {}
    filtered_metadata = {}
    length_distribution = {}
    
    for idx, seq in sequences.items():
        # Count non-padding tokens
        if seq.dtype == torch.long:
            # For integer sequences, count non--100 tokens
            actual_length = (seq != -100).sum().item()
        else:
            # For float sequences, just use length
            actual_length = len(seq)
        
        if min_length <= actual_length <= max_length:
            filtered_sequences[idx] = seq
            if metadata:
                filtered_metadata[idx] = metadata.get(idx)
            
            # Update length distribution
            if actual_length not in length_distribution:
                length_distribution[actual_length] = 0
            length_distribution[actual_length] += 1
    
    # Compute new stats
    if filtered_sequences:
        lengths = []
        for seq in filtered_sequences.values():
            if seq.dtype == torch.long:
                actual_length = (seq != -100).sum().item()
            else:
                actual_length = len(seq)
            lengths.append(actual_length)
        
        new_stats = {
            'total': len(filtered_sequences),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': sum(lengths) / len(lengths),
            'length_distribution': length_distribution,
            'filter_min': min_length,
            'filter_max': max_length,
            'original_total': stats['total'],
            'filtered_out': stats['total'] - len(filtered_sequences),
        }
    else:
        new_stats = {
            'total': 0,
            'filter_min': min_length,
            'filter_max': max_length,
            'original_total': stats['total'],
            'filtered_out': stats['total'],
        }
    
    print(f"\nFiltered: {len(filtered_sequences)} sequences ({100*len(filtered_sequences)/len(sequences):.1f}%)")
    print(f"Removed: {len(sequences) - len(filtered_sequences)} sequences")
    print(f"New length range: {new_stats['min_length']}-{new_stats['max_length']}")
    print(f"New mean length: {new_stats['mean_length']:.1f}")
    
    # Print length distribution
    print(f"\nLength distribution:")
    for length in sorted(length_distribution.keys()):
        count = length_distribution[length]
        percentage = 100 * count / len(filtered_sequences)
        bar = '█' * int(percentage / 2)
        print(f"  Length {length:2d}: {count:4d} ({percentage:5.2f}%) {bar}")
    
    # Save filtered data
    output_data = {
        'sequences': filtered_sequences,
        'stats': new_stats,
    }
    if filtered_metadata:
        output_data['metadata'] = filtered_metadata
    
    torch.save(output_data, output_file)
    print(f"\nSaved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter action sequences by length"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing input .pt files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save filtered .pt files"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum sequence length to keep"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=36,
        help="Maximum sequence length to keep"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_filtered",
        help="Suffix to add to output filename"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and val splits
    for split in ["train", "val"]:
        input_file = input_dir / f"action_sequences_{split}.pt"
        if not input_file.exists():
            print(f"⚠️  {input_file} not found, skipping...")
            continue
        
        output_file = output_dir / f"action_sequences_{split}{args.suffix}.pt"
        
        print(f"\n{'='*80}")
        print(f"Filtering {split.upper()} split")
        print(f"{'='*80}")
        
        filter_sequences(
            input_file,
            output_file,
            min_length=args.min_length,
            max_length=args.max_length,
        )
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Filtered sequences to length range: [{args.min_length}, {args.max_length}]")
    print(f"\n✓ Use max_path_len={args.max_length} in your OneShotPrefixProbe model")
    print(f"  This means {args.max_length} learned query tokens")


if __name__ == "__main__":
    main()
