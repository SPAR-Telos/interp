#!/usr/bin/env python3
"""
Script to create a balanced grid dataset for probing by sampling agent and goal positions
from multiple grids to ensure equal representation of all cell types.
"""

import argparse
import os
import random
from pathlib import Path

import pandas as pd
import numpy as np


def create_balanced_grid_dataset(
    input_csv_path: str,
    output_csv_path: str,
    target_samples_per_class: int = 1000,
    seed: int = 42
) -> None:
    """
    Create a balanced dataset by sampling equal numbers of each cell type.
    
    Args:
        input_csv_path: Path to the original grid CSV
        output_csv_path: Path to save the balanced CSV
        target_samples_per_class: Target number of samples per class
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Loading data from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Analyze current class distribution
    print("\n📊 Current class distribution:")
    class_counts = df['cell_type'].value_counts()
    for cell_type, count in class_counts.items():
        print(f"  {cell_type}: {count:,}")
    
    # Group by cell type
    balanced_samples = []
    
    for cell_type in ['wall', 'empty', 'agent', 'goal']:
        if cell_type not in class_counts:
            print(f"⚠️  Warning: No {cell_type} samples found in dataset")
            continue
            
        cell_data = df[df['cell_type'] == cell_type]
        available_samples = len(cell_data)
        
        print(f"\n🎯 Processing {cell_type} class:")
        print(f"  Available samples: {available_samples:,}")
        print(f"  Target samples: {target_samples_per_class:,}")
        
        if available_samples >= target_samples_per_class:
            # Sample without replacement
            sampled_data = cell_data.sample(n=target_samples_per_class, random_state=seed)
            print(f"  ✅ Sampled {target_samples_per_class:,} samples")
        else:
            # Sample with replacement to reach target
            sampled_data = cell_data.sample(n=target_samples_per_class, replace=True, random_state=seed)
            print(f"  ⚠️  Sampled {target_samples_per_class:,} samples (with replacement)")
        
        balanced_samples.append(sampled_data)
    
    # Combine all balanced samples
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    
    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the balanced dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    balanced_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Balanced dataset saved to {output_csv_path}")
    print(f"📊 Final class distribution:")
    final_counts = balanced_df['cell_type'].value_counts()
    for cell_type, count in final_counts.items():
        print(f"  {cell_type}: {count:,}")
    
    print(f"\n📈 Total samples: {len(balanced_df):,}")


def create_balanced_dataset_with_oversampling(
    input_csv_path: str,
    output_csv_path: str,
    minority_multiplier: int = 5,
    seed: int = 42
) -> None:
    """
    Create a balanced dataset by oversampling minority classes.
    
    Args:
        input_csv_path: Path to the original grid CSV
        output_csv_path: Path to save the balanced CSV
        minority_multiplier: How many times to oversample minority classes
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Loading data from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Analyze current class distribution
    print("\n📊 Current class distribution:")
    class_counts = df['cell_type'].value_counts()
    for cell_type, count in class_counts.items():
        print(f"  {cell_type}: {count:,}")
    
    # Find the majority class size
    majority_size = class_counts.max()
    print(f"\n🎯 Majority class size: {majority_size:,}")
    
    balanced_samples = []
    
    for cell_type in ['wall', 'empty', 'agent', 'goal']:
        if cell_type not in class_counts:
            print(f"⚠️  Warning: No {cell_type} samples found in dataset")
            continue
            
        cell_data = df[df['cell_type'] == cell_type]
        current_size = len(cell_data)
        
        print(f"\n🔄 Processing {cell_type} class:")
        print(f"  Current size: {current_size:,}")
        
        if cell_type in ['agent', 'goal']:
            # Oversample minority classes
            target_size = min(current_size * minority_multiplier, majority_size)
            if target_size > current_size:
                # Sample with replacement
                additional_samples = cell_data.sample(
                    n=target_size - current_size, 
                    replace=True, 
                    random_state=seed
                )
                sampled_data = pd.concat([cell_data, additional_samples], ignore_index=True)
                print(f"  ✅ Oversampled to {target_size:,} samples")
            else:
                sampled_data = cell_data
                print(f"  ✅ Kept original {current_size:,} samples")
        else:
            # For wall and empty, sample down to a reasonable size
            if current_size > majority_size:
                sampled_data = cell_data.sample(n=majority_size, random_state=seed)
                print(f"  ✅ Downsampled to {majority_size:,} samples")
            else:
                sampled_data = cell_data
                print(f"  ✅ Kept original {current_size:,} samples")
        
        balanced_samples.append(sampled_data)
    
    # Combine all balanced samples
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    
    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save the balanced dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    balanced_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Balanced dataset saved to {output_csv_path}")
    print(f"📊 Final class distribution:")
    final_counts = balanced_df['cell_type'].value_counts()
    for cell_type, count in final_counts.items():
        print(f"  {cell_type}: {count:,}")
    
    print(f"\n📈 Total samples: {len(balanced_df):,}")


def main():
    parser = argparse.ArgumentParser(description="Create balanced grid datasets for probing")
    parser.add_argument("input_csv", help="Path to input grid CSV file")
    parser.add_argument("output_csv", help="Path to output balanced CSV file")
    parser.add_argument(
        "--method", 
        choices=["equal", "oversample"], 
        default="equal",
        help="Balancing method: 'equal' for equal samples per class, 'oversample' for oversampling minority classes"
    )
    parser.add_argument(
        "--target-samples", 
        type=int, 
        default=1000,
        help="Target number of samples per class (for equal method)"
    )
    parser.add_argument(
        "--minority-multiplier", 
        type=int, 
        default=5,
        help="Multiplier for oversampling minority classes (for oversample method)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.method == "equal":
        create_balanced_grid_dataset(
            args.input_csv, 
            args.output_csv, 
            args.target_samples, 
            args.seed
        )
    elif args.method == "oversample":
        create_balanced_dataset_with_oversampling(
            args.input_csv, 
            args.output_csv, 
            args.minority_multiplier, 
            args.seed
        )


if __name__ == "__main__":
    main()
