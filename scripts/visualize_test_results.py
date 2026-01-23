"""Create publication-quality visualizations of test results.

This script generates figures showing:
1. Prefix accuracy degradation by grid size
2. Comparison with random baseline
3. Token accuracy by grid size

Usage:
    python scripts/visualize_test_results.py \
      --results-file results/decoder_probe_layer15_full_test_results.json \
      --output-dir figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Publication-quality settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8


def compute_random_baseline(num_actions=4, max_steps=15):
    """Compute random baseline prefix accuracies.
    
    For N consecutive correct predictions with random guessing:
    P(all N correct) = (1/num_actions)^N
    
    Args:
        num_actions: Number of possible actions (4 for LEFT/RIGHT/UP/DOWN)
        max_steps: Maximum number of steps to compute
    
    Returns:
        List of probabilities for each prefix length
    """
    baseline = []
    p = 1.0 / num_actions  # 0.25 for 4 actions
    
    for n in range(1, max_steps + 1):
        baseline.append(p ** n * 100)  # Convert to percentage
    
    return baseline


def plot_prefix_accuracies(results_data, output_dir):
    """Create prefix accuracy plot comparing grid sizes with random baseline.
    
    Args:
        results_data: Dictionary containing test results
        output_dir: Directory to save figures
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for different grid sizes (colorblind-friendly palette)
    colors = {
        7: '#0173B2',   # Blue
        9: '#DE8F05',   # Orange  
        11: '#029E73',  # Green
        13: '#CC78BC',  # Purple
        15: '#CA9161',  # Brown
    }
    
    markers = {
        7: 'o',
        9: 's',
        11: '^',
        13: 'D',
        15: 'v',
    }
    
    # Plot each grid size
    grid_sizes = sorted(results_data['grid_sizes'].keys(), key=lambda x: int(x))
    
    for size_str in grid_sizes:
        size = int(size_str)
        metrics = results_data['grid_sizes'][size_str]['metrics']
        prefix_accs = metrics['prefix_accuracies']
        
        # Convert to lists
        steps = sorted([int(k) for k in prefix_accs.keys()])
        accuracies = [prefix_accs[str(step)] * 100 for step in steps]  # Convert to percentage
        
        # Plot line
        ax.plot(steps, accuracies, 
                marker=markers[size], 
                color=colors[size],
                label=f'Grid Size {size}',
                linewidth=2.5,
                markersize=8,
                markeredgewidth=1.5,
                markeredgecolor='white',
                alpha=0.9)
    
    # Compute and plot random baseline
    max_steps = max([max([int(k) for k in results_data['grid_sizes'][s]['metrics']['prefix_accuracies'].keys()]) 
                     for s in grid_sizes])
    random_baseline = compute_random_baseline(num_actions=4, max_steps=max_steps)
    baseline_steps = list(range(1, len(random_baseline) + 1))
    
    ax.plot(baseline_steps, random_baseline,
            linestyle='--',
            color='black',
            linewidth=2.5,
            label='Random Baseline (25% per step)',
            alpha=0.7,
            zorder=1)
    
    # Styling
    ax.set_xlabel('Prefix Length (First N Steps)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Prefix Accuracy Degradation by Grid Size', fontweight='bold', pad=20)
    
    # Set x-axis to integer values only
    max_steps_shown = min(15, max_steps)  # Show up to 15 steps
    ax.set_xticks(range(1, max_steps_shown + 1))
    ax.set_xlim(0.5, max_steps_shown + 0.5)
    
    # Set y-axis to start at 0
    max_acc = max([max(accuracies) for size_str in grid_sizes 
                   for accuracies in [[results_data['grid_sizes'][size_str]['metrics']['prefix_accuracies'][str(k)] * 100 
                                      for k in sorted([int(x) for x in results_data['grid_sizes'][size_str]['metrics']['prefix_accuracies'].keys()])]]])
    ax.set_ylim(0, min(50, max_acc + 5))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fancybox=False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'prefix_accuracy_by_grid_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path}")
    
    # Also save as PDF for publications
    output_path_pdf = Path(output_dir) / 'prefix_accuracy_by_grid_size.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path_pdf}")
    
    plt.close()


def plot_token_accuracy_by_grid_size(results_data, output_dir):
    """Create bar chart of token accuracy by grid size.
    
    Args:
        results_data: Dictionary containing test results
        output_dir: Directory to save figures
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    grid_sizes = sorted([int(k) for k in results_data['grid_sizes'].keys()])
    token_accs = [results_data['grid_sizes'][str(size)]['metrics']['token_accuracy'] * 100 
                  for size in grid_sizes]
    
    # Colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Create bar chart
    bars = ax.bar(grid_sizes, token_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, token_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add random baseline line
    ax.axhline(y=25, color='black', linestyle='--', linewidth=2, alpha=0.7,
               label='Random Baseline (25%)')
    
    # Styling
    ax.set_xlabel('Grid Size', fontweight='bold')
    ax.set_ylabel('Token Accuracy (%)', fontweight='bold')
    ax.set_title('Token Accuracy by Grid Size', fontweight='bold', pad=20)
    ax.set_xticks(grid_sizes)
    ax.set_ylim(0, max(token_accs) + 10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1, axis='y')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fancybox=False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'token_accuracy_by_grid_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path}")
    
    output_path_pdf = Path(output_dir) / 'token_accuracy_by_grid_size.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path_pdf}")
    
    plt.close()


def plot_sequence_accuracy_by_grid_size(results_data, output_dir):
    """Create bar chart of sequence accuracy by grid size.
    
    Args:
        results_data: Dictionary containing test results
        output_dir: Directory to save figures
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    grid_sizes = sorted([int(k) for k in results_data['grid_sizes'].keys()])
    seq_accs = [results_data['grid_sizes'][str(size)]['metrics']['sequence_accuracy'] * 100 
                for size in grid_sizes]
    
    # Colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Create bar chart
    bars = ax.bar(grid_sizes, seq_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, seq_accs):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Grid Size', fontweight='bold')
    ax.set_ylabel('Sequence Accuracy (%)', fontweight='bold')
    ax.set_title('Perfect Sequence Accuracy by Grid Size', fontweight='bold', pad=20)
    ax.set_xticks(grid_sizes)
    ax.set_ylim(0, max(seq_accs) + 3 if max(seq_accs) > 0 else 10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1, axis='y')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'sequence_accuracy_by_grid_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path}")
    
    output_path_pdf = Path(output_dir) / 'sequence_accuracy_by_grid_size.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {output_path_pdf}")
    
    plt.close()


def create_summary_table(results_data, output_dir):
    """Create a LaTeX-formatted summary table.
    
    Args:
        results_data: Dictionary containing test results
        output_dir: Directory to save table
    """
    grid_sizes = sorted([int(k) for k in results_data['grid_sizes'].keys()])
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Test Set Performance by Grid Size}")
    lines.append("\\label{tab:test_results}")
    lines.append("\\begin{tabular}{c|ccc|ccc}")
    lines.append("\\hline")
    lines.append("Grid & Token & Sequence & N=1 & N=2 & N=3 & N=4 \\\\")
    lines.append("Size & Acc. (\\%) & Acc. (\\%) & Prefix & Prefix & Prefix & Prefix \\\\")
    lines.append("\\hline")
    
    for size in grid_sizes:
        size_str = str(size)
        metrics = results_data['grid_sizes'][size_str]['metrics']
        token_acc = metrics['token_accuracy'] * 100
        seq_acc = metrics['sequence_accuracy'] * 100
        prefix_accs = metrics['prefix_accuracies']
        
        n1 = prefix_accs.get('1', 0) * 100
        n2 = prefix_accs.get('2', 0) * 100
        n3 = prefix_accs.get('3', 0) * 100
        n4 = prefix_accs.get('4', 0) * 100
        
        lines.append(f"{size} & {token_acc:.1f} & {seq_acc:.1f} & {n1:.1f} & {n2:.1f} & {n3:.1f} & {n4:.1f} \\\\")
    
    lines.append("\\hline")
    
    # Overall
    overall = results_data['overall']
    token_acc_overall = overall['token_accuracy'] * 100
    seq_acc_overall = overall['sequence_accuracy'] * 100
    lines.append(f"\\textbf{{Overall}} & \\textbf{{{token_acc_overall:.1f}}} & \\textbf{{{seq_acc_overall:.1f}}} & \\multicolumn{{4}}{{c}}{{---}} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Save
    output_path = Path(output_dir) / 'results_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Saved LaTeX table: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize test results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/decoder_probe_layer15_full_test_results.json",
        help="Path to test results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/",
        help="Directory to save figures",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}")
    print(f"  Results file: {args.results_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Load results
    with open(args.results_file, 'r') as f:
        results_data = json.load(f)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    print("Generating figures...")
    
    print("\n1. Prefix Accuracy Plot")
    plot_prefix_accuracies(results_data, args.output_dir)
    
    print("\n2. Token Accuracy Bar Chart")
    plot_token_accuracy_by_grid_size(results_data, args.output_dir)
    
    print("\n3. Sequence Accuracy Bar Chart")
    plot_sequence_accuracy_by_grid_size(results_data, args.output_dir)
    
    print("\n4. Summary Table (LaTeX)")
    create_summary_table(results_data, args.output_dir)
    
    print(f"\n{'='*80}")
    print("Visualization Complete!")
    print(f"{'='*80}")
    print(f"  Figures saved to: {args.output_dir}")
    print(f"  - prefix_accuracy_by_grid_size.png/pdf")
    print(f"  - token_accuracy_by_grid_size.png/pdf")
    print(f"  - sequence_accuracy_by_grid_size.png/pdf")
    print(f"  - results_table.tex")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
