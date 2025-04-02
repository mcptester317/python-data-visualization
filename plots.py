#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.stats import norm

def load_stats(csv_path):
    """Load statistics from CSV file."""
    print(f"Loading statistics from {csv_path}")
    stats = pd.read_csv(csv_path, index_col=0)
    print("Statistics loaded:")
    print(stats)
    return stats

def plot_normal_distribution(stats, output_path=None):
    """Plot approximated normal distributions of token lengths."""
    # Extract data for correct and incorrect
    correct_stats = stats.loc['Correct']
    incorrect_stats = stats.loc['Incorrect']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Generate x values
    x = np.linspace(0, max(correct_stats['max'], incorrect_stats['max']), 1000)
    
    # Generate normal distributions based on mean and std
    correct_pdf = norm.pdf(x, correct_stats['mean'], correct_stats['std'])
    incorrect_pdf = norm.pdf(x, incorrect_stats['mean'], incorrect_stats['std'])
    
    # Plot approximated distributions
    plt.plot(x, correct_pdf, label='Correct', color='blue')
    plt.plot(x, incorrect_pdf, label='Incorrect', color='orange')
    plt.fill_between(x, correct_pdf, alpha=0.3, color='blue')
    plt.fill_between(x, incorrect_pdf, alpha=0.3, color='orange')
    
    plt.title('Approximated Distribution of Token Lengths by Correctness')
    plt.xlabel('Token Length')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path / 'token_distribution.png', dpi=300)
        print(f"Saved distribution plot to {output_path / 'token_distribution.png'}")
    
    plt.close()

def plot_boxplot(stats, output_path=None):
    """Plot box plot of token lengths."""
    # Extract data for correct and incorrect
    correct_stats = stats.loc['Correct']
    incorrect_stats = stats.loc['Incorrect']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create box plot data
    box_data = [
        {'label': 'Correct', 
         'whislo': correct_stats['min'], 
         'q1': correct_stats['25%'], 
         'med': correct_stats['50%'], 
         'q3': correct_stats['75%'], 
         'whishi': correct_stats['max'],
         'fliers': []},
        {'label': 'Incorrect', 
         'whislo': incorrect_stats['min'], 
         'q1': incorrect_stats['25%'], 
         'med': incorrect_stats['50%'], 
         'q3': incorrect_stats['75%'], 
         'whishi': incorrect_stats['max'],
         'fliers': []}
    ]
    
    # Use ax.bxp instead of plt.bxp
    ax.bxp(box_data, positions=[1, 2], showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Correct', 'Incorrect'])
    ax.set_title('Token Length by Correctness')
    ax.set_ylabel('Token Length')
    ax.grid(axis='y', alpha=0.3)
    
    if output_path:
        plt.savefig(output_path / 'token_boxplot.png', dpi=300)
        print(f"Saved box plot to {output_path / 'token_boxplot.png'}")
    
    plt.close()

def plot_means_with_error(stats, output_path=None):
    """Plot bar chart of means with error bars."""
    # Extract data for correct and incorrect
    correct_stats = stats.loc['Correct']
    incorrect_stats = stats.loc['Incorrect']
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    means = [correct_stats['mean'], incorrect_stats['mean']]
    stds = [correct_stats['std'], incorrect_stats['std']]
    
    plt.bar([1, 2], means, yerr=stds, capsize=10, width=0.6, 
            color=['blue', 'orange'], alpha=0.7)
    plt.xticks([1, 2], ['Correct', 'Incorrect'])
    plt.title('Mean Token Length by Correctness')
    plt.ylabel('Mean Token Length')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate([1, 2]):
        plt.text(v, means[i] + stds[i] + 0.5, f'{means[i]:.2f}', 
                 ha='center', va='bottom', fontweight='bold')
    
    if output_path:
        plt.savefig(output_path / 'token_means.png', dpi=300)
        print(f"Saved means plot to {output_path / 'token_means.png'}")
    
    plt.close()

def plot_percentiles(stats, output_path=None):
    """Plot percentile comparison."""
    # Extract data for correct and incorrect
    correct_stats = stats.loc['Correct']
    incorrect_stats = stats.loc['Incorrect']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    percentiles = ['10%', '25%', '50%', '75%', '90%', '95%', '99%']
    
    # Extract percentile values
    correct_percentiles = [correct_stats[p] for p in percentiles]
    incorrect_percentiles = [incorrect_stats[p] for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    
    plt.bar(x - width/2, correct_percentiles, width, label='Correct', alpha=0.7, color='blue')
    plt.bar(x + width/2, incorrect_percentiles, width, label='Incorrect', alpha=0.7, color='orange')
    
    plt.xlabel('Percentile')
    plt.ylabel('Token Length')
    plt.title('Token Length Percentiles by Correctness')
    plt.xticks(x, percentiles)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    if output_path:
        plt.savefig(output_path / 'token_percentiles.png', dpi=300)
        print(f"Saved percentiles plot to {output_path / 'token_percentiles.png'}")
    
    plt.close()

def print_statistics(stats):
    """Print key statistical differences."""
    # Extract data for correct and incorrect
    correct_stats = stats.loc['Correct']
    incorrect_stats = stats.loc['Incorrect']
    
    # Calculate key differences
    diff_mean = correct_stats['mean'] - incorrect_stats['mean']
    diff_median = correct_stats['50%'] - incorrect_stats['50%']
    
    print("\nKey Differences (Correct - Incorrect):")
    print(f"Mean difference: {diff_mean:.2f} tokens")
    print(f"Median difference: {diff_median:.2f} tokens")
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((correct_stats['std']**2 + incorrect_stats['std']**2) / 2)
    cohens_d = diff_mean / pooled_std
    
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    print(f"Effect interpretation: {effect}")

def generate_all_plots(stats_file, output_dir="token_analysis_plots"):
    """Generate all individual plots from statistics CSV file."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load statistics
    stats = load_stats(stats_file)
    
    # Generate individual plots
    plot_normal_distribution(stats, output_path)
    plot_boxplot(stats, output_path)
    plot_means_with_error(stats, output_path)
    plot_percentiles(stats, output_path)
    
    # Print statistics
    print_statistics(stats)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Generate individual token length analysis visualizations")
    parser.add_argument("--stats", type=str, default="token_length_stats.csv",
                       help="Path to token length statistics CSV file")
    parser.add_argument("--output", type=str, default="token_analysis_plots",
                       help="Output directory for visualizations")
    parser.add_argument("--plot", type=str, choices=["all", "distribution", "boxplot", "means", "percentiles"],
                       default="all", help="Specific plot to generate (default: all)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load statistics
    stats = load_stats(args.stats)
    
    # Generate requested plots
    if args.plot == "all" or args.plot == "distribution":
        plot_normal_distribution(stats, output_path)
    
    if args.plot == "all" or args.plot == "boxplot":
        plot_boxplot(stats, output_path)
    
    if args.plot == "all" or args.plot == "means":
        plot_means_with_error(stats, output_path)
    
    if args.plot == "all" or args.plot == "percentiles":
        plot_percentiles(stats, output_path)
    
    # Print statistics
    print_statistics(stats)

if __name__ == "__main__":
    main()