import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from bisect import bisect
import os
from fdr.fdr_control import *

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_estimation(df, q_value_column, method_name, corruption_name, save_dir):
    """Plot accuracy estimation curve with threshold-matched true accuracy."""
    os.makedirs(save_dir, exist_ok=True)
    
    pi0       = estimate_pi0_storey_from_df(df, q_value_column)
    df_gt     = compute_ground_truth_curve(df)
    df_method = compute_method_estimation_curve(df, q_value_column, pi0)
    
    if len(df_method) == 0:
        print(f"  No valid data for {corruption_name}")
        return None, pi0
    
    # ── Find best threshold ───────────────────────────────────────────────
    best_idx      = df_method['Accuracy_est'].idxmax()
    best_q        = df_method.loc[best_idx, 'q_value_method']
    best_acc_est  = df_method.loc[best_idx, 'Accuracy_est']
    best_acc_true = df_method.loc[best_idx, 'Accuracy_true_at_threshold']
    best_error    = df_method.loc[best_idx, 'error_at_threshold']
    best_n        = df_method.loc[best_idx, 'n_discoveries']

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    
    ax.plot(df_method['q_value_method'], df_method['Accuracy_est'],
            color='#E53935', linewidth=2.5,
            label=f'{method_name} (est)', alpha=0.9)
    
    ax.plot(df_method['q_value_method'], df_method['Accuracy_true_at_threshold'],
            color='#1976D2', linewidth=2.5, linestyle='-.',
            label='True acc @ threshold', alpha=0.9)
    
    ax.plot(df_gt['q_value_gt'], df_gt['accuracy_gt'],
            color='black', linewidth=2.5, linestyle='--',
            label='Ground Truth', alpha=0.7)
    
    # Mark best threshold
    ax.axvline(best_q, color='#E53935', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.scatter([best_q], [best_acc_est],  color='#E53935', s=80, zorder=5)
    ax.scatter([best_q], [best_acc_true], color='#1976D2', s=80, zorder=5)
    
    # FDR reference lines
    for fdr in [0.05, 0.10, 0.20]:
        ax.axvline(fdr, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax.text(fdr, 0.02, f'{fdr:.2f}', rotation=0,
                va='bottom', ha='center', fontsize=9, alpha=0.6)
    
    ax.set_xlabel('Q-value (FDR threshold)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, min(0.5, df_method['q_value_method'].max() + 0.02))
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    corruption_title = corruption_name.replace('_', ' ').title()
    ax.set_title(f'{corruption_title}', fontweight='bold')
    
    filename = f'{save_dir}/accuracy_{corruption_name}'
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {corruption_name}: "
          f"π₀={pi0:.3f}  "
          f"best_q={best_q:.3f}  "
          f"n_accepted={best_n}/{len(df)}  "
          f"est_acc={best_acc_est:.3f}  "
          f"true_acc={best_acc_true:.3f}  "
          f"error={best_error:.3f}")
    
    return df_method, pi0


print("✓ Updated estimation functions loaded")


def plot_fdr_curve(df, corruption_name, save_dir):
    """Plot FDR control curve with Mix-Max (publication quality, 6×4 inches)."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    
    methods = [
        ('q_values_ground_truth', 'Ground Truth',    '#2E7D32', '--', 2.5),
        ('q_values_fdr2',         'FDR2 (TDC)',       '#F57C00', '-',  2.0),
        ('q_values_bh',           'Benjamini-Hochberg','#7B1FA2', '-',  2.0),
        ('q_values_mixmax',       'Mix-Max',           '#E53935', '-',  2.5),
    ]
    
    for col_name, label, color, linestyle, linewidth in methods:
        if col_name not in df.columns:
            continue
        df_plot = df[~df[col_name].isna()].copy()
        if len(df_plot) == 0:
            continue
        df_sorted    = df_plot.sort_values(col_name).reset_index(drop=True)
        q_vals       = df_sorted[col_name].values
        n_discoveries = np.arange(1, len(df_sorted) + 1)
        ax.plot(q_vals, n_discoveries, label=label, linewidth=linewidth,
                color=color, linestyle=linestyle, alpha=0.8)
    
    for fdr in [0.01, 0.05, 0.1]:
        ax.axvline(fdr, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.text(fdr, ax.get_ylim()[1] * 0.95, f'{fdr:.2f}',
                rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Q-value (FDR)', fontweight='bold')
    ax.set_ylabel('Number of Discoveries', fontweight='bold')
    ax.set_xlim(0, 0.2)
    ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    
    corruption_title = corruption_name.replace('_', ' ').title()
    ax.set_title(f'{corruption_title}', fontweight='bold')
    
    filename = f'{save_dir}/fdr_{corruption_name}'
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ {corruption_name}")


def plot_method_comparison(results_df, corruption_name, save_dir='figures/comparison'):
    """Compare all methods for a single corruption (publication quality, 6×4 inches)."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    
    df_corr  = results_df[results_df['corruption'] == corruption_name]
    true_acc = df_corr['true_accuracy'].iloc[0]
    df_sorted = df_corr.sort_values('estimated_accuracy', ascending=False)
    
    methods   = df_sorted['method'].values
    estimates = df_sorted['estimated_accuracy'].values
    errors    = np.abs(estimates - true_acc)
    
    colors = []
    for method in methods:
        if 'Mix-Max' in method:
            colors.append('#E53935')   # Red  – Mix-Max
        elif method in ['FDR2', 'B-H']:
            colors.append('#7B1FA2')   # Purple – other FDR
        else:
            colors.append('#1976D2')   # Blue   – baselines
    
    x_pos = np.arange(len(methods))
    bars  = ax.bar(x_pos, estimates, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)
    ax.axhline(true_acc, color='red', linestyle='--', linewidth=2.5,
               label=f'True Accuracy ({true_acc:.3f})', alpha=0.8)
    
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{err:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Estimated Accuracy', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    corruption_title = corruption_name.replace('_', ' ').title()
    ax.set_title(f'Method Comparison: {corruption_title}', fontweight='bold')
    
    filename = f'{save_dir}/comparison_{corruption_name}'
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ {corruption_name}")


def plot_overall_comparison(results_df, save_dir='figures/comparison'):
    """Overall MAE comparison across all corruptions."""
    os.makedirs(save_dir, exist_ok=True)
    
    mae_data = []
    for method in results_df['method'].unique():
        df_m = results_df[results_df['method'] == method]
        mae  = np.abs(df_m['estimated_accuracy'] - df_m['true_accuracy']).mean()
        mae_data.append({'Method': method, 'MAE': mae})
    mae_df = pd.DataFrame(mae_data).sort_values('MAE')
    
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    
    colors = []
    for method in mae_df['Method']:
        if 'Mix-Max' in method:
            colors.append('#E53935')
        elif method in ['FDR2', 'B-H']:
            colors.append('#7B1FA2')
        else:
            colors.append('#1976D2')
    
    x_pos = np.arange(len(mae_df))
    ax.bar(x_pos, mae_df['MAE'], color=colors, alpha=0.7,
           edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mae_df['Method'], rotation=45, ha='right')
    ax.set_title('Overall Performance Across All Corruptions', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (_, row) in enumerate(mae_df.iterrows()):
        ax.text(i, row['MAE'] + 0.005, f"{row['MAE']:.4f}",
                ha='center', va='bottom', fontsize=9)
    
    filename = f'{save_dir}/overall_comparison'
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()
    
    print("\n✓ Overall comparison plot saved")
    print("\nMean Absolute Error (MAE) by Method:")
    print(mae_df.to_string(index=False))
    return mae_df


def plot_scatter_estimated_vs_true(results_df, save_dir='figures/comparison'):
    """
    Scatter plot: estimated accuracy vs true accuracy.
    One point per corruption per method.
    Perfect estimation = diagonal line.
    """
    os.makedirs(save_dir, exist_ok=True)

    methods = results_df['method'].unique()

    # ── 1. One plot per method ────────────────────────────────────────────────
    for method in methods:
        df_m = results_df[results_df['method'] == method].copy()

        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

        color = '#E53935' if 'Mix-Max' in method else \
                '#7B1FA2' if method in ['FDR2', 'B-H'] else '#1976D2'

        ax.scatter(df_m['true_accuracy'], df_m['estimated_accuracy'],
                   color=color, s=60, alpha=0.8, edgecolors='black', linewidths=0.5,
                   label=method)

        # Annotate each point with corruption name
        for _, row in df_m.iterrows():
            ax.annotate(row['corruption'].replace('_', ' '),
                        (row['true_accuracy'], row['estimated_accuracy']),
                        fontsize=6, alpha=0.7,
                        xytext=(4, 4), textcoords='offset points')

        # Perfect diagonal
        lo = min(df_m['true_accuracy'].min(), df_m['estimated_accuracy'].min()) - 0.02
        hi = max(df_m['true_accuracy'].max(), df_m['estimated_accuracy'].max()) + 0.02
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, alpha=0.6, label='Perfect')

        mae = np.abs(df_m['estimated_accuracy'] - df_m['true_accuracy']).mean()
        ax.set_xlabel('True Accuracy', fontweight='bold')
        ax.set_ylabel('Estimated Accuracy', fontweight='bold')
        ax.set_title(f'{method}\nMAE = {mae:.4f}', fontweight='bold')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        safe_name = method.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        filename  = f'{save_dir}/scatter_{safe_name}'
        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(filename + '.pdf', bbox_inches='tight')
        plt.close()

    # ── 2. All methods on one axes ────────────────────────────────────────────
    method_colors = {}
    for method in methods:
        method_colors[method] = (
            '#E53935' if 'Mix-Max' in method else
            '#7B1FA2' if method in ['FDR2', 'B-H'] else
            '#1976D2'
        )

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    for method in methods:
        df_m  = results_df[results_df['method'] == method]
        color = method_colors[method]
        ax.scatter(df_m['true_accuracy'], df_m['estimated_accuracy'],
                   color=color, s=50, alpha=0.75,
                   edgecolors='black', linewidths=0.4,
                   label=method)

    # Perfect diagonal
    all_vals = pd.concat([results_df['true_accuracy'], results_df['estimated_accuracy']])
    lo, hi   = all_vals.min() - 0.02, all_vals.max() + 0.02
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=2, alpha=0.7, label='Perfect estimation')

    ax.set_xlabel('True Accuracy', fontweight='bold')
    ax.set_ylabel('Estimated Accuracy', fontweight='bold')
    ax.set_title('Estimated vs True Accuracy (All Methods)', fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.9, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')

    filename = f'{save_dir}/scatter_all_methods'
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    print("✓ Scatter plots saved")


print("✓ Plotting functions loaded (including Mix-Max + scatter)")


def plot_method_comparison(results_df, corruption_name, save_dir='figures/comparison'):
    """
    Compare all methods (FDR + Baselines) for a single corruption.
    Publication quality 6x4 inches.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    
    # Filter for this corruption
    df_corr = results_df[results_df['corruption'] == corruption_name]
    
    # Get ground truth
    true_acc = df_corr['true_accuracy'].iloc[0]
    
    # Sort methods by estimate
    df_sorted = df_corr.sort_values('estimated_accuracy', ascending=False)
    
    methods = df_sorted['method'].values
    estimates = df_sorted['estimated_accuracy'].values
    errors = np.abs(estimates - true_acc)
    
    # Color mapping
    colors = []
    for method in methods:
        if 'BC-Min' in method:
            colors.append('#FF9800')  # Orange - our method
        elif method in ['FDR2', 'B-H']:
            colors.append('#7B1FA2')  # Purple - other FDR
        else:
            colors.append('#1976D2')  # Blue - baselines
    
    # Bar plot
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, estimates, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Ground truth line
    ax.axhline(true_acc, color='red', linestyle='--', linewidth=2.5, 
               label=f'True Accuracy ({true_acc:.3f})', alpha=0.8)
    
    # Add error annotations on bars
    for i, (bar, err) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{err:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Estimated Accuracy', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    corruption_title = corruption_name.replace('_', ' ').title()
    ax.set_title(f'Method Comparison: {corruption_title}', fontweight='bold')
    
    filename = f'{save_dir}/comparison_{corruption_name}'
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {corruption_name}")


def plot_overall_comparison(results_df, save_dir='figures/comparison'):
    """
    Overall comparison across all corruptions.
    Shows mean absolute error for each method.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute MAE for each method
    mae_data = []
    for method in results_df['method'].unique():
        df_method = results_df[results_df['method'] == method]
        mae = np.abs(df_method['estimated_accuracy'] - df_method['true_accuracy']).mean()
        mae_data.append({'Method': method, 'MAE': mae})
    
    mae_df = pd.DataFrame(mae_data).sort_values('MAE')
    
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    
    # Color by method type
    colors = []
    for method in mae_df['Method']:
        if 'BC-Min' in method:
            colors.append('#FF9800')
        elif method in ['FDR2', 'B-H']:
            colors.append('#7B1FA2')
        else:
            colors.append('#1976D2')
    
    x_pos = np.arange(len(mae_df))
    ax.bar(x_pos, mae_df['MAE'], color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mae_df['Method'], rotation=45, ha='right')
    ax.set_title('Overall Performance Across All Corruptions', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(mae_df.iterrows()):
        ax.text(i, row['MAE'] + 0.005, f"{row['MAE']:.4f}",
                ha='center', va='bottom', fontsize=9)
    
    filename = f'{save_dir}/overall_comparison'
    plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()
    
    print("\n✓ Overall comparison plot saved")
    print("\nMean Absolute Error (MAE) by Method:")
    print(mae_df.to_string(index=False))
    
    return mae_df


print("✓ Comparison plotting functions loaded")