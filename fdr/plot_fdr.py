import os
import matplotlib.pyplot as plt
import numpy as np
from fdr.fdr_control import control_fdr
import pandas as pd


def plot_fdr_comprehensive(df, class_idx, dataset_name, save_dir='fdr_results'):
    os.makedirs(save_dir, exist_ok=True)

    methods = [
        ('q_values_tdc', 'TDC', 'blue', '-'),
        ('q_values_ground_truth', 'Ground Truth', 'green', '-'),
        ('q_values_bh_storey', 'BH (Storey π₀)', 'red', '-'),
        ('q_values_bh_fixed', 'BH (π₀=1)', 'orange', '-'),
    ]

    if 'q_values_negative_training' in df.columns:
        methods.append(('q_values_negative_training', 'Negative Training', 'brown', ':'))

    def create_plot(xlim_max, suffix):
        plt.figure(figsize=(16, 10))

        for col_name, label, color, linestyle in methods:
            if col_name in df.columns:
                df_sorted = df.sort_values(by=col_name, ascending=True).reset_index(drop=True)
                plt.plot(df_sorted[col_name], np.arange(len(df_sorted)),
                        marker='none', linestyle=linestyle, label=label, linewidth=2, color=color)

        plt.ylabel('Number of Discoveries', fontsize=12)
        plt.xlabel('Q-values', fontsize=12)

        title_suffix = f" (xlim={xlim_max})" if xlim_max != 1 else ""
        plt.title(f'FDR Control Methods - Class {class_idx} ({dataset_name}){title_suffix}',
                  fontsize=14, fontweight='bold')

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='lower right')
        plt.xlim(-0.01, xlim_max + 0.01)
        plt.ylim(0, len(df) * 1.05)

        for fdr_level in [0.01, 0.05, 0.1]:
            if fdr_level <= xlim_max:
                plt.axvline(x=fdr_level, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                plt.text(fdr_level, plt.ylim()[1]*0.95, f'FDR={fdr_level}',
                        rotation=90, va='top', fontsize=9)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'fdr_class_{class_idx}_{dataset_name}{suffix}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    save_paths = []

    save_path_full = create_plot(1.0, '')
    save_paths.append(save_path_full)
    print(f"  Plot saved to: {save_path_full}")

    save_path_zoomed = create_plot(0.2, '_zoom')
    save_paths.append(save_path_zoomed)
    print(f"  Zoomed plot saved to: {save_path_zoomed}")

    return save_paths


def process_multiclass_fdr(cnn_scores, decoy_scores, labels, dataset_name,
                           train_cnn_scores=None, train_labels=None,
                           num_classes=10, save_dir='fdr_results'):

    results_dict = {}
    summary_stats = []

    for class_idx in range(num_classes):

        class_cnn_scores = cnn_scores[:, class_idx]
        class_decoy_scores = decoy_scores[:, class_idx]

        binary_labels = (labels == class_idx).astype(int)

        if train_cnn_scores is not None and train_labels is not None and dataset_name != 'train':
            df_fdr = control_fdr(
                class_cnn_scores, binary_labels, class_decoy_scores,
                train_cnn_scores, train_labels, cnn_scores, class_idx
            )
        else:
            df_fdr = control_fdr(class_cnn_scores, binary_labels, class_decoy_scores)

        csv_path = save_fdr_data(df_fdr, class_idx, dataset_name, save_dir)
        plot_path = plot_fdr_comprehensive(df_fdr, class_idx, dataset_name, save_dir)

        results_dict[f'class_{class_idx}'] = df_fdr

        fdr_levels = [0.01, 0.05, 0.1]

        available_methods = ['q_values_tdc', 'q_values_ground_truth',
                           'q_values_bh_storey', 'q_values_bh_fixed',]

        if 'q_values_negative_training' in df_fdr.columns:
            available_methods.append('q_values_negative_training')
        if 'q_values_model_knockoff' in df_fdr.columns:
            available_methods.append('q_values_model_knockoff')

        for fdr_level in fdr_levels:
            for method in available_methods:
                n_discoveries = (df_fdr[method] <= fdr_level).sum()

                summary_stats.append({
                    'dataset': dataset_name,
                    'class': class_idx,
                    'method': method.replace('q_values_', ''),
                    'fdr_level': fdr_level,
                    'n_discoveries': n_discoveries,
                    'n_positive': binary_labels.sum(),
                    'n_total': len(labels)
                })

    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)

        print(f"\n{'='*60}")
        print(f"SUMMARY: Discoveries at FDR=0.05 ({dataset_name})")
        print(f"{'='*60}")

        # Filter for FDR=0.05
        fdr_05_data = summary_df[summary_df['fdr_level'] == 0.05]

        if not fdr_05_data.empty:
            pivot = fdr_05_data.pivot_table(
                index='class',
                columns='method',
                values='n_discoveries',
                aggfunc='first'
            )
            print(pivot)
        else:
            print("No data available for FDR=0.05")
    else:
        summary_df = pd.DataFrame()
        print("Warning: No summary statistics generated")

    return results_dict, summary_df


def save_fdr_data(df, class_idx, dataset_name, save_dir='fdr_results'):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'fdr_data_class_{class_idx}_{dataset_name}.csv')
    df.to_csv(save_path, index=False)
    print(f"  Data saved to: {save_path}")

    return save_path



def run_complete_fdr_analysis(test_cnn_scores, test_labels, separate_decoy_scores,
                              test_shifted_cnn_scores, test_shifted_labels,
                              separate_shifted_decoy_scores,
                              train_cnn_scores=None, train_labels=None,
                              train_decoy_scores=None,
                              num_classes=10, save_dir='fdr_results'):
    all_results = {}
    all_summaries = {}



    test_results, test_summary = process_multiclass_fdr(
        test_cnn_scores,
        separate_decoy_scores,
        test_labels,
        dataset_name='test',
        train_cnn_scores=train_cnn_scores,
        train_labels=train_labels,
        num_classes=num_classes,
        save_dir=save_dir
    )
    all_results['test'] = test_results
    all_summaries['test'] = test_summary


    shifted_results, shifted_summary = process_multiclass_fdr(
        test_shifted_cnn_scores,
        separate_shifted_decoy_scores,
        test_shifted_labels,
        dataset_name='shifted',
        train_cnn_scores=train_cnn_scores,
        train_labels=train_labels,
        num_classes=num_classes,
        save_dir=save_dir
    )
    all_results['shifted'] = shifted_results
    all_summaries['shifted'] = shifted_summary

    # Process train dataset if provided (no benchmarks for train)
    if train_cnn_scores is not None and train_labels is not None and train_decoy_scores is not None:
        print("\n" + "="*80)
        print("PROCESSING TRAIN DATASET")
        print("="*80)
        train_results, train_summary = process_multiclass_fdr(
            train_cnn_scores,
            train_decoy_scores,
            train_labels,
            dataset_name='train',
            num_classes=num_classes,
            save_dir=save_dir
        )
        all_results['train'] = train_results
        all_summaries['train'] = train_summary

    print(f"ALL RESULTS SAVED TO: {save_dir}/")
    print(f"{'='*80}\n")

    return all_results, all_summaries



def plot_fdr_multiclass(df, filename="fdr_multiclass"):
    """Plot FDR curves for multiclass classification"""
    plt.figure(figsize=(6, 4))
    
    methods = [
        ('q_values_mm', 'Mix-Max', 'blue', '-'),
        ('q_values_ground_truth', 'Ground Truth', 'green', '-'),
        ('q_values_bh_storey', 'BH (Storey π₀)', 'red', '--'),
    ]
    
    for col_name, label, color, linestyle in methods:
        if col_name in df.columns:
            # Remove NaN values first
            df_valid = df[~df[col_name].isna()].copy()
            df_sorted = df_valid.sort_values(by=col_name, ascending=True).reset_index(drop=True)
            
            plt.plot(df_sorted[col_name], 
                    np.arange(len(df_sorted)), 
                    marker='none', linestyle=linestyle, 
                    label=label, linewidth=2, color=color)
    
    plt.ylabel('Number of Discoveries')
    plt.xlabel('Q-values')
    plt.title('Multi-class FDR Control Methods')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    
    plt.savefig("BCSS/" + filename + ".png", bbox_inches='tight', dpi=300)
    plt.savefig("BCSS/" + filename + ".pdf", bbox_inches='tight')
    plt.show()
    
    return plt