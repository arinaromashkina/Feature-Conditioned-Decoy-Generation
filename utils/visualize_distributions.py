import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def create_plots_directory(base_dir="plots"):
    plots_dir = Path(base_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def plot_class_score_distributions(model_scores, decoy_scores, labels, method_name,
                                   bins=100, num_classes=10, save_dir="plots", 
                                   save_prefix="", show_plots=False):
    plots_dir = create_plots_directory(save_dir)
    n_cols = 5
    n_rows = (num_classes + n_cols - 1) // n_cols

    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = axes.flatten()

    for class_idx in range(num_classes):
        ax = axes[class_idx]
        class_model_scores = model_scores[:, class_idx]
        class_decoy_scores = decoy_scores[:, class_idx]
        mask_correct = (labels == class_idx)
        mask_incorrect = (labels != class_idx)
        
        ax.hist(class_model_scores[mask_incorrect], bins=bins//2, alpha=0.5,
                label='Wrong Class', color='red', density=True)
        ax.hist(class_decoy_scores[mask_incorrect], bins=bins//2, alpha=0.5,
                label='Decoy (Wrong)', color='blue', density=True)
        ax.hist(class_model_scores[mask_correct], bins=bins//2, alpha=0.3,
                label='Correct Class', color='green', density=True)

        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Class {class_idx}', fontsize=10)
        ax.grid(True, alpha=0.3)

        if class_idx == 0:
            ax.legend(fontsize=8)

    for i in range(num_classes, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Separate Score Distributions by Class - {method_name}',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename1 = f"{save_prefix}class_distributions_{method_name.replace(' ', '_').lower()}.png"
    save_path1 = plots_dir / filename1
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig1)
    
    fig2, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    correct_scores = []
    correct_decoys = []

    for i in range(len(labels)):
        correct_class = labels[i]
        correct_scores.append(model_scores[i, correct_class])
        correct_decoys.append(decoy_scores[i, correct_class])

    correct_scores = np.array(correct_scores)
    correct_decoys = np.array(correct_decoys)

    axes[0].hist(correct_scores, bins=bins, alpha=0.5,
                label=f'CNN Scores (Correct Class)', color='green',
                density=True, edgecolor='black', linewidth=0.5)
    axes[0].hist(correct_decoys, bins=bins, alpha=0.5,
                label=f'Decoy Scores (Correct Class)', color='orange',
                density=True, edgecolor='black', linewidth=0.5)

    axes[0].set_xlabel('Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'Score Distribution for Correct Class - {method_name}',
                     fontsize=16, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scatter = axes[1].scatter(correct_scores, correct_decoys,
                            c=labels, cmap='tab10', alpha=0.6, s=20)

    min_val = min(np.min(correct_scores), np.min(correct_decoys))
    max_val = max(np.max(correct_scores), np.max(correct_decoys))
    axes[1].plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=2, label='y = x')

    axes[1].set_xlabel('CNN Score (Correct Class)', fontsize=12)
    axes[1].set_ylabel('Decoy Score (Correct Class)', fontsize=12)
    axes[1].set_title(f'Correct Class: CNN vs Decoy Scores - {method_name}',
                     fontsize=16, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=axes[1], label='True Label')
    plt.tight_layout()
    
    filename2 = f"{save_prefix}correct_class_comparison_{method_name.replace(' ', '_').lower()}.png"
    save_path2 = plots_dir / filename2
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig2)
    
    return save_path1, save_path2


def plot_dataset_comparison(original_cnn_scores, shifted_cnn_scores,
                           original_decoy_scores, shifted_decoy_scores,
                           num_classes=None, save_dir="plots", 
                           save_prefix="", show_plots=True):
    plots_dir = create_plots_directory(save_dir)
    
    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(original_cnn_scores.flatten(), bins=100, alpha=0.5,
                 label='Original', density=True, color='blue')
    axes[0].hist(shifted_cnn_scores.flatten(), bins=100, alpha=0.5,
                 label='Shifted', density=True, color='red')
    axes[0].set_xlabel('CNN Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('CNN Score Distributions Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(original_decoy_scores.flatten(), bins=100, alpha=0.5,
                 label='Original', density=True, color='blue')
    axes[1].hist(shifted_decoy_scores.flatten(), bins=100, alpha=0.5,
                 label='Shifted', density=True, color='red')
    axes[1].set_xlabel('Decoy Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Decoy Score Distributions Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Comparison: Original vs Shifted Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename1 = f"{save_prefix}dataset_distributions_comparison.png"
    save_path1 = plots_dir / filename1
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig1)
    
    if num_classes is not None:
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        original_means = original_cnn_scores.mean(axis=0)
        shifted_means = shifted_cnn_scores.mean(axis=0)
        
        x = np.arange(num_classes)
        width = 0.35

        ax.bar(x - width/2, original_means, width, label='Original', color='blue', alpha=0.7)
        ax.bar(x + width/2, shifted_means, width, label='Shifted', color='red', alpha=0.7)

        ax.set_xlabel('Class')
        ax.set_ylabel('Average CNN Score')
        ax.set_title('Average CNN Scores per Class: Original vs Shifted')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        
        filename2 = f"{save_prefix}class_averages_comparison.png"
        save_path2 = plots_dir / filename2
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig2)
        
        return save_path1, save_path2
    
    return save_path1, None


def plot_score_distributions_combined(model_scores_list, decoy_scores_list, 
                                      labels_list, method_names, 
                                      bins=50, num_classes=10, 
                                      save_dir="plots", save_prefix="",
                                      show_plots=True):
    plots_dir = create_plots_directory(save_dir)
    
    n_methods = len(method_names)
    fig, axes = plt.subplots(n_methods, 2, figsize=(15, 4 * n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_scores, decoy_scores, labels, method_name) in enumerate(
        zip(model_scores_list, decoy_scores_list, labels_list, method_names)):
        
        correct_scores = []
        correct_decoys = []
        
        for j in range(len(labels)):
            correct_class = labels[j]
            correct_scores.append(model_scores[j, correct_class])
            correct_decoys.append(decoy_scores[j, correct_class])
        
        correct_scores = np.array(correct_scores)
        correct_decoys = np.array(correct_decoys)
        
        axes[i, 0].hist(correct_scores, bins=bins, alpha=0.5,
                       label=f'CNN Scores', color='green',
                       density=True, edgecolor='black', linewidth=0.5)
        axes[i, 0].hist(correct_decoys, bins=bins, alpha=0.5,
                       label=f'Decoy Scores', color='orange',
                       density=True, edgecolor='black', linewidth=0.5)
        
        axes[i, 0].set_xlabel('Score')
        axes[i, 0].set_ylabel('Density')
        axes[i, 0].set_title(f'{method_name}: Score Distribution')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        scatter = axes[i, 1].scatter(correct_scores, correct_decoys,
                                    c=labels, cmap='tab10', alpha=0.6, s=10)
        
        min_val = min(np.min(correct_scores), np.min(correct_decoys))
        max_val = max(np.max(correct_scores), np.max(correct_decoys))
        axes[i, 1].plot([min_val, max_val], [min_val, max_val],
                       'k--', linewidth=1.5, label='y = x')
        
        axes[i, 1].set_xlabel('CNN Score')
        axes[i, 1].set_ylabel('Decoy Score')
        axes[i, 1].set_title(f'{method_name}: CNN vs Decoy Scores')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=axes[i, 1], label='True Label')
    
    plt.suptitle('Comparison Across Different Methods/Experiments', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f"{save_prefix}methods_comparison.png"
    save_path = plots_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    
    return save_path