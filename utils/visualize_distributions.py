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


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Print numpy version
print(np.__version__)
np.set_printoptions(threshold=np.inf)

# Set global font sizes for publication quality
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Axis titles
    'axes.labelsize': 14,      # Axis labels
    'xtick.labelsize': 12,     # X-axis tick labels
    'ytick.labelsize': 12,     # Y-axis tick labels
    'legend.fontsize': 10,     # Legend (increased from 10)
    'figure.titlesize': 18     # Figure title
})

# Define global path for saving figures
PATH = './figures/'
Path(PATH).mkdir(exist_ok=True)


def plot_score_distribution(train_scores, train_labels, test_scores, test_labels, 
                            filename="score_distribution", title="Score Distribution Comparison",
                            bins=50, xlim=None, show_kde=True):
    """
    Publication-quality visualization of score distributions for train and test sets.
    
    Parameters:
    -----------
    train_scores : array-like
        Training set scores
    train_labels : array-like
        Training set labels (0 for negative, 1 for positive)
    test_scores : array-like
        Test set scores
    test_labels : array-like
        Test set labels (0 for negative, 1 for positive)
    filename : str
        Name for saved files (without extension)
    title : str
        Plot title
    bins : int or array-like
        Number of bins or bin edges for histograms
    xlim : tuple or None
        X-axis limits (min, max)
    show_kde : bool
        Whether to show KDE curves
    """
    plt.figure(figsize=(6, 4))  # Updated figure size
    
    # Convert to numpy arrays and flatten
    train_scores = np.array(train_scores).flatten()
    train_labels = np.array(train_labels).flatten()
    test_scores = np.array(test_scores).flatten()
    test_labels = np.array(test_labels).flatten()
    
    # Separate scores by class and dataset
    train_neg = train_scores[train_labels == 0]
    train_pos = train_scores[train_labels == 1]
    test_neg = test_scores[test_labels == 0]
    test_pos = test_scores[test_labels == 1]
    
    # Plot histograms with visible edges
    plt.hist(train_neg, bins=bins, density=True, color='blue', 
             alpha=0.3, label='Train Negative', edgecolor='blue', linewidth=0.5)
    plt.hist(train_pos, bins=bins, density=True, color='red', 
             alpha=0.3, label='Train Positive', edgecolor='red', linewidth=0.5)
    plt.hist(test_neg, bins=bins, density=True, color='green', 
             alpha=0.3, label='Test Negative', edgecolor='green', linewidth=0.5)
    plt.hist(test_pos, bins=bins, density=True, color='orange', 
             alpha=0.3, label='Test Positive', edgecolor='orange', linewidth=0.5)
    
    # Add KDE curves if needed
    if show_kde:
        all_scores = np.concatenate([train_scores, test_scores])
        x_range = np.linspace(all_scores.min(), all_scores.max(), 200)
        
        if len(train_neg) > 1:
            try:
                kde_train_neg = stats.gaussian_kde(train_neg.flatten())
                plt.plot(x_range, kde_train_neg(x_range), color='blue', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(train_pos) > 1:
            try:
                kde_train_pos = stats.gaussian_kde(train_pos.flatten())
                plt.plot(x_range, kde_train_pos(x_range), color='red', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_neg) > 1:
            try:
                kde_test_neg = stats.gaussian_kde(test_neg.flatten())
                plt.plot(x_range, kde_test_neg(x_range), color='green', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_pos) > 1:
            try:
                kde_test_pos = stats.gaussian_kde(test_pos.flatten())
                plt.plot(x_range, kde_test_pos(x_range), color='orange', 
                        linewidth=2, linestyle='-')
            except:
                pass
    
    plt.xlabel('Scores')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if xlim is not None:
        plt.xlim(xlim)
    
    # Save in both formats
    plt.savefig(PATH + filename + ".png", bbox_inches='tight', dpi=300)
    plt.savefig(PATH + filename + ".pdf", bbox_inches='tight')
    plt.show()


def plot_score_distribution_with_decoys(train_scores, train_labels, 
                                        test_scores, test_labels, 
                                        test_decoy_scores,
                                        filename="score_distribution_decoys", 
                                        title="Score Distribution with Decoys",
                                        bins=50, xlim=None, show_kde=True):
    """
    Publication-quality visualization including decoy scores for test set.
    
    Parameters:
    -----------
    train_scores : array-like
        Training set scores
    train_labels : array-like
        Training set labels (0 for negative, 1 for positive)
    test_scores : array-like
        Test set scores
    test_labels : array-like
        Test set labels (0 for negative, 1 for positive)
    test_decoy_scores : array-like
        Decoy scores for test set
    filename : str
        Name for saved files (without extension)
    title : str
        Plot title
    bins : int or array-like
        Number of bins or bin edges for histograms
    xlim : tuple or None
        X-axis limits (min, max)
    show_kde : bool
        Whether to show KDE curves
    """
    plt.figure(figsize=(6, 4))  # Updated figure size
    
    # Convert to numpy arrays and flatten
    train_scores = np.array(train_scores).flatten()
    train_labels = np.array(train_labels).flatten()
    test_scores = np.array(test_scores).flatten()
    test_labels = np.array(test_labels).flatten()
    test_decoy_scores = np.array(test_decoy_scores).flatten()
    
    # Separate scores by class and dataset
    train_neg = train_scores[train_labels == 0]
    train_pos = train_scores[train_labels == 1]
    test_neg = test_scores[test_labels == 0]
    test_pos = test_scores[test_labels == 1]
    
    # Plot histograms with visible edges
    plt.hist(train_neg, bins=bins, density=True, color='green', 
             alpha=0.3, label='Training -', edgecolor='green', linewidth=0.5)
    plt.hist(train_pos, bins=bins, density=True, color='red', 
             alpha=0.3, label='Training +', edgecolor='red', linewidth=0.5)
    plt.hist(test_neg, bins=bins, density=True, color='purple', 
             alpha=0.3, label='Test -', edgecolor='purple', linewidth=0.5)
    plt.hist(test_pos, bins=bins, density=True, color='orange', 
             alpha=0.3, label='Test +', edgecolor='orange', linewidth=0.5)
    plt.hist(test_decoy_scores, bins=bins, density=True, color='dodgerblue', 
             alpha=0.3, label='Test decoys', edgecolor='dodgerblue', linewidth=0.5)
    
    # Add KDE curves if needed
    if show_kde:
        all_scores = np.concatenate([train_scores, test_scores, test_decoy_scores])
        x_range = np.linspace(all_scores.min(), all_scores.max(), 200)
        
        if len(train_neg) > 1:
            try:
                kde_train_neg = stats.gaussian_kde(train_neg.flatten())
                plt.plot(x_range, kde_train_neg(x_range), color='green', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(train_pos) > 1:
            try:
                kde_train_pos = stats.gaussian_kde(train_pos.flatten())
                plt.plot(x_range, kde_train_pos(x_range), color='red', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_neg) > 1:
            try:
                kde_test_neg = stats.gaussian_kde(test_neg.flatten())
                plt.plot(x_range, kde_test_neg(x_range), color='purple', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_pos) > 1:
            try:
                kde_test_pos = stats.gaussian_kde(test_pos.flatten())
                plt.plot(x_range, kde_test_pos(x_range), color='orange', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_decoy_scores) > 1:
            try:
                kde_decoys = stats.gaussian_kde(test_decoy_scores.flatten())
                plt.plot(x_range, kde_decoys(x_range), color='dodgerblue', 
                        linewidth=2, linestyle='-')
            except:
                pass
    
    plt.xlabel('Discriminative scores')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if xlim is not None:
        plt.xlim(xlim)
    
    # Save in both formats
    plt.savefig(PATH + filename + ".png", bbox_inches='tight', dpi=300)
    plt.savefig(PATH + filename + ".pdf", bbox_inches='tight')
    plt.show()