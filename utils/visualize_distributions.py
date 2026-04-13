import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

print(np.__version__)
np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Axis titles
    'axes.labelsize': 14,      # Axis labels
    'xtick.labelsize': 12,     # X-axis tick labels
    'ytick.labelsize': 12,     # Y-axis tick labels
    'legend.fontsize': 10,     # Legend (increased from 10)
    'figure.titlesize': 18     # Figure title
})


def plot_score_distribution_with_decoys(train_scores, train_labels, 
                                        test_scores, test_labels, 
                                        test_decoy_scores,
                                        filename="score_distribution_decoys", 
                                        title="Score Distribution with Decoys",
                                        xlim=(-10, 10), show_kde=True, class_id=None):
    
    plt.figure(figsize=(6, 4))  # Updated figure size
    
    # Convert to numpy arrays and flatten
    train_scores = np.array(train_scores).flatten()
    train_labels = np.array(train_labels).flatten()
    test_scores = np.array(test_scores).flatten()
    test_labels = np.array(test_labels).flatten()
    test_decoy_scores = np.array(test_decoy_scores).flatten()
    
    # Separate scores by class and dataset
    train_neg = train_scores[train_labels != class_id]
    train_pos = train_scores[train_labels == class_id]
    test_neg = test_scores[test_labels != class_id]
    test_pos = test_scores[test_labels == class_id]
    
    # Определяем бины как в примере
    bin_width = 0.1
    bins = np.arange(xlim[0], xlim[1] + bin_width, bin_width)
    
    # Plot histograms with stat="density" как в примере
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
        
        kde_bw = 0.2  # Параметр bandwidth как в примере
        
        if len(train_neg) > 1:
            try:
                kde_train_neg = stats.gaussian_kde(train_neg.flatten(), bw_method=kde_bw)
                plt.plot(x_range, kde_train_neg(x_range), color='green', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(train_pos) > 1:
            try:
                kde_train_pos = stats.gaussian_kde(train_pos.flatten(), bw_method=kde_bw)
                plt.plot(x_range, kde_train_pos(x_range), color='red', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_neg) > 1:
            try:
                kde_test_neg = stats.gaussian_kde(test_neg.flatten(), bw_method=kde_bw)
                plt.plot(x_range, kde_test_neg(x_range), color='purple', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_pos) > 1:
            try:
                kde_test_pos = stats.gaussian_kde(test_pos.flatten(), bw_method=kde_bw)
                plt.plot(x_range, kde_test_pos(x_range), color='orange', 
                        linewidth=2, linestyle='-')
            except:
                pass
        
        if len(test_decoy_scores) > 1:
            try:
                kde_decoys = stats.gaussian_kde(test_decoy_scores.flatten(), bw_method=kde_bw)
                plt.plot(x_range, kde_decoys(x_range), color='dodgerblue', 
                        linewidth=2, linestyle='-')
            except:
                pass
    
    plt.xlabel('Discriminative scores')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Используем xlim для ограничения диапазона
    if xlim is not None:
        plt.xlim(xlim)
    
    plt.title(title)
    plt.show()
    plt.savefig(filename)