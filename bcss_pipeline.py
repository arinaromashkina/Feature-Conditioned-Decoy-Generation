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

from data_processing.score_feature_dataset import ScoreFeatureDataset, create_score_feature_dataset, create_score_feature_dataset_bcss
from data_processing.negative_scores_pool import collect_negative_scores
from flows.separate_flows import SeparateClassFlows

DEVICE = 'cuda'
NUM_CLASSES = 5
PATH_DATA = '../../../../blob/BCSS/training/bcss.medium.training.torch'


data = torch.load(PATH_DATA)
train_ds = create_score_feature_dataset_bcss(data, DEVICE)
print('Train loaded')

print("Training separate flows for each class...")
separate_flows = SeparateClassFlows(num_classes=NUM_CLASSES, n_flows=15, feature_dim=64, hidden_dim=64).to(DEVICE)
# separate_flows = separate_flows.train_separate(train_ds, epochs=8, lr=1e-3, device=DEVICE)
# torch.save(separate_flows.state_dict(), 'BCSS/bcss_cond_flows_model.pth')
separate_flows.load_state_dict(torch.load('BCSS/bcss_cond_flows_model.pth'))


test_file_name = '../../../../blob/BCSS/test/TCGA-S3-AA10-DX1_xmin43039_ymin23986_MPP-0.2500.png.tensor'
test_data = torch.load(test_file_name, weights_only=False)
features = test_data['features']
predictions = test_data['predictions']
labels = torch.tensor(test_data['mask'])
predictions = torch.flatten(predictions, start_dim=2).squeeze(0).T    
features = torch.flatten(features, start_dim=2).squeeze(0).T    
labels = torch.flatten(labels, start_dim=0)
test_ds = ScoreFeatureDataset(predictions, features, predictions, labels)


scores_flow, decoys_flow, labels_flow = separate_flows.generate_decoys(test_ds, device='cuda')
torch.save(scores_flow, 'scores_flow.pt')
torch.save(decoys_flow, 'decoys_flow.pt')
torch.save(labels_flow, 'labels_flow.pt')
scores_flow = torch.load('scores_flow.pt')
decoys_flow = torch.load('decoys_flow.pt')
labels_flow = torch.load('labels_flow.pt')
labels_flow = np.where(
    labels_flow <= 3,
    labels_flow,           # оставляем 0,1,2,3 как есть
    torch.tensor(4)   # все что >3 становится 4
)
print('Test loaded')

def get_scores_from_ds(score_dataset, device='cuda'):
    test_cnn_scores = []
    test_labels = []
    test_loader = DataLoader(score_dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for cnn_scores, features, target_decoy, labels in test_loader:
            features = features.to(DEVICE)

            test_cnn_scores.append(cnn_scores.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_cnn_scores = np.concatenate(test_cnn_scores, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    return test_cnn_scores, test_labels


train_scores, train_labels = get_scores_from_ds(train_ds)
print('got flat train scores!')


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Print numpy version
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
                                        bins=150, xlim=None, show_kde=True, class_id=None):
   
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
    
    # Plot histograms with visible edges
    plt.hist(train_neg, bins=bins, density=False, color='green', 
             alpha=0.3, label='Training -', edgecolor='green', linewidth=0.5)
    plt.hist(train_pos, bins=bins, density=False, color='red', 
             alpha=0.3, label='Training +', edgecolor='red', linewidth=0.5)
    plt.hist(test_neg, bins=bins, density=False, color='purple', 
             alpha=0.3, label='Test -', edgecolor='purple', linewidth=0.5)
    plt.hist(test_pos, bins=bins, density=False, color='orange', 
             alpha=0.3, label='Test +', edgecolor='orange', linewidth=0.5)
    plt.hist(test_decoy_scores, bins=bins, density=False, color='dodgerblue', 
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
    plt.savefig('BCSS/' + filename + ".png", bbox_inches='tight', dpi=300)
    plt.savefig('BCSS/' + filename + ".pdf", bbox_inches='tight')
    plt.show()


def plot_score_distribution_with_decoys(train_scores, train_labels, 
                                        test_scores, test_labels, 
                                        test_decoy_scores,
                                        filename="score_distribution_decoys", 
                                        title="Score Distribution with Decoys",
                                        xlim=(-10, 10), show_kde=False, class_id=None):
    
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
    
    # Save in both formats
    plt.savefig('BCSS/' + filename + ".png", bbox_inches='tight', dpi=300)
    plt.savefig('BCSS/' + filename + ".pdf", bbox_inches='tight')
    plt.show()


for i in range(NUM_CLASSES):
    plot_score_distribution_with_decoys(train_scores[:, i], train_labels, 
                                        scores_flow[:, i], labels_flow, 
                                        decoys_flow[:, i],
                                        filename=f"score_distribution_decoys_{i}", 
                                        title=f"Score Distribution with Decoys for class {i}",
                                        xlim=(-10, 10), show_kde=True, class_id=i)



from fdr.fdr_control import control_fdr_multiclass
from fdr.plot_fdr import plot_fdr_multiclass

print('labels_flow', np.min(labels_flow), np.max(labels_flow))
print(np.unique(labels_flow))
for value in [0, 1, 2, 3, 4]:
    print(value, len(labels_flow[labels_flow == value]), len(labels_flow[labels_flow == value]) / len(labels_flow))
print('train_labels', np.min(train_labels), np.max(train_labels))

print('start df flow')
df, pi0_overall, pi0_estimates = control_fdr_multiclass(scores_flow, labels_flow, decoys_flow,
                           train_cnn_scores=train_scores, train_labels=train_labels,
                           test_cnn_scores=scores_flow, num_classes=5)
print('finish df flow')
print('pi0_overall', pi0_overall)
print('pi0_estimates', pi0_estimates)
plot_fdr_multiclass(df, filename="fdr_multiclass")
print('end')