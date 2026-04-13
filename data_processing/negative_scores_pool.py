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
from tqdm import tqdm
# ─────────────────────────────────────────────────────────────────────────────
# Updated dataset creation — null = full score vector with true class replaced
# ─────────────────────────────────────────────────────────────────────────────

def collect_negative_scores(model, train_dataset, num_classes=10, device='cuda'):
    """Collect negative scores for each class (unchanged)."""
    negative_scores_pools = {i: [] for i in range(num_classes)}
    model.eval()
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting negative scores'):
            images, labels = images.to(device), labels.to(device)
            features = model.get_features(images)
            #scores   = model.linear2(F.relu(model.linear1(features)))
            scores = model.linear2(features)

            for class_idx in range(num_classes):
                neg_mask = labels != class_idx
                if neg_mask.sum() > 0:
                    negative_scores_pools[class_idx].append(
                        scores[neg_mask, class_idx].cpu())

    for class_idx in range(num_classes):
        if negative_scores_pools[class_idx]:
            negative_scores_pools[class_idx] = \
                torch.cat(negative_scores_pools[class_idx]).numpy()
        else:
            negative_scores_pools[class_idx] = np.array([])

    return negative_scores_pools

print("✓ ScoreShiftFlow defined — single flow over full score vector")

# def collect_negative_scores(model, train_dataset, bool_multiclass=True, num_classes=10, device='cuda'):
#     if bool_multiclass:
#         negative_scores_pools = {i: [] for i in range(num_classes)}
#     else:
#         negative_scores_pools = []
#     model.eval()
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
#     with torch.no_grad():
#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)
#             features = model.get_features(images)
#             scores = model.fc2(features)
#             if bool_multiclass:
#                 for class_idx in range(num_classes):
#                     neg_mask = labels != class_idx
#                     if neg_mask.sum() > 0:
#                         class_scores = scores[neg_mask, class_idx]
#                         negative_scores_pools[class_idx].append(class_scores.cpu())
#             else:
#                 neg_mask = labels != 1
#                 if neg_mask.sum() > 0:
#                         class_scores = scores[neg_mask]
#                         negative_scores_pools.append(class_scores.cpu())
#     if bool_multiclass:
#         for class_idx in range(num_classes):
#             if negative_scores_pools[class_idx]:
#                 negative_scores_pools[class_idx] = torch.cat(negative_scores_pools[class_idx]).numpy()
#                 print(f"Class {class_idx}: collected {len(negative_scores_pools[class_idx])} negative scores")
#             else:
#                 negative_scores_pools[class_idx] = np.array([])
#                 print(f"Class {class_idx}: no negative scores collected")
#     return negative_scores_pools