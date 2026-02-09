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

print("Training separate flows for each class...")
separate_flows = SeparateClassFlows(num_classes=NUM_CLASSES, n_flows=8, feature_dim=64, hidden_dim=64).to(DEVICE)
separate_flows = separate_flows.train_separate(train_ds, epochs=5, lr=1e-3, device=DEVICE)
torch.save(separate_flows.state_dict(), 'BCSS/bcss_cond_flows_model.pth')

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