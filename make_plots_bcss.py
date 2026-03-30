import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr
import os
from bisect import bisect
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import glob
import pickle

# Print versions
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Set numpy print options
np.set_printoptions(threshold=np.inf)

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
print(f"Device: {DEVICE}")

# Create output directories
os.makedirs('figures/accuracy', exist_ok=True)
os.makedirs('figures/fdr_curves', exist_ok=True)
os.makedirs('figures/mano', exist_ok=True)
os.makedirs('figures/comparison', exist_ok=True)
print("✓ Setup complete")

from data_processing.score_feature_dataset import ScoreFeatureDataset, create_score_feature_dataset, create_score_feature_dataset_bcss
from data_processing.negative_scores_pool import collect_negative_scores
from flows.separate_flows import SeparateClassFlows


def negentropy(logits):
    """Compute negentropy (negative entropy)."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    max_entropy = np.log(logits.shape[1])
    return max_entropy - entropy


def calibration_temp(logits, labels, num_bins=15):
    """Find optimal temperature for calibration."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()
    
    temps = torch.linspace(0.1, 5.0, 50)
    best_temp = 1.0
    best_ece = float('inf')
    
    for temp in temps:
        scaled_probs = torch.softmax(logits / temp, dim=1)
        confidences, predictions = scaled_probs.max(dim=1)
        accuracies = (predictions == labels).float()
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        for i in range(num_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = accuracies[mask].mean()
                ece += mask.float().mean() * torch.abs(bin_conf - bin_acc)
        
        if ece < best_ece:
            best_ece = ece
            best_temp = temp.item()
    
    return best_temp


def _to_tensor(x):
    """Convert numpy array or tensor to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x


def predict_ATC_maxconf(source_logits, source_labels, target_logits):
    """Average Threshold Confidence with max confidence - SOFTMAX NORMALIZED."""
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    
    # Apply softmax normalization
    source_probs = torch.softmax(source_logits, dim=1)
    target_probs = torch.softmax(target_logits, dim=1)
    
    source_scores = source_probs.amax(1)
    target_scores = target_probs.amax(1)
    
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_probs.argmax(1) == source_labels).sum()]
    
    estimate = (target_scores > threshold).float().mean().item()
    return np.clip(estimate, 0.0, 1.0)


def predict_ATC_negent(source_logits, source_labels, target_logits):
    """Average Threshold Confidence with negentropy."""
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    
    source_scores = negentropy(source_logits)
    target_scores = negentropy(target_logits)
    
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_logits.argmax(1) == source_labels).sum()]
    
    estimate = (target_scores > threshold).float().mean().item()
    return np.clip(estimate, 0.0, 1.0)


def predict_AC(source_logits, source_labels, target_logits):
    """Average Confidence - SOFTMAX NORMALIZED."""
    target_logits = _to_tensor(target_logits)
    target_probs = torch.softmax(target_logits, dim=1)
    return np.clip(target_probs.amax(1).mean().item(), 0.0, 1.0)


def predict_DOC(source_logits, source_labels, target_logits):
    """Difference of Confidences - SOFTMAX NORMALIZED."""
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    
    # Apply softmax normalization
    source_probs = torch.softmax(source_logits, dim=1)
    target_probs = torch.softmax(target_logits, dim=1)
    
    avg_source_conf = source_probs.amax(1).mean().item()
    avg_target_conf = target_probs.amax(1).mean().item()
    
    source_acc = (source_probs.argmax(1) == source_labels).float().mean().item()
    
    estimate = source_acc + (avg_target_conf - avg_source_conf)
    return np.clip(estimate, 0.0, 1.0)


try:
    import ot
    
    def predict_COT(source_logits, source_labels, target_logits):
        """Confidence Optimal Transport - WORKS WITH PROBABILITY SIMPLEXES."""
        source_logits = _to_tensor(source_logits)
        source_labels = _to_tensor(source_labels).long()
        target_logits = _to_tensor(target_logits)
        
        num_classes = source_logits.shape[1]
        
        # Apply softmax normalization to get probability simplexes
        source_probs = torch.softmax(source_logits, dim=1)
        target_probs = torch.softmax(target_logits, dim=1)
        
        # Source label distribution (empirical class distribution)
        source_label_dist = torch.nn.functional.one_hot(source_labels, num_classes).float().mean(0)
        
        # Cost matrix: L1 distance from each target sample to each class simplex vertex
        # For each target sample, compute distance to each one-hot class vector
        # Shape: [n_target, num_classes]
        eye_matrix = torch.eye(num_classes, device=target_logits.device)
        cost_matrix = torch.cdist(target_probs, eye_matrix, p=1)  # L1 distance
        
        # Normalize target distribution (uniform over samples)
        n_target = len(target_probs)
        target_dist = np.ones(n_target) / n_target
        
        # Source distribution is the empirical label distribution
        source_dist = source_label_dist.cpu().numpy()
        cost_matrix_np = cost_matrix.cpu().numpy()
        
        # Ensure distributions sum to 1
        target_dist = target_dist / target_dist.sum()
        source_dist = source_dist / source_dist.sum()
        
        # Solve optimal transport
        ot_plan = ot.emd(target_dist, source_dist, cost_matrix_np)
        ot_cost = np.sum(ot_plan * cost_matrix_np)
        
        # Source statistics (on probabilities)
        s_conf = source_probs.amax(1).mean().item()
        s_acc = (source_probs.argmax(1) == source_labels).float().mean().item()
        
        # Confidence gap
        conf_gap = s_conf - s_acc
        
        # Error estimate
        err_est = ot_cost + conf_gap
        
        estimate = 1. - err_est
        return np.clip(estimate, 0.0, 1.0)
    
    COT_AVAILABLE = True
    print("✓ COT available")
except ImportError:
    COT_AVAILABLE = False
    print("⚠ COT unavailable (install: pip install POT)")

BASELINE_METHODS = {
    'ATC': predict_ATC_maxconf,
    'ATC-NE': predict_ATC_negent,
    'AC': predict_AC,
    'DOC': predict_DOC,
}
if COT_AVAILABLE:
    BASELINE_METHODS['COT'] = predict_COT

print(f"✓ Baseline methods: {list(BASELINE_METHODS.keys())}")


def empirical_p_values(distribution, query):
    """Calculate empirical p-values from decoy distribution."""
    dist_len = len(distribution)
    p_values = np.zeros(len(query))
    sorted_dist = np.sort(distribution)
    for i, score in enumerate(query):
        p_values[i] = (dist_len - bisect(sorted_dist, score)) / dist_len
    return p_values


def estimate_pi0_storey(p_values, lambda_range=np.arange(0.05, 0.95, 0.05)):
    """Estimate proportion of nulls using Storey's method (LABEL-FREE)."""
    pi0_estimates = [np.mean(p_values > lam) / (1 - lam) for lam in lambda_range]
    pi0 = min(1.0, max(0, np.mean(pi0_estimates)))
    return pi0


def benjamini_hochberg(p_values, pi0=1.0):
    """Standard Benjamini-Hochberg procedure with π₀ adjustment - OPTIMIZED."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Vectorized q-value computation
    ranks = np.arange(1, n + 1)
    q_values = np.minimum(1.0, (n * sorted_p * pi0) / ranks)
    
    # Enforce monotonicity (backwards cumulative minimum)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    
    # Map back to original order
    original_q = np.zeros(n)
    original_q[sorted_idx] = q_values
    
    return original_q


def calculate_fdr2_qvalues(target_scores, decoy_scores):
    """Calculate FDR2 q-values using TDC (LABEL-FREE) - OPTIMIZED."""
    n = len(target_scores)
    combined_scores = np.maximum(target_scores, decoy_scores)
    is_target_win = (target_scores > decoy_scores).astype(int)
    
    sort_idx = np.argsort(combined_scores)[::-1]
    sorted_wins = is_target_win[sort_idx]
    
    # Vectorized FDR2 calculation using cumsum
    n_decoy_wins_cumsum = np.cumsum(1 - sorted_wins)
    ranks = np.arange(1, n + 1)
    fdr2_values = (2 * n_decoy_wins_cumsum) / ranks
    
    # Enforce monotonicity
    q_fdr2 = np.minimum.accumulate(fdr2_values[::-1])[::-1]
    
    # Map back to original order
    final_q_fdr2 = np.zeros(n)
    final_q_fdr2[sort_idx] = q_fdr2
    
    return final_q_fdr2


def control_fdr_mixmax(model_scores, target_labels, decoy_scores):
    """Mix-Max FDR control with FDR2, B-H, and Ground Truth."""
    n_samples = len(model_scores)
    
    df = pd.DataFrame({
        'original_index': np.arange(n_samples),
        'label': target_labels,
        'max_model_score': model_scores.max(axis=1),
        'max_decoy_score': decoy_scores.max(axis=1),
        'predicted_class': model_scores.argmax(axis=1)
    })
    
    df['max_score'] = df[['max_model_score', 'max_decoy_score']].max(axis=1)
    
    # Ground Truth FDR - OPTIMIZED
    df_sorted_gt = df.sort_values(by='max_model_score', ascending=False).reset_index(drop=True)
    is_incorrect = (df_sorted_gt['predicted_class'] != df_sorted_gt['label']).astype(int)
    num_incorrect_cumsum = np.cumsum(is_incorrect)
    ranks = np.arange(1, len(df_sorted_gt) + 1)
    fdr_gt_list = num_incorrect_cumsum / ranks
    
    # Enforce monotonicity
    q_values_gt = np.minimum.accumulate(fdr_gt_list[::-1])[::-1]
    
    df_sorted_gt['q_values_ground_truth'] = q_values_gt
    gt_mapping = dict(zip(df_sorted_gt['original_index'], df_sorted_gt['q_values_ground_truth']))
    df['q_values_ground_truth'] = df['original_index'].map(gt_mapping)
    
    # FDR2 (TDC)
    target_scores = df['max_model_score'].values
    decoy_scores_vals = df['max_decoy_score'].values
    q_fdr2 = calculate_fdr2_qvalues(target_scores, decoy_scores_vals)
    df['q_values_fdr2'] = q_fdr2
    
    # Benjamini-Hochberg with Storey π₀
    p_values = empirical_p_values(decoy_scores_vals, target_scores)
    pi0 = estimate_pi0_storey(p_values)
    q_bh = benjamini_hochberg(p_values, pi0=pi0)
    
    df['q_values_bh'] = q_bh
    df['p_values'] = p_values
    df['pi0_storey'] = pi0
    
    return df


def control_fdr_binary_then_combine(model_scores, target_labels, decoy_scores, 
                                      method='min', use_bh=True, verbose=False):
    """Binary-then-Combine multi-class FDR control (LABEL-FREE) - OPTIMIZED."""
    n_samples = len(model_scores)
    num_classes = model_scores.shape[1]
    
    # Preallocate array for q-values
    all_q_values = np.ones((n_samples, num_classes))
    
    # Store diagnostics
    pi0_estimates = []
    pi0_true = []
    
    # Step 1: Binary FDR control for each class - VECTORIZED
    if verbose:
        print("\nComputing binary FDR control for each class...")
    
    for class_k in range(num_classes):
        model_scores_k = model_scores[:, class_k]
        decoy_scores_k = decoy_scores[:, class_k]
        
        if use_bh:
            p_values_k = empirical_p_values(decoy_scores_k, model_scores_k)
            pi0_k = estimate_pi0_storey(p_values_k)
            q_values_k = benjamini_hochberg(p_values_k, pi0=pi0_k)
        else:
            q_values_k = calculate_fdr2_qvalues(model_scores_k, decoy_scores_k)
            pi0_k = np.nan
        
        all_q_values[:, class_k] = q_values_k
        
        # Calculate TRUE pi0 for this class
        predicted_class_k = model_scores.argmax(axis=1)
        samples_predicted_as_k = (predicted_class_k == class_k)
        if samples_predicted_as_k.sum() > 0:
            true_pi0_k = 1 - (target_labels[samples_predicted_as_k] == class_k).mean()
        else:
            true_pi0_k = np.nan
        
        pi0_estimates.append(pi0_k)
        pi0_true.append(true_pi0_k)
        
        if verbose:
            print(f"  Class {class_k}: π₀_estimated={pi0_k:.3f}, π₀_true={true_pi0_k:.3f}, "
                  f"n_predicted={samples_predicted_as_k.sum()}")
    
    # Step 2: Combine q-values - VECTORIZED
    predicted_classes = model_scores.argmax(axis=1)
    
    if method == 'predicted':
        # Take q-value from predicted class for each sample
        combined_q_values = all_q_values[np.arange(n_samples), predicted_classes]
    elif method == 'min':
        # Take minimum q-value across all classes for each sample
        combined_q_values = all_q_values.min(axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 3: Create final DataFrame
    df_final = pd.DataFrame({
        'index': np.arange(n_samples),
        'label': target_labels,
        'predicted_class': predicted_classes,
        'q_values_binary_combined': combined_q_values,
        'max_model_score': model_scores.max(axis=1)
    })
    
    # Add ground truth - OPTIMIZED
    df_final_sorted = df_final.sort_values('max_model_score', ascending=False).reset_index(drop=True)
    is_incorrect = (df_final_sorted['predicted_class'] != df_final_sorted['label']).astype(int)
    num_incorrect_cumsum = np.cumsum(is_incorrect)
    ranks = np.arange(1, len(df_final_sorted) + 1)
    fdr_gt = num_incorrect_cumsum / ranks
    
    # Enforce monotonicity
    q_gt = np.minimum.accumulate(fdr_gt[::-1])[::-1]
    
    df_final_sorted['q_values_ground_truth'] = q_gt
    gt_mapping = dict(zip(df_final_sorted['index'], df_final_sorted['q_values_ground_truth']))
    df_final['q_values_ground_truth'] = df_final['index'].map(gt_mapping)
    
    # Return diagnostics
    diagnostics = {
        'pi0_estimated': pi0_estimates,
        'pi0_true': pi0_true,
        'pi0_error': [abs(est - true) if not np.isnan(est) and not np.isnan(true) else np.nan 
                      for est, true in zip(pi0_estimates, pi0_true)]
    }
    
    return df_final, diagnostics


print("✓ FDR control functions loaded (OPTIMIZED)")


def estimate_pi0_storey_from_df(df, q_value_column):
    """Estimate π₀ from DataFrame (LABEL-FREE) with better fallback."""
    if 'p_values' in df.columns:
        p_values = df['p_values'].dropna().values
        if len(p_values) > 0:
            pi0 = estimate_pi0_storey(p_values)
            return pi0
    
    if 'pi0_storey' in df.columns:
        pi0 = df['pi0_storey'].iloc[0]
        if not np.isnan(pi0):
            return pi0
    
    # Improved fallback: use top 20% predictions
    score_column = 'max_model_score' if 'max_model_score' in df.columns else 'model_score'
    if score_column not in df.columns:
        return 0.5
    
    df_sorted = df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    top_k = max(100, int(0.2 * len(df_sorted)))
    top_samples = df_sorted.head(top_k)
    
    is_correct = (top_samples['predicted_class'] == top_samples['label']).astype(int)
    pi0_est = 1 - is_correct.mean()
    
    # Clip to reasonable range
    return np.clip(pi0_est, 0.05, 0.95)


def compute_method_estimation_curve(df, q_value_column, pi0):
    """Compute LABEL-FREE estimation curve with clipping to [0, 1]."""
    total_samples = len(df)
    df_method = df[~df[q_value_column].isna()].copy()
    
    if len(df_method) == 0:
        return pd.DataFrame()
    
    df_method = df_method.sort_values(by=q_value_column, ascending=True).reset_index(drop=True)
    df_method['n_discoveries'] = np.arange(1, len(df_method) + 1)
    
    # Label-free estimation
    df_method['FP_est'] = df_method['n_discoveries'] * df_method[q_value_column]
    df_method['TP_est'] = df_method['n_discoveries'] - df_method['FP_est']
    df_method['TN_est'] = total_samples * pi0 - df_method['FP_est']
    df_method['FN_est'] = total_samples * (1 - pi0) - df_method['TP_est']
    
    # Clip to non-negative
    df_method['FP_est'] = np.maximum(0, df_method['FP_est'])
    df_method['TP_est'] = np.maximum(0, df_method['TP_est'])
    df_method['TN_est'] = np.maximum(0, df_method['TN_est'])
    df_method['FN_est'] = np.maximum(0, df_method['FN_est'])
    
    # CLIP ACCURACY TO [0, 1]
    df_method['Accuracy_est'] = np.clip(
        (df_method['TP_est'] + df_method['TN_est']) / total_samples,
        0.0,
        1.0
    )
    
    # True metrics (for comparison)
    df_method['is_correct'] = (df_method['predicted_class'] == df_method['label']).astype(int)
    df_method['TP_true'] = df_method['is_correct'].cumsum()
    df_method['FP_true'] = df_method['n_discoveries'] - df_method['TP_true']
    
    total_correct = df_method['is_correct'].sum()
    total_incorrect = len(df_method) - total_correct
    
    df_method['TN_true'] = total_incorrect - df_method['FP_true']
    df_method['FN_true'] = total_correct - df_method['TP_true']
    df_method['Accuracy_true'] = (df_method['TP_true'] + df_method['TN_true']) / total_samples
    
    return df_method[[q_value_column, 'Accuracy_est', 'Accuracy_true',
                      'FP_est', 'TP_est', 'FN_est', 'TN_est',
                      'FP_true', 'TP_true', 'FN_true', 'TN_true']].rename(
        columns={q_value_column: 'q_value_method'})


def compute_best_gt_accuracy(model_scores, labels):
    """
    Oracle: best possible accuracy by thresholding max confidence - SOFTMAX NORMALIZED.
    """
    # Apply softmax normalization
    model_probs = np.exp(model_scores) / np.exp(model_scores).sum(axis=1, keepdims=True)
    
    max_scores = model_probs.max(axis=1)
    preds = model_probs.argmax(axis=1)
    correct = (preds == labels).astype(int)
    
    # Sort by confidence (descending)
    order = np.argsort(-max_scores)
    correct_sorted = correct[order]
    
    n = len(labels)
    total_correct = correct.sum()
    total_incorrect = n - total_correct
    
    # Vectorized computation
    tp = np.cumsum(correct_sorted)
    fp = np.arange(1, n + 1) - tp
    tn = total_incorrect - fp
    fn = total_correct - tp
    
    acc = (tp + tn) / n
    
    return acc.max()


def get_scores_from_ds(score_dataset, device=DEVICE):
    """Extract scores and labels from dataset."""
    test_cnn_scores = []
    test_labels = []
    test_loader = DataLoader(score_dataset, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for cnn_scores, features, target_decoy, labels in test_loader:
            test_cnn_scores.append(cnn_scores.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
    
    test_cnn_scores = np.concatenate(test_cnn_scores, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    return test_cnn_scores, test_labels


print("✓ All helper functions loaded (OPTIMIZED)")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

PATH_DATA = '../../data/BCSS/training/bcss.mini.training.torch'
data = torch.load(PATH_DATA)
train_ds = create_score_feature_dataset_bcss(data, DEVICE)
print('✓ Train loaded')

train_scores, train_labels = get_scores_from_ds(train_ds)
print(f'  Class 0: {len(train_labels[train_labels==0])} samples')
print(f'  Class 1: {len(train_labels[train_labels==1])} samples')
print(f'  Class 2: {len(train_labels[train_labels==2])} samples')
print(f'  Class 3: {len(train_labels[train_labels==3])} samples')
print(f'  Class 4: {len(train_labels[train_labels==4])} samples')

print("\n" + "="*80)
print("LOADING FLOW MODEL")
print("="*80)

separate_flows = SeparateClassFlows(
    num_classes=NUM_CLASSES, 
    n_flows=11, 
    feature_dim=64, 
    hidden_dim=64
).to(DEVICE)
separate_flows.load_state_dict(torch.load('BCSS/bcss_cond_flows_model_mini.pth'))
separate_flows = separate_flows.to(DEVICE)
print('✓ Flow model uploaded')

print("\n" + "="*80)
print("PREPARING SOURCE DATA (40% SUBSAMPLE)")
print("="*80)

train_size = len(train_ds)
subset_size = int(0.4 * train_size)
np.random.seed(42)
indices = np.random.choice(train_size, subset_size, replace=False)
subset_ds = Subset(train_ds, indices)

source_logits, source_decoy_scores, source_labels = separate_flows.generate_decoys(
    subset_ds, device=DEVICE
)
print(f'✓ Source data: {len(source_labels)} samples')

# Calibration temperature
temp = calibration_temp(source_logits, source_labels)
print(f'✓ Calibration temperature: {temp:.3f}')

print("\n" + "="*80)
print("PROCESSING TEST FILES")
print("="*80)

test_folder = '../../data/BCSS/test/'
test_files = glob.glob(test_folder + '*.tensor')
print(f'Found {len(test_files)} test files')

results = []
pi0_diagnostics_all = []

for test_file_name in tqdm(test_files, desc="Processing files"):
    NAME = os.path.basename(test_file_name)
    
    try:
        test_data = torch.load(test_file_name, weights_only=False)
    except (EOFError, pickle.UnpicklingError, FileNotFoundError, RuntimeError) as e:
        print(f"⚠️ Skipping corrupted file {NAME}: {e}")
        continue
    except Exception as e:
        print(f"⚠️ Unexpected error with {NAME}: {e}")
        continue
    
    features = test_data['features']
    predictions = test_data['predictions']
    labels = torch.tensor(test_data['mask'])
    
    predictions = torch.flatten(predictions, start_dim=2).squeeze(0).T    
    features = torch.flatten(features, start_dim=2).squeeze(0).T    
    labels = torch.flatten(labels, start_dim=0)
    
    test_ds = ScoreFeatureDataset(predictions, features, predictions, labels)
    
    test_scores, test_decoys, test_labels = separate_flows.generate_decoys(
        test_ds, device=DEVICE
    )
    
    # Subsample if needed
    test_scores = test_scores[::10]
    test_decoys = test_decoys[::10]
    test_labels = test_labels[::10]
    
    model_scores = test_scores
    labels_np = test_labels
    
    # --- TRUE ACC (SOFTMAX NORMALIZED) ---
    model_probs = np.exp(model_scores) / np.exp(model_scores).sum(axis=1, keepdims=True)
    true_acc = (model_probs.argmax(1) == labels_np).mean()
    
    # --- BEST (ORACLE) ---
    best_acc = compute_best_gt_accuracy(model_scores, labels_np)
    
    # --- OUR METHOD (BC-Min) WITH DIAGNOSTICS ---
    df_bc_min_bh, diagnostics = control_fdr_binary_then_combine(
        model_scores, labels_np, test_decoys, method='min', use_bh=True, verbose=False
    )
    
    pi0_diagnostics_all.append({
        'file': NAME,
        **diagnostics
    })
    
    pi0 = estimate_pi0_storey_from_df(df_bc_min_bh, 'q_values_binary_combined')
    df_method = compute_method_estimation_curve(
        df_bc_min_bh, 'q_values_binary_combined', pi0
    )
    
    est_acc_ours = df_method['Accuracy_est'].max() if len(df_method) > 0 else 0.0
    
    # --- BASELINES (SOFTMAX NORMALIZED) ---
    target_logits = torch.tensor(model_scores).to(DEVICE)
    target_labels = torch.tensor(labels_np).to(DEVICE)
    
    scaled_source = source_logits / temp
    scaled_target = target_logits / temp
    
    baseline_results = {}
    for method_name, method_func in BASELINE_METHODS.items():
        estimate = method_func(scaled_source, source_labels, scaled_target)
        baseline_results[method_name] = estimate
    
    # Store results
    results.append({
        'file': NAME,
        'true_acc': true_acc,
        'best_acc': best_acc,
        'ours_est': est_acc_ours,
        **baseline_results
    })

print(f"\n✓ Processed {len(results)} test images successfully")

# ============================================================================
# π₀ DIAGNOSTICS
# ============================================================================

print("\n" + "="*80)
print("π₀ ESTIMATION DIAGNOSTICS")
print("="*80)

# Aggregate π₀ statistics across all images
all_pi0_estimated = []
all_pi0_true = []
all_pi0_errors = []

for diag in pi0_diagnostics_all:
    for est, true, err in zip(diag['pi0_estimated'], diag['pi0_true'], diag['pi0_error']):
        if not np.isnan(est) and not np.isnan(true):
            all_pi0_estimated.append(est)
            all_pi0_true.append(true)
            all_pi0_errors.append(err)

all_pi0_estimated = np.array(all_pi0_estimated)
all_pi0_true = np.array(all_pi0_true)
all_pi0_errors = np.array(all_pi0_errors)

print(f"\nOverall π₀ Statistics (across all images and classes):")
print(f"  Mean π₀ estimated: {all_pi0_estimated.mean():.3f} ± {all_pi0_estimated.std():.3f}")
print(f"  Mean π₀ true:      {all_pi0_true.mean():.3f} ± {all_pi0_true.std():.3f}")
print(f"  Mean π₀ error:     {all_pi0_errors.mean():.3f} ± {all_pi0_errors.std():.3f}")
print(f"  Median π₀ error:   {np.median(all_pi0_errors):.3f}")

# Per-class statistics
print(f"\nπ₀ Statistics by Class:")
print("-" * 60)
for class_k in range(NUM_CLASSES):
    class_pi0_est = []
    class_pi0_true = []
    
    for diag in pi0_diagnostics_all:
        if class_k < len(diag['pi0_estimated']):
            est = diag['pi0_estimated'][class_k]
            true = diag['pi0_true'][class_k]
            if not np.isnan(est) and not np.isnan(true):
                class_pi0_est.append(est)
                class_pi0_true.append(true)
    
    if len(class_pi0_est) > 0:
        class_pi0_est = np.array(class_pi0_est)
        class_pi0_true = np.array(class_pi0_true)
        class_error = np.abs(class_pi0_est - class_pi0_true)
        
        print(f"\nClass {class_k}:")
        print(f"  π₀ estimated: {class_pi0_est.mean():.3f} ± {class_pi0_est.std():.3f}")
        print(f"  π₀ true:      {class_pi0_true.mean():.3f} ± {class_pi0_true.std():.3f}")
        print(f"  π₀ error:     {class_error.mean():.3f} ± {class_error.std():.3f}")

# Correlation analysis
print(f"\n" + "-" * 60)
print("Correlation Analysis:")
from scipy.stats import pearsonr, spearmanr

if len(all_pi0_errors) > 1:
    # Does π₀ error correlate with accuracy estimation error?
    accuracy_errors = np.abs(results_df['ours_est'] - results_df['true_acc'])
    
    # Compute average π₀ error per image
    image_pi0_errors = []
    for diag in pi0_diagnostics_all:
        errors = [e for e in diag['pi0_error'] if not np.isnan(e)]
        if len(errors) > 0:
            image_pi0_errors.append(np.mean(errors))
        else:
            image_pi0_errors.append(np.nan)
    
    image_pi0_errors = np.array(image_pi0_errors)
    valid_mask = ~np.isnan(image_pi0_errors)
    
    if valid_mask.sum() > 1:
        corr_pearson, p_pearson = pearsonr(
            image_pi0_errors[valid_mask], 
            accuracy_errors[valid_mask]
        )
        corr_spearman, p_spearman = spearmanr(
            image_pi0_errors[valid_mask], 
            accuracy_errors[valid_mask]
        )
        
        print(f"  π₀ error vs Accuracy error:")
        print(f"    Pearson:  r={corr_pearson:.3f}, p={p_pearson:.4f}")
        print(f"    Spearman: ρ={corr_spearman:.3f}, p={p_spearman:.4f}")

# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("COMPUTING MEAN ERRORS")
print("="*80)

methods = ['ours_est', 'ATC', 'ATC-NE', 'AC', 'DOC']
if 'COT' in results_df.columns:
    methods.append('COT')

print("\nMean Absolute Error (MAE) by Method:")
print("-" * 40)
for method in methods:
    errors = np.abs(results_df[method] - results_df['true_acc'])
    mae = errors.mean()
    std = errors.std()
    print(f"{method:15s}: {mae:.4f} ± {std:.4f}")

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: True vs Best Accuracy
plt.figure(figsize=(6, 6))
plt.scatter(results_df['true_acc'], results_df['best_acc'], alpha=0.7, s=50)
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
plt.xlabel('True Accuracy', fontweight='bold')
plt.ylabel('Best Accuracy (Oracle Threshold)', fontweight='bold')
plt.title('True vs Best Accuracy per Image', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('figures/scatter_true_vs_best.png', dpi=300)
plt.savefig('figures/scatter_true_vs_best.pdf')
plt.close()
print("✓ Saved: scatter_true_vs_best.png/pdf")

# Plot 2: Estimated vs True Accuracy (all methods)
colors_map = {
    'ours_est': '#FF9800',  # Orange
    'ATC': '#1976D2',       # Blue
    'ATC-NE': '#0288D1',    # Light Blue
    'AC': '#388E3C',        # Green
    'DOC': '#7B1FA2',       # Purple
    'COT': '#D32F2F'        # Red
}

plt.figure(figsize=(7, 7))
for method in methods:
    plt.scatter(
        results_df['true_acc'],
        results_df[method],
        label=method,
        alpha=0.6,
        s=50,
        color=colors_map.get(method, 'gray')
    )

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect')
plt.xlabel('True Accuracy', fontweight='bold')
plt.ylabel('Estimated Accuracy', fontweight='bold')
plt.title('Estimated vs True Accuracy (Per Image)', fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('figures/scatter_est_vs_true.png', dpi=300)
plt.savefig('figures/scatter_est_vs_true.pdf')
plt.close()
print("✓ Saved: scatter_est_vs_true.png/pdf")

# Plot 3: Error Distribution with CONSISTENT BINNING
fig, ax = plt.subplots(figsize=(8, 5))

# Compute global min/max for consistent bins
all_errors = []
for method in methods:
    errors = np.abs(results_df[method] - results_df['true_acc'])
    all_errors.extend(errors)

bins = np.linspace(0, max(all_errors), 30)

for method in methods:
    errors = np.abs(results_df[method] - results_df['true_acc'])
    ax.hist(errors, bins=bins, alpha=0.4, label=method, 
            color=colors_map.get(method, 'gray'))

ax.set_xlabel('Absolute Error', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Estimation Error Distribution (Consistent Binning)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/error_hist.png', dpi=300)
plt.savefig('figures/error_hist.pdf')
plt.close()
print("✓ Saved: error_hist.png/pdf")

# Plot 4: Mean Error Comparison Bar Chart
mae_data = []
for method in methods:
    errors = np.abs(results_df[method] - results_df['true_acc'])
    mae = errors.mean()
    mae_data.append({'Method': method, 'MAE': mae})

mae_df = pd.DataFrame(mae_data).sort_values('MAE')

fig, ax = plt.subplots(figsize=(8, 5))
colors_list = [colors_map.get(m, 'gray') for m in mae_df['Method']]
bars = ax.bar(range(len(mae_df)), mae_df['MAE'], color=colors_list, 
              alpha=0.7, edgecolor='black', linewidth=1)

ax.set_xlabel('Method', fontweight='bold')
ax.set_ylabel('Mean Absolute Error', fontweight='bold')
ax.set_title('Mean Error by Method (Lower is Better)', fontweight='bold')
ax.set_xticks(range(len(mae_df)))
ax.set_xticklabels(mae_df['Method'], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (idx, row) in enumerate(mae_df.iterrows()):
    ax.text(i, row['MAE'] + 0.002, f"{row['MAE']:.4f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/mean_error_comparison.png', dpi=300)
plt.savefig('figures/mean_error_comparison.pdf')
plt.close()
print("✓ Saved: mean_error_comparison.png/pdf")

# Plot 5: π₀ Estimation Quality
if len(all_pi0_estimated) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter: Estimated vs True π₀
    axes[0].scatter(all_pi0_true, all_pi0_estimated, alpha=0.5, s=30)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect estimation')
    axes[0].set_xlabel('True π₀', fontweight='bold')
    axes[0].set_ylabel('Estimated π₀', fontweight='bold')
    axes[0].set_title('π₀ Estimation Quality', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Histogram: π₀ Errors
    axes[1].hist(all_pi0_errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].axvline(np.mean(all_pi0_errors), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(all_pi0_errors):.3f}')
    axes[1].set_xlabel('|π₀_estimated - π₀_true|', fontweight='bold')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('π₀ Estimation Error Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/pi0_diagnostics.png', dpi=300)
    plt.savefig('figures/pi0_diagnostics.pdf')
    plt.close()
    print("✓ Saved: pi0_diagnostics.png/pdf")

# Plot 6: π₀ Error vs Accuracy Error
if len(image_pi0_errors) > 0 and valid_mask.sum() > 1:
    fig, ax = plt.subplots(figsize=(7, 6))
    
    ax.scatter(image_pi0_errors[valid_mask], accuracy_errors[valid_mask], 
               alpha=0.6, s=60, color='darkblue')
    
    # Add trend line
    z = np.polyfit(image_pi0_errors[valid_mask], accuracy_errors[valid_mask], 1)
    p = np.poly1d(z)
    x_line = np.linspace(image_pi0_errors[valid_mask].min(), 
                         image_pi0_errors[valid_mask].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    
    ax.set_xlabel('Mean π₀ Error (per image)', fontweight='bold')
    ax.set_ylabel('Accuracy Estimation Error', fontweight='bold')
    ax.set_title('π₀ Error vs Accuracy Error', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add correlation text
    ax.text(0.05, 0.95, 
            f"Pearson: r={corr_pearson:.3f}\nSpearman: ρ={corr_spearman:.3f}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=11)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/pi0_vs_accuracy_error.png', dpi=300)
    plt.savefig('figures/pi0_vs_accuracy_error.pdf')
    plt.close()
    print("✓ Saved: pi0_vs_accuracy_error.png/pdf")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to: figures/")
print(f"Total images processed: {len(results_df)}")
print(f"\nBest performing method: {mae_df.iloc[0]['Method']} (MAE: {mae_df.iloc[0]['MAE']:.4f})")

# ============================================================================
# DECOY QUALITY ANALYSIS (on a sample image for diagnostics)
# ============================================================================


"""
Decoy Quality Diagnostics and Method Improvement Suggestions

This script analyzes:
1. Decoy score quality (separation from target scores)
2. Decoy calibration (do decoys represent true negatives?)
3. Suggestions for improving BC-Min method
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu

def analyze_decoy_quality(target_scores, decoy_scores, target_labels, predicted_labels):
    """
    Analyze the quality of decoy score generation.
    
    Returns diagnostics about:
    - Score separation
    - Distribution differences
    - Class-specific quality
    """
    n_samples, n_classes = target_scores.shape
    
    print("\n" + "="*80)
    print("DECOY QUALITY ANALYSIS")
    print("="*80)
    
    diagnostics = {
        'overall': {},
        'per_class': []
    }
    
    # Overall statistics
    target_max = target_scores.max(axis=1)
    decoy_max = decoy_scores.max(axis=1)
    
    print("\n1. OVERALL SCORE STATISTICS:")
    print("-" * 60)
    print(f"Target scores (max):")
    print(f"  Mean: {target_max.mean():.3f}, Std: {target_max.std():.3f}")
    print(f"  Median: {np.median(target_max):.3f}")
    print(f"  Range: [{target_max.min():.3f}, {target_max.max():.3f}]")
    
    print(f"\nDecoy scores (max):")
    print(f"  Mean: {decoy_max.mean():.3f}, Std: {decoy_max.std():.3f}")
    print(f"  Median: {np.median(decoy_max):.3f}")
    print(f"  Range: [{decoy_max.min():.3f}, {decoy_max.max():.3f}]")
    
    # Score separation
    separation = target_max.mean() - decoy_max.mean()
    print(f"\nScore Separation (target - decoy): {separation:.3f}")
    
    # Statistical tests
    ks_stat, ks_pval = ks_2samp(target_max, decoy_max)
    mw_stat, mw_pval = mannwhitneyu(target_max, decoy_max, alternative='greater')
    
    print(f"\nDistribution Tests:")
    print(f"  KS test: stat={ks_stat:.3f}, p={ks_pval:.4e}")
    print(f"  Mann-Whitney U: stat={mw_stat:.1f}, p={mw_pval:.4e}")
    
    diagnostics['overall'] = {
        'target_mean': target_max.mean(),
        'decoy_mean': decoy_max.mean(),
        'separation': separation,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval
    }
    
    # Per-class analysis
    print("\n2. PER-CLASS ANALYSIS:")
    print("-" * 60)
    
    for class_k in range(n_classes):
        print(f"\nClass {class_k}:")
        
        # Samples predicted as this class
        mask_predicted = (predicted_labels == class_k)
        
        if mask_predicted.sum() == 0:
            print(f"  No samples predicted as class {class_k}")
            continue
        
        target_k = target_scores[mask_predicted, class_k]
        decoy_k = decoy_scores[mask_predicted, class_k]
        
        # True positives vs false positives
        mask_tp = (target_labels[mask_predicted] == class_k)
        mask_fp = ~mask_tp
        
        n_tp = mask_tp.sum()
        n_fp = mask_fp.sum()
        
        print(f"  Predicted as class {class_k}: {mask_predicted.sum()} samples")
        print(f"    True Positives: {n_tp}")
        print(f"    False Positives: {n_fp}")
        print(f"    Precision: {n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0:.3f}")
        
        if n_tp > 0:
            print(f"  Target scores (TP): mean={target_k[mask_tp].mean():.3f}, "
                  f"std={target_k[mask_tp].std():.3f}")
        if n_fp > 0:
            print(f"  Target scores (FP): mean={target_k[mask_fp].mean():.3f}, "
                  f"std={target_k[mask_fp].std():.3f}")
        
        print(f"  Decoy scores: mean={decoy_k.mean():.3f}, std={decoy_k.std():.3f}")
        
        # Key insight: Are FP target scores similar to decoys?
        if n_fp > 0:
            fp_target_mean = target_k[mask_fp].mean()
            decoy_mean = decoy_k.mean()
            decoy_quality = abs(fp_target_mean - decoy_mean)
            print(f"  Decoy Quality (|FP_target - decoy|): {decoy_quality:.3f}")
            print(f"    {'✓ GOOD' if decoy_quality < 0.5 else '✗ POOR'} - Decoys "
                  f"{'well' if decoy_quality < 0.5 else 'poorly'} match FP scores")
        
        diagnostics['per_class'].append({
            'class': class_k,
            'n_predicted': mask_predicted.sum(),
            'n_tp': n_tp,
            'n_fp': n_fp,
            'precision': n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
        })
    
    return diagnostics


def suggest_improvements(diagnostics, mae_comparison):
    """
    Based on diagnostics, suggest improvements to the BC-Min method.
    """
    print("\n" + "="*80)
    print("IMPROVEMENT SUGGESTIONS")
    print("="*80)
    
    print("\n🔍 DIAGNOSIS:")
    print("-" * 60)
    
    # Check decoy quality
    separation = diagnostics['overall']['separation']
    ks_pval = diagnostics['overall']['ks_pval']
    
    issues = []
    
    if separation < 1.0:
        issues.append("LOW_SEPARATION")
        print("⚠️  Issue: Low score separation between targets and decoys")
        print(f"    Current separation: {separation:.3f}")
        print(f"    Decoys may not adequately represent false positives")
    
    if ks_pval > 0.05:
        issues.append("SIMILAR_DISTRIBUTIONS")
        print("⚠️  Issue: Target and decoy distributions are too similar")
        print(f"    KS test p-value: {ks_pval:.4f}")
        print(f"    Decoys may be too 'easy' to distinguish")
    
    # Check π₀ estimation
    # This would come from the main script results
    
    # Check method performance
    if mae_comparison is not None:
        our_mae = mae_comparison.get('ours_est', None)
        baseline_maes = {k: v for k, v in mae_comparison.items() if k != 'ours_est'}
        
        if our_mae is not None and len(baseline_maes) > 0:
            best_baseline_mae = min(baseline_maes.values())
            
            if our_mae > best_baseline_mae * 1.2:
                issues.append("UNDERPERFORMING")
                print(f"⚠️  Issue: BC-Min underperforms baselines significantly")
                print(f"    BC-Min MAE: {our_mae:.4f}")
                print(f"    Best baseline: {best_baseline_mae:.4f}")
    
    print("\n💡 SUGGESTED IMPROVEMENTS:")
    print("-" * 60)
    
    if "LOW_SEPARATION" in issues or "SIMILAR_DISTRIBUTIONS" in issues:
        print("\n1. IMPROVE DECOY GENERATION:")
        print("   a) Train flows longer or with better architecture")
        print("   b) Use negative samples from other classes as decoys")
        print("   c) Mix synthetic decoys with real negative samples")
        print("   d) Add noise/perturbation to make decoys more diverse")
    
    if "UNDERPERFORMING" in issues:
        print("\n2. IMPROVE π₀ ESTIMATION:")
        print("   a) Use more robust π₀ estimators (e.g., bootstrap)")
        print("   b) Estimate π₀ globally instead of per-class")
        print("   c) Use convex optimization for π₀")
        print("   d) Try different lambda ranges in Storey's method")
        
        print("\n3. ALTERNATIVE COMBINATION STRATEGIES:")
        print("   a) Try 'predicted' instead of 'min' combination")
        print("   b) Weighted combination based on prediction confidence")
        print("   c) Use FDR2 instead of B-H for binary control")
        print("   d) Calibrate q-values after combination")
        
        print("\n4. LEVERAGE SOURCE DATA:")
        print("   a) Use source accuracy as prior in estimation")
        print("   b) Adaptive thresholding based on source distribution")
        print("   c) Transfer learning: train on source, fine-tune on target decoys")
    
    print("\n5. HYBRID APPROACHES:")
    print("   a) Combine BC-Min with best baseline (ensemble)")
    print("   b) Use BC-Min for high-confidence, baseline for low-confidence")
    print("   c) Weighted average based on per-class confidence")
    
    return issues


def plot_decoy_quality(target_scores, decoy_scores, target_labels, predicted_labels, 
                       save_path='figures/decoy_quality.png'):
    """
    Visualize decoy quality across classes.
    """
    n_samples, n_classes = target_scores.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for class_k in range(min(n_classes, 6)):
        ax = axes[class_k]
        
        # Get samples predicted as this class
        mask_predicted = (predicted_labels == class_k)
        
        if mask_predicted.sum() == 0:
            ax.text(0.5, 0.5, f'No predictions\nfor class {class_k}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'Class {class_k}')
            continue
        
        target_k = target_scores[mask_predicted, class_k]
        decoy_k = decoy_scores[mask_predicted, class_k]
        
        # Separate TP and FP
        mask_tp = (target_labels[mask_predicted] == class_k)
        mask_fp = ~mask_tp
        
        # Plot distributions
        if mask_tp.sum() > 0:
            ax.hist(target_k[mask_tp], bins=30, alpha=0.5, label='Target (TP)', 
                   color='green', density=True)
        if mask_fp.sum() > 0:
            ax.hist(target_k[mask_fp], bins=30, alpha=0.5, label='Target (FP)', 
                   color='orange', density=True)
        
        ax.hist(decoy_k, bins=30, alpha=0.5, label='Decoy', 
               color='red', density=True)
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Class {class_k} (n={mask_predicted.sum()})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_classes, 6):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")

print("\n" + "="*80)
print("DECOY QUALITY ANALYSIS (Sample)")
print("="*80)

# Load one test file for detailed analysis
test_file_sample = test_files[0]
print(f"\nAnalyzing: {os.path.basename(test_file_sample)}")

try:
    test_data = torch.load(test_file_sample, weights_only=False)
    features = test_data['features']
    predictions = test_data['predictions']
    labels = torch.tensor(test_data['mask'])
    
    predictions = torch.flatten(predictions, start_dim=2).squeeze(0).T    
    features = torch.flatten(features, start_dim=2).squeeze(0).T    
    labels = torch.flatten(labels, start_dim=0)
    
    test_ds = ScoreFeatureDataset(predictions, features, predictions, labels)
    
    test_scores, test_decoys, test_labels = separate_flows.generate_decoys(
        test_ds, device=DEVICE
    )
    
    test_scores = test_scores[::10]
    test_decoys = test_decoys[::10]
    test_labels = test_labels[::10]
    
    predicted_labels = test_scores.argmax(axis=1)
    
    # Run diagnostics
    
    diagnostics = analyze_decoy_quality(test_scores, test_decoys, test_labels, predicted_labels)
    
    # Create visualization
    plot_decoy_quality(test_scores, test_decoys, test_labels, predicted_labels,
                      save_path='figures/decoy_quality.png')
    
    # Get MAE comparison dict
    mae_dict = {row['Method']: mae_df[mae_df['Method'] == row['Method']]['MAE'].iloc[0] 
                for idx, row in mae_df.iterrows()}
    
    # Suggest improvements
    suggest_improvements(diagnostics, mae_dict)
    
except Exception as e:
    print(f"Could not run decoy analysis: {e}")

print("\n" + "="*80)
print("ALL ANALYSIS COMPLETE")
print("="*80)