import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from bisect import bisect
import os
from sklearn.neighbors import KernelDensity



def empirical_p_values(distribution, query):
    dist_len = len(distribution)
    query_len = len(query)
    p_values = np.zeros(query_len)
    sorted_dist = np.sort(distribution)
    for i, score in enumerate(query):
        p_values[i] = (dist_len - bisect(sorted_dist, score)) / dist_len

    return p_values


def estimate_pi0_storey(p_values, lambda_range=np.arange(0.05, 0.95, 0.05)):
    pi0_estimates = []
    for lam in lambda_range:
        pi0_lam = np.mean(p_values > lam) / (1 - lam)
        pi0_estimates.append(pi0_lam)
    pi0 = np.mean(pi0_estimates)
    return min(1.0, max(0, pi0))


def benjamini_hochberg_storey(p_values):
    n = len(p_values)
    pi0 = estimate_pi0_storey(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    q_values = np.zeros(n)
    prev_min = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        q_value = pi0 * sorted_p[i] * n / rank
        q_value = min(q_value, prev_min)
        q_values[sorted_indices[i]] = q_value
        prev_min = q_value

    return q_values


def benjamini_hochberg_fixed(p_values, pi0=1.0):
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    q_values = np.zeros(n)
    prev_min = 1.0
    for i in range(n-1, -1, -1):
        rank = i + 1
        q_value = pi0 * sorted_p[i] * n / rank
        q_value = min(q_value, prev_min)
        q_values[sorted_indices[i]] = q_value
        prev_min = q_value

    return q_values


def calculate_qvalues_from_pvalues(distribution, query, pi_0=0.9):
    p_values = empirical_p_values(np.sort(distribution), query)
    q_values = p_values * len(p_values) * pi_0
    q_values = q_values / np.arange(1, len(p_values) + 1)
    for i in range(len(p_values)-1, 0, -1):
        q_values[i-1] = min(q_values[i-1], q_values[i])
    return np.sort(q_values)


def negative_training_benchmark(train_cnn_scores, train_labels, test_cnn_scores,
                                target_class, fdr_level=0.05):
    neg_train_mask = (train_labels != target_class)
    null_distribution = train_cnn_scores[neg_train_mask]
    test_scores = test_cnn_scores
    p_values = empirical_p_values(null_distribution, test_scores)
    q_values = benjamini_hochberg_storey(np.sort(p_values))
    n_discoveries = (q_values <= fdr_level).sum()
    return n_discoveries, q_values, p_values

def control_fdr(model_scores, target_labels, decoys,
                train_cnn_scores=None, train_labels=None,
                test_cnn_scores=None, target_class=None):
    
    # Create dataframe with original index preserved
    df = pd.DataFrame({
        'model_score': model_scores,
        'label': target_labels,
        'decoy_score': decoys,
        'original_index': np.arange(len(model_scores))
    })
    
    # Compute p-values first (before any sorting)
    df['p_value'] = empirical_p_values(np.sort(decoys), model_scores)
    
    # Compute q-values using BH (before any sorting)
    df['q_values_bh_storey'] = benjamini_hochberg_storey(df['p_value'].values)
    df['q_values_bh_fixed'] = benjamini_hochberg_fixed(df['p_value'].values)
    
    # Now sort by model_score for ground truth FDR calculation
    df = df.sort_values(by='model_score', ascending=False).reset_index(drop=True)
    
    # Ground truth FDR
    df['fdr'] = 0.0
    num_neg = 0
    num_pos = 0
    
    for index, row in df.iterrows():
        if row['label'] == 0:
            num_neg += 1
        else:
            num_pos += 1
        if num_pos + num_neg > 0:
            df.at[index, 'fdr'] = (num_neg + 1.0) / (num_pos + num_neg)
    
    df['q_values_ground_truth'] = df['fdr'].copy()
    prev = 1
    for index in range(len(df) - 1, -1, -1):
        df.loc[index, 'q_values_ground_truth'] = min(prev, df.loc[index, 'q_values_ground_truth'])
        prev = df.loc[index, 'q_values_ground_truth']
    
    # TDC method - compute on sorted data by max_score
    df['max_score'] = df[['model_score', 'decoy_score']].max(axis=1)
    
    # Sort by max_score for TDC, preserving original_index
    df_sorted_by_max = df.sort_values(by='max_score', ascending=False).reset_index(drop=True)
    
    fdr_tdc_list = []
    num_decoy_wins = 0
    num_target_wins = 0
    
    for index, row in df_sorted_by_max.iterrows():
        if row['model_score'] >= row['decoy_score']:  # Target wins
            num_target_wins += 1
            if num_target_wins > 0:
                fdr_tdc_list.append((num_decoy_wins + 1.0) / num_target_wins)
            else:
                fdr_tdc_list.append(1.0)
        else:  # Decoy wins - not counted as discovery
            num_decoy_wins += 1
            fdr_tdc_list.append(np.nan)  # No FDR value for decoy wins
    
    df_sorted_by_max['fdr_tdc'] = fdr_tdc_list
    
    # Make monotone (only for target wins)
    q_values_tdc_list = df_sorted_by_max['fdr_tdc'].tolist()
    prev = 1.0
    for index in range(len(q_values_tdc_list) - 1, -1, -1):
        if not np.isnan(q_values_tdc_list[index]):
            q_values_tdc_list[index] = min(prev, q_values_tdc_list[index])
            prev = q_values_tdc_list[index]
    
    df_sorted_by_max['q_values_tdc'] = q_values_tdc_list
    
    # Map TDC q-values back using original_index
    tdc_mapping = dict(zip(df_sorted_by_max['original_index'], 
                           df_sorted_by_max['q_values_tdc']))
    df['q_values_tdc'] = df['original_index'].map(tdc_mapping)
    
    if (train_cnn_scores is not None and train_labels is not None and
        test_cnn_scores is not None and target_class is not None):
        # Negative training - compute on original order using original_index
        neg_train_mask = (train_labels != target_class)
        null_distribution = train_cnn_scores[neg_train_mask]
        
        # Create mapping from original_index to values
        p_values_neg_full = empirical_p_values(null_distribution, model_scores)
        q_values_neg_full = benjamini_hochberg_storey(p_values_neg_full)
        
        # Map to current df using original_index
        neg_p_mapping = dict(zip(np.arange(len(model_scores)), p_values_neg_full))
        neg_q_mapping = dict(zip(np.arange(len(model_scores)), q_values_neg_full))
        
        df['p_values_negative_training'] = df['original_index'].map(neg_p_mapping)
        df['q_values_negative_training'] = df['original_index'].map(neg_q_mapping)
    
    return df

def plot_fdr_comprehensive(df, filename="fdr_comprehensive"):
    """Comprehensive FDR plot with all methods"""
    plt.figure(figsize=(6, 4))  # Updated figure size
    
    methods = [
        ('q_values_tdc', 'TDC', 'blue', '-'),
        ('q_values_ground_truth', 'Ground Truth', 'green', '-'),
        ('q_values_bh_storey', 'BH (Storey π₀)', 'red', '-'),
    ]
    if 'q_values_negative_training' in df.columns:
        methods.append(('q_values_negative_training', 'Negative Training', 'orange', ':'))
    
    for col_name, label, color, linestyle in methods:
        df_sorted = df.sort_values(by=col_name, ascending=True).reset_index(drop=True)
        plt.plot(df_sorted[col_name], np.arange(len(df_sorted)), 
                marker='none', linestyle=linestyle, label=label, linewidth=2, color=color)
    
    plt.ylabel('Number of Discoveries')
    plt.xlabel('Q-values')
    plt.title('Comparison of FDR Control Methods')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    
    # Save in both formats
    plt.savefig(PATH + filename + ".png", bbox_inches='tight', dpi=300)
    plt.savefig(PATH + filename + ".pdf", bbox_inches='tight')
    plt.show()
    
    return plt

def plot_fdr_comprehensive(df, filename="fdr_comprehensive"):
    """Comprehensive FDR plot with all methods"""
    
    methods = [
        ('q_values_ground_truth', 'Ground Truth', 'green', '-'),
        ('q_values_bh_storey', 'BH (Storey π₀)', 'red', '-'),
        ('q_values_tdc', 'TDC', 'blue', '-'),
    ]
    if 'q_values_negative_training' in df.columns:
        methods.append(('q_values_negative_training', 'Negative Training', 'orange', ':'))
    
    # Создаем два графика с разными xlim
    xlim_configs = [
        (1.0, "full"),    # График до 1.0
        (0.2, "zoom")     # График до 0.2
    ]
    
    for xmax, suffix in xlim_configs:
        plt.figure(figsize=(6, 4))  # Updated figure size
        
        for col_name, label, color, linestyle in methods:
            df_sorted = df.sort_values(by=col_name, ascending=True).reset_index(drop=True)
            plt.plot(df_sorted[col_name], np.arange(len(df_sorted)), 
                    marker='none', linestyle=linestyle, label=label, linewidth=2, color=color)
        
        plt.ylabel('Number of discoveries')
        plt.xlabel('Estimated FDR (Q-values)')
        plt.title(f'Comparison of FDR Control Methods')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, xmax)
        
        # Сохраняем с разными именами файлов
        plt.savefig(PATH + f"{filename}_{suffix}.png", bbox_inches='tight', dpi=300)
        plt.savefig(PATH + f"{filename}_{suffix}.pdf", bbox_inches='tight')
        plt.show()
    
    return plt


def estimate_confusion_matrix_1(df, pi0):
    df_cm = df.copy()
    total_samples = len(df)
    
    # Sort by model score descending (from highest to lowest threshold)
    df_cm = df_cm.sort_values(by='model_score', ascending=False).reset_index(drop=True)
    
    # Threshold is the model_score at each position
    df_cm['threshold'] = df_cm['model_score']
    
    # Number of discoveries at this threshold (samples with score >= threshold)
    df_cm['n_discoveries'] = np.arange(1, len(df_cm) + 1)
    
    # Estimate confusion matrix components using formulas from Section 5
    # FP(s_th) = |{f(t) >= s_th}| * FDP(s_th)
    df_cm['FP_est'] = df_cm['n_discoveries'] * (df_cm['q_values_bh_storey'])
    
    # TP(s_th) = |{f(t) >= s_th}| * (1 - FDP(s_th))
    df_cm['TP_est'] = df_cm['n_discoveries'] * (1 - df_cm['q_values_bh_storey'])
    
    # TN(s_th) = |T| * pi0 - FP(s_th)
    df_cm['TN_est'] = total_samples * pi0 - df_cm['FP_est']
    
    # FN(s_th) = |T| * (1 - pi0) - TP(s_th)
    df_cm['FN_est'] = total_samples * (1 - pi0) - df_cm['TP_est']
    
    # Accuracy = (TP + TN) / |T|
    df_cm['Accuracy_est'] = (df_cm['TP_est'] + df_cm['TN_est']) / total_samples
    
    # Ground truth accuracy for comparison
    # At each threshold: TP_true = positives with score >= threshold
    #                    TN_true = negatives with score < threshold
    df_cm['TP_true'] = (df_cm['label'] == 1).cumsum()
    df_cm['TN_true'] = (df_cm['label'] == 0).sum() - (df_cm['label'] == 0).cumsum()
    df_cm['Accuracy_true'] = (df_cm['TP_true'] + df_cm['TN_true']) / total_samples
    
    return df_cm


def estimate_confusion_matrix(df, pi0):
    df_cm = df.copy()
    total_samples = len(df)
    
    # Make sure we're sorted by model score descending
    df_cm = df_cm.sort_values(by='model_score', ascending=False).reset_index(drop=True)
    
    # Threshold is the model_score at each position
    df_cm['threshold'] = df_cm['model_score']
    
    # Number of discoveries at this threshold (samples with score >= threshold)
    # This is just the rank (1, 2, 3, ...)
    df_cm['n_discoveries'] = np.arange(1, len(df_cm) + 1)
    
    # For each threshold, estimate FDP using p-values and BH procedure
    # The q-value at position i tells us the FDR if we make i discoveries
    # But we want the FDP at each threshold
    
    # Method 1: Use p-values directly to estimate FDP
    # At threshold corresponding to rank i:
    # Expected number of false positives = n_discoveries * p_value * pi0
    # (because p_value estimates P(null score >= threshold))
    
    # FDP estimate at each threshold
    df_cm['FDP_est'] = df_cm['p_value'] * pi0
    
    # Now compute confusion matrix
    # FP(s_th) = |{f(t) >= s_th}| * FDP(s_th)
    df_cm['FP_est'] = df_cm['n_discoveries'] * df_cm['FDP_est']
    
    # TP(s_th) = |{f(t) >= s_th}| * (1 - FDP(s_th))
    df_cm['TP_est'] = df_cm['n_discoveries'] * (1 - df_cm['FDP_est'])
    
    # TN(s_th) = |T| * pi0 - FP(s_th)
    df_cm['TN_est'] = total_samples * pi0 - df_cm['FP_est']
    
    # FN(s_th) = |T| * (1 - pi0) - TP(s_th)
    df_cm['FN_est'] = total_samples * (1 - pi0) - df_cm['TP_est']
    
    # Clip negative values (can happen due to estimation error)
    df_cm['FP_est'] = np.maximum(0, df_cm['FP_est'])
    df_cm['TP_est'] = np.maximum(0, df_cm['TP_est'])
    df_cm['TN_est'] = np.maximum(0, df_cm['TN_est'])
    df_cm['FN_est'] = np.maximum(0, df_cm['FN_est'])
    
    # Accuracy = (TP + TN) / |T|
    df_cm['Accuracy_est'] = (df_cm['TP_est'] + df_cm['TN_est']) / total_samples
    
    # Ground truth accuracy for comparison
    # At each threshold: TP_true = positives with score >= threshold
    #                    TN_true = negatives with score < threshold
    df_cm['TP_true'] = (df_cm['label'] == 1).cumsum()
    df_cm['FP_true'] = (df_cm['label'] == 0).cumsum()
    df_cm['TN_true'] = (df_cm['label'] == 0).sum() - df_cm['FP_true']
    df_cm['FN_true'] = (df_cm['label'] == 1).sum() - df_cm['TP_true']
    df_cm['Accuracy_true'] = (df_cm['TP_true'] + df_cm['TN_true']) / total_samples
    
    return df_cm


def estimate_confusion_matrix_v2(df, pi0):
    """
    Alternative version using q-values directly
    """
    df_cm = df.copy()
    total_samples = len(df)
    
    df_cm = df_cm.sort_values(by='model_score', ascending=False).reset_index(drop=True)
    df_cm['threshold'] = df_cm['model_score']
    df_cm['n_discoveries'] = np.arange(1, len(df_cm) + 1)
    
    # q-value IS the FDR estimate for making n_discoveries
    # So: E[FP] = n_discoveries * q_value
    df_cm['FP_est'] = df_cm['n_discoveries'] * df_cm['q_values_bh_storey']
    df_cm['TP_est'] = df_cm['n_discoveries'] - df_cm['FP_est']
    
    # For TN and FN, we need pi0
    df_cm['TN_est'] = total_samples * pi0 - df_cm['FP_est']
    df_cm['FN_est'] = total_samples * (1 - pi0) - df_cm['TP_est']
    
    df_cm['FP_est'] = np.maximum(0, df_cm['FP_est'])
    df_cm['TP_est'] = np.maximum(0, df_cm['TP_est'])
    df_cm['TN_est'] = np.maximum(0, df_cm['TN_est'])
    df_cm['FN_est'] = np.maximum(0, df_cm['FN_est'])
    
    df_cm['Accuracy_est'] = (df_cm['TP_est'] + df_cm['TN_est']) / total_samples
    
    # Ground truth
    df_cm['TP_true'] = (df_cm['label'] == 1).cumsum()
    df_cm['FP_true'] = (df_cm['label'] == 0).cumsum()
    df_cm['TN_true'] = (df_cm['label'] == 0).sum() - df_cm['FP_true']
    df_cm['FN_true'] = (df_cm['label'] == 1).sum() - df_cm['TP_true']
    df_cm['Accuracy_true'] = (df_cm['TP_true'] + df_cm['TN_true']) / total_samples
    
    return df_cm


def plot_confusion_matrix_analysis(df, pi0, filename="confusion_matrix_analysis"):
    """
    Create a 3-panel plot showing:
    1. TP and FP estimates vs threshold
    2. TN and FN estimates vs threshold
    3. Accuracy estimates vs threshold
    """
    df_cm = estimate_confusion_matrix_v2(df, pi0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: TP and FP vs threshold
    axes[0].plot(df_cm['threshold'], df_cm['TP_est'], 
                 label='TP (estimated)', color='green', linewidth=2)
    axes[0].plot(df_cm['threshold'], df_cm['FP_est'], 
                 label='FP (estimated)', color='red', linewidth=2)
    # Add ground truth for comparison
    axes[0].plot(df_cm['threshold'], df_cm['TP_true'], 
                 label='TP (true)', color='green', linewidth=1, linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Threshold (score)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('True Positives and False Positives')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: TN and FN vs threshold
    axes[1].plot(df_cm['threshold'], df_cm['TN_est'], 
                 label='TN (estimated)', color='blue', linewidth=2)
    axes[1].plot(df_cm['threshold'], df_cm['FN_est'], 
                 label='FN (estimated)', color='orange', linewidth=2)
    # Add ground truth for comparison
    axes[1].plot(df_cm['threshold'], df_cm['TN_true'], 
                 label='TN (true)', color='blue', linewidth=1, linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Threshold (score)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('True Negatives and False Negatives')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Accuracy vs threshold
    axes[2].plot(df_cm['threshold'], df_cm['Accuracy_est'], 
                 label='Accuracy (estimated)', color='dodgerblue', linewidth=2, linestyle='-')
    axes[2].plot(df_cm['threshold'], df_cm['Accuracy_true'], 
                 label='Accuracy (ground truth)', color='black', linewidth=2, linestyle='--')
    axes[2].set_xlabel('Threshold (score)')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy Estimation')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(PATH + filename + ".png", bbox_inches='tight', dpi=300)
    plt.savefig(PATH + filename + ".pdf", bbox_inches='tight')
    plt.show()
    
    return df_cm


def control_fdr_multiclass(model_scores, target_labels, decoy_scores,
                           train_cnn_scores=None, train_labels=None,
                           test_cnn_scores=None, num_classes=None):
    """
    Multi-class FDR control using Mix-Max method.
    
    Args:
        model_scores: array of shape (n_samples, num_classes) - scores for each class
        target_labels: array of shape (n_samples,) - true class labels
        decoy_scores: array of shape (n_samples, num_classes) - decoy scores for each class
    """
    
    n_samples = len(model_scores)
    if num_classes is None:
        num_classes = model_scores.shape[1]
    
    # Create dataframe with original index preserved
    df = pd.DataFrame({
        'original_index': np.arange(n_samples),
        'label': target_labels
    })
    
    # Add model scores and decoy scores for each class
    for k in range(num_classes):
        df[f'model_score_class_{k}'] = model_scores[:, k]
        df[f'decoy_score_class_{k}'] = decoy_scores[:, k]
    
    # Compute max scores
    df['max_model_score'] = model_scores.max(axis=1)
    df['max_decoy_score'] = decoy_scores.max(axis=1)
    df['predicted_class'] = model_scores.argmax(axis=1)
    
    # Overall max score (competition between all model scores and decoy scores)
    df['max_score'] = df[['max_model_score', 'max_decoy_score']].max(axis=1)
    
    # Estimate π₀ for each class using Storey method
    pi0_estimates = []
    for k in range(num_classes):
        # Get samples where this class is NOT the true class (negatives for class k)
        neg_mask = (target_labels != k)
        null_distribution_k = model_scores[neg_mask, k]
        query_k = model_scores[:, k]
        
        # Compute p-values for class k
        p_values_k = empirical_p_values(null_distribution_k, query_k)
        
        # Estimate π₀^k
        pi0_k = estimate_pi0_storey(p_values_k)
        pi0_estimates.append(pi0_k)
        
        # Store p-values for this class
        df[f'p_value_class_{k}'] = p_values_k
    
    # Estimate overall π₀ = [1 - Σ(1 - π₀^k)]_{[0,1]}
    pi0_overall = 1 - sum([(1 - pi0_k) for pi0_k in pi0_estimates])
    pi0_overall = max(0.0, min(1.0, pi0_overall))  # Crop to [0,1]
    
    df['pi0_overall'] = pi0_overall
    
    # ============ GROUND TRUTH FDR ============
    # Ground truth doesn't know about decoys!
    # Sort only by max_model_score
    df_sorted_gt = df.sort_values(by='max_model_score', ascending=False).reset_index(drop=True)
    
    num_incorrect = 0
    num_total = 0
    
    fdr_gt_list = []
    for index, row in df_sorted_gt.iterrows():
        num_total += 1
        # Check if prediction is correct
        if row['predicted_class'] != row['label']:
            num_incorrect += 1
        
        fdr_gt = num_incorrect / num_total
        fdr_gt_list.append(fdr_gt)
    
    df_sorted_gt['fdr_gt'] = fdr_gt_list
    
    # Make monotone - идем с КОНЦА к НАЧАЛУ
    q_values_gt = df_sorted_gt['fdr_gt'].values.copy()
    for i in range(len(q_values_gt) - 2, -1, -1):  # от предпоследнего к первому
        q_values_gt[i] = min(q_values_gt[i], q_values_gt[i + 1])
    
    df_sorted_gt['q_values_ground_truth'] = q_values_gt
    
    # ВАЖНО: маппим обратно используя original_index
    gt_mapping = dict(zip(df_sorted_gt['original_index'], 
                          df_sorted_gt['q_values_ground_truth']))
    df['q_values_ground_truth'] = df['original_index'].map(gt_mapping)
    
    # ============ MIX-MAX FDR (TDC-like) ============
    # Sort by max_score (this includes competition with decoys)
    df_sorted_by_max = df.sort_values(by='max_score', ascending=False).reset_index(drop=True)
    
    # ВАЖНО: Сначала посчитаем cumulative counts идя СВЕРХУ ВНИЗ
    num_decoy_wins_cumsum = []
    num_target_wins_cumsum = []
    decoy_count = 0
    target_count = 0
    
    for index, row in df_sorted_by_max.iterrows():
        if row['max_model_score'] >= row['max_decoy_score']:  # Target wins
            target_count += 1
        else:  # Decoy wins
            decoy_count += 1
        
        num_decoy_wins_cumsum.append(decoy_count)
        num_target_wins_cumsum.append(target_count)
    
    df_sorted_by_max['num_decoy_wins'] = num_decoy_wins_cumsum
    df_sorted_by_max['num_target_wins'] = num_target_wins_cumsum
    
    # Теперь считаем FDR только для target wins
    fdr_mm_list = []
    for index, row in df_sorted_by_max.iterrows():
        if row['max_model_score'] >= row['max_decoy_score']:  # Target wins
            if row['num_target_wins'] > 0:
                fdr_mm = (row['num_decoy_wins'] + 1.0) / row['num_target_wins']
                fdr_mm_list.append(fdr_mm)
            else:
                fdr_mm_list.append(1.0)
        else:  # Decoy wins
            fdr_mm_list.append(np.nan)
    
    df_sorted_by_max['fdr_mm'] = fdr_mm_list
    
    # Make q-values monotone - ТОЛЬКО ДЛЯ TARGET WINS, идем с конца к началу
    q_values_mm = df_sorted_by_max['fdr_mm'].values.copy()
    
    # Находим последний target win и начинаем с него
    last_valid_idx = -1
    for i in range(len(q_values_mm) - 1, -1, -1):
        if not np.isnan(q_values_mm[i]):
            last_valid_idx = i
            break
    
    if last_valid_idx >= 0:
        prev = q_values_mm[last_valid_idx]
        for i in range(last_valid_idx - 1, -1, -1):
            if not np.isnan(q_values_mm[i]):
                q_values_mm[i] = min(prev, q_values_mm[i])
                prev = q_values_mm[i]
    
    df_sorted_by_max['q_values_mm'] = q_values_mm
    
    print("\nMM Debug:")
    print(f"First 30 samples (sorted by max_score DESC):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 300)
    print(df_sorted_by_max[['original_index', 'label', 'predicted_class', 'max_model_score', 
                            'max_decoy_score', 'max_score', 'num_decoy_wins', 'num_target_wins',
                            'fdr_mm', 'q_values_mm']].head(30))
    
    # После подсчета FDR
    print(f"\nFirst target win index: {df_sorted_by_max[df_sorted_by_max['max_model_score'] >= df_sorted_by_max['max_decoy_score']].index[0] if len(df_sorted_by_max[df_sorted_by_max['max_model_score'] >= df_sorted_by_max['max_decoy_score']]) > 0 else 'None'}")

    # Покажи первые несколько target wins
    target_wins_df = df_sorted_by_max[df_sorted_by_max['max_model_score'] >= df_sorted_by_max['max_decoy_score']].head(20)
    print("\nFirst 20 target wins:")
    print(target_wins_df[['original_index', 'max_model_score', 'max_decoy_score', 'num_decoy_wins', 'num_target_wins', 'fdr_mm', 'q_values_mm']])
    
    # Map back to original order
    mm_mapping = dict(zip(df_sorted_by_max['original_index'], 
                          df_sorted_by_max['q_values_mm']))
    df['q_values_mm'] = df['original_index'].map(mm_mapping)
    
    # ============ BH method ============
    # Use decoys as null distribution
    p_values_overall = []
    for i in range(n_samples):
        max_model = model_scores[i].max()
        decoy_vals = decoy_scores[i]
        p_val = (decoy_vals >= max_model).sum() / num_classes
        p_values_overall.append(p_val)
    
    p_values_overall = np.array(p_values_overall)
    df['p_value_overall'] = p_values_overall
    df['q_values_bh_storey'] = benjamini_hochberg_storey(p_values_overall)
    
    print("\nQ-values statistics:")
    print(f"GT q-values range: {df['q_values_ground_truth'].min():.4f} - {df['q_values_ground_truth'].max():.4f}")
    print(f"MM q-values range: {df['q_values_mm'].min():.4f} - {df['q_values_mm'].max():.4f} (excluding NaN)")
    print(f"MM NaN count: {df['q_values_mm'].isna().sum()}")
    print(f"Samples with q < 0.1:")
    print(f"  GT: {(df['q_values_ground_truth'] < 0.1).sum()}")
    print(f"  MM: {(df['q_values_mm'] < 0.1).sum()}")
    
    return df, pi0_overall, pi0_estimates