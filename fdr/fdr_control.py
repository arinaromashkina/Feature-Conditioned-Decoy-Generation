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
    return min(1.0, max(0, np.mean(pi0_estimates)))


def benjamini_hochberg(p_values, pi0=1.0):
    """Standard Benjamini-Hochberg procedure with π₀ adjustment."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    q_values = np.zeros(n)
    
    for i in range(n):
        q = min(1.0, (n * sorted_p[i] * pi0) / (i + 1))
        q_values[i] = q
    
    # Enforce monotonicity
    for i in range(n-2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i+1])
    
    # Map back to original order
    original_q = np.zeros(n)
    original_q[sorted_idx] = q_values
    return original_q


def calculate_fdr2_qvalues(target_scores, decoy_scores):
    """Calculate FDR2 q-values using TDC (LABEL-FREE)."""
    n = len(target_scores)
    combined_scores = np.maximum(target_scores, decoy_scores)
    is_target_win = (target_scores > decoy_scores).astype(int)
    
    sort_idx = np.argsort(combined_scores)[::-1]
    sorted_wins = is_target_win[sort_idx]
    
    fdr2_values = np.zeros(n)
    n_decoy_wins = 0
    
    for i in range(n):
        if sorted_wins[i] == 0:
            n_decoy_wins += 1
        fdr2_values[i] = (2 * n_decoy_wins) / (i + 1) if (i + 1) > 0 else 0
    
    # Enforce monotonicity
    q_fdr2 = np.minimum.accumulate(fdr2_values[::-1])[::-1]
    
    # Map back to original order
    final_q_fdr2 = np.zeros(n)
    for i in range(n):
        final_q_fdr2[sort_idx[i]] = q_fdr2[i]
    
    return final_q_fdr2


def calculate_mixmax_qvalues(model_scores, decoy_scores):
    """
    Mix-Max FDR control from Keich et al. with pi0=0.
    
    With pi0=0, FDP_MM(s_th) simplifies to:
    
    FDP_MM(s_th) = 
        SUM_{Zj > s_th} [ P(f(t) <= Zj) / P(g(t) <= Zj) ]_[0,1]
        ─────────────────────────────────────────────────────────
                        1{ f(t) >= s_th }
    
    Where:
        f(t)  = target score per sample (model score for predicted class)
        Zj    = decoy scores (null distribution)
        g(t)  = mix_max_score = max(f(t), Z) per sample
        s_th  = threshold swept from high to low
    
    P(f(t) <= z)  estimated empirically over all samples
    P(g(t) <= z)  estimated empirically over all samples
    """
    n = len(model_scores)
    pi0 = 0.0

    # ── Scores ────────────────────────────────────────────────────────────
    pred_classes  = model_scores.argmax(axis=1)           # (n,)
    target_scores = model_scores[np.arange(n), pred_classes]  # f(t), shape (n,)
    null_scores   = decoy_scores[np.arange(n), pred_classes]  # Zj,   shape (n,)

    mix_max_scores = np.maximum(target_scores, null_scores)    # g(t), shape (n,)

    print(f"  target_scores : [{target_scores.min():.3f}, {target_scores.max():.3f}]"
          f" mean={target_scores.mean():.3f}")
    print(f"  null_scores   : [{null_scores.min():.3f}, {null_scores.max():.3f}]"
          f" mean={null_scores.mean():.3f}")
    print(f"  mix_max_scores: [{mix_max_scores.min():.3f}, {mix_max_scores.max():.3f}]"
          f" mean={mix_max_scores.mean():.3f}")
    print(f"  target wins   : {(target_scores > null_scores).mean():.3f}")

    # ── Unique null values and their counts ───────────────────────────────
    # We need: for each unique Zj value, 
    #   how many targets f(t) <= Zj  → P_F(Zj)
    #   how many mix_max g(t) <= Zj  → P_G(Zj)
    unique_z, counts_z = np.unique(null_scores, return_counts=True)
    n_unique_z = len(unique_z)
    n_obs      = len(null_scores)

    sorted_targets  = np.sort(target_scores)    # for searchsorted
    sorted_mixmax   = np.sort(mix_max_scores)   # for searchsorted

    # Empirical CDFs at each unique Zj
    # P(f(t) <= z)
    counts_f_leq_z = np.searchsorted(sorted_targets, unique_z, side='right')
    P_F_leq_z      = counts_f_leq_z / n

    # P(g(t) <= z)  where g(t) = mix_max
    counts_g_leq_z = np.searchsorted(sorted_mixmax, unique_z, side='right')
    P_G_leq_z      = counts_g_leq_z / n

    # R_j = [ P(f(t) <= Zj) / P(g(t) <= Zj) ]_[0,1]
    # This is the per-Zj contribution to FDP numerator
    R_j = np.divide(
        P_F_leq_z, P_G_leq_z,
        out=np.zeros_like(P_F_leq_z),
        where=P_G_leq_z > 0
    )
    R_j = np.clip(R_j, 0, 1)

    print(f"  R_j: min={R_j.min():.4f}, max={R_j.max():.4f}, mean={R_j.mean():.4f}")

    # ── Sweep thresholds from high to low ─────────────────────────────────
    # Thresholds = sorted target scores descending
    # At each threshold s_th:
    #   denominator = number of target scores >= s_th = i+1 (discoveries)
    #   numerator   = SUM_{Zj > s_th} R_j[j]
    
    all_thresholds = np.sort(target_scores)[::-1]   # sweep target scores
    fdr_values     = np.zeros(n)

    for i, s_th in enumerate(all_thresholds):
        D = i + 1   # number of discoveries (f(t) >= s_th)

        # Sum over Zj > s_th of R_j
        # unique_z is sorted → find first index where unique_z > s_th
        idx_start = np.searchsorted(unique_z, s_th, side='right')

        if idx_start >= n_unique_z:
            numerator = 0.0
        else:
            # Weight each R_j by how many null scores equal that unique_z value
            numerator = np.sum(R_j[idx_start:] * counts_z[idx_start:])

        fdr_values[i] = numerator / D if D > 0 else 0.0

    print(f"  fdr_values: min={fdr_values.min():.4f}, max={fdr_values.max():.4f}")
    print(f"  fdr_values first 5: {fdr_values[:5]}")

    # ── Monotone q-values ─────────────────────────────────────────────────
    q_values = np.minimum.accumulate(fdr_values[::-1])[::-1]

    print(f"  q_values: min={q_values.min():.4f}, max={q_values.max():.4f}")
    print(f"  unique q_values: {len(np.unique(q_values))}")

    # ── Map back to original order ────────────────────────────────────────
    sorted_idx = np.argsort(target_scores)[::-1]
    final_q    = np.zeros(n)
    final_q[sorted_idx] = q_values

    return final_q


def control_fdr_mixmax(model_scores, target_labels, decoy_scores):
    """Mix-Max FDR control. pi0=0 (no truly null samples)."""
    n_samples = len(model_scores)

    df = pd.DataFrame({
        'original_index'  : np.arange(n_samples),
        'label'           : target_labels,
        'max_model_score' : model_scores.max(axis=1),
        'max_decoy_score' : decoy_scores.max(axis=1),
        'predicted_class' : model_scores.argmax(axis=1),
        'pred_class_score': model_scores[np.arange(n_samples),
                                         model_scores.argmax(axis=1)],
    })

    # Mix-Max q-values
    q_mixmax = calculate_mixmax_qvalues(model_scores, decoy_scores)
    df['q_values_mixmax'] = q_mixmax

    # Ground Truth (oracle, uses labels)
    df_gt = df.sort_values('pred_class_score', ascending=False).reset_index(drop=True)
    num_incorrect = 0
    fdr_gt = []
    for idx, row in df_gt.iterrows():
        if row['predicted_class'] != row['label']:
            num_incorrect += 1
        fdr_gt.append(num_incorrect / (idx + 1))
    q_gt = np.array(fdr_gt)
    for i in range(len(q_gt) - 2, -1, -1):
        q_gt[i] = min(q_gt[i], q_gt[i + 1])
    df_gt['q_values_ground_truth'] = q_gt
    gt_map = dict(zip(df_gt['original_index'], df_gt['q_values_ground_truth']))
    df['q_values_ground_truth'] = df['original_index'].map(gt_map)

    return df


print("✓ Mix-Max (correct formula) loaded")

# ─────────────────────────────────────────────────────────────────────────────
# Accuracy-estimation helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pi0_storey_from_df(df, q_value_column):
    """Estimate π₀ from DataFrame (LABEL-FREE)."""
    if 'p_values' in df.columns:
        p_values = df['p_values'].dropna().values
        if len(p_values) > 0:
            lambda_range = np.arange(0.05, 0.95, 0.05)
            pi0_estimates = [np.mean(p_values > lam) / (1 - lam) for lam in lambda_range]
            return min(1.0, max(0, np.mean(pi0_estimates)))
    
    if 'pi0_storey' in df.columns:
        pi0 = df['pi0_storey'].iloc[0]
        if not np.isnan(pi0):
            return pi0
    
    # Mix-Max: pi0 = 0 by design
    if q_value_column == 'q_values_mixmax':
        return 0.0

    # Fallback: empirical estimate
    score_column = 'max_model_score' if 'max_model_score' in df.columns else 'model_score'
    if score_column not in df.columns:
        return 0.5
    
    df_sorted = df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    top_k = max(100, int(0.2 * len(df_sorted)))
    top_samples = df_sorted.head(top_k)
    is_correct = (top_samples['predicted_class'] == top_samples['label']).astype(int)
    return np.clip(1 - is_correct.mean(), 0.01, 0.99)


def compute_ground_truth_curve(df):
    """Compute Ground Truth curve (uses labels — oracle)."""
    score_column = 'max_model_score' if 'max_model_score' in df.columns else 'model_score'
    if score_column not in df.columns:
        return pd.DataFrame({'q_value': [0.05], 'accuracy': [0.5]})
    
    df_gt = df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    
    if 'q_values_ground_truth' in df_gt.columns:
        df_gt = df_gt[~df_gt['q_values_ground_truth'].isna()].copy()
        q_col = 'q_values_ground_truth'
    else:
        df_gt['is_correct'] = (df_gt['predicted_class'] == df_gt['label']).astype(int)
        num_incorrect = 0
        q_values_gt = []
        for idx in range(len(df_gt)):
            if df_gt.loc[idx, 'is_correct'] == 0:
                num_incorrect += 1
            q_values_gt.append(num_incorrect / (idx + 1))
        df_gt['q_values_ground_truth'] = q_values_gt
        q_col = 'q_values_ground_truth'
    
    df_gt['is_correct'] = (df_gt['predicted_class'] == df_gt['label']).astype(int)
    df_gt['n_discoveries'] = np.arange(1, len(df_gt) + 1)
    df_gt['TP_true'] = df_gt['is_correct'].cumsum()
    df_gt['FP_true'] = df_gt['n_discoveries'] - df_gt['TP_true']
    
    total_correct   = df_gt['is_correct'].sum()
    total_incorrect = len(df_gt) - total_correct
    total_samples   = len(df)
    
    df_gt['TN_true']       = total_incorrect - df_gt['FP_true']
    df_gt['FN_true']       = total_correct   - df_gt['TP_true']
    df_gt['Accuracy_true'] = (df_gt['TP_true'] + df_gt['TN_true']) / total_samples
    
    return df_gt[[q_col, 'Accuracy_true']].rename(
        columns={q_col: 'q_value_gt', 'Accuracy_true': 'accuracy_gt'})


def compute_method_estimation_curve(df, q_value_column, pi0):
    """Compute LABEL-FREE estimation curve with correct threshold-matched true accuracy."""
    total_samples = len(df)
    df_method = df[~df[q_value_column].isna()].copy()
    
    if len(df_method) == 0:
        return pd.DataFrame()
    
    # Sort by q-value ascending (best discoveries first)
    df_method = df_method.sort_values(by=q_value_column, ascending=True).reset_index(drop=True)
    df_method['n_discoveries'] = np.arange(1, len(df_method) + 1)
    
    # ── Label-free estimation (pi0=0 for Mix-Max) ─────────────────────────
    df_method['FP_est'] = df_method['n_discoveries'] * df_method[q_value_column]
    df_method['TP_est'] = df_method['n_discoveries'] - df_method['FP_est']
    df_method['TN_est'] = total_samples * pi0 - df_method['FP_est']
    df_method['FN_est'] = total_samples * (1 - pi0) - df_method['TP_est']
    
    df_method['FP_est'] = np.maximum(0, df_method['FP_est'])
    df_method['TP_est'] = np.maximum(0, df_method['TP_est'])
    df_method['TN_est'] = np.maximum(0, df_method['TN_est'])
    df_method['FN_est'] = np.maximum(0, df_method['FN_est'])
    
    df_method['Accuracy_est'] = (df_method['TP_est'] + df_method['TN_est']) / total_samples

    # ── True accuracy AT SAME threshold (uses labels) ─────────────────────
    # For each row i: true accuracy among the top i discoveries
    df_method['is_correct'] = (
        df_method['predicted_class'] == df_method['label']
    ).astype(int)
    
    cum_correct = df_method['is_correct'].cumsum()
    
    # True acc at threshold = correct among accepted / total samples
    # (same denominator as estimated acc)
    df_method['Accuracy_true_at_threshold'] = cum_correct / total_samples

    # Also compute standard cumulative accuracy among accepted only
    df_method['Accuracy_true_among_accepted'] = (
        cum_correct / df_method['n_discoveries']
    )

    # ── Error at each threshold ───────────────────────────────────────────
    df_method['error_at_threshold'] = np.abs(
        df_method['Accuracy_est'] - df_method['Accuracy_true_at_threshold']
    )

    return df_method[[
        q_value_column,
        'n_discoveries',
        'Accuracy_est',
        'Accuracy_true_at_threshold',
        'Accuracy_true_among_accepted',
        'error_at_threshold',
        'FP_est', 'TP_est', 'FN_est', 'TN_est',
        'is_correct',
    ]].rename(columns={q_value_column: 'q_value_method'})

