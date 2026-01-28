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
    null_distribution = train_cnn_scores[neg_train_mask, target_class]
    test_scores = test_cnn_scores[:, target_class]
    p_values = empirical_p_values(null_distribution, test_scores)
    q_values = benjamini_hochberg_storey(np.sort(p_values))
    n_discoveries = (q_values <= fdr_level).sum()
    return n_discoveries, q_values, p_values


def control_fdr(model_scores, target_labels, decoys,
                train_cnn_scores=None, train_labels=None,
                test_cnn_scores=None, target_class=None):
    df = pd.DataFrame({
        'model_score': model_scores,
        'label': target_labels,
        'decoy_score': decoys
    })
    df = df.sort_values(by='model_score', ascending=False, inplace=False).reset_index(drop=True)
    df['fdr'] = 0.0
    num_neg = 0
    num_pos = 0

    for index, row in df.iterrows():
        if row['label'] == 0:
            num_neg += 1
        else:
            num_pos += 1
        df.at[index, 'fdr'] = (num_neg + 1.0) / (num_pos + num_neg)


    df['q_values_ground_truth'] = df['fdr'].copy()
    prev = 1
    for index in range(len(df) - 1, -1, -1):
        df.loc[index, 'q_values_ground_truth'] = min(prev, df.loc[index, 'q_values_ground_truth'])
        prev = df.loc[index, 'q_values_ground_truth']

    df['max_score'] = df[['model_score', 'decoy_score']].max(axis=1)
    df = df.sort_values(by='max_score', ascending=False).reset_index(drop=True)

    df['fdr'] = 0.0

    num_neg = 0
    num_pos = 0

    for index, row in df.iterrows():
        if row['model_score'] < row['decoy_score']:
            num_neg += 1
        else:
            num_pos += 1
        df.loc[index, 'fdr'] = (2 * num_neg + 1.0) / (num_pos + num_neg)

    df['q_values_tdc'] = df['fdr'].copy()
    prev = 1.0
    for index in range(len(df) - 1, -1, -1):
        df.loc[index, 'q_values_tdc'] = min(prev, df.loc[index, 'q_values_tdc'])
        prev = df.loc[index, 'q_values_tdc']

    df['p_value'] = empirical_p_values(np.sort(decoys), model_scores)

    df['q_values_bh_storey'] = benjamini_hochberg_storey(np.sort(df['p_value']))
    df['q_values_bh_fixed'] = benjamini_hochberg_fixed(np.sort(df['p_value']))

    if (train_cnn_scores is not None and train_labels is not None and
        test_cnn_scores is not None and target_class is not None):
        n_discoveries_neg, q_values_neg, p_values_neg = negative_training_benchmark(
            train_cnn_scores, train_labels, test_cnn_scores, target_class
        )
        df['q_values_negative_training'] = q_values_neg
        df['p_values_negative_training'] = p_values_neg
    return df