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

MIN_POOL = 50

def create_score_feature_dataset(dataset, cnn_model, negative_scores_pools,
                                  device='cuda'):
    """
    Create dataset where:
      cnn_scores    = original logit vector  (10,)
      features      = CNN penultimate features (640,)
      target_decoy  = null score vector:
                        same as cnn_scores EXCEPT true-class score
                        is replaced by a random negative score.
                        This teaches the flow what the score vector looks like
                        when the sample does NOT belong to its true class.
      labels        = true class
    """
    cnn_scores_list    = []
    features_list      = []
    target_decoy_list  = []
    labels_list        = []

    cnn_model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for images, labels in tqdm(loader,
                                   desc='Creating score dataset', leave=False):
            images   = images.to(device)
            features = cnn_model.get_features(images)
            #scores   = cnn_model.linear2(F.relu(cnn_model.linear1(features)))
            scores = cnn_model.linear2(features)

            scores_cpu   = scores.cpu()
            features_cpu = features.cpu()

            # Null vector: replace true-class score with negative score
            null_vectors = scores_cpu.clone()
            for i, label in enumerate(labels):
                label_val = label.item()
                neg_pool  = negative_scores_pools[label_val]
                if len(neg_pool) > 0:
                    neg_score = np.random.choice(neg_pool)
                    null_vectors[i, label_val] = torch.tensor(
                        neg_score, dtype=null_vectors.dtype)

            cnn_scores_list.append(scores_cpu)
            features_list.append(features_cpu)
            target_decoy_list.append(null_vectors)
            labels_list.append(labels)

    class ScoreDataset:
        def __init__(self, scores, features, decoys, labels):
            self.cnn_scores          = scores
            self.features            = features
            self.target_decoy_scores = decoys
            self.labels              = labels

        def __len__(self):
            return len(self.cnn_scores)

        def __getitem__(self, idx):
            return (self.cnn_scores[idx], self.features[idx],
                    self.target_decoy_scores[idx], self.labels[idx])

    return ScoreDataset(
        torch.cat(cnn_scores_list),
        torch.cat(features_list),
        torch.cat(target_decoy_list),
        torch.cat(labels_list),
    )


class ScoreFeatureDataset(Dataset):
    def __init__(self, cnn_scores, features, target_decoy_scores, labels):
        self.cnn_scores = cnn_scores
        self.target_decoy_scores = target_decoy_scores
        self.features = features
        self.labels = labels

        if len(cnn_scores) != len(features):
            print(f"Warning: cnn_scores ({len(cnn_scores)}) and features ({len(features)}) length mismatch")

    def __len__(self):
        return len(self.cnn_scores)

    def __getitem__(self, idx):
        return (
            self.cnn_scores[idx],
            self.features[idx],
            self.target_decoy_scores[idx],
            self.labels[idx]
        )

def create_score_feature_dataset_(dataset, cnn_model, negative_scores_pools, bool_multiclass=True, device='cuda'):
    cnn_scores_list = []
    features_list = []
    target_decoy_list = []
    labels_list = []
    cnn_model.eval()
    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            features = cnn_model.get_features(images)
            scores = cnn_model.fc2(features)
            scores_cpu = scores.cpu()
            features_cpu = features.cpu()
            target_decoy_scores = scores_cpu.clone()
            for i, label in enumerate(labels):
                label_val = label.item()
                if bool_multiclass:
                    neg_pool = negative_scores_pools[label_val]
                else:
                    neg_pool = negative_scores_pools
                if len(neg_pool) > 0:
                    random_neg_score = np.random.choice(neg_pool)
                    target_decoy_scores[i, label_val] = torch.tensor(
                                                          random_neg_score,
                                                          dtype=target_decoy_scores.dtype
                                                      )

            cnn_scores_list.append(scores_cpu)
            features_list.append(features_cpu)
            target_decoy_list.append(target_decoy_scores)
            labels_list.append(labels)

    all_cnn_scores = torch.cat(cnn_scores_list)
    all_features = torch.cat(features_list)
    all_target_decoy = torch.cat(target_decoy_list)
    all_labels = torch.cat(labels_list)
    return ScoreFeatureDataset(all_cnn_scores, all_features, all_target_decoy, all_labels)



def create_score_feature_dataset_bcss(
    data,
    bool_multiclass=True,
    device='cpu'
):
    cnn_scores_list = []
    features_list = []
    target_decoy_list = []
    labels_list = []

    total_preds = data['total_preds']
    total_features = data['total_features']
    classes = sorted(total_preds.keys())
    num_classes = len(classes)
    negative_scores_pools = {}

    for c in classes:
        neg_scores = []
        for other_c in classes:
            if other_c == c:
                continue
            neg_scores.append(total_preds[other_c][c]) 
        negative_scores_pools[c] = torch.cat(neg_scores).cpu().numpy()

    for c in classes:
        preds = total_preds[c]           
        feats = total_features[c]         
        preds = preds.T                  
        feats = feats.T                   

        N_c = preds.shape[0]

        labels = torch.full((N_c,), c, dtype=torch.long)

        target_decoy_scores = preds.clone()

        neg_pool = negative_scores_pools[c]

        if len(neg_pool) == 0:
            raise RuntimeError(f"No negative scores for class {c}")

        random_neg_scores = np.random.choice(neg_pool, size=N_c)
        target_decoy_scores[:, c] = torch.tensor(
            random_neg_scores,
            dtype=target_decoy_scores.dtype
        )

        cnn_scores_list.append(preds)
        features_list.append(feats)
        target_decoy_list.append(target_decoy_scores)
        labels_list.append(labels)

    all_cnn_scores = torch.cat(cnn_scores_list, dim=0)
    all_features = torch.cat(features_list, dim=0)
    all_target_decoy = torch.cat(target_decoy_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return ScoreFeatureDataset(
        all_cnn_scores,
        all_features,
        all_target_decoy,
        all_labels
    )

def build_error_conditioned_pools(train_scores: np.ndarray,
                                   train_labels: np.ndarray,
                                   num_classes:  int,
                                   verbose:      bool = True) -> dict:
    """
    Строит два типа пулов для двух стратегий decoy:

    pool_score[c] = score_c на примерах где argmax=c И label≠c
                    (для стратегий 'score_coord' и 'pool_replace')

    pool_vectors[c] = полные score-векторы примеров где label≠c
                      (для стратегии 'full_vector')

    Если ошибок < MIN_POOL — fallback на label≠c (только для pool_score).

    Логика Mix-Max (π_mm = 0):
      f(t) = model_scores[i, ĉ_i]   — target score
      Z(t) ~ pool_score[ĉ_i]        — decoy score
      Под H₀ (ошибка): f(t) и Z(t) должны быть exchangeable.
    """
    pred_classes = train_scores.argmax(axis=1)
    pool_score   = {}
    pool_vectors = {}

    if verbose:
        acc = (pred_classes == train_labels).mean()
        print(f"\nBuilding error-conditioned decoy pools  "
              f"(train acc={acc:.4f})")

    for c in range(num_classes):
        # ── pool_score: coord c при ошибке ───────────────────────────────
        error_mask = (pred_classes == c) & (train_labels != c)
        n_err      = error_mask.sum()

        if n_err >= MIN_POOL:
            pool_score[c] = train_scores[error_mask, c]
            src = f"per-class errors (argmax=c & label≠c)"
        else:
            neg_mask      = train_labels != c
            pool_score[c] = train_scores[neg_mask, c]
            src = f"fallback (label≠c)"

        # ── pool_vectors: полные векторы при label≠c ─────────────────────
        neg_mask         = train_labels != c
        pool_vectors[c]  = train_scores[neg_mask]   # shape [M, C]

        if verbose:
            p = pool_score[c]
            print(f"  class {c}: n_score={len(p):6d}  "
                  f"n_vec={pool_vectors[c].shape[0]:6d}  "
                  f"[{p.min():.3f}, {p.max():.3f}]  "
                  f"mean={p.mean():.3f}  n_errors={n_err:4d}  src={src}")

    return pool_score, pool_vectors

def _build_decoy_score_coord(sc_np: np.ndarray,
                              pool_score: dict,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Стратегия A: заменяем только координату pred_class.
    decoy[i, ĉ_i] ~ pool_score[ĉ_i]
    decoy[i, j]   = score[i, j]  для j ≠ ĉ_i
    """
    pred_classes = sc_np.argmax(axis=1)
    dc_np        = sc_np.copy()
    for c in range(sc_np.shape[1]):
        mask = pred_classes == c
        if not mask.any():
            continue
        pool = pool_score[c]
        if len(pool) == 0:
            continue
        dc_np[mask, c] = rng.choice(pool, size=mask.sum(), replace=True)
    return dc_np

# ============================================================
# BCSS: BUILD ERROR-CONDITIONED POOLS
# ============================================================

def build_error_conditioned_pools_bcss(
    data:        dict,
    verbose:     bool = True
) -> tuple:
    """
    Аналог build_error_conditioned_pools для BCSS.

    data['total_preds'][c]    — тензор [C, N_c]: логиты для примеров класса c
    data['total_features'][c] — тензор [D, N_c]: признаки для примеров класса c

    pool_score[c]   = score_c на примерах где argmax=c И label≠c
                      (fallback: label≠c если ошибок < MIN_POOL)
    pool_vectors[c] = полные score-векторы примеров где label≠c  [M, C]
    """
    total_preds = data['total_preds']
    classes     = sorted(total_preds.keys())
    num_classes = len(classes)

    # Собираем все скоры и метки в единые массивы
    # preds[c] shape: [C, N_c] → нам нужно [N_c, C]
    all_scores_list  = []
    all_labels_list  = []

    for c in classes:
        preds_c = total_preds[c].T.cpu().numpy()   # [N_c, C]
        N_c     = preds_c.shape[0]
        labels_c = np.full(N_c, c, dtype=np.int64)
        all_scores_list.append(preds_c)
        all_labels_list.append(labels_c)

    train_scores = np.concatenate(all_scores_list, axis=0)  # [N, C]
    train_labels = np.concatenate(all_labels_list, axis=0)  # [N]

    if verbose:
        pred_classes = train_scores.argmax(axis=1)
        acc = (pred_classes == train_labels).mean()
        print(f"\nBuilding error-conditioned decoy pools  "
              f"(train acc={acc:.4f})")

    pool_score, pool_vectors = build_error_conditioned_pools(
        train_scores, train_labels, num_classes, verbose=verbose
    )

    return pool_score, pool_vectors, train_scores, train_labels


# ============================================================
# BCSS: CREATE SCORE FEATURE DATASET
# ============================================================

def create_score_feature_dataset_bcss_v2(
    data:         dict,
    pool_score:   dict,
    pool_vectors: dict,
    strategy:     str  = 'score_coord',
    device:       str  = 'cpu'
) -> ScoreFeatureDataset:
    """
    Аналог create_score_dataset_with_decoys для BCSS.

    Данные уже предвычислены — не нужно гнать батчи через модель.

    strategy:
      'score_coord'  — заменяем только coord pred_class (Strategy A)
      'full_vector'  — заменяем весь вектор (Strategy B)
      'pool_replace' — как score_coord (Strategy C)
    """
    assert strategy in ('score_coord', 'full_vector', 'pool_replace'), \
        f"Unknown strategy: {strategy}"

    total_preds    = data['total_preds']
    total_features = data['total_features']
    classes        = sorted(total_preds.keys())

    sc_list, ft_list, dc_list, lb_list = [], [], [], []
    rng = np.random.default_rng(0)

    for c in classes:
        preds_c = total_preds[c].T.cpu()       # [N_c, C]
        feats_c = total_features[c].T.cpu()    # [N_c, D]
        N_c     = preds_c.shape[0]

        labels_c = torch.full((N_c,), c, dtype=torch.long)
        sc_np    = preds_c.numpy()

        # ── строим decoy вектор ──────────────────────────────────────────
        if strategy == 'score_coord':
            dc_np = _build_decoy_score_coord(sc_np, pool_score, rng)

        dc = torch.from_numpy(dc_np).float()

        sc_list.append(preds_c)
        ft_list.append(feats_c)
        dc_list.append(dc)
        lb_list.append(labels_c)

    return ScoreFeatureDataset(
        torch.cat(sc_list,  dim=0),
        torch.cat(ft_list,  dim=0),
        torch.cat(dc_list,  dim=0),
        torch.cat(lb_list,  dim=0),
    )


# ============================================================
# BCSS: NO DECOYS (для инференса)
# ============================================================

def create_score_feature_dataset_bcss_no_decoys(
    data:   dict,
    device: str = 'cpu'
) -> ScoreFeatureDataset:
    """
    Без замены decoys (placeholder = scores).
    Используется при инференсе Flow.
    """
    total_preds    = data['total_preds']
    total_features = data['total_features']
    classes        = sorted(total_preds.keys())

    sc_list, ft_list, lb_list = [], [], []

    for c in classes:
        preds_c  = total_preds[c].T.cpu()     # [N_c, C]
        feats_c  = total_features[c].T.cpu()  # [N_c, D]
        N_c      = preds_c.shape[0]
        labels_c = torch.full((N_c,), c, dtype=torch.long)

        sc_list.append(preds_c)
        ft_list.append(feats_c)
        lb_list.append(labels_c)

    all_scores = torch.cat(sc_list, dim=0)
    all_feats  = torch.cat(ft_list, dim=0)
    all_labels = torch.cat(lb_list, dim=0)

    return ScoreFeatureDataset(
        all_scores,
        all_feats,
        all_scores.clone(),   # placeholder
        all_labels,
    )