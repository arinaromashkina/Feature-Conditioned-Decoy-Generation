import numpy as np
from tqdm import tqdm

MIN_POOL = 50


def collect_negative_scores(model, train_dataset, num_classes=10, device='cuda'):
    """
    Collect per-class pools of negative logit scores from the training set.
    For class k, the pool contains score[k] for all samples with label != k.
    """
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    negative_scores_pools = {i: [] for i in range(num_classes)}
    model.eval()
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting negative scores'):
            images, labels = images.to(device), labels.to(device)
            features = model.get_features(images)
            scores   = model.linear2(features)

            for class_idx in range(num_classes):
                neg_mask = labels != class_idx
                if neg_mask.sum() > 0:
                    negative_scores_pools[class_idx].append(
                        scores[neg_mask, class_idx].cpu())

    for class_idx in range(num_classes):
        if negative_scores_pools[class_idx]:
            import torch
            negative_scores_pools[class_idx] = \
                torch.cat(negative_scores_pools[class_idx]).numpy()
        else:
            negative_scores_pools[class_idx] = np.array([])

    return negative_scores_pools


def build_error_conditioned_pools(train_scores: np.ndarray,
                                   train_labels: np.ndarray,
                                   num_classes:  int,
                                   verbose:      bool = True) -> tuple:
    """
    Build two types of decoy pools:

    pool_score[c]   = score_c on samples where argmax == c AND label != c
                      (used by 'score_coord' strategy; falls back to label != c
                       if fewer than MIN_POOL errors exist)

    pool_vectors[c] = full score vectors for samples where label != c

    Under Mix-Max with pi0 = 0:
      f(t) = model_scores[i, c_hat]  — target score
      Z(t) ~ pool_score[c_hat]       — decoy score
      Under H_0 (incorrect prediction), f(t) and Z(t) should be exchangeable.
    """
    pred_classes = train_scores.argmax(axis=1)
    pool_score   = {}
    pool_vectors = {}

    if verbose:
        acc = (pred_classes == train_labels).mean()
        print(f"\nBuilding error-conditioned decoy pools (train acc={acc:.4f})")

    for c in range(num_classes):
        error_mask = (pred_classes == c) & (train_labels != c)
        n_err      = error_mask.sum()

        if n_err >= MIN_POOL:
            pool_score[c] = train_scores[error_mask, c]
            src = "per-class errors (argmax=c & label!=c)"
        else:
            neg_mask      = train_labels != c
            pool_score[c] = train_scores[neg_mask, c]
            src = "fallback (label!=c)"

        pool_vectors[c] = train_scores[train_labels != c]

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
    Score-coordinate decoy strategy:
      decoy[i, c_hat] ~ pool_score[c_hat]
      decoy[i, j]      = score[i, j]  for j != c_hat

    Approximates the null distribution while preserving the multivariate
    correlation structure of the logit vector.
    """
    pred_classes = sc_np.argmax(axis=1)
    dc_np        = sc_np.copy()
    for c in range(sc_np.shape[1]):
        mask = pred_classes == c
        if not mask.any():
            continue
        pool = pool_score.get(c, [])
        if len(pool) == 0:
            continue
        dc_np[mask, c] = rng.choice(pool, size=mask.sum(), replace=True)
    return dc_np


def build_error_conditioned_pools_bcss(data: dict, verbose: bool = True) -> tuple:
    """
    Variant of build_error_conditioned_pools for BCSS pre-computed logits.

    data['total_preds'][c]    — tensor [C, N_c]: logits for pixels of class c
    data['total_features'][c] — tensor [D, N_c]: features for pixels of class c

    Returns (pool_score, pool_vectors, train_scores [N, C], train_labels [N]).
    """
    total_preds = data['total_preds']
    classes     = sorted(total_preds.keys())
    num_classes = len(classes)

    all_scores_list, all_labels_list = [], []
    for c in classes:
        preds_c  = total_preds[c].T.cpu().numpy()
        N_c      = preds_c.shape[0]
        all_scores_list.append(preds_c)
        all_labels_list.append(np.full(N_c, c, dtype=np.int64))

    train_scores = np.concatenate(all_scores_list, axis=0)
    train_labels = np.concatenate(all_labels_list, axis=0)

    pool_score, pool_vectors = build_error_conditioned_pools(
        train_scores, train_labels, num_classes, verbose=verbose)

    return pool_score, pool_vectors, train_scores, train_labels
