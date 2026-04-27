import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ScoreFeatureDataset(Dataset):
    """
    Dataset of (logit_scores, penultimate_features, null_decoy_scores, labels).

    Each item is a tuple:
      cnn_scores          : (C,)   classifier logit vector
      features            : (D,)   penultimate-layer feature vector
      target_decoy_scores : (C,)   null logit vector (for CNF training)
      labels              : ()     true class label
    """
    def __init__(self, cnn_scores, features, target_decoy_scores, labels):
        assert len(cnn_scores) == len(features), (
            f"cnn_scores ({len(cnn_scores)}) and features ({len(features)}) length mismatch")
        self.cnn_scores          = cnn_scores
        self.features            = features
        self.target_decoy_scores = target_decoy_scores
        self.labels              = labels

    def __len__(self):
        return len(self.cnn_scores)

    def __getitem__(self, idx):
        return (self.cnn_scores[idx], self.features[idx],
                self.target_decoy_scores[idx], self.labels[idx])


def create_score_feature_dataset(dataset, cnn_model, negative_scores_pools, device='cuda'):
    """
    Build a ScoreFeatureDataset by running cnn_model over dataset.

    The null vector for each sample is identical to the model's logit vector
    except that the true-class coordinate is replaced by a random draw from
    the corresponding negative score pool.
    """
    import torch.nn.functional as F

    cnn_scores_list, features_list, target_decoy_list, labels_list = [], [], [], []
    cnn_model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Creating score dataset', leave=False):
            images   = images.to(device)
            features = cnn_model.get_features(images)
            scores   = cnn_model.linear2(features)

            scores_cpu   = scores.cpu()
            features_cpu = features.cpu()
            null_vectors = scores_cpu.clone()

            for i, label in enumerate(labels):
                label_val = label.item()
                neg_pool  = negative_scores_pools[label_val]
                if len(neg_pool) > 0:
                    null_vectors[i, label_val] = torch.tensor(
                        np.random.choice(neg_pool), dtype=null_vectors.dtype)

            cnn_scores_list.append(scores_cpu)
            features_list.append(features_cpu)
            target_decoy_list.append(null_vectors)
            labels_list.append(labels)

    return ScoreFeatureDataset(
        torch.cat(cnn_scores_list),
        torch.cat(features_list),
        torch.cat(target_decoy_list),
        torch.cat(labels_list),
    )


def create_score_feature_dataset_bcss(data, device='cpu'):
    """
    Build a ScoreFeatureDataset from BCSS pre-computed logits and features.

    data['total_preds'][c]    — tensor [C, N_c]: logits for pixels of class c
    data['total_features'][c] — tensor [D, N_c]: features for pixels of class c

    The null score for each pixel replaces the true-class coordinate with a
    random draw from negative-class logits of that class.
    """
    total_preds    = data['total_preds']
    total_features = data['total_features']
    classes        = sorted(total_preds.keys())
    num_classes    = len(classes)

    negative_scores_pools = {}
    for c in classes:
        neg_scores = [total_preds[other_c][c] for other_c in classes if other_c != c]
        negative_scores_pools[c] = torch.cat(neg_scores).cpu().numpy()

    sc_list, ft_list, dc_list, lb_list = [], [], [], []

    for c in classes:
        preds = total_preds[c].T     # [N_c, C]
        feats = total_features[c].T  # [N_c, D]
        N_c   = preds.shape[0]

        labels           = torch.full((N_c,), c, dtype=torch.long)
        target_decoy     = preds.clone()
        neg_pool         = negative_scores_pools[c]

        if len(neg_pool) == 0:
            raise RuntimeError(f"No negative scores for class {c}")

        target_decoy[:, c] = torch.tensor(
            np.random.choice(neg_pool, size=N_c), dtype=target_decoy.dtype)

        sc_list.append(preds)
        ft_list.append(feats)
        dc_list.append(target_decoy)
        lb_list.append(labels)

    return ScoreFeatureDataset(
        torch.cat(sc_list), torch.cat(ft_list),
        torch.cat(dc_list), torch.cat(lb_list),
    )


def create_score_feature_dataset_bcss_v2(data, pool_score, pool_vectors,
                                          strategy='score_coord', device='cpu'):
    """
    Build a ScoreFeatureDataset from BCSS pre-computed data using error-conditioned pools.

    Requires pool_score and pool_vectors from build_error_conditioned_pools_bcss().
    strategy: 'score_coord' — replace only the predicted-class coordinate.
    """
    from data_processing.negative_scores_pool import _build_decoy_score_coord

    assert strategy == 'score_coord', f"Unknown strategy: {strategy}"

    total_preds    = data['total_preds']
    total_features = data['total_features']
    classes        = sorted(total_preds.keys())

    sc_list, ft_list, dc_list, lb_list = [], [], [], []
    rng = np.random.default_rng(0)

    for c in classes:
        preds_c  = total_preds[c].T.cpu()    # [N_c, C]
        feats_c  = total_features[c].T.cpu() # [N_c, D]
        N_c      = preds_c.shape[0]
        labels_c = torch.full((N_c,), c, dtype=torch.long)
        dc_np    = _build_decoy_score_coord(preds_c.numpy(), pool_score, rng)

        sc_list.append(preds_c)
        ft_list.append(feats_c)
        dc_list.append(torch.from_numpy(dc_np).float())
        lb_list.append(labels_c)

    return ScoreFeatureDataset(
        torch.cat(sc_list), torch.cat(ft_list),
        torch.cat(dc_list), torch.cat(lb_list),
    )


def create_score_feature_dataset_bcss_no_decoys(data, device='cpu'):
    """
    Build a ScoreFeatureDataset from BCSS data without null replacement.
    Used for CNF inference (decoy slot is filled with the original scores).
    """
    total_preds    = data['total_preds']
    total_features = data['total_features']
    classes        = sorted(total_preds.keys())

    sc_list, ft_list, lb_list = [], [], []
    for c in classes:
        preds_c  = total_preds[c].T.cpu()
        feats_c  = total_features[c].T.cpu()
        N_c      = preds_c.shape[0]
        sc_list.append(preds_c)
        ft_list.append(feats_c)
        lb_list.append(torch.full((N_c,), c, dtype=torch.long))

    all_scores = torch.cat(sc_list)
    return ScoreFeatureDataset(
        all_scores, torch.cat(ft_list),
        all_scores.clone(),  # placeholder decoys
        torch.cat(lb_list),
    )
