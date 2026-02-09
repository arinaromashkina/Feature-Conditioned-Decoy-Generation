import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
import numpy as np

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

def create_score_feature_dataset(dataset, cnn_model, negative_scores_pools, bool_multiclass=True, device='cuda'):
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