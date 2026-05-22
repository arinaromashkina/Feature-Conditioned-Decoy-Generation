import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from robustness.tools.breeds_helpers import make_entity13

from data_processing.score_feature_dataset import ScoreFeatureDataset
from data_processing.negative_scores_pool import (
    build_error_conditioned_pools, _build_decoy_score_coord,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR        = "/home/arina/imagenet"
BATCH_SIZE      = 64
DEVICE          = 'cuda:2' if torch.cuda.is_available() else 'cpu'
FEATURE_DIM     = 640
CLASSIFIER_PATH = 'breeds_entity13_classifier.pth'
OUT_DIR         = 'results_ds/entity13_clean'
SEED            = 42

os.makedirs(OUT_DIR, exist_ok=True)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seeds(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading — clean ImageNet val only, no corruptions
# ─────────────────────────────────────────────────────────────────────────────

hierarchy_dir        = f"{DATA_DIR}/imagenet_class_hierarchy"
ret                  = make_entity13(hierarchy_dir, split='good')
source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])
target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])
NUM_CLASSES          = len(ret[1][0])

print(f"Entity-13: {NUM_CLASSES} source classes, {len(ret[1][1])} target classes")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4717, 0.4499, 0.3837],
                         [0.2600, 0.2516, 0.2575]),
])

trainset = folder.ImageFolder(
    root=f"{DATA_DIR}/imagenetv1/train/",
    transform=transform, label_mapping=source_label_mapping,
)
idx = np.arange(len(trainset))
np.random.seed(SEED)
np.random.shuffle(idx)
train_idx, val_idx = idx[:-10000], idx[-10000:]
train_subset = torch.utils.data.Subset(trainset, train_idx)
val_subset   = torch.utils.data.Subset(trainset, val_idx)

trainloader = torch.utils.data.DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
valloader   = torch.utils.data.DataLoader(
    val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

clean_val_source = folder.ImageFolder(
    root=f"{DATA_DIR}/imagenetv1/val/",
    transform=transform, label_mapping=source_label_mapping,
)
clean_val_target = folder.ImageFolder(
    root=f"{DATA_DIR}/imagenetv1/val/",
    transform=transform, label_mapping=target_label_mapping,
)

print(f"Train: {len(train_subset)}  Val: {len(val_subset)}")
print(f"Clean val source: {len(clean_val_source)}  target: {len(clean_val_target)}")

# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class BREEDSClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, feature_dim=640):
        super().__init__()
        backbone      = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.linear1  = nn.Linear(2048, feature_dim)
        self.linear2  = nn.Linear(feature_dim, num_classes)

    def get_features(self, x):
        return self.linear1(self.backbone(x).flatten(1))

    def forward(self, x):
        return self.linear2(F.relu(self.get_features(x)))


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_classifier(model, trainloader, valloader, epochs=30, lr=0.01,
                     device='cuda', save_path='classifier.pth', patience=5):
    model     = model.to(device)
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, no_improve = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(trainloader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)
        scheduler.step()
        val_acc = evaluate_accuracy(model, valloader, device)
        print(f"Epoch {epoch:3d} | loss={total_loss/total:.4f} "
              f"| train={correct/total:.4f} | val={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}")
                break
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    print(f"Training done. Best val_acc={best_acc:.4f}")


model = BREEDSClassifier(num_classes=NUM_CLASSES, pretrained=True,
                          feature_dim=FEATURE_DIM).to(DEVICE)

if os.path.exists(CLASSIFIER_PATH):
    model.load_state_dict(
        torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False))
    print(f"Classifier loaded from {CLASSIFIER_PATH}")
else:
    print("Training classifier from scratch...")
    set_seeds(SEED)
    train_classifier(model, trainloader, valloader, epochs=30, lr=0.01,
                     device=DEVICE, save_path=CLASSIFIER_PATH, patience=5)

model.eval()
print(f"Val acc (source): {evaluate_accuracy(model, valloader, DEVICE):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Collect train scores and build decoy pools
# ─────────────────────────────────────────────────────────────────────────────

def collect_scores(model, dataset, batch_size=256, device='cuda'):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    sc_list, lb_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting scores'):
            feats  = model.get_features(images.to(device))
            scores = model.linear2(F.relu(feats))
            sc_list.append(scores.cpu().numpy())
            lb_list.append(labels.numpy())
    return np.concatenate(sc_list), np.concatenate(lb_list)


def build_score_dataset(dataset, model, pool_score, pool_vectors,
                        device='cuda', batch_size=256):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    sc_list, ft_list, dc_list, lb_list = [], [], [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Building score dataset', leave=False):
            feats  = model.get_features(images.to(device))
            scores = model.linear2(F.relu(feats))
            sc_np  = scores.cpu().numpy()
            dc_np  = _build_decoy_score_coord(sc_np, pool_score, rng)
            sc_list.append(scores.cpu())
            ft_list.append(feats.cpu())
            dc_list.append(torch.from_numpy(dc_np).float())
            lb_list.append(labels)
    return ScoreFeatureDataset(
        torch.cat(sc_list), torch.cat(ft_list),
        torch.cat(dc_list), torch.cat(lb_list),
    )


print("\nCollecting train scores...")
train_scores, train_labels = collect_scores(
    model, train_subset, batch_size=256, device=DEVICE)
print(f"  shape={train_scores.shape}  "
      f"acc={(train_scores.argmax(1) == train_labels).mean():.4f}")

print("Building decoy pools...")
pool_score, pool_vectors = build_error_conditioned_pools(
    train_scores, train_labels, NUM_CLASSES, verbose=True)

# ─────────────────────────────────────────────────────────────────────────────
# Build score datasets for clean test sets and save npz
# ─────────────────────────────────────────────────────────────────────────────

def save_npz(score_ds, path):
    np.savez(
        path,
        logits=score_ds.cnn_scores.numpy(),
        labels=score_ds.labels.numpy(),
        decoys=score_ds.target_decoy_scores.numpy(),
    )
    print(f"  Saved {len(score_ds)} samples → {path}.npz")


for split_name, dataset in [
    ('imagenet_val_source', clean_val_source),
    ('imagenet_val_target', clean_val_target),
]:
    print(f"\nProcessing {split_name} (N={len(dataset)})...")
    ds = build_score_dataset(dataset, model, pool_score, pool_vectors, device=DEVICE)
    save_npz(ds, os.path.join(OUT_DIR, split_name))

print(f"\nDone. Files saved to {OUT_DIR}/")
