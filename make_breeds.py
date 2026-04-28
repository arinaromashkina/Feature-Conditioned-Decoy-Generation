import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from robustness.tools.breeds_helpers import (
    make_living17, make_entity13, make_entity30, make_nonliving26,
)

from data_processing.score_feature_dataset import ScoreFeatureDataset
from data_processing.negative_scores_pool import (
    build_error_conditioned_pools, _build_decoy_score_coord,
)
from flows.flow_FN import ScoreShiftFlowWrapper
from utils.other_methods import (
    predict_ATC_maxconf, predict_ATC_negent,
    predict_AC, predict_DOC, calibration_temp,
)
from fdr.fdr_control import *
from fdr.plot_fdr import *

matplotlib.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   16,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  10,
    'figure.titlesize': 18,
})

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_C = [
    "fog", "frost", "motion_blur", "brightness", "zoom_blur",
    "snow", "defocus_blur", "glass_blur", "gaussian_noise",
    "shot_noise", "impulse_noise", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression", "speckle_noise", "spatter",
    "gaussian_blur", "saturate",
]
SEVERITIES = [1, 2, 3, 4, 5]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def get_breeds_ret(hierarchy_dir: str, name: str):
    """Return ret = make_*(hierarchy_dir, split='good') for the chosen subset."""
    makers = {
        'living17':   make_living17,
        'entity13':   make_entity13,
        'entity30':   make_entity30,
        'nonliving26': make_nonliving26,
    }
    if name not in makers:
        raise ValueError(f"Unknown BREEDS name: {name}. "
                         f"Choose from: {list(makers)}")
    return makers[name](hierarchy_dir, split='good')


def get_imagenet_breeds(batch_size, data_dir, name='living17'):
    hierarchy_dir = f"{data_dir}/imagenet_class_hierarchy"
    ret = get_breeds_ret(hierarchy_dir, name)

    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4717, 0.4499, 0.3837],
                             [0.2600, 0.2516, 0.2575]),
    ])

    trainset = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform, label_mapping=source_label_mapping,
    )
    targetset = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform, label_mapping=target_label_mapping,
    )

    idx = np.arange(len(trainset))
    np.random.seed(42)
    np.random.shuffle(idx)
    train_idx, val_idx = idx[:-10000], idx[-10000:]

    train_subset = torch.utils.data.Subset(trainset, train_idx)
    val_subset   = torch.utils.data.Subset(trainset, val_idx)

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4)

    testsets, testloaders = [], []

    def add_loader(ds):
        testsets.append(ds)
        testloaders.append(torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4))

    add_loader(val_subset)   # [0] val source
    add_loader(targetset)    # [1] val target (full train)
    add_loader(folder.ImageFolder(f"{data_dir}/imagenetv1/val/",
               transform=transform, label_mapping=source_label_mapping))  # [2]
    add_loader(folder.ImageFolder(f"{data_dir}/imagenetv1/val/",
               transform=transform, label_mapping=target_label_mapping))  # [3]

    print(f"\n  Loading corruptions (source)...")
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            path = f"{data_dir}/imagenet-c/{corruption}/{severity}"
            if os.path.isdir(path):
                add_loader(folder.ImageFolder(root=path, transform=transform,
                                              label_mapping=source_label_mapping))
            else:
                print(f"    Skipping (not found): {path}")

    print(f"  Loading corruptions (target)...")
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            path = f"{data_dir}/imagenet-c/{corruption}/{severity}"
            if os.path.isdir(path):
                add_loader(folder.ImageFolder(root=path, transform=transform,
                                              label_mapping=target_label_mapping))
            else:
                print(f"    Skipping (not found): {path}")

    print(f"\nBREEDS '{name}' loaded.")
    print(f"  Train: {len(train_subset)}  Val: {len(val_subset)}")
    print(f"  Source classes: {len(ret[1][0])}  Target classes: {len(ret[1][1])}")
    print(f"  Total test sets: {len(testsets)}")
    return trainset, train_subset, val_subset, trainloader, testsets, testloaders


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class BREEDSClassifier(nn.Module):
    """
    ResNet-50 (ImageNet pretrained) adapted for BREEDS classification.

    Head: backbone → avgpool → flatten [2048]
              → linear1 [2048 → feature_dim]
              → ReLU
              → linear2 [feature_dim → num_classes]

    get_features() returns the feature_dim-dimensional penultimate vector.
    """
    def __init__(self, num_classes: int, pretrained: bool = True,
                 feature_dim: int = 640, freeze_backbone: bool = False):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.linear1     = nn.Linear(2048, feature_dim)
        self.linear2     = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(1)
        return self.linear1(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.get_features(x)))

    def forward_with_features(self, x: torch.Tensor):
        feats  = self.get_features(x)
        logits = self.linear2(F.relu(feats))
        return logits, feats


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_accuracy(model, loader, device='cuda') -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_breeds_classifier(model, trainloader, valloader,
                             epochs=30, lr=0.01, device='cuda',
                             save_path='breeds_classifier.pth', patience=5):
    model = model.to(device)
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
              f"| train_acc={correct/total:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nTraining done. Best val_acc={best_acc:.4f}")


def collect_train_scores_breeds(model, train_dataset, batch_size=256, device='cuda'):
    """Return (scores [N, C], labels [N]) for the training set."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    sc_list, lb_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting train scores'):
            feats  = model.get_features(images.to(device))
            scores = model.linear2(F.relu(feats))
            sc_list.append(scores.cpu().numpy())
            lb_list.append(labels.numpy())
    return np.concatenate(sc_list), np.concatenate(lb_list)


def create_score_dataset_breeds(dataset, model, pool_score, pool_vectors,
                                 strategy='score_coord', device='cuda', batch_size=256):
    """Build a ScoreFeatureDataset for one BREEDS split."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    sc_list, ft_list, dc_list, lb_list = [], [], [], []
    rng = np.random.default_rng(0)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'Score dataset [{strategy}]', leave=False):
            feats  = model.get_features(images.to(device))
            scores = model.linear2(F.relu(feats))
            sc_np  = scores.cpu().numpy()

            if strategy == 'score_coord':
                dc_np = _build_decoy_score_coord(sc_np, pool_score, rng)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            sc_list.append(scores.cpu())
            ft_list.append(feats.cpu())
            dc_list.append(torch.from_numpy(dc_np).float())
            lb_list.append(labels)

    return ScoreFeatureDataset(
        torch.cat(sc_list), torch.cat(ft_list),
        torch.cat(dc_list), torch.cat(lb_list),
    )


def build_testset_names(n_testsets: int) -> list:
    names = [
        'val_source', 'val_target',
        'imagenet_val_source', 'imagenet_val_target',
    ]
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            names.append(f'corr_source_{corruption}_sev{severity}')
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            names.append(f'corr_target_{corruption}_sev{severity}')
    names = names[:n_testsets]
    while len(names) < n_testsets:
        names.append(f'unknown_{len(names)}')
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR        = "/home/arina/imagenet"
BREEDS_NAME     = "nonliving26"   # living17 | entity13 | entity30 | nonliving26
BATCH_SIZE      = 64
DEVICE          = 'cuda:2' if torch.cuda.is_available() else 'cpu'
DECOY_STRATEGY  = 'score_coord'
CLASSIFIER_PATH = f'breeds_{BREEDS_NAME}_classifier.pth'
FLOWS_PATH      = f'cond_flows_breeds_{BREEDS_NAME}_normalize.pth'
FEATURE_DIM     = 640

COLOR_ENGPE    = '#1976D2'
COLOR_ENGPE_TA = '#FF6D00'

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Data
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\nBREEDS: {BREEDS_NAME}\n{'='*60}")

(trainset, train_subset, val_subset,
 trainloader, testsets, testloaders) = get_imagenet_breeds(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, name=BREEDS_NAME)

hierarchy_dir = f"{DATA_DIR}/imagenet_class_hierarchy"
ret           = get_breeds_ret(hierarchy_dir, BREEDS_NAME)
NUM_CLASSES   = len(ret[1][0])
valloader     = testloaders[0]
print(f"NUM_CLASSES (source): {NUM_CLASSES}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Classifier
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\nMODEL\n{'='*60}")

breeds_model = BREEDSClassifier(
    num_classes=NUM_CLASSES, pretrained=True,
    feature_dim=FEATURE_DIM, freeze_backbone=False,
).to(DEVICE)

if os.path.exists(CLASSIFIER_PATH):
    breeds_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    print(f"Classifier loaded from {CLASSIFIER_PATH}")
else:
    print("Training classifier from scratch...")
    train_breeds_classifier(
        model=breeds_model, trainloader=trainloader, valloader=valloader,
        epochs=5, lr=0.01, device=DEVICE, save_path=CLASSIFIER_PATH, patience=3,
    )

breeds_model.eval()
print(f"Val accuracy (source): {evaluate_accuracy(breeds_model, valloader, DEVICE):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Training scores and decoy pools
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\nCOLLECTING TRAIN SCORES\n{'='*60}")

train_scores_raw, train_labels_raw = collect_train_scores_breeds(
    breeds_model, train_subset, batch_size=256, device=DEVICE)
print(f"Scores: {train_scores_raw.shape}  "
      f"acc={(train_scores_raw.argmax(1) == train_labels_raw).mean():.4f}")

print(f"\n{'='*60}\nBUILDING DECOY POOLS\n{'='*60}")
pool_score, pool_vectors = build_error_conditioned_pools(
    train_scores_raw, train_labels_raw, NUM_CLASSES, verbose=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Conditional normalizing flow
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\nFLOW MODEL\n{'='*60}")

score_shift_flow = ScoreShiftFlowWrapper(
    num_classes=NUM_CLASSES, n_flows=12,
    feature_dim=FEATURE_DIM, hidden_dim=256, encoder_dim=128, clip_val=5.0,
).to(DEVICE)

print(f"Creating train score dataset (strategy='{DECOY_STRATEGY}')...")
train_score_dataset = create_score_dataset_breeds(
    train_subset, breeds_model,
    pool_score=pool_score, pool_vectors=pool_vectors,
    strategy=DECOY_STRATEGY, device=DEVICE,
)

if os.path.exists(FLOWS_PATH):
    score_shift_flow.load_state_dict(torch.load(FLOWS_PATH, map_location=DEVICE))
    print(f"Flow loaded from {FLOWS_PATH}")
else:
    print("Training flow...")
    score_shift_flow.train_flow(
        train_score_dataset, epochs=30, lr=3e-4,
        batch_size=256, device=DEVICE, patience=5, grad_clip=1.0,
    )
    torch.save(score_shift_flow.state_dict(), FLOWS_PATH)
    print(f"Flow saved to {FLOWS_PATH}")

score_shift_flow.eval()

# Temperature calibration on training set
source_logits, _, source_labels = score_shift_flow.generate_decoys(
    train_score_dataset, device=DEVICE)
source_logits_t = torch.tensor(source_logits).to(DEVICE)
source_labels_t = torch.tensor(source_labels).to(DEVICE)
temp            = calibration_temp(source_logits_t, source_labels_t)
scaled_source   = source_logits_t / temp
print(f"Source: {len(source_labels)} samples, temp={temp:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Baseline methods
# ─────────────────────────────────────────────────────────────────────────────

try:
    from utils.other_methods import predict_COT
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

BASELINE_METHODS = {
    'ATC':    predict_ATC_maxconf,
    'ATC-NE': predict_ATC_negent,
    'AC':     predict_AC,
    'DOC':    predict_DOC,
}
if COT_AVAILABLE:
    BASELINE_METHODS['COT'] = predict_COT

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Test set names
# ─────────────────────────────────────────────────────────────────────────────

TESTSET_NAMES = build_testset_names(len(testsets))
print(f"Total test sets: {len(testsets)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs('BREEDS/figures/diagnostics', exist_ok=True)
os.makedirs('BREEDS/figures/accuracy',    exist_ok=True)
os.makedirs('BREEDS/figures/comparison',  exist_ok=True)

all_results = []
tile_data   = {}

for testset_idx, (testset, testloader) in enumerate(zip(testsets, testloaders)):

    ds_name = TESTSET_NAMES[testset_idx]
    print(f"\n{'─'*60}")
    print(f"  TESTSET [{testset_idx+1}/{len(testsets)}]: {ds_name}")
    print(f"{'─'*60}")

    try:
        test_score_ds = create_score_dataset_breeds(
            testset, breeds_model,
            pool_score=pool_score, pool_vectors=pool_vectors,
            strategy=DECOY_STRATEGY, device=DEVICE,
        )
    except Exception as e:
        print(f"  Failed to create score dataset: {e}")
        continue

    ms, ds_flow, ls = score_shift_flow.generate_decoys(test_score_ds, device=DEVICE)
    n = len(ls)
    if n == 0:
        print(f"  Empty dataset, skipping")
        continue

    target_logits = torch.tensor(ms).to(DEVICE)
    target_labels = torch.tensor(ls).to(DEVICE)
    true_acc_full = float((ms.argmax(axis=1) == ls).mean())
    scaled_target = target_logits / temp
    print(f"  N={n}, true_acc={true_acc_full:.4f}")

    # Baseline estimates
    baseline_estimates, baseline_errors = {}, {}
    for method_name, method_func in BASELINE_METHODS.items():
        try:
            estimate = float(method_func(scaled_source, source_labels_t, scaled_target))
        except Exception as e:
            print(f"  {method_name}: FAILED ({e})")
            estimate = np.nan
        estimate = np.clip(estimate, 0.0, 1.0)
        baseline_estimates[method_name] = estimate
        baseline_errors[method_name]    = abs(estimate - true_acc_full)
        print(f"  {method_name}: est={estimate:.4f}  err={abs(estimate - true_acc_full):.4f}")

    pred_scores  = np.max(ms,      axis=1)
    pred_label   = np.argmax(ms,   axis=1)
    decoy_scores = np.max(ds_flow, axis=1)
    true_labels  = ls

    probs_np      = F.softmax(torch.tensor(ms).float(), dim=1).numpy()
    mano_fro      = np.linalg.norm(probs_np, ord='fro') / np.sqrt(n)
    mano_fro_norm = float(np.clip(
        (mano_fro - 1.0 / np.sqrt(NUM_CLASSES)) / (1.0 - 1.0 / np.sqrt(NUM_CLASSES)),
        0.0, 1.0))

    correct_pred = (true_labels == pred_label).astype(int)

    # Sort by prediction score (ascending)
    sort_idx           = np.argsort(pred_scores)
    pred_scores_sorted = pred_scores[sort_idx]
    label_sorted       = true_labels[sort_idx]
    pred_label_sorted  = pred_label[sort_idx]
    correct_pred_s     = (label_sorted == pred_label_sorted).astype(int)

    # True FDR curve
    FD        = 1 - correct_pred_s
    FD_CF     = np.cumsum(FD[::-1])[::-1]
    D_CF      = np.arange(0, n)[::-1] + 1
    FDR_true  = np.clip(FD_CF / D_CF, 0, 1)
    QVAL_true = np.clip(np.minimum.accumulate(FDR_true), 0, 1)

    # Mix-Max FDR
    pi0 = 0.0
    sorted_decoys           = np.sort(decoy_scores)
    unique_z_vals, counts_z = np.unique(decoy_scores, return_counts=True)
    n_unique_z              = len(unique_z_vals)

    counts_w_leq_z = np.searchsorted(pred_scores_sorted, unique_z_vals, side='left')
    counts_z_leq_z = np.searchsorted(sorted_decoys,      unique_z_vals, side='left')
    P_W_leq_z = np.clip(
        (counts_w_leq_z - pi0 * counts_z_leq_z) / ((1 - pi0) * n), 0, 1)
    P_Y_leq_z = np.clip(counts_z_leq_z / n, 0, 1)
    R_j = np.clip(
        np.divide(P_W_leq_z, P_Y_leq_z,
                  out=np.zeros_like(P_W_leq_z), where=P_Y_leq_z > 0),
        0, 1)

    fdr_values = np.zeros(n)
    for i, T in enumerate(pred_scores_sorted[::-1]):
        D     = i + 1
        F_0   = pi0 * np.sum(decoy_scores > T)
        z_idx = np.searchsorted(unique_z_vals, T, side='left')
        F_1   = 0.0 if z_idx >= n_unique_z else (
            (1 - pi0) * np.sum(R_j[z_idx:] * counts_z[z_idx:]))
        fdr_values[i] = (F_0 + F_1) / D if D > 0 else 0.0

    QVAL_mixmax = np.clip(np.minimum.accumulate(np.clip(fdr_values, 0, 1)[::-1]), 0, 1)

    # TDC FDR
    TDC_score = np.maximum(pred_scores, decoy_scores)
    TDC_win   = (pred_scores > decoy_scores).astype(int)
    tdc_idx   = np.argsort(TDC_score)
    FD_CF_tdc = np.cumsum((1 - TDC_win[tdc_idx])[::-1])[::-1]
    D_CF_tdc  = np.maximum(np.arange(0, n)[::-1] + 1 - FD_CF_tdc, 1)
    QVAL_TDC  = np.clip(np.minimum.accumulate(np.clip(FD_CF_tdc / D_CF_tdc, 0, 1)), 0, 1)

    # Accuracy estimation
    pi0_tdc = np.clip(float(QVAL_TDC[0]),    0.0, 1.0)
    pi0_mm  = np.clip(float(QVAL_mixmax[0]), 0.0, 1.0)

    Acc_est    = np.zeros(n)
    Acc_est_MM = np.zeros(n)
    Acc_true   = np.zeros(n)

    for i in range(n):
        TP_true     = correct_pred_s[i:].sum()
        TN_true     = (1 - correct_pred_s[:i]).sum()
        Acc_true[i] = (TP_true + TN_true) / n

        accepted      = n - i
        FP_tdc        = accepted * QVAL_TDC[i]
        TP_tdc        = accepted * (1 - QVAL_TDC[i])
        TN_tdc        = n * pi0_tdc - FP_tdc
        Acc_est[i]    = np.clip((TP_tdc + TN_tdc) / n, 0.0, 1.0)

        FP_mm         = accepted * QVAL_mixmax[i]
        TP_mm         = accepted * (1 - QVAL_mixmax[i])
        TN_mm         = n * pi0_mm - FP_mm
        Acc_est_MM[i] = np.clip((TP_mm + TN_mm) / n, 0.0, 1.0)

    Acc_true = np.clip(Acc_true, 0.0, 1.0)

    acc_st_true    = float(Acc_true[0])
    acc_ta_true    = float(Acc_true.max())
    acc_st_est_tdc = float(Acc_est[0])
    acc_ta_est_tdc = float(Acc_est.max())
    acc_st_est_mm  = float(Acc_est_MM[0])
    acc_ta_est_mm  = float(Acc_est_MM.max())

    err_st_tdc = abs(acc_st_est_tdc - acc_st_true)
    err_ta_tdc = abs(acc_ta_est_tdc - acc_ta_true)
    err_st_mm  = abs(acc_st_est_mm  - acc_st_true)
    err_ta_mm  = abs(acc_ta_est_mm  - acc_ta_true)

    baseline_errors_ta = {m: abs(baseline_estimates[m] - acc_ta_true)
                          for m in BASELINE_METHODS}

    print(f"\n  {'':28} {'ACC_ST':>10}  {'ACC_TA':>10}")
    print(f"  {'─'*52}")
    print(f"  {'True':<28} {acc_st_true:>10.4f}  {acc_ta_true:>10.4f}")
    print(f"  {'TDC  (est|err)':<28} "
          f"{acc_st_est_tdc:>6.4f} {err_st_tdc:>+.4f}  "
          f"{acc_ta_est_tdc:>6.4f} {err_ta_tdc:>+.4f}")
    print(f"  {'Mix-Max (est|err)':<28} "
          f"{acc_st_est_mm:>6.4f} {err_st_mm:>+.4f}  "
          f"{acc_ta_est_mm:>6.4f} {err_ta_mm:>+.4f}")

    # Store curve data
    total_TP  = int(correct_pred_s.sum())
    TP_from_i = np.cumsum(correct_pred_s[::-1])[::-1]
    D_from_i  = np.arange(n, 0, -1)
    tile_data[ds_name] = dict(
        normalized_rank  = np.arange(n) / n,
        logit_threshold  = pred_scores_sorted.copy(),
        QVAL_TDC        = QVAL_TDC.copy(),
        QVAL_mixmax     = QVAL_mixmax.copy(),
        QVAL_true       = QVAL_true.copy(),
        Acc_true        = Acc_true.copy(),
        Acc_est         = Acc_est.copy(),
        Acc_est_MM      = Acc_est_MM.copy(),
        precision_true  = np.where(D_from_i > 0, TP_from_i / D_from_i, 0.0),
        recall_true     = TP_from_i / max(total_TP, 1),
        precision_est   = np.clip(1 - QVAL_mixmax, 0.0, 1.0),
        recall_est      = np.clip((1 - QVAL_mixmax) * D_from_i / max(total_TP, 1), 0.0, 1.0),
        n_samples       = n,
        ds_name         = ds_name,
        mano_fro_norm   = mano_fro_norm,
        acc_st_true     = acc_st_true,
    )

    # Diagnostic figures (first 10 test sets only)
    if testset_idx < 10:
        safe_name       = f'breeds_{BREEDS_NAME}_{ds_name}'
        normalized_rank = np.arange(n) / n

        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.arange(pred_scores.min() - 0.5, pred_scores.max() + 0.5, 0.05)
        incorrect_mask = true_labels != pred_label
        sns.histplot(pred_scores, bins=bins, stat='density', color='blue',
                     kde=True, fill=True, alpha=0.3, label='model', ax=ax)
        sns.histplot(decoy_scores, bins=bins, stat='density', color='orange',
                     kde=True, fill=True, alpha=0.3, label='null', ax=ax)
        if incorrect_mask.any():
            sns.histplot(pred_scores[incorrect_mask], bins=bins, stat='density',
                         color='red', kde=True, fill=True, alpha=0.3,
                         label='incorrect', ax=ax)
        ax.set_title(f'{ds_name} — Score Distributions')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'BREEDS/figures/diagnostics/{safe_name}_scores.png', dpi=150)
        plt.savefig(f'BREEDS/figures/diagnostics/{safe_name}_scores.pdf')
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pred_scores_sorted, QVAL_mixmax, label='Mix-Max FDR')
        ax.plot(pred_scores_sorted, QVAL_true,   label='True FDR')
        ax.set_xlabel('Score threshold')
        ax.set_ylabel('q-value (FDR)')
        ax.set_title(f'{ds_name} — FDR vs Score Threshold')
        ax.legend(); ax.grid()
        plt.tight_layout()
        plt.savefig(f'BREEDS/figures/diagnostics/{safe_name}_fdr_vs_score.png', dpi=150)
        plt.savefig(f'BREEDS/figures/diagnostics/{safe_name}_fdr_vs_score.pdf')
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(normalized_rank, QVAL_TDC,    label='TDC FDR')
        ax.plot(normalized_rank, QVAL_mixmax, label='Mix-Max FDR')
        ax.plot(normalized_rank, QVAL_true,   label='True FDR')
        ax.plot(normalized_rank, Acc_true,    label='True Acc')
        ax.plot(normalized_rank, Acc_est,     label='ENGPE (TDC)')
        ax.plot(normalized_rank, Acc_est_MM,  label='ENGPE-TA (Mix-Max)')
        ax.set_xlabel('Normalized rank (fraction accepted)')
        ax.set_ylabel('Value')
        ax.set_title(f'{ds_name}')
        ax.legend(); ax.grid()
        plt.tight_layout()
        plt.savefig(f'BREEDS/figures/accuracy/{safe_name}_acc_fdr.png', dpi=150)
        plt.savefig(f'BREEDS/figures/accuracy/{safe_name}_acc_fdr.pdf')
        plt.close()

    # Collect results row
    row = dict(
        testset=ds_name, testset_idx=testset_idx, n_samples=n,
        is_source=('source' in ds_name), is_target=('target' in ds_name),
        is_clean=(testset_idx < 4), true_acc=true_acc_full,
        acc_st_true=acc_st_true, acc_ta_true=acc_ta_true,
        acc_st_tdc=acc_st_est_tdc, acc_ta_tdc=acc_ta_est_tdc,
        err_st_tdc=err_st_tdc,     err_ta_tdc=err_ta_tdc,
        acc_st_mm=acc_st_est_mm,   acc_ta_mm=acc_ta_est_mm,
        err_st_mm=err_st_mm,       err_ta_mm=err_ta_mm,
        mano_fro_norm=mano_fro_norm,
    )
    for m in BASELINE_METHODS:
        row[f'est_{m}']    = baseline_estimates[m]
        row[f'err_{m}']    = baseline_errors[m]
        row[f'err_ta_{m}'] = baseline_errors_ta[m]
    all_results.append(row)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Aggregate results and publication figures
# ─────────────────────────────────────────────────────────────────────────────

if not all_results:
    print("No results collected.")
    exit(0)

df = pd.DataFrame(all_results)
df.to_csv(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_results.csv', index=False)
print(f"\nresults.csv saved ({len(df)} test sets)")

groups = {
    'all':         df,
    'clean':       df[df['is_clean']],
    'corruptions': df[~df['is_clean']],
    'source':      df[df['is_source']],
    'target':      df[df['is_target']],
}

print("\n" + "="*80 + "\nAGGREGATE SUMMARY\n" + "="*80)

for group_name, gdf in groups.items():
    if len(gdf) == 0:
        continue
    print(f"\n  Group: {group_name} (n={len(gdf)})")
    print(f"  {'Metric':<38} {'Mean_Est':>9}  {'Mean_True':>9}  {'MAE':>9}  {'Std':>9}")
    print(f"  {'─'*80}")

    def _print_row(label, col_est, col_true, col_err):
        est_v  = gdf[col_est].dropna().values  if col_est  else None
        true_v = gdf[col_true].dropna().values if col_true else None
        err_v  = gdf[col_err].dropna().values  if col_err  else None
        est_m  = est_v.mean()  if est_v  is not None and len(est_v)  > 0 else float('nan')
        true_m = true_v.mean() if true_v is not None and len(true_v) > 0 else float('nan')
        err_m  = err_v.mean()  if err_v  is not None and len(err_v)  > 0 else float('nan')
        err_s  = err_v.std()   if err_v  is not None and len(err_v)  > 0 else float('nan')
        print(f"  {label:<38} {est_m:>9.4f}  {true_m:>9.4f}  {err_m:>9.4f}  {err_s:>9.4f}")

    true_st = gdf['acc_st_true'].dropna().values
    true_ta = gdf['acc_ta_true'].dropna().values
    print(f"  {'True ACC_ST':<38} {'─':>9}  {true_st.mean():>9.4f}  {'─':>9}  {'─':>9}")
    print(f"  {'True ACC_TA':<38} {'─':>9}  {true_ta.mean():>9.4f}  {'─':>9}  {'─':>9}")
    print(f"  {'─'*80}")

    _print_row('ENGPE    (ACC_ST)',    'acc_st_tdc', 'acc_st_true', 'err_st_tdc')
    _print_row('ENGPE    (ACC_TA)',    'acc_ta_tdc', 'acc_ta_true', 'err_ta_tdc')
    _print_row('ENGPE-TA (ACC_ST)',    'acc_st_mm',  'acc_st_true', 'err_st_mm')
    _print_row('ENGPE-TA (ACC_TA)',    'acc_ta_mm',  'acc_ta_true', 'err_ta_mm')
    print(f"  {'─'*80}")
    for m in BASELINE_METHODS:
        _print_row(f'{m} (est vs ACC_ST)', f'est_{m}', 'acc_st_true', f'err_{m}')
        _print_row(f'{m} (est vs ACC_TA)', f'est_{m}', 'acc_ta_true', f'err_ta_{m}')

# Figure A: Bar chart of MAE per method
method_names_bar = (
    ['ENGPE (ST)', 'ENGPE (TA)', 'ENGPE-TA (ST)', 'ENGPE-TA (TA)'] +
    [f'{m} (ST)' for m in BASELINE_METHODS] +
    [f'{m} (TA)' for m in BASELINE_METHODS]
)
method_cols_bar = (
    ['err_st_tdc', 'err_ta_tdc', 'err_st_mm', 'err_ta_mm'] +
    [f'err_{m}' for m in BASELINE_METHODS] +
    [f'err_ta_{m}' for m in BASELINE_METHODS]
)
colors_bar = (
    ['#2196F3', '#1565C0', '#FF9800', '#E65100'] +
    ['#4CAF50'] * len(BASELINE_METHODS) +
    ['#388E3C'] * len(BASELINE_METHODS)
)

for group_name, gdf in groups.items():
    if len(gdf) == 0:
        continue
    means_bar = [gdf[c].dropna().mean() for c in method_cols_bar]
    stds_bar  = [gdf[c].dropna().std()  for c in method_cols_bar]

    fig, ax = plt.subplots(figsize=(max(8, len(method_names_bar) * 0.9), 5))
    x_pos = np.arange(len(method_names_bar))
    ax.bar(x_pos, means_bar, yerr=stds_bar, color=colors_bar,
           capsize=4, alpha=0.85, edgecolor='black', linewidth=1,
           error_kw=dict(elinewidth=1.2, ecolor='black'))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names_bar, rotation=35, ha='right')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'BREEDS {BREEDS_NAME} — MAE [{group_name}]')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_{group_name}_bar_error.png', dpi=150)
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_{group_name}_bar_error.pdf')
    plt.close()

# Figure B: Scatter estimated vs true accuracy
scatter_methods = [
    dict(label='ENGPE',    col_est='acc_st_tdc', col_true='acc_st_true', panel=0, marker='o', color=COLOR_ENGPE,    size=35),
    dict(label='ENGPE-TA', col_est='acc_st_mm',  col_true='acc_st_true', panel=0, marker='s', color=COLOR_ENGPE_TA, size=50),
    dict(label='ENGPE',    col_est='acc_ta_tdc', col_true='acc_ta_true', panel=1, marker='o', color=COLOR_ENGPE,    size=35),
    dict(label='ENGPE-TA', col_est='acc_ta_mm',  col_true='acc_ta_true', panel=1, marker='s', color=COLOR_ENGPE_TA, size=50),
]
baseline_colors  = plt.cm.tab10(np.linspace(0, 0.9, len(BASELINE_METHODS)))
baseline_markers = ['D', '^', 'v', 'P', 'X', '*', 'h', '8']
for idx, m in enumerate(BASELINE_METHODS):
    c  = baseline_colors[idx]
    mk = baseline_markers[idx % len(baseline_markers)]
    scatter_methods += [
        dict(label=m, col_est=f'est_{m}', col_true='acc_st_true', panel=0, marker=mk, color=c, size=25),
        dict(label=m, col_est=f'est_{m}', col_true='acc_ta_true', panel=1, marker=mk, color=c, size=25),
    ]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for panel_idx, ax in enumerate(axes):
    panel_methods = [sm for sm in scatter_methods if sm['panel'] == panel_idx]
    all_vals = []
    for sm in panel_methods:
        all_vals.extend(df[sm['col_true']].values.tolist())
        all_vals.extend(df[sm['col_est']].dropna().values.tolist())
    lo, hi = min(all_vals) - 0.02, max(all_vals) + 0.02
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y')
    plotted = set()
    for sm in panel_methods:
        lbl = sm['label'] if sm['label'] not in plotted else '_nolegend_'
        ax.scatter(df[sm['col_true']].values, df[sm['col_est']].values,
                   label=lbl, marker=sm['marker'], color=sm['color'],
                   s=sm.get('size', 25), alpha=0.75)
        plotted.add(sm['label'])
    ax.set_xlabel('True Accuracy')
    ax.set_ylabel('Estimated Accuracy')
    ax.set_title(['ACC$_{ST}$', 'ACC$_{TA}$'][panel_idx])
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_est_vs_true.png', dpi=150)
plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_est_vs_true.pdf')
plt.close()

# Figure C: True ACC_ST vs True ACC_TA
fig, ax = plt.subplots(figsize=(6, 4))
lo = min(df['acc_st_true'].min(), df['acc_ta_true'].min()) - 0.02
hi = max(df['acc_st_true'].max(), df['acc_ta_true'].max()) + 0.02
ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y')
src_mask = df['is_source'].values
ax.scatter(df.loc[src_mask,  'acc_st_true'], df.loc[src_mask,  'acc_ta_true'],
           color='steelblue',  s=40, alpha=0.8, label='source', zorder=3)
ax.scatter(df.loc[~src_mask, 'acc_st_true'], df.loc[~src_mask, 'acc_ta_true'],
           color='darkorange', s=40, alpha=0.8, label='target', zorder=3)
ax.set_xlabel('True ACC$_{ST}$')
ax.set_ylabel('True ACC$_{TA}$')
ax.legend(); ax.grid(linestyle='--', alpha=0.4)
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_accst_vs_accta.png', dpi=150)
plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_accst_vs_accta.pdf')
plt.close()

# Figure D: Accuracy vs corruption severity
corr_df = df[~df['is_clean']].copy()
if len(corr_df) > 0:
    sev_extracted = corr_df['testset'].str.extract(r'sev(\d)')
    if sev_extracted[0].notna().any():
        corr_df['severity'] = sev_extracted[0].astype(int)
        corr_df['split']    = np.where(corr_df['is_source'], 'source', 'target')
        fig, ax = plt.subplots(figsize=(7, 4))
        for split, color in [('source', '#2196F3'), ('target', '#FF9800')]:
            for col, ls, lbl in [
                ('true_acc',   '-',  f'True Acc ({split})'),
                ('acc_st_tdc', '--', f'ENGPE ({split})'),
                ('acc_st_mm',  ':',  f'ENGPE-TA ({split})'),
            ]:
                sev_vals = sorted(corr_df['severity'].unique())
                vals = [corr_df[(corr_df['severity'] == sev) &
                                (corr_df['split'] == split)][col].mean()
                        for sev in sev_vals]
                ax.plot(sev_vals, vals, color=color, linestyle=ls, marker='o', label=lbl)
        ax.set_xlabel('Severity'); ax.set_ylabel('Accuracy')
        ax.set_title(f'BREEDS {BREEDS_NAME} — Acc vs Severity')
        ax.legend(fontsize=8); ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_acc_vs_severity.png', dpi=150)
        plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_acc_vs_severity.pdf')
        plt.close()

# Summary CSV
summary_rows = []
for col_est, col_true, col_err, method, kind in [
    ('acc_st_tdc', 'acc_st_true', 'err_st_tdc', 'ENGPE',    'ST'),
    ('acc_ta_tdc', 'acc_ta_true', 'err_ta_tdc', 'ENGPE',    'TA'),
    ('acc_st_mm',  'acc_st_true', 'err_st_mm',  'ENGPE-TA', 'ST'),
    ('acc_ta_mm',  'acc_ta_true', 'err_ta_mm',  'ENGPE-TA', 'TA'),
]:
    v_err = df[col_err].dropna().values
    summary_rows.append(dict(
        method=method, type=kind,
        mean_est=df[col_est].dropna().values.mean(),
        mean_true=df[col_true].dropna().values.mean(),
        mean_mae=v_err.mean(), std_mae=v_err.std(),
    ))
for m in BASELINE_METHODS:
    for kind, col_true, col_err in [
        ('ST', 'acc_st_true', f'err_{m}'),
        ('TA', 'acc_ta_true', f'err_ta_{m}'),
    ]:
        v_err = df[col_err].dropna().values
        summary_rows.append(dict(
            method=m, type=kind,
            mean_est=df[f'est_{m}'].dropna().values.mean(),
            mean_true=df[col_true].dropna().values.mean(),
            mean_mae=v_err.mean(), std_mae=v_err.std(),
        ))
pd.DataFrame(summary_rows).to_csv(
    f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_summary.csv',
    index=False, float_format='%.6f')
print(f"summary.csv saved")

# Figure E: MANO scatter
if 'mano_fro_norm' in df.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    lo = min(df['acc_st_true'].min(), df['mano_fro_norm'].min()) - 0.02
    hi = max(df['acc_st_true'].max(), df['mano_fro_norm'].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y')
    ax.scatter(df['acc_st_true'], df['mano_fro_norm'],
               color='#7B1FA2', s=30, alpha=0.75, label='MANO')
    ax.set_xlabel('True Accuracy (ACC$_{ST}$)')
    ax.set_ylabel('MANO (Frobenius norm)')
    ax.legend(); ax.grid(linestyle='--', alpha=0.4)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_mano.png', dpi=150)
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_mano.pdf')
    plt.close()

# Figure F: Best test set — Acc/FDR vs rank
if tile_data:
    best_key = max(tile_data, key=lambda k: tile_data[k]['acc_st_true'])
    td = tile_data[best_key]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(td['normalized_rank'], td['Acc_true'],    label='True Acc',     linewidth=1.5, linestyle='--', color='steelblue')
    ax.plot(td['normalized_rank'], td['QVAL_true'],   label='True FDR',     linewidth=1.5, linestyle='--', color='#78909C')
    ax.plot(td['normalized_rank'], td['Acc_est_MM'],  label='ENGPE-TA',     linewidth=1.8, color=COLOR_ENGPE_TA)
    ax.plot(td['normalized_rank'], td['QVAL_mixmax'], label='ENGPE-TA FDR', linewidth=1.5, color=COLOR_ENGPE_TA, linestyle=':')
    ax.set_xlabel('Fraction accepted'); ax.set_ylabel('Value')
    ax.legend(); ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_acc_fdr.png', dpi=150)
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_acc_fdr.pdf')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(td['recall_true'], td['precision_true'], label='True PR',     linewidth=1.5, color='steelblue')
    ax.plot(td['recall_est'],  td['precision_est'],  label='ENGPE-TA PR', linewidth=1.8, color=COLOR_ENGPE_TA)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.legend(); ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_pr_curve.png', dpi=150)
    plt.savefig(f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_pr_curve.pdf')
    plt.close()

# Subsampled curves CSV
N_CURVE_PTS = 100
curve_rows  = []
for key, td in tile_data.items():
    n_td = td['n_samples']
    idx  = np.linspace(0, n_td - 1, N_CURVE_PTS, dtype=int)
    for i in idx:
        curve_rows.append(dict(
            dataset='breeds', breeds_name=BREEDS_NAME, testset=key,
            logit_threshold=float(td['logit_threshold'][i]),
            frac_accepted=float(td['normalized_rank'][i]),
            acc_true=float(td['Acc_true'][i]),
            acc_est_mm=float(td['Acc_est_MM'][i]),
            acc_est_tdc=float(td['Acc_est'][i]),
            fdr_mixmax=float(td['QVAL_mixmax'][i]),
            fdr_tdc=float(td['QVAL_TDC'][i]),
            fdr_true=float(td['QVAL_true'][i]),
        ))

df_curves = pd.DataFrame(curve_rows)
base_csv  = f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}'

# All testsets
df_curves.to_csv(f'{base_csv}_acc_curves_all.csv', index=False, float_format='%.6f')

# Corruptions only
corr_mask = df_curves['testset'].str.contains('corr_', na=False)
df_curves[corr_mask].to_csv(
    f'{base_csv}_acc_curves_corruptions.csv', index=False, float_format='%.6f')

# Best testset: highest acc_st_true among corruption test sets (closest to train)
corr_keys = [k for k in tile_data if 'corr_' in k]
if corr_keys:
    best_key = max(corr_keys, key=lambda k: tile_data[k]['acc_st_true'])
else:
    best_key = max(tile_data, key=lambda k: tile_data[k]['acc_st_true'])
df_curves[df_curves['testset'] == best_key].to_csv(
    f'{base_csv}_acc_curves_best.csv', index=False, float_format='%.6f')

# Backward-compat single file (now includes logit_threshold)
df_curves.to_csv(f'{base_csv}_acc_curves.csv', index=False, float_format='%.6f')

print(f"acc_curves saved: all={len(df_curves)}, "
      f"corruptions={corr_mask.sum()}, best_testset='{best_key}'")

print(f"\n{'='*80}")
print(f"BREEDS '{BREEDS_NAME}' pipeline complete.")
print(f"  Test sets processed : {len(df)}")
print(f"  Results saved in    : BREEDS/figures/")
print(f"{'='*80}")
