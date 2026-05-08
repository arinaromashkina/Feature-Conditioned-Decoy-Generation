import os
import time
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
    makers = {
        'living17':    make_living17,
        'entity13':    make_entity13,
        'entity30':    make_entity30,
        'nonliving26': make_nonliving26,
    }
    if name not in makers:
        raise ValueError(f"Unknown BREEDS name: {name}. Choose from: {list(makers)}")
    return makers[name](hierarchy_dir, split='good')


def get_imagenet_breeds(batch_size, data_dir, name='living17', seed=42):
    hierarchy_dir = f"{data_dir}/imagenet_class_hierarchy"
    ret           = get_breeds_ret(hierarchy_dir, name)

    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4717, 0.4499, 0.3837],
                             [0.2600, 0.2516, 0.2575]),
    ])

    trainset  = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform, label_mapping=source_label_mapping,
    )
    targetset = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform, label_mapping=target_label_mapping,
    )

    idx = np.arange(len(trainset))
    np.random.seed(seed)
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

    add_loader(val_subset)
    add_loader(targetset)
    add_loader(folder.ImageFolder(f"{data_dir}/imagenetv1/val/",
               transform=transform, label_mapping=source_label_mapping))
    add_loader(folder.ImageFolder(f"{data_dir}/imagenetv1/val/",
               transform=transform, label_mapping=target_label_mapping))

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
# Z-score normalizer (ablation: compare against RobustFeatureNormalizer)
# ─────────────────────────────────────────────────────────────────────────────

class ZScoreNormalizer(nn.Module):
    """
    Standard running mean/std normalizer.
    Ablation baseline for RobustFeatureNormalizer (which uses median/IQR + tanh).
    """
    def __init__(self, feature_dim, eps=1e-6, momentum=0.01):
        super().__init__()
        self.eps      = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_std',  torch.ones(feature_dim))
        self.register_buffer('initialized',  torch.tensor(False))

    @torch.no_grad()
    def _update_stats(self, x):
        batch_mean = x.mean(0)
        batch_std  = x.std(0).clamp(min=self.eps)
        if not self.initialized:
            self.running_mean.copy_(batch_mean)
            self.running_std.copy_(batch_std)
            self.initialized.fill_(True)
        else:
            self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
            self.running_std.mul_(1 - self.momentum).add_(batch_std   * self.momentum)

    def forward(self, x):
        if self.training:
            self._update_stats(x)
        if not self.initialized:
            return x
        return (x - self.running_mean) / (self.running_std + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_classifier_inference_time(model, loader, device, n_batches=20):
    """Mean inference time in ms per sample."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(images)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / images.size(0) * 1000)
    return float(np.mean(times)) if times else float('nan')


def measure_flow_inference_time(flow, score_dataset, device, n_batches=20):
    """Mean flow forward-pass time in ms per sample."""
    flow.eval()
    loader = torch.utils.data.DataLoader(
        score_dataset, batch_size=256, shuffle=False, num_workers=0)
    times = []
    with torch.no_grad():
        for i, (_, feats, _, _) in enumerate(loader):
            if i >= n_batches:
                break
            feats = feats.to(device)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = flow.flow.sample(feats)
            if 'cuda' in str(device):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / feats.size(0) * 1000)
    return float(np.mean(times)) if times else float('nan')


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

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    print(f"\nTraining done. Best val_acc={best_acc:.4f}")


def collect_train_scores_breeds(model, train_dataset, batch_size=256, device='cuda'):
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
# Global config
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR        = "/home/arina/imagenet"
BREEDS_NAME     = "nonliving26"   # living17 | entity13 | entity30 | nonliving26
BATCH_SIZE      = 64
DEVICE          = 'cuda:2' if torch.cuda.is_available() else 'cpu'
DECOY_STRATEGY  = 'score_coord'
FEATURE_DIM     = 640
CLASSIFIER_PATH = f'breeds_{BREEDS_NAME}_classifier.pth'

COLOR_ENGPE    = '#1976D2'
COLOR_ENGPE_TA = '#FF6D00'

# Set QUICK_MODE=True to run only the baseline config (for testing)
QUICK_MODE = False

# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations (flow only — classifier is shared)
#
# name            – unique run identifier, used in filenames
# seed            – controls flow weight init + training randomness
# n_flows         – number of coupling layers (baseline = 12)
# encoder_dim     – feature encoder output dim (baseline = 128)
# use_robust_norm – True  = RobustFeatureNormalizer (median/IQR + tanh)
#                   False = ZScoreNormalizer (mean/std)
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    # stability: same architecture, different random seeds for flow training
    dict(name='seed_42',     seed=42,  n_flows=12, encoder_dim=128, use_robust_norm=True),
    dict(name='seed_123',    seed=123, n_flows=12, encoder_dim=128, use_robust_norm=True),
    dict(name='seed_456',    seed=456, n_flows=12, encoder_dim=128, use_robust_norm=True),
    # ablation: number of coupling layers
    dict(name='flows_6',     seed=42,  n_flows=6,  encoder_dim=128, use_robust_norm=True),
    dict(name='flows_24',    seed=42,  n_flows=24, encoder_dim=128, use_robust_norm=True),
    # ablation: encoder dimension
    dict(name='enc_64',      seed=42,  n_flows=12, encoder_dim=64,  use_robust_norm=True),
    dict(name='enc_256',     seed=42,  n_flows=12, encoder_dim=256, use_robust_norm=True),
    # ablation: normalizer type
    dict(name='zscore_norm', seed=42,  n_flows=12, encoder_dim=128, use_robust_norm=False),
]

CONFIGS_TO_RUN = ABLATION_CONFIGS[:1] if QUICK_MODE else ABLATION_CONFIGS


# ─────────────────────────────────────────────────────────────────────────────
# Flow-only experiment function
# (classifier, pools, test score datasets, and temperature are shared externally)
# ─────────────────────────────────────────────────────────────────────────────

def run_flow_experiment(cfg,
                        train_score_dataset,
                        test_score_datasets,
                        testset_names,
                        NUM_CLASSES,
                        scaled_source,
                        source_labels_t,
                        temp,
                        BASELINE_METHODS):
    """
    Train (or load) a flow for one ablation config, then evaluate on all test sets.
    Everything classifier-dependent is passed in as precomputed data.
    Returns dict with results DataFrame, timing dict, and tile_data.
    """
    cfg_name = cfg['name']
    seed     = cfg['seed']
    n_flows  = cfg['n_flows']
    enc_dim  = cfg['encoder_dim']

    print(f"\n{'#'*70}")
    print(f"  FLOW CONFIG: {cfg_name}  "
          f"(seed={seed}, n_flows={n_flows}, encoder_dim={enc_dim}, "
          f"robust_norm={cfg['use_robust_norm']})")
    print(f"{'#'*70}")

    set_seeds(seed)

    out_base = f'BREEDS/{cfg_name}'
    os.makedirs(f'{out_base}/diagnostics', exist_ok=True)
    os.makedirs(f'{out_base}/accuracy',    exist_ok=True)
    os.makedirs(f'{out_base}/comparison',  exist_ok=True)

    timing = {
        'config':          cfg_name,
        'seed':            seed,
        'n_flows':         n_flows,
        'encoder_dim':     enc_dim,
        'use_robust_norm': cfg['use_robust_norm'],
    }

    # ── Build flow ────────────────────────────────────────────────────────────
    FLOWS_PATH = f'cond_flows_breeds_{BREEDS_NAME}_{cfg_name}.pth'

    score_shift_flow = ScoreShiftFlowWrapper(
        num_classes=NUM_CLASSES, n_flows=n_flows,
        feature_dim=FEATURE_DIM, hidden_dim=256,
        encoder_dim=enc_dim, clip_val=5.0,
    ).to(DEVICE)

    if not cfg['use_robust_norm']:
        score_shift_flow.flow.feature_norm = ZScoreNormalizer(
            FEATURE_DIM, momentum=0.01).to(DEVICE)

    timing['flow_n_params'] = count_parameters(score_shift_flow)

    if os.path.exists(FLOWS_PATH):
        score_shift_flow.load_state_dict(
            torch.load(FLOWS_PATH, map_location=DEVICE, weights_only=False))
        print(f"  Flow loaded from {FLOWS_PATH}")
        timing['flow_train_time_s'] = float('nan')
    else:
        print("  Training flow...")
        t0 = time.perf_counter()
        score_shift_flow.train_flow(
            train_score_dataset, epochs=30, lr=3e-4,
            batch_size=256, device=DEVICE, patience=5, grad_clip=1.0,
        )
        timing['flow_train_time_s'] = time.perf_counter() - t0
        torch.save(score_shift_flow.state_dict(), FLOWS_PATH)
        print(f"  Flow saved to {FLOWS_PATH}  "
              f"({timing['flow_train_time_s']:.1f} s)")

    score_shift_flow.eval()
    timing['flow_inference_ms_per_sample'] = measure_flow_inference_time(
        score_shift_flow, train_score_dataset, DEVICE)
    print(f"  Flow inference: {timing['flow_inference_ms_per_sample']:.3f} ms/sample  "
          f"| params: {timing['flow_n_params']:,}")

    # ── Evaluate on all test sets ─────────────────────────────────────────────
    all_results = []
    tile_data   = {}
    t_eval_start = time.perf_counter()

    for testset_idx, (test_score_ds, ds_name) in enumerate(
            zip(test_score_datasets, testset_names)):

        if test_score_ds is None:
            continue

        print(f"  [{testset_idx+1}/{len(test_score_datasets)}] {ds_name}")

        ms, ds_flow, ls = score_shift_flow.generate_decoys(test_score_ds, device=DEVICE)
        n = len(ls)
        if n == 0:
            continue

        true_acc_full = float((ms.argmax(axis=1) == ls).mean())
        scaled_target = torch.tensor(ms).to(DEVICE) / temp

        # Baseline estimates
        baseline_estimates, baseline_errors = {}, {}
        for method_name, method_func in BASELINE_METHODS.items():
            try:
                estimate = float(method_func(scaled_source, source_labels_t, scaled_target))
            except Exception:
                estimate = np.nan
            estimate = np.clip(estimate, 0.0, 1.0)
            baseline_estimates[method_name] = estimate
            baseline_errors[method_name]    = abs(estimate - true_acc_full)

        pred_scores  = np.max(ms,      axis=1)
        pred_label   = np.argmax(ms,   axis=1)
        decoy_scores = np.max(ds_flow, axis=1)
        true_labels  = ls

        probs_np      = F.softmax(torch.tensor(ms).float(), dim=1).numpy()
        mano_fro      = np.linalg.norm(probs_np, ord='fro') / np.sqrt(n)
        mano_fro_norm = float(np.clip(
            (mano_fro - 1.0 / np.sqrt(NUM_CLASSES)) / (1.0 - 1.0 / np.sqrt(NUM_CLASSES)),
            0.0, 1.0))

        sort_idx           = np.argsort(pred_scores)
        pred_scores_sorted = pred_scores[sort_idx]
        label_sorted       = true_labels[sort_idx]
        pred_label_sorted  = pred_label[sort_idx]
        correct_pred_s     = (label_sorted == pred_label_sorted).astype(int)

        FD        = 1 - correct_pred_s
        FD_CF     = np.cumsum(FD[::-1])[::-1]
        D_CF      = np.arange(0, n)[::-1] + 1
        FDR_true  = np.clip(FD_CF / D_CF, 0, 1)
        QVAL_true = np.clip(np.minimum.accumulate(FDR_true), 0, 1)

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

        QVAL_mixmax = np.clip(
            np.minimum.accumulate(np.clip(fdr_values, 0, 1)[::-1]), 0, 1)

        TDC_score = np.maximum(pred_scores, decoy_scores)
        TDC_win   = (pred_scores > decoy_scores).astype(int)
        tdc_idx   = np.argsort(TDC_score)
        FD_CF_tdc = np.cumsum((1 - TDC_win[tdc_idx])[::-1])[::-1]
        D_CF_tdc  = np.maximum(np.arange(0, n)[::-1] + 1 - FD_CF_tdc, 1)
        QVAL_TDC  = np.clip(
            np.minimum.accumulate(np.clip(FD_CF_tdc / D_CF_tdc, 0, 1)), 0, 1)

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

        total_TP        = int(correct_pred_s.sum())
        TP_from_i       = np.cumsum(correct_pred_s[::-1])[::-1]
        D_from_i        = np.arange(n, 0, -1)
        normalized_rank = np.arange(n) / n

        tile_data[ds_name] = dict(
            normalized_rank  = normalized_rank,
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
            recall_est      = np.clip(
                (1 - QVAL_mixmax) * D_from_i / max(total_TP, 1), 0.0, 1.0),
            pred_scores     = pred_scores.copy(),
            decoy_scores    = decoy_scores.copy(),
            true_labels     = true_labels.copy(),
            pred_label      = pred_label.copy(),
            n_samples       = n,
            ds_name         = ds_name,
            mano_fro_norm   = mano_fro_norm,
            acc_st_true     = acc_st_true,
        )

        # ── Diagnostic figures (first 10 test sets) ───────────────────────────
        if testset_idx < 10:
            safe_name      = f'breeds_{BREEDS_NAME}_{ds_name}'
            incorrect_mask = true_labels != pred_label

            # Score distribution: model vs decoy vs incorrect
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = np.arange(pred_scores.min() - 0.5, pred_scores.max() + 0.5, 0.05)
            sns.histplot(pred_scores,  bins=bins, stat='density', color='blue',
                         kde=True, fill=True, alpha=0.3, label='model scores', ax=ax)
            sns.histplot(decoy_scores, bins=bins, stat='density', color='orange',
                         kde=True, fill=True, alpha=0.3, label='decoy (null)', ax=ax)
            if incorrect_mask.any():
                sns.histplot(pred_scores[incorrect_mask], bins=bins, stat='density',
                             color='red', kde=True, fill=True, alpha=0.3,
                             label='incorrect', ax=ax)
            ax.set_title(f'{cfg_name} | {ds_name} — Score Distributions')
            ax.set_xlabel('Max logit (prediction score)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{out_base}/diagnostics/{safe_name}_scores.png', dpi=150)
            plt.savefig(f'{out_base}/diagnostics/{safe_name}_scores.pdf')
            plt.close()

            # ECDF: model scores vs decoy scores
            fig, ax = plt.subplots(figsize=(6, 4))
            cdf = np.arange(1, n + 1) / n
            ax.plot(np.sort(pred_scores),  cdf, color='blue',   label='model scores', lw=1.5)
            ax.plot(np.sort(decoy_scores), cdf, color='orange',  label='decoy (null)', lw=1.5)
            ax.set_title(f'{cfg_name} | {ds_name} — ECDF')
            ax.set_xlabel('Max logit')
            ax.set_ylabel('Cumulative proportion')
            ax.legend(); ax.grid(linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(f'{out_base}/diagnostics/{safe_name}_ecdf.png', dpi=150)
            plt.savefig(f'{out_base}/diagnostics/{safe_name}_ecdf.pdf')
            plt.close()

            # FDR vs score threshold
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(pred_scores_sorted, QVAL_mixmax, label='Mix-Max FDR')
            ax.plot(pred_scores_sorted, QVAL_true,   label='True FDR')
            ax.set_xlabel('Score threshold')
            ax.set_ylabel('q-value (FDR)')
            ax.set_title(f'{cfg_name} | {ds_name} — FDR vs Threshold')
            ax.legend(); ax.grid()
            plt.tight_layout()
            plt.savefig(f'{out_base}/diagnostics/{safe_name}_fdr_vs_score.png', dpi=150)
            plt.savefig(f'{out_base}/diagnostics/{safe_name}_fdr_vs_score.pdf')
            plt.close()

            # Acc / FDR curves
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(normalized_rank, QVAL_TDC,    label='TDC FDR')
            ax.plot(normalized_rank, QVAL_mixmax, label='Mix-Max FDR')
            ax.plot(normalized_rank, QVAL_true,   label='True FDR')
            ax.plot(normalized_rank, Acc_true,    label='True Acc')
            ax.plot(normalized_rank, Acc_est,     label='ENGPE (TDC)')
            ax.plot(normalized_rank, Acc_est_MM,  label='ENGPE-TA (Mix-Max)')
            ax.set_xlabel('Fraction accepted')
            ax.set_ylabel('Value')
            ax.set_title(f'{cfg_name} | {ds_name}')
            ax.legend(); ax.grid()
            plt.tight_layout()
            plt.savefig(f'{out_base}/accuracy/{safe_name}_acc_fdr.png', dpi=150)
            plt.savefig(f'{out_base}/accuracy/{safe_name}_acc_fdr.pdf')
            plt.close()

        row = dict(
            config=cfg_name, testset=ds_name, testset_idx=testset_idx, n_samples=n,
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

    timing['eval_total_time_s'] = time.perf_counter() - t_eval_start

    if not all_results:
        return dict(config=cfg, results_df=pd.DataFrame(), timing=timing, tile_data=tile_data)

    df = pd.DataFrame(all_results)
    df.to_csv(f'{out_base}/comparison/breeds_{BREEDS_NAME}_{cfg_name}_results.csv', index=False)

    # ── Per-config summary figures ─────────────────────────────────────────────
    groups = {
        'all':         df,
        'clean':       df[df['is_clean']],
        'corruptions': df[~df['is_clean']],
        'source':      df[df['is_source']],
        'target':      df[df['is_target']],
    }

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
        ax.set_title(f'BREEDS {BREEDS_NAME} [{cfg_name}] — MAE [{group_name}]')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(
            f'{out_base}/comparison/breeds_{BREEDS_NAME}_{cfg_name}_{group_name}_bar_error.png',
            dpi=150)
        plt.savefig(
            f'{out_base}/comparison/breeds_{BREEDS_NAME}_{cfg_name}_{group_name}_bar_error.pdf')
        plt.close()

    # Best corruption test set: score dist + acc/FDR summary
    corr_keys = [k for k in tile_data if 'corr_' in k]
    best_key  = (max(corr_keys, key=lambda k: tile_data[k]['acc_st_true'])
                 if corr_keys else max(tile_data, key=lambda k: tile_data[k]['acc_st_true']))
    if best_key in tile_data:
        td = tile_data[best_key]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        bins = np.arange(td['pred_scores'].min() - 0.5,
                         td['pred_scores'].max() + 0.5, 0.05)
        incorrect_mask = td['true_labels'] != td['pred_label']
        sns.histplot(td['pred_scores'],  bins=bins, stat='density', color='blue',
                     kde=True, fill=True, alpha=0.3, label='model scores', ax=ax)
        sns.histplot(td['decoy_scores'], bins=bins, stat='density', color='orange',
                     kde=True, fill=True, alpha=0.3, label='decoy (null)', ax=ax)
        if incorrect_mask.any():
            sns.histplot(td['pred_scores'][incorrect_mask], bins=bins, stat='density',
                         color='red', kde=True, fill=True, alpha=0.3,
                         label='incorrect', ax=ax)
        ax.set_title(f'{best_key}\nScore Distribution')
        ax.set_xlabel('Max logit')
        ax.legend()

        ax = axes[1]
        ax.plot(td['normalized_rank'], td['Acc_true'],   label='True Acc',
                lw=1.5, ls='--', color='steelblue')
        ax.plot(td['normalized_rank'], td['Acc_est_MM'], label='ENGPE-TA',
                lw=1.8, color=COLOR_ENGPE_TA)
        ax.plot(td['normalized_rank'], td['QVAL_true'],  label='True FDR',
                lw=1.5, ls='--', color='gray')
        ax.plot(td['normalized_rank'], td['QVAL_mixmax'],label='ENGPE-TA FDR',
                lw=1.5, color=COLOR_ENGPE_TA, ls=':')
        ax.set_xlabel('Fraction accepted')
        ax.set_ylabel('Value')
        ax.set_title(f'{best_key}\nAcc / FDR curves')
        ax.legend(fontsize=9); ax.grid(linestyle='--', alpha=0.4)

        plt.suptitle(f'{cfg_name} — best corruption: {best_key}', fontsize=13)
        plt.tight_layout()
        plt.savefig(
            f'{out_base}/comparison/breeds_{BREEDS_NAME}_{cfg_name}_best_summary.png', dpi=150)
        plt.savefig(
            f'{out_base}/comparison/breeds_{BREEDS_NAME}_{cfg_name}_best_summary.pdf')
        plt.close()

    # Summary + curves CSVs
    summary_rows = []
    for col_est, col_true, col_err, method, kind in [
        ('acc_st_tdc', 'acc_st_true', 'err_st_tdc', 'ENGPE',    'ST'),
        ('acc_ta_tdc', 'acc_ta_true', 'err_ta_tdc', 'ENGPE',    'TA'),
        ('acc_st_mm',  'acc_st_true', 'err_st_mm',  'ENGPE-TA', 'ST'),
        ('acc_ta_mm',  'acc_ta_true', 'err_ta_mm',  'ENGPE-TA', 'TA'),
    ]:
        v_err = df[col_err].dropna().values
        summary_rows.append(dict(
            config=cfg_name, method=method, type=kind,
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
                config=cfg_name, method=m, type=kind,
                mean_est=df[f'est_{m}'].dropna().values.mean(),
                mean_true=df[col_true].dropna().values.mean(),
                mean_mae=v_err.mean(), std_mae=v_err.std(),
            ))
    pd.DataFrame(summary_rows).to_csv(
        f'{out_base}/comparison/breeds_{BREEDS_NAME}_{cfg_name}_summary.csv',
        index=False, float_format='%.6f')

    N_CURVE_PTS = 100
    curve_rows  = []
    for key, td in tile_data.items():
        n_td = td['n_samples']
        idx  = np.linspace(0, n_td - 1, N_CURVE_PTS, dtype=int)
        for i in idx:
            curve_rows.append(dict(
                dataset='breeds', breeds_name=BREEDS_NAME, config=cfg_name, testset=key,
                logit_threshold=float(td['logit_threshold'][i]),
                frac_accepted=float(td['normalized_rank'][i]),
                acc_true=float(td['Acc_true'][i]),
                acc_est_mm=float(td['Acc_est_MM'][i]),
                acc_est_tdc=float(td['Acc_est'][i]),
                fdr_mixmax=float(td['QVAL_mixmax'][i]),
                fdr_tdc=float(td['QVAL_TDC'][i]),
                fdr_true=float(td['QVAL_true'][i]),
            ))
    pd.DataFrame(curve_rows).to_csv(
        f'{out_base}/comparison/breeds_{BREEDS_NAME}_{cfg_name}_acc_curves.csv',
        index=False, float_format='%.6f')

    print(f"  Config '{cfg_name}' done. → {out_base}/comparison/")
    return dict(config=cfg, results_df=df, timing=timing, tile_data=tile_data)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Data (loaded once)
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

print(f"\n{'='*70}\nBREEDS: {BREEDS_NAME}\n{'='*70}")

(trainset, train_subset, val_subset,
 trainloader, testsets, testloaders) = get_imagenet_breeds(
    batch_size=BATCH_SIZE, data_dir=DATA_DIR, name=BREEDS_NAME, seed=42)

hierarchy_dir = f"{DATA_DIR}/imagenet_class_hierarchy"
ret           = get_breeds_ret(hierarchy_dir, BREEDS_NAME)
NUM_CLASSES   = len(ret[1][0])
valloader     = testloaders[0]
TESTSET_NAMES = build_testset_names(len(testsets))
print(f"NUM_CLASSES (source): {NUM_CLASSES}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Classifier (once, shared across all flow configs)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}\nCLASSIFIER\n{'='*70}")

breeds_model = BREEDSClassifier(
    num_classes=NUM_CLASSES, pretrained=True,
    feature_dim=FEATURE_DIM, freeze_backbone=False,
).to(DEVICE)

if os.path.exists(CLASSIFIER_PATH):
    breeds_model.load_state_dict(
        torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False))
    print(f"Classifier loaded from {CLASSIFIER_PATH}")
    classifier_train_time_s = float('nan')
else:
    print("Training classifier from scratch...")
    set_seeds(42)
    t0 = time.perf_counter()
    train_breeds_classifier(
        model=breeds_model, trainloader=trainloader, valloader=valloader,
        epochs=5, lr=0.01, device=DEVICE, save_path=CLASSIFIER_PATH, patience=3,
    )
    classifier_train_time_s = time.perf_counter() - t0

breeds_model.eval()
val_acc = evaluate_accuracy(breeds_model, valloader, DEVICE)
classifier_inference_ms = measure_classifier_inference_time(breeds_model, valloader, DEVICE)
print(f"Val accuracy (source): {val_acc:.4f}  "
      f"| inference: {classifier_inference_ms:.3f} ms/sample  "
      f"| params: {count_parameters(breeds_model):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train scores, decoy pools, train score dataset (once)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}\nCOLLECTING TRAIN SCORES\n{'='*70}")

train_scores_raw, train_labels_raw = collect_train_scores_breeds(
    breeds_model, train_subset, batch_size=256, device=DEVICE)
print(f"Scores: {train_scores_raw.shape}  "
      f"acc={(train_scores_raw.argmax(1) == train_labels_raw).mean():.4f}")

pool_score, pool_vectors = build_error_conditioned_pools(
    train_scores_raw, train_labels_raw, NUM_CLASSES, verbose=True)

print(f"\nCreating train score dataset (strategy='{DECOY_STRATEGY}')...")
train_score_dataset = create_score_dataset_breeds(
    train_subset, breeds_model,
    pool_score=pool_score, pool_vectors=pool_vectors,
    strategy=DECOY_STRATEGY, device=DEVICE,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Precompute all test score datasets (once)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*70}\nPRECOMPUTING TEST SCORE DATASETS ({len(testsets)} sets)\n{'='*70}")

test_score_datasets = []
for idx, testset in enumerate(testsets):
    print(f"  [{idx+1}/{len(testsets)}] {TESTSET_NAMES[idx]}", end='  ')
    try:
        ds = create_score_dataset_breeds(
            testset, breeds_model,
            pool_score=pool_score, pool_vectors=pool_vectors,
            strategy=DECOY_STRATEGY, device=DEVICE,
        )
        print(f"N={len(ds)}")
    except Exception as e:
        print(f"FAILED: {e}")
        ds = None
    test_score_datasets.append(ds)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Temperature calibration (once, depends only on classifier)
# ─────────────────────────────────────────────────────────────────────────────

source_logits_t = train_score_dataset.cnn_scores.to(DEVICE)
source_labels_t = train_score_dataset.labels.to(DEVICE)
temp            = calibration_temp(source_logits_t, source_labels_t)
scaled_source   = source_logits_t / temp
print(f"\nTemperature calibration: temp={temp:.4f}  "
      f"(N={len(source_labels_t)} train samples)")

os.makedirs('BREEDS/ablation', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Ablation loop (flow only)
# ─────────────────────────────────────────────────────────────────────────────

all_experiment_results = []
all_timing_records     = []

for cfg in CONFIGS_TO_RUN:
    exp = run_flow_experiment(
        cfg=cfg,
        train_score_dataset=train_score_dataset,
        test_score_datasets=test_score_datasets,
        testset_names=TESTSET_NAMES,
        NUM_CLASSES=NUM_CLASSES,
        scaled_source=scaled_source,
        source_labels_t=source_labels_t,
        temp=temp,
        BASELINE_METHODS=dict(BASELINE_METHODS),
    )
    all_experiment_results.append(exp)
    all_timing_records.append(exp['timing'])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Timing summary
# ─────────────────────────────────────────────────────────────────────────────

timing_df = pd.DataFrame(all_timing_records)
# Prepend classifier row (shared baseline)
classifier_row = pd.DataFrame([{
    'config': 'classifier (shared)',
    'seed': 42, 'n_flows': '-', 'encoder_dim': '-', 'use_robust_norm': '-',
    'classifier_train_time_s': classifier_train_time_s,
    'classifier_inference_ms_per_sample': classifier_inference_ms,
    'classifier_n_params': count_parameters(breeds_model),
}])
timing_df.to_csv('BREEDS/ablation/timing_flow.csv', index=False, float_format='%.4f')
classifier_row.to_csv('BREEDS/ablation/timing_classifier.csv', index=False, float_format='%.4f')
print(f"\nTiming saved to BREEDS/ablation/")

print("\n" + timing_df[['config', 'n_flows', 'encoder_dim', 'use_robust_norm',
                         'flow_train_time_s',
                         'flow_inference_ms_per_sample',
                         'flow_n_params']].to_string(index=False))

if len(timing_df) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in [
        (axes[0], 'flow_train_time_s',            'Flow training time (s)'),
        (axes[1], 'flow_inference_ms_per_sample',  'Flow inference (ms/sample)'),
    ]:
        vals = timing_df[col].fillna(0)
        bars = ax.bar(timing_df['config'], vals, color='#1976D2',
                      alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.set_xticklabels(timing_df['config'], rotation=40, ha='right')
        ax.set_ylabel(title.split('(')[1].rstrip(')'))
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0)
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    plt.suptitle(f'BREEDS {BREEDS_NAME} — Flow Compute Cost', fontsize=14)
    plt.tight_layout()
    plt.savefig('BREEDS/ablation/timing_comparison.png', dpi=150)
    plt.savefig('BREEDS/ablation/timing_comparison.pdf')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Ablation comparison figures (across all configs)
# ─────────────────────────────────────────────────────────────────────────────

if len(all_experiment_results) > 1:
    combined_df = pd.concat(
        [r['results_df'] for r in all_experiment_results if len(r['results_df']) > 0],
        ignore_index=True,
    )
    combined_df.to_csv('BREEDS/ablation/all_configs_results.csv',
                       index=False, float_format='%.6f')

    cfg_names   = [r['config']['name'] for r in all_experiment_results
                   if len(r['results_df']) > 0]
    corr_mask   = combined_df['testset'].str.contains('corr_', na=False)
    corr_df     = combined_df[corr_mask]

    # ── A: MAE by config (corruptions) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(cfg_names) * 1.2), 5))
    mae_mm  = [corr_df[corr_df['config'] == c]['err_st_mm'].dropna().mean()  for c in cfg_names]
    std_mm  = [corr_df[corr_df['config'] == c]['err_st_mm'].dropna().std()   for c in cfg_names]
    mae_tdc = [corr_df[corr_df['config'] == c]['err_st_tdc'].dropna().mean() for c in cfg_names]
    std_tdc = [corr_df[corr_df['config'] == c]['err_st_tdc'].dropna().std()  for c in cfg_names]

    x = np.arange(len(cfg_names)); w = 0.35
    ax.bar(x - w/2, mae_mm,  w, yerr=std_mm,  label='ENGPE-TA (Mix-Max)',
           color=COLOR_ENGPE_TA, capsize=4, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.bar(x + w/2, mae_tdc, w, yerr=std_tdc, label='ENGPE (TDC)',
           color=COLOR_ENGPE,    capsize=4, alpha=0.85, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cfg_names, rotation=35, ha='right')
    ax.set_ylabel('Mean Absolute Error (ACC_ST, corruptions)')
    ax.set_title(f'BREEDS {BREEDS_NAME} — Ablation: MAE by Flow Config')
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5); ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('BREEDS/ablation/ablation_mae_by_config.png', dpi=150)
    plt.savefig('BREEDS/ablation/ablation_mae_by_config.pdf')
    plt.close()

    # ── B: Stability across seeds ─────────────────────────────────────────────
    seed_cfgs = [c for c in cfg_names if c.startswith('seed_')]
    if len(seed_cfgs) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, col, color in [
            ('ENGPE-TA', 'err_st_mm',  COLOR_ENGPE_TA),
            ('ENGPE',    'err_st_tdc', COLOR_ENGPE),
        ]:
            means = [corr_df[corr_df['config'] == c][col].dropna().mean() for c in seed_cfgs]
            stds  = [corr_df[corr_df['config'] == c][col].dropna().std()  for c in seed_cfgs]
            ax.errorbar(seed_cfgs, means, yerr=stds, marker='o', capsize=5,
                        label=label, color=color, linewidth=1.8)
        ax.set_xlabel('Flow seed')
        ax.set_ylabel('MAE (ACC_ST, corruptions)')
        ax.set_title(f'BREEDS {BREEDS_NAME} — Flow Stability Across Seeds')
        ax.legend(); ax.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('BREEDS/ablation/stability_seeds.png', dpi=150)
        plt.savefig('BREEDS/ablation/stability_seeds.pdf')
        plt.close()

    # ── C: n_flows ablation ───────────────────────────────────────────────────
    flow_cfgs = [c for c in cfg_names if c.startswith('flows_') or c == 'seed_42']
    flow_nmap = {r['config']['name']: r['config']['n_flows']
                 for r in all_experiment_results}
    if len(flow_cfgs) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, col, color in [
            ('ENGPE-TA', 'err_st_mm',  COLOR_ENGPE_TA),
            ('ENGPE',    'err_st_tdc', COLOR_ENGPE),
        ]:
            pairs = sorted([(flow_nmap[c],
                             corr_df[corr_df['config'] == c][col].dropna().mean(),
                             corr_df[corr_df['config'] == c][col].dropna().std())
                            for c in flow_cfgs], key=lambda x: x[0])
            nv, mv, sv = zip(*pairs)
            ax.errorbar(nv, mv, yerr=sv, marker='o', capsize=5,
                        label=label, color=color, linewidth=1.8)
        ax.set_xlabel('n_flows (coupling layers)')
        ax.set_ylabel('MAE (ACC_ST, corruptions)')
        ax.set_title(f'BREEDS {BREEDS_NAME} — Ablation: n_flows')
        ax.legend(); ax.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('BREEDS/ablation/ablation_n_flows.png', dpi=150)
        plt.savefig('BREEDS/ablation/ablation_n_flows.pdf')
        plt.close()

    # ── D: encoder_dim ablation ───────────────────────────────────────────────
    enc_cfgs = [c for c in cfg_names if c.startswith('enc_') or c == 'seed_42']
    enc_dmap = {r['config']['name']: r['config']['encoder_dim']
                for r in all_experiment_results}
    if len(enc_cfgs) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, col, color in [
            ('ENGPE-TA', 'err_st_mm',  COLOR_ENGPE_TA),
            ('ENGPE',    'err_st_tdc', COLOR_ENGPE),
        ]:
            pairs = sorted([(enc_dmap[c],
                             corr_df[corr_df['config'] == c][col].dropna().mean(),
                             corr_df[corr_df['config'] == c][col].dropna().std())
                            for c in enc_cfgs], key=lambda x: x[0])
            dv, mv, sv = zip(*pairs)
            ax.errorbar(dv, mv, yerr=sv, marker='s', capsize=5,
                        label=label, color=color, linewidth=1.8)
        ax.set_xlabel('Encoder dimension')
        ax.set_ylabel('MAE (ACC_ST, corruptions)')
        ax.set_title(f'BREEDS {BREEDS_NAME} — Ablation: encoder_dim')
        ax.legend(); ax.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('BREEDS/ablation/ablation_encoder_dim.png', dpi=150)
        plt.savefig('BREEDS/ablation/ablation_encoder_dim.pdf')
        plt.close()

    # ── E: normalizer ablation ────────────────────────────────────────────────
    norm_cfgs = [c for c in cfg_names if c in ('seed_42', 'zscore_norm')]
    if len(norm_cfgs) == 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, col, color in [
            ('ENGPE-TA', 'err_st_mm',  COLOR_ENGPE_TA),
            ('ENGPE',    'err_st_tdc', COLOR_ENGPE),
        ]:
            means = [corr_df[corr_df['config'] == c][col].dropna().mean() for c in norm_cfgs]
            stds  = [corr_df[corr_df['config'] == c][col].dropna().std()  for c in norm_cfgs]
            x_pos = np.arange(len(norm_cfgs))
            offset = 0.18 if label == 'ENGPE-TA' else -0.18
            ax.bar(x_pos + offset, means, 0.32, yerr=stds, label=label,
                   color=color, capsize=4, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Robust\n(median/IQR + tanh)', 'Z-score\n(mean/std)'])
        ax.set_ylabel('MAE (ACC_ST, corruptions)')
        ax.set_title(f'BREEDS {BREEDS_NAME} — Normalizer Ablation')
        ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5); ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig('BREEDS/ablation/ablation_normalizer.png', dpi=150)
        plt.savefig('BREEDS/ablation/ablation_normalizer.pdf')
        plt.close()

    # ── F: Score distribution comparison across configs (same test set) ───────
    sets_per_cfg = [set(r['tile_data'].keys()) for r in all_experiment_results
                    if r['tile_data']]
    if sets_per_cfg:
        common_corr = [s for s in set.intersection(*sets_per_cfg) if 'corr_' in s]
        if common_corr:
            ref_cfg = all_experiment_results[0]
            ref_key = max(common_corr,
                          key=lambda k: ref_cfg['tile_data'].get(k, {}).get('acc_st_true', 0))
            n_show  = min(len(all_experiment_results), 4)
            fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4), sharey=True)
            if n_show == 1:
                axes = [axes]
            for ax_i, exp in zip(axes, all_experiment_results[:n_show]):
                td    = exp['tile_data'].get(ref_key)
                cname = exp['config']['name']
                if td is None:
                    ax_i.set_title(f'{cname}\n(no data)')
                    continue
                bins = np.arange(td['pred_scores'].min() - 0.5,
                                 td['pred_scores'].max() + 0.5, 0.05)
                sns.histplot(td['pred_scores'],  bins=bins, stat='density', color='blue',
                             kde=True, fill=True, alpha=0.3, label='model', ax=ax_i)
                sns.histplot(td['decoy_scores'], bins=bins, stat='density', color='orange',
                             kde=True, fill=True, alpha=0.3, label='decoy', ax=ax_i)
                ax_i.set_title(cname)
                ax_i.set_xlabel('Max logit')
                if ax_i is axes[0]:
                    ax_i.set_ylabel('Density')
                ax_i.legend(fontsize=8)
            fig.suptitle(f'Score Distributions — {ref_key}', fontsize=13)
            plt.tight_layout()
            plt.savefig('BREEDS/ablation/score_dist_comparison.png', dpi=150)
            plt.savefig('BREEDS/ablation/score_dist_comparison.pdf')
            plt.close()

print(f"\n{'='*70}")
print(f"BREEDS '{BREEDS_NAME}' ablation complete.")
print(f"  Classifier : {CLASSIFIER_PATH}  (val_acc={val_acc:.4f})")
print(f"  Flow configs run : {len(all_experiment_results)}")
print(f"  Per-config  : BREEDS/<config_name>/comparison/")
print(f"  Ablation    : BREEDS/ablation/")
print(f"{'='*70}")
