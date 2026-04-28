import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr, ks_2samp
import os
from bisect import bisect
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
print(f"Device: {DEVICE}")

os.makedirs('figures/accuracy',      exist_ok=True)
os.makedirs('figures/diagnostics',   exist_ok=True)
os.makedirs('figures/comparison',    exist_ok=True)
os.makedirs('figures/distributions', exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH   = '/kaggle/input/models/arinaromashkina/wide-res-net-1/pytorch/default/1/best_model.pth'
CIFAR_C_PATH = '/kaggle/input/datasets/arinaromashkina/cifar-10-c/CIFAR-10-C'
FLOWS_PATH   = 'cond_flows_model_cifar10_big_new.pth'
NUM_CLASSES  = 10
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
SEVERITIES   = [1, 2, 3, 4, 5]
MIN_POOL     = 30

# Стратегия формирования target для decoy при обучении Flow:
# 'score_coord'  — заменяем только координату pred_class на отрицат. скор из пула
# 'full_vector'  — заменяем весь вектор скоров на вектор другого (неправильного) примера
# 'pool_replace' — как score_coord, но с полным вектором из пула (текущий вариант)
DECOY_STRATEGY = 'score_coord'  # выбери: 'score_coord', 'full_vector', 'pool_replace'

CORRUPTIONS_TO_TEST = [
    'gaussian_noise',           # Гауссов шум
    'shot_noise',               # Дробовой шум
    'impulse_noise',            # Импульсный шум
    'defocus_blur',             # Дефокусировка
    'glass_blur',               # Стеклянное размытие
    'motion_blur',              # Размытие движения
    'zoom_blur',                # Размытие при зуме
    'snow',                     # Снег
    'frost',                    # Иней
    'fog',                      # Туман
    'brightness',               # Яркость
    'contrast',                 # Контраст
    'elastic_transform',        # Эластичная трансформация
    'pixelate',                 # Пикселизация
    'jpeg_compression',         # JPEG-сжатие
    'speckle_noise',            # Спекл-шум
    'gaussian_blur',            # Гауссово размытие
    'spatter',                  # Брызги
    'saturate'                  # Насыщенность
]


cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std  = [0.2470, 0.2435, 0.2616]

print(f"Device: {DEVICE}")
print(f"Decoy strategy: {DECOY_STRATEGY}")

# ============================================================
# TRANSFORMS & DATA LOADING
# ============================================================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

def cifar_c_transform(img: np.ndarray) -> torch.Tensor:
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)
    return transforms.Normalize(cifar_mean, cifar_std)(img)

print("Loading CIFAR-10...")
cifar_train = datasets.CIFAR10('./data', train=True,  download=True,
                                transform=transform_train)
cifar_test  = datasets.CIFAR10('./data', train=False, download=True,
                                transform=transform_test)
print(f"✓ train={len(cifar_train)}, test={len(cifar_test)}")

# ============================================================
# CIFAR-10-C DATASET
# ============================================================
class CIFAR10CDataset:
    """Single severity slice of CIFAR-10-C."""
    def __init__(self, data_path, corruption_type, severity, transform=None):
        all_images = np.load(f"{data_path}/{corruption_type}.npy")
        all_labels = np.load(f"{data_path}/labels.npy")
        s, e = (severity - 1) * 10_000, severity * 10_000
        self.images    = all_images[s:e]
        self.labels    = all_labels[s:e].astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


class CIFAR10CAllSeverities:
    """All 5 severity slices concatenated for one corruption."""
    def __init__(self, data_path, corruption_type, transform=None):
        all_images = np.load(f"{data_path}/{corruption_type}.npy")
        all_labels = np.load(f"{data_path}/labels.npy").astype(np.int64)
        self.images    = all_images
        self.labels    = all_labels
        self.transform = transform
        self.severity_datasets = {
            sev: CIFAR10CDataset(data_path, corruption_type, sev, transform)
            for sev in SEVERITIES
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


print("Loading CIFAR-10-C datasets (19 corruptions × 5 severities = 95 sets)...")
test_corrupted_datasets: dict = {}
for corruption in CORRUPTIONS_TO_TEST:
    for severity in SEVERITIES:
        key = f"{corruption}_s{severity}"
        try:
            test_corrupted_datasets[key] = CIFAR10CDataset(
                CIFAR_C_PATH, corruption, severity, transform=cifar_c_transform)
            print(f"  ✓ {key}  ({len(test_corrupted_datasets[key])} samples)")
        except Exception as exc:
            print(f"  ✗ {key}: {exc}")
print(f"✓ Loaded {len(test_corrupted_datasets)} corrupted datasets")

# ============================================================
# MODEL  (WideResNet must be defined before this cell)
# ============================================================
print("\nLoading WideResNet...")
cifar_model = WideResNet(depth=28, widen_factor=10,
                          dropout_rate=0.3, num_classes=NUM_CLASSES)
cifar_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
cifar_model = cifar_model.to(DEVICE)
cifar_model.eval()
print("✓ Model ready")

# ============================================================
# SCORE FEATURE DATASET
# ============================================================
class ScoreFeatureDataset(torch.utils.data.Dataset):
    """
    Хранит:
      cnn_scores          — логиты модели  [N, C]
      features            — признаки penultimate layer  [N, D]
      target_decoy_scores — decoy-векторы для обучения Flow  [N, C]
      labels              — истинные метки  [N]
    """
    def __init__(self, cnn_scores, features, target_decoy_scores, labels):
        self.cnn_scores          = cnn_scores
        self.features            = features
        self.target_decoy_scores = target_decoy_scores
        self.labels              = labels

    def __len__(self):
        return len(self.cnn_scores)

    def __getitem__(self, idx):
        return (self.cnn_scores[idx], self.features[idx],
                self.target_decoy_scores[idx], self.labels[idx])

# ============================================================
# COLLECT TRAIN SCORES
# ============================================================
def collect_train_scores(model, train_dataset,
                          num_classes: int = 10,
                          device: str = 'cuda') -> tuple:
    """Returns (train_scores [N×C], train_labels [N])."""
    model.eval()
    loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    sc_list, lb_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting train scores'):
            images = images.to(device)
            feats  = model.get_features(images)
            scores = model.linear2(F.relu(model.linear1(feats)))
            sc_list.append(scores.cpu().numpy())
            lb_list.append(labels.numpy())
    return np.concatenate(sc_list), np.concatenate(lb_list)

# ============================================================
# DECOY POOLS
#
# Теория Mix-Max (π_mm = 0):
#   H₀(t) = "предсказанный класс ĉ неверный"
#   Decoy Z(t) должен симулировать P(score_ĉ | argmax=ĉ, label≠ĉ)
#   Т.е. score_c на примерах, где модель предсказала c, но ошиблась.
#   Тогда выполняется exchangeability: f(t)|wrong ~ Z(t)
# ============================================================

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

# Все стратегии симулируют P(score | wrong) для обучения Flow.

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



def create_score_dataset_with_decoys(
        dataset,
        cnn_model,
        pool_score:   dict,
        pool_vectors: dict,
        strategy:     str  = 'score_coord',
        device:       str  = 'cuda') -> 'ScoreFeatureDataset':
    """
    Создаёт ScoreFeatureDataset с target_decoy для обучения Flow.

    strategy:
      'score_coord'  — заменяем только coord pred_class (Strategy A)
      'full_vector'  — заменяем весь вектор (Strategy B)
      'pool_replace' — как score_coord (Strategy C, legacy)

    target_decoy используется только при ОБУЧЕНИИ Flow.
    При инференсе Flow генерирует decoys сам.
    """
    assert strategy in ('score_coord', 'full_vector', 'pool_replace'), \
        f"Unknown strategy: {strategy}"

    cnn_model.eval()
    loader  = DataLoader(dataset, batch_size=256, shuffle=False)
    sc_list, ft_list, dc_list, lb_list = [], [], [], []
    rng     = np.random.default_rng(0)

    with torch.no_grad():
        for images, labels in tqdm(loader,
                                   desc=f'Score dataset [{strategy}]',
                                   leave=False):
            images = images.to(device)
            feats  = cnn_model.get_features(images)
            scores = cnn_model.linear2(F.relu(cnn_model.linear1(feats)))
            sc     = scores.cpu()
            ft     = feats.cpu()
            sc_np  = sc.numpy()

            if strategy == 'score_coord':
                dc_np = _build_decoy_score_coord(sc_np, pool_score, rng)
            elif strategy == 'full_vector':
                dc_np = _build_decoy_full_vector(sc_np, pool_vectors, rng)
            else:   # 'pool_replace'
                dc_np = _build_decoy_pool_replace(sc_np, pool_score, rng)

            dc = torch.from_numpy(dc_np).float()

            sc_list.append(sc)
            ft_list.append(ft)
            dc_list.append(dc)
            lb_list.append(labels)

    return ScoreFeatureDataset(
        torch.cat(sc_list),
        torch.cat(ft_list),
        torch.cat(dc_list),
        torch.cat(lb_list),
    )


def create_score_dataset_no_decoys(dataset, cnn_model,
                                    device: str = 'cuda') -> 'ScoreFeatureDataset':
    """
    Создаёт ScoreFeatureDataset без замены decoys (placeholder = scores).
    Используется там, где decoys генерируются отдельно (инференс Flow).
    """
    cnn_model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    sc_list, ft_list, lb_list = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Score dataset', leave=False):
            images = images.to(device)
            feats  = cnn_model.get_features(images)
            scores = cnn_model.linear2(F.relu(cnn_model.linear1(feats)))
            sc_list.append(scores.cpu())
            ft_list.append(feats.cpu())
            lb_list.append(labels)

    all_scores = torch.cat(sc_list)
    all_feats  = torch.cat(ft_list)
    all_labels = torch.cat(lb_list)

    return ScoreFeatureDataset(
        all_scores, all_feats, all_scores.clone(), all_labels)

print("\n" + "="*70)
print("COLLECTING TRAIN SCORES")
print("="*70)
train_scores_raw, train_labels_raw = collect_train_scores(
    cifar_model, cifar_train, NUM_CLASSES, DEVICE)
print(f"Train scores: {train_scores_raw.shape}  "
      f"acc={(train_scores_raw.argmax(1) == train_labels_raw).mean():.4f}")

print("\n" + "="*70)
print("BUILDING ERROR-CONDITIONED DECOY POOLS")
print("="*70)
pool_score, pool_vectors = build_error_conditioned_pools(
    train_scores_raw, train_labels_raw, NUM_CLASSES, verbose=True)

# ── Логиты train для baselines ────────────────────────────────────────────────
source_logits_np = train_scores_raw
source_labels_np = train_labels_raw

# ============================================================
# FLOW MODEL
# ============================================================
print("\n" + "="*70)
print("FLOW MODEL")
print("="*70)

FEATURE_DIM=640
score_shift_flow = ScoreShiftFlowWrapper(
    num_classes = NUM_CLASSES,
    n_flows     = 12,
    feature_dim = FEATURE_DIM,
    hidden_dim  = 256,
    encoder_dim = 128,
    clip_val    = 5.0,      # ← OOD защита
).to(DEVICE)


# score_shift_flow = ScoreShiftFlowWrapper(
#     num_classes = NUM_CLASSES,
#     n_flows     = 12,
#     feature_dim = 640,
#     hidden_dim  = 256,
#     encoder_dim = 128,
# ).to(DEVICE)

print(f"Creating train score dataset (strategy='{DECOY_STRATEGY}')...")
train_score_dataset = create_score_dataset_with_decoys(
    cifar_train, cifar_model,
    pool_score   = pool_score,
    pool_vectors = pool_vectors,
    strategy     = DECOY_STRATEGY,
    device       = DEVICE,
)

if os.path.exists(FLOWS_PATH):
    score_shift_flow.load_state_dict(
        torch.load(FLOWS_PATH, map_location=DEVICE))
    print(f"✓ Flow loaded from {FLOWS_PATH}")
else:
    print("Training ScoreShiftFlow...")
    score_shift_flow.train_flow(
        train_score_dataset, epochs=30, lr=3e-4,
        batch_size=256, device=DEVICE, patience=5, grad_clip=1.0)
    torch.save(score_shift_flow.state_dict(), FLOWS_PATH)
    print(f"✓ Flow saved to {FLOWS_PATH}")

score_shift_flow.eval()

source_logits, ds_train, source_labels =  ms, ds_flow, ls = score_shift_flow.generate_decoys(
        train_score_dataset, device=DEVICE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr, ks_2samp
import os
from bisect import bisect
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
print(f"Device: {DEVICE}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr, ks_2samp
import os
from bisect import bisect
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch

# ── Global settings ───────────────────────────────────────────────────────────
print(np.__version__)
np.set_printoptions(threshold=np.inf)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18
})

import os
for d in ['figures/distributions', 'figures/diagnostics', 'figures/accuracy', 'figures/summary']:
    os.makedirs(d, exist_ok=True)

COLOR_ENGPE    = '#1976D2'   # blue          — ENGPE
COLOR_ENGPE_TA = '#FF6D00'   # bright orange — ENGPE-TA

# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════
all_results = []
tile_data    = {}

for key, dataset in test_corrupted_datasets.items():
    corruption, sev_str = key.rsplit('_s', 1)
    severity = int(sev_str)
    print(f"\n{'─'*60}")
    print(f"  CORRUPTION: {corruption}  severity={severity}")
    print(f"{'─'*60}")

    score_ds = create_score_dataset_no_decoys(dataset, cifar_model, DEVICE)
    ms, ds_flow, ls = score_shift_flow.generate_decoys(score_ds, device=DEVICE)

    target_logits = torch.tensor(ms).to(DEVICE)
    target_labels = torch.tensor(ls).to(DEVICE)

    true_acc = (target_logits.argmax(1) == target_labels).float().mean().item()

    temp          = calibration_temp(source_logits, source_labels)
    scaled_source = source_logits / temp
    scaled_target = target_logits / temp

    # ── Baseline methods ──────────────────────────────────────────────────
    baseline_estimates = {}
    baseline_errors    = {}

    for method_name, method_func in BASELINE_METHODS.items():
        estimate = method_func(scaled_source, source_labels, scaled_target)
        error    = abs(estimate - true_acc)
        baseline_estimates[method_name] = float(estimate)
        baseline_errors[method_name]    = float(error)
        print(f"  {method_name}: {estimate:.3f} (error: {error:.3f})")
        print(f"{'─'*60}")

    print(f"{'─'*60}")
    true_acc_full = float((ms.argmax(axis=1) == ls).mean())
    bin_width  = 0.05
    bins       = np.arange(-3, 8, bin_width)
    stat       = 'density'
    plot_alpha = 0.3

    true_labels = ls
    n_samples   = len(true_labels)

    pred_scores  = np.max(ms, axis=1)
    pred_label   = np.argmax(ms, axis=1)
    decoy_scores = np.max(ds_flow, axis=1)

    probs_np      = F.softmax(torch.tensor(ms).float(), dim=1).numpy()
    mano_fro      = np.linalg.norm(probs_np, ord='fro') / np.sqrt(n_samples)
    mano_fro_norm = np.clip(
        (mano_fro - 1.0 / np.sqrt(NUM_CLASSES)) / (1.0 - 1.0 / np.sqrt(NUM_CLASSES)),
        0.0, 1.0)

    # ── Figure 1: Score distributions ────────────────────────────────────
    plt.figure(figsize=(6, 4))
    sns.histplot(pred_scores,
                 bins=bins, stat=stat, color='blue', kde=True,
                 fill=True, alpha=plot_alpha, label='model_mixture')
    sns.histplot(decoy_scores,
                 bins=bins, stat=stat, color='orange', kde=True,
                 fill=True, alpha=plot_alpha, label='null')
    sns.histplot(pred_scores[true_labels != pred_label],
                 bins=bins, stat=stat, color='red', kde=True,
                 fill=True, alpha=plot_alpha, label='incorrect')
    plt.title(f'{corruption} s{severity} — Score Distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/distributions/{key}_scores.png', dpi=300)
    plt.savefig(f'figures/distributions/{key}_scores.pdf')
    plt.show()

    # ── True FDR ─────────────────────────────────────────────────────────
    sort_idx           = np.argsort(pred_scores)
    pred_scores_sorted = pred_scores[sort_idx]
    label_sorted       = true_labels[sort_idx]
    pred_label_sorted  = pred_label[sort_idx]
    correct_pred       = (label_sorted == pred_label_sorted).astype(int)

    print(f"  accuracy : {correct_pred.sum() / n_samples:.4f}")
    print(f"  error/pi0: {1 - correct_pred.sum() / n_samples:.4f}")

    FD       = 1 - correct_pred
    FD_CF    = np.cumsum(FD[::-1])[::-1]
    D_CF     = np.arange(0, n_samples)[::-1] + 1
    FDR_true = np.clip(FD_CF / D_CF, 0, 1)
    QVAL_true = np.minimum.accumulate(FDR_true)
    QVAL_true = np.clip(QVAL_true, 0, 1)

    # ── Mix-Max FDR ───────────────────────────────────────────────────────
    n   = n_samples
    pi0 = 0.0

    unique_z_vals, counts_z = np.unique(decoy_scores, return_counts=True)
    n_unique_z    = len(unique_z_vals)
    sorted_decoys = np.sort(decoy_scores)

    counts_w_leq_z = np.searchsorted(pred_scores_sorted, unique_z_vals, side='left')
    counts_z_leq_z = np.searchsorted(sorted_decoys,      unique_z_vals, side='left')

    P_W_leq_z = np.clip((counts_w_leq_z - pi0 * counts_z_leq_z) / ((1 - pi0) * n), 0, 1)
    P_Y_leq_z = np.clip(counts_z_leq_z / n, 0, 1)

    R_j = np.divide(P_W_leq_z, P_Y_leq_z,
                    out=np.zeros_like(P_W_leq_z),
                    where=P_Y_leq_z > 0)
    R_j = np.clip(R_j, 0, 1)

    all_thresholds = pred_scores_sorted[::-1]
    fdr_values     = np.zeros(n)

    for i, T in enumerate(all_thresholds):
        D     = i + 1
        F_0   = pi0 * np.sum(decoy_scores > T)
        z_idx = np.searchsorted(unique_z_vals, T, side='left')
        F_1   = 0.0 if z_idx >= n_unique_z else (1 - pi0) * np.sum(R_j[z_idx:] * counts_z[z_idx:])
        fdr_values[i] = (F_0 + F_1) / D if D > 0 else 0

    QVAL_mixmax = np.clip(np.minimum.accumulate(fdr_values[::-1]), 0, 1)

    # ── Figure 2: FDR vs score threshold ─────────────────────────────────
    plt.figure(figsize=(6, 4))
    plt.plot(pred_scores_sorted, QVAL_mixmax, label='Mix-Max FDR')
    plt.plot(pred_scores_sorted, QVAL_true,   label='True FDR')
    plt.xlabel('Score threshold')
    plt.ylabel('q-value (FDR)')
    plt.title(f'{corruption} s{severity} — FDR vs Score Threshold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figures/diagnostics/{key}_fdr_vs_score.png', dpi=300)
    plt.savefig(f'figures/diagnostics/{key}_fdr_vs_score.pdf')
    plt.show()

    # ── TDC FDR ───────────────────────────────────────────────────────────
    TDC       = (pred_scores > decoy_scores).astype(int)
    TDC_score = np.maximum(pred_scores, decoy_scores)

    tdc_sort_idx = np.argsort(TDC_score)
    TDC_score_s  = TDC_score[tdc_sort_idx]
    TDC_label_s  = TDC[tdc_sort_idx]

    FD_CF   = np.cumsum((1 - TDC_label_s)[::-1])[::-1]
    D_CF    = np.maximum(np.arange(0, n_samples)[::-1] + 1 - FD_CF, 1)
    TDC_FDR = np.clip(FD_CF / D_CF, 0, 1)
    QVAL_TDC = np.clip(np.minimum.accumulate(TDC_FDR), 0, 1)

    # ── Acc estimation ────────────────────────────────────────────────────
    pi0_tdc = np.clip(QVAL_TDC[0],   0.0, 1.0)
    pi0_mm  = np.clip(QVAL_mixmax[0], 0.0, 1.0)

    Acc_est    = np.zeros(n)
    Acc_est_MM = np.zeros(n)
    Acc_true   = np.zeros(n)

    for i in range(n):
        TP_true     = correct_pred[i:].sum()
        TN_true     = (1 - correct_pred[:i]).sum()
        Acc_true[i] = (TP_true + TN_true) / n

        accepted       = n - i
        FP             = accepted * QVAL_TDC[i]
        TP             = accepted * (1 - QVAL_TDC[i])
        TN             = n * pi0_tdc - FP
        Acc_est[i]     = np.clip((TP + TN) / n, 0.0, 1.0)

        FP             = accepted * QVAL_mixmax[i]
        TP             = accepted * (1 - QVAL_mixmax[i])
        TN             = n * pi0_mm - FP
        Acc_est_MM[i]  = np.clip((TP + TN) / n, 0.0, 1.0)

    Acc_true = np.clip(Acc_true, 0.0, 1.0)

    normalized_rank = np.arange(n) / n
    total_TP        = int(correct_pred.sum())
    TP_from_i       = np.cumsum(correct_pred[::-1])[::-1]
    D_from_i        = np.arange(n, 0, -1)

    tile_data[key] = dict(
        normalized_rank = normalized_rank.copy(),
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
        n_samples       = n,
        corruption      = corruption,
        severity        = severity,
    )

    # ── Figure 3: Acc & FDR vs normalised rank ───────────────────────────
    plt.figure(figsize=(6, 4))
    plt.plot(normalized_rank, QVAL_TDC,    label='TDC FDR')
    plt.plot(normalized_rank, QVAL_mixmax, label='Mix-Max FDR')
    plt.plot(normalized_rank, QVAL_true,   label='True FDR')
    plt.plot(normalized_rank, Acc_true,    label='True Acc')
    plt.plot(normalized_rank, Acc_est,     label='Est Acc with TDC')
    plt.plot(normalized_rank, Acc_est_MM,  label='Est Acc with Mix-Max')
    plt.xlabel('Normalized rank (fraction accepted)')
    plt.ylabel('Value')
    plt.title(f'{corruption} s{severity}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figures/accuracy/{key}_acc_fdr.png', dpi=300)
    plt.savefig(f'figures/accuracy/{key}_acc_fdr.pdf')
    plt.show()

    # ── ACC_ST / ACC_TA metrics ───────────────────────────────────────────
    acc_st_true    = Acc_true[0]
    acc_ta_true    = Acc_true.max()

    acc_st_est_tdc = Acc_est[0]
    acc_ta_est_tdc = Acc_est.max()

    acc_st_est_mm  = Acc_est_MM[0]
    acc_ta_est_mm  = Acc_est_MM.max()

    err_st_tdc = abs(acc_st_est_tdc - acc_st_true)
    err_ta_tdc = abs(acc_ta_est_tdc - acc_ta_true)
    err_st_mm  = abs(acc_st_est_mm  - acc_st_true)
    err_ta_mm  = abs(acc_ta_est_mm  - acc_ta_true)

    baseline_errors_ta = {
        method_name: abs(baseline_estimates[method_name] - acc_ta_true)
        for method_name in BASELINE_METHODS
    }

    print(f"\n  {'':28} {'ACC_ST':>10}  {'ACC_TA':>10}")
    print(f"  {'─'*52}")
    print(f"  {'True':<28} {acc_st_true:>10.4f}  {acc_ta_true:>10.4f}")
    print(f"  {'TDC  (est | err)':<28} "
          f"{acc_st_est_tdc:>6.4f}  {err_st_tdc:>+.4f}  "
          f"{acc_ta_est_tdc:>6.4f}  {err_ta_tdc:>+.4f}")
    print(f"  {'Mix-Max (est | err)':<28} "
          f"{acc_st_est_mm:>6.4f}  {err_st_mm:>+.4f}  "
          f"{acc_ta_est_mm:>6.4f}  {err_ta_mm:>+.4f}")

    # ── Collect results ───────────────────────────────────────────────────
    row = dict(
        corruption     = corruption,
        severity       = severity,
        true_acc       = true_acc,
        mano_fro_norm  = mano_fro_norm,
        acc_st_true    = acc_st_true,
        acc_ta_true    = acc_ta_true,
        acc_st_tdc     = acc_st_est_tdc,
        acc_ta_tdc     = acc_ta_est_tdc,
        err_st_tdc     = err_st_tdc,
        err_ta_tdc     = err_ta_tdc,
        acc_st_mm      = acc_st_est_mm,
        acc_ta_mm      = acc_ta_est_mm,
        err_st_mm      = err_st_mm,
        err_ta_mm      = err_ta_mm,
    )
    for method_name in BASELINE_METHODS:
        row[f'est_{method_name}']    = baseline_estimates[method_name]
        row[f'err_{method_name}']    = baseline_errors[method_name]
        row[f'err_ta_{method_name}'] = baseline_errors_ta[method_name]

    all_results.append(row)


# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
if all_results:
    df = pd.DataFrame(all_results)

    # ── Save raw results to CSV ───────────────────────────────────────────
    df.to_csv('figures/results_per_corruption.csv', index=False, float_format='%.6f')
    print("\nSaved per-corruption results → figures/results_per_corruption.csv")

    # Save subsampled acc/fdr curves for ALL corruptions (100 pts each)
    # for later cross-dataset plots
    N_CURVE_PTS = 100
    curve_rows  = []
    for td_key, td in tile_data.items():
        n   = td['n_samples']
        idx = np.linspace(0, n - 1, N_CURVE_PTS, dtype=int)
        for i in idx:
            curve_rows.append(dict(
                dataset       = 'cifar10c',
                corruption    = td['corruption'],
                severity      = td['severity'],
                frac_accepted = float(td['normalized_rank'][i]),
                acc_true      = float(td['Acc_true'][i]),
                acc_est_mm    = float(td['Acc_est_MM'][i]),
                acc_est_tdc   = float(td['Acc_est'][i]),
                fdr_mixmax    = float(td['QVAL_mixmax'][i]),
                fdr_tdc       = float(td['QVAL_TDC'][i]),
                fdr_true      = float(td['QVAL_true'][i]),
            ))
    pd.DataFrame(curve_rows).to_csv(
        'figures/results_acc_curves.csv', index=False, float_format='%.6f')
    print("Saved acc curves            → figures/results_acc_curves.csv")

    # ── Build summary table (mean ± std) ─────────────────────────────────
    summary_rows = []

    acc_report = [
        ('acc_st_true', 'True ACC_ST',         'True',    'ST'),
        ('acc_ta_true', 'True ACC_TA',          'True',    'TA'),
        ('acc_st_tdc',  'ENGPE est ACC_ST',     'ENGPE',   'ST'),
        ('acc_ta_tdc',  'ENGPE est ACC_TA',     'ENGPE',   'TA'),
        ('acc_st_mm',   'ENGPE-TA est ACC_ST',  'ENGPE-TA','ST'),
        ('acc_ta_mm',   'ENGPE-TA est ACC_TA',  'ENGPE-TA','TA'),
    ]
    for col, name, method, kind in acc_report:
        v = df[col].values
        summary_rows.append(dict(metric=name, method=method, type=kind,
                                 mean=v.mean(), std=v.std()))

    for method_name in BASELINE_METHODS:
        col = f'est_{method_name}'
        v   = df[col].values
        summary_rows.append(dict(metric=f'{method_name} est ACC',
                                 method=method_name, type='ST',
                                 mean=v.mean(), std=v.std()))

    err_report = [
        ('err_st_tdc', 'MAE ENGPE vs ACC_ST',    'ENGPE',    'ST'),
        ('err_ta_tdc', 'MAE ENGPE vs ACC_TA',    'ENGPE',    'TA'),
        ('err_st_mm',  'MAE ENGPE-TA vs ACC_ST', 'ENGPE-TA', 'ST'),
        ('err_ta_mm',  'MAE ENGPE-TA vs ACC_TA', 'ENGPE-TA', 'TA'),
    ]
    for col, name, method, kind in err_report:
        v = df[col].values
        summary_rows.append(dict(metric=name, method=method, type=kind,
                                 mean=v.mean(), std=v.std()))

    for method_name in BASELINE_METHODS:
        for col_key, kind, true_type in [(f'err_{method_name}',    'ST', 'ACC_ST'),
                                         (f'err_ta_{method_name}', 'TA', 'ACC_TA')]:
            v = df[col_key].values
            summary_rows.append(dict(metric=f'MAE {method_name} vs {true_type}',
                                     method=method_name, type=kind,
                                     mean=v.mean(), std=v.std()))

    # MANO
    v_mano = df['mano_fro_norm'].values
    summary_rows.append(dict(metric='MANO est ACC (Frobenius norm)',
                             method='MANO', type='ST',
                             mean=v_mano.mean(), std=v_mano.std()))
    mae_mano = (df['mano_fro_norm'] - df['acc_st_true']).abs().values
    summary_rows.append(dict(metric='MAE MANO vs ACC_ST',
                             method='MANO', type='ST',
                             mean=mae_mano.mean(), std=mae_mano.std()))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('figures/results_summary.csv', index=False, float_format='%.6f')
    print("Saved summary results    → figures/results_summary.csv")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("AGGREGATE SUMMARY  (mean ± std across corruptions)")
    print("="*70)
    print(f"\n  {'Metric':<38} {'Mean':>8}   {'Std':>8}   {'Type':>6}")
    print(f"  {'─'*65}")
    for r in summary_rows:
        print(f"  {r['metric']:<38} {r['mean']:>8.4f}   {r['std']:>8.4f}   {r['type']:>6}")

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 1 (summary): True ACC_TA vs True ACC_ST
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    lo = min(df['acc_st_true'].min(), df['acc_ta_true'].min()) - 0.02
    hi = max(df['acc_st_true'].max(), df['acc_ta_true'].max()) + 0.02

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y', zorder=2)
    ax.scatter(
        df['acc_st_true'].values,
        df['acc_ta_true'].values,
        color='steelblue', s=60, alpha=0.85, zorder=3
    )
    ax.set_xlabel('True ACC ST')
    ax.set_ylabel('True ACC TA')
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig('figures/summary/scatter_true_accst_vs_accta.png', dpi=300)
    plt.savefig('figures/summary/scatter_true_accst_vs_accta.pdf')
    plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 2 (summary): Estimated Accuracy vs True Accuracy
    # All competitors + ENGPE (true=ACC_ST) + ENGPE-TA (true=ACC_TA)
    # ─────────────────────────────────────────────────────────────────────

    # Build color palette for baselines
    baseline_list    = list(BASELINE_METHODS.keys())
    n_base           = len(baseline_list)
    baseline_palette = plt.cm.tab10(np.linspace(0, 0.9, n_base))
    baseline_markers = ['D', '^', 'v', 'P', 'X', '*', 'h', '8']

    # Collect all values to determine axis range
    all_scatter_vals = []
    all_scatter_vals.extend(df['acc_st_true'].values.tolist())
    all_scatter_vals.extend(df['acc_ta_true'].values.tolist())
    all_scatter_vals.extend(df['acc_st_tdc'].values.tolist())
    all_scatter_vals.extend(df['acc_ta_mm'].values.tolist())
    for method_name in baseline_list:
        all_scatter_vals.extend(df[f'est_{method_name}'].values.tolist())

    lo2 = min(all_scatter_vals) - 0.02
    hi2 = max(all_scatter_vals) + 0.02

    fig, ax = plt.subplots(figsize=(6, 4))

    # Diagonal reference
    ax.plot([lo2, hi2], [lo2, hi2], 'r--', linewidth=1.5,
            label='x=y', zorder=2)

    # ── Baselines (True = ACC_ST, same as competitors compare vs full acc) ─
    for idx, method_name in enumerate(baseline_list):
        ax.scatter(
            df['acc_st_true'].values,          # true accuracy axis (x)
            df[f'est_{method_name}'].values,   # estimated accuracy axis (y)
            label  = method_name,
            marker = baseline_markers[idx % len(baseline_markers)],
            color  = baseline_palette[idx],
            s      = 25,
            alpha  = 0.70,
            zorder = 3
        )

    # ── ENGPE: true accuracy = ACC_ST ────────────────────────────────────
    ax.scatter(
        df['acc_st_true'].values,   # x = true
        df['acc_st_tdc'].values,    # y = estimated
        label  = 'ENGPE',
        marker = 'o',
        color  = COLOR_ENGPE,
        s      = 35,
        alpha  = 0.80,
        zorder = 4
    )

    # ── ENGPE-TA: true accuracy = ACC_TA (brighter / larger) ─────────────
    ax.scatter(
        df['acc_ta_true'].values,   # x = true
        df['acc_ta_mm'].values,     # y = estimated
        label  = 'ENGPE-TA',
        marker = 's',
        color  = COLOR_ENGPE_TA,
        s      = 50,
        alpha  = 1.0,
        zorder = 5
    )

    ax.set_xlabel('True Accuracy')
    ax.set_ylabel('Estimated Accuracy')
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)
    ax.set_aspect('equal', 'box')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('figures/summary/scatter_estimated_vs_true_all.png', dpi=300)
    plt.savefig('figures/summary/scatter_estimated_vs_true_all.pdf')
    plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 3 (summary): Bar chart — MAE per method (mean ± std)
    # ─────────────────────────────────────────────────────────────────────
    bar_entries = []

    # Our methods
    bar_entries.append(('ENGPE\n(ST)',    'err_st_tdc', '#2196F3'))
    bar_entries.append(('ENGPE\n(TA)',    'err_ta_tdc', '#1565C0'))
    bar_entries.append(('ENGPE-TA\n(ST)', 'err_st_mm',  '#FF9800'))
    bar_entries.append(('ENGPE-TA\n(TA)', 'err_ta_mm',  '#E65100'))

    # Baselines (vs ACC_ST)
    for idx, method_name in enumerate(baseline_list):
        bar_entries.append((f'{method_name}\n(ST)',
                            f'err_{method_name}',
                            baseline_palette[idx]))

    labels_bar = [e[0] for e in bar_entries]
    means_bar  = [df[e[1]].mean() for e in bar_entries]
    stds_bar   = [df[e[1]].std()  for e in bar_entries]
    colors_bar = [e[2]            for e in bar_entries]

    fig, ax = plt.subplots(figsize=(max(8, len(bar_entries) * 1.0), 4))
    x_pos = np.arange(len(bar_entries))

    ax.bar(x_pos, means_bar, yerr=stds_bar,
           color=colors_bar, capsize=4, alpha=0.85,
           error_kw=dict(elinewidth=1.2, ecolor='black'))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_bar, rotation=30, ha='right')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Method')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    legend_elements = [
        mpatches.Patch(facecolor='#2196F3', alpha=0.85, label='ENGPE (ST)'),
        mpatches.Patch(facecolor='#1565C0', alpha=0.85, label='ENGPE (TA)'),
        mpatches.Patch(facecolor='#FF9800', alpha=0.85, label='ENGPE-TA (ST)'),
        mpatches.Patch(facecolor='#E65100', alpha=0.85, label='ENGPE-TA (TA)'),
        mpatches.Patch(facecolor='grey',    alpha=0.85, label='Baseline (ST)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/summary/bar_mae_per_method.png', dpi=300)
    plt.savefig('figures/summary/bar_mae_per_method.pdf')
    plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 4 (summary): MANO Frobenius score vs True Accuracy
    # ─────────────────────────────────────────────────────────────────────
    lo_m = min(df['acc_st_true'].min(), df['mano_fro_norm'].min()) - 0.02
    hi_m = max(df['acc_st_true'].max(), df['mano_fro_norm'].max()) + 0.02

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([lo_m, hi_m], [lo_m, hi_m], 'r--', linewidth=1.5, label='x=y', zorder=2)
    ax.scatter(
        df['acc_st_true'].values,
        df['mano_fro_norm'].values,
        label='MANO (Frobenius)', marker='D',
        color='#5C6BC0', s=25, alpha=0.75, zorder=3
    )
    ax.set_xlabel('True Accuracy')
    ax.set_ylabel('MANO Score (normalized Frobenius)')
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig('figures/summary/scatter_mano.png', dpi=300)
    plt.savefig('figures/summary/scatter_mano.pdf')
    plt.show()
    plt.close()
    print("✓ scatter_mano")

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 5: Acc & FDR vs rank for the best corruption (min err_ta_mm)
    # ─────────────────────────────────────────────────────────────────────
    best_key = df.loc[df['err_ta_mm'].idxmin(), 'corruption']
    print(f"\nBest corruption (min err_ta_mm): {best_key}")

    if best_key in tile_data:
        td = tile_data[best_key]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(td['normalized_rank'], td['Acc_true'],    label='True Acc',     linewidth=1.5, linestyle='--', color='steelblue')
        ax.plot(td['normalized_rank'], td['QVAL_true'],   label='True FDR',     linewidth=1.5, linestyle='--', color='#78909C')
        ax.plot(td['normalized_rank'], td['Acc_est_MM'],  label='ENGPE-TA',     linewidth=1.8, color=COLOR_ENGPE_TA)
        ax.plot(td['normalized_rank'], td['QVAL_mixmax'], label='ENGPE-TA FDR', linewidth=1.5, color=COLOR_ENGPE_TA, linestyle=':')
        ax.set_xlabel('Fraction accepted')
        ax.set_ylabel('Value')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('figures/summary/best_corruption_acc_fdr.png', dpi=300)
        plt.savefig('figures/summary/best_corruption_acc_fdr.pdf')
        plt.show()
        plt.close()
        print("✓ best_corruption_acc_fdr")

        # ─────────────────────────────────────────────────────────────────
        # FIGURE 6: Precision-Recall curve — true vs estimated (best corruption)
        # ─────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(td['recall_true'], td['precision_true'],
                color='steelblue', linewidth=1.8, label='True PR')
        ax.plot(td['recall_est'],  td['precision_est'],
                color=COLOR_ENGPE_TA, linewidth=1.8, linestyle='--',
                label='ENGPE-TA PR (estimated)')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=10)
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('figures/summary/best_corruption_pr_curve.png', dpi=300)
        plt.savefig('figures/summary/best_corruption_pr_curve.pdf')
        plt.show()
        plt.close()
        print("✓ best_corruption_pr_curve")

    print("\nAll figures saved.")
    print(f"Full table:\n{df.to_string(index=False, float_format='{:.4f}'.format)}")