import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob

from data_processing.score_feature_dataset import *
from data_processing.negative_scores_pool import *
from flows.shift_flow import ScoreShiftFlowWrapper
from utils.other_methods import *
from utils.visualize_distributions import *
from fdr.fdr_control import *
from fdr.plot_fdr import *

try:
    from utils.other_methods import predict_COT
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES    = 5
FEATURE_DIM    = 64
PATH_DATA      = '../../data/BCSS/training/bcss.mini.training.torch'
TEST_FOLDER    = '../../data/BCSS/test/'
FLOWS_PATH     = 'BCSS/bcss_score_shift_flow_normalize.pth'
SUBSAMPLE_STEP = 10
MIN_PIXELS     = 50        # минимум пикселей чтобы считать threshold-accuracy

os.makedirs('BCSS/figures/accuracy',    exist_ok=True)
os.makedirs('BCSS/figures/diagnostics', exist_ok=True)
os.makedirs('BCSS/figures/comparison',  exist_ok=True)

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 10, 'figure.titlesize': 18,
})

BASELINE_METHODS = {
    'ATC'   : predict_ATC_maxconf,
    'ATC-NE': predict_ATC_negent,
    'AC'    : predict_AC,
    'DOC'   : predict_DOC,
}
if COT_AVAILABLE:
    BASELINE_METHODS['COT'] = predict_COT

COLORS = {
    'Mix-Max': '#E53935',
    'ATC'    : '#1976D2',
    'ATC-NE' : '#0288D1',
    'AC'     : '#388E3C',
    'DOC'    : '#7B1FA2',
    'COT'    : '#F57C00',
}

print(f"Device    : {DEVICE}")
print(f"Baselines : {list(BASELINE_METHODS.keys())}")


# ============================================================================
# HELPERS
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class RobustFeatureNormalizer(nn.Module):
    """
    Робастная нормализация признаков для OOD тест-данных.
    
    Алгоритм:
      1. Накапливает median и IQR на train (устойчиво к выбросам)
      2. Нормализует: x_norm = clip((x - median) / IQR, -clip_val, clip_val)
      3. После clip — tanh сжимает всё в (-1, 1) без потери информации
    
    Всё сохраняется в state_dict как буферы.
    """
    def __init__(self, feature_dim, clip_val=5.0, momentum=0.01, eps=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.clip_val    = clip_val
        self.momentum    = momentum
        self.eps         = eps

        # Робастные статистики (median + IQR вместо mean + std)
        self.register_buffer('running_median', torch.zeros(feature_dim))
        self.register_buffer('running_iqr',    torch.ones(feature_dim))
        self.register_buffer('initialized',    torch.tensor(False))

    @torch.no_grad()
    def _update_stats(self, x):
        """Обновляем running median и IQR через EMA."""
        batch_median = x.median(dim=0).values

        q75 = torch.quantile(x, 0.75, dim=0)
        q25 = torch.quantile(x, 0.25, dim=0)
        batch_iqr = (q75 - q25).clamp(min=self.eps)

        if not self.initialized:
            self.running_median.copy_(batch_median)
            self.running_iqr.copy_(batch_iqr)
            self.initialized.fill_(True)
        else:
            self.running_median.mul_(1 - self.momentum).add_(
                batch_median * self.momentum)
            self.running_iqr.mul_(1 - self.momentum).add_(
                batch_iqr * self.momentum)

    def forward(self, x):
        if self.training:
            self._update_stats(x)

        if not self.initialized:
            # Если статистика ещё не собрана — просто tanh
            return torch.tanh(x * 0.01)

        # 1. Центрируем и масштабируем по IQR
        x_norm = (x - self.running_median) / (self.running_iqr + self.eps)

        # 2. Clip жёсткий — убивает совсем дикие выбросы
        x_norm = x_norm.clamp(-self.clip_val, self.clip_val)

        # 3. Tanh — мягкое сжатие в (-1, 1), сохраняет градиенты
        x_norm = torch.tanh(x_norm / self.clip_val)

        return x_norm


class ActNorm(nn.Module):
    """
    Activation normalization — data-driven per-channel scale+bias.
    Initialized from the first FORWARD batch so output has mean=0, std=1.
    `initialized` is registered as a buffer so it is saved/loaded with state_dict.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim       = dim
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias      = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, x, reverse=False):
        if not self.initialized and not reverse:
            with torch.no_grad():
                self.bias.data      = -x.mean(0)
                self.log_scale.data = -x.std(0).clamp(min=1e-6).log()
            self.initialized.fill_(True)

        if not reverse:
            y       = (x + self.bias) * self.log_scale.exp()
            log_det = self.log_scale.sum().expand(x.size(0))
            return y, log_det
        else:
            x_rec   = x * (-self.log_scale).exp() - self.bias
            log_det = -self.log_scale.sum().expand(x.size(0))
            return x_rec, log_det


class CouplingLayer(nn.Module):
    """
    Affine coupling layer operating on the full score vector.
    Conditioned on encoded CNN features.

    Split  : x1 = x[:d],  x2 = x[d:]
    Forward: z2 = x2 * exp(s(x1, feat)) + t(x1, feat)
    Reverse: x2 = (z2 - t) * exp(-s)

    Scale head uses Tanh so s ∈ (-1, 1) — prevents exp blow-up.
    """
    def __init__(self, dim, feature_dim, hidden_dim=256, mask_type='first_half'):
        super().__init__()
        self.dim       = dim
        self.mask_type = mask_type

        if mask_type == 'first_half':
            self.d_in  = dim // 2
            self.d_out = dim - dim // 2
        else:
            self.d_in  = dim - dim // 2
            self.d_out = dim // 2

        net_in = self.d_in + feature_dim

        self.net = nn.Sequential(
            nn.Linear(net_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, self.d_out),
            nn.Tanh(),
        )
        self.translate_head = nn.Linear(hidden_dim // 2, self.d_out)

        # Initialize to identity transform
        nn.init.zeros_(self.scale_head[0].weight)
        nn.init.zeros_(self.scale_head[0].bias)
        nn.init.zeros_(self.translate_head.weight)
        nn.init.zeros_(self.translate_head.bias)

    def _split(self, x):
        if self.mask_type == 'first_half':
            return x[:, :self.d_in], x[:, self.d_in:]
        else:
            return x[:, self.d_out:], x[:, :self.d_out]

    def _merge(self, x1, x2):
        if self.mask_type == 'first_half':
            return torch.cat([x1, x2], dim=1)
        else:
            return torch.cat([x2, x1], dim=1)

    def forward(self, x, features, reverse=False):
        x1, x2     = self._split(x)
        cond_input = torch.cat([x1, features], dim=1)
        h          = self.net(cond_input)
        s          = self.scale_head(h)
        t          = self.translate_head(h)

        if not reverse:
            z2      = x2 * torch.exp(s) + t
            log_det = s.sum(dim=1)
            return self._merge(x1, z2), log_det
        else:
            x2_rec  = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=1)
            return self._merge(x1, x2_rec), log_det


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────

class ScoreShiftFlow(nn.Module):
    """
    Single normalizing flow that models P(decoy_score_vector | features).

    Architecture:
      - RobustFeatureNormalizer: median/IQR + clamp + tanh → always ∈ (-1, 1)
      - Feature encoder: feature_dim → encoder_dim
      - n_flows CouplingLayers with alternating masks
      - ActNorm between every pair of coupling layers
    """
    def __init__(self,
                 score_dim   = 10,
                 feature_dim = 640,
                 n_flows     = 12,
                 hidden_dim  = 256,
                 encoder_dim = 128,
                 clip_val    = 5.0):
        super().__init__()
        self.score_dim   = score_dim
        self.feature_dim = feature_dim
        self.n_flows     = n_flows
        self._log_2pi    = float(np.log(2 * np.pi))

        # ── Робастная нормализация признаков ─────────────────────────────
        # Любые OOD значения → выход всегда ∈ (-1, 1)
        self.feature_norm = RobustFeatureNormalizer(
            feature_dim, clip_val=clip_val, momentum=0.01)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
        )

        # Alternating coupling layers + actnorm
        self.layers = nn.ModuleList()
        for i in range(n_flows):
            mask = 'first_half' if i % 2 == 0 else 'second_half'
            self.layers.append(
                CouplingLayer(score_dim, encoder_dim, hidden_dim, mask)
            )
            if i < n_flows - 1:
                self.layers.append(ActNorm(score_dim))

    def encode(self, features):
        # Нормализуем → всегда ∈ (-1, 1) даже при OOD в миллион раз
        features_norm = self.feature_norm(features)
        return self.feature_encoder(features_norm)

    def forward(self, scores, features, reverse=False):
        """
        reverse=False : scores → latent z,  returns (z, log_det)
        reverse=True  : latent z → scores,  returns (scores, log_det)
        """
        enc         = self.encode(features)
        log_det_sum = torch.zeros(scores.size(0), device=scores.device)

        if not reverse:
            x = scores
            for layer in self.layers:
                if isinstance(layer, ActNorm):
                    x, ld = layer(x, reverse=False)
                else:
                    x, ld = layer(x, enc, reverse=False)
                log_det_sum = log_det_sum + ld
            return x, log_det_sum

        else:
            z = scores
            for layer in reversed(self.layers):
                if isinstance(layer, ActNorm):
                    z, ld = layer(z, reverse=True)
                else:
                    z, ld = layer(z, enc, reverse=True)
                log_det_sum = log_det_sum + ld
            return z, log_det_sum

    def log_prob(self, scores, features):
        """log P(scores | features) under the learned null distribution."""
        z, log_det = self.forward(scores, features, reverse=False)
        log_pz     = -0.5 * (z ** 2).sum(dim=1) \
                     - 0.5 * self.score_dim * self._log_2pi
        return log_pz + log_det

    def sample(self, features):
        """Sample one null score vector per row of features."""
        z = torch.randn(features.size(0), self.score_dim, device=features.device)
        scores, _ = self.forward(z, features, reverse=True)
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ScoreShiftFlowWrapper(nn.Module):
    """
    Wraps ScoreShiftFlow with train / generate_decoys helpers.
    """
    def __init__(self,
                 num_classes = 10,
                 n_flows     = 12,
                 feature_dim = 640,
                 hidden_dim  = 256,
                 encoder_dim = 128,
                 clip_val    = 5.0):
        super().__init__()
        self.num_classes = num_classes
        self.flow = ScoreShiftFlow(
            score_dim   = num_classes,
            feature_dim = feature_dim,
            n_flows     = n_flows,
            hidden_dim  = hidden_dim,
            encoder_dim = encoder_dim,
            clip_val    = clip_val,
        )
        print("✓ ScoreShiftFlow defined — single flow over full score vector")
        print(f"  clip_val={clip_val}  (OOD robust feature normalization)")

    # ── Training ──────────────────────────────────────────────────────────────

    def train_flow(self,
                   score_dataset,
                   epochs     = 30,
                   lr         = 3e-4,
                   batch_size = 256,
                   device     = 'cuda',
                   patience   = 5,
                   grad_clip  = 1.0):
        self.flow.to(device)
        self.flow.train()   # RobustFeatureNormalizer накапливает статистику

        loader    = DataLoader(score_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
        optimizer = torch.optim.AdamW(self.flow.parameters(), lr=lr,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

        best_loss  = float('inf')
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches  = 0

            for cnn_scores, features, target_decoy, labels in loader:
                features     = features.to(device)
                target_decoy = target_decoy.to(device)

                log_prob = self.flow.log_prob(target_decoy, features)
                loss     = -log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.flow.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  "
                      f"loss={avg_loss:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

            if avg_loss < best_loss - 1e-4:
                best_loss  = avg_loss
                no_improve = 0
                best_state = {k: v.clone()
                              for k, v in self.flow.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    self.flow.load_state_dict(best_state)
                    break

        if best_state is not None:
            self.flow.load_state_dict(best_state)
        print(f"  ✓ Training complete. Best loss: {best_loss:.4f}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate_decoys(self, score_dataset, device='cuda', n_samples=1):
        """
        Returns:
          model_scores  : (N, C)  original CNN scores
          decoy_scores  : (N, C)  sampled null score vectors
          labels        : (N,)    true labels
        """
        self.flow.to(device)
        self.flow.eval()   # RobustFeatureNormalizer использует frozen статистику

        cnn_list   = []
        decoy_list = []
        lbl_list   = []

        loader = DataLoader(score_dataset, batch_size=256,
                            shuffle=False, num_workers=0)

        with torch.no_grad():
            for cnn_scores, features, target_decoy, labels in loader:
                features = features.to(device)

                if n_samples == 1:
                    decoy = self.flow.sample(features)
                else:
                    samples = torch.stack(
                        [self.flow.sample(features) for _ in range(n_samples)],
                        dim=0)
                    decoy = samples.mean(dim=0)

                cnn_list.append(cnn_scores.cpu().numpy())
                decoy_list.append(decoy.cpu().numpy())
                lbl_list.append(labels.cpu().numpy())

        return (np.concatenate(cnn_list,   axis=0),
                np.concatenate(decoy_list, axis=0),
                np.concatenate(lbl_list,   axis=0))

    # ── OOD диагностика ───────────────────────────────────────────────────────

    def diagnose_feature_ood(self, score_dataset, device='cuda', tile_name=''):
        """
        Показывает насколько фичи тайла OOD относительно train статистики.
        Вызывать после train_flow (когда статистика уже собрана).
        """
        self.flow.to(device)
        self.flow.eval()

        norm = self.flow.feature_norm
        if not norm.initialized:
            print("  ⚠ Normalizer not initialized — train first")
            return

        loader = DataLoader(score_dataset, batch_size=256, shuffle=False)
        all_features = []
        with torch.no_grad():
            for _, features, _, _ in loader:
                all_features.append(features.to(device))
        features = torch.cat(all_features, dim=0)

        # Отклонение от train median в единицах IQR
        deviation = ((features - norm.running_median) /
                     (norm.running_iqr + 1e-6)).abs()

        print(f"  OOD диагностика [{tile_name}]:")
        print(f"    median deviation (IQR units) : {deviation.median():.2f}")
        print(f"    95th pct deviation            : {deviation.quantile(0.95):.2f}")
        print(f"    max deviation                 : {deviation.max():.2f}")
        print(f"    % features с deviation >  5  : "
              f"{(deviation > 5).float().mean() * 100:.1f}%")
        print(f"    % features с deviation > 10  : "
              f"{(deviation > 10).float().mean() * 100:.1f}%")
        print(f"    % features с deviation > 100 : "
              f"{(deviation > 100).float().mean() * 100:.1f}%")


def get_scores_from_ds(dataset):
    """Вытащить scores и labels из ScoreFeatureDataset."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    cnn_list, lbl_list = [], []
    with torch.no_grad():
        for cnn_scores, features, target_decoy, labels in loader:
            cnn_list.append(cnn_scores.cpu().numpy())
            lbl_list.append(labels.cpu().numpy())
    return np.concatenate(cnn_list), np.concatenate(lbl_list)


def load_bcss_test_file(fpath):
    """Загрузить один .tensor тайл → ScoreFeatureDataset."""
    data        = torch.load(fpath, weights_only=False)
    predictions = torch.flatten(data['predictions'], start_dim=2).squeeze(0).T
    features    = torch.flatten(data['features'],    start_dim=2).squeeze(0).T
    labels      = torch.flatten(torch.tensor(data['mask']), start_dim=0)
    labels      = torch.where(labels <= 3, labels, torch.tensor(4))
    return ScoreFeatureDataset(predictions, features, predictions, labels)


# ============================================================================
# ДАННЫЕ И FLOW
# ============================================================================

print("\n" + "="*70)
print("LOADING TRAINING DATA")
print("="*70)
data = torch.load(PATH_DATA)

# ── Создаём train dataset с decoys ───────────────────────────────────────
train_ds = create_score_feature_dataset_bcss(
    data,
    bool_multiclass=True,
    device='cpu'
)

train_scores, train_labels = get_scores_from_ds(train_ds)

print(f"Train scores shape: {train_scores.shape}")
for c in range(NUM_CLASSES):
    print(f"  Class {c}: {(train_labels == c).sum()} samples")

print("\n" + "="*70)
print("FLOW MODEL")
print("="*70)

flow = ScoreShiftFlowWrapper(
    num_classes = NUM_CLASSES,
    n_flows     = 12,
    feature_dim = FEATURE_DIM,
    hidden_dim  = 256,
    encoder_dim = 128,
    clip_val    = 5.0,      # ← OOD защита
).to(DEVICE)

if os.path.exists(FLOWS_PATH):
    flow.load_state_dict(torch.load(FLOWS_PATH, map_location=DEVICE))
    print(f"✓ Flow loaded from {FLOWS_PATH}")
else:
    print("Training ScoreShiftFlow...")
    flow.train_flow(
        train_ds, epochs=40, lr=3e-4, batch_size=256,
        device=DEVICE, patience=5, grad_clip=1.0)
    torch.save(flow.state_dict(), FLOWS_PATH)
    print(f"✓ Flow saved to {FLOWS_PATH}")

# Source данные для baseline методов (40% subsample от train)
print("\nПодготовка source данных для baselines...")
np.random.seed(42)
subset_idx = np.random.choice(len(train_ds), int(0.4 * len(train_ds)), replace=False)
subset_ds  = Subset(train_ds, subset_idx.tolist())

source_logits, _, source_labels = flow.generate_decoys(subset_ds, device=DEVICE)
source_logits_t = torch.tensor(source_logits).to(DEVICE)
source_labels_t = torch.tensor(source_labels).to(DEVICE)
temp            = calibration_temp(source_logits_t, source_labels_t)
scaled_source   = source_logits_t / temp
print(f"✓ Source: {len(source_labels)} samples, temp={temp:.4f}")


# ============================================================================
# ОБРАБОТКА ТЕСТОВЫХ ФАЙЛОВ
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import os

# ── Global settings ──────────────────────────────────────────────────────────
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

for d in ['BCSS/figures/distributions', 'BCSS/figures/diagnostics',
          'BCSS/figures/accuracy', 'BCSS/figures/comparison']:
    os.makedirs(d, exist_ok=True)

# ── Единые цвета для всех графиков ───────────────────────────────────────────
COLOR_ENGPE    = '#FF9800'   # синий      — ENGPE
COLOR_ENGPE_TA = '#FF6D00'   # оранжевый  — ENGPE-TA

print("\n" + "="*70)
print("PROCESSING TEST FILES")
print("="*70)

test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, '*.tensor')))
print(f"Найдено {len(test_files)} тестовых файлов")

all_results = []

for fpath in tqdm(test_files, desc="Test files"):
    tile_name = os.path.basename(fpath)
    print(f"\n{'─'*60}")
    print(f"  TILE: {tile_name}")
    print(f"{'─'*60}")

    # ── Загрузка тайла ────────────────────────────────────────────────────
    try:
        test_ds = load_bcss_test_file(fpath)
    except Exception as e:
        print(f"  ✗ {tile_name}: {e}")
        continue

    # ── Генерация decoys ──────────────────────────────────────────────────
    ms, ds_flow, ls = flow.generate_decoys(test_ds, device=DEVICE)

    ms      = ms[::SUBSAMPLE_STEP]
    ds_flow = ds_flow[::SUBSAMPLE_STEP]
    ls      = ls[::SUBSAMPLE_STEP]
    n       = len(ls)

    if n < MIN_PIXELS:
        print(f"  ✗ Too few pixels: {n}")
        continue

    target_logits = torch.tensor(ms).to(DEVICE)
    target_labels = torch.tensor(ls).to(DEVICE)

    true_acc_full = float((ms.argmax(axis=1) == ls).mean())

    scaled_target = target_logits / temp

    # ── Baseline methods ──────────────────────────────────────────────────
    baseline_estimates = {}
    baseline_errors    = {}

    for method_name, method_func in BASELINE_METHODS.items():
        try:
            estimate = float(method_func(scaled_source, source_labels_t, scaled_target))
        except Exception as e:
            print(f"  {method_name}: FAILED ({e})")
            estimate = np.nan
        estimate = np.clip(estimate, 0.0, 1.0)
        error    = abs(estimate - true_acc_full)
        baseline_estimates[method_name] = estimate
        baseline_errors[method_name]    = error
        print(f"  {method_name}: {estimate:.3f} (error: {error:.3f})")

    print(f"{'─'*60}")

    # ── Подготовка данных ─────────────────────────────────────────────────
    true_labels  = ls
    n_samples    = len(true_labels)

    pred_scores  = np.max(ms, axis=1)
    pred_label   = np.argmax(ms, axis=1)
    decoy_scores = np.max(ds_flow, axis=1)

    correct_pred = (true_labels == pred_label).astype(int)

    print(f"  accuracy : {correct_pred.sum() / n_samples:.4f}")
    print(f"  error/pi0: {1 - correct_pred.sum() / n_samples:.4f}")

    safe_name = 'norm_flow_' + tile_name.replace('.tensor', '')

    # ── Figure 1: Score distributions ─────────────────────────────────────
    bin_width  = 0.05
    bins       = np.arange(pred_scores.min() - 0.5,
                           pred_scores.max() + 0.5, bin_width)
    stat       = 'density'
    plot_alpha = 0.3

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(pred_scores,
                 bins=bins, stat=stat, color='blue', kde=True,
                 fill=True, alpha=plot_alpha, label='model_mixture', ax=ax)
    sns.histplot(decoy_scores,
                 bins=bins, stat=stat, color='orange', kde=True,
                 fill=True, alpha=plot_alpha, label='null', ax=ax)
    sns.histplot(pred_scores[true_labels != pred_label],
                 bins=bins, stat=stat, color='red', kde=True,
                 fill=True, alpha=plot_alpha, label='incorrect', ax=ax)
    ax.set_title(f'{tile_name[:30]} — Score Distributions')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_scores.png', dpi=300)
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_scores.pdf')
    plt.show()
    plt.close()

    # ── True FDR ──────────────────────────────────────────────────────────
    sort_idx           = np.argsort(pred_scores)
    pred_scores_sorted = pred_scores[sort_idx]
    label_sorted       = true_labels[sort_idx]
    pred_label_sorted  = pred_label[sort_idx]
    correct_pred_s     = (label_sorted == pred_label_sorted).astype(int)

    FD        = 1 - correct_pred_s
    FD_CF     = np.cumsum(FD[::-1])[::-1]
    D_CF      = np.arange(0, n_samples)[::-1] + 1
    FDR_true  = np.clip(FD_CF / D_CF, 0, 1)
    QVAL_true = np.clip(np.minimum.accumulate(FDR_true), 0, 1)

    # ── Mix-Max FDR ───────────────────────────────────────────────────────
    pi0 = 0.0

    sorted_decoys           = np.sort(decoy_scores)
    unique_z_vals, counts_z = np.unique(decoy_scores, return_counts=True)
    n_unique_z              = len(unique_z_vals)

    counts_w_leq_z = np.searchsorted(pred_scores_sorted, unique_z_vals, side='left')
    counts_z_leq_z = np.searchsorted(sorted_decoys,      unique_z_vals, side='left')

    P_W_leq_z = np.clip(
        (counts_w_leq_z - pi0 * counts_z_leq_z) / ((1 - pi0) * n_samples), 0, 1)
    P_Y_leq_z = np.clip(counts_z_leq_z / n_samples, 0, 1)

    R_j = np.divide(P_W_leq_z, P_Y_leq_z,
                    out=np.zeros_like(P_W_leq_z),
                    where=P_Y_leq_z > 0)
    R_j = np.clip(R_j, 0, 1)

    all_thresholds = pred_scores_sorted[::-1]
    fdr_values     = np.zeros(n_samples)

    for i, T in enumerate(all_thresholds):
        D     = i + 1
        F_0   = pi0 * np.sum(decoy_scores > T)
        z_idx = np.searchsorted(unique_z_vals, T, side='left')
        F_1   = (0.0 if z_idx >= n_unique_z
                 else (1 - pi0) * np.sum(R_j[z_idx:] * counts_z[z_idx:]))
        fdr_values[i] = (F_0 + F_1) / D if D > 0 else 0

    QVAL_mixmax = np.clip(np.minimum.accumulate(fdr_values[::-1]), 0, 1)

    # ── Figure 2: FDR vs score threshold ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pred_scores_sorted, QVAL_mixmax, label='Mix-Max FDR')
    ax.plot(pred_scores_sorted, QVAL_true,   label='True FDR')
    ax.set_xlabel('Score threshold')
    ax.set_ylabel('q-value (FDR)')
    ax.set_title(f'{tile_name[:30]} — FDR vs Score Threshold')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_fdr_vs_score.png', dpi=300)
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_fdr_vs_score.pdf')
    plt.show()
    plt.close()

    # ── TDC FDR ───────────────────────────────────────────────────────────
    TDC       = (pred_scores > decoy_scores).astype(int)
    TDC_score = np.maximum(pred_scores, decoy_scores)

    tdc_sort_idx = np.argsort(TDC_score)
    TDC_score_s  = TDC_score[tdc_sort_idx]
    TDC_label_s  = TDC[tdc_sort_idx]

    FD_CF_tdc = np.cumsum((1 - TDC_label_s)[::-1])[::-1]
    D_CF_tdc  = np.maximum(np.arange(0, n_samples)[::-1] + 1 - FD_CF_tdc, 1)
    TDC_FDR   = np.clip(FD_CF_tdc / D_CF_tdc, 0, 1)
    QVAL_TDC  = np.clip(np.minimum.accumulate(TDC_FDR), 0, 1)

    # ── Acc estimation ─────────────────────────────────────────────────────
    pi0_tdc = np.clip(float(QVAL_TDC[0]),    0.0, 1.0)
    pi0_mm  = np.clip(float(QVAL_mixmax[0]), 0.0, 1.0)

    Acc_est    = np.zeros(n_samples)
    Acc_est_MM = np.zeros(n_samples)
    Acc_true   = np.zeros(n_samples)

    for i in range(n_samples):
        TP_true     = correct_pred_s[i:].sum()
        TN_true     = (1 - correct_pred_s[:i]).sum()
        Acc_true[i] = (TP_true + TN_true) / n_samples

        accepted       = n_samples - i
        FP_tdc         = accepted * QVAL_TDC[i]
        TP_tdc         = accepted * (1 - QVAL_TDC[i])
        TN_tdc         = n_samples * pi0_tdc - FP_tdc
        Acc_est[i]     = np.clip((TP_tdc + TN_tdc) / n_samples, 0.0, 1.0)

        FP_mm          = accepted * QVAL_mixmax[i]
        TP_mm          = accepted * (1 - QVAL_mixmax[i])
        TN_mm          = n_samples * pi0_mm - FP_mm
        Acc_est_MM[i]  = np.clip((TP_mm + TN_mm) / n_samples, 0.0, 1.0)

    Acc_true = np.clip(Acc_true, 0.0, 1.0)

    # ── Figure 3: Acc & FDR vs normalised rank ────────────────────────────
    normalized_rank = np.arange(n_samples) / n_samples

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(normalized_rank, QVAL_TDC,    label='TDC FDR')
    ax.plot(normalized_rank, QVAL_mixmax, label='Mix-Max FDR')
    ax.plot(normalized_rank, QVAL_true,   label='True FDR')
    ax.plot(normalized_rank, Acc_true,    label='True Acc')
    ax.plot(normalized_rank, Acc_est,     label='Est Acc with TDC')
    ax.plot(normalized_rank, Acc_est_MM,  label='Est Acc with Mix-Max')
    ax.set_xlabel('Normalized rank (fraction accepted)')
    ax.set_ylabel('Value')
    ax.set_title(f'{tile_name[:30]}')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(f'BCSS/figures/accuracy/{safe_name}_acc_fdr.png', dpi=300)
    plt.savefig(f'BCSS/figures/accuracy/{safe_name}_acc_fdr.pdf')
    plt.show()
    plt.close()

    # ── ACC_ST / ACC_TA metrics ────────────────────────────────────────────
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

    # ── Collect results ────────────────────────────────────────────────────
    row = dict(
        tile           = tile_name,
        n_pixels       = n_samples,
        true_acc       = true_acc_full,
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
    df.to_csv('BCSS/figures/comparison/norm_flow_results.csv',
              index=False, float_format='%.6f')
    print(f"\n✓ results.csv сохранён  ({len(df)} тайлов)")

    # ── Build summary table (mean ± std) ──────────────────────────────────
    summary_rows = []

    acc_report = [
        ('acc_st_true', 'True ACC_ST',        'True',     'ST'),
        ('acc_ta_true', 'True ACC_TA',         'True',     'TA'),
        ('acc_st_tdc',  'ENGPE est ACC_ST',    'ENGPE',    'ST'),
        ('acc_ta_tdc',  'ENGPE est ACC_TA',    'ENGPE',    'TA'),
        ('acc_st_mm',   'ENGPE-TA est ACC_ST', 'ENGPE-TA', 'ST'),
        ('acc_ta_mm',   'ENGPE-TA est ACC_TA', 'ENGPE-TA', 'TA'),
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
        for col_key, kind, true_type in [
            (f'err_{method_name}',    'ST', 'ACC_ST'),
            (f'err_ta_{method_name}', 'TA', 'ACC_TA'),
        ]:
            v = df[col_key].values
            summary_rows.append(dict(metric=f'MAE {method_name} vs {true_type}',
                                     method=method_name, type=kind,
                                     mean=v.mean(), std=v.std()))

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('BCSS/figures/comparison/norm_flow_results_summary.csv',
                      index=False, float_format='%.6f')
    print("✓ summary CSV сохранён")

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("AGGREGATE SUMMARY  (mean ± std across tiles)")
    print("="*70)
    print(f"\n  {'Metric':<38} {'Mean':>8}   {'Std':>8}   {'Type':>6}")
    print(f"  {'─'*65}")
    for r in summary_rows:
        print(f"  {r['metric']:<38} {r['mean']:>8.4f}   {r['std']:>8.4f}   {r['type']:>6}")

    print(f"\n  Full results table:")
    print(df.to_string(index=False, float_format='{:.4f}'.format))

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
    plt.savefig('BCSS/figures/comparison/norm_scatter_true_accst_vs_accta.png',
                dpi=300)
    plt.savefig('BCSS/figures/comparison/norm_scatter_true_accst_vs_accta.pdf')
    plt.show()
    plt.close()
    print("✓ scatter_true_accst_vs_accta")

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 2 (summary): Estimated Accuracy vs True Accuracy
    # All competitors + ENGPE (true=ACC_ST) + ENGPE-TA (true=ACC_TA)
    # ─────────────────────────────────────────────────────────────────────
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

    # ── Baselines ──────────────────────────────────────────────────────────
    for idx, method_name in enumerate(baseline_list):
        ax.scatter(
            df['acc_st_true'].values,
            df[f'est_{method_name}'].values,
            label  = method_name,
            marker = baseline_markers[idx % len(baseline_markers)],
            color  = baseline_palette[idx],
            s      = 55,
            alpha  = 0.70,
            zorder = 3
        )

    # ── ENGPE: true accuracy = ACC_ST ─────────────────────────────────────
    ax.scatter(
        df['acc_st_true'].values,
        df['acc_st_tdc'].values,
        label  = 'ENGPE',
        marker = 'o',
        color  = COLOR_ENGPE,       # '#2196F3'
        s      = 70,
        alpha  = 0.80,
        zorder = 4
    )

    # ── ENGPE-TA: true accuracy = ACC_TA (brighter / larger) ──────────────
    ax.scatter(
        df['acc_ta_true'].values,
        df['acc_ta_mm'].values,
        label  = 'ENGPE-TA',
        marker = 's',
        color  = COLOR_ENGPE_TA,    # '#FF6D00'
        s      = 90,
        alpha  = 0.95,
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
    plt.savefig('BCSS/figures/comparison/norm_scatter_estimated_vs_true_all.png',
                dpi=300)
    plt.savefig('BCSS/figures/comparison/norm_scatter_estimated_vs_true_all.pdf')
    plt.show()
    plt.close()
    print("✓ scatter_estimated_vs_true_all")

    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 3 (summary): Bar chart — MAE per method (mean ± std)
    # ─────────────────────────────────────────────────────────────────────
    bar_entries = []

    # Our methods — используем те же COLOR_ENGPE / COLOR_ENGPE_TA
    bar_entries.append(('ENGPE\n(ST)',     'err_st_tdc', COLOR_ENGPE))
    bar_entries.append(('ENGPE\n(TA)',     'err_ta_tdc', COLOR_ENGPE))
    bar_entries.append(('ENGPE-TA\n(ST)', 'err_st_mm',  COLOR_ENGPE_TA))
    bar_entries.append(('ENGPE-TA\n(TA)', 'err_ta_mm',  COLOR_ENGPE_TA))

    # Baselines (vs ACC_ST)
    for idx, method_name in enumerate(baseline_list):
        bar_entries.append((
            f'{method_name}\n(ST)',
            f'err_{method_name}',
            baseline_palette[idx]
        ))

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
        mpatches.Patch(facecolor=COLOR_ENGPE,    alpha=0.85, label='ENGPE'),
        mpatches.Patch(facecolor=COLOR_ENGPE_TA, alpha=0.85, label='ENGPE-TA'),
        mpatches.Patch(facecolor='grey',         alpha=0.85, label='Baseline'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    plt.savefig('BCSS/figures/comparison/norm_summary_bar_mean_error.png', dpi=300)
    plt.savefig('BCSS/figures/comparison/norm_summary_bar_mean_error.pdf')
    plt.show()
    plt.close()
    print("✓ bar_mae_per_method")

    print("\nAll figures saved.")


# ============================================================================
# PLOT: Score distributions для первого тайла
# ============================================================================
print("\n" + "="*70)
print("SCORE DISTRIBUTIONS (первый тайл)")
print("="*70)

first_tile = test_files[0]
tile_name  = os.path.basename(first_tile)
print(f"Тайл: {tile_name}")

try:
    test_ds_dist = load_bcss_test_file(first_tile)

    ms, ds_decoy, ls = flow.generate_decoys(test_ds_dist, device=DEVICE)
    ms       = ms[::SUBSAMPLE_STEP]
    ds_decoy = ds_decoy[::SUBSAMPLE_STEP]
    ls       = ls[::SUBSAMPLE_STEP]

    for i in range(NUM_CLASSES):
        plot_score_distribution_with_decoys(
            train_scores[:, i],
            train_labels,
            ms[:, i],
            ls,
            ds_decoy[:, i],
            filename = f"BCSS/figures/diagnostics/norm_scores_dist_class{i}",
            title    = f"Score Distribution with Decoys — Class {i}",
            xlim     = (-10, 10),
            show_kde = True,
            class_id = i,
        )
    print(f"✓ Score distributions сохранены для {NUM_CLASSES} классов")

except Exception as e:
    print(f"✗ Ошибка при построении distributions: {e}")