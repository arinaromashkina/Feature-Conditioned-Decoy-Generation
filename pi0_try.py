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
from scipy.stats import spearmanr
import os
import tarfile
import io
from bisect import bisect
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

# Print versions
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Set numpy print options
np.set_printoptions(threshold=np.inf)

# Publication-quality matplotlib settings
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

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10

print(f"Device: {DEVICE}")

# Create output directories
os.makedirs('figures/accuracy', exist_ok=True)
os.makedirs('figures/fdr_curves', exist_ok=True)
os.makedirs('figures/mano', exist_ok=True)
os.makedirs('figures/comparison', exist_ok=True)

print("✓ Setup complete")


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
from scipy.stats import spearmanr
import os
from bisect import bisect
from collections import defaultdict


def negentropy(logits):
    """Compute negentropy (negative entropy)."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    max_entropy = np.log(logits.shape[1])
    return max_entropy - entropy


def calibration_temp(logits, labels, num_bins=15):
    """Find optimal temperature for calibration."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()
    
    temps = torch.linspace(0.1, 5.0, 50)
    best_temp = 1.0
    best_ece = float('inf')
    
    for temp in temps:
        scaled_probs = torch.softmax(logits / temp, dim=1)
        confidences, predictions = scaled_probs.max(dim=1)
        accuracies = (predictions == labels).float()
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = accuracies[mask].mean()
                ece += mask.float().mean() * torch.abs(bin_conf - bin_acc)
        
        if ece < best_ece:
            best_ece = ece
            best_temp = temp.item()
    
    return best_temp


def _to_tensor(x):
    """Convert numpy array or tensor to torch tensor."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x


def predict_ATC_maxconf(source_logits, source_labels, target_logits):
    """Average Threshold Confidence with max confidence."""
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    
    source_scores = torch.softmax(source_logits, dim=1).amax(1)
    target_scores = torch.softmax(target_logits, dim=1).amax(1)
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_logits.argmax(1) == source_labels).sum()]
    estimate = (target_scores > threshold).float().mean().item()
    return estimate


def predict_ATC_negent(source_logits, source_labels, target_logits):
    """Average Threshold Confidence with negentropy."""
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    
    source_scores = negentropy(source_logits)
    target_scores = negentropy(target_logits)
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_logits.argmax(1) == source_labels).sum()]
    estimate = (target_scores > threshold).float().mean().item()
    return estimate


def predict_AC(source_logits, source_labels, target_logits):
    """Average Confidence."""
    target_logits = _to_tensor(target_logits)
    return torch.softmax(target_logits, dim=1).amax(1).mean().item()


def predict_DOC(source_logits, source_labels, target_logits):
    """Difference of Confidences."""
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    
    avg_source_conf = torch.softmax(source_logits, dim=1).amax(1).mean().item()
    avg_target_conf = torch.softmax(target_logits, dim=1).amax(1).mean().item()
    source_acc = (source_logits.argmax(1) == source_labels).float().mean().item()
    return source_acc + (avg_target_conf - avg_source_conf)


try:
    import ot
    
    def predict_COT(source_logits, source_labels, target_logits):
        """Confidence Optimal Transport."""
        source_logits = _to_tensor(source_logits)
        source_labels = _to_tensor(source_labels).long()
        target_logits = _to_tensor(target_logits)
        
        num_classes = source_logits.shape[1]
        source_label_dist = torch.nn.functional.one_hot(source_labels, num_classes).float().mean(0)
        target_probs = torch.softmax(target_logits, dim=1)
        
        cost_matrix = torch.stack([
            (target_probs - onehot).abs().sum(1)
            for onehot in torch.eye(num_classes, device=target_logits.device)
        ], dim=1) / 2
        
        # IMPORTANT: ot.emd() requires all arrays to be numpy
        uniform_dist = np.ones(len(target_probs)) / len(target_probs)
        source_dist = source_label_dist.cpu().numpy()
        cost_matrix_np = cost_matrix.cpu().numpy()
        
        ot_plan = ot.emd(uniform_dist, source_dist, cost_matrix_np)
        ot_cost = np.sum(ot_plan * cost_matrix_np)
        
        s_conf = torch.softmax(source_logits, dim=1).amax(1).mean().item()
        s_acc = (source_logits.argmax(1) == source_labels).float().mean().item()
        conf_gap = s_conf - s_acc
        err_est = ot_cost + conf_gap
        return 1. - err_est
    
    COT_AVAILABLE = True
    print("✓ COT available")
except ImportError:
    COT_AVAILABLE = False
    print("⚠ COT unavailable (install: pip install POT)")


BASELINE_METHODS = {
    'ATC': predict_ATC_maxconf,
    'ATC-NE': predict_ATC_negent,
    'AC': predict_AC,
    'DOC': predict_DOC,
}

if COT_AVAILABLE:
    BASELINE_METHODS['COT'] = predict_COT

print(f"✓ Baseline methods: {list(BASELINE_METHODS.keys())}")


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


MIN_POOL = 30


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
                                   verbose:      bool = True,
                                   build_vectors: bool = False,
                                   max_pool_size: int  = 10_000) -> dict:
    """
    Строит пулы для decoy-стратегий.

    pool_score[c]   — score_c на примерах где argmax=c И label≠c
                      (или fallback label≠c). Ограничен max_pool_size.
    pool_vectors[c] — полные score-векторы label≠c (только если build_vectors=True).

    Логика Mix-Max (π_mm = 0):
      f(t) = model_scores[i, ĉ_i]   — target score
      Z(t) ~ pool_score[ĉ_i]        — decoy score
    """
    rng          = np.random.default_rng(0)
    pred_classes = train_scores.argmax(axis=1)
    pool_score   = {}
    pool_vectors = {} if build_vectors else None

    if verbose:
        acc = (pred_classes == train_labels).mean()
        print(f"\nBuilding error-conditioned decoy pools  "
              f"(train acc={acc:.4f}, max_pool_size={max_pool_size})")

    for c in range(num_classes):
        # ── pool_score: coord c при ошибке ───────────────────────────────
        error_mask = (pred_classes == c) & (train_labels != c)
        n_err      = error_mask.sum()

        if n_err >= MIN_POOL:
            candidates = train_scores[error_mask, c]
            src = "per-class errors (argmax=c & label≠c)"
        else:
            neg_mask   = train_labels != c
            candidates = train_scores[neg_mask, c]
            src = "fallback (label≠c)"

        # Субсэмплинг — достаточно max_pool_size для rng.choice
        if len(candidates) > max_pool_size:
            idx = rng.choice(len(candidates), size=max_pool_size, replace=False)
            pool_score[c] = candidates[idx]
        else:
            pool_score[c] = candidates

        # ── pool_vectors: только если явно запрошено ──────────────────────
        if build_vectors:
            neg_mask = train_labels != c
            vecs     = train_scores[neg_mask]
            if len(vecs) > max_pool_size:
                idx = rng.choice(len(vecs), size=max_pool_size, replace=False)
                vecs = vecs[idx]
            pool_vectors[c] = vecs

        if verbose:
            p = pool_score[c]
            n_vec = pool_vectors[c].shape[0] if build_vectors else 0
            print(f"  class {c}: n_score={len(p):6d}  "
                  f"n_vec={n_vec:6d}  "
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


# ============================================================
# LOAD PRECOMPUTED DATA (plant-daniella)
# ============================================================
TRAIN_DATA_PATH = '/kaggle/input/datasets/arinaromashkina/plantlet/train_data2 (1).tar.xz'
TEST_DATA_PATH  = '/kaggle/input/datasets/arinaromashkina/plantlet/test_data2.tar.xz'
FLOWS_PATH      = 'score_shift_flow_plant.pt'
DECOY_STRATEGY  = 'score_coord'
TEST_DECOYS_SAVE_PATH = 'test_decoys_plant.pt'


def load_precomputed_data(path):
    if path.endswith(('.tar.xz', '.tar.gz', '.tar.bz2', '.tar')):
        with tarfile.open(path, 'r:*') as tar:
            pt_members = [m for m in tar.getmembers() if m.name.endswith('.pt')]
            if not pt_members:
                raise FileNotFoundError(f"No .pt file found inside {path}")
            f = tar.extractfile(pt_members[0])
            data = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)
    else:
        data = torch.load(path, map_location='cpu', weights_only=False)
    print(data.keys())
    feat_key  = 'hidden_features' if 'hidden_features' in data else 'test_hidden_features'
    logit_key = 'logits'          if 'logits'          in data else 'test_logits'
    label_key = 'labels'          if 'labels'          in data else 'test_labels'
    return (
        data[feat_key].float(),
        data[logit_key].float(),
        data[label_key].long(),
    )


def create_train_score_dataset_from_tensors(logits, features, labels,
                                             pool_score, strategy='score_coord'):
    """Build ScoreFeatureDataset from pre-computed tensors (no CNN needed)."""
    sc_np = logits.numpy()
    rng   = np.random.default_rng(0)
    if strategy == 'score_coord':
        dc_np = _build_decoy_score_coord(sc_np, pool_score, rng)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return ScoreFeatureDataset(
        logits, features, torch.from_numpy(dc_np).float(), labels)


def create_test_score_dataset_from_tensors(logits, features, labels):
    """Build ScoreFeatureDataset for test inference (placeholder decoys)."""
    return ScoreFeatureDataset(logits, features, logits.clone(), labels)


print("\n" + "="*70)
print("LOADING PRECOMPUTED DATA")
print("="*70)
train_features, train_logits, train_labels_t = load_precomputed_data(TRAIN_DATA_PATH)
test_features,  test_logits,  test_labels_t  = load_precomputed_data(TEST_DATA_PATH)

train_scores_raw = train_logits.numpy()
train_labels_raw = train_labels_t.numpy()

NUM_CLASSES = train_logits.shape[1]
FEATURE_DIM = train_features.shape[1]
print(f"Train: logits={train_logits.shape}, features={train_features.shape}, labels={train_labels_t.shape}")
print(f"Test:  logits={test_logits.shape},  features={test_features.shape},  labels={test_labels_t.shape}")
print(f"NUM_CLASSES={NUM_CLASSES}, FEATURE_DIM={FEATURE_DIM}")
print(f"Train acc={(train_scores_raw.argmax(1) == train_labels_raw).mean():.4f}")

print("\n" + "="*70)
print("BUILDING ERROR-CONDITIONED DECOY POOLS")
print("="*70)
pool_score, pool_vectors = build_error_conditioned_pools(
    train_scores_raw, train_labels_raw, NUM_CLASSES,
    verbose=True, build_vectors=False, max_pool_size=10_000)

# ── Логиты train для baselines ────────────────────────────────────────────────
source_logits_np = train_scores_raw
source_labels_np = train_labels_raw

# ============================================================
# STRATEGY COMPARISON SETUP
# ============================================================

NOISE_STD = 0.5

STRATEGIES = [
    ('score_coord',            0.0),
    ('score_coord_noise',      NOISE_STD),
    ('nearest_neighbor',       0.0),
    ('nearest_neighbor_noise', NOISE_STD),
]
STRATEGY_LABELS = {
    'score_coord':            'Random (SC)',
    'score_coord_noise':      'Random + noise',
    'nearest_neighbor':       'Nearest-neighbor',
    'nearest_neighbor_noise': 'NN + noise',
}
STRATEGY_COLORS = {
    'score_coord':            '#1976D2',
    'score_coord_noise':      '#42A5F5',
    'nearest_neighbor':       '#E65100',
    'nearest_neighbor_noise': '#FF8A65',
}


def build_error_vector_pool(train_scores, train_labels, num_classes, min_pool=30):
    """Pool of full logit vectors where argmax==k AND label!=k (for NN strategy)."""
    pred_classes = train_scores.argmax(axis=1)
    pool = {}
    for k in range(num_classes):
        mask  = (pred_classes == k) & (train_labels != k)
        pool[k] = train_scores[mask] if mask.sum() >= min_pool \
                  else train_scores[train_labels != k]
    return pool


def _build_decoy_nearest_neighbor(sc_np, pool_error_vectors, rng, noise_std=0.0):
    """NN decoy: find nearest training error vector in subspace excluding coord k."""
    n_samples, n_classes = sc_np.shape
    pred_classes = sc_np.argmax(axis=1)
    dc_np        = sc_np.copy()
    all_coords   = np.arange(n_classes)
    for k in range(n_classes):
        mask = pred_classes == k
        if not mask.any():
            continue
        pool_k = pool_error_vectors.get(k)
        if pool_k is None or len(pool_k) == 0:
            continue
        samples_k      = sc_np[mask]
        compare_coords = all_coords[all_coords != k]
        S_proj = samples_k[:, compare_coords]
        V_proj = pool_k[:,   compare_coords]
        S_sq   = (S_proj ** 2).sum(axis=1, keepdims=True)
        V_sq   = (V_proj ** 2).sum(axis=1, keepdims=True)
        dists  = np.maximum(S_sq + V_sq.T - 2.0 * (S_proj @ V_proj.T), 0.0)
        nn_idx = dists.argmin(axis=1)
        dc_np[mask, k] = pool_k[nn_idx, k]
    if noise_std > 0.0:
        dc_np = dc_np + rng.normal(0.0, noise_std, size=dc_np.shape)
    return dc_np


def apply_strategy(scores_np, pool_score, pool_error_vectors, strategy, noise_std):
    rng    = np.random.default_rng(42)
    _noise = noise_std if strategy in ('score_coord_noise', 'nearest_neighbor_noise') else 0.0
    if strategy in ('nearest_neighbor', 'nearest_neighbor_noise'):
        return _build_decoy_nearest_neighbor(scores_np, pool_error_vectors, rng, _noise)
    else:
        return _build_decoy_score_coord(scores_np, pool_score, rng)


def compute_fdr_acc_curves(scores_np, decoy_np, labels_np, pi0=0.0):
    n            = len(labels_np)
    pred_scores  = scores_np.max(axis=1)
    pred_label   = scores_np.argmax(axis=1)
    decoy_scores = decoy_np.max(axis=1)
    correct      = (pred_label == labels_np).astype(int)

    sort_idx           = np.argsort(pred_scores)
    pred_scores_sorted = pred_scores[sort_idx]
    correct_sorted     = correct[sort_idx]

    # True FDR
    FD        = 1 - correct_sorted
    FD_CF     = np.cumsum(FD[::-1])[::-1]
    D_CF      = np.arange(n, 0, -1)
    FDR_true  = np.clip(FD_CF / D_CF, 0, 1)
    QVAL_true = np.clip(np.minimum.accumulate(FDR_true), 0, 1)

    # TDC
    TDC_score = np.maximum(pred_scores, decoy_scores)
    TDC_win   = (pred_scores > decoy_scores).astype(int)
    tdc_idx   = np.argsort(TDC_score)
    FD_CF_tdc = np.cumsum((1 - TDC_win[tdc_idx])[::-1])[::-1]
    D_CF_tdc  = np.maximum(np.arange(n, 0, -1) - FD_CF_tdc, 1)
    QVAL_TDC  = np.clip(np.minimum.accumulate(np.clip(FD_CF_tdc / D_CF_tdc, 0, 1)), 0, 1)

    # Mix-Max
    sorted_decoys           = np.sort(decoy_scores)
    unique_z_vals, counts_z = np.unique(decoy_scores, return_counts=True)
    n_unique_z              = len(unique_z_vals)
    counts_w_leq_z = np.searchsorted(pred_scores_sorted, unique_z_vals, side='left')
    counts_z_leq_z = np.searchsorted(sorted_decoys,      unique_z_vals, side='left')
    P_W_leq_z = np.clip((counts_w_leq_z - pi0 * counts_z_leq_z) / ((1 - pi0) * n), 0, 1)
    P_Y_leq_z = np.clip(counts_z_leq_z / n, 0, 1)
    R_j       = np.clip(np.divide(P_W_leq_z, P_Y_leq_z,
                                   out=np.zeros_like(P_W_leq_z),
                                   where=P_Y_leq_z > 0), 0, 1)
    fdr_values = np.zeros(n)
    for i, T in enumerate(pred_scores_sorted[::-1]):
        D     = i + 1
        F_0   = pi0 * np.sum(decoy_scores > T)
        z_idx = np.searchsorted(unique_z_vals, T, side='left')
        F_1   = 0.0 if z_idx >= n_unique_z else (
            (1 - pi0) * np.sum(R_j[z_idx:] * counts_z[z_idx:]))
        fdr_values[i] = (F_0 + F_1) / D if D > 0 else 0.0
    QVAL_mixmax = np.clip(np.minimum.accumulate(np.clip(fdr_values, 0, 1)[::-1]), 0, 1)

    # Acc curves
    pi0_tdc = float(np.clip(QVAL_TDC[0],    0, 1))
    pi0_mm  = float(np.clip(QVAL_mixmax[0], 0, 1))
    Acc_true = np.zeros(n); Acc_est = np.zeros(n); Acc_est_MM = np.zeros(n)
    for i in range(n):
        TP_true     = correct_sorted[i:].sum()
        TN_true     = (1 - correct_sorted[:i]).sum()
        Acc_true[i] = (TP_true + TN_true) / n
        accepted      = n - i
        FP_tdc        = accepted * QVAL_TDC[i]
        TN_tdc        = n * pi0_tdc - FP_tdc
        Acc_est[i]    = np.clip((accepted * (1 - QVAL_TDC[i]) + TN_tdc) / n, 0, 1)
        FP_mm         = accepted * QVAL_mixmax[i]
        TN_mm         = n * pi0_mm - FP_mm
        Acc_est_MM[i] = np.clip((accepted * (1 - QVAL_mixmax[i]) + TN_mm) / n, 0, 1)

    Acc_true = np.clip(Acc_true, 0, 1)
    normalized_rank = np.arange(n) / n
    total_TP  = int(correct_sorted.sum())
    TP_from_i = np.cumsum(correct_sorted[::-1])[::-1]
    D_from_i  = np.arange(n, 0, -1)
    acc_st_true = float(Acc_true[0]); acc_ta_true = float(Acc_true.max())
    acc_st_mm   = float(Acc_est_MM[0]); acc_ta_mm = float(Acc_est_MM.max())
    return dict(
        normalized_rank    = normalized_rank,
        pred_scores_sorted = pred_scores_sorted,
        pred_scores        = pred_scores,
        decoy_scores       = decoy_scores,
        QVAL_true          = QVAL_true,
        QVAL_TDC           = QVAL_TDC,
        QVAL_mixmax        = QVAL_mixmax,
        Acc_true           = Acc_true,
        Acc_est            = Acc_est,
        Acc_est_MM         = Acc_est_MM,
        precision_true     = np.where(D_from_i > 0, TP_from_i / D_from_i, 0),
        recall_true        = TP_from_i / max(total_TP, 1),
        precision_est      = np.clip(1 - QVAL_mixmax, 0, 1),
        recall_est         = np.clip((1 - QVAL_mixmax) * D_from_i / max(total_TP, 1), 0, 1),
        correct            = correct,
        pred_label         = pred_label,
        labels             = labels_np,
        true_acc           = float(correct.mean()),
        acc_st_true        = acc_st_true,
        acc_ta_true        = acc_ta_true,
        acc_st_est_mm      = acc_st_mm,
        acc_ta_est_mm      = acc_ta_mm,
        err_st_mm          = abs(acc_st_mm - acc_st_true),
        err_ta_mm          = abs(acc_ta_mm - acc_ta_true),
        n                  = n,
    )


print("\nBuilding NN error vector pool...")
pool_error_vectors = build_error_vector_pool(
    train_scores_raw, train_labels_raw, NUM_CLASSES)

sc_np = test_logits.numpy()
ft_np = test_features.numpy()
lb_np = test_labels_t.numpy()
print(f"Test: n={len(lb_np)}  acc={(sc_np.argmax(1) == lb_np).mean():.4f}")

# ============================================================
# RAW DECOY STRATEGY COMPARISON
# ============================================================
print("\n" + "="*70)
print("RAW DECOY STRATEGY COMPARISON")
print("="*70)

raw_results = {}
for strat_name, noise_std in STRATEGIES:
    decoy_np = apply_strategy(sc_np, pool_score, pool_error_vectors, strat_name, noise_std)
    curves   = compute_fdr_acc_curves(sc_np, decoy_np, lb_np)
    raw_results[strat_name] = curves
    print(f"  {strat_name}: true_acc={curves['true_acc']:.3f}  "
          f"err_st={curves['err_st_mm']:.3f}  err_ta={curves['err_ta_mm']:.3f}")

n_strats = len(STRATEGIES)

# ── Figure 1: Score distributions ────────────────────────────────────────────
fig, axes = plt.subplots(1, n_strats, figsize=(5 * n_strats, 4))
ref_sc = raw_results['score_coord']
bins   = np.linspace(ref_sc['pred_scores'].min() - 0.3,
                     ref_sc['pred_scores'].max() + 0.3, 60)
inc_mask = ref_sc['correct'] == 0

for ax, (strat_name, _) in zip(axes, STRATEGIES):
    c = raw_results[strat_name]
    color = STRATEGY_COLORS[strat_name]
    sns.histplot(c['pred_scores'],  bins=bins, stat='density', color='steelblue',
                 kde=True, fill=True, alpha=0.3, label='model', ax=ax)
    sns.histplot(c['decoy_scores'], bins=bins, stat='density', color=color,
                 kde=True, fill=True, alpha=0.4, label='decoy', ax=ax)
    if inc_mask.any():
        sns.histplot(c['pred_scores'][inc_mask], bins=bins, stat='density',
                     color='crimson', kde=True, fill=True, alpha=0.3, label='incorrect', ax=ax)
    ax.set_title(f'{STRATEGY_LABELS[strat_name]}\ntrue_acc={c["true_acc"]:.3f}')
    ax.set_xlabel('Max logit'); ax.legend(fontsize=8)
    if ax is axes[0]: ax.set_ylabel('Density')

plt.suptitle('Plant — Score Distributions (Raw Decoy)', fontsize=13)
plt.tight_layout(); plt.show()

# ── Figure 2: FDR curves ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, n_strats, figsize=(5 * n_strats, 4))
for ax, (strat_name, _) in zip(axes, STRATEGIES):
    c = raw_results[strat_name]; r = c['normalized_rank']
    ax.plot(r, c['QVAL_true'],   color='gray',                    lw=1.5, ls='--', label='True FDR')
    ax.plot(r, c['QVAL_mixmax'], color=STRATEGY_COLORS[strat_name], lw=1.8,        label='Mix-Max')
    ax.plot(r, c['QVAL_TDC'],   color='navy',                    lw=1.2, ls=':',  label='TDC')
    ax.axhline(0.1, color='black', lw=0.8, ls=':', alpha=0.4)
    ax.set_title(f'{STRATEGY_LABELS[strat_name]}\nMAE_ST={c["err_st_mm"]:.3f}')
    ax.set_xlabel('Fraction accepted'); ax.legend(fontsize=8)
    ax.grid(ls='--', alpha=0.3); ax.set_ylim(0, 1.05)
    if ax is axes[0]: ax.set_ylabel('q-value')

plt.suptitle('Plant — FDR Curves', fontsize=13)
plt.tight_layout(); plt.show()

# ── Figure 3: Accuracy curves overlay ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
r = ref_sc['normalized_rank']

ax = axes[0]
ax.plot(r, ref_sc['Acc_true'], color='black', lw=2, ls='--', label='True Acc')
for strat_name, _ in STRATEGIES:
    c = raw_results[strat_name]
    ax.plot(r, c['Acc_est_MM'], color=STRATEGY_COLORS[strat_name], lw=1.8,
            label=f'{STRATEGY_LABELS[strat_name]} (MAE={c["err_st_mm"]:.3f})')
ax.set_xlabel('Fraction accepted'); ax.set_ylabel('Accuracy')
ax.set_title('Accuracy curves (MixMax)'); ax.legend(fontsize=8); ax.grid(ls='--', alpha=0.4)

ax = axes[1]
ax.plot(r, ref_sc['QVAL_true'], color='black', lw=2, ls='--', label='True FDR')
for strat_name, _ in STRATEGIES:
    c = raw_results[strat_name]
    ax.plot(r, c['QVAL_mixmax'], color=STRATEGY_COLORS[strat_name], lw=1.8,
            label=STRATEGY_LABELS[strat_name])
ax.set_xlabel('Fraction accepted'); ax.set_ylabel('q-value')
ax.set_title('FDR curves comparison'); ax.legend(fontsize=8)
ax.grid(ls='--', alpha=0.4); ax.set_ylim(0, 1.05)

plt.suptitle('Plant — Strategy Comparison', fontsize=13)
plt.tight_layout(); plt.show()

# ── Figure 4: Scatter test (decoy vs model) ──────────────────────────────────
fig, axes = plt.subplots(1, n_strats, figsize=(5 * n_strats, 4))
correct_mask_test = raw_results['score_coord']['correct'].astype(bool)

for ax, (strat_name, _) in zip(axes, STRATEGIES):
    c = raw_results[strat_name]
    ms = c['pred_scores']; ds = c['decoy_scores']
    idx = np.random.default_rng(0).choice(len(ms), min(3000, len(ms)), replace=False)
    ms_s = ms[idx]; ds_s = ds[idx]; cor_s = correct_mask_test[idx]
    ax.scatter(ms_s[cor_s],  ds_s[cor_s],  s=6, alpha=0.3, color='steelblue',
               label='correct', rasterized=True)
    ax.scatter(ms_s[~cor_s], ds_s[~cor_s], s=6, alpha=0.5, color='crimson',
               label='incorrect', rasterized=True)
    lims = [min(ms.min(), ds.min()) - 0.1, max(ms.max(), ds.max()) + 0.1]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(STRATEGY_LABELS[strat_name])
    ax.set_xlabel('Model score (max logit)'); ax.legend(fontsize=8); ax.grid(ls='--', alpha=0.3)
    if ax is axes[0]: ax.set_ylabel('Decoy score (max logit)')

plt.suptitle('Plant — Decoy vs Model Score (TEST)', fontsize=13)
plt.tight_layout(); plt.show()

# ── Figure 5: Scatter TRAINING data ──────────────────────────────────────────
print('\nTraining scatter...')
fig, axes = plt.subplots(1, n_strats, figsize=(5 * n_strats, 4))
tr_sc   = train_scores_raw
tr_lb   = train_labels_raw
corr_tr = (tr_sc.argmax(1) == tr_lb)

for ax, (strat_name, noise_std) in zip(axes, STRATEGIES):
    dc = apply_strategy(tr_sc, pool_score, pool_error_vectors, strat_name, noise_std)
    ms = tr_sc.max(axis=1); ds = dc.max(axis=1)
    idx = np.random.default_rng(0).choice(len(ms), min(3000, len(ms)), replace=False)
    ms_s = ms[idx]; ds_s = ds[idx]; cor_s = corr_tr[idx]
    frac_above = (ds[~corr_tr] > ms[~corr_tr]).mean()
    ax.scatter(ms_s[cor_s],  ds_s[cor_s],  s=6, alpha=0.3, color='steelblue',
               label='correct', rasterized=True)
    ax.scatter(ms_s[~cor_s], ds_s[~cor_s], s=6, alpha=0.6, color='crimson',
               label='incorrect', rasterized=True)
    lims = [min(ms.min(), ds.min()) - 0.1, max(ms.max(), ds.max()) + 0.1]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f'{STRATEGY_LABELS[strat_name]}\n'
                 f'train_acc={corr_tr.mean():.3f}  decoy>model(incorr):{frac_above:.2f}')
    ax.set_xlabel('Model score'); ax.legend(fontsize=8); ax.grid(ls='--', alpha=0.3)
    if ax is axes[0]: ax.set_ylabel('Decoy score')

plt.suptitle('Plant — TRAINING data scatter (Decoy vs Model Score)', fontsize=13)
plt.tight_layout(); plt.show()

# ── Figure 6: Pool debug ──────────────────────────────────────────────────────
print('\nPool debug analysis...')
pred_classes_test = sc_np.argmax(axis=1)
correct_test      = pred_classes_test == lb_np
scores_at_chat    = sc_np[np.arange(len(sc_np)), pred_classes_test]
train_pred        = tr_sc.argmax(axis=1)
train_err         = train_pred != tr_lb
all_pool_vals     = np.concatenate([pool_score[c] for c in range(NUM_CLASSES)])
scores_tr_err     = tr_sc[train_err][np.arange(train_err.sum()), train_pred[train_err]]

frac_pool = np.mean([np.mean(pool_score[c] >= s)
                     for s, c in zip(scores_at_chat[~correct_test],
                                     pred_classes_test[~correct_test])
                     ]) if (~correct_test).sum() > 0 else float('nan')
shift = np.median(all_pool_vals) - np.median(scores_at_chat[~correct_test])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.hist(scores_tr_err, bins=50, density=True, alpha=0.6, color='crimson',
        label=f'train errors at c_hat (n={train_err.sum()})')
ax.hist(all_pool_vals, bins=50, density=True, alpha=0.5, color='orange',
        label=f'pool values (n={len(all_pool_vals)})')
ax.set_title('TRAIN: error scores vs pool')
ax.set_xlabel('Score at c_hat'); ax.set_ylabel('Density')
ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

ax = axes[1]
ax.hist(scores_at_chat[~correct_test], bins=50, density=True, alpha=0.6,
        color='crimson', label='test incorrect')
ax.hist(scores_at_chat[correct_test],  bins=50, density=True, alpha=0.4,
        color='steelblue', label='test correct')
ax.hist(all_pool_vals, bins=50, density=True, alpha=0.4, color='orange',
        label='pool', linestyle='--')
ax.set_title(f'TEST: scores vs pool\npool ≥ incorrect: {frac_pool:.2f}  '
             f'pool-median shift: {shift:+.2f}')
ax.set_xlabel('Score at c_hat')
ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

plt.suptitle('Plant — Pool coverage diagnostic', fontsize=13)
plt.tight_layout(); plt.show()

print(f'\nPool coverage:  pool ≥ incorrect test: {frac_pool:.3f}  (ideal 0.5)')
print(f'  pool median: {np.median(all_pool_vals):.3f}  '
      f'incorrect test median: {np.median(scores_at_chat[~correct_test]):.3f}  '
      f'shift: {shift:+.3f}')

# ── Figure 7: P-values + calibration ─────────────────────────────────────────
print('\nP-value diagnostics...')


def compute_pvalues(sc_np, lb_np, pool_score):
    pred_classes   = sc_np.argmax(axis=1)
    scores_at_chat = sc_np[np.arange(len(sc_np)), pred_classes]
    p_vals  = np.array([np.mean(pool_score[c_hat] >= s)
                        for s, c_hat in zip(scores_at_chat, pred_classes)])
    correct = (pred_classes == lb_np)
    return p_vals, correct, pred_classes


def build_calibrated_pool(pool_score, test_scores_per_class, num_classes):
    """Shift+scale pool per class to match test median + IQR."""
    calibrated = {}
    for c in range(num_classes):
        pool    = pool_score.get(c, np.array([]))
        test_sc = test_scores_per_class.get(c, np.array([]))
        if len(pool) == 0 or len(test_sc) == 0:
            calibrated[c] = pool; continue
        pool_med = np.median(pool);  test_med = np.median(test_sc)
        pool_iqr = np.percentile(pool, 75) - np.percentile(pool, 25)
        test_iqr = np.percentile(test_sc, 75) - np.percentile(test_sc, 25)
        if pool_iqr < 1e-6 or test_iqr < 1e-6:
            calibrated[c] = pool + (test_med - pool_med)
        else:
            calibrated[c] = (pool - pool_med) * (test_iqr / pool_iqr) + test_med
    return calibrated


p_vals_raw, correct_bool, pred_cls = compute_pvalues(sc_np, lb_np, pool_score)

test_scores_per_class = {c: sc_np[pred_cls == c, c]
                          for c in range(NUM_CLASSES)
                          if (pred_cls == c).sum() > 0}
pool_cal   = build_calibrated_pool(pool_score, test_scores_per_class, NUM_CLASSES)
p_vals_cal, _, _ = compute_pvalues(sc_np, lb_np, pool_cal)

print(f'  Raw pool:   incorrect mean_p={p_vals_raw[~correct_bool].mean():.3f}  '
      f'correct mean_p={p_vals_raw[correct_bool].mean():.3f}')
print(f'  Calibrated: incorrect mean_p={p_vals_cal[~correct_bool].mean():.3f}  '
      f'correct mean_p={p_vals_cal[correct_bool].mean():.3f}  (ideal incorrect≈0.5)')

bins_p = np.linspace(0, 1, 21)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

ax = axes[0]
ax.hist(p_vals_raw[correct_bool],  bins=bins_p, density=True, alpha=0.6,
        color='steelblue', label=f'Correct (n={correct_bool.sum()})')
ax.hist(p_vals_raw[~correct_bool], bins=bins_p, density=True, alpha=0.6,
        color='crimson',   label=f'Incorrect (n={(~correct_bool).sum()})')
ax.axhline(1.0, ls='--', color='black', lw=1.2, label='Uniform')
ax.set_title(f'Raw pool p-values\nincorrect mean={p_vals_raw[~correct_bool].mean():.3f}')
ax.set_xlabel('p-value [P(pool ≥ score at c_hat)]'); ax.set_ylabel('Density')
ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

ax = axes[1]
ax.hist(p_vals_cal[correct_bool],  bins=bins_p, density=True, alpha=0.6,
        color='steelblue', label='Correct')
ax.hist(p_vals_cal[~correct_bool], bins=bins_p, density=True, alpha=0.6,
        color='crimson',   label='Incorrect')
ax.axhline(1.0, ls='--', color='black', lw=1.2, label='Uniform')
ax.set_title(f'Calibrated pool p-values\nincorrect mean={p_vals_cal[~correct_bool].mean():.3f}')
ax.set_xlabel('p-value [P(pool_cal ≥ score at c_hat)]')
ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

ax = axes[2]
for pv, label, color in [(p_vals_raw[~correct_bool], 'Raw (H0)',  'crimson'),
                          (p_vals_cal[~correct_bool], 'Cal (H0)',  'darkorange')]:
    if len(pv) == 0: continue
    pv_s = np.sort(pv); unif = np.linspace(0, 1, len(pv_s))
    ax.plot(unif, pv_s, color=color, lw=1.8, label=label)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.6, label='Uniform')
ax.set_xlabel('Uniform quantile'); ax.set_ylabel('Empirical p-value quantile')
ax.set_title('QQ plot — incorrect preds vs Uniform\n(H0 ideal = diagonal)')
ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

ax = axes[3]
class_raw = [p_vals_raw[(pred_cls == c) & ~correct_bool].mean()
             if ((pred_cls == c) & ~correct_bool).sum() > 0 else np.nan
             for c in range(NUM_CLASSES)]
class_cal = [p_vals_cal[(pred_cls == c) & ~correct_bool].mean()
             if ((pred_cls == c) & ~correct_bool).sum() > 0 else np.nan
             for c in range(NUM_CLASSES)]
x = np.arange(NUM_CLASSES)
ax.bar(x - 0.2, class_raw, 0.4, color='crimson',   alpha=0.7, label='Raw pool')
ax.bar(x + 0.2, class_cal, 0.4, color='darkorange', alpha=0.7, label='Calibrated')
ax.axhline(0.5, ls='--', color='black', lw=1, label='Ideal (0.5)')
ax.set_xlabel('Predicted class c_hat'); ax.set_ylabel('Mean p-value (incorrect only)')
ax.set_title('Per-class pool calibration'); ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

plt.suptitle('Plant — P-value diagnostics', fontsize=13)
plt.tight_layout(); plt.show()

# Calibrated pool effect on FDR/Acc
decoy_cal  = apply_strategy(sc_np, pool_cal, pool_error_vectors, 'score_coord', 0.0)
curves_cal = compute_fdr_acc_curves(sc_np, decoy_cal, lb_np)
sc_ref     = raw_results['score_coord']
r          = sc_ref['normalized_rank']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(r, sc_ref['QVAL_true'],       color='black',     lw=2,   ls='--', label='True FDR')
ax.plot(r, sc_ref['QVAL_mixmax'],     color='#1976D2',   lw=1.8,          label=f'Raw  err_st={sc_ref["err_st_mm"]:.3f}')
ax.plot(r, curves_cal['QVAL_mixmax'], color='darkorange', lw=1.8,         label=f'Cal  err_st={curves_cal["err_st_mm"]:.3f}')
ax.set_title('FDR: raw pool vs calibrated'); ax.set_xlabel('Fraction accepted')
ax.set_ylabel('q-value'); ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

ax = axes[1]
ax.plot(r, sc_ref['Acc_true'],        color='black',     lw=2,   ls='--', label='True Acc')
ax.plot(r, sc_ref['Acc_est_MM'],      color='#1976D2',   lw=1.8,          label=f'Raw  err_ta={sc_ref["err_ta_mm"]:.3f}')
ax.plot(r, curves_cal['Acc_est_MM'],  color='darkorange', lw=1.8,         label=f'Cal  err_ta={curves_cal["err_ta_mm"]:.3f}')
ax.set_title('Acc: raw pool vs calibrated'); ax.set_xlabel('Fraction accepted')
ax.set_ylabel('Accuracy'); ax.legend(fontsize=9); ax.grid(ls='--', alpha=0.3)

plt.suptitle('Plant — Effect of pool calibration on FDR/Acc', fontsize=13)
plt.tight_layout(); plt.show()

# MAE summary (raw)
print('\n' + '='*60)
print('MAE SUMMARY — Raw Decoy Strategies')
print('='*60)
print(f"  {'Strategy':<25} {'MAE_ST':>8}  {'MAE_TA':>8}")
print(f"  {'-'*45}")
for strat_name, _ in STRATEGIES:
    c = raw_results[strat_name]
    print(f"  {STRATEGY_LABELS[strat_name]:<25} {c['err_st_mm']:>8.4f}  {c['err_ta_mm']:>8.4f}")
print(f"  {'Calibrated (SC)':<25} {curves_cal['err_st_mm']:>8.4f}  {curves_cal['err_ta_mm']:>8.4f}")

# ============================================================
# FLOW MODEL — all 4 strategies
# ============================================================
print("\n" + "="*70)
print("FLOW TRAINING — all 4 strategies")
print("="*70)

FLOW_EPOCHS    = 30
FLOW_LR        = 3e-4
FLOW_PATIENCE  = 5
FLOW_N         = 12
FLOW_ENC_DIM   = 128
FLOW_SUBSAMPLE = 0.5
FLOW_SEED      = 42

# Subsample 50% of train for flow
n_total  = len(train_scores_raw)
n_flow   = int(n_total * FLOW_SUBSAMPLE)
rng_sub  = np.random.default_rng(FLOW_SEED)
sub_idx  = rng_sub.choice(n_total, size=n_flow, replace=False)
sub_idx.sort()

flow_tr_sc = train_scores_raw[sub_idx]
flow_tr_ft = train_features.numpy()[sub_idx]
flow_tr_lb = train_labels_raw[sub_idx]
print(f"Flow train subset: {n_flow}/{n_total} ({FLOW_SUBSAMPLE*100:.0f}%)"
      f"  acc={(flow_tr_sc.argmax(1) == flow_tr_lb).mean():.4f}")

flow_results = {}
for strat_name, noise_std in STRATEGIES:
    print(f'\n{"#"*60}')
    print(f'  strategy: {strat_name}  (noise={noise_std})')

    train_decoy = apply_strategy(flow_tr_sc, pool_score, pool_error_vectors, strat_name, noise_std)
    train_ds = ScoreFeatureDataset(
        torch.from_numpy(flow_tr_sc).float(),
        torch.from_numpy(flow_tr_ft).float(),
        torch.from_numpy(train_decoy).float(),
        torch.from_numpy(flow_tr_lb).long(),
    )

    flow_path = f'plant_flow_{strat_name}_half.pth'
    flow = ScoreShiftFlowWrapper(
        num_classes = NUM_CLASSES,
        n_flows     = FLOW_N,
        feature_dim = FEATURE_DIM,
        hidden_dim  = 256,
        encoder_dim = FLOW_ENC_DIM,
        clip_val    = 5.0,
    ).to(DEVICE)

    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=DEVICE, weights_only=False))
        print(f"  Loaded from {flow_path}")
    else:
        print(f"  Training → {flow_path}")
        flow.train_flow(train_ds, epochs=FLOW_EPOCHS, lr=FLOW_LR,
                         batch_size=256, device=str(DEVICE),
                         patience=FLOW_PATIENCE, grad_clip=1.0)
        torch.save(flow.state_dict(), flow_path)
        print(f"  Saved → {flow_path}")

    flow.eval()

    # Build test dataset (pool decoys as placeholder target)
    tset_decoy_pool = apply_strategy(sc_np, pool_score, pool_error_vectors, strat_name, noise_std)
    test_ds = ScoreFeatureDataset(
        torch.from_numpy(sc_np).float(),
        torch.from_numpy(ft_np).float(),
        torch.from_numpy(tset_decoy_pool).float(),
        torch.from_numpy(lb_np).long(),
    )
    ms_np, ds_flow_np, ls_np = flow.generate_decoys(test_ds, device=str(DEVICE))
    curves_flow = compute_fdr_acc_curves(ms_np, ds_flow_np, ls_np)
    flow_results[strat_name] = curves_flow
    print(f"  true_acc={curves_flow['true_acc']:.3f}  "
          f"err_st={curves_flow['err_st_mm']:.3f}  err_ta={curves_flow['err_ta_mm']:.3f}")

print('\nAll flow experiments done.')

# ── Figure: Raw vs Flow FDR/Acc ──────────────────────────────────────────────
fig, axes = plt.subplots(2, n_strats, figsize=(5 * n_strats, 8))
r = raw_results['score_coord']['normalized_rank']

for col, (strat_name, _) in enumerate(STRATEGIES):
    raw_c  = raw_results[strat_name]
    flow_c = flow_results[strat_name]
    color  = STRATEGY_COLORS[strat_name]

    ax = axes[0][col]
    ax.plot(r, raw_c['QVAL_true'],    color='black',      lw=2,    ls='--', label='True FDR')
    ax.plot(r, raw_c['QVAL_mixmax'],  color=color,        lw=1.8,           label=f'Raw  err_st={raw_c["err_st_mm"]:.3f}')
    ax.plot(r, flow_c['QVAL_mixmax'], color='darkorange', lw=1.8,  ls='--', label=f'Flow err_st={flow_c["err_st_mm"]:.3f}')
    ax.set_title(STRATEGY_LABELS[strat_name])
    ax.set_xlabel('Fraction accepted'); ax.legend(fontsize=7); ax.grid(ls='--', alpha=0.3); ax.set_ylim(0, 1.05)
    if col == 0: ax.set_ylabel('q-value (FDR)')

    ax = axes[1][col]
    ax.plot(r, raw_c['Acc_true'],     color='black',      lw=2,    ls='--', label='True Acc')
    ax.plot(r, raw_c['Acc_est_MM'],   color=color,        lw=1.8,           label=f'Raw  err_ta={raw_c["err_ta_mm"]:.3f}')
    ax.plot(r, flow_c['Acc_est_MM'],  color='darkorange', lw=1.8,  ls='--', label=f'Flow err_ta={flow_c["err_ta_mm"]:.3f}')
    ax.set_xlabel('Fraction accepted'); ax.legend(fontsize=7); ax.grid(ls='--', alpha=0.3)
    if col == 0: ax.set_ylabel('Accuracy')

plt.suptitle('Plant — Raw vs Flow decoys (all strategies)', fontsize=13)
plt.tight_layout(); plt.show()

# ── Figure: Raw vs Flow score distributions ───────────────────────────────────
ref_sc = raw_results['score_coord']
bins = np.linspace(ref_sc['pred_scores'].min() - 0.3,
                   ref_sc['pred_scores'].max() + 0.3, 60)
inc_mask = ref_sc['correct'] == 0

fig, axes = plt.subplots(2, n_strats, figsize=(5 * n_strats, 8))
for col, (strat_name, _) in enumerate(STRATEGIES):
    raw_c  = raw_results[strat_name]
    flow_c = flow_results[strat_name]
    color  = STRATEGY_COLORS[strat_name]

    ax = axes[0][col]
    sns.histplot(raw_c['pred_scores'],  bins=bins, stat='density', color='steelblue',
                 kde=True, fill=True, alpha=0.3, label='model', ax=ax)
    sns.histplot(raw_c['decoy_scores'], bins=bins, stat='density', color=color,
                 kde=True, fill=True, alpha=0.4, label='decoy (raw)', ax=ax)
    ax.set_title(f'{STRATEGY_LABELS[strat_name]}  RAW  err_st={raw_c["err_st_mm"]:.3f}')
    ax.set_xlabel('Max logit'); ax.legend(fontsize=7)
    if col == 0: ax.set_ylabel('Density (raw)')

    ax = axes[1][col]
    sns.histplot(flow_c['pred_scores'],  bins=bins, stat='density', color='steelblue',
                 kde=True, fill=True, alpha=0.3, label='model', ax=ax)
    sns.histplot(flow_c['decoy_scores'], bins=bins, stat='density', color='darkorange',
                 kde=True, fill=True, alpha=0.4, label='decoy (flow)', ax=ax)
    ax.set_title(f'FLOW  err_st={flow_c["err_st_mm"]:.3f}')
    ax.set_xlabel('Max logit'); ax.legend(fontsize=7)
    if col == 0: ax.set_ylabel('Density (flow)')

plt.suptitle('Plant — Raw vs Flow score distributions', fontsize=13)
plt.tight_layout(); plt.show()

# MAE summary: Raw vs Flow
print('\n' + '='*65)
print('MAE SUMMARY — Raw vs Flow (all strategies)')
print('='*65)
print(f"  {'Strategy':<25} {'Raw ST':>8}  {'Raw TA':>8}  {'Flow ST':>8}  {'Flow TA':>8}")
print(f"  {'-'*65}")
for strat_name, _ in STRATEGIES:
    rc = raw_results[strat_name]; fc = flow_results[strat_name]
    print(f"  {STRATEGY_LABELS[strat_name]:<25} "
          f"{rc['err_st_mm']:>8.4f}  {rc['err_ta_mm']:>8.4f}  "
          f"{fc['err_st_mm']:>8.4f}  {fc['err_ta_mm']:>8.4f}")
print(f"  {'Calibrated pool (SC)':<25} {curves_cal['err_st_mm']:>8.4f}  {curves_cal['err_ta_mm']:>8.4f}")
