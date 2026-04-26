import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from data_processing.score_feature_dataset import *
from data_processing.negative_scores_pool import *
from flows.shift_flow import ScoreShiftFlowWrapper
from utils.other_methods import *
from utils.visualize_distributions import *
from fdr.fdr_control import *
from fdr.plot_fdr import *

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   16,
    'axes.labelsize':   14,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  10,
    'figure.titlesize': 18,
})

from robustness.tools.helpers import get_label_mapping
from robustness.tools import folder
from robustness.tools.breeds_helpers import (
    make_living17,
    make_entity13,
    make_entity30,
    make_nonliving26
)

DATA_DIR = "/home/arina/imagenet"
BATCH_SIZE = 64

IMAGENET_C = [
    "fog", "frost", "motion_blur", "brightness", "zoom_blur",
    "snow", "defocus_blur", "glass_blur", "gaussian_noise",
    "shot_noise", "impulse_noise", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression", "speckle_noise", "spatter",
    "gaussian_blur", "saturate"
]
SEVERITIES = [1, 2, 3, 4, 5]

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


# ============================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: получить make_* и ret для breeds
# ============================================================
def get_breeds_ret(hierarchy_dir: str, name: str):
    """Возвращает ret = make_*(hierarchy_dir, split='good') для нужного датасета."""
    if name == "living17":
        return make_living17(hierarchy_dir, split="good")
    elif name == "entity13":
        return make_entity13(hierarchy_dir, split="good")
    elif name == "entity30":
        return make_entity30(hierarchy_dir, split="good")
    elif name == "nonliving26":
        return make_nonliving26(hierarchy_dir, split="good")
    else:
        raise ValueError(f"Unknown breeds name: {name}. "
                         f"Choose from: living17, entity13, entity30, nonliving26")


def get_imagenet_breeds(batch_size, data_dir, name="living17"):

    hierarchy_dir = f"{data_dir}/imagenet_class_hierarchy"

    # --- Выбор датасета ---
    ret = get_breeds_ret(hierarchy_dir, name)

    # --- Label mappings ---
    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4717, 0.4499, 0.3837],
                             [0.2600, 0.2516, 0.2575])
    ])

    # --- Train / Val split ---
    trainset = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform,
        label_mapping=source_label_mapping
    )
    targetset = folder.ImageFolder(
        root=f"{data_dir}/imagenetv1/train/",
        transform=transform,
        label_mapping=target_label_mapping
    )

    idx = np.arange(len(trainset))
    np.random.seed(42)
    np.random.shuffle(idx)

    train_idx = idx[:len(idx) - 10000]
    val_idx   = idx[len(idx) - 10000:]

    train_subset = torch.utils.data.Subset(trainset, train_idx)
    val_subset   = torch.utils.data.Subset(trainset, val_idx)

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    testsets    = []
    testloaders = []

    def add_loader(ds):
        testsets.append(ds)
        testloaders.append(
            torch.utils.data.DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=4
            )
        )

    # --- Clean sets ---
    # [0] val из train (source)
    add_loader(val_subset)
    # [1] targetset (target, full train)
    add_loader(targetset)

    # [2] ImageNet val source
    add_loader(folder.ImageFolder(
        f"{data_dir}/imagenetv1/val/",
        transform=transform,
        label_mapping=source_label_mapping
    ))

    # [3] ImageNet val target
    add_loader(folder.ImageFolder(
        f"{data_dir}/imagenetv1/val/",
        transform=transform,
        label_mapping=target_label_mapping
    ))

    # --- Corruptions SOURCE mapping ---
    print(f"\n  Загружаем corruptions (source)...")
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            path = f"{data_dir}/imagenet-c/{corruption}/{severity}"
            if os.path.isdir(path):
                add_loader(folder.ImageFolder(
                    root=path,
                    transform=transform,
                    label_mapping=source_label_mapping
                ))
            else:
                print(f"    ⚠ Пропуск (не найден): {path}")

    # --- Corruptions TARGET mapping ---
    print(f"  Загружаем corruptions (target)...")
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            path = f"{data_dir}/imagenet-c/{corruption}/{severity}"
            if os.path.isdir(path):
                add_loader(folder.ImageFolder(
                    root=path,
                    transform=transform,
                    label_mapping=target_label_mapping
                ))
            else:
                print(f"    ⚠ Пропуск (не найден): {path}")

    print(f"\n✅ BREEDS '{name}' готов!")
    print(f"   Train size       : {len(train_subset)}")
    print(f"   Val size         : {len(val_subset)}")
    print(f"   Source classes   : {len(ret[1][0])}")
    print(f"   Target classes   : {len(ret[1][1])}")
    print(f"   Total test sets  : {len(testsets)}")
    print(f"     - 4 clean sets")
    print(f"     - corruption sets (source + target)")

    return trainset, train_subset, val_subset, trainloader, testsets, testloaders


# ============================================================
# MODEL
# ============================================================
class BREEDSClassifier(nn.Module):
    """
    ResNet50 (pretrained ImageNet) адаптированный под BREEDS.

    Архитектура хвоста:
        backbone → avgpool → flatten [2048]
                → linear1 [2048→feature_dim]  (penultimate)
                → ReLU
                → linear2 [feature_dim→num_classes]

    get_features() возвращает feature_dim-мерный вектор.
    """
    def __init__(self, num_classes: int, pretrained: bool = True,
                 feature_dim: int = 640, freeze_backbone: bool = False):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Убираем оригинальный fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → [B, 2048, 1, 1]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Head ──────────────────────────────────────────────────────────
        self.linear1 = nn.Linear(2048, feature_dim)
        self.linear2 = nn.Linear(feature_dim, num_classes)

        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает feature_dim-мерный вектор (penultimate layer)."""
        x = self.backbone(x)   # [B, 2048, 1, 1]
        x = x.flatten(1)       # [B, 2048]
        x = self.linear1(x)    # [B, feature_dim]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.get_features(x)
        logits = self.linear2(F.relu(feats))
        return logits

    def forward_with_features(self, x: torch.Tensor):
        """Возвращает (logits, features) одним проходом."""
        feats  = self.get_features(x)
        logits = self.linear2(F.relu(feats))
        return logits, feats


# ============================================================
# TRAINING
# ============================================================
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train_breeds_classifier(
        model,
        trainloader,
        valloader,
        epochs:    int   = 30,
        lr:        float = 0.01,
        device:    str   = 'cuda',
        save_path: str   = 'breeds_classifier.pth',
        patience:  int   = 5,
) -> dict:
    """
    Обучает BREEDSClassifier на source-split.
    Возвращает историю лоссов/аккуратностей.
    """
    model = model.to(device)

    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True
    )
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = nn.CrossEntropyLoss()

    history    = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc   = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(trainloader,
                                   desc=f'Epoch {epoch}/{epochs}',
                                   leave=False):
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

        train_loss = total_loss / total
        train_acc  = correct   / total

        # ── Val ───────────────────────────────────────────────────────────
        val_acc = evaluate_accuracy(model, valloader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} "
              f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model  (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Загружаем лучшие веса
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\n✅ Training done. Best val_acc={best_acc:.4f}")
    return history


# ============================================================
# UTILITIES
# ============================================================
def evaluate_accuracy(model, loader, device='cuda') -> float:
    """Считает top-1 accuracy по даталоадеру."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits  = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


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
        pool = pool_score.get(c, [])
        if len(pool) == 0:
            continue
        dc_np[mask, c] = rng.choice(pool, size=mask.sum(), replace=True)
    return dc_np


def collect_train_scores_breeds(
        model,
        train_dataset,
        batch_size: int = 256,
        device:     str = 'cuda',
) -> tuple:
    """
    Аналог collect_train_scores из CIFAR-пайплайна.
    Возвращает (scores [N, C], labels [N]).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4
    )
    sc_list, lb_list = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Collecting train scores'):
            images  = images.to(device)
            feats   = model.get_features(images)
            scores  = model.linear2(F.relu(feats))
            sc_list.append(scores.cpu().numpy())
            lb_list.append(labels.numpy())

    return np.concatenate(sc_list), np.concatenate(lb_list)


def create_score_dataset_breeds(
        dataset,
        model,
        pool_score:   dict,
        pool_vectors: dict,
        strategy:     str = 'score_coord',
        device:       str = 'cuda',
        batch_size:   int = 256,
) -> 'ScoreFeatureDataset':
    """
    Аналог create_score_dataset_with_decoys для BREEDS.
    Возвращает ScoreFeatureDataset.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=4
    )
    sc_list, ft_list, dc_list, lb_list = [], [], [], []
    rng = np.random.default_rng(0)

    with torch.no_grad():
        for images, labels in tqdm(loader,
                                   desc=f'Score dataset [{strategy}]',
                                   leave=False):
            images = images.to(device)
            feats  = model.get_features(images)
            scores = model.linear2(F.relu(feats))

            sc_np = scores.cpu().numpy()

            if strategy == 'score_coord':
                dc_np = _build_decoy_score_coord(sc_np, pool_score, rng)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            sc_list.append(scores.cpu())
            ft_list.append(feats.cpu())
            dc_list.append(torch.from_numpy(dc_np).float())
            lb_list.append(labels)

    return ScoreFeatureDataset(
        torch.cat(sc_list),
        torch.cat(ft_list),
        torch.cat(dc_list),
        torch.cat(lb_list),
    )


# ============================================================
# CONFIG — BREEDS
# ============================================================
DATA_DIR        = "/home/arina/imagenet"
BREEDS_NAME     = "nonliving26"          # living17 | entity13 | entity30 | nonliving26
BATCH_SIZE      = 64
DEVICE          = 'cuda:2' if torch.cuda.is_available() else 'cpu'
DECOY_STRATEGY  = 'score_coord'
CLASSIFIER_PATH = f'breeds_{BREEDS_NAME}_classifier.pth'
FLOWS_PATH      = f'cond_flows_breeds_{BREEDS_NAME}_normalize.pth'
FEATURE_DIM     = 640

# ============================================================
# STEP 1: ЗАГРУЗКА ДАННЫХ
# ============================================================
print(f"\n{'='*60}")
print(f"BREEDS: {BREEDS_NAME}")
print(f"{'='*60}")

(trainset, train_subset, val_subset,
 trainloader, testsets, testloaders) = get_imagenet_breeds(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    name=BREEDS_NAME
)

# Количество классов из source split — надёжно берём из маппинга
hierarchy_dir = f"{DATA_DIR}/imagenet_class_hierarchy"
ret           = get_breeds_ret(hierarchy_dir, BREEDS_NAME)
NUM_CLASSES   = len(ret[1][0])   # source классов
print(f"NUM_CLASSES (source): {NUM_CLASSES}")

# val loader = testsets[0] = val_subset (source mapping)
valloader = testloaders[0]

# ============================================================
# STEP 2: МОДЕЛЬ
# ============================================================
print(f"\n{'='*60}")
print("MODEL")
print(f"{'='*60}")

breeds_model = BREEDSClassifier(
    num_classes     = NUM_CLASSES,
    pretrained      = True,
    feature_dim     = FEATURE_DIM,
    freeze_backbone = False,
).to(DEVICE)

if os.path.exists(CLASSIFIER_PATH):
    breeds_model.load_state_dict(
        torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    print(f"✓ Classifier loaded from {CLASSIFIER_PATH}")
else:
    print("Training classifier from scratch...")
    history = train_breeds_classifier(
        model       = breeds_model,
        trainloader = trainloader,
        valloader   = valloader,
        epochs      = 5,
        lr          = 0.01,
        device      = DEVICE,
        save_path   = CLASSIFIER_PATH,
        patience    = 3,
    )

breeds_model.eval()

val_acc = evaluate_accuracy(breeds_model, valloader, DEVICE)
print(f"Val accuracy (source): {val_acc:.4f}")

# ============================================================
# STEP 3: COLLECT TRAIN SCORES + POOLS
# ============================================================
print(f"\n{'='*60}")
print("COLLECTING TRAIN SCORES")
print(f"{'='*60}")

train_scores_raw, train_labels_raw = collect_train_scores_breeds(
    breeds_model, train_subset, batch_size=256, device=DEVICE
)
print(f"Scores: {train_scores_raw.shape}  "
      f"acc={(train_scores_raw.argmax(1) == train_labels_raw).mean():.4f}")

print(f"\n{'='*60}")
print("BUILDING DECOY POOLS")
print(f"{'='*60}")

pool_score, pool_vectors = build_error_conditioned_pools(
    train_scores_raw, train_labels_raw, NUM_CLASSES, verbose=True
)

source_logits_np = train_scores_raw
source_labels_np = train_labels_raw

# ============================================================
# STEP 4: FLOW
# ============================================================
print(f"\n{'='*60}")
print("FLOW MODEL")
print(f"{'='*60}")

# score_shift_flow = ScoreShiftFlowWrapper(
#     num_classes = NUM_CLASSES,
#     n_flows     = 12,
#     feature_dim = FEATURE_DIM,
#     hidden_dim  = 256,
#     encoder_dim = 128,
# ).to(DEVICE)

score_shift_flow = ScoreShiftFlowWrapper(
    num_classes = NUM_CLASSES,
    n_flows     = 12,
    feature_dim = FEATURE_DIM,
    hidden_dim  = 256,
    encoder_dim = 128,
    clip_val    = 5.0,      # ← OOD защита
).to(DEVICE)

print(f"Creating train score dataset (strategy='{DECOY_STRATEGY}')...")
train_score_dataset = create_score_dataset_breeds(
    train_subset, breeds_model,
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
    print("Training Flow...")
    score_shift_flow.train_flow(
        train_score_dataset, epochs=30, lr=3e-4,
        batch_size=256, device=DEVICE, patience=5, grad_clip=1.0
    )
    torch.save(score_shift_flow.state_dict(), FLOWS_PATH)
    print(f"✓ Flow saved to {FLOWS_PATH}")

score_shift_flow.eval()

# ── Калибровка температуры на train ──────────────────────────────────────────
source_logits, _, source_labels = score_shift_flow.generate_decoys(
    train_score_dataset, device=DEVICE
)
source_logits_t = torch.tensor(source_logits).to(DEVICE)
source_labels_t = torch.tensor(source_labels).to(DEVICE)
temp            = calibration_temp(source_logits_t, source_labels_t)
scaled_source   = source_logits_t / temp
print(f"✓ Source: {len(source_labels)} samples, temp={temp:.4f}")

# ============================================================
# STEP 5: BASELINE METHODS
# ============================================================
try:
    from utils.other_methods import predict_COT
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

BASELINE_METHODS = {
    'ATC'   : predict_ATC_maxconf,
    'ATC-NE': predict_ATC_negent,
    'AC'    : predict_AC,
    'DOC'   : predict_DOC,
}
if COT_AVAILABLE:
    BASELINE_METHODS['COT'] = predict_COT

# ============================================================
# STEP 6: ТЕСТОВЫЕ НАБОРЫ — ИМЕНА
# ============================================================
# Структура testsets (см. get_imagenet_breeds):
#   [0]  val_subset          (source, train split)
#   [1]  targetset           (target, full train)
#   [2]  imagenetv1/val      (source mapping)
#   [3]  imagenetv1/val      (target mapping)
#   [4..4+N_CORRUPTIONS-1]   corruptions SOURCE
#   [4+N_CORRUPTIONS..]      corruptions TARGET
# (N_CORRUPTIONS может быть меньше 19*5, если папки отсутствуют)

def build_testset_names(n_testsets: int) -> list:
    """
    Строит список имён тестовых наборов.
    Первые 4 — clean, дальше corruption source, потом corruption target.
    Если папок не хватало, имён генерируется ровно n_testsets.
    """
    names = [
        'val_source',
        'val_target',
        'imagenet_val_source',
        'imagenet_val_target',
    ]
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            names.append(f'corr_source_{corruption}_sev{severity}')
    for corruption in IMAGENET_C:
        for severity in SEVERITIES:
            names.append(f'corr_target_{corruption}_sev{severity}')

    # Обрезаем или дополняем до реального числа датасетов
    if len(names) > n_testsets:
        names = names[:n_testsets]
    while len(names) < n_testsets:
        names.append(f'unknown_{len(names)}')
    return names

TESTSET_NAMES = build_testset_names(len(testsets))
print(f"Total test sets: {len(testsets)}, names: {len(TESTSET_NAMES)}")

# ============================================================
# STEP 7: ОСНОВНОЙ ЦИКЛ ПО ТЕСТОВЫМ НАБОРАМ
# ============================================================
os.makedirs('BREEDS/figures/diagnostics', exist_ok=True)
os.makedirs('BREEDS/figures/accuracy',    exist_ok=True)
os.makedirs('BREEDS/figures/comparison',  exist_ok=True)

all_results = []
tile_data = {}

COLOR_ENGPE    = '#1976D2'
COLOR_ENGPE_TA = '#FF6D00'

for testset_idx, (testset, testloader) in enumerate(zip(testsets, testloaders)):

    ds_name = TESTSET_NAMES[testset_idx]
    print(f"\n{'─'*60}")
    print(f"  TESTSET [{testset_idx+1}/{len(testsets)}]: {ds_name}")
    print(f"{'─'*60}")

    # ── Создаём score dataset для текущего тестового набора ──────────────
    try:
        test_score_ds = create_score_dataset_breeds(
            testset, breeds_model,
            pool_score   = pool_score,
            pool_vectors = pool_vectors,
            strategy     = DECOY_STRATEGY,
            device       = DEVICE,
        )
    except Exception as e:
        print(f"  ✗ Ошибка создания score dataset: {e}")
        continue

    # ── Генерация decoys через flow ───────────────────────────────────────
    ms, ds_flow, ls = score_shift_flow.generate_decoys(
        test_score_ds, device=DEVICE
    )

    n = len(ls)
    if n == 0:
        print(f"  ✗ Пустой датасет, пропускаем")
        continue

    target_logits = torch.tensor(ms).to(DEVICE)
    target_labels = torch.tensor(ls).to(DEVICE)

    true_acc_full = float((ms.argmax(axis=1) == ls).mean())
    scaled_target = target_logits / temp

    print(f"  N={n}, true_acc={true_acc_full:.4f}")

    # ── Baseline methods ──────────────────────────────────────────────────
    baseline_estimates = {}
    baseline_errors    = {}

    for method_name, method_func in BASELINE_METHODS.items():
        try:
            estimate = float(method_func(
                scaled_source, source_labels_t, scaled_target
            ))
        except Exception as e:
            print(f"  {method_name}: FAILED ({e})")
            estimate = np.nan
        estimate = np.clip(estimate, 0.0, 1.0)
        error    = abs(estimate - true_acc_full)
        baseline_estimates[method_name] = estimate
        baseline_errors[method_name]    = error
        print(f"  {method_name}: est={estimate:.4f}  true={true_acc_full:.4f}  "
              f"err={error:.4f}")

    # ── Подготовка ────────────────────────────────────────────────────────
    pred_scores  = np.max(ms,      axis=1)
    pred_label   = np.argmax(ms,   axis=1)
    decoy_scores = np.max(ds_flow, axis=1)
    true_labels  = ls
    n_samples    = len(true_labels)

    probs_np      = F.softmax(torch.tensor(ms).float(), dim=1).numpy()
    mano_fro      = np.linalg.norm(probs_np, ord='fro') / np.sqrt(n_samples)
    mano_fro_norm = float(np.clip(
        (mano_fro - 1.0 / np.sqrt(NUM_CLASSES)) / (1.0 - 1.0 / np.sqrt(NUM_CLASSES)),
        0.0, 1.0))

    correct_pred = (true_labels == pred_label).astype(int)
    print(f"  accuracy : {correct_pred.mean():.4f}")

    # ── Sort по pred_score ────────────────────────────────────────────────
    sort_idx           = np.argsort(pred_scores)
    pred_scores_sorted = pred_scores[sort_idx]
    label_sorted       = true_labels[sort_idx]
    pred_label_sorted  = pred_label[sort_idx]
    correct_pred_s     = (label_sorted == pred_label_sorted).astype(int)

    # ── True FDR ──────────────────────────────────────────────────────────
    FD        = 1 - correct_pred_s
    FD_CF     = np.cumsum(FD[::-1])[::-1]
    D_CF      = np.arange(0, n_samples)[::-1] + 1
    FDR_true  = np.clip(FD_CF / D_CF, 0, 1)
    QVAL_true = np.minimum.accumulate(FDR_true)
    QVAL_true = np.clip(QVAL_true, 0, 1)

    # ── Mix-Max FDR ───────────────────────────────────────────────────────
    pi0 = 0.0

    sorted_decoys           = np.sort(decoy_scores)
    unique_z_vals, counts_z = np.unique(decoy_scores, return_counts=True)
    n_unique_z              = len(unique_z_vals)

    counts_w_leq_z = np.searchsorted(pred_scores_sorted, unique_z_vals, side='left')
    counts_z_leq_z = np.searchsorted(sorted_decoys,      unique_z_vals, side='left')

    P_W_leq_z = np.clip(
        (counts_w_leq_z - pi0 * counts_z_leq_z) / ((1 - pi0) * n_samples),
        0, 1
    )
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
        F_1   = 0.0 if z_idx >= n_unique_z else (
            (1 - pi0) * np.sum(R_j[z_idx:] * counts_z[z_idx:])
        )
        fdr_values[i] = (F_0 + F_1) / D if D > 0 else 0.0

    fdr_values  = np.clip(fdr_values, 0, 1)
    QVAL_mixmax = np.clip(
        np.minimum.accumulate(fdr_values[::-1]), 0, 1
    )

    # ── TDC FDR ───────────────────────────────────────────────────────────
    TDC       = (pred_scores > decoy_scores).astype(int)
    TDC_score = np.maximum(pred_scores, decoy_scores)

    tdc_sort_idx = np.argsort(TDC_score)
    TDC_score_s  = TDC_score[tdc_sort_idx]
    TDC_label_s  = TDC[tdc_sort_idx]

    FD_CF_tdc = np.cumsum((1 - TDC_label_s)[::-1])[::-1]
    D_CF_tdc  = np.maximum(
        np.arange(0, n_samples)[::-1] + 1 - FD_CF_tdc, 1
    )
    QVAL_TDC = np.clip(
        np.minimum.accumulate(np.clip(FD_CF_tdc / D_CF_tdc, 0, 1)),
        0, 1
    )

    # ── Acc estimation ────────────────────────────────────────────────────
    pi0_tdc = np.clip(float(QVAL_TDC[0]),    0.0, 1.0)
    pi0_mm  = np.clip(float(QVAL_mixmax[0]), 0.0, 1.0)

    Acc_est    = np.zeros(n_samples)
    Acc_est_MM = np.zeros(n_samples)
    Acc_true   = np.zeros(n_samples)

    for i in range(n_samples):
        TP_true     = correct_pred_s[i:].sum()
        TN_true     = (1 - correct_pred_s[:i]).sum()
        Acc_true[i] = (TP_true + TN_true) / n_samples

        accepted      = n_samples - i
        FP_tdc        = accepted * QVAL_TDC[i]
        TP_tdc        = accepted * (1 - QVAL_TDC[i])
        TN_tdc        = n_samples * pi0_tdc - FP_tdc
        Acc_est[i]    = np.clip((TP_tdc + TN_tdc) / n_samples, 0.0, 1.0)

        FP_mm         = accepted * QVAL_mixmax[i]
        TP_mm         = accepted * (1 - QVAL_mixmax[i])
        TN_mm         = n_samples * pi0_mm - FP_mm
        Acc_est_MM[i] = np.clip((TP_mm + TN_mm) / n_samples, 0.0, 1.0)

    Acc_true = np.clip(Acc_true, 0.0, 1.0)

    # ── Metrics ───────────────────────────────────────────────────────────
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

    baseline_errors_ta = {
        m: abs(baseline_estimates[m] - acc_ta_true)
        for m in BASELINE_METHODS
    }

    print(f"\n  {'':28} {'ACC_ST':>10}  {'ACC_TA':>10}")
    print(f"  {'─'*52}")
    print(f"  {'True':<28} {acc_st_true:>10.4f}  {acc_ta_true:>10.4f}")
    print(f"  {'TDC  (est|err)':<28} "
          f"{acc_st_est_tdc:>6.4f} {err_st_tdc:>+.4f}  "
          f"{acc_ta_est_tdc:>6.4f} {err_ta_tdc:>+.4f}")
    print(f"  {'Mix-Max (est|err)':<28} "
          f"{acc_st_est_mm:>6.4f} {err_st_mm:>+.4f}  "
          f"{acc_ta_est_mm:>6.4f} {err_ta_mm:>+.4f}")

    # ── Store tile data for aggregate plots (all testsets) ────────────────
    total_TP  = int(correct_pred_s.sum())
    TP_from_i = np.cumsum(correct_pred_s[::-1])[::-1]
    D_from_i  = np.arange(n_samples, 0, -1)
    tile_data[ds_name] = dict(
        normalized_rank = np.arange(n_samples) / n_samples,
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
        n_samples       = n_samples,
        ds_name         = ds_name,
        mano_fro_norm   = mano_fro_norm,
        acc_st_true     = acc_st_true,
    )

    # ── Figures (только для первых 10 наборов) ────────────────────────────
    if testset_idx < 10:
        safe_name       = f'breeds_{BREEDS_NAME}_{ds_name}'
        normalized_rank = np.arange(n_samples) / n_samples

        # Figure 1: Score distributions
        fig, ax = plt.subplots(figsize=(6, 4))
        bin_width = 0.05
        bins = np.arange(
            pred_scores.min() - 0.5,
            pred_scores.max() + 0.5,
            bin_width
        )
        incorrect_mask = true_labels != pred_label
        sns.histplot(pred_scores, bins=bins, stat='density', color='blue',
                     kde=True, fill=True, alpha=0.3,
                     label='model_mixture', ax=ax)
        sns.histplot(decoy_scores, bins=bins, stat='density', color='orange',
                     kde=True, fill=True, alpha=0.3,
                     label='null', ax=ax)
        if incorrect_mask.any():
            sns.histplot(pred_scores[incorrect_mask], bins=bins,
                         stat='density', color='red',
                         kde=True, fill=True, alpha=0.3,
                         label='incorrect', ax=ax)
        ax.set_title(f'{ds_name} — Score Distributions')
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            f'BREEDS/figures/diagnostics/{safe_name}_scores.png', dpi=150)
        plt.savefig(
            f'BREEDS/figures/diagnostics/{safe_name}_scores.pdf')
        plt.close()

        # Figure 2: FDR vs score threshold
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pred_scores_sorted, QVAL_mixmax, label='Mix-Max FDR')
        ax.plot(pred_scores_sorted, QVAL_true,   label='True FDR')
        ax.set_xlabel('Score threshold')
        ax.set_ylabel('q-value (FDR)')
        ax.set_title(f'{ds_name} — FDR vs Score Threshold')
        ax.legend(); ax.grid()
        plt.tight_layout()
        plt.savefig(
            f'BREEDS/figures/diagnostics/{safe_name}_fdr_vs_score.png',
            dpi=150)
        plt.savefig(
            f'BREEDS/figures/diagnostics/{safe_name}_fdr_vs_score.pdf')
        plt.close()

        # Figure 3: Acc & FDR vs normalised rank
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(normalized_rank, QVAL_TDC,    label='TDC FDR')
        ax.plot(normalized_rank, QVAL_mixmax, label='Mix-Max FDR')
        ax.plot(normalized_rank, QVAL_true,   label='True FDR')
        ax.plot(normalized_rank, Acc_true,    label='True Acc')
        ax.plot(normalized_rank, Acc_est,     label='Est Acc with TDC')
        ax.plot(normalized_rank, Acc_est_MM,  label='Est Acc with Mix-Max')
        ax.set_xlabel('Normalized rank (fraction accepted)')
        ax.set_ylabel('Value')
        ax.set_title(f'{ds_name}')
        ax.legend(); ax.grid()
        plt.tight_layout()
        plt.savefig(
            f'BREEDS/figures/accuracy/{safe_name}_acc_fdr.png', dpi=150)
        plt.savefig(
            f'BREEDS/figures/accuracy/{safe_name}_acc_fdr.pdf')
        plt.close()

    # ── Собираем результаты ───────────────────────────────────────────────
    row = dict(
        testset        = ds_name,
        testset_idx    = testset_idx,
        n_samples      = n_samples,
        is_source      = ('source' in ds_name),
        is_target      = ('target' in ds_name),
        is_clean       = testset_idx < 4,
        true_acc       = true_acc_full,
        # ── True ──────────────────────────────────────────────────────────
        acc_st_true    = acc_st_true,
        acc_ta_true    = acc_ta_true,
        # ── ENPE (TDC) ────────────────────────────────────────────────────
        acc_st_tdc     = acc_st_est_tdc,
        acc_ta_tdc     = acc_ta_est_tdc,
        err_st_tdc     = err_st_tdc,
        err_ta_tdc     = err_ta_tdc,
        # ── ENPE-TA (Mix-Max) ─────────────────────────────────────────────
        acc_st_mm      = acc_st_est_mm,
        acc_ta_mm      = acc_ta_est_mm,
        err_st_mm      = err_st_mm,
        err_ta_mm      = err_ta_mm,
        # ── MANO ──────────────────────────────────────────────────────────
        mano_fro_norm  = mano_fro_norm,
    )
    # baselines
    for m in BASELINE_METHODS:
        row[f'est_{m}']    = baseline_estimates[m]
        row[f'err_{m}']    = baseline_errors[m]
        row[f'err_ta_{m}'] = baseline_errors_ta[m]

    all_results.append(row)

# ============================================================
# STEP 8: AGGREGATE SUMMARY
# ============================================================
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(
        f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_results.csv',
        index=False
    )
    print(f"\n✓ results.csv сохранён ({len(df)} тестовых наборов)")

    # ── Группировки ───────────────────────────────────────────────────────
    groups = {
        'all':         df,
        'clean':       df[df['is_clean']],
        'corruptions': df[~df['is_clean']],
        'source':      df[df['is_source']],
        'target':      df[df['is_target']],
    }

    # ── Текстовый summary ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("AGGREGATE SUMMARY")
    print("="*80)

    for group_name, gdf in groups.items():
        if len(gdf) == 0:
            continue

        print(f"\n  ── Group: {group_name} (n={len(gdf)}) ──")
        print(f"  {'Metric':<38} {'Mean_Est':>9}  {'Mean_True':>9}  "
              f"{'Mean_Err':>9}  {'Std_Err':>9}")
        print(f"  {'─'*80}")

        # Helper: вывести строку estimation + true + error
        def _print_row(label, col_est, col_true, col_err):
            est_v  = gdf[col_est].dropna().values  if col_est  else None
            true_v = gdf[col_true].dropna().values if col_true else None
            err_v  = gdf[col_err].dropna().values  if col_err  else None
            est_m  = est_v.mean()  if est_v  is not None and len(est_v)  > 0 else float('nan')
            true_m = true_v.mean() if true_v is not None and len(true_v) > 0 else float('nan')
            err_m  = err_v.mean()  if err_v  is not None and len(err_v)  > 0 else float('nan')
            err_s  = err_v.std()   if err_v  is not None and len(err_v)  > 0 else float('nan')
            print(f"  {label:<38} {est_m:>9.4f}  {true_m:>9.4f}  "
                  f"{err_m:>9.4f}  {err_s:>9.4f}")

        # ── True baselines ──────────────────────────────────────────────
        true_st = gdf['acc_st_true'].dropna().values
        true_ta = gdf['acc_ta_true'].dropna().values
        print(f"  {'True ACC_ST':<38} {'─':>9}  {true_st.mean():>9.4f}  "
              f"{'─':>9}  {'─':>9}")
        print(f"  {'True ACC_TA':<38} {'─':>9}  {true_ta.mean():>9.4f}  "
              f"{'─':>9}  {'─':>9}")
        print(f"  {'─'*80}")

        # ── ENPE (TDC) ─────────────────────────────────────────────────
        _print_row('ENPE (ACC_ST est vs true)',
                   'acc_st_tdc', 'acc_st_true', 'err_st_tdc')
        _print_row('ENPE (ACC_TA est vs true)',
                   'acc_ta_tdc', 'acc_ta_true', 'err_ta_tdc')

        # ── ENPE-TA (Mix-Max) ──────────────────────────────────────────
        _print_row('ENPE-TA (ACC_ST est vs true)',
                   'acc_st_mm', 'acc_st_true', 'err_st_mm')
        _print_row('ENPE-TA (ACC_TA est vs true)',
                   'acc_ta_mm', 'acc_ta_true', 'err_ta_mm')

        print(f"  {'─'*80}")

        # ── Baselines ──────────────────────────────────────────────────
        for m in BASELINE_METHODS:
            _print_row(f'{m} (est vs ACC_ST)',
                       f'est_{m}', 'acc_st_true', f'err_{m}')
            _print_row(f'{m} (est vs ACC_TA)',
                       f'est_{m}', 'acc_ta_true', f'err_ta_{m}')

    # ── Figure A: Bar chart по группам ────────────────────────────────────
    method_names_bar = (
        ['ENPE (ST)', 'ENPE (TA)', 'ENPE-TA (ST)', 'ENPE-TA (TA)'] +
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

        fig, ax = plt.subplots(
            figsize=(max(8, len(method_names_bar) * 0.9), 5)
        )
        x_pos = np.arange(len(method_names_bar))
        ax.bar(x_pos, means_bar, yerr=stds_bar,
               color=colors_bar, capsize=4, alpha=0.85,
               edgecolor='black', linewidth=1,
               error_kw=dict(elinewidth=1.2, ecolor='black'))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names_bar, rotation=35, ha='right')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(
            f'BREEDS {BREEDS_NAME} — Mean Error [{group_name}]'
        )
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(
            f'BREEDS/figures/comparison/'
            f'breeds_{BREEDS_NAME}_{group_name}_bar_error.png', dpi=150)
        plt.savefig(
            f'BREEDS/figures/comparison/'
            f'breeds_{BREEDS_NAME}_{group_name}_bar_error.pdf')
        plt.close()

    # ── Figure B: Scatter estimated vs true acc ────────────────────────────
    scatter_methods = []
    scatter_methods.append(dict(
        label='ENPE',    col_est='acc_st_tdc',
        col_true='acc_st_true', panel=0, marker='o',
        color=COLOR_ENGPE, size=35))
    scatter_methods.append(dict(
        label='ENPE-TA', col_est='acc_st_mm',
        col_true='acc_st_true', panel=0, marker='s',
        color=COLOR_ENGPE_TA, size=50))
    scatter_methods.append(dict(
        label='ENPE',    col_est='acc_ta_tdc',
        col_true='acc_ta_true', panel=1, marker='o',
        color=COLOR_ENGPE, size=35))
    scatter_methods.append(dict(
        label='ENPE-TA', col_est='acc_ta_mm',
        col_true='acc_ta_true', panel=1, marker='s',
        color=COLOR_ENGPE_TA, size=50))

    baseline_colors  = plt.cm.tab10(np.linspace(0, 0.9, len(BASELINE_METHODS)))
    baseline_markers = ['D', '^', 'v', 'P', 'X', '*', 'h', '8']
    for idx, m in enumerate(BASELINE_METHODS):
        c  = baseline_colors[idx]
        mk = baseline_markers[idx % len(baseline_markers)]
        scatter_methods.append(dict(
            label=m, col_est=f'est_{m}',
            col_true='acc_st_true', panel=0, marker=mk, color=c, size=25))
        scatter_methods.append(dict(
            label=m, col_est=f'est_{m}',
            col_true='acc_ta_true', panel=1, marker=mk, color=c, size=25))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for panel_idx, ax in enumerate(axes):
        panel_methods = [sm for sm in scatter_methods if sm['panel'] == panel_idx]
        all_vals = []
        for sm in panel_methods:
            all_vals.extend(df[sm['col_true']].values.tolist())
            all_vals.extend(df[sm['col_est']].dropna().values.tolist())
        lo = min(all_vals) - 0.02
        hi = max(all_vals) + 0.02
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y')

        plotted = set()
        for sm in panel_methods:
            lbl = sm['label'] if sm['label'] not in plotted else '_nolegend_'
            ax.scatter(
                df[sm['col_true']].values,
                df[sm['col_est']].values,
                label=lbl, marker=sm['marker'],
                color=sm['color'],
                s=sm.get('size', 25), alpha=0.75
            )
            plotted.add(sm['label'])

        ax.set_xlabel('True Accuracy')
        ax.set_ylabel('Estimated Accuracy')
        ax.set_title(['ACC$_{ST}$', 'ACC$_{TA}$'][panel_idx])
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig(
        f'BREEDS/figures/comparison/'
        f'breeds_{BREEDS_NAME}_scatter_est_vs_true.png', dpi=150)
    plt.savefig(
        f'BREEDS/figures/comparison/'
        f'breeds_{BREEDS_NAME}_scatter_est_vs_true.pdf')
    plt.close()

    # ── Figure C: True ACC_ST vs True ACC_TA ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    lo = min(df['acc_st_true'].min(), df['acc_ta_true'].min()) - 0.02
    hi = max(df['acc_st_true'].max(), df['acc_ta_true'].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y')

    src_mask = df['is_source'].values
    ax.scatter(
        df.loc[src_mask,  'acc_st_true'],
        df.loc[src_mask,  'acc_ta_true'],
        color='steelblue', s=40, alpha=0.8, label='source', zorder=3
    )
    ax.scatter(
        df.loc[~src_mask, 'acc_st_true'],
        df.loc[~src_mask, 'acc_ta_true'],
        color='darkorange', s=40, alpha=0.8, label='target', zorder=3
    )
    ax.set_xlabel('True ACC$_{ST}$')
    ax.set_ylabel('True ACC$_{TA}$')
    ax.legend(); ax.grid(linestyle='--', alpha=0.4)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(
        f'BREEDS/figures/comparison/'
        f'breeds_{BREEDS_NAME}_scatter_accst_vs_accta.png', dpi=150)
    plt.savefig(
        f'BREEDS/figures/comparison/'
        f'breeds_{BREEDS_NAME}_scatter_accst_vs_accta.pdf')
    plt.close()

    # ── Figure D: ACC по уровням severity ─────────────────────────────────
    corr_df = df[~df['is_clean']].copy()
    if len(corr_df) > 0:
        sev_extracted = corr_df['testset'].str.extract(r'sev(\d)')
        if sev_extracted[0].notna().any():
            corr_df['severity'] = sev_extracted[0].astype(int)
            corr_df['split']    = np.where(corr_df['is_source'], 'source', 'target')

            fig, ax = plt.subplots(figsize=(7, 4))
            for split, color in [('source', '#2196F3'), ('target', '#FF9800')]:
                for col, ls, lbl in [
                    ('true_acc',  '-',  f'True Acc ({split})'),
                    ('acc_st_tdc', '--', f'ENPE ({split})'),
                    ('acc_st_mm',  ':',  f'ENPE-TA ({split})'),
                ]:
                    vals     = []
                    sev_vals = sorted(corr_df['severity'].unique())
                    for sev in sev_vals:
                        sub = corr_df[
                            (corr_df['severity'] == sev) &
                            (corr_df['split'] == split)
                        ]
                        vals.append(sub[col].mean() if len(sub) > 0 else np.nan)
                    ax.plot(sev_vals, vals, color=color,
                            linestyle=ls, marker='o', label=lbl)

            ax.set_xlabel('Severity')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'BREEDS {BREEDS_NAME} — Acc vs Severity')
            ax.legend(fontsize=8); ax.grid(linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(
                f'BREEDS/figures/comparison/'
                f'breeds_{BREEDS_NAME}_acc_vs_severity.png', dpi=150)
            plt.savefig(
                f'BREEDS/figures/comparison/'
                f'breeds_{BREEDS_NAME}_acc_vs_severity.pdf')
            plt.close()
            print("✓ acc_vs_severity")

    # ── Summary CSV (mean±std per method) ────────────────────────────────
    summary_rows = []
    for col_est, col_true, col_err, method, kind in [
        ('acc_st_tdc',    'acc_st_true', 'err_st_tdc', 'ENGPE',    'ST'),
        ('acc_ta_tdc',    'acc_ta_true', 'err_ta_tdc', 'ENGPE',    'TA'),
        ('acc_st_mm',     'acc_st_true', 'err_st_mm',  'ENGPE-TA', 'ST'),
        ('acc_ta_mm',     'acc_ta_true', 'err_ta_mm',  'ENGPE-TA', 'TA'),
    ]:
        v_est  = df[col_est].dropna().values
        v_true = df[col_true].dropna().values
        v_err  = df[col_err].dropna().values
        summary_rows.append(dict(
            metric=f'Est ACC ({kind})', method=method, type=kind,
            mean_est=v_est.mean(),   std_est=v_est.std(),
            mean_true=v_true.mean(), std_true=v_true.std(),
            mean_mae=v_err.mean(),   std_mae=v_err.std(),
        ))
    for m in BASELINE_METHODS:
        v_est  = df[f'est_{m}'].dropna().values
        v_true = df['acc_st_true'].dropna().values
        v_err  = df[f'err_{m}'].dropna().values
        summary_rows.append(dict(
            metric='Est ACC (ST)', method=m, type='ST',
            mean_est=v_est.mean(),   std_est=v_est.std(),
            mean_true=v_true.mean(), std_true=v_true.std(),
            mean_mae=v_err.mean(),   std_mae=v_err.std(),
        ))
        v_err_ta = df[f'err_ta_{m}'].dropna().values
        v_true_ta = df['acc_ta_true'].dropna().values
        summary_rows.append(dict(
            metric='Est ACC (TA)', method=m, type='TA',
            mean_est=v_est.mean(),   std_est=v_est.std(),
            mean_true=v_true_ta.mean(), std_true=v_true_ta.std(),
            mean_mae=v_err_ta.mean(), std_mae=v_err_ta.std(),
        ))
    if 'mano_fro_norm' in df.columns:
        v_mano = df['mano_fro_norm'].dropna().values
        v_true = df['acc_st_true'].dropna().values
        mae_m  = np.abs(v_mano - v_true)
        summary_rows.append(dict(
            metric='Est ACC (MANO Frobenius)', method='MANO', type='ST',
            mean_est=v_mano.mean(), std_est=v_mano.std(),
            mean_true=v_true.mean(), std_true=v_true.std(),
            mean_mae=mae_m.mean(),  std_mae=mae_m.std(),
        ))
    pd.DataFrame(summary_rows).to_csv(
        f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_summary.csv',
        index=False, float_format='%.6f')
    print(f"✓ summary.csv сохранён")

    # ── Figure 4: MANO scatter ─────────────────────────────────────────────
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
        plt.savefig(
            f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_mano.png',
            dpi=150)
        plt.savefig(
            f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_scatter_mano.pdf')
        plt.close()
        print("✓ scatter_mano")

    # ── Figure 5: Best testset — Acc/FDR vs rank ──────────────────────────
    if tile_data:
        best_key = max(
            tile_data,
            key=lambda k: tile_data[k]['acc_st_true']
        )
        td = tile_data[best_key]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(td['normalized_rank'], td['Acc_true'],
                label='True Acc',     linewidth=1.5, linestyle='--',
                color='steelblue')
        ax.plot(td['normalized_rank'], td['QVAL_true'],
                label='True FDR',     linewidth=1.5, linestyle='--',
                color='#78909C')
        ax.plot(td['normalized_rank'], td['Acc_est_MM'],
                label='ENGPE-TA',     linewidth=1.8, color=COLOR_ENGPE_TA)
        ax.plot(td['normalized_rank'], td['QVAL_mixmax'],
                label='ENGPE-TA FDR', linewidth=1.5, color=COLOR_ENGPE_TA,
                linestyle=':')
        ax.set_xlabel('Fraction accepted')
        ax.set_ylabel('Value')
        ax.legend(); ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_acc_fdr.png',
            dpi=150)
        plt.savefig(
            f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_acc_fdr.pdf')
        plt.close()
        print(f"✓ best_acc_fdr  (testset={best_key})")

    # ── Figure 6: Best testset — Precision-Recall curve ───────────────────
    if tile_data:
        td = tile_data[best_key]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(td['recall_true'], td['precision_true'],
                label='True PR',     linewidth=1.5, color='steelblue')
        ax.plot(td['recall_est'],  td['precision_est'],
                label='ENGPE-TA PR', linewidth=1.8, color=COLOR_ENGPE_TA)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(); ax.grid(linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_pr_curve.png',
            dpi=150)
        plt.savefig(
            f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_best_pr_curve.pdf')
        plt.close()
        print("✓ best_pr_curve")

    # ── Subsampled curves CSV (for cross-dataset combining) ───────────────
    N_CURVE_PTS = 100
    curve_rows  = []
    for key, td in tile_data.items():
        n   = td['n_samples']
        idx = np.linspace(0, n - 1, N_CURVE_PTS, dtype=int)
        for i in idx:
            curve_rows.append(dict(
                dataset      = 'breeds',
                breeds_name  = BREEDS_NAME,
                testset      = key,
                frac_accepted= float(td['normalized_rank'][i]),
                acc_true     = float(td['Acc_true'][i]),
                acc_est_mm   = float(td['Acc_est_MM'][i]),
                acc_est_tdc  = float(td['Acc_est'][i]),
                fdr_mixmax   = float(td['QVAL_mixmax'][i]),
                fdr_tdc      = float(td['QVAL_TDC'][i]),
                fdr_true     = float(td['QVAL_true'][i]),
            ))
    pd.DataFrame(curve_rows).to_csv(
        f'BREEDS/figures/comparison/breeds_{BREEDS_NAME}_acc_curves.csv',
        index=False, float_format='%.6f')
    print(f"✓ acc_curves.csv сохранён ({len(curve_rows)} строк)")

    # ── Финальный summary ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"✅ BREEDS '{BREEDS_NAME}' pipeline завершён.")
    print(f"   Обработано тестовых наборов : {len(df)}")
    print(f"   Результаты сохранены в      : BREEDS/figures/")
    print(f"{'='*80}")