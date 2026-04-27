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
    Robust feature normalization for OOD test data.

    Algorithm:
      1. Accumulates median and IQR on train data (robust to outliers).
      2. Normalizes: x_norm = tanh(clip((x - median) / IQR, -clip_val, clip_val))
      3. tanh compresses everything to (-1, 1) without information loss.

    Running statistics are stored in state_dict as buffers.
    """
    def __init__(self, feature_dim, clip_val=5.0, momentum=0.01, eps=1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.clip_val    = clip_val
        self.momentum    = momentum
        self.eps         = eps

        self.register_buffer('running_median', torch.zeros(feature_dim))
        self.register_buffer('running_iqr',    torch.ones(feature_dim))
        self.register_buffer('initialized',    torch.tensor(False))

    @torch.no_grad()
    def _update_stats(self, x):
        """Update running median and IQR via exponential moving average."""
        batch_median = x.median(dim=0).values
        q75 = torch.quantile(x, 0.75, dim=0)
        q25 = torch.quantile(x, 0.25, dim=0)
        batch_iqr = (q75 - q25).clamp(min=self.eps)

        if not self.initialized:
            self.running_median.copy_(batch_median)
            self.running_iqr.copy_(batch_iqr)
            self.initialized.fill_(True)
        else:
            self.running_median.mul_(1 - self.momentum).add_(batch_median * self.momentum)
            self.running_iqr.mul_(1 - self.momentum).add_(batch_iqr * self.momentum)

    def forward(self, x):
        if self.training:
            self._update_stats(x)

        if not self.initialized:
            return torch.tanh(x * 0.01)

        x_norm = (x - self.running_median) / (self.running_iqr + self.eps)
        x_norm = x_norm.clamp(-self.clip_val, self.clip_val)
        x_norm = torch.tanh(x_norm / self.clip_val)
        return x_norm


class ActNorm(nn.Module):
    """
    Activation normalization — data-driven per-channel scale+bias.
    Initialized from the first forward batch so output has mean=0, std=1.
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
    Affine coupling layer operating on the full score vector,
    conditioned on encoded CNN features.

    Split  : x1 = x[:d],  x2 = x[d:]
    Forward: z2 = x2 * exp(s(x1, feat)) + t(x1, feat)
    Reverse: x2 = (z2 - t) * exp(-s)

    Scale head uses Tanh so s ∈ (-1, 1), preventing exp blow-up.
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
        self.scale_head     = nn.Sequential(nn.Linear(hidden_dim // 2, self.d_out), nn.Tanh())
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
    Conditional normalizing flow that models P(null_score_vector | features).

    Architecture:
      - RobustFeatureNormalizer: median/IQR + clamp + tanh → always ∈ (-1, 1)
      - Feature encoder: feature_dim → encoder_dim (2-layer MLP with LayerNorm + GELU)
      - n_flows CouplingLayers with alternating first-half/second-half masks
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

        self.feature_norm = RobustFeatureNormalizer(
            feature_dim, clip_val=clip_val, momentum=0.01)

        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for i in range(n_flows):
            mask = 'first_half' if i % 2 == 0 else 'second_half'
            self.layers.append(CouplingLayer(score_dim, encoder_dim, hidden_dim, mask))
            if i < n_flows - 1:
                self.layers.append(ActNorm(score_dim))

    def encode(self, features):
        return self.feature_encoder(self.feature_norm(features))

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
        log_pz     = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.score_dim * self._log_2pi
        return log_pz + log_det

    def sample(self, features):
        """Sample one null score vector per row of features."""
        z = torch.randn(features.size(0), self.score_dim, device=features.device)
        scores, _ = self.forward(z, features, reverse=True)
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper with training and inference helpers
# ─────────────────────────────────────────────────────────────────────────────

class ScoreShiftFlowWrapper(nn.Module):
    """
    Wraps ScoreShiftFlow with train_flow() and generate_decoys() helpers.
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

    def train_flow(self,
                   score_dataset,
                   epochs     = 30,
                   lr         = 3e-4,
                   batch_size = 256,
                   device     = 'cuda',
                   patience   = 5,
                   grad_clip  = 1.0):
        self.flow.to(device)
        self.flow.train()

        loader    = DataLoader(score_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
        optimizer = torch.optim.AdamW(self.flow.parameters(), lr=lr, weight_decay=1e-4)
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

                loss = -self.flow.log_prob(target_decoy, features).mean()

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
                      f"loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

            if avg_loss < best_loss - 1e-4:
                best_loss  = avg_loss
                no_improve = 0
                best_state = {k: v.clone() for k, v in self.flow.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    self.flow.load_state_dict(best_state)
                    break

        if best_state is not None:
            self.flow.load_state_dict(best_state)
        print(f"  Training complete. Best loss: {best_loss:.4f}")
        return self

    def generate_decoys(self, score_dataset, device='cuda', n_samples=1):
        """
        Returns:
          model_scores : (N, C)  original classifier logits
          decoy_scores : (N, C)  sampled null score vectors
          labels       : (N,)    true labels
        """
        self.flow.to(device)
        self.flow.eval()

        cnn_list, decoy_list, lbl_list = [], [], []
        loader = DataLoader(score_dataset, batch_size=256, shuffle=False, num_workers=0)

        with torch.no_grad():
            for cnn_scores, features, target_decoy, labels in loader:
                features = features.to(device)

                if n_samples == 1:
                    decoy = self.flow.sample(features)
                else:
                    samples = torch.stack(
                        [self.flow.sample(features) for _ in range(n_samples)], dim=0)
                    decoy = samples.mean(dim=0)

                cnn_list.append(cnn_scores.cpu().numpy())
                decoy_list.append(decoy.cpu().numpy())
                lbl_list.append(labels.cpu().numpy())

        return (np.concatenate(cnn_list,   axis=0),
                np.concatenate(decoy_list, axis=0),
                np.concatenate(lbl_list,   axis=0))

    def diagnose_feature_ood(self, score_dataset, device='cuda', tile_name=''):
        """
        Reports how far test features are from the training distribution
        in IQR units. Call after train_flow() when statistics are frozen.
        """
        self.flow.to(device)
        self.flow.eval()

        norm = self.flow.feature_norm
        if not norm.initialized:
            print("  Normalizer not initialized — train first")
            return

        loader = DataLoader(score_dataset, batch_size=256, shuffle=False)
        all_features = []
        with torch.no_grad():
            for _, features, _, _ in loader:
                all_features.append(features.to(device))
        features = torch.cat(all_features, dim=0)

        deviation = ((features - norm.running_median) / (norm.running_iqr + 1e-6)).abs()

        print(f"  OOD diagnostics [{tile_name}]:")
        print(f"    median deviation (IQR units) : {deviation.median():.2f}")
        print(f"    95th pct deviation            : {deviation.quantile(0.95):.2f}")
        print(f"    max deviation                 : {deviation.max():.2f}")
        print(f"    % features with deviation >  5: {(deviation > 5).float().mean() * 100:.1f}%")
        print(f"    % features with deviation > 10: {(deviation > 10).float().mean() * 100:.1f}%")
        print(f"    % features with deviation >100: {(deviation > 100).float().mean() * 100:.1f}%")
