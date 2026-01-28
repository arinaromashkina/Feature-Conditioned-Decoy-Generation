import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from conditional_coupling_layer import ConditionalCouplingLayer

class ConditionalNormalizingFlow(nn.Module):    
    def __init__(self, n_flows=8, feature_dim=128, hidden_dim=64):
        super().__init__()
        self.input_dim = 2
        self.n_flows = n_flows
        
        self.flows = nn.ModuleList([
            ConditionalCouplingLayer(self.input_dim, feature_dim, hidden_dim)
            for _ in range(n_flows)
        ])
    
    def forward(self, scores, features, reverse=False):
        if scores.shape[1] == 1:
            x = torch.cat([scores, torch.zeros_like(scores)], dim=1)
        else:
            x = scores
        log_det_total = 0
        if not reverse:
            for i, flow in enumerate(self.flows):
                x, log_det = flow(x, features, reverse=False)
                log_det_total += log_det
                if i < len(self.flows) - 1:
                    x = x.flip(dims=[1])
            return x, log_det_total
        else:
            for i, flow in enumerate(reversed(self.flows)):
                if i > 0:
                    x = x.flip(dims=[1])
                x, log_det = flow(x, features, reverse=True)
                log_det_total += log_det
            return x[:, 0:1], log_det_total
    
    def log_prob(self, scores, features):
        z, log_det = self.forward(scores, features, reverse=False)
        log_pz = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * self.input_dim * torch.log(torch.tensor(2 * 3.14159))
        return log_pz + log_det
    
    def sample(self, features, n_samples=1):
        batch_size = features.shape[0]
        device = features.device
        z = torch.randn(batch_size, self.input_dim).to(device)
        with torch.no_grad():
            scores, _ = self.forward(z, features, reverse=True)
        return scores.squeeze()

    def train_flow(self, score_dataset, epochs=20, lr=1e-3, device='cuda'):
        train_loader = DataLoader(score_dataset, batch_size=256, shuffle=True)
        
        optimizer = torch.optim.Adam(self.flows[class_idx].parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0

            for cnn_scores, features, target_decoy, labels in train_loader:
                features = features.to(device)
                target_decoy = target_decoy.to(device)
                log_prob = self.flows.log_prob(target_decoy, features.to(device))
                loss = -log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 1 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"  Class {class_idx}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return self

    def generate_decoys(self, score_dataset, device='cuda'):
        test_cnn_scores = []
        test_labels = []
        separate_decoy_scores = []
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        with torch.no_grad():
            for cnn_scores, features, target_decoy, labels in test_loader:
                features = features.to(DEVICE)
                decoy_scores = self.flows.sample(features)
                separate_decoy_scores.append(decoy_scores.cpu().numpy())
                test_cnn_scores.append(cnn_scores.cpu().numpy())
                test_labels.append(labels.cpu().numpy())

        separate_decoy_scores = np.concatenate(separate_decoy_scores, axis=0)
        test_cnn_scores = np.concatenate(test_cnn_scores, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        return test_cnn_scores, separate_decoy_scores, test_labels