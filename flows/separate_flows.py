import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flows.conditional_normalizing_flow import ConditionalNormalizingFlow

class SeparateClassFlows(nn.Module):
    def __init__(self, num_classes=10, n_flows=4, feature_dim=128, hidden_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.flows = nn.ModuleList([
            ConditionalNormalizingFlow(n_flows=n_flows, feature_dim=feature_dim, hidden_dim=hidden_dim)
            for _ in range(num_classes)
        ])

    def log_prob(self, scores, features):
        batch_size = scores.shape[0]
        log_probs = torch.zeros(batch_size, self.num_classes).to(scores.device)

        for class_idx in range(self.num_classes):
            class_score = scores[:, class_idx:class_idx+1]
            log_probs[:, class_idx] = self.flows[class_idx].log_prob(class_score, features)

        return log_probs.sum(dim=1)

    def sample(self, features):
        batch_size = features.shape[0]
        device = features.device
        decoy_scores = torch.zeros(batch_size, self.num_classes).to(device)
        for class_idx in range(self.num_classes):
            decoy_scores[:, class_idx] = self.flows[class_idx].sample(features).squeeze()

        return decoy_scores

    def train_separate(self, score_dataset, epochs=20, lr=1e-3, device='cuda'):
        train_loader = DataLoader(score_dataset, batch_size=256, shuffle=True)
        for class_idx in range(self.num_classes):
            print(f"\nTraining flow for class {class_idx}...")

            optimizer = torch.optim.Adam(self.flows[class_idx].parameters(), lr=lr)

            for epoch in range(epochs):
                total_loss = 0
                n_batches = 0

                for cnn_scores, features, target_decoy, labels in train_loader:
                    features = features.to(device)
                    target_decoy = target_decoy.to(device)
                    class_decoy_score = target_decoy[:, class_idx:class_idx+1].to(device)
                    log_prob = self.flows[class_idx].log_prob(class_decoy_score, features.to(device))
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