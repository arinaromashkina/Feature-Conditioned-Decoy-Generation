import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
from bisect import bisect
import os


NUM_CLASSES = 10
GPU_ID = 0
DEVICE = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"Date: {date.today()}")

from data_processing.score_feature_dataset import ScoreFeatureDataset, create_score_feature_dataset
from data_processing.negative_scores_pool import collect_negative_scores
from flows.separate_flows import SeparateClassFlows

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        features = self.get_features(x)
        return self.fc2(F.relu(features))

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        return self.fc1(x)


class BrightnessAdjustment():
    def __init__(self, factor=1.0):
        self.factor = factor

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = img * self.factor
        return transforms.ToPILImage()(img)





def train_cnn_model(train_dataset, epochs=4):
    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    return model


def accuracy_score(labels, preds):
    return np.mean(np.array(labels) == np.array(preds))


def evaluate_model(model, test_dataset):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    with torch.no_grad():
        inputs, labels = next(iter(test_loader))
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        acc = accuracy_score(labels.numpy(), preds.cpu().numpy())
        print(f"Accuracy: {acc:.4f}")
    return acc


def main():
    shift = 0.6

    transform_shift = transforms.Compose([
        BrightnessAdjustment(shift),
        transforms.ToTensor()
    ])

    transform = transforms.Compose([
        BrightnessAdjustment(1.0),
        transforms.ToTensor()
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    mnist_test_shifted = datasets.MNIST('./data', train=False, transform=transform_shift)

    print(f"MNIST train size: {len(mnist_train)}")
    print(f"MNIST test size: {len(mnist_test)}")

    print("\nTraining CNN model...")
    cnn_model = train_cnn_model(mnist_train, epochs=2)
    print("\nEvaluating on test data...")
    evaluate_model(cnn_model, mnist_test)
    negative_scores_pools = collect_negative_scores(cnn_model, mnist_train, True, 10)

    print("\nCreating ScoreFeatureDatasets...")
    train_score_dataset = create_score_feature_dataset(mnist_train, cnn_model, negative_scores_pools)
    test_score_dataset = create_score_feature_dataset(mnist_test, cnn_model, negative_scores_pools)
    test_shifted_score_dataset = create_score_feature_dataset(mnist_test_shifted, cnn_model, negative_scores_pools)

    print(f"\nDataset sizes:")
    print(f"Train dataset: {len(train_score_dataset)} samples")
    print(f"Test dataset: {len(test_score_dataset)} samples")
    print(f"Shifted test dataset: {len(test_shifted_score_dataset)} samples")

    print("\nSample check:")
    sample_cnn_score, sample_features, sample_target_decoy, sample_label = train_score_dataset[0]
    print(f"Label: {sample_label}")
    print(f"CNN score for class {sample_label}: {sample_cnn_score[sample_label]:.4f}")
    print(f"Target decoy for class {sample_label}: {sample_target_decoy[sample_label]:.4f}")
    print(f"Are they different? {abs(sample_cnn_score[sample_label] - sample_target_decoy[sample_label]) > 1e-6}")

    return cnn_model, train_score_dataset, test_score_dataset, test_shifted_score_dataset


cnn_model, train_ds, test_ds, test_shifted_ds = main()

print("Training separate flows for each class...")
separate_flows = SeparateClassFlows(num_classes=NUM_CLASSES, n_flows=4, feature_dim=128, hidden_dim=64).to(DEVICE)
separate_flows = separate_flows.train_separate(train_ds, epochs=3, lr=1e-3, device=DEVICE)
test_cnn_scores, separate_decoy_scores, test_labels = separate_flows.generate_decoys(test_ds, device='cuda')
test_shifted_cnn_scores, separate_shifted_decoy_scores, test_shifted_labels = separate_flows.generate_decoys(test_shifted_ds, device='cuda')


from utils.visualize_distributions import plot_class_score_distributions, plot_dataset_comparison

plot_class_score_distributions(
        model_scores=test_cnn_scores,
        decoy_scores=separate_decoy_scores,
        labels=test_labels,
        method_name="Separate flows for test data",
        num_classes=10,
        save_dir="distribution_plots",
        save_prefix="mnist_multiclass_test",
        show_plots=True
    )

plot_class_score_distributions(
        model_scores=test_shifted_cnn_scores,
        decoy_scores=separate_shifted_decoy_scores,
        labels=test_shifted_labels,
        method_name="Separate flows for shifted data",
        num_classes=10,
        save_dir="distribution_plots",
        save_prefix="mnist_multiclass_shifted",
        show_plots=True
    )


plot_dataset_comparison(
    original_cnn_scores=test_cnn_scores,
    shifted_cnn_scores=test_shifted_cnn_scores,
    original_decoy_scores=separate_decoy_scores,
    shifted_decoy_scores=separate_shifted_decoy_scores,
    num_classes=NUM_CLASSES,
    save_dir="comparison_plots",
    save_prefix="mnist_multiclass",
    show_plots=True
)