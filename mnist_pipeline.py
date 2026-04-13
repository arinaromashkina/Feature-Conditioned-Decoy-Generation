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

from data_processing.score_feature_dataset import *
from data_processing.negative_scores_pool import collect_negative_scores
from flows.shift_flow import ScoreShiftFlowWrapper
from utils.other_methods import *
from utils.visualize_distributions import *
from fdr.fdr_control import *
from fdr.plot_fdr import *

NUM_CLASSES = 10
GPU_ID = 0
DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"Date: {date.today()}") 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.linear1 = nn.Linear(64 * 7 * 7, 128)
        self.linear2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        features = self.get_features(x)
        return self.linear2(features)  # features уже с ReLU

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.linear1(x)
        return F.relu(x)  # возвращаем признаки с ReLU



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

    negative_scores_pools = collect_negative_scores(cnn_model, mnist_train, num_classes=10, device=DEVICE)

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
score_shift_flow = ScoreShiftFlowWrapper(
    num_classes = 10,
    n_flows     = 12,
    feature_dim = 128,
    hidden_dim  = 256,
    encoder_dim = 128,
).to(DEVICE)


score_shift_flow.train_flow(
        train_ds,
        epochs     = 30,
        lr         = 3e-4,
        batch_size = 256,
        device     = DEVICE,
        patience   = 5,
        grad_clip  = 1.0,
    )

print(f"✓ Flow saved to")


train_scores, train_decoy_scores, train_labels = score_shift_flow.generate_decoys(train_ds, device=DEVICE)
test_cnn_scores, separate_decoy_scores, test_labels = score_shift_flow.generate_decoys(test_ds, device='cuda')
test_shifted_cnn_scores, separate_shifted_decoy_scores, test_shifted_labels = score_shift_flow.generate_decoys(test_shifted_ds, device='cuda')

for i in range(NUM_CLASSES):
    plot_score_distribution_with_decoys(train_scores[:, i], train_labels, 
                                        test_shifted_cnn_scores[:, i], test_shifted_labels, 
                                        separate_shifted_decoy_scores[:, i],
                                        filename=f"score_distribution_decoys_mini_long_{i}", 
                                        title=f"Score Distribution with Decoys for class {i}",
                                        xlim=(-4, 6), show_kde=True, class_id=i)

print("\n" + "="*80)
print("RUNNING FDR CONTROL  [Mix-Max]")
print("="*80)

all_fdr_results = {}   # corruption → DataFrame with q_values_mixmax

test_shifted_cnn_scores, separate_shifted_decoy_scores, test_shifted_labels 
df_mixmax_clean = control_fdr_mixmax(test_shifted_cnn_scores, test_shifted_labels, separate_shifted_decoy_scores)
all_fdr_results['mnist'] = df_mixmax_clean

plot_accuracy_estimation(
        df_mixmax_clean,
        q_value_column='q_values_mixmax',
        method_name='Mix-Max',
        corruption_name='mnist',
        save_dir='figures/accuracy'
    )

print("\n" + "="*80)
print("RUNNING EVALUATIONS")
print("="*80)

# Source logits (clean train) — computed once for baselines
source_logits, source_decoy_scores, source_labels = score_shift_flow.generate_decoys(
    train_ds, device=DEVICE)
source_logits_t  = torch.tensor(source_logits).to(DEVICE)
source_labels_t  = torch.tensor(source_labels).to(DEVICE)
temp = calibration_temp(source_logits_t, source_labels_t)

all_results = []
corruption_name = 'mnist_shifted'

model_scores, decoy_scores, labels = score_shift_flow.generate_decoys(
    test_shifted_ds, device=DEVICE)

target_logits_t = torch.tensor(model_scores).to(DEVICE)
target_labels_t = torch.tensor(labels).to(DEVICE)

# True accuracy on FULL dataset (for baselines comparison)
true_acc_full = (target_logits_t.argmax(1) == target_labels_t).float().mean().item()

# ── Mix-Max ───────────────────────────────────────────────────────────
df_mixmax = control_fdr_mixmax(model_scores, labels, decoy_scores)
pi0       = 0.0
df_curve  = compute_method_estimation_curve(df_mixmax, 'q_values_mixmax', pi0)

if len(df_curve) > 0:
    best_idx      = df_curve['Accuracy_est'].idxmax()
    mixmax_est    = df_curve.loc[best_idx, 'Accuracy_est']
    # True accuracy at the SAME threshold (honest comparison)
    mixmax_true_at_thresh = df_curve.loc[best_idx, 'Accuracy_true_at_threshold']
    mixmax_error  = df_curve.loc[best_idx, 'error_at_threshold']
    best_q        = df_curve.loc[best_idx, 'q_value_method']
    n_accepted    = int(df_curve.loc[best_idx, 'n_discoveries'])
else:
    mixmax_est            = 0.0
    mixmax_true_at_thresh = 0.0
    mixmax_error          = 0.0
    best_q                = np.nan
    n_accepted            = 0

all_results.append({
        'corruption'          : corruption_name,
        'method'              : 'Mix-Max',
        'estimated_accuracy'  : mixmax_est,
        # For Mix-Max: true_accuracy = true acc at best threshold
        'true_accuracy'       : mixmax_true_at_thresh,
        # Also store full-dataset true acc for reference
        'true_accuracy_full'  : true_acc_full,
        'error'               : mixmax_error,
        'best_q'              : best_q,
        'n_accepted'          : n_accepted,
        'n_total'             : len(labels),
        'frac_accepted'       : n_accepted / len(labels),
    })
print(f"  Mix-Max: est={mixmax_est:.3f}  "
          f"true@thresh={mixmax_true_at_thresh:.3f}  "
          f"true_full={true_acc_full:.3f}  "
          f"error={mixmax_error:.3f}  "
          f"q*={best_q:.3f}  "
          f"accepted={n_accepted}/{len(labels)}")

# ── Baseline methods (compare against full-dataset true acc) ──────────
scaled_source = source_logits_t / temp
scaled_target = target_logits_t / temp

for method_name, method_func in BASELINE_METHODS.items():
    try:
        estimate = method_func(scaled_source, source_labels_t, scaled_target)
        error    = abs(estimate - true_acc_full)
        all_results.append({
                'corruption'         : corruption_name,
                'method'             : method_name,
                'estimated_accuracy' : estimate,
                'true_accuracy'      : true_acc_full,
                'true_accuracy_full' : true_acc_full,
                'error'              : error,
                'best_q'             : np.nan,
                'n_accepted'         : len(labels),
                'n_total'            : len(labels),
                'frac_accepted'      : 1.0,
        })
        print(f"  {method_name}: est={estimate:.3f}  "
                  f"true={true_acc_full:.3f}  error={error:.3f}")
    except Exception as e:
        print(f"  {method_name}: FAILED ({e})")

results_df = pd.DataFrame(all_results)
print("\n✓ All evaluations complete")


print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# ── Note about comparison ─────────────────────────────────────────────────
print("""
NOTE on comparison:
  Mix-Max  → error = |est_acc - true_acc_at_best_threshold|
             (true_acc computed on ACCEPTED samples only)
  Baselines → error = |est_acc - true_acc_full_dataset|
             (true_acc computed on ALL samples)
  These measure different things — Mix-Max selects a subset,
  baselines estimate accuracy on the whole dataset.
""")

stats_data = []
for method in results_df['method'].unique():
    df_m   = results_df[results_df['method'] == method]
    errors = df_m['estimated_accuracy'] - df_m['true_accuracy']
    abs_e  = np.abs(errors)

    is_mixmax = method == 'Mix-Max'
    stats_data.append({
        'Method'       : method,
        'MAE'          : abs_e.mean(),
        'RMSE'         : np.sqrt((errors**2).mean()),
        'Mean Bias'    : errors.mean(),
        'Std Dev'      : errors.std(),
        'Max Error'    : abs_e.max(),
        # For Mix-Max: also report avg fraction accepted
        'Avg Accept %' : f"{df_m['frac_accepted'].mean()*100:.1f}%" \
                         if is_mixmax else "100.0%",
        'True Acc Type': 'at threshold' if is_mixmax else 'full dataset',
    })

stats_df = pd.DataFrame(stats_data).sort_values('MAE')

print("\nPerformance Statistics:")
print(stats_df.to_string(index=False))

# ── Separate summary for Mix-Max ──────────────────────────────────────────
mm_df = results_df[results_df['method'] == 'Mix-Max']
print(f"\nMix-Max detailed:")
print(f"  Mean est_acc       : {mm_df['estimated_accuracy'].mean():.3f}")
print(f"  Mean true@thresh   : {mm_df['true_accuracy'].mean():.3f}")
print(f"  Mean true_full     : {mm_df['true_accuracy_full'].mean():.3f}")
print(f"  MAE vs thresh      : {mm_df['error'].mean():.4f}")
print(f"  MAE vs full        : {(mm_df['estimated_accuracy'] - mm_df['true_accuracy_full']).abs().mean():.4f}")
print(f"  Avg accepted       : {mm_df['frac_accepted'].mean()*100:.1f}%")
print(f"  Avg best q*        : {mm_df['best_q'].mean():.3f}")

stats_df.to_csv('figures/comparison/statistics.csv', index=False)
results_df.to_csv('figures/comparison/all_results.csv', index=False)
print("\n✓ Statistics saved to figures/comparison/statistics.csv")
print("✓ Results saved to figures/comparison/all_results.csv")