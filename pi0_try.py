import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from data_processing.score_feature_dataset import *
from data_processing.negative_scores_pool import collect_negative_scores
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
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 5
FEATURE_DIM = 768
SAVE_DIR    = "./data/amazon_data/score_feature_datasets"
FLOWS_PATH  = "./amazon_score_shift_flow.pth"
MIN_PIXELS  = 50

os.makedirs('amazon/figures/accuracy',    exist_ok=True)
os.makedirs('amazon/figures/comparison',  exist_ok=True)
os.makedirs('amazon/figures/diagnostics', exist_ok=True)

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
SPLIT_MARKERS = {'id_val': 'o', 'ood_val': 's', 'test': '*'}
SPLIT_COLORS  = {
    'id_val' : '#1976D2',
    'ood_val': '#E53935',
    'test'   : '#388E3C',
}
SPLIT_LABELS  = {
    'id_val' : 'Amazon ID Val',
    'ood_val': 'Amazon OOD Val',
    'test'   : 'Amazon Test',
}

print(f"Device    : {DEVICE}")
print(f"Baselines : {list(BASELINE_METHODS.keys())}")


# ============================================================================
# DATASET
# ============================================================================

class ScoreFeatureDataset(Dataset):
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


def load_score_feature_datasets(save_dir=SAVE_DIR):
    splits = {}
    for split in ("train", "id_val", "ood_val", "test"):
        path = f"{save_dir}/{split}.pt"
        raw  = torch.load(path, map_location="cpu")
        splits[split] = ScoreFeatureDataset(
            cnn_scores          = raw["cnn_scores"],
            features            = raw["features"],
            target_decoy_scores = raw["target_decoy_scores"],
            labels              = raw["labels"],
        )
        print(f"  {split:8s}: {len(splits[split]):,} samples  "
              f"feature_dim={splits[split].features.shape[1]}")
    return splits


# ============================================================================
# HELPERS
# ============================================================================

def get_scores_from_ds(ds, device=DEVICE):
    """Вытащить CNN scores и labels из датасета."""
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    cnn_list, lbl_list = [], []
    with torch.no_grad():
        for cnn_scores, _, _, labels in loader:
            cnn_list.append(cnn_scores.cpu().numpy())
            lbl_list.append(labels.cpu().numpy())
    return np.concatenate(cnn_list), np.concatenate(lbl_list)


def true_accuracy_full(model_scores, labels_np):
    """
    Обычная accuracy по всем сэмплам (pred == label).
    Это единственный ground truth с которым сравниваем ВСЕХ.
    """
    pred = model_scores.argmax(axis=1)
    return float((pred == labels_np).mean())


def true_accuracy_at_threshold(model_scores, labels_np, conf_threshold):
    """
    Реальная accuracy только среди сэмплов
    где max_score >= conf_threshold.
    Знаменатель = число принятых сэмплов.
    """
    pred       = model_scores.argmax(axis=1)
    max_scores = model_scores[np.arange(len(model_scores)), pred]
    mask       = max_scores >= conf_threshold
    n_accepted = mask.sum()
    if n_accepted < MIN_PIXELS:
        return np.nan, 0
    return float((pred[mask] == labels_np[mask]).mean()), int(n_accepted)


def find_best_mixmax_threshold(df_curve):
    """
    Найти q* где Accuracy_est максимальна.
    Возвращает словарь:
        mixmax_est       — наша оценка accuracy при q*
        mixmax_true_at_q — реальная accuracy среди принятых при q*
        mixmax_error     — |est - true_at_q|
        mixmax_best_q    — сам порог q*
        mixmax_n_accepted— сколько сэмплов принято
    """
    if len(df_curve) == 0:
        return dict(
            mixmax_est        = np.nan,
            mixmax_true_at_q  = np.nan,
            mixmax_error      = np.nan,
            mixmax_best_q     = np.nan,
            mixmax_n_accepted = 0,
        )

    best_idx = df_curve['Accuracy_est'].idxmax()
    row      = df_curve.loc[best_idx]
    return dict(
        mixmax_est        = float(row['Accuracy_est']),
        mixmax_true_at_q  = float(row['Accuracy_true_at_threshold']),
        mixmax_error      = float(row['error_at_threshold']),
        mixmax_best_q     = float(row['q_value_method']),
        mixmax_n_accepted = int(row['n_discoveries']),
    )


def quick_diagnostics(model_scores, decoy_scores, labels_np, name=""):
    """Печатает target-wins диагностику."""
    n             = len(model_scores)
    pred_classes  = model_scores.argmax(axis=1)
    target_scores = model_scores[np.arange(n), pred_classes]
    null_scores   = decoy_scores[np.arange(n), pred_classes]
    correct       = pred_classes == labels_np

    tw_c  = (target_scores[correct]  > null_scores[correct]).mean()
    tw_i  = (target_scores[~correct] > null_scores[~correct]).mean() \
            if (~correct).sum() > 0 else np.nan
    gap_c = (target_scores - null_scores)[correct].mean()
    gap_i = (target_scores - null_scores)[~correct].mean() \
            if (~correct).sum() > 0 else np.nan

    print(f"  [{name}] n={n:,}  true_acc={correct.mean():.3f}")
    print(f"    target wins correct  : {tw_c:.3f}")
    print(f"    target wins incorrect: {tw_i:.3f}")
    print(f"    gap correct          : {gap_c:.3f}")
    print(f"    gap incorrect        : {gap_i:.3f}")


# ============================================================================
# ДАННЫЕ
# ============================================================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

datasets_dict = load_score_feature_datasets()
train_ds   = datasets_dict["train"]
id_val_ds  = datasets_dict["id_val"]
ood_val_ds = datasets_dict["ood_val"]
test_ds    = datasets_dict["test"]

print("\nSanity check:")
for split, ds in datasets_dict.items():
    cnn_s, feat, decoy, lbl = ds[0]
    c = lbl.item()
    print(f"  {split:8s}: label={c}  "
          f"cnn[{c}]={cnn_s[c]:.4f}  "
          f"decoy[{c}]={decoy[c]:.4f}  "
          f"cnn>decoy: {cnn_s[c] > decoy[c]}")

train_scores_np, train_labels_np = get_scores_from_ds(train_ds)
print(f"\nTrain class distribution:")
for c in range(NUM_CLASSES):
    n_c = (train_labels_np == c).sum()
    print(f"  Class {c}: {n_c:,} ({n_c/len(train_labels_np)*100:.1f}%)")


# ============================================================================
# FLOW
# ============================================================================

print("\n" + "="*70)
print("SCORE SHIFT FLOW")
print("="*70)

flow = ScoreShiftFlowWrapper(
    num_classes = NUM_CLASSES,
    n_flows     = 12,
    feature_dim = FEATURE_DIM,
    hidden_dim  = 256,
    encoder_dim = 128,
).to(DEVICE)

if os.path.exists(FLOWS_PATH):
    flow.load_state_dict(torch.load(FLOWS_PATH, map_location=DEVICE))
    print(f"✓ Flow loaded from {FLOWS_PATH}")
else:
    print("Training ScoreShiftFlow...")
    flow.train_flow(
        train_ds, epochs=30, lr=3e-4, batch_size=256,
        device=DEVICE, patience=5, grad_clip=1.0,
    )
    torch.save(flow.state_dict(), FLOWS_PATH)
    print(f"✓ Flow saved to {FLOWS_PATH}")


# ============================================================================
# ГЕНЕРАЦИЯ DECOYS
# ============================================================================

print("\n" + "="*70)
print("GENERATING DECOYS")
print("="*70)

decoy_results = {}   # split → (model_scores, decoy_scores, labels_np)
train_scores_flow, train_labels_flow = None, None

for split, ds in [("train",   train_ds),
                   ("id_val",  id_val_ds),
                   ("ood_val", ood_val_ds),
                   ("test",    test_ds)]:
    print(f"\n{split}:")
    ms, ds_decoy, ls = flow.generate_decoys(ds, device=DEVICE)

    if split == 'train':
        # сохраняем train scores для plot_score_distribution_with_decoys
        train_scores_flow  = ms
        train_labels_flow  = ls

    decoy_results[split] = (ms, ds_decoy, ls)
    quick_diagnostics(ms, ds_decoy, ls, name=split)


# ============================================================================
# SOURCE ДЛЯ BASELINES
# ============================================================================

print("\n" + "="*70)
print("PREPARING SOURCE FOR BASELINES")
print("="*70)

source_ms, _, source_ls = decoy_results["id_val"]
source_logits_t = torch.tensor(source_ms).to(DEVICE)
source_labels_t = torch.tensor(source_ls).to(DEVICE)
temp            = calibration_temp(source_logits_t, source_labels_t)
scaled_source   = source_logits_t / temp
print(f"✓ Source: id_val ({len(source_ls):,} samples), temp={temp:.4f}")


# ============================================================================
# SCORE DISTRIBUTIONS (для ood_val — самый интересный сдвиг)
# ============================================================================

print("\n" + "="*70)
print("SCORE DISTRIBUTIONS")
print("="*70)

ms_ood, ds_decoy_ood, ls_ood = decoy_results["ood_val"]

for i in range(NUM_CLASSES):
    plot_score_distribution_with_decoys(
        train_scores_flow[:, i],   # train scores класса i
        train_labels_flow,
        ms_ood[:, i],              # ood_val scores класса i
        ls_ood,
        ds_decoy_ood[:, i],        # decoy scores класса i
        filename = f"amazon/figures/accuracy/scores_dist_{i}",
        title    = f"Score Distribution with Decoys — Class {i}",
        xlim     = (-1, 1.5),
        show_kde = True,
        class_id = i,
    )
print(f"✓ Score distributions сохранены для {NUM_CLASSES} классов")


# ============================================================================
# ACCURACY ESTIMATION PLOTS (per split)
# ============================================================================

print("\n" + "="*70)
print("ACCURACY ESTIMATION PLOTS")
print("="*70)

fdr_dfs = {}   # split → df_mm

for split in ("id_val", "ood_val", "test"):
    print(f"\n{split}:")
    ms, ds_d, ls = decoy_results[split]
    df_mm        = control_fdr_mixmax(ms, ls, ds_d)
    fdr_dfs[split] = df_mm

    plot_accuracy_estimation(
        df_mm,
        q_value_column  = 'q_values_mixmax',
        method_name     = 'Mix-Max',
        corruption_name = f'amazon_{split}',
        save_dir        = 'amazon/figures/accuracy',
    )


# ============================================================================
# ПОЛНАЯ ОЦЕНКА
# ============================================================================

print("\n" + "="*70)
print("FULL EVALUATION")
print("="*70)

results = []   # один dict на split

for split in ("id_val", "ood_val", "test"):
    print(f"\n── {SPLIT_LABELS[split]} ──")
    ms, ds_d, ls = decoy_results[split]
    n            = len(ls)

    # ── True accuracy — единственный ground truth для сравнения ВСЕХ ─────
    acc_full = true_accuracy_full(ms, ls)
    print(f"  true_acc_full = {acc_full:.4f}")

    # ── Mix-Max ───────────────────────────────────────────────────────────
    df_mm    = fdr_dfs[split]
    df_curve = compute_method_estimation_curve(df_mm, 'q_values_mixmax', pi0=0.0)
    mm       = find_best_mixmax_threshold(df_curve)

    print(f"  Mix-Max: est={mm['mixmax_est']:.4f}  "
          f"true@q*={mm['mixmax_true_at_q']:.4f}  "
          f"error={mm['mixmax_error']:.4f}  "
          f"q*={mm['mixmax_best_q']:.4f}  "
          f"accepted={mm['mixmax_n_accepted']}/{n}")

    # ── Baselines ─────────────────────────────────────────────────────────
    target_logits_t = torch.tensor(ms).to(DEVICE)
    scaled_target   = target_logits_t / temp

    baseline_est = {}
    for mname, mfunc in BASELINE_METHODS.items():
        try:
            baseline_est[mname] = float(
                mfunc(scaled_source, source_labels_t, scaled_target))
            print(f"  {mname}: est={baseline_est[mname]:.4f}  "
                  f"error={abs(baseline_est[mname] - acc_full):.4f}")
        except Exception as e:
            print(f"  {mname}: FAILED ({e})")
            baseline_est[mname] = np.nan

    results.append({
        'split'             : split,
        'split_label'       : SPLIT_LABELS[split],
        'n'                 : n,
        # ── Ground truth (один для всех) ──────────────────────────────
        'true_acc_full'     : acc_full,
        # ── Mix-Max ───────────────────────────────────────────────────
        **mm,               # mixmax_est, mixmax_true_at_q, mixmax_error,
                            # mixmax_best_q, mixmax_n_accepted
        'mixmax_frac_accepted': mm['mixmax_n_accepted'] / max(n, 1),
        # ── Baselines ─────────────────────────────────────────────────
        **baseline_est,
    })

results_df = pd.DataFrame(results)
baseline_methods = list(BASELINE_METHODS.keys())


# ============================================================================
# ОШИБКИ
# ============================================================================

print("\n" + "="*70)
print("ОШИБКИ")
print("="*70)

# Mix-Max vs true_acc_full  (честное сравнение с конкурентами)
results_df['mixmax_err_vs_full'] = np.abs(
    results_df['mixmax_est'] - results_df['true_acc_full'])

# Baselines vs true_acc_full
for m in baseline_methods:
    results_df[f'err_{m}'] = np.abs(
        results_df[m] - results_df['true_acc_full'])

# ── MAE таблица ───────────────────────────────────────────────────────────
mae_table = {}
mae_table['Mix-Max'] = (
    results_df['mixmax_err_vs_full'].mean(),
    results_df['mixmax_err_vs_full'].std(),
)
for m in baseline_methods:
    col = f'err_{m}'
    if col in results_df:
        mae_table[m] = (
            results_df[col].mean(),
            results_df[col].std(),
        )

# Mix-Max vs true_at_q* (threshold-matched — отдельная метрика)
mae_at_q = results_df['mixmax_error'].mean()
std_at_q = results_df['mixmax_error'].std()

print("\nMAE vs true_acc_full (одинаковый ground truth для всех):")
print("-"*55)
for method, (mae, std) in sorted(mae_table.items(), key=lambda x: x[1][0]):
    marker = ' ← наш' if method == 'Mix-Max' else ''
    print(f"  {method:<12}: {mae:.4f} ± {std:.4f}{marker}")

print(f"\nMix-Max MAE vs true_acc_at_q* (threshold-matched):")
print(f"  {'Mix-Max':<12}: {mae_at_q:.4f} ± {std_at_q:.4f}")

print("\nPer-split сводка:")
print(f"{'Split':<16} {'true_full':>9} {'mm_est':>7} "
      f"{'mm_true@q':>10} {'err@q':>7} {'q*':>5} {'accept%':>8}")
print("-"*70)
for _, row in results_df.iterrows():
    print(
        f"  {row['split_label']:<14} "
        f"{row['true_acc_full']:9.4f} "
        f"{row['mixmax_est']:7.4f} "
        f"{row['mixmax_true_at_q']:10.4f} "
        f"{row['mixmax_error']:7.4f} "
        f"{row['mixmax_best_q']:5.3f} "
        f"{row['mixmax_frac_accepted']*100:7.1f}%"
    )


# ============================================================================
# PLOT 1: Scatter — estimated vs true_acc_full (все методы)
# ============================================================================

print("\n" + "="*70)
print("SCATTER: estimated vs true_acc_full")
print("="*70)

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

# Mix-Max
for _, row in results_df.iterrows():
    s = row['split']
    ax.scatter(
        row['true_acc_full'], row['mixmax_est'],
        color=SPLIT_COLORS[s], s=180, marker='*',
        alpha=0.9, zorder=5,
        label=f"Mix-Max ({row['split_label']})  "
              f"err={row['mixmax_err_vs_full']:.4f}",
    )

# Baselines
for m in baseline_methods:
    col = f'err_{m}'
    if m not in results_df.columns:
        continue
    for _, row in results_df.iterrows():
        ax.scatter(
            row['true_acc_full'], row[m],
            color=COLORS.get(m, 'gray'), s=70,
            alpha=0.75, marker='o',
        )
    # Одна запись в легенду на метод
    mae_m = mae_table[m][0] if m in mae_table else np.nan
    ax.scatter([], [], color=COLORS.get(m, 'gray'), s=70,
               label=f"{m}  MAE={mae_m:.4f}")

lo = results_df['true_acc_full'].min() - 0.03
hi = results_df['true_acc_full'].max() + 0.03
ax.plot([lo, hi], [lo, hi], 'k--', lw=2, alpha=0.5, label='Perfect')

ax.set_xlabel('True Accuracy (все сэмплы)', fontweight='bold')
ax.set_ylabel('Estimated Accuracy', fontweight='bold')
ax.set_title('Amazon: Estimated vs True Accuracy\n(все методы, одинаковый ground truth)',
             fontweight='bold')
ax.set_xlim(lo, hi)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

plt.savefig('amazon/figures/comparison/scatter_all_methods.png',
            dpi=300, bbox_inches='tight')
plt.savefig('amazon/figures/comparison/scatter_all_methods.pdf',
            bbox_inches='tight')
plt.show(); plt.close()
print("✓ scatter_all_methods")


# ============================================================================
# PLOT 2: Scatter — Mix-Max est vs true_acc_at_q* (threshold-matched)
# ============================================================================

print("\n" + "="*70)
print("SCATTER: Mix-Max est vs true_acc_at_q*")
print("="*70)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

for _, row in results_df.iterrows():
    s = row['split']
    ax.scatter(
        row['mixmax_true_at_q'], row['mixmax_est'],
        color=SPLIT_COLORS[s], s=150,
        marker=SPLIT_MARKERS[s], alpha=0.9, zorder=3,
        label=row['split_label'],
    )
    ax.annotate(
        row['split_label'],
        (row['mixmax_true_at_q'], row['mixmax_est']),
        fontsize=9, xytext=(6, 6), textcoords='offset points',
    )

lo = min(results_df['mixmax_true_at_q'].min(),
         results_df['mixmax_est'].min()) - 0.02
hi = max(results_df['mixmax_true_at_q'].max(),
         results_df['mixmax_est'].max()) + 0.02
ax.plot([lo, hi], [lo, hi], 'k--', lw=2, alpha=0.5)

ax.set_xlabel('True Accuracy при q* (среди принятых)', fontweight='bold')
ax.set_ylabel('Mix-Max Estimated Accuracy', fontweight='bold')
ax.set_title(f'Amazon Mix-Max: Est vs True при оптимальном q*\n'
             f'MAE={mae_at_q:.4f}', fontweight='bold')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.savefig('amazon/figures/comparison/scatter_mixmax_at_q.png',
            dpi=300, bbox_inches='tight')
plt.savefig('amazon/figures/comparison/scatter_mixmax_at_q.pdf',
            bbox_inches='tight')
plt.show(); plt.close()
print("✓ scatter_mixmax_at_q")


# ============================================================================
# PLOT 3: MAE bar chart
# ============================================================================

print("\n" + "="*70)
print("MAE BAR CHART")
print("="*70)

# Left panel: все методы vs true_acc_full
sorted_mae = sorted(mae_table.items(), key=lambda x: x[1][0])
m_names    = [x[0] for x in sorted_mae]
m_maes     = [x[1][0] for x in sorted_mae]
m_stds     = [x[1][1] for x in sorted_mae]
m_cols     = [COLORS.get(m, 'gray') for m in m_names]
m_xlbls    = ['Mix-Max\n(ours)' if m == 'Mix-Max' else m for m in m_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

axes[0].bar(range(len(m_names)), m_maes, yerr=m_stds, capsize=5,
            color=m_cols, alpha=0.8, edgecolor='black', linewidth=1)
axes[0].set_title('MAE vs True Accuracy (все сэмплы)\nОдинаковый ground truth для всех',
                  fontweight='bold')
axes[0].set_xlabel('Method', fontweight='bold')
axes[0].set_ylabel('MAE', fontweight='bold')
axes[0].set_xticks(range(len(m_names)))
axes[0].set_xticklabels(m_xlbls, rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)
for i, (mae, std) in enumerate(zip(m_maes, m_stds)):
    axes[0].text(i, mae + std + 0.002, f'{mae:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

# Right panel: Mix-Max (vs full) + Mix-Max (vs true@q*) + baselines (vs full)
right_names = (['Mix-Max\nvs all', 'Mix-Max\nvs true@q*'] +
               [m for m in m_names if m != 'Mix-Max'])
right_maes  = ([mae_table['Mix-Max'][0], mae_at_q] +
               [mae_table[m][0] for m in m_names if m != 'Mix-Max'])
right_stds  = ([mae_table['Mix-Max'][1], std_at_q] +
               [mae_table[m][1] for m in m_names if m != 'Mix-Max'])
right_cols  = (['#E53935', '#EF9A9A'] +
               [COLORS.get(m, 'gray') for m in m_names if m != 'Mix-Max'])

axes[1].bar(range(len(right_names)), right_maes, yerr=right_stds, capsize=5,
            color=right_cols, alpha=0.8, edgecolor='black', linewidth=1)
axes[1].set_title('Mix-Max: две метрики качества\nvs конкуренты (vs all)',
                  fontweight='bold')
axes[1].set_xlabel('Method', fontweight='bold')
axes[1].set_ylabel('MAE', fontweight='bold')
axes[1].set_xticks(range(len(right_names)))
axes[1].set_xticklabels(right_names, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)
for i, (mae, std) in enumerate(zip(right_maes, right_stds)):
    axes[1].text(i, mae + std + 0.002, f'{mae:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.savefig('amazon/figures/comparison/mae_bar.png', dpi=300, bbox_inches='tight')
plt.savefig('amazon/figures/comparison/mae_bar.pdf', bbox_inches='tight')
plt.show(); plt.close()
print("✓ mae_bar")


# ============================================================================
# PLOT 4: Score diagnostics per split
# ============================================================================

print("\n" + "="*70)
print("SCORE DIAGNOSTICS")
print("="*70)

for split in ("id_val", "ood_val", "test"):
    ms, ds_d, ls = decoy_results[split]
    n            = len(ms)
    pred         = ms.argmax(axis=1)
    tgt          = ms[np.arange(n),  pred]
    null         = ds_d[np.arange(n), pred]
    correct      = pred == ls
    acc          = correct.mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(f'Amazon {SPLIT_LABELS[split]}  —  true_acc={acc:.4f}',
                 fontweight='bold')

    for i, (arr_c, arr_i, title, xlabel) in enumerate([
        (tgt[correct],        tgt[~correct],
         'Target Score (pred class)', 'Score'),
        (null[correct],       null[~correct],
         'Null Score (pred class)', 'Score'),
        ((tgt-null)[correct], (tgt-null)[~correct],
         'Gap = Target − Null', 'Score difference'),
    ]):
        axes[i].hist(arr_c, bins=80, alpha=0.6, density=True,
                     color='#2196F3',
                     label=f'Correct (mean={np.nanmean(arr_c):.3f})')
        if (~correct).sum() > 0:
            axes[i].hist(arr_i, bins=80, alpha=0.6, density=True,
                         color='#F44336',
                         label=f'Incorrect (mean={np.nanmean(arr_i):.3f})')
        if i == 2:
            axes[i].axvline(0, color='black', linestyle='--',
                            linewidth=1.5, alpha=0.7)
        axes[i].set_title(title, fontweight='bold')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.savefig(f'amazon/figures/diagnostics/score_diag_{split}.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'amazon/figures/diagnostics/score_diag_{split}.pdf',
                bbox_inches='tight')
    plt.show(); plt.close()
    print(f"✓ score_diag_{split}")


# ============================================================================
# СОХРАНЕНИЕ
# ============================================================================

results_df.to_csv('amazon/figures/comparison/results.csv', index=False)
print("\n✓ results.csv сохранён")

print("\n" + "="*70)
print("ИТОГ")
print("="*70)
print(f"\n  MAE vs true_acc_full:")
for method, (mae, std) in sorted(mae_table.items(), key=lambda x: x[1][0]):
    marker = ' ← наш' if method == 'Mix-Max' else ''
    print(f"    {method:<12}: {mae:.4f} ± {std:.4f}{marker}")
print(f"\n  Mix-Max MAE vs true@q* : {mae_at_q:.4f} ± {std_at_q:.4f}")

print(f"\n  Per-split details:")
for _, row in results_df.iterrows():
    print(f"\n  {row['split_label']}:")
    print(f"    n                 : {row['n']:,}")
    print(f"    true_acc_full     : {row['true_acc_full']:.4f}")
    print(f"    mixmax_est        : {row['mixmax_est']:.4f}  "
          f"(err vs full={row['mixmax_err_vs_full']:.4f})")
    print(f"    mixmax_true@q*    : {row['mixmax_true_at_q']:.4f}  "
          f"(err@q={row['mixmax_error']:.4f})")
    print(f"    mixmax_best_q*    : {row['mixmax_best_q']:.4f}")
    print(f"    accepted          : {row['mixmax_n_accepted']:,}/{row['n']:,} "
          f"({row['mixmax_frac_accepted']*100:.1f}%)")
    for m in baseline_methods:
        if m in row:
            err = abs(row[m] - row['true_acc_full'])
            print(f"    {m:<12}    : est={row[m]:.4f}  err={err:.4f}")