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
        xlim     = (-6, 6),
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ============================================================
# CONFIG
# ============================================================
DEVICE       = "cuda:1"
NUM_CLASSES  = 5
FEATURE_DIM  = 768
MAX_TOK_LEN  = 512
BATCH_SIZE   = 8
LR           = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS       = 3
DATA_DIR     = "./data"

TRAIN_CATEGORIES = [
    "Books",
    "Electronics",
    "Movies_and_TV",
    "CDs_and_Vinyl",
    "Clothing_Shoes_and_Jewelry",
]
OOD_CATEGORIES = [
    "Sports_and_Outdoors",
    "Tools_and_Home_Improvement",
]


# ============================================================
# 1. TOKENIZER / TRANSFORM
# ============================================================
def get_distilbert_transform(max_token_length: int = MAX_TOK_LEN):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def transform(text: str) -> torch.Tensor:
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )
        x = torch.stack(
            (tokens["input_ids"], tokens["attention_mask"]), dim=2
        )
        return x.squeeze(0)  # (max_length, 2)

    return transform


# ============================================================
# 2. МОДЕЛЬ
# ============================================================
class AmazonDistilBERT(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.distilbert     = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier     = nn.Linear(768, num_classes)
        self.dropout        = nn.Dropout(0.1)
        self.relu           = nn.ReLU()

    def forward(self, x: torch.Tensor, return_features: bool = False):
        input_ids      = x[:, :, 0]
        attention_mask = x[:, :, 1]

        hidden = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        cls    = hidden[:, 0]
        feat   = self.dropout(self.relu(self.pre_classifier(cls)))
        logits = self.classifier(feat)

        return (logits, feat) if return_features else logits


# ============================================================
# 3. DATASET (HuggingFace)
# ============================================================
class AmazonHFDataset(Dataset):
    def __init__(self, texts, labels, reviewers, categories, transform=None):
        self.texts      = texts
        self.labels     = torch.tensor(labels,     dtype=torch.long)
        self.reviewers  = torch.tensor(reviewers,  dtype=torch.long)
        self.categories = torch.tensor(categories, dtype=torch.long)
        self.transform  = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y    = self.labels[idx]
        meta = torch.stack([self.reviewers[idx], self.categories[idx]])
        x    = self.transform(text) if self.transform else text
        return x, y, meta


# ============================================================
# 4. ЗАГРУЗКА ДАННЫХ из HF кэша
# ============================================================
def load_amazon_hf_split(
    categories,
    cat_offset: int = 0,
    max_per_cat     = None,
):
    all_texts, all_labels, all_reviewers, all_cats = [], [], [], []
    reviewer_map = {}

    for cat_idx, category in enumerate(categories):
        print(f"  Loading {category} …")
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_review_{category}",
                split="full",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  ⚠ Skipping {category}: {e}")
            continue

        if max_per_cat:
            ds = ds.select(range(min(max_per_cat, len(ds))))

        for item in ds:
            text   = (item.get("text") or "").strip()
            rating = item.get("rating", 3)
            user   = item.get("user_id", "unknown")

            if not text:
                continue

            label = max(0, min(4, int(round(float(rating))) - 1))

            if user not in reviewer_map:
                reviewer_map[user] = len(reviewer_map)

            all_texts.append(text)
            all_labels.append(label)
            all_reviewers.append(reviewer_map[user])
            all_cats.append(cat_offset + cat_idx)

    return all_texts, all_labels, all_reviewers, all_cats


def get_amazon_hf_data(
    train_categories = TRAIN_CATEGORIES,
    ood_categories   = OOD_CATEGORIES,
    max_per_cat      = 50_000,
    val_fraction     = 0.1,
    max_token_length = MAX_TOK_LEN,
    batch_size       = BATCH_SIZE,
) -> dict:
    transform = get_distilbert_transform(max_token_length)

    print("Loading TRAIN categories …")
    texts, labels, reviewers, cats = load_amazon_hf_split(
        train_categories, cat_offset=0, max_per_cat=max_per_cat
    )

    n   = len(texts)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    val_n     = int(n * val_fraction)
    val_idx   = idx[:val_n]
    train_idx = idx[val_n:]

    def make_subset(indices):
        return AmazonHFDataset(
            texts      = [texts[i]     for i in indices],
            labels     = [labels[i]    for i in indices],
            reviewers  = [reviewers[i] for i in indices],
            categories = [cats[i]      for i in indices],
            transform  = transform,
        )

    train_set  = make_subset(train_idx)
    id_val_set = make_subset(val_idx)

    print("Loading OOD categories …")
    ood_texts, ood_labels, ood_reviewers, ood_cats = load_amazon_hf_split(
        ood_categories,
        cat_offset=len(train_categories),
        max_per_cat=max_per_cat,
    )

    n_ood   = len(ood_texts)
    ood_idx = np.arange(n_ood)
    np.random.seed(0)
    np.random.shuffle(ood_idx)

    half        = n_ood // 2
    ood_val_idx = ood_idx[:half]
    test_idx    = ood_idx[half:]

    def make_ood_subset(indices):
        return AmazonHFDataset(
            texts      = [ood_texts[i]     for i in indices],
            labels     = [ood_labels[i]    for i in indices],
            reviewers  = [ood_reviewers[i] for i in indices],
            categories = [ood_cats[i]      for i in indices],
            transform  = transform,
        )

    ood_val_set = make_ood_subset(ood_val_idx)
    test_set    = make_ood_subset(test_idx)

    def make_loader(dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = 0,
            pin_memory  = False,
        )

    result = {
        "train_set"     : train_set,
        "train_loader"  : make_loader(train_set, shuffle=True),
        "id_val_set"    : id_val_set,
        "id_val_loader" : make_loader(id_val_set),
        "ood_val_set"   : ood_val_set,
        "ood_val_loader": make_loader(ood_val_set),
        "test_set"      : test_set,
        "test_loader"   : make_loader(test_set),
    }

    print("\nDataset sizes:")
    for name in ("train", "id_val", "ood_val", "test"):
        print(f"  {name:8s}: {len(result[f'{name}_set']):>8,} samples")

    return result


# ============================================================
# 5. ОБУЧЕНИЕ / ОЦЕНКА
# ============================================================
def train_amazon_model(
    trainloader,
    val_loader,
    epochs = EPOCHS,
    device = DEVICE,
) -> AmazonDistilBERT:
    model     = AmazonDistilBERT(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = 0,
        num_training_steps = len(trainloader) * epochs,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y, _ in tqdm(trainloader, desc=f"[Train] Epoch {epoch}/{epochs}"):
            x, y   = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            correct      += (logits.detach().argmax(1) == y).sum().item()
            total        += y.size(0)

        print(f"Epoch {epoch}: loss={running_loss/len(trainloader):.4f}  "
              f"train_acc={correct/total:.4f}")

        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, device)
            print(f"          id_val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(model, dataloader, device=DEVICE) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y    = x.to(device), y.to(device)
            preds   = model(x).argmax(1)  # argmax на логитах — OK
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total


# ============================================================
# 6. NEGATIVE SCORES POOL (логиты, без softmax)
# ============================================================
@torch.no_grad()
def collect_negative_scores(
    model,
    dataloader,
    num_classes = NUM_CLASSES,
    device      = DEVICE,
) -> dict:
    model.eval()
    all_scores, all_labels = [], []

    safe_loader = DataLoader(
        dataloader.dataset,
        batch_size  = dataloader.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    for x, y, _ in tqdm(safe_loader, desc="Collecting negative scores"):
        # логиты без softmax
        logits = model(x.to(device)).cpu()
        all_scores.append(logits)
        all_labels.append(y)

    all_scores = torch.cat(all_scores)  # (N, num_classes)
    all_labels = torch.cat(all_labels)  # (N,)

    negative_pools = {}
    for c in range(num_classes):
        mask              = all_labels != c
        negative_pools[c] = all_scores[mask]
        print(f"  class {c+1}: negative pool size = {len(negative_pools[c]):,}")

    return negative_pools


# ============================================================
# 7. SCORE-FEATURE DATASET (логиты, без softmax)
# ============================================================
class ScoreFeatureDataset(Dataset):
    """
    Каждый элемент: (cnn_scores, features, target_decoy_scores, label)
        cnn_scores         : (num_classes,) — логиты модели
        features           : (768,)         — penultimate слой
        target_decoy_scores: (num_classes,) — логиты негативного примера
        label              : int            — истинный класс (0-4)
    """
    def __init__(self, cnn_scores, features, target_decoy_scores, labels):
        self.cnn_scores          = cnn_scores
        self.features            = features
        self.target_decoy_scores = target_decoy_scores
        self.labels              = labels

    def __len__(self):
        return len(self.cnn_scores)

    def __getitem__(self, idx):
        return (
            self.cnn_scores[idx],
            self.features[idx],
            self.target_decoy_scores[idx],
            self.labels[idx],
        )


@torch.no_grad()
def create_score_feature_dataset(
    dataloader,
    model,
    negative_pools,
    device = DEVICE,
) -> ScoreFeatureDataset:
    model.eval()
    all_cnn_scores, all_features, all_decoys, all_labels = [], [], [], []

    safe_loader = DataLoader(
        dataloader.dataset,
        batch_size  = dataloader.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    for x, y, _ in tqdm(safe_loader, desc="Building ScoreFeatureDataset"):
        x = x.to(device)
        logits, feats = model(x, return_features=True)

        # логиты без softmax
        scores = logits.cpu()
        feats  = feats.cpu()

        decoys = []
        for label in y.tolist():
            pool = negative_pools[label]
            i    = torch.randint(0, len(pool), (1,)).item()
            decoys.append(pool[i])
        decoys = torch.stack(decoys)

        all_cnn_scores.append(scores)
        all_features.append(feats)
        all_decoys.append(decoys)
        all_labels.append(y)

    return ScoreFeatureDataset(
        torch.cat(all_cnn_scores),
        torch.cat(all_features),
        torch.cat(all_decoys),
        torch.cat(all_labels),
    )


# ============================================================
# 8. ЗАГРУЗКА СОХРАНЁННЫХ ДАТАСЕТОВ
# ============================================================
def load_score_feature_datasets(
    save_dir = "./data/amazon_data/score_feature_datasets",
):
    datasets = {}
    for split in ("train", "id_val", "ood_val", "test"):
        path = f"{save_dir}/{split}.pt"
        raw  = torch.load(path, map_location="cpu")
        datasets[split] = ScoreFeatureDataset(
            cnn_scores          = raw["cnn_scores"],
            features            = raw["features"],
            target_decoy_scores = raw["target_decoy_scores"],
            labels              = raw["labels"],
        )
        print(f"  {split:8s}: {len(datasets[split]):,} samples")

    negative_pools = torch.load(
        f"{save_dir}/negative_pools.pt", map_location="cpu"
    )
    print("  negative_pools: loaded ✓")
    return datasets, negative_pools


# ============================================================
# 9. ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================================
def main(
    train_model = False,
    model_path  = "amazon_distilbert.pt",
    save_dir    = "./data/amazon_data/score_feature_datasets",
):
    print("=" * 60)

    # только кэш, без интернета
    os.environ["HF_DATASETS_OFFLINE"]  = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # --- данные ---
    print("Loading Amazon data from HF cache …")
    data = get_amazon_hf_data(max_per_cat=50_000)

    # --- модель ---
    if train_model:
        print("\nTraining DistilBERT …")
        model = train_amazon_model(
            data["train_loader"],
            val_loader = data["id_val_loader"],
        )
        torch.save(model.state_dict(), model_path)
        print(f"Model saved → {model_path}")
    else:
        print(f"\nLoading model from {model_path} …")
        model = AmazonDistilBERT(NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("Model loaded ✓")

    # --- оценка ---
    print("\nEvaluation:")
    for split in ("id_val", "ood_val", "test"):
        acc = evaluate_model(model, data[f"{split}_loader"])
        print(f"  {split:8s} acc = {acc:.4f}")

    # --- negative pools ---
    print("\nCollecting negative score pools …")
    negative_pools = collect_negative_scores(
        model, data["train_loader"], NUM_CLASSES, DEVICE
    )

    # --- ScoreFeatureDatasets ---
    print("\nBuilding ScoreFeatureDatasets …")
    score_datasets = {}
    for split in ("train", "id_val", "ood_val", "test"):
        score_datasets[split] = create_score_feature_dataset(
            data[f"{split}_loader"], model, negative_pools
        )
        n    = len(score_datasets[split])
        fdim = score_datasets[split].features.shape[1]
        print(f"  {split:8s}: {n:,} samples | feature_dim={fdim}")

    # --- сохраняем на диск ---
    os.makedirs(save_dir, exist_ok=True)

    for split, ds in score_datasets.items():
        path = f"{save_dir}/{split}.pt"
        torch.save({
            "cnn_scores"          : ds.cnn_scores,
            "features"            : ds.features,
            "target_decoy_scores" : ds.target_decoy_scores,
            "labels"              : ds.labels,
        }, path)
        print(f"  Saved {split} → {path}")

    torch.save(negative_pools, f"{save_dir}/negative_pools.pt")
    print(f"  Saved negative_pools → {save_dir}/negative_pools.pt")

    # --- sanity check ---
    cnn_s, feat, decoy, lbl = score_datasets["train"][0]
    c = lbl.item()
    print(f"\nSanity check → label={c+1}★")
    print(f"  cnn_logits  : {cnn_s.numpy().round(3)}")
    print(f"  decoy_logits: {decoy.numpy().round(3)}")
    print(f"  cnn[{c}]={cnn_s[c]:.4f}  decoy[{c}]={decoy[c]:.4f}")
    print(f"  feat shape  : {feat.shape}")

    return model, score_datasets


# ============================================================
if __name__ == "__main__":
    model, score_datasets = main(train_model=False)

    train_ds   = score_datasets["train"]
    id_val_ds  = score_datasets["id_val"]
    ood_val_ds = score_datasets["ood_val"]
    test_ds    = score_datasets["test"]