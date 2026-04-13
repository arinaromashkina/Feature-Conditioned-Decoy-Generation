import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob

from data_processing.score_feature_dataset import (
    ScoreFeatureDataset,
    create_score_feature_dataset_bcss,
)
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
DEVICE         = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES    = 5
FEATURE_DIM    = 64
PATH_DATA      = '../../data/BCSS/training/bcss.mini.training.torch'
TEST_FOLDER    = '../../data/BCSS/test/'
FLOWS_PATH     = 'BCSS/bcss_score_shift_flow.pth'
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


def true_accuracy_full(model_scores, labels_np):
    """
    Обычная accuracy по всем пикселям.
    Это ground truth с которым сравниваем ВСЕХ.
    """
    pred = model_scores.argmax(axis=1)
    return float((pred == labels_np).mean())


def true_accuracy_at_threshold(model_scores, labels_np, conf_threshold):
    """
    Реальная accuracy только среди пикселей
    где max_score >= conf_threshold.
    Знаменатель = число принятых пикселей.
    """
    pred          = model_scores.argmax(axis=1)
    max_scores    = model_scores[np.arange(len(model_scores)), pred]
    mask          = max_scores >= conf_threshold
    n_accepted    = mask.sum()
    if n_accepted < MIN_PIXELS:
        return np.nan, 0
    acc = float((pred[mask] == labels_np[mask]).mean())
    return acc, int(n_accepted)


def find_best_mixmax_threshold(df_curve):
    """
    Найти q* где Accuracy_est максимальна.
    Вернуть: est_acc, true_acc_at_q, error, q*, n_accepted
    
    df_curve — результат compute_method_estimation_curve().
    Колонки которые нам нужны:
        q_value_method          — q-value порог
        Accuracy_est            — наша оценка accuracy (label-free)
        Accuracy_true_at_threshold — реальная accuracy среди принятых
        error_at_threshold      — |est - true_at_threshold|
        n_discoveries           — сколько пикселей принято
    """
    if len(df_curve) == 0:
        return dict(
            mixmax_est=np.nan,
            mixmax_true_at_q=np.nan,
            mixmax_error=np.nan,
            mixmax_best_q=np.nan,
            mixmax_n_accepted=0,
        )

    best_idx = df_curve['Accuracy_est'].idxmax()
    row      = df_curve.loc[best_idx]

    return dict(
        mixmax_est       = float(row['Accuracy_est']),
        mixmax_true_at_q = float(row['Accuracy_true_at_threshold']),
        mixmax_error     = float(row['error_at_threshold']),
        mixmax_best_q    = float(row['q_value_method']),
        mixmax_n_accepted= int(row['n_discoveries']),
    )


# ============================================================================
# ДАННЫЕ И FLOW
# ============================================================================

print("\n" + "="*70)
print("LOADING TRAINING DATA")
print("="*70)

data     = torch.load(PATH_DATA)
train_ds = create_score_feature_dataset_bcss(data, DEVICE)
train_scores, train_labels = get_scores_from_ds(train_ds)
print(f"Train scores shape: {train_scores.shape}")
for c in range(NUM_CLASSES):
    print(f"  Class {c}: {(train_labels == c).sum()} samples")

print("\n" + "="*70)
print("FLOW MODEL")
print("="*70)

flow = ScoreShiftFlowWrapper(
    num_classes=NUM_CLASSES,
    n_flows=12,
    feature_dim=FEATURE_DIM,
    hidden_dim=256,
    encoder_dim=128,
).to(DEVICE)

if os.path.exists(FLOWS_PATH):
    flow.load_state_dict(torch.load(FLOWS_PATH, map_location=DEVICE))
    print(f"✓ Flow loaded from {FLOWS_PATH}")
else:
    print("Training ScoreShiftFlow...")
    flow.train_flow(
        train_ds, epochs=30, lr=3e-4, batch_size=256,
        device=DEVICE, patience=5, grad_clip=1.0)
    torch.save(flow.state_dict(), FLOWS_PATH)
    print(f"✓ Flow saved to {FLOWS_PATH}")

# Source данные для baseline методов (40% subsample от train)
print("\nПодготовка source данных для baselines...")
np.random.seed(42)
subset_idx  = np.random.choice(len(train_ds), int(0.4 * len(train_ds)), replace=False)
subset_ds   = Subset(train_ds, subset_idx.tolist())

source_logits, _, source_labels = flow.generate_decoys(subset_ds, device=DEVICE)
source_logits_t = torch.tensor(source_logits).to(DEVICE)
source_labels_t = torch.tensor(source_labels).to(DEVICE)
temp            = calibration_temp(source_logits_t, source_labels_t)
scaled_source   = source_logits_t / temp
print(f"✓ Source: {len(source_labels)} samples, temp={temp:.4f}")


# ============================================================================
# ОБРАБОТКА ТЕСТОВЫХ ФАЙЛОВ
# ============================================================================

print("\n" + "="*70)
print("PROCESSING TEST FILES")
print("="*70)

#test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, '*.tensor')))
test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, '*.tensor')))[:10]
print(f"Найдено {len(test_files)} тестовых файлов")

results     = []
all_fdr_dfs = {}   # tile_name → df_mm (для графиков accuracy)

for fpath in tqdm(test_files, desc="Test files"):
    tile_name = os.path.basename(fpath)

    # ── Загрузка тайла ───────────────────────────────────────────────────
    try:
        test_ds = load_bcss_test_file(fpath)
    except Exception as e:
        print(f"  ✗ {tile_name}: {e}")
        continue

    # ── Генерация decoys ─────────────────────────────────────────────────
    model_scores, decoy_scores, labels_np = flow.generate_decoys(
        test_ds, device=DEVICE)

    # Subsample
    model_scores = model_scores[::SUBSAMPLE_STEP]
    decoy_scores = decoy_scores[::SUBSAMPLE_STEP]
    labels_np    = labels_np[::SUBSAMPLE_STEP]
    n            = len(labels_np)

    # ── True accuracy (обычная, по всем пикселям) ─────────────────────────
    # ЭТО главная метрика с которой сравниваем всех
    acc_full = true_accuracy_full(model_scores, labels_np)

    # ── Mix-Max FDR curve ────────────────────────────────────────────────
    df_mm    = control_fdr_mixmax(model_scores, labels_np, decoy_scores)
    df_curve = compute_method_estimation_curve(df_mm, 'q_values_mixmax', pi0=0.0)

    # Находим q* где наша оценка accuracy максимальна
    mm_result = find_best_mixmax_threshold(df_curve)

    all_fdr_dfs[tile_name] = (df_mm, df_curve)

    # ── Baseline методы ──────────────────────────────────────────────────
    # Каждый возвращает одно число — оценку accuracy по всему тайлу
    target_logits_t = torch.tensor(model_scores).to(DEVICE)
    scaled_target   = target_logits_t / temp

    baseline_est = {}
    for mname, mfunc in BASELINE_METHODS.items():
        try:
            baseline_est[mname] = float(
                mfunc(scaled_source, source_labels_t, scaled_target))
        except Exception as e:
            print(f"  {tile_name} – {mname}: FAILED ({e})")
            baseline_est[mname] = np.nan

    results.append({
        'tile'            : tile_name,
        'n_pixels'        : n,
        # ── Единственный ground truth для сравнения всех методов ──────
        'true_acc_full'   : acc_full,
        # ── Mix-Max результаты ────────────────────────────────────────
        **mm_result,       # mixmax_est, mixmax_true_at_q, mixmax_error,
                           # mixmax_best_q, mixmax_n_accepted
        'mixmax_frac_accepted': mm_result['mixmax_n_accepted'] / max(n, 1),
        # ── Baselines ─────────────────────────────────────────────────
        **baseline_est,
    })

results_df = pd.DataFrame(results)
print(f"\n✓ Обработано {len(results_df)} тайлов")


# ============================================================================
# ОШИБКИ
# ============================================================================

print("\n" + "="*70)
print("ОШИБКИ")
print("="*70)

baseline_methods = list(BASELINE_METHODS.keys())

# ── Ошибка Mix-Max ────────────────────────────────────────────────────────
# 1) vs true_acc_full   (сравниваем честно с конкурентами)
# 2) vs true_acc_at_q*  (то что Mix-Max реально предсказывает)
results_df['mixmax_err_vs_full'] = np.abs(
    results_df['mixmax_est'] - results_df['true_acc_full'])

# mixmax_error уже есть — это |mixmax_est - mixmax_true_at_q|

# ── Ошибка baselines vs true_acc_full ────────────────────────────────────
for m in baseline_methods:
    results_df[f'err_{m}'] = np.abs(results_df[m] - results_df['true_acc_full'])

# ── Сводка MAE ───────────────────────────────────────────────────────────
print("\n" + "-"*60)
print("MAE vs true_acc_full (одинаковый ground truth для всех)")
print("-"*60)

mae_table = {}
mae_table['Mix-Max'] = (
    results_df['mixmax_err_vs_full'].mean(),
    results_df['mixmax_err_vs_full'].std(),
)
for m in baseline_methods:
    mae_table[m] = (
        results_df[f'err_{m}'].mean(),
        results_df[f'err_{m}'].std(),
    )

for method, (mae, std) in sorted(mae_table.items(), key=lambda x: x[1][0]):
    print(f"  {method:<12}: MAE = {mae:.4f} ± {std:.4f}")

print("\n" + "-"*60)
print("Mix-Max: est vs true_acc_at_q* (threshold-matched, без конкурентов)")
print("-"*60)
mae_at_q = results_df['mixmax_error'].mean()
std_at_q = results_df['mixmax_error'].std()
print(f"  Mix-Max       : MAE = {mae_at_q:.4f} ± {std_at_q:.4f}")

print("\n" + "-"*60)
print("Per-tile сводка")
print("-"*60)
print(f"{'Tile':<45} {'true_full':>9} {'mm_est':>7} "
      f"{'mm_true@q':>10} {'err@q':>7} {'q*':>5} {'accept%':>8}")
print("-"*105)
for _, row in results_df.sort_values('true_acc_full').iterrows():
    print(
        f"  {row['tile'][:43]:<43} "
        f"{row['true_acc_full']:9.3f} "
        f"{row['mixmax_est']:7.3f} "
        f"{row['mixmax_true_at_q']:10.3f} "
        f"{row['mixmax_error']:7.3f} "
        f"{row['mixmax_best_q']:5.3f} "
        f"{row['mixmax_frac_accepted']*100:7.1f}%"
    )
print(
    f"\n  {'MEAN':<43} "
    f"{results_df['true_acc_full'].mean():9.3f} "
    f"{results_df['mixmax_est'].mean():7.3f} "
    f"{results_df['mixmax_true_at_q'].mean():10.3f} "
    f"{results_df['mixmax_error'].mean():7.3f} "
    f"{'':5} "
    f"{results_df['mixmax_frac_accepted'].mean()*100:7.1f}%"
)


# ============================================================================
# PLOT 1: Accuracy estimation curve (per tile)
# ============================================================================

print("\n" + "="*70)
print("ACCURACY ESTIMATION PLOTS (per tile)")
print("="*70)

for tile_name, (df_mm, _) in all_fdr_dfs.items():
    plot_accuracy_estimation(
        df_mm,
        q_value_column  = 'q_values_mixmax',
        method_name     = 'Mix-Max',
        corruption_name = tile_name.replace('.tensor', ''),
        save_dir        = 'BCSS/figures/accuracy',
    )


# ============================================================================
# PLOT 2: Scatter — estimated vs true_acc_full (все методы)
# ============================================================================

print("\n" + "="*70)
print("SCATTER: estimated vs true_acc_full")
print("="*70)

fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

# Mix-Max (est vs true_acc_full)
ax.scatter(
    results_df['true_acc_full'],
    results_df['mixmax_est'],
    color=COLORS['Mix-Max'], s=120, marker='*',
    alpha=0.9, zorder=4,
    label=f"Mix-Max  MAE={mae_table['Mix-Max'][0]:.4f}",
)

# Baselines
for m in baseline_methods:
    mae_m = mae_table[m][0]
    ax.scatter(
        results_df['true_acc_full'],
        results_df[m],
        color=COLORS.get(m, 'gray'), s=60, alpha=0.75,
        label=f"{m}  MAE={mae_m:.4f}",
    )

lo = results_df['true_acc_full'].min() - 0.03
hi = results_df['true_acc_full'].max() + 0.03
ax.plot([lo, hi], [lo, hi], 'k--', lw=2, alpha=0.5, label='Perfect')

ax.set_xlabel('True Accuracy (все пиксели)', fontweight='bold')
ax.set_ylabel('Estimated Accuracy', fontweight='bold')
ax.set_title('Estimated vs True ACC\n(все методы, одинаковый ground truth)',
             fontweight='bold')
ax.set_xlim(lo, hi)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.savefig('BCSS/figures/comparison/scatter_all_methods.png', dpi=300, bbox_inches='tight')
plt.savefig('BCSS/figures/comparison/scatter_all_methods.pdf', bbox_inches='tight')
plt.show(); plt.close()
print("✓ scatter_all_methods")


# ============================================================================
# PLOT 3: Scatter — Mix-Max est vs true_acc_at_q* (threshold-matched)
# ============================================================================

print("\n" + "="*70)
print("SCATTER: Mix-Max est vs true_acc_at_q*")
print("="*70)

fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

ax.scatter(
    results_df['mixmax_true_at_q'],
    results_df['mixmax_est'],
    color=COLORS['Mix-Max'], s=100, alpha=0.85, zorder=3,
)

for _, row in results_df.iterrows():
    ax.annotate(
        row['tile'][:18],
        (row['mixmax_true_at_q'], row['mixmax_est']),
        fontsize=6, alpha=0.6, xytext=(4, 4), textcoords='offset points',
    )

lo = min(results_df['mixmax_true_at_q'].min(),
         results_df['mixmax_est'].min()) - 0.02
hi = max(results_df['mixmax_true_at_q'].max(),
         results_df['mixmax_est'].max()) + 0.02
ax.plot([lo, hi], [lo, hi], 'k--', lw=2, alpha=0.5)

ax.set_xlabel('True ACC при q* (среди принятых пикселей)', fontweight='bold')
ax.set_ylabel('Mix-Max Estimated ACC', fontweight='bold')
ax.set_title(f'Mix-Max: Est vs True ACC при оптимальном q*\nMAE={mae_at_q:.4f}',
             fontweight='bold')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.grid(True, alpha=0.3)

plt.savefig('BCSS/figures/comparison/scatter_mixmax_at_q.png', dpi=300, bbox_inches='tight')
plt.savefig('BCSS/figures/comparison/scatter_mixmax_at_q.pdf', bbox_inches='tight')
plt.show(); plt.close()
print("✓ scatter_mixmax_at_q")


# ============================================================================
# PLOT 4: MAE bar chart
# ============================================================================

print("\n" + "="*70)
print("MAE BAR CHART")
print("="*70)

# Все методы vs true_acc_full — честное сравнение
sorted_mae = sorted(mae_table.items(), key=lambda x: x[1][0])
m_names = [x[0] for x in sorted_mae]
m_maes  = [x[1][0] for x in sorted_mae]
m_stds  = [x[1][1] for x in sorted_mae]
m_cols  = [COLORS.get(m, 'gray') for m in m_names]
m_xlbls = ['Mix-Max\n(ours)' if m == 'Mix-Max' else m for m in m_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Left: все методы vs true_acc_full
axes[0].bar(range(len(m_names)), m_maes, yerr=m_stds, capsize=5,
            color=m_cols, alpha=0.8, edgecolor='black', linewidth=1)
axes[0].set_title('MAE vs True Accuracy (все пиксели)\nОдинаковый ground truth для всех',
                  fontweight='bold')
axes[0].set_xlabel('Method', fontweight='bold')
axes[0].set_ylabel('MAE', fontweight='bold')
axes[0].set_xticks(range(len(m_names)))
axes[0].set_xticklabels(m_xlbls, rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)
for i, (mae, std) in enumerate(zip(m_maes, m_stds)):
    axes[0].text(i, mae + std + 0.002, f'{mae:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

# Right: Mix-Max vs true_acc_full  +  Mix-Max vs true_at_q*
right_names = (['Mix-Max\nvs all pixels', 'Mix-Max\nvs true@q*'] +
               [m for m in m_names if m != 'Mix-Max'])
right_maes  = ([mae_table['Mix-Max'][0], mae_at_q] +
               [mae_table[m][0] for m in m_names if m != 'Mix-Max'])
right_stds  = ([mae_table['Mix-Max'][1], std_at_q] +
               [mae_table[m][1] for m in m_names if m != 'Mix-Max'])
right_cols  = (['#E53935', '#EF9A9A'] +
               [COLORS.get(m, 'gray') for m in m_names if m != 'Mix-Max'])

axes[1].bar(range(len(right_names)), right_maes, yerr=right_stds, capsize=5,
            color=right_cols, alpha=0.8, edgecolor='black', linewidth=1)
axes[1].set_title('Mix-Max: две метрики качества\nvs конкуренты (vs all pixels)',
                  fontweight='bold')
axes[1].set_xlabel('Method', fontweight='bold')
axes[1].set_ylabel('MAE', fontweight='bold')
axes[1].set_xticks(range(len(right_names)))
axes[1].set_xticklabels(right_names, rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)
for i, (mae, std) in enumerate(zip(right_maes, right_stds)):
    axes[1].text(i, mae + std + 0.002, f'{mae:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.savefig('BCSS/figures/comparison/mae_bar.png', dpi=300, bbox_inches='tight')
plt.savefig('BCSS/figures/comparison/mae_bar.pdf', bbox_inches='tight')
plt.show(); plt.close()
print("✓ mae_bar")


# ============================================================================
# PLOT 5: Score distributions для первого тайла
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

    # plot_score_distribution_with_decoys для каждого класса
    for i in range(NUM_CLASSES):
        plot_score_distribution_with_decoys(
            train_scores[:, i],   # train scores для класса i
            train_labels,
            ms[:, i],             # test scores для класса i
            ls,
            ds_decoy[:, i],       # decoy scores для класса i
            filename  = f"BCSS/figures/diagnostics/scores_dist_class{i}",
            title     = f"Score Distribution with Decoys — Class {i}",
            xlim      = (-10, 10),
            show_kde  = True,
            class_id  = i,
        )
    print(f"✓ Score distributions сохранены для {NUM_CLASSES} классов")

except Exception as e:
    print(f"✗ Ошибка при построении distributions: {e}")


# ============================================================================
# СОХРАНЕНИЕ
# ============================================================================

results_df.to_csv('BCSS/figures/comparison/results.csv', index=False)
print("\n✓ results.csv сохранён")

print("\n" + "="*70)
print("ИТОГ")
print("="*70)
print(f"  Тайлов обработано   : {len(results_df)}")
print(f"  Mean true_acc_full  : {results_df['true_acc_full'].mean():.3f} "
      f"± {results_df['true_acc_full'].std():.3f}")
print(f"\n  MAE (vs true_acc_full):")
for method, (mae, std) in sorted(mae_table.items(), key=lambda x: x[1][0]):
    marker = ' ← наш' if method == 'Mix-Max' else ''
    print(f"    {method:<12}: {mae:.4f} ± {std:.4f}{marker}")
print(f"\n  Mix-Max MAE vs true@q* : {mae_at_q:.4f} ± {std_at_q:.4f}")
print(f"  Mix-Max avg accepted   : "
      f"{results_df['mixmax_frac_accepted'].mean()*100:.1f}%")