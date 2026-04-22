import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob

from data_processing.score_feature_dataset import *
from data_processing.negative_scores_pool import *
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
FLOWS_PATH     = 'BCSS/bcss_score_shift_flow_new.pth'
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


# ============================================================================
# ДАННЫЕ И FLOW
# ============================================================================

print("\n" + "="*70)
print("LOADING TRAINING DATA")
print("="*70)
data = torch.load(PATH_DATA)

# ── Строим пулы (новая логика) ────────────────────────────────────────────
pool_score, pool_vectors, train_scores, train_labels = \
    build_error_conditioned_pools_bcss(data, verbose=True)

# ── Создаём train dataset с decoys ───────────────────────────────────────
train_ds = create_score_feature_dataset_bcss_v2(
    data,
    pool_score,
    pool_vectors,
    strategy='score_coord',
    device=DEVICE
)

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
        train_ds, epochs=40, lr=3e-4, batch_size=256,
        device=DEVICE, patience=5, grad_clip=1.0)
    torch.save(flow.state_dict(), FLOWS_PATH)
    print(f"✓ Flow saved to {FLOWS_PATH}")

# Source данные для baseline методов (40% subsample от train)
print("\nПодготовка source данных для baselines...")
np.random.seed(42)
subset_idx = np.random.choice(len(train_ds), int(0.4 * len(train_ds)), replace=False)
subset_ds  = Subset(train_ds, subset_idx.tolist())

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

test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, '*.tensor')))
print(f"Найдено {len(test_files)} тестовых файлов")

results     = []
all_fdr_dfs = {}   # tile_name → df_mm (для графиков accuracy)

all_results = []

for fpath in tqdm(test_files, desc="Test files"):
    tile_name = os.path.basename(fpath)
    print(f"\n{'─'*60}")
    print(f"  TILE: {tile_name}")
    print(f"{'─'*60}")

    # ── Загрузка тайла ───────────────────────────────────────────────────
    try:
        test_ds = load_bcss_test_file(fpath)
    except Exception as e:
        print(f"  ✗ {tile_name}: {e}")
        continue

    # ── Генерация decoys ─────────────────────────────────────────────────
    ms, ds_flow, ls = flow.generate_decoys(test_ds, device=DEVICE)

    # Subsample
    ms       = ms[::SUBSAMPLE_STEP]
    ds_flow  = ds_flow[::SUBSAMPLE_STEP]
    ls       = ls[::SUBSAMPLE_STEP]
    n        = len(ls)

    if n < MIN_PIXELS:
        print(f"  ✗ Too few pixels: {n}")
        continue

    target_logits = torch.tensor(ms).to(DEVICE)
    target_labels = torch.tensor(ls).to(DEVICE)

    true_acc_full = float((ms.argmax(axis=1) == ls).mean())

    scaled_target = target_logits / temp

    # ── Baseline methods ──────────────────────────────────────────────────
    baseline_estimates = {}
    baseline_errors    = {}

    for method_name, method_func in BASELINE_METHODS.items():
        try:
            estimate = float(method_func(scaled_source, source_labels_t, scaled_target))
        except Exception as e:
            print(f"  {method_name}: FAILED ({e})")
            estimate = np.nan
        estimate = np.clip(estimate, 0.0, 1.0)
        error = abs(estimate - true_acc_full)
        baseline_estimates[method_name] = estimate
        baseline_errors[method_name]    = error
        print(f"  {method_name}: {estimate:.3f} (error: {error:.3f})")

    print(f"{'─'*60}")

    # ── Подготовка данных ─────────────────────────────────────────────────
    true_labels   = ls
    n_samples     = len(true_labels)

    pred_scores   = np.max(ms, axis=1)
    pred_label    = np.argmax(ms, axis=1)
    decoy_scores  = np.max(ds_flow, axis=1)

    correct_pred  = (true_labels == pred_label).astype(int)

    print(f"  accuracy : {correct_pred.sum() / n_samples:.4f}")
    print(f"  error/pi0: {1 - correct_pred.sum() / n_samples:.4f}")

    # ── Figure 1: Score distributions ─────────────────────────────────────
    bin_width  = 0.05
    bins       = np.arange(pred_scores.min() - 0.5,
                           pred_scores.max() + 0.5, bin_width)
    stat       = 'density'
    plot_alpha = 0.3

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(pred_scores,
                 bins=bins, stat=stat, color='blue',   kde=True,
                 fill=True, alpha=plot_alpha, label='model_mixture', ax=ax)
    sns.histplot(decoy_scores,
                 bins=bins, stat=stat, color='orange', kde=True,
                 fill=True, alpha=plot_alpha, label='null', ax=ax)
    sns.histplot(pred_scores[true_labels != pred_label],
                 bins=bins, stat=stat, color='red',    kde=True,
                 fill=True, alpha=plot_alpha, label='incorrect', ax=ax)
    ax.set_title(f'{tile_name[:30]} — Score Distributions')
    ax.legend()
    plt.tight_layout()
    safe_name = 'shift_flow_' + tile_name.replace('.tensor', '')
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_scores.png', dpi=150)
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_scores.pdf')
    plt.show(); plt.close()

    # ── True FDR (ground truth) ───────────────────────────────────────────
    sort_idx           = np.argsort(pred_scores)
    pred_scores_sorted = pred_scores[sort_idx]
    label_sorted       = true_labels[sort_idx]
    pred_label_sorted  = pred_label[sort_idx]
    correct_pred_s     = (label_sorted == pred_label_sorted).astype(int)

    FD       = 1 - correct_pred_s
    FD_CF    = np.cumsum(FD[::-1])[::-1]
    D_CF     = np.arange(0, n_samples)[::-1] + 1
    FDR_true = FD_CF / D_CF
    FDR_true = np.clip(FDR_true, 0, 1)
    QVAL_true = np.minimum.accumulate(FDR_true)
    QVAL_true = np.clip(QVAL_true, 0, 1)

    # ── Mix-Max FDR ───────────────────────────────────────────────────────
    pi0 = 0.0

    sorted_decoys            = np.sort(decoy_scores)
    unique_z_vals, counts_z  = np.unique(decoy_scores, return_counts=True)
    n_unique_z               = len(unique_z_vals)

    counts_w_leq_z = np.searchsorted(pred_scores_sorted, unique_z_vals, side='left')
    counts_z_leq_z = np.searchsorted(sorted_decoys,      unique_z_vals, side='left')

    P_W_leq_z = (counts_w_leq_z - pi0 * counts_z_leq_z) / ((1 - pi0) * n_samples)
    P_W_leq_z = np.clip(P_W_leq_z, 0, 1)
    P_Y_leq_z = counts_z_leq_z / n_samples
    P_Y_leq_z = np.clip(P_Y_leq_z, 0, 1)

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
        if z_idx >= n_unique_z:
            F_1 = 0.0
        else:
            F_1 = (1 - pi0) * np.sum(R_j[z_idx:] * counts_z[z_idx:])
        fdr_values[i] = (F_0 + F_1) / D if D > 0 else 0

    fdr_values  = np.clip(fdr_values, 0, 1)
    QVAL_mixmax = np.minimum.accumulate(fdr_values[::-1])
    QVAL_mixmax = np.clip(QVAL_mixmax, 0, 1)

    # ── Figure 2: FDR vs score threshold ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pred_scores_sorted, QVAL_mixmax, label='Mix-Max FDR')
    ax.plot(pred_scores_sorted, QVAL_true,   label='True FDR')
    ax.set_xlabel('Score threshold')
    ax.set_ylabel('q-value (FDR)')
    ax.set_title(f'{tile_name[:30]} — FDR vs Score Threshold')
    ax.legend(); ax.grid()
    plt.tight_layout()
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_fdr_vs_score.png', dpi=150)
    plt.savefig(f'BCSS/figures/diagnostics/{safe_name}_fdr_vs_score.pdf')
    plt.show(); plt.close()

    # ── TDC FDR ───────────────────────────────────────────────────────────
    TDC       = (pred_scores > decoy_scores).astype(int)
    TDC_score = np.maximum(pred_scores, decoy_scores)

    tdc_sort_idx = np.argsort(TDC_score)
    TDC_score_s  = TDC_score[tdc_sort_idx]
    TDC_label_s  = TDC[tdc_sort_idx]

    FD_CF_tdc = np.cumsum((1 - TDC_label_s)[::-1])[::-1]
    D_CF_tdc  = np.arange(0, n_samples)[::-1] + 1 - FD_CF_tdc
    D_CF_tdc  = np.maximum(D_CF_tdc, 1)
    TDC_FDR   = FD_CF_tdc / D_CF_tdc
    TDC_FDR   = np.clip(TDC_FDR, 0, 1)
    QVAL_TDC  = np.minimum.accumulate(TDC_FDR)
    QVAL_TDC  = np.clip(QVAL_TDC, 0, 1)

    # ── Acc estimation ─────────────────────────────────────────────────────
    pi0_tdc = np.clip(float(QVAL_TDC[0]),   0.0, 1.0)
    pi0_mm  = np.clip(float(QVAL_mixmax[0]), 0.0, 1.0)

    Acc_est    = np.zeros(n_samples)
    Acc_est_MM = np.zeros(n_samples)
    Acc_true   = np.zeros(n_samples)

    for i in range(n_samples):
        TP_true     = correct_pred_s[i:].sum()
        TN_true     = (1 - correct_pred_s[:i]).sum()
        Acc_true[i] = (TP_true + TN_true) / n_samples

        accepted       = n_samples - i
        FP_tdc         = accepted * QVAL_TDC[i]
        TP_tdc         = accepted * (1 - QVAL_TDC[i])
        TN_tdc         = n_samples * pi0_tdc - FP_tdc
        Acc_est[i]     = np.clip((TP_tdc + TN_tdc) / n_samples, 0.0, 1.0)

        FP_mm          = accepted * QVAL_mixmax[i]
        TP_mm          = accepted * (1 - QVAL_mixmax[i])
        TN_mm          = n_samples * pi0_mm - FP_mm
        Acc_est_MM[i]  = np.clip((TP_mm + TN_mm) / n_samples, 0.0, 1.0)

    Acc_true = np.clip(Acc_true, 0.0, 1.0)

    # ── Figure 3: Acc & FDR vs normalised rank ────────────────────────────
    normalized_rank = np.arange(n_samples) / n_samples

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(normalized_rank, QVAL_TDC,    label='TDC FDR')
    ax.plot(normalized_rank, QVAL_mixmax, label='Mix-Max FDR')
    ax.plot(normalized_rank, QVAL_true,   label='True FDR')
    ax.plot(normalized_rank, Acc_true,    label='True Acc')
    ax.plot(normalized_rank, Acc_est,     label='Est Acc with TDC')
    ax.plot(normalized_rank, Acc_est_MM,  label='Est Acc with Mix-Max')
    ax.set_xlabel('Normalized rank (fraction accepted)')
    ax.set_ylabel('Value')
    ax.set_title(f'{tile_name[:30]}')
    ax.legend(); ax.grid()
    plt.tight_layout()
    plt.savefig(f'BCSS/figures/accuracy/{safe_name}_acc_fdr.png', dpi=150)
    plt.savefig(f'BCSS/figures/accuracy/{safe_name}_acc_fdr.pdf')
    plt.show(); plt.close()

    # ── ACC_ST / ACC_TA metrics ────────────────────────────────────────────
    acc_st_true    = Acc_true[0]
    acc_ta_true    = Acc_true.max()

    acc_st_est_tdc = Acc_est[0]
    acc_ta_est_tdc = Acc_est.max()

    acc_st_est_mm  = Acc_est_MM[0]
    acc_ta_est_mm  = Acc_est_MM.max()

    err_st_tdc = abs(acc_st_est_tdc - acc_st_true)
    err_ta_tdc = abs(acc_ta_est_tdc - acc_ta_true)
    err_st_mm  = abs(acc_st_est_mm  - acc_st_true)
    err_ta_mm  = abs(acc_ta_est_mm  - acc_ta_true)

    baseline_errors_ta = {}
    for method_name in BASELINE_METHODS:
        baseline_errors_ta[method_name] = abs(
            baseline_estimates[method_name] - acc_ta_true)

    print(f"\n  {'':28} {'ACC_ST':>10}  {'ACC_TA':>10}")
    print(f"  {'─'*52}")
    print(f"  {'True':<28} {acc_st_true:>10.4f}  {acc_ta_true:>10.4f}")
    print(f"  {'TDC  (est | err)':<28} "
          f"{acc_st_est_tdc:>6.4f}  {err_st_tdc:>+.4f}  "
          f"{acc_ta_est_tdc:>6.4f}  {err_ta_tdc:>+.4f}")
    print(f"  {'Mix-Max (est | err)':<28} "
          f"{acc_st_est_mm:>6.4f}  {err_st_mm:>+.4f}  "
          f"{acc_ta_est_mm:>6.4f}  {err_ta_mm:>+.4f}")

    # ── Собираем результаты ────────────────────────────────────────────────
    row = dict(
        tile           = tile_name,
        n_pixels       = n_samples,
        true_acc       = true_acc_full,
        acc_st_true    = acc_st_true,
        acc_ta_true    = acc_ta_true,
        acc_st_tdc     = acc_st_est_tdc,
        acc_ta_tdc     = acc_ta_est_tdc,
        err_st_tdc     = err_st_tdc,
        err_ta_tdc     = err_ta_tdc,
        acc_st_mm      = acc_st_est_mm,
        acc_ta_mm      = acc_ta_est_mm,
        err_st_mm      = err_st_mm,
        err_ta_mm      = err_ta_mm,
    )
    for method_name in BASELINE_METHODS:
        row[f'est_{method_name}']    = baseline_estimates[method_name]
        row[f'err_{method_name}']    = baseline_errors[method_name]
        row[f'err_ta_{method_name}'] = baseline_errors_ta[method_name]

    all_results.append(row)


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv('BCSS/figures/comparison/shift_flow_results.csv', index=False)
    print(f"\n✓ results.csv сохранён  ({len(df)} тайлов)")

    print("\n" + "="*70)
    print("AGGREGATE SUMMARY  (mean ± std across tiles)")
    print("="*70)
    print(f"\n  {'Metric':<35} {'Mean':>8}   {'Std':>8}   {'Type':>6}")
    print(f"  {'─'*60}")

    acc_report = [
        ('acc_st_true', 'True ACC_ST',       ''),
        ('acc_ta_true', 'True ACC_TA',        ''),
        ('acc_st_tdc',  'ENPE    est ACC_ST', 'ST'),
        ('acc_ta_tdc',  'ENPE    est ACC_TA', 'TA'),
        ('acc_st_mm',   'ENPE-TA est ACC_ST', 'ST'),
        ('acc_ta_mm',   'ENPE-TA est ACC_TA', 'TA'),
    ]
    for col, name, kind in acc_report:
        v = df[col].values
        print(f"  {name:<35} {v.mean():>8.4f}   {v.std():>8.4f}   {kind:>6}")

    print(f"  {'─'*60}")
    for method_name in BASELINE_METHODS:
        col  = f'est_{method_name}'
        name = f'{method_name} est ACC_ST'
        v    = df[col].values
        print(f"  {name:<35} {v.mean():>8.4f}   {v.std():>8.4f}   {'ST':>6}")

    print(f"\n  {'─'*60}")
    print(f"  ERRORS")
    print(f"  {'─'*60}")

    fdr_report = [
        ('err_st_tdc', '|err| ENPE    vs ACC_ST'),
        ('err_ta_tdc', '|err| ENPE    vs ACC_TA'),
        ('err_st_mm',  '|err| ENPE-TA vs ACC_ST'),
        ('err_ta_mm',  '|err| ENPE-TA vs ACC_TA'),
    ]
    for col, name in fdr_report:
        v = df[col].values
        print(f"  {name:<35} {v.mean():>8.4f}   {v.std():>8.4f}")

    print(f"  {'─'*60}")
    for method_name in BASELINE_METHODS:
        v_st = df[f'err_{method_name}'].values
        v_ta = df[f'err_ta_{method_name}'].values
        print(f"  {'|err| ' + method_name + ' vs ACC_ST':<35} "
              f"{v_st.mean():>8.4f}   {v_st.std():>8.4f}")
        print(f"  {'|err| ' + method_name + ' vs ACC_TA':<35} "
              f"{v_ta.mean():>8.4f}   {v_ta.std():>8.4f}")

    print(f"\n  Full results table:")
    print(df.to_string(index=False, float_format='{:.4f}'.format))

    # =========================================================================
    # FIGURE A: Bar chart — mean absolute error per method
    # =========================================================================
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
    means_bar = [df[c].mean() for c in method_cols_bar]
    stds_bar  = [df[c].std()  for c in method_cols_bar]
    colors_bar = (
        ['#2196F3', '#1565C0', '#FF9800', '#E65100'] +
        ['#4CAF50'] * len(BASELINE_METHODS) +
        ['#388E3C'] * len(BASELINE_METHODS)
    )

    fig, ax = plt.subplots(figsize=(max(8, len(method_names_bar) * 0.9), 5))
    x_pos = np.arange(len(method_names_bar))
    ax.bar(x_pos, means_bar, yerr=stds_bar,
           color=colors_bar, capsize=4, alpha=0.85,
           edgecolor='black', linewidth=1,
           error_kw=dict(elinewidth=1.2, ecolor='black'))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names_bar, rotation=35, ha='right')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Method')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(bottom=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', alpha=0.85, label='ENPE (ST)'),
        Patch(facecolor='#1565C0', alpha=0.85, label='ENPE (TA)'),
        Patch(facecolor='#FF9800', alpha=0.85, label='ENPE-TA (ST)'),
        Patch(facecolor='#E65100', alpha=0.85, label='ENPE-TA (TA)'),
        Patch(facecolor='#4CAF50', alpha=0.85, label='Baseline (ST)'),
        Patch(facecolor='#388E3C', alpha=0.85, label='Baseline (TA)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    plt.savefig('BCSS/figures/comparison/shift_summary_bar_mean_error.png', dpi=150)
    plt.savefig('BCSS/figures/comparison/shift_summary_bar_mean_error.pdf')
    plt.show(); plt.close()
    print("✓ summary_bar_mean_error")

    # =========================================================================
    # FIGURE B: Scatter — estimated vs true accuracy
    # =========================================================================
    scatter_methods = []

    scatter_methods.append(dict(
        label='ENPE',      col_est='acc_st_tdc', col_true='acc_st_true',
        panel=0, marker='o', color='#2196F3', zorder=4))
    scatter_methods.append(dict(
        label='ENPE-TA',   col_est='acc_st_mm',  col_true='acc_st_true',
        panel=0, marker='s', color='#FF9800', zorder=4))
    scatter_methods.append(dict(
        label='ENPE',      col_est='acc_ta_tdc', col_true='acc_ta_true',
        panel=1, marker='o', color='#2196F3', zorder=4))
    scatter_methods.append(dict(
        label='ENPE-TA',   col_est='acc_ta_mm',  col_true='acc_ta_true',
        panel=1, marker='s', color='#FF9800', zorder=4))

    baseline_colors  = plt.cm.tab10(np.linspace(0, 0.9, len(BASELINE_METHODS)))
    baseline_markers = ['D', '^', 'v', 'P', 'X', '*', 'h', '8']
    for idx, method_name in enumerate(BASELINE_METHODS):
        c = baseline_colors[idx]
        m = baseline_markers[idx % len(baseline_markers)]
        scatter_methods.append(dict(
            label=method_name, col_est=f'est_{method_name}',
            col_true='acc_st_true', panel=0,
            marker=m, color=c, zorder=3))
        scatter_methods.append(dict(
            label=method_name, col_est=f'est_{method_name}',
            col_true='acc_ta_true', panel=1,
            marker=m, color=c, zorder=3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panel_titles = ['ACC$_{ST}$', 'ACC$_{TA}$']

    for panel_idx, ax in enumerate(axes):
        methods_in_panel = [sm for sm in scatter_methods
                            if sm['panel'] == panel_idx]
        all_vals = []
        for sm in methods_in_panel:
            all_vals.extend(df[sm['col_true']].values.tolist())
            all_vals.extend(df[sm['col_est']].values.tolist())

        lo = min(all_vals) - 0.02
        hi = max(all_vals) + 0.02
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5,
                label='x=y', zorder=2)

        plotted_labels = set()
        for sm in methods_in_panel:
            lbl = (sm['label'] if sm['label'] not in plotted_labels
                   else '_nolegend_')
            ax.scatter(
                df[sm['col_true']].values,
                df[sm['col_est']].values,
                label=lbl,
                marker=sm['marker'],
                color=sm['color'],
                s=60, alpha=0.85,
                zorder=sm['zorder']
            )
            plotted_labels.add(sm['label'])

        ax.set_xlabel('True Accuracy')
        ax.set_ylabel('Estimated Accuracy')
        ax.set_title(panel_titles[panel_idx])
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(linestyle='--', alpha=0.4)
        ax.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig('BCSS/figures/comparison/shift_scatter_estimated_vs_true_acc.png',
                dpi=150)
    plt.savefig('BCSS/figures/comparison/shift_scatter_estimated_vs_true_acc.pdf')
    plt.show(); plt.close()
    print("✓ scatter_estimated_vs_true_acc")

    # =========================================================================
    # FIGURE C: True ACC_ST vs True ACC_TA
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    lo = min(df['acc_st_true'].min(), df['acc_ta_true'].min()) - 0.02
    hi = max(df['acc_st_true'].max(), df['acc_ta_true'].max()) + 0.02

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y')
    ax.scatter(
        df['acc_st_true'].values,
        df['acc_ta_true'].values,
        color='steelblue', s=70, alpha=0.85, zorder=3
    )
    ax.set_xlabel('True ACC ST')
    ax.set_ylabel('True ACC TA')
    ax.set_title('True ACC_ST vs ACC_TA (per tile)')
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig('BCSS/figures/comparison/shift_scatter_true_accst_vs_accta.png',
                dpi=150)
    plt.savefig('BCSS/figures/comparison/shift_scatter_true_accst_vs_accta.pdf')
    plt.show(); plt.close()
    print("✓ scatter_true_accst_vs_accta")


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

    for i in range(NUM_CLASSES):
        plot_score_distribution_with_decoys(
            train_scores[:, i],
            train_labels,
            ms[:, i],
            ls,
            ds_decoy[:, i],
            filename = f"BCSS/figures/diagnostics/shift_scores_dist_class{i}",
            title    = f"Score Distribution with Decoys — Class {i}",
            xlim     = (-10, 10),
            show_kde = True,
            class_id = i,
        )
    print(f"✓ Score distributions сохранены для {NUM_CLASSES} классов")

except Exception as e:
    print(f"✗ Ошибка при построении distributions: {e}")