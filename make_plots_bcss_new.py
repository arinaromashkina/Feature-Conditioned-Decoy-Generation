import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from tqdm import tqdm
import os
import glob

from data_processing.score_feature_dataset import (
    ScoreFeatureDataset,
    create_score_feature_dataset_bcss,
)
from flows.shift_flow import ScoreShiftFlowWrapper
from utils.visualize_distributions import plot_score_distribution_with_decoys

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES    = 5
FEATURE_DIM    = 64
PATH_DATA      = '../../data/BCSS/training/bcss.mini.training.torch'
TEST_FOLDER    = '../../data/BCSS/test/'
FLOWS_PATH     = 'BCSS/bcss_score_shift_flow_new.pth'
SUBSAMPLE_STEP = 10
MIN_PIXELS     = 50

os.makedirs('BCSS/figures/accuracy',    exist_ok=True)
os.makedirs('BCSS/figures/diagnostics', exist_ok=True)
os.makedirs('BCSS/figures/comparison',  exist_ok=True)

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 10, 'figure.titlesize': 18,
})
print(f"Device: {DEVICE}")


# ============================================================================
# ДАННЫЕ
# ============================================================================

def get_scores_from_ds(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    cnn_list, lbl_list = [], []
    with torch.no_grad():
        for cnn_scores, features, target_decoy, labels in loader:
            cnn_list.append(cnn_scores.cpu().numpy())
            lbl_list.append(labels.cpu().numpy())
    return np.concatenate(cnn_list), np.concatenate(lbl_list)


def load_bcss_test_file(fpath):
    data        = torch.load(fpath, weights_only=False)
    predictions = torch.flatten(data['predictions'], start_dim=2).squeeze(0).T
    features    = torch.flatten(data['features'],    start_dim=2).squeeze(0).T
    labels      = torch.flatten(torch.tensor(data['mask']), start_dim=0)
    labels      = torch.where(labels <= 3, labels, torch.tensor(4))
    return ScoreFeatureDataset(predictions, features, predictions, labels)


# ============================================================================
# BASELINES
# ============================================================================

def _to_tensor(x):
    return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x


def negentropy(logits):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    probs   = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
    return np.log(logits.shape[1]) - entropy


def predict_ATC_maxconf(source_logits, source_labels, target_logits):
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    src_scores    = torch.softmax(source_logits, dim=1).amax(1)
    tgt_scores    = torch.softmax(target_logits, dim=1).amax(1)
    n_correct     = (source_logits.argmax(1) == source_labels).sum()
    threshold     = torch.sort(src_scores)[0][-n_correct]
    return (tgt_scores > threshold).float().mean().item()


def predict_ATC_negent(source_logits, source_labels, target_logits):
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    src_scores    = negentropy(source_logits)
    tgt_scores    = negentropy(target_logits)
    n_correct     = (source_logits.argmax(1) == source_labels).sum()
    threshold     = torch.sort(src_scores)[0][-n_correct]
    return (tgt_scores > threshold).float().mean().item()


def predict_AC(source_logits, source_labels, target_logits):
    return _to_tensor(target_logits).softmax(dim=1).amax(1).mean().item()


def predict_DOC(source_logits, source_labels, target_logits):
    source_logits = _to_tensor(source_logits)
    source_labels = _to_tensor(source_labels).long()
    target_logits = _to_tensor(target_logits)
    src_conf = torch.softmax(source_logits, dim=1).amax(1).mean().item()
    tgt_conf = torch.softmax(target_logits, dim=1).amax(1).mean().item()
    src_acc  = (source_logits.argmax(1) == source_labels).float().mean().item()
    return src_acc + (tgt_conf - src_conf)


try:
    import ot
    def predict_COT(source_logits, source_labels, target_logits):
        source_logits  = _to_tensor(source_logits)
        source_labels  = _to_tensor(source_labels).long()
        target_logits  = _to_tensor(target_logits)
        num_classes    = source_logits.shape[1]
        src_label_dist = torch.nn.functional.one_hot(
            source_labels, num_classes).float().mean(0)
        target_probs   = torch.softmax(target_logits, dim=1)
        cost_matrix    = torch.stack([
            (target_probs - onehot).abs().sum(1)
            for onehot in torch.eye(num_classes)
        ], dim=1) / 2
        ot_plan = ot.emd(
            np.ones(len(target_probs)) / len(target_probs),
            src_label_dist.cpu().numpy(),
            cost_matrix.cpu().numpy()
        )
        ot_cost  = np.sum(ot_plan * cost_matrix.cpu().numpy())
        s_conf   = torch.softmax(source_logits, dim=1).amax(1).mean().item()
        s_acc    = (source_logits.argmax(1) == source_labels).float().mean().item()
        return 1.0 - (ot_cost + s_conf - s_acc)
    COT_AVAILABLE = True
    print("✓ COT available")
except ImportError:
    COT_AVAILABLE = False
    print("⚠ COT unavailable")

BASELINE_METHODS = {
    'ATC'   : predict_ATC_maxconf,
    'ATC-NE': predict_ATC_negent,
    'AC'    : predict_AC,
    'DOC'   : predict_DOC,
}
if COT_AVAILABLE:
    BASELINE_METHODS['COT'] = predict_COT

print(f"Baselines: {list(BASELINE_METHODS.keys())}")


# ============================================================================
# EMPIRICAL DECOYS — ИСПРАВЛЕННАЯ ВЕРСИЯ
#
# Теоретическое обоснование:
# --------------------------
# Нулевая гипотеза H₀(t): предсказанный класс ĉ = argmax f(t) НЕВЕРНЫЙ.
#
# Decoy должен симулировать:
#   P(score_ĉ(t) | предсказание неверное)
#
# Эмпирическая оценка из train:
#   Для класса c: берём score_c на примерах где
#     (a) модель предсказала c  [argmax = c]
#     (b) label ≠ c             [предсказание неверное]
#   → это и есть распределение скора под H₀
#
# Почему это правильно:
#   - target f(t) = score_ĉ(t) на test
#   - decoy Z(t) ~ P(score_ĉ | ошибка класса ĉ) из train
#   - При H₀ (t неверно классифицирован) f(t) и Z(t)
#     имеют одинаковое распределение → exchangeability ✓
# ============================================================================

def build_error_conditioned_pools(train_scores: np.ndarray,
                                   train_labels: np.ndarray,
                                   num_classes: int,
                                   verbose: bool = True) -> dict:
    """
    Строит пулы decoy-скоров для каждого класса c:
      pool[c] = score_c на примерах где argmax=c И label≠c
                (т.е. модель предсказала c, но ошиблась)

    Если таких примеров мало — fallback на score_c | label≠c
    (менее точно, но работает при малом числе ошибок на train).

    Parameters
    ----------
    train_scores : (N, C) logit/softmax scores из train
    train_labels : (N,)   истинные метки
    num_classes  : C
    verbose      : печать статистики

    Returns
    -------
    pools : dict[int → np.ndarray]  decoy скоры для каждого класса
    """
    pred_classes = train_scores.argmax(axis=1)
    pools        = {}
    MIN_POOL     = 30  # минимальный размер пула

    if verbose:
        print("\nBuilding error-conditioned decoy pools:")
        print(f"  Train accuracy: "
              f"{(pred_classes == train_labels).mean():.4f}")

    for c in range(num_classes):
        # Основной пул: argmax=c И label≠c (истинные ошибки класса c)
        error_mask = (pred_classes == c) & (train_labels != c)
        n_errors   = error_mask.sum()

        if n_errors >= MIN_POOL:
            pools[c] = train_scores[error_mask, c]
            source   = f"errors (argmax=c, label≠c)"
        else:
            # Fallback: label≠c (все негативные примеры класса c)
            # Менее точно но состоятельно при достаточном overlap
            neg_mask = train_labels != c
            pools[c] = train_scores[neg_mask, c]
            source   = f"fallback (label≠c)"

        if verbose:
            p = pools[c]
            print(f"  Class {c}: n={len(p):5d}  "
                  f"[{p.min():.3f}, {p.max():.3f}]  "
                  f"mean={p.mean():.3f}  source={source}  "
                  f"n_errors={n_errors}")

    return pools


def generate_decoys(model_scores: np.ndarray,
                    neg_pools:    dict,
                    seed:         int = 42) -> np.ndarray:
    """
    Генерирует decoy вектор для каждого примера.

    Для примера i с предсказанным классом ĉ:
      decoy[i, ĉ] ~ pool[ĉ]   (из error-conditioned pool)
      decoy[i, j] = model_scores[i, j]  для j ≠ ĉ

    Логика: Mix-Max использует только Z_i = decoy[i, pred_class],
    поэтому только координата предсказанного класса критична.
    Остальные координаты можно оставить как есть.

    Parameters
    ----------
    model_scores : (n, C) скоры модели на test
    neg_pools    : dict из build_error_conditioned_pools
    seed         : random seed

    Returns
    -------
    decoys : (n, C) decoy скоры
    """
    rng          = np.random.default_rng(seed)
    n, C         = model_scores.shape
    pred_classes = model_scores.argmax(axis=1)
    decoys       = model_scores.copy()  # начинаем с копии

    for c in range(C):
        mask = pred_classes == c
        if mask.sum() == 0:
            continue
        pool = neg_pools[c]
        if len(pool) == 0:
            # крайний fallback: uniform на диапазоне скоров
            lo, hi = model_scores[:, c].min(), model_scores[:, c].max()
            decoys[mask, c] = rng.uniform(lo, hi, size=mask.sum())
        else:
            decoys[mask, c] = rng.choice(pool, size=mask.sum(), replace=True)

    return decoys


def verify_exchangeability(model_scores:  np.ndarray,
                            decoy_scores:  np.ndarray,
                            labels:        np.ndarray,
                            verbose:       bool = True) -> dict:
    """
    Проверяет выполнение exchangeability:
    Под H₀ (неверные предсказания) f(t) и Z(t) должны иметь
    одинаковое распределение.

    KS-тест: если p >> 0.05 → exchangeability не отвергается ✓
    """
    n            = len(model_scores)
    pred_classes = model_scores.argmax(axis=1)
    f_t          = model_scores[np.arange(n), pred_classes]
    Z            = decoy_scores[np.arange(n), pred_classes]
    is_wrong     = pred_classes != labels

    result = {}

    # KS-тест на неверных предсказаниях
    if is_wrong.sum() >= 10:
        ks_stat, ks_pval = ks_2samp(f_t[is_wrong], Z[is_wrong])
        result['ks_stat_wrong'] = ks_stat
        result['ks_pval_wrong'] = ks_pval
        exchangeable = ks_pval > 0.05
    else:
        ks_stat, ks_pval = np.nan, np.nan
        exchangeable = None

    # Win rate: P(f(t) > Z) должен быть > 0.5 для верных предсказаний
    # и ≈ 0.5 для неверных (под H₀)
    win_rate_correct = (f_t[~is_wrong] > Z[~is_wrong]).mean() \
                       if (~is_wrong).sum() > 0 else np.nan
    win_rate_wrong   = (f_t[is_wrong]  > Z[is_wrong]).mean()  \
                       if is_wrong.sum() > 0 else np.nan

    result.update({
        'ks_stat': ks_stat, 'ks_pval': ks_pval,
        'exchangeable': exchangeable,
        'win_rate_correct': win_rate_correct,
        'win_rate_wrong':   win_rate_wrong,
        'n_wrong': int(is_wrong.sum()),
        'n_correct': int((~is_wrong).sum()),
    })

    if verbose:
        print(f"\n  Exchangeability check:")
        print(f"    KS test (wrong preds):  "
              f"stat={ks_stat:.4f}  p={ks_pval:.4f}  "
              f"OK={exchangeable}")
        print(f"    Win rate correct: {win_rate_correct:.4f}  "
              f"(should be >> 0.5)")
        print(f"    Win rate wrong:   {win_rate_wrong:.4f}  "
              f"(should be ≈ 0.5 under H₀)")

    return result


# ============================================================================
# MIX-MAX FDR
#
# Для примера t с предсказанным классом ĉ:
#   f(t) = model_scores[t, ĉ]   (target score)
#   Z(t) = decoy_scores[t, ĉ]   (decoy score ~ H₀)
#   g(t) = max(f(t), Z(t))      (mix-max score)
#
# FDP_MM(s) = Σ_{Z_j > s} P(f ≤ Z_j) / P(g ≤ Z_j)  /  #{f(t) > s}
#
# π₀ = FDP_MM(-∞) = доля неверных предсказаний (оценивается label-free)
# ============================================================================

def calculate_mixmax_qvalues(model_scores: np.ndarray,
                              decoy_scores: np.ndarray,
                              verbose:      bool = True) -> np.ndarray:
    """
    Вычисляет Mix-Max q-values.

    Параметризация: сортировка по f(t) убывая — это стандарт для FDR,
    где порог по скору определяет список принятых гипотез.

    Returns
    -------
    q_values : (n,) q-value для каждого примера
               (монотонны в порядке убывания f(t))
    """
    n            = len(model_scores)
    pred_classes = model_scores.argmax(axis=1)
    f_t          = model_scores[np.arange(n), pred_classes]
    Z            = decoy_scores[np.arange(n), pred_classes]
    g_t          = np.maximum(f_t, Z)

    if verbose:
        overlap = ((Z >= f_t.min()) & (Z <= f_t.max())).mean()
        print(f"  f(t): [{f_t.min():.3f}, {f_t.max():.3f}]  "
              f"mean={f_t.mean():.3f}")
        print(f"  Z   : [{Z.min():.3f}, {Z.max():.3f}]  "
              f"mean={Z.mean():.3f}")
        print(f"  Z in f(t) range: {overlap:.3f}  "
              f"target wins: {(f_t > Z).mean():.3f}")

    # Эмпирические CDF для R_j = P(f ≤ z) / P(g ≤ z)
    sorted_f = np.sort(f_t)
    sorted_g = np.sort(g_t)
    unique_Z, counts_Z = np.unique(Z, return_counts=True)

    P_F = np.searchsorted(sorted_f, unique_Z, side='right') / n
    P_G = np.searchsorted(sorted_g, unique_Z, side='right') / n

    with np.errstate(divide='ignore', invalid='ignore'):
        R_j = np.where(
            (P_G > 0) & (P_F > 0),
            np.clip(P_F / P_G, 0.0, 1.0),
            0.0
        )

    # Сортировка по f(t) убывая — стандартный порядок для FDR
    order         = np.argsort(f_t)[::-1]
    sorted_f_desc = f_t[order]
    n_unique      = len(unique_Z)
    fdp_values    = np.zeros(n)

    for i, s_th in enumerate(sorted_f_desc):
        D         = i + 1                                   # #{f(t) ≥ s_th}
        # Сумма по Z_j > s_th (decoys превышающие текущий порог)
        idx_start = np.searchsorted(unique_Z, s_th, side='right')
        numerator = float(
            np.sum(R_j[idx_start:] * counts_Z[idx_start:])
        ) if idx_start < n_unique else 0.0
        fdp_values[i] = min(numerator / D, 1.0)

    # Монотонизация: q(i) = min(fdp(i), fdp(i+1), ...) по убыванию f(t)
    q_desc         = np.minimum.accumulate(fdp_values[::-1])[::-1]
    final_q        = np.empty(n)
    final_q[order] = q_desc

    if verbose:
        print(f"  FDP range: [{final_q.min():.4f}, {final_q.max():.4f}]")

    return final_q


def _compute_gt_qvalues(pred_classes: np.ndarray,
                         f_t:          np.ndarray,
                         labels:       np.ndarray) -> np.ndarray:
    """Ground-truth FDP (требует меток — только для валидации метода)."""
    n     = len(f_t)
    order = np.argsort(f_t)[::-1]
    n_incorrect, fdp_gt = 0, np.zeros(n)
    for rank, idx in enumerate(order):
        if pred_classes[idx] != labels[idx]:
            n_incorrect += 1
        fdp_gt[rank] = n_incorrect / (rank + 1)
    # Монотонизация
    for i in range(n - 2, -1, -1):
        fdp_gt[i] = min(fdp_gt[i], fdp_gt[i + 1])
    final_q        = np.empty(n)
    final_q[order] = fdp_gt
    return final_q


def control_fdr_mixmax(model_scores:  np.ndarray,
                        target_labels: np.ndarray,
                        decoy_scores:  np.ndarray,
                        verbose:       bool = True) -> pd.DataFrame:
    n            = len(model_scores)
    pred_classes = model_scores.argmax(axis=1)
    f_t          = model_scores[np.arange(n), pred_classes]

    df = pd.DataFrame({
        'original_index'  : np.arange(n),
        'label'           : target_labels,
        'predicted_class' : pred_classes,
        'pred_class_score': f_t,
    })
    df['q_values_mixmax']       = calculate_mixmax_qvalues(
        model_scores, decoy_scores, verbose=verbose)
    df['q_values_ground_truth'] = _compute_gt_qvalues(
        pred_classes, f_t, target_labels)
    return df


# ============================================================================
# ESTIMATION CURVE — ИСПРАВЛЕННАЯ ВЕРСИЯ
#
# Ключевые исправления:
# 1. Обе кривые (est и true) параметризованы по f(t) убывая — единая ось
# 2. π₀ = FDP_MM(-∞) = последнее q-value (все приняты) = доля ошибок
# 3. Явная проверка: TN ≥ 0 (если нет — предупреждение, не заглушка)
# 4. ACC_ST = ACC при s_th → -∞ = все приняты
# 5. ACC_TA = max ACC_est по f(t) порогам
# ============================================================================

def compute_method_estimation_curve(df: pd.DataFrame,
                                     q_col: str) -> pd.DataFrame:
    """
    Строит кривую ACC(s_th) в единой параметризации по f(t).

    Параметризация:
      - Сортировка по pred_class_score УБЫВАЯ
      - i-я точка: порог s_th = f(t_i), принимаем топ-i примеров
      - n_discoveries = i (число принятых)

    Returns
    -------
    df_curve : DataFrame с колонками:
        pred_class_score    — текущий порог (f(t) убывает)
        q_value_method      — q-value Mix-Max при этом пороге
        n_discoveries       — число принятых
        FP_est, TP_est      — estimated confusion
        TN_est, FN_est      — estimated confusion
        Accuracy_est        — estimated accuracy
        Accuracy_true_at_threshold — true accuracy (для валидации)
        tn_negative         — флаг: TN < 0 (диагностика)
    """
    total    = len(df)
    true_pi0 = 1.0 - float((df['predicted_class'] == df['label']).mean())
    true_acc_full = 1.0 - true_pi0

    df_m = df[~df[q_col].isna()].copy()
    if len(df_m) == 0:
        return pd.DataFrame()

    # ── Единая сортировка по f(t) убывая ─────────────────────────────────
    # Это обеспечивает что обе кривые (est и true) параметризованы одинаково
    df_m = df_m.sort_values('pred_class_score',
                             ascending=False).reset_index(drop=True)
    df_m['n_discoveries'] = np.arange(1, len(df_m) + 1)

    # ── π₀ = FDP_MM(-∞): q-value когда приняты ВСЕ примеры ──────────────
    # Это label-free оценка доли ошибок классификатора
    # Физический смысл: при s_th → -∞ принимаем всё,
    # FDP = доля неверных среди всех = 1 - ACC
    pi0_est = float(np.clip(df_m[q_col].iloc[-1], 0.0, 1.0))

    print(f"  π₀_est  = FDP_MM(-∞) = {pi0_est:.4f}  "
          f"[label-free estimate of error rate]")
    print(f"  π₀_true = true error = {true_pi0:.4f}  "
          f"[ground truth, for validation only]")
    print(f"  Δπ₀ = {abs(pi0_est - true_pi0):.4f}")

    # ── Estimated confusion matrix ────────────────────────────────────────
    # FP(s_th) = D * FDP(s_th)       где D = n_discoveries
    # TP(s_th) = D - FP(s_th)        = D * (1 - FDP)
    # TN(s_th) = |T| * π₀ - FP(s_th)
    # FN(s_th) = |T| * (1-π₀) - TP(s_th)
    #
    # Заметим: TP + FP + TN + FN = D + (|T|-D) = |T| ✓
    # TN ≥ 0 ⟺ FP ≤ |T|*π₀ ⟺ D*FDP ≤ |T|*π₀
    # При хорошем decoy это должно выполняться для всех s_th
    df_m['FP_est'] = df_m['n_discoveries'] * df_m[q_col]
    df_m['TP_est'] = df_m['n_discoveries'] - df_m['FP_est']
    df_m['TN_est'] = total * pi0_est - df_m['FP_est']
    df_m['FN_est'] = total * (1.0 - pi0_est) - df_m['TP_est']

    # Диагностика: фиксируем нарушения, НЕ заглушаем
    df_m['tn_negative'] = df_m['TN_est'] < -0.5  # допуск на float погрешность
    df_m['fn_negative'] = df_m['FN_est'] < -0.5

    n_tn_neg = df_m['tn_negative'].sum()
    n_fn_neg = df_m['fn_negative'].sum()
    if n_tn_neg > 0:
        print(f"  ⚠ TN < 0 в {n_tn_neg}/{total} точках — "
              f"decoy distribution слишком тяжёлый (FDP завышен)")
    if n_fn_neg > 0:
        print(f"  ⚠ FN < 0 в {n_fn_neg}/{total} точках — "
              f"FDP занижен или π₀ недооценён")

    # Clip только для вычисления ACC (не скрываем флаги)
    FP_c = df_m['FP_est'].clip(lower=0.0)
    TP_c = df_m['TP_est'].clip(lower=0.0)
    TN_c = df_m['TN_est'].clip(lower=0.0)
    FN_c = df_m['FN_est'].clip(lower=0.0)
    df_m['Accuracy_est'] = (TP_c + TN_c) / total

    # ── True confusion matrix — та же параметризация (по f(t) убывая) ─────
    df_m['is_correct'] = (df_m['predicted_class'] == df_m['label']).astype(int)
    df_m['TP_true']    = df_m['is_correct'].cumsum().astype(float)
    df_m['FP_true']    = df_m['n_discoveries'] - df_m['TP_true']
    df_m['TN_true']    = np.maximum(0.0, total * true_pi0 - df_m['FP_true'])
    df_m['Accuracy_true_at_threshold'] = \
        (df_m['TP_true'] + df_m['TN_true']) / total

    # ── Метаданные ────────────────────────────────────────────────────────
    df_m['true_acc_full'] = true_acc_full
    df_m['pi0_est']       = pi0_est
    df_m['true_pi0']      = true_pi0

    df_m['error_est_vs_thresh'] = np.abs(
        df_m['Accuracy_est'] - df_m['Accuracy_true_at_threshold'])
    df_m['error_est_vs_full'] = np.abs(
        df_m['Accuracy_est'] - true_acc_full)

    return df_m.rename(columns={q_col: 'q_value_method'})


def find_best_mixmax_threshold(df_curve: pd.DataFrame) -> dict:
    """
    Извлекает ключевые метрики из кривой.

    ACC_ST = ACC(-∞) = accuracy когда принимаем всё = стандартная accuracy
    ACC_TA = max ACC_est = accuracy с адаптацией порога
    """
    if len(df_curve) == 0:
        return dict(pi0_est=np.nan,
                    acc_st_est=np.nan,  acc_ta_est=np.nan,
                    acc_st_true=np.nan, acc_ta_true=np.nan,
                    err_st=np.nan,      err_ta=np.nan,
                    best_q=np.nan,      best_threshold=np.nan,
                    n_accepted=0,       frac_accepted=np.nan,
                    n_tn_negative=np.nan)

    pi0     = float(df_curve['pi0_est'].iloc[0])
    total   = len(df_curve)

    # ACC_ST: последняя строка = все приняты (s_th → -∞)
    last        = df_curve.iloc[-1]
    acc_st_est  = float(last['Accuracy_est'])
    acc_st_true = float(last['true_acc_full'])

    # ACC_TA: максимум по кривой
    best_idx    = df_curve['Accuracy_est'].idxmax()
    best        = df_curve.loc[best_idx]
    acc_ta_est  = float(best['Accuracy_est'])
    acc_ta_true = float(best['Accuracy_true_at_threshold'])

    return dict(
        pi0_est       = pi0,
        acc_st_est    = acc_st_est,
        acc_ta_est    = acc_ta_est,
        acc_st_true   = acc_st_true,
        acc_ta_true   = acc_ta_true,
        err_st        = abs(acc_st_est  - acc_st_true),
        err_ta        = abs(acc_ta_est  - acc_ta_true),
        best_q        = float(best['q_value_method']),
        best_threshold= float(best['pred_class_score']),
        n_accepted    = int(best['n_discoveries']),
        frac_accepted = float(best['n_discoveries']) / total,
        n_tn_negative = int(df_curve['tn_negative'].sum()),
    )


# ============================================================================
# ДИАГНОСТИКА
# ============================================================================

def diagnose_tile(model_scores:  np.ndarray,
                  decoy_scores:  np.ndarray,
                  labels:        np.ndarray,
                  tile_name:     str = "tile",
                  exch_result:   dict = None) -> None:
    """
    Расширенная диагностика с проверкой exchangeability.
    """
    n            = len(model_scores)
    pred_classes = model_scores.argmax(axis=1)
    f_t          = model_scores[np.arange(n), pred_classes]
    Z            = decoy_scores[np.arange(n), pred_classes]
    g_t          = np.maximum(f_t, Z)
    is_correct   = pred_classes == labels
    is_wrong     = ~is_correct

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(f'Diagnostics: {tile_name}', fontweight='bold')

    # 1. Target vs Decoy — раздельно для верных и неверных
    ax   = axes[0, 0]
    bins = np.linspace(min(f_t.min(), Z.min()), max(f_t.max(), Z.max()), 80)
    ax.hist(f_t[is_correct],  bins=bins, alpha=0.5, color='#388E3C',
            density=True, label=f'f(t) correct  n={is_correct.sum()}')
    ax.hist(f_t[is_wrong],    bins=bins, alpha=0.5, color='#E53935',
            density=True, label=f'f(t) wrong  n={is_wrong.sum()}')
    ax.hist(Z,                bins=bins, alpha=0.4, color='#1976D2',
            density=True, label=f'Decoy Z  mean={Z.mean():.2f}')
    ax.set_xlabel('Score'); ax.set_ylabel('Density')
    ax.set_title('Target (correct/wrong) vs Decoy')
    ax.legend(fontsize=8)

    # Ключевая диагностика: f(t)|wrong vs Z должны быть похожи
    if is_wrong.sum() >= 5:
        ks_stat, ks_pval = ks_2samp(f_t[is_wrong], Z[is_wrong])
        ax.set_title(f'f(t) vs Decoy\n'
                     f'KS(wrong,Z): stat={ks_stat:.3f} p={ks_pval:.3f}')

    # 2. Win rate per class — отдельно для верных/неверных
    ax = axes[0, 1]
    C  = model_scores.shape[1]
    x  = np.arange(C)
    w  = 0.35
    wr_correct, wr_wrong, ns = [], [], []
    for c in range(C):
        mc = pred_classes == c
        wrc = (f_t[mc &  is_correct] > Z[mc &  is_correct]).mean() \
              if (mc & is_correct).sum() > 0 else np.nan
        wrw = (f_t[mc & is_wrong]   > Z[mc & is_wrong]).mean()    \
              if (mc & is_wrong).sum()   > 0 else np.nan
        wr_correct.append(wrc); wr_wrong.append(wrw)
        ns.append(mc.sum())

    bars_c = ax.bar(x - w/2, wr_correct, w, color='#388E3C',
                    alpha=0.7, label='Win rate | correct')
    bars_w = ax.bar(x + w/2, wr_wrong,   w, color='#E53935',
                    alpha=0.7, label='Win rate | wrong')
    ax.axhline(0.5, color='black', ls='--', lw=1.5,
               label='H₀ benchmark (0.5)')
    ax.axhline(1.0, color='gray',  ls=':', lw=1)
    for bar, sz in zip(bars_c, ns):
        ax.text(bar.get_x() + bar.get_width()/2,
                0.02, f'n={sz}', ha='center', fontsize=7, color='white',
                fontweight='bold')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('P(f(t) > Z)')
    ax.set_title('Win Rate per Class\n(wrong ≈ 0.5 → exchangeability ✓)')
    ax.set_ylim(0, 1.15); ax.legend(fontsize=8)

    # 3. CDF — ключевая: f(t)|wrong vs Z
    ax = axes[0, 2]
    p  = np.linspace(0, 1, n)
    ax.plot(np.sort(f_t),            np.linspace(0,1,n),
            color='#388E3C', lw=2, label='CDF f(t) all')
    ax.plot(np.sort(g_t),            np.linspace(0,1,n),
            color='#7B1FA2', lw=2, ls='--', label='CDF g(t)')
    ax.plot(np.sort(Z),              np.linspace(0,1,n),
            color='#1976D2', lw=2, label='CDF Z (decoy)')
    if is_wrong.sum() >= 5:
        ax.plot(np.sort(f_t[is_wrong]),
                np.linspace(0,1,is_wrong.sum()),
                color='#E53935', lw=2.5, ls='-.',
                label='CDF f(t)|wrong  ← should ≈ CDF Z')
    ax.set_xlabel('Score'); ax.set_ylabel('CDF')
    ax.set_title('CDF: exchangeability check\nf(t)|wrong ≈ Z?')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Scatter f(t) vs Z
    ax    = axes[1, 0]
    sub   = 2000
    idx_c = np.where( is_correct)[0]
    idx_w = np.where( is_wrong)[0]
    idx_c = idx_c[np.random.choice(
        len(idx_c), min(sub, len(idx_c)), replace=False)]
    idx_w = idx_w[np.random.choice(
        len(idx_w), min(sub, len(idx_w)), replace=False)]
    ax.scatter(f_t[idx_c], Z[idx_c], alpha=0.3, s=5,
               color='#388E3C', label='Correct')
    ax.scatter(f_t[idx_w], Z[idx_w], alpha=0.3, s=5,
               color='#E53935', label='Wrong')
    lo = min(f_t.min(), Z.min()); hi = max(f_t.max(), Z.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.5,
            label='f(t) = Z')
    ax.set_xlabel('Target f(t)'); ax.set_ylabel('Decoy Z')
    ax.set_title('Scatter\nCorrect: f(t) > Z  |  Wrong: f(t) ≈ Z')
    ax.legend(fontsize=8, markerscale=3)

    # 5. R_j weights
    ax       = axes[1, 1]
    unique_Z_= np.unique(Z)
    sorted_f_= np.sort(f_t)
    sorted_g_= np.sort(g_t)
    P_F_     = np.searchsorted(sorted_f_, unique_Z_, side='right') / n
    P_G_     = np.searchsorted(sorted_g_, unique_Z_, side='right') / n
    R_j_     = np.where((P_G_ > 0) & (P_F_ > 0),
                         np.clip(P_F_ / P_G_, 0, 1), 0)
    ax.plot(unique_Z_, R_j_, color='#E53935', lw=1.5)
    ax.fill_between(unique_Z_, R_j_, alpha=0.2, color='#E53935')
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='R_j=1')
    ax.axhline(0.5, color='gray', ls=':',  lw=1, label='R_j=0.5')
    ax.set_xlabel('Decoy score Z'); ax.set_ylabel('R_j = P(f≤Z)/P(g≤Z)')
    ax.set_title('Mix-Max weights R_j\n(low Z → R_j≈1 expected)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. FDP: Mix-Max vs True (единая параметризация по f(t))
    ax     = axes[1, 2]
    order  = np.argsort(f_t)[::-1]
    n_disc = np.arange(1, n + 1)
    q_mm   = calculate_mixmax_qvalues(model_scores, decoy_scores, verbose=False)
    # Оба в порядке убывания f(t)
    ax.plot(n_disc / n, q_mm[order],  color='#E53935', lw=2,
            label='Mix-Max FDP est')
    is_wrong_sorted = (~is_correct[order]).astype(int)
    fdp_gt  = np.cumsum(is_wrong_sorted) / n_disc
    for i in range(len(fdp_gt) - 2, -1, -1):
        fdp_gt[i] = min(fdp_gt[i], fdp_gt[i + 1])
    ax.plot(n_disc / n, fdp_gt, color='black', lw=2, ls='--',
            label='True FDP')
    ax.set_xlabel('Fraction accepted (by f(t) rank)')
    ax.set_ylabel('FDP')
    ax.set_title('FDP: Mix-Max vs True\n(единая ось: f(t) убывает)')
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Консоль
    print(f"\n{'='*55}")
    print(f"DIAGNOSTICS: {tile_name}")
    print(f"{'='*55}")
    print(f"  n_pixels        : {n}")
    print(f"  true_accuracy   : {is_correct.mean():.4f}")
    print(f"  target_win_rate : {(f_t > Z).mean():.4f}  "
          f"(all, should be > 0.5)")
    if is_wrong.sum() >= 5:
        ks_s, ks_p = ks_2samp(f_t[is_wrong], Z[is_wrong])
        print(f"  KS(f|wrong, Z)  : stat={ks_s:.4f}  p={ks_p:.4f}  "
              f"exchange={'✓' if ks_p > 0.05 else '✗ WARNING'}")
    print(f"  f(t) mean/std   : {f_t.mean():.3f} / {f_t.std():.3f}")
    print(f"  Z   mean/std    : {Z.mean():.3f} / {Z.std():.3f}")
    if is_wrong.sum() > 0:
        print(f"  f(t)|wrong mean : {f_t[is_wrong].mean():.3f}  "
              f"(should ≈ Z mean = {Z.mean():.3f})")

    for ext in ('png', 'pdf'):
        plt.savefig(
            f'BCSS/figures/diagnostics/diagnose_{tile_name}.{ext}',
            dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved")


# ============================================================================
# ГРАФИК ACCURACY — ИСПРАВЛЕННАЯ ВЕРСИЯ
#
# Обе кривые параметризованы по f(t) (pred_class_score убывает)
# Ось X — q-value (монотонный по f(t)), для читаемости
# ============================================================================

def plot_accuracy_curve(df_curve:         pd.DataFrame,
                         metrics:          dict,
                         baseline_results: dict,
                         title:            str  = '',
                         save_path:        str  = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(title, fontweight='bold')

    # ── Левый: ACC vs q-value ─────────────────────────────────────────────
    ax  = axes[0]
    q   = df_curve['q_value_method'].values       # монотонен по f(t)
    est = df_curve['Accuracy_est'].values
    tru = df_curve['Accuracy_true_at_threshold'].values

    ax.plot(q, est, color='#E53935', lw=2.5,
            label='Mix-Max est (label-free)')
    ax.plot(q, tru, color='#1976D2', lw=2.0, ls='-.',
            label='True ACC @ threshold')
    ax.axhline(metrics['acc_st_true'], color='black', lw=2, ls='--',
               label=f"True ACC_ST = {metrics['acc_st_true']:.3f}")

    best_q = metrics['best_q']
    ax.axvline(best_q, color='gray', ls=':', lw=1.5)
    ax.scatter([best_q], [metrics['acc_ta_est']],
               color='#E53935', s=90, zorder=5,
               label=f"ACC_TA est={metrics['acc_ta_est']:.3f}")
    ax.scatter([best_q], [metrics['acc_ta_true']],
               color='#1976D2', s=90, zorder=5,
               label=f"ACC_TA true={metrics['acc_ta_true']:.3f}")

    bl_colors = ['#388E3C', '#0288D1', '#7B1FA2', '#F57C00', '#00838F']
    for (name, val), col in zip(baseline_results.items(), bl_colors):
        if not np.isnan(val):
            ax.axhline(val, color=col, lw=1.5, ls='--', alpha=0.7,
                       label=f'{name}={val:.3f}')

    for fdr in [0.05, 0.10, 0.20]:
        ax.axvline(fdr, color='gray', ls=':', alpha=0.25, lw=1)
        ax.text(fdr + 0.002, 0.02, f'{fdr:.2f}', fontsize=8, alpha=0.5)

    # Выделяем зоны где TN < 0 (ненадёжные оценки)
    tn_neg = df_curve['tn_negative'].values
    if tn_neg.any():
        ax.fill_between(q, 0, 1, where=tn_neg,
                        color='orange', alpha=0.15,
                        label='TN<0 (unreliable)')

    ax.set_xlabel('Q-value (= FDP threshold)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.01, min(0.55, float(q.max()) + 0.03))
    ax.legend(fontsize=7.5, framealpha=0.9, loc='lower left')
    ax.grid(True, alpha=0.3, ls='--')
    ax.set_title('ACC vs Q-value\n(единая параметризация по f(t))')

    # ── Правый: ACC vs fraction accepted ─────────────────────────────────
    ax2   = axes[1]
    frac  = df_curve['n_discoveries'].values / len(df_curve)

    ax2.plot(frac, est, color='#E53935', lw=2.5,
             label='Mix-Max est')
    ax2.plot(frac, tru, color='#1976D2', lw=2.0, ls='-.',
             label='True ACC @ threshold')
    ax2.axhline(metrics['acc_st_true'], color='black', lw=2, ls='--',
                label=f"True ACC_ST = {metrics['acc_st_true']:.3f}")

    best_frac = metrics['frac_accepted']
    ax2.axvline(best_frac, color='gray', ls=':', lw=1.5)
    ax2.scatter([best_frac], [metrics['acc_ta_est']],
                color='#E53935', s=90, zorder=5,
                label=f"ACC_TA est={metrics['acc_ta_est']:.3f}")
    ax2.scatter([best_frac], [metrics['acc_ta_true']],
                color='#1976D2', s=90, zorder=5,
                label=f"ACC_TA true={metrics['acc_ta_true']:.3f}")

    if tn_neg.any():
        ax2.fill_between(frac, 0, 1, where=tn_neg,
                         color='orange', alpha=0.15, label='TN<0 zone')

    ax2.set_xlabel('Fraction accepted (f(t) rank)', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_ylim(0, 1.05); ax2.set_xlim(0, 1.05)
    ax2.legend(fontsize=7.5, framealpha=0.9, loc='lower right')
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.set_title('ACC vs Fraction Accepted\n(f(t) убывает →)')

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        for ext in ('png', 'pdf'):
            plt.savefig(f'{save_path}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def print_metrics(met: dict, title: str = "") -> None:
    print(f"\n  [{title}]")
    print(f"  π₀  = {met['pi0_est']:.4f}")
    print(f"  ACC_ST: est={met['acc_st_est']:.4f}  "
          f"true={met['acc_st_true']:.4f}  "
          f"err={met['err_st']:.4f}")
    print(f"  ACC_TA: est={met['acc_ta_est']:.4f}  "
          f"true={met['acc_ta_true']:.4f}  "
          f"err={met['err_ta']:.4f}")
    print(f"  best_q={met['best_q']:.4f}  "
          f"thresh={met['best_threshold']:.4f}  "
          f"n_accepted={met['n_accepted']}  "
          f"frac={met['frac_accepted']:.3f}")
    if met['n_tn_negative'] > 0:
        print(f"  ⚠ TN<0 в {met['n_tn_negative']} точках")


# ============================================================================
# ЗАГРУЗКА ДАННЫХ И FLOW
# ============================================================================

print("\n" + "="*70)
print("LOADING TRAINING DATA")
print("="*70)
data           = torch.load(PATH_DATA)
train_ds       = create_score_feature_dataset_bcss(data, DEVICE)
train_scores, train_labels = get_scores_from_ds(train_ds)
print(f"Train: {train_scores.shape}")
for c in range(NUM_CLASSES):
    print(f"  Class {c}: {(train_labels == c).sum()}")

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
print("✓ Flow loaded")

test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, '*.tensor')))
print(f"Test files: {len(test_files)}")

# ── Строим error-conditioned пулы из train ────────────────────────────────────
print("\n" + "="*70)
print("BUILDING ERROR-CONDITIONED DECOY POOLS")
print("="*70)
neg_pools = build_error_conditioned_pools(
    train_scores, train_labels, NUM_CLASSES, verbose=True)


# ============================================================================
# TILE 0
# ============================================================================

print("\n" + "="*70)
print("TILE 0")
print("="*70)

test_ds_0           = load_bcss_test_file(test_files[0])
ms0, ds_flow0, ls0  = flow.generate_decoys(test_ds_0, device=DEVICE)
ms0      = ms0     [::SUBSAMPLE_STEP]
ds_flow0 = ds_flow0[::SUBSAMPLE_STEP]
ls0      = ls0     [::SUBSAMPLE_STEP]

# Baselines
bl0 = {}
for name, fn in BASELINE_METHODS.items():
    try:    bl0[name] = fn(train_scores, train_labels, ms0)
    except: bl0[name] = np.nan
print("Baselines tile 0:", {k: f"{v:.4f}" for k, v in bl0.items()})

# ── Flow decoys ───────────────────────────────────────────────────────────────
print("\n--- Flow decoys ---")
exch0_flow = verify_exchangeability(ms0, ds_flow0, ls0)
diagnose_tile(ms0, ds_flow0, ls0,
              tile_name='tile0_flow', exch_result=exch0_flow)
df_mm0      = control_fdr_mixmax(ms0, ls0, ds_flow0)
df_cur0     = compute_method_estimation_curve(df_mm0, 'q_values_mixmax')
met0        = find_best_mixmax_threshold(df_cur0)
print_metrics(met0, "Tile 0 | Flow")
plot_accuracy_curve(df_cur0, met0, bl0,
                    title='Tile 0 — Flow Decoys',
                    save_path='BCSS/figures/accuracy/acc_tile0_flow')

# ── Empirical decoys (error-conditioned) ──────────────────────────────────────
print("\n--- Empirical decoys (error-conditioned) ---")
ds_emp0     = generate_decoys(ms0, neg_pools, seed=42)
exch0_emp   = verify_exchangeability(ms0, ds_emp0, ls0)
diagnose_tile(ms0, ds_emp0, ls0,
              tile_name='tile0_empirical', exch_result=exch0_emp)
df_mm0_e    = control_fdr_mixmax(ms0, ls0, ds_emp0)
df_cur0_e   = compute_method_estimation_curve(df_mm0_e, 'q_values_mixmax')
met0_e      = find_best_mixmax_threshold(df_cur0_e)
print_metrics(met0_e, "Tile 0 | Empirical")
plot_accuracy_curve(df_cur0_e, met0_e, bl0,
                    title='Tile 0 — Empirical Decoys (error-conditioned)',
                    save_path='BCSS/figures/accuracy/acc_tile0_empirical')


# ============================================================================
# TILE 1
# ============================================================================

print("\n" + "="*70)
print("TILE 1")
print("="*70)

test_ds_1           = load_bcss_test_file(test_files[1])
ms1, ds_flow1, ls1  = flow.generate_decoys(test_ds_1, device=DEVICE)
ms1      = ms1     [::SUBSAMPLE_STEP]
ds_flow1 = ds_flow1[::SUBSAMPLE_STEP]
ls1      = ls1     [::SUBSAMPLE_STEP]

bl1 = {}
for name, fn in BASELINE_METHODS.items():
    try:    bl1[name] = fn(train_scores, train_labels, ms1)
    except: bl1[name] = np.nan
print("Baselines tile 1:", {k: f"{v:.4f}" for k, v in bl1.items()})

# ── Flow decoys ───────────────────────────────────────────────────────────────
print("\n--- Flow decoys ---")
exch1_flow = verify_exchangeability(ms1, ds_flow1, ls1)
diagnose_tile(ms1, ds_flow1, ls1,
              tile_name='tile1_flow', exch_result=exch1_flow)
df_mm1      = control_fdr_mixmax(ms1, ls1, ds_flow1)
df_cur1     = compute_method_estimation_curve(df_mm1, 'q_values_mixmax')
met1        = find_best_mixmax_threshold(df_cur1)
print_metrics(met1, "Tile 1 | Flow")
plot_accuracy_curve(df_cur1, met1, bl1,
                    title='Tile 1 — Flow Decoys',
                    save_path='BCSS/figures/accuracy/acc_tile1_flow')

# ── Empirical decoys (error-conditioned) ──────────────────────────────────────
print("\n--- Empirical decoys (error-conditioned) ---")
ds_emp1     = generate_decoys(ms1, neg_pools, seed=42)
exch1_emp   = verify_exchangeability(ms1, ds_emp1, ls1)
diagnose_tile(ms1, ds_emp1, ls1,
              tile_name='tile1_empirical', exch_result=exch1_emp)
df_mm1_e    = control_fdr_mixmax(ms1, ls1, ds_emp1)
df_cur1_e   = compute_method_estimation_curve(df_mm1_e, 'q_values_mixmax')
met1_e      = find_best_mixmax_threshold(df_cur1_e)
print_metrics(met1_e, "Tile 1 | Empirical")
plot_accuracy_curve(df_cur1_e, met1_e, bl1,
                    title='Tile 1 — Empirical Decoys (error-conditioned)',
                    save_path='BCSS/figures/accuracy/acc_tile1_empirical')


# ============================================================================
# ИТОГОВАЯ ТАБЛИЦА
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

header = (f"\n{'Case':<38} {'π₀':>6} {'ST_est':>8} {'ST_true':>8} "
          f"{'err_ST':>7} {'TA_est':>8} {'TA_true':>8} "
          f"{'err_TA':>7} {'TN<0':>5}")
print(header)
print("-" * 100)

summary = [
    ("Tile 0 | Flow decoys",                met0),
    ("Tile 0 | Empirical (error-cond.)",    met0_e),
    ("Tile 1 | Flow decoys",                met1),
    ("Tile 1 | Empirical (error-cond.)",    met1_e),
]
for name, m in summary:
    print(f"  {name:<36} {m['pi0_est']:>6.3f} "
          f"{m['acc_st_est']:>8.4f} {m['acc_st_true']:>8.4f} "
          f"{m['err_st']:>7.4f} "
          f"{m['acc_ta_est']:>8.4f} {m['acc_ta_true']:>8.4f} "
          f"{m['err_ta']:>7.4f} "
          f"{m['n_tn_negative']:>5}")

print("\nBaselines vs True ACC_ST:")
print(f"  {'Method':<10} {'T0_true':>8} {'T0_bl':>8} "
      f"{'T1_true':>8} {'T1_bl':>8}")
print("-" * 40)
for name in BASELINE_METHODS:
    print(f"  {name:<10} {met0['acc_st_true']:>8.4f} "
          f"{bl0.get(name, np.nan):>8.4f} "
          f"{met1['acc_st_true']:>8.4f} "
          f"{bl1.get(name, np.nan):>8.4f}")

# Exchangeability summary
print("\nExchangeability (KS test, f(t)|wrong vs Z):")
print(f"  {'Case':<38} {'KS stat':>9} {'p-value':>9} {'OK':>4}")
print("-" * 65)
for name, exch in [
    ("Tile 0 | Flow",           exch0_flow),
    ("Tile 0 | Empirical",      exch0_emp),
    ("Tile 1 | Flow",           exch1_flow),
    ("Tile 1 | Empirical",      exch1_emp),
]:
    ok = "✓" if exch.get('exchangeable') else "✗"
    print(f"  {name:<38} "
          f"{exch.get('ks_stat', float('nan')):>9.4f} "
          f"{exch.get('ks_pval', float('nan')):>9.4f} "
          f"{ok:>4}")

print("\n✓ Done")