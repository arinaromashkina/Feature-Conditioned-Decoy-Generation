"""
Publication plots for BREEDS entity30:
  Fig A – Accuracy vs score-threshold quantile  (averaged across all test sets)
  Fig B – Precision-Recall curve                (averaged across all test sets)

X-axis for Fig A = frac_accepted, which is the empirical quantile of the
prediction score used as threshold.  At frac_accepted=0 every sample is
accepted (min threshold); at frac_accepted≈1 almost nothing is accepted
(max threshold).  The relationship is monotone, so this axis is equivalent
to "logit threshold" up to a score-rank transform.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Settings ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
})

OUT_DIR = 'figures/publication'
os.makedirs(OUT_DIR, exist_ok=True)

C_TRUE = '#1565C0'        # dark blue  – true curves
C_EST  = '#FF0066'        # pink-red   – ENGPE-TA curves

THR_GRID    = np.linspace(0.0, 0.98, 200)   # threshold quantile axis
RECALL_GRID = np.linspace(0.0, 1.0,  201)   # recall axis for PR curve


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {name}')


# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('results_ds/breeds/breeds_entity30_acc_curves.csv')
print(f'Loaded {len(df)} rows, {df["testset"].nunique()} test sets')


# ── Curve averaging ────────────────────────────────────────────────────────────

def _interp(x, y, grid):
    """Monotone interpolation to grid (edge-fill)."""
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order],
                     left=y[order][0], right=y[order][-1])


def _prec_at_recall(rec, prec, grid):
    """
    Map (rec, prec) parametric curve onto a fixed recall grid.
    rec may be non-monotone; we take the first occurrence at each unique recall.
    """
    order = np.argsort(rec)
    r_s, p_s = rec[order], prec[order]
    _, first = np.unique(r_s, return_index=True)
    return np.interp(grid, r_s[first], p_s[first],
                     left=p_s[first][0], right=p_s[first][-1])


def average_curves(curves_df):
    acc_true_curves, acc_est_curves = [], []
    fdr_true_curves, fdr_est_curves = [], []
    pr_true_curves,  pr_est_curves  = [], []

    for ts, group in curves_df.groupby('testset'):
        group = group.sort_values('frac_accepted')
        fa = group['frac_accepted'].values
        at = group['acc_true'].values
        ae = group['acc_est_mm'].values
        ft = np.clip(group['fdr_true'].values,   0.0, 1.0)
        fe = np.clip(group['fdr_mixmax'].values,  0.0, 1.0)

        if len(fa) < 2:
            continue

        acc_true_curves.append(_interp(fa, at, THR_GRID))
        acc_est_curves.append( _interp(fa, ae, THR_GRID))
        fdr_true_curves.append(_interp(fa, ft, THR_GRID))
        fdr_est_curves.append( _interp(fa, fe, THR_GRID))

        # PR curve:
        #   precision(fa) = 1 - FDR(fa)
        #   recall(fa)    = precision(fa) * (1 - fa) / acc_true(fa=0)
        #
        # acc_true(fa=0) = at[0] = true accuracy when all samples are accepted
        # = total_TP / n.  (1-fa) = fraction of samples accepted = D/n.
        # So recall = TP/n / (total_TP/n) = TP / total_TP. Correct.
        acc0 = max(float(at[0]), 1e-6)
        pt   = np.clip(1.0 - ft, 0.0, 1.0)
        pe   = np.clip(1.0 - fe, 0.0, 1.0)
        rt   = np.clip(pt * (1.0 - fa) / acc0, 0.0, 1.0)
        re   = np.clip(pe * (1.0 - fa) / acc0, 0.0, 1.0)

        pr_true_curves.append(_prec_at_recall(rt, pt, RECALL_GRID))
        pr_est_curves.append( _prec_at_recall(re, pe, RECALL_GRID))

    def _ms(lst):
        arr = np.array(lst)
        return arr.mean(axis=0), arr.std(axis=0)

    am,  as_  = _ms(acc_true_curves)
    em,  es   = _ms(acc_est_curves)
    ftm, fts  = _ms(fdr_true_curves)
    fem, fes  = _ms(fdr_est_curves)
    ptm, pts  = _ms(pr_true_curves)
    pem, pes  = _ms(pr_est_curves)

    return dict(
        threshold     = THR_GRID,
        acc_true_mean = am,  acc_true_std = as_,
        acc_est_mean  = em,  acc_est_std  = es,
        fdr_true_mean = ftm, fdr_true_std = fts,
        fdr_est_mean  = fem, fdr_est_std  = fes,
        recall        = RECALL_GRID,
        pr_true_mean  = ptm, pr_true_std  = pts,
        pr_est_mean   = pem, pr_est_std   = pes,
        n_curves      = len(acc_true_curves),
    )


# ── Fig A: Precision vs threshold quantile ────────────────────────────────────
# Precision = accuracy among accepted = 1 - FDR.
# Rises monotonically from model accuracy (fa=0, accept all) toward 1 (fa→1).

def plot_acc_vs_threshold(avg, filename, subset_label='all test sets'):
    fig, ax = plt.subplots(figsize=(6, 4))
    thr = avg['threshold']

    prec_true = np.clip(1.0 - avg['fdr_true_mean'], 0, 1)
    prec_est  = np.clip(1.0 - avg['fdr_est_mean'],  0, 1)
    prec_true_lo = np.clip(prec_true - avg['fdr_true_std'], 0, 1)
    prec_true_hi = np.clip(prec_true + avg['fdr_true_std'], 0, 1)
    prec_est_lo  = np.clip(prec_est  - avg['fdr_est_std'],  0, 1)
    prec_est_hi  = np.clip(prec_est  + avg['fdr_est_std'],  0, 1)

    ax.plot(thr, prec_true, color=C_TRUE, lw=1.8, label='True precision')
    ax.fill_between(thr, prec_true_lo, prec_true_hi, alpha=0.15, color=C_TRUE)

    ax.plot(thr, prec_est, color=C_EST, lw=1.8, ls='--', label='ENGPE-TA precision')
    ax.fill_between(thr, prec_est_lo, prec_est_hi, alpha=0.15, color=C_EST)

    ax.set_xlabel('Score threshold quantile  (0 = accept all,  1 = reject all)')
    ax.set_ylabel('Precision  (accuracy among accepted)')
    ax.set_title(f'BREEDS entity30 — Precision vs Threshold\n({subset_label}, n={avg["n_curves"]})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


# ── Fig B: Precision-Recall curve ─────────────────────────────────────────────

def plot_pr(avg, filename, subset_label='all test sets'):
    """
    Standard PR curve parametrised by the score threshold.
    As threshold rises: recall falls (fewer accepted), precision rises.
    Traversal direction: from (recall=1, prec=avg_accuracy) → (recall=0, prec→1).
    Precision at recall=1 equals the model's average accuracy over the group,
    which is typically > 0 (the curve starts above the x-axis).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    rec = avg['recall']

    # no-skill baseline: precision = recall (iso-F1 diagonal)
    ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.3, label='no-skill')

    ax.plot(rec, avg['pr_true_mean'], color=C_TRUE, lw=1.8, label='True PR')
    ax.fill_between(rec,
                    np.clip(avg['pr_true_mean'] - avg['pr_true_std'], 0, 1),
                    np.clip(avg['pr_true_mean'] + avg['pr_true_std'], 0, 1),
                    alpha=0.15, color=C_TRUE)

    ax.plot(rec, avg['pr_est_mean'],  color=C_EST,  lw=1.8, label='ENGPE-TA PR', ls='--')
    ax.fill_between(rec,
                    np.clip(avg['pr_est_mean'] - avg['pr_est_std'], 0, 1),
                    np.clip(avg['pr_est_mean'] + avg['pr_est_std'], 0, 1),
                    alpha=0.15, color=C_EST)

    # Annotate the recall=1 endpoint (model's average precision when accepting all)
    p_at_1 = float(avg['pr_true_mean'][np.argmin(np.abs(rec - 1.0))])
    ax.annotate(f'prec={p_at_1:.2f}',
                xy=(1.0, p_at_1), xytext=(0.75, p_at_1 - 0.12),
                arrowprops=dict(arrowstyle='->', color='grey', lw=1.0),
                fontsize=9, color='grey')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'BREEDS entity30 — Precision-Recall\n({subset_label}, n={avg["n_curves"]})')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


# ── Compute and save ───────────────────────────────────────────────────────────

# All test sets
avg_all = average_curves(df)
print(f'\nAveraged over {avg_all["n_curves"]} test sets (all)')

plot_acc_vs_threshold(avg_all, 'fig_entity30_acc_vs_threshold_all', 'all test sets')
plot_pr(avg_all,              'fig_entity30_pr_all',               'all test sets')

# Corrupted only (harder, lower accuracy → PR curve starts closer to (1, 0))
df_corr = df[df['testset'].str.startswith('corr')]
if len(df_corr) > 0:
    avg_corr = average_curves(df_corr)
    print(f'Averaged over {avg_corr["n_curves"]} corrupted test sets')
    plot_acc_vs_threshold(avg_corr, 'fig_entity30_acc_vs_threshold_corr', 'corrupted test sets')
    plot_pr(avg_corr,              'fig_entity30_pr_corr',               'corrupted test sets')

print('\nDone. Figures saved to', os.path.abspath(OUT_DIR))
