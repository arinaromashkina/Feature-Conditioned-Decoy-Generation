import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Print numpy version
print(np.__version__)
np.set_printoptions(threshold=np.inf)

# Set global font sizes for publication quality
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})

OUT_DIR = 'figures/publication'
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {name}')


# ── Load data ──────────────────────────────────────────────────────────────────
df_cifar = pd.read_csv('results_ds/cifar-10-c/results_per_corruption.csv')
df_bcss  = pd.read_csv('results_ds/bcss/norm_flow_results.csv')

BREEDS_NAMES = ['living17', 'entity13', 'entity30', 'nonliving26']
dfs_breeds = {}
for bname in BREEDS_NAMES:
    dfs_breeds[bname] = pd.read_csv(f'results_ds/breeds/breeds_{bname}_results.csv')
df_breeds = pd.concat(dfs_breeds.values(), ignore_index=True)

print(f'CIFAR-10-C : {len(df_cifar)} test sets')
print(f'BCSS       : {len(df_bcss)} test sets')
print(f'BREEDS     : {len(df_breeds)} test sets (all 4 combined)')

# ── Colors and styles ──────────────────────────────────────────────────────────
COLOR_CIFAR    = '#1976D2'
COLOR_BCSS     = "#FFA914"
COLOR_BREEDS   = "#19BE3D"
COLOR_ENGPE_TA = "#FF0066"

BREEDS_COLORS = {
    'living17':    '#E53935',
    'entity13':    '#8E24AA',
    'entity30':    '#F4511E',
    'nonliving26': '#039BE5',
}

COMP_METHODS  = ['ATC', 'ATC-NE', 'AC', 'DOC', 'COT']
COMP_MARKERS  = ['D',   '^',      'v',  'P',   'X']
COMP_COLORS   = ["#47AFF0", "#2618EA", "#F9F335", '#6A1B9A', "#FF9B36"]


# ── Helper ─────────────────────────────────────────────────────────────────────
def axis_lims(*arrs, margin=0.03):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrs])
    vals = vals[~np.isnan(vals)]
    return vals.min() - margin, vals.max() + margin


# ════════════════════════════════════════════════════════════════════════════════
# PLOT 1.1: Overall scatter  True ACC TA (y) vs True ACC ST (x)
# Two versions: breeds combined / breeds detailed
# ════════════════════════════════════════════════════════════════════════════════
def plot_ta_vs_st(experiment_data, filename):
    """experiment_data: list of dict(label, df, color)"""
    fig, ax = plt.subplots(figsize=(7, 7))

    all_x = np.concatenate([d['df']['acc_st_true'].values for d in experiment_data])
    all_y = np.concatenate([d['df']['acc_ta_true'].values for d in experiment_data])
    lo, hi = axis_lims(all_x, all_y)

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2.0, label='x=y', zorder=2)

    # CIFAR-10-C is plotted last so its points appear on top
    others = [d for d in experiment_data if d['label'] != 'CIFAR-10-C']
    cifar  = [d for d in experiment_data if d['label'] == 'CIFAR-10-C']
    for d in others + cifar:
        ax.scatter(
            d['df']['acc_st_true'].values,
            d['df']['acc_ta_true'].values,
            label=d['label'],
            color=d['color'],
            s=50, alpha=0.75, zorder=3,
        )

    ax.set_xlabel('True ACC ST')
    ax.set_ylabel('True ACC TA')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


# Version A – BREEDS as one group
exp_v1 = [
    dict(label='BREEDS',     df=df_breeds, color=COLOR_BREEDS),
    dict(label='CIFAR-10-C', df=df_cifar,  color=COLOR_CIFAR),
    dict(label='BCSS',       df=df_bcss,   color=COLOR_BCSS),
    
]
plot_ta_vs_st(exp_v1, 'fig1_ta_vs_st_breeds_combined')

# Version B – BREEDS split into 4 sub-datasets
exp_v2 = [
    dict(label='CIFAR-10-C', df=df_cifar, color=COLOR_CIFAR),
    dict(label='BCSS',       df=df_bcss,  color=COLOR_BCSS),
]
for bname, bdf in dfs_breeds.items():
    exp_v2.append(dict(label=f'BREEDS {bname}', df=bdf, color=BREEDS_COLORS[bname]))
plot_ta_vs_st(exp_v2, 'fig1_ta_vs_st_breeds_detailed')

print('✓ Plot 1.1 done\n')


# ════════════════════════════════════════════════════════════════════════════════
# PLOT 1.2: Overall scatter  MANO (y) vs True ACC ST (x)
# Two versions: breeds combined / breeds detailed
# ════════════════════════════════════════════════════════════════════════════════
def plot_mano_vs_st(experiment_data, filename):
    fig, ax = plt.subplots(figsize=(7, 7))

    all_x = np.concatenate([d['df']['acc_st_true'].values for d in experiment_data])
    all_y = np.concatenate([d['df']['mano_fro_norm'].values for d in experiment_data])
    lo, hi = axis_lims(all_x, all_y)

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2.0, label='x=y', zorder=2)

    for d in experiment_data:
        ax.scatter(
            d['df']['acc_st_true'].values,
            d['df']['mano_fro_norm'].values,
            label=d['label'],
            color=d['color'],
            s=50, alpha=0.75, zorder=3,
        )

    ax.set_xlabel('True ACC ST')
    ax.set_ylabel('MANO')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


plot_mano_vs_st(exp_v1, 'fig1_mano_breeds_combined')
plot_mano_vs_st(exp_v2, 'fig1_mano_breeds_detailed')

print('✓ Plot 1.2 done\n')


# ════════════════════════════════════════════════════════════════════════════════
# PLOT 2: Per-experiment scatter  Estimated Accuracy (y) vs True Accuracy (x)
#
# Competitors (ATC, ATC-NE, AC, DOC, COT, MANO): x = acc_st_true
# ENGPE-TA (ours, drawn last):                    x = acc_ta_true, y = acc_ta_mm
# ════════════════════════════════════════════════════════════════════════════════
def plot_est_vs_true(df, filename):
    available = [m for m in COMP_METHODS if f'est_{m}' in df.columns]

    fig, ax = plt.subplots(figsize=(7, 7))

    all_x = np.concatenate([df['acc_st_true'].values, df['acc_ta_true'].values])
    all_y_lists = [df['acc_ta_mm'].values]
    for m in available:
        all_y_lists.append(df[f'est_{m}'].dropna().values)
    all_y = np.concatenate(all_y_lists)
    lo, hi = axis_lims(all_x, all_y)

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2.0, label='x=y', zorder=2)

    # ── Competitors ───────────────────────────────────────────────────────────
    for idx, m in enumerate(COMP_METHODS):
        if f'est_{m}' not in df.columns:
            continue
        ax.scatter(
            df['acc_st_true'].values,
            df[f'est_{m}'].values,
            label=m,
            marker=COMP_MARKERS[idx],
            color=COMP_COLORS[idx],
            s=25, alpha=0.55, zorder=3,
        )

    # ── ENGPE-TA (our method, drawn last, bright) ─────────────────────────────
    ax.scatter(
        df['acc_ta_true'].values,
        df['acc_ta_mm'].values,
        label='ENGPE',
        marker='o',
        color=COLOR_ENGPE_TA,
        s=55, alpha=1.0, zorder=5,
        edgecolors='#BF360C', linewidths=0.5,
    )

    ax.set_xlabel('True Accuracy')
    ax.set_ylabel('Estimated Accuracy')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


plot_est_vs_true(df_cifar,  'fig2_est_vs_true_cifar10c')
plot_est_vs_true(df_bcss,   'fig2_est_vs_true_bcss')
plot_est_vs_true(df_breeds, 'fig2_est_vs_true_breeds')
for bname, bdf in dfs_breeds.items():
    plot_est_vs_true(bdf, f'fig2_est_vs_true_{bname}')

print('✓ Plot 2 done\n')


# ════════════════════════════════════════════════════════════════════════════════
# PLOTS 3 & 4: Acc/FDR curves and Precision-Recall curves
# Source: acc_curves CSVs (100 subsampled points per test set)
# ════════════════════════════════════════════════════════════════════════════════

# ── Load acc_curves data ───────────────────────────────────────────────────────
curves_cifar    = pd.read_csv('results_ds/cifar-10-c/results_acc_curves.csv')
curves_bcss     = pd.read_csv('results_ds/bcss/norm_flow_acc_curves.csv')
curves_entity13 = pd.read_csv('results_ds/breeds/breeds_entity13_acc_curves.csv')

# Breeds: separate splits (logit-based x-axis)
curves_entity13_all  = pd.read_csv('results_ds/breeds/breeds_entity13_acc_curves_all.csv')
curves_entity13_corr = pd.read_csv('results_ds/breeds/breeds_entity13_acc_curves_corruptions.csv')
curves_entity13_best = pd.read_csv('results_ds/breeds/breeds_entity13_acc_curves_best.csv')

curves_entity30    = pd.read_csv('results_ds/breeds/breeds_entity30_acc_curves.csv')
curves_living17    = pd.read_csv('results_ds/breeds/breeds_living17_acc_curves.csv')
curves_nonliving26 = pd.read_csv('results_ds/breeds/breeds_nonliving26_acc_curves.csv')

RECALL_GRID = np.linspace(0, 1.0,  200)
C_TRUE  = '#1565C0'       # dark blue  – true curves
C_EST   = COLOR_ENGPE_TA  # pink-red   – ENGPE-TA curves


def _interp(x, y, grid):
    """Interpolate (x, y) to grid, extrapolating with edge values."""
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order],
                     left=y[order][0], right=y[order][-1])


def _prec_at_recall(rec, prec, recall_grid):
    """
    Given parametric (rec, prec) curve, return precision at fixed recall levels.

    rec may be non-monotone (clipped at 1) so we sort by rec ascending and
    interpolate.  At recall=1 this gives precision at the lowest threshold
    (most permissive), which is the correct PR-curve starting point.
    """
    order  = np.argsort(rec)
    r_sort = rec[order]
    p_sort = prec[order]
    # np.interp requires strictly-increasing xp; duplicates at rec=1 are
    # handled by keeping the first occurrence (smallest fa → lowest prec).
    _, first = np.unique(r_sort, return_index=True)
    return np.interp(recall_grid, r_sort[first], p_sort[first],
                     left=p_sort[first][0], right=p_sort[first][-1])


def average_curves(curves_df, id_col, x_col='frac_accepted'):
    """
    For each test set, interpolate acc/fdr curves to a common x grid and
    PR curves to RECALL_GRID.  Returns dict of mean/std arrays.

    x_col: 'frac_accepted' (CIFAR) or 'logit_threshold' (breeds).
    The x grid is derived from the 1st–99th percentile of x_col across all rows.
    frac_accepted is always used for PR recall computation regardless of x_col.
    """
    acc_true_list, acc_est_list = [], []
    fdr_true_list, fdr_est_list = [], []
    pr_true_list,  pr_est_list  = [], []

    all_x = curves_df[x_col].values
    x_min = np.nanpercentile(all_x, 1)
    x_max = np.nanpercentile(all_x, 99)
    X_GRID = np.linspace(x_min, x_max, 200)

    has_fa = 'frac_accepted' in curves_df.columns

    for _, group in curves_df.groupby(id_col):
        group = group.sort_values(x_col)
        x    = group[x_col].values
        at   = group['acc_true'].values
        ae   = group['acc_est_mm'].values
        ft   = np.clip(group['fdr_true'].values,   0, 1)
        fe   = np.clip(group['fdr_mixmax'].values, 0, 1)
        fa   = group['frac_accepted'].values if has_fa else x

        if len(x) < 2:
            continue

        acc_true_list.append(_interp(x, at, X_GRID))
        acc_est_list.append( _interp(x, ae, X_GRID))
        fdr_true_list.append(_interp(x, ft, X_GRID))
        fdr_est_list.append( _interp(x, fe, X_GRID))

        # Precision-Recall:
        #   precision(threshold) = 1 - FDR(threshold)
        #   recall(threshold)    = (1 - FDR(threshold)) * (1 - fa) / acc_true(0)
        #   because total_TP = acc_true(fa=0)*N  and  D = N*(1-fa)
        #
        # recall_est can exceed 1 when ENGPE-TA underestimates FDR at low
        # thresholds.  We parametrise the PR curve by RECALL (not threshold) to
        # avoid the degenerate flat segment that arises from clipping.
        acc0 = max(at[0], 1e-6)
        # Interpolate frac_accepted to X_GRID for recall computation
        fa_grid = _interp(x, fa, X_GRID)
        ft_grid = _interp(x, ft, X_GRID)
        fe_grid = _interp(x, fe, X_GRID)
        pt = np.clip(1 - ft_grid, 0, 1)
        pe = np.clip(1 - fe_grid, 0, 1)
        rt = np.clip((1 - ft_grid) * (1 - fa_grid) / acc0, 0, 1)
        re = np.clip((1 - fe_grid) * (1 - fa_grid) / acc0, 0, 1)

        pr_true_list.append(_prec_at_recall(rt, pt, RECALL_GRID))
        pr_est_list.append( _prec_at_recall(re, pe, RECALL_GRID))

    def _ms(lst):
        arr = np.array(lst)
        return arr.mean(axis=0), arr.std(axis=0)

    am,  as_  = _ms(acc_true_list)
    em,  es   = _ms(acc_est_list)
    ftm, fts  = _ms(fdr_true_list)
    fem, fes  = _ms(fdr_est_list)
    ptm, pts  = _ms(pr_true_list)
    pem, pes  = _ms(pr_est_list)

    return dict(
        x_grid        = X_GRID,
        acc_true_mean = am,  acc_true_std = as_,
        acc_est_mean  = em,  acc_est_std  = es,
        fdr_true_mean = ftm, fdr_true_std = fts,
        fdr_est_mean  = fem, fdr_est_std  = fes,
        pr_recall     = RECALL_GRID,
        pr_true_mean  = ptm, pr_true_std  = pts,
        pr_est_mean   = pem, pr_est_std   = pes,
    )


def plot_acc_fdr(avg, filename, x_label='Fraction accepted'):
    fig, ax = plt.subplots(figsize=(7, 7))
    x = avg['x_grid']

    # True Acc & FDR
    ax.plot(x, avg['acc_true_mean'], color=C_TRUE, lw=1.8, label='True Acc')
    ax.fill_between(x,
                    np.clip(avg['acc_true_mean'] - avg['acc_true_std'], 0, 1),
                    np.clip(avg['acc_true_mean'] + avg['acc_true_std'], 0, 1),
                    alpha=0.15, color=C_TRUE)
    ax.plot(x, avg['fdr_true_mean'], color=C_TRUE, lw=1.5, ls='--', label='True FDR')
    ax.fill_between(x,
                    np.clip(avg['fdr_true_mean'] - avg['fdr_true_std'], 0, 1),
                    np.clip(avg['fdr_true_mean'] + avg['fdr_true_std'], 0, 1),
                    alpha=0.15, color=C_TRUE)

    # ENGPE-TA Acc & FDR
    ax.plot(x, avg['acc_est_mean'], color=C_EST, lw=1.8, label='ENGPE Acc')
    ax.fill_between(x,
                    np.clip(avg['acc_est_mean'] - avg['acc_est_std'], 0, 1),
                    np.clip(avg['acc_est_mean'] + avg['acc_est_std'], 0, 1),
                    alpha=0.15, color=C_EST)
    ax.plot(x, avg['fdr_est_mean'], color=C_EST, lw=1.5, ls='--', label='ENGPE FDR')
    ax.fill_between(x,
                    np.clip(avg['fdr_est_mean'] - avg['fdr_est_std'], 0, 1),
                    np.clip(avg['fdr_est_mean'] + avg['fdr_est_std'], 0, 1),
                    alpha=0.15, color=C_EST)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Value')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


def plot_pr_curve(avg, filename):
    fig, ax = plt.subplots(figsize=(7, 7))
    rec = avg['pr_recall']

    ax.plot(rec, avg['pr_true_mean'], color=C_TRUE, lw=1.8, label='True PR')
    ax.fill_between(rec,
                    np.clip(avg['pr_true_mean'] - avg['pr_true_std'], 0, 1),
                    np.clip(avg['pr_true_mean'] + avg['pr_true_std'], 0, 1),
                    alpha=0.15, color=C_TRUE)

    ax.plot(rec, avg['pr_est_mean'], color=C_EST, lw=1.8, ls='--', label='ENGPE PR')
    ax.fill_between(rec,
                    np.clip(avg['pr_est_mean'] - avg['pr_est_std'], 0, 1),
                    np.clip(avg['pr_est_mean'] + avg['pr_est_std'], 0, 1),
                    alpha=0.15, color=C_EST)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


# ── Compute averages and save plots ───────────────────────────────────────────

# CIFAR-10-C
avg_cifar = average_curves(curves_cifar, 'corruption', x_col='frac_accepted')
plot_acc_fdr(avg_cifar,  'fig3_acc_fdr_cifar10c',  x_label='Fraction accepted')
plot_pr_curve(avg_cifar, 'fig4_pr_cifar10c')

# BCSS
avg_bcss = average_curves(curves_bcss, 'tile', x_col='frac_accepted')
plot_acc_fdr(avg_bcss,  'fig3_acc_fdr_bcss',  x_label='Fraction accepted')
plot_pr_curve(avg_bcss, 'fig4_pr_bcss')

# Entity13 — all test sets (logit threshold x-axis)
avg_entity13_all = average_curves(curves_entity13_all, 'testset', x_col='logit_threshold')
plot_acc_fdr(avg_entity13_all,  'fig3_acc_fdr_entity13_all',  x_label='Logit threshold')
plot_pr_curve(avg_entity13_all, 'fig4_pr_entity13_all')

# Entity13 — corruptions only
avg_entity13_corr = average_curves(curves_entity13_corr, 'testset', x_col='logit_threshold')
plot_acc_fdr(avg_entity13_corr,  'fig3_acc_fdr_entity13_corruptions',  x_label='Logit threshold')
plot_pr_curve(avg_entity13_corr, 'fig4_pr_entity13_corruptions')

# Entity13 — best (single closest-to-train corruption test set)
avg_entity13_best = average_curves(curves_entity13_best, 'testset', x_col='logit_threshold')
plot_acc_fdr(avg_entity13_best,  'fig3_acc_fdr_entity13_best',  x_label='Logit threshold')
plot_pr_curve(avg_entity13_best, 'fig4_pr_entity13_best')

# Entity30
avg_entity30 = average_curves(curves_entity30, 'testset', x_col='frac_accepted')
plot_acc_fdr(avg_entity30,  'fig3_acc_fdr_entity30',  x_label='Fraction accepted')
plot_pr_curve(avg_entity30, 'fig4_pr_entity30')

# Living17
avg_living17 = average_curves(curves_living17, 'testset', x_col='frac_accepted')
plot_acc_fdr(avg_living17,  'fig3_acc_fdr_living17',  x_label='Fraction accepted')
plot_pr_curve(avg_living17, 'fig4_pr_living17')

# Nonliving26
avg_nonliving26 = average_curves(curves_nonliving26, 'testset', x_col='frac_accepted')
plot_acc_fdr(avg_nonliving26,  'fig3_acc_fdr_nonliving26',  x_label='Fraction accepted')
plot_pr_curve(avg_nonliving26, 'fig4_pr_nonliving26')

print('✓ Plots 3 & 4 done\n')
print('All publication plots saved to:', os.path.abspath(OUT_DIR))
