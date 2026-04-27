import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Print numpy version
print(np.__version__)
np.set_printoptions(threshold=np.inf)

# Set global font sizes for publication quality
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 18
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
    fig, ax = plt.subplots(figsize=(6, 4))

    all_x = np.concatenate([d['df']['acc_st_true'].values for d in experiment_data])
    all_y = np.concatenate([d['df']['acc_ta_true'].values for d in experiment_data])
    lo, hi = axis_lims(all_x, all_y)

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y', zorder=2)

    # CIFAR-10-C is plotted last so its points appear on top
    others = [d for d in experiment_data if d['label'] != 'CIFAR-10-C']
    cifar  = [d for d in experiment_data if d['label'] == 'CIFAR-10-C']
    for d in others + cifar:
        ax.scatter(
            d['df']['acc_st_true'].values,
            d['df']['acc_ta_true'].values,
            label=d['label'],
            color=d['color'],
            s=25, alpha=0.75, zorder=3,
        )

    ax.set_xlabel('True ACC ST')
    ax.set_ylabel('True ACC TA')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.legend(fontsize=10)
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
    fig, ax = plt.subplots(figsize=(6, 4))

    all_x = np.concatenate([d['df']['acc_st_true'].values for d in experiment_data])
    all_y = np.concatenate([d['df']['mano_fro_norm'].values for d in experiment_data])
    lo, hi = axis_lims(all_x, all_y)

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y', zorder=2)

    for d in experiment_data:
        ax.scatter(
            d['df']['acc_st_true'].values,
            d['df']['mano_fro_norm'].values,
            label=d['label'],
            color=d['color'],
            s=25, alpha=0.75, zorder=3,
        )

    ax.set_xlabel('True ACC ST')
    ax.set_ylabel('MANO')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.legend(fontsize=10)
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

    fig, ax = plt.subplots(figsize=(6, 4))

    all_x = np.concatenate([df['acc_st_true'].values, df['acc_ta_true'].values])
    all_y_lists = [df['acc_ta_mm'].values]
    for m in available:
        all_y_lists.append(df[f'est_{m}'].dropna().values)
    all_y = np.concatenate(all_y_lists)
    lo, hi = axis_lims(all_x, all_y)

    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='x=y', zorder=2)

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
            s=12, alpha=0.50, zorder=3,
        )

    # ── ENGPE-TA (our method, drawn last, bright) ─────────────────────────────
    ax.scatter(
        df['acc_ta_true'].values,
        df['acc_ta_mm'].values,
        label='ENGPE-TA',
        marker='o',
        color=COLOR_ENGPE_TA,
        s=30, alpha=1.0, zorder=5,
        edgecolors='#BF360C', linewidths=0.5,
    )

    ax.set_xlabel('True Accuracy')
    ax.set_ylabel('Estimated Accuracy')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


plot_est_vs_true(df_cifar,  'fig2_est_vs_true_cifar10c')
plot_est_vs_true(df_bcss,   'fig2_est_vs_true_bcss')
plot_est_vs_true(df_breeds, 'fig2_est_vs_true_breeds')

print('✓ Plot 2 done\n')


# ════════════════════════════════════════════════════════════════════════════════
# PLOTS 3 & 4: Acc/FDR curves and Precision-Recall curves
# Source: acc_curves CSVs (100 subsampled points per test set)
# ════════════════════════════════════════════════════════════════════════════════

# ── Load acc_curves data ───────────────────────────────────────────────────────
curves_cifar  = pd.read_csv('results_ds/cifar-10-c/results_acc_curves.csv')
curves_bcss   = pd.read_csv('results_ds/bcss/norm_flow_acc_curves.csv')
curves_breeds = pd.concat([
    pd.read_csv(f'results_ds/breeds/breeds_{n}_acc_curves.csv')
    for n in BREEDS_NAMES
], ignore_index=True)

# For BREEDS, create unique test-set id (testset names repeat across sub-datasets)
curves_breeds['_uid'] = curves_breeds['breeds_name'] + '/' + curves_breeds['testset']

FA_GRID = np.linspace(0, 0.98, 200)
C_TRUE  = '#1565C0'    # dark blue  – true curves
C_EST   = COLOR_ENGPE_TA  # pink-red  – ENGPE-TA curves


def _interp(x, y, grid):
    """Interpolate (x, y) to grid, extrapolating with edge values."""
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order],
                     left=y[order][0], right=y[order][-1])


def average_curves(curves_df, id_col):
    """
    For each test set, interpolate acc/fdr curves and PR curves to FA_GRID.
    Returns dict of mean/std arrays.
    """
    acc_true_list, acc_est_list = [], []
    fdr_true_list, fdr_est_list = [], []
    pr_true_p, pr_true_r       = [], []
    pr_est_p,  pr_est_r        = [], []

    for _, group in curves_df.groupby(id_col):
        group = group.sort_values('frac_accepted')
        fa   = group['frac_accepted'].values
        at   = group['acc_true'].values
        ae   = group['acc_est_mm'].values
        ft   = np.clip(group['fdr_true'].values,    0, 1)
        fe   = np.clip(group['fdr_mixmax'].values,  0, 1)

        if len(fa) < 2:
            continue

        acc_true_list.append(_interp(fa, at, FA_GRID))
        acc_est_list.append( _interp(fa, ae, FA_GRID))
        fdr_true_list.append(_interp(fa, ft, FA_GRID))
        fdr_est_list.append( _interp(fa, fe, FA_GRID))

        # Precision-Recall derivation:
        # precision(t) = 1 - FDR(t)
        # recall(t) = (1 - FDR(t)) * (1 - t) / acc_true(0)
        # because: total_TP = acc_true(0)*N and D(t)=N*(1-t)
        acc0 = max(at[0], 1e-6)
        pt = np.clip(1 - ft, 0, 1)
        pe = np.clip(1 - fe, 0, 1)
        rt = np.clip((1 - ft) * (1 - fa) / acc0, 0, 1)
        re = np.clip((1 - fe) * (1 - fa) / acc0, 0, 1)

        pr_true_p.append(_interp(fa, pt, FA_GRID))
        pr_true_r.append(_interp(fa, rt, FA_GRID))
        pr_est_p.append( _interp(fa, pe, FA_GRID))
        pr_est_r.append( _interp(fa, re, FA_GRID))

    def _ms(lst):
        arr = np.array(lst)
        return arr.mean(axis=0), arr.std(axis=0)

    am, as_ = _ms(acc_true_list)
    em, es  = _ms(acc_est_list)
    ftm, fts = _ms(fdr_true_list)
    fem, fes = _ms(fdr_est_list)

    return dict(
        fa       = FA_GRID,
        acc_true_mean=am, acc_true_std=as_,
        acc_est_mean =em, acc_est_std =es,
        fdr_true_mean=ftm, fdr_true_std=fts,
        fdr_est_mean =fem, fdr_est_std =fes,
        pr_true_p    = np.array(pr_true_p).mean(axis=0),
        pr_true_r    = np.array(pr_true_r).mean(axis=0),
        pr_est_p     = np.array(pr_est_p).mean(axis=0),
        pr_est_r     = np.array(pr_est_r).mean(axis=0),
    )


def plot_acc_fdr(avg, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    fa = avg['fa']

    # True Acc & FDR
    ax.plot(fa, avg['acc_true_mean'], color=C_TRUE, lw=1.8, label='True Acc')
    ax.fill_between(fa,
                    np.clip(avg['acc_true_mean'] - avg['acc_true_std'], 0, 1),
                    np.clip(avg['acc_true_mean'] + avg['acc_true_std'], 0, 1),
                    alpha=0.15, color=C_TRUE)
    ax.plot(fa, avg['fdr_true_mean'], color=C_TRUE, lw=1.5, ls='--', label='True FDR')
    ax.fill_between(fa,
                    np.clip(avg['fdr_true_mean'] - avg['fdr_true_std'], 0, 1),
                    np.clip(avg['fdr_true_mean'] + avg['fdr_true_std'], 0, 1),
                    alpha=0.15, color=C_TRUE)

    # ENGPE-TA Acc & FDR
    ax.plot(fa, avg['acc_est_mean'], color=C_EST, lw=1.8, label='ENGPE-TA Acc')
    ax.fill_between(fa,
                    np.clip(avg['acc_est_mean'] - avg['acc_est_std'], 0, 1),
                    np.clip(avg['acc_est_mean'] + avg['acc_est_std'], 0, 1),
                    alpha=0.15, color=C_EST)
    ax.plot(fa, avg['fdr_est_mean'], color=C_EST, lw=1.5, ls='--', label='ENGPE-TA FDR')
    ax.fill_between(fa,
                    np.clip(avg['fdr_est_mean'] - avg['fdr_est_std'], 0, 1),
                    np.clip(avg['fdr_est_mean'] + avg['fdr_est_std'], 0, 1),
                    alpha=0.15, color=C_EST)

    ax.set_xlabel('Fraction accepted')
    ax.set_ylabel('Value')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


def plot_pr_curve(avg, filename):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(avg['pr_true_r'], avg['pr_true_p'],
            color=C_TRUE, lw=1.8, label='True PR')
    ax.plot(avg['pr_est_r'], avg['pr_est_p'],
            color=C_EST, lw=1.8, ls='--', label='ENGPE-TA PR')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_fig(fig, filename)


# ── Compute averages and save plots ───────────────────────────────────────────
avg_cifar  = average_curves(curves_cifar,  'corruption')
avg_bcss   = average_curves(curves_bcss,   'tile')
avg_breeds = average_curves(curves_breeds, '_uid')

plot_acc_fdr(avg_cifar,  'fig3_acc_fdr_cifar10c')
plot_acc_fdr(avg_bcss,   'fig3_acc_fdr_bcss')
plot_acc_fdr(avg_breeds, 'fig3_acc_fdr_breeds')

plot_pr_curve(avg_cifar,  'fig4_pr_cifar10c')
plot_pr_curve(avg_bcss,   'fig4_pr_bcss')
plot_pr_curve(avg_breeds, 'fig4_pr_breeds')

print('✓ Plots 3 & 4 done\n')
print('All publication plots saved to:', os.path.abspath(OUT_DIR))
