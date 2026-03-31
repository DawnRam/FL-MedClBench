"""
plot_param_selection.py
生成三张参数选择依据图：
  1. T (epoch数) 收敛图 — 训练曲线，说明100 rounds足够
  2. E (本地epoch) 对比图 — 不同E下的Best val F1
  3. lr 对比图 — 不同lr下的训练曲线
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR  = os.path.dirname(SCRIPT_DIR)            # FL-MedClsBench/
GRID_BASE  = os.path.join(BENCH_DIR, 'results', 'FLSkin', 'grid')
OUT_DIR    = os.path.join(BENCH_DIR, 'results', 'FLSkin', 'param_selection_plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'legend.fontsize':  9.5,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linestyle':   '--',
})

# colour palette (colorblind-friendly)
COLORS = {
    'LocalTrain': '#333333',
    'FedAvg':     '#1f77b4',
    'FedProx':    '#aec7e8',
    'MOON':       '#ff7f0e',
    'FedNova':    '#d62728',
    'FedRDN':     '#9467bd',
    'FedBN':      '#2ca02c',
    'SioBN':      '#98df8a',
    'FedPer':     '#8c564b',
    'FedRoD':     '#e377c2',
    'Ditto':      '#7f7f7f',
    'FedAWA':     '#bcbd22',
    'FedLWS':     '#17becf',
    'PN':         '#ffbb78',
}

LINESTYLES = {
    'LocalTrain': '-',
    'FedAvg':     '-',
    'FedProx':    '--',
    'MOON':       '--',
    'FedNova':    '-.',
    'FedRDN':     ':',
    'FedBN':      '-',
    'SioBN':      '--',
    'FedPer':     '-.',
    'FedRoD':     ':',
}

# ─── helpers ─────────────────────────────────────────────────────────────────

def load_method_curves(setting_dir, method, metric='val_f1', max_seeds=3):
    """Return (rounds, mean_per_round, std_per_round) averaged over sites & seeds."""
    seed_avgs = []
    for seed in range(max_seeds):
        csv = os.path.join(setting_dir, method, f'metrics_seed{seed}.csv')
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        avg = df.groupby('round')[metric].mean()
        seed_avgs.append(avg)
    if not seed_avgs:
        return None, None, None
    combined = pd.concat(seed_avgs, axis=1)
    return (combined.index.values,
            combined.mean(axis=1).values,
            combined.std(axis=1).values)


def load_best_val(setting_dir, method, metric='val_f1', max_seeds=3):
    """Return (best_mean, best_std) = best epoch val metric, averaged over seeds."""
    bests = []
    for seed in range(max_seeds):
        csv = os.path.join(setting_dir, method, f'metrics_seed{seed}.csv')
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        bests.append(df.groupby('round')[metric].mean().max())
    if not bests:
        return None, None
    return np.mean(bests), np.std(bests)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Convergence: T determination
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence():
    setting_dir = os.path.join(GRID_BASE, '100epoch_1e-4_1')

    # Representative methods: 2 global, 3 personalized, 1 local
    plot_methods = ['LocalTrain', 'FedAvg', 'FedNova', 'FedBN', 'FedPer', 'SioBN']
    category     = {'LocalTrain':'Local', 'FedAvg':'Global FL', 'FedNova':'Global FL',
                    'FedBN':'Personal. FL', 'FedPer':'Personal. FL', 'SioBN':'Personal. FL'}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for method in plot_methods:
        rounds, mean, std = load_method_curves(setting_dir, method)
        if rounds is None:
            continue
        c  = COLORS.get(method, '#999999')
        ls = LINESTYLES.get(method, '-')
        ax.plot(rounds, mean, color=c, linestyle=ls, linewidth=1.8,
                label=f'{method}')
        ax.fill_between(rounds, mean - std, mean + std, color=c, alpha=0.12)

    # Mark best-round window (r50–r80) with a shaded band
    ax.axvspan(50, 80, color='gold', alpha=0.15, label='Typical best-val window (r50–r80)')

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Avg Val F1 (%) — 4 sites, 3 seeds')
    ax.set_title('(a) Convergence Analysis: T = 100 Rounds is Sufficient\n'
                 r'(FL$_{\rm Skin}$, lr = 1e-4, E = 1, Total = 100 local epochs)')
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 80)

    # Legend: split by category
    handles, labels = ax.get_legend_handles_labels()
    # Add category separators via proxy artists
    sep = Line2D([], [], color='none')
    leg_handles = [
        mpatches.Patch(color='none', label='— Local —'),
        *[h for h, l in zip(handles, labels) if l in ['LocalTrain']],
        mpatches.Patch(color='none', label='— Global FL —'),
        *[h for h, l in zip(handles, labels) if l in ['FedAvg','FedNova']],
        mpatches.Patch(color='none', label='— Personalized FL —'),
        *[h for h, l in zip(handles, labels) if l in ['FedBN','FedPer','SioBN']],
        *[h for h, l in zip(handles, labels) if 'window' in l],
    ]
    leg_labels = [
        '— Local —', 'LocalTrain',
        '— Global FL —', 'FedAvg', 'FedNova',
        '— Personalized FL —', 'FedBN', 'FedPer', 'SioBN',
        'Typical best-val window (r50–r80)',
    ]
    ax.legend(leg_handles, leg_labels, ncol=2, fontsize=8.5,
              loc='lower right', framealpha=0.85)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'fig1_convergence_T.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — E (local epochs) comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_E_comparison():
    E_configs = [
        (1,  '100epoch_1e-4_1',  3),   # T=100, 3 seeds
        (2,  '100epoch_1e-4_2',  3),   # T=50,  3 seeds
        (5,  '100epoch_1e-4_5',  1),   # T=20,  1 seed (data limited)
        (10, '100epoch_1e-4_10', 3),   # T=10,  1-3 seeds
    ]

    # Methods with data in all E settings (at least 1 seed for E=5,10)
    methods = ['LocalTrain', 'FedAvg', 'FedNova', 'FedBN', 'FedPer', 'SioBN',
               'FedProx', 'FedRDN', 'FedRoD']

    # Collect best val F1 per (method, E)
    records = {}
    for method in methods:
        records[method] = {'E': [], 'mean': [], 'std': [], 'n': []}
        for E, setting, max_s in E_configs:
            sd = os.path.join(GRID_BASE, setting)
            m, s = load_best_val(sd, method, max_seeds=max_s)
            if m is not None:
                records[method]['E'].append(E)
                records[method]['mean'].append(m)
                records[method]['std'].append(s if s is not None else 0)
                records[method]['n'].append(max_s)

    # ── subplot layout: 3×3 grid ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), sharey=False)
    axes = axes.flatten()

    E_vals = [1, 2, 5, 10]
    x = np.arange(len(E_vals))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        rec = records[method]
        if not rec['E']:
            ax.set_visible(False)
            continue

        # map available E to x positions
        means  = []
        stds   = []
        colors_bar = []
        for e in E_vals:
            if e in rec['E']:
                i = rec['E'].index(e)
                means.append(rec['mean'][i])
                stds.append(rec['std'][i])
                n = rec['n'][i]
                colors_bar.append('#1f77b4' if n >= 3 else '#aec7e8')
            else:
                means.append(np.nan)
                stds.append(0)
                colors_bar.append('#dddddd')

        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6,
                      color=colors_bar, edgecolor='white', linewidth=0.8,
                      error_kw={'elinewidth': 1.2, 'ecolor': '#555555'})

        # Highlight best E
        valid = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if valid:
            best_i = max(valid, key=lambda x: x[1])[0]
            bars[best_i].set_edgecolor('#d62728')
            bars[best_i].set_linewidth(2.0)

        ax.set_xticks(x)
        ax.set_xticklabels([f'E={e}' for e in E_vals], fontsize=8.5)
        ax.set_title(method, fontsize=10, fontweight='bold',
                     color=COLORS.get(method, '#333333'))
        ax.set_ylabel('Best Val F1 (%)', fontsize=8)

        ymin = max(0, min(m for m in means if not np.isnan(m)) - 8)
        ymax = max(m for m in means if not np.isnan(m)) + 8
        ax.set_ylim(ymin, ymax)

    # Hide unused axes
    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    full_patch  = mpatches.Patch(color='#1f77b4', label='3 seeds (full)')
    part_patch  = mpatches.Patch(color='#aec7e8', label='1 seed (limited data)')
    best_patch  = mpatches.Patch(facecolor='white', edgecolor='#d62728',
                                 linewidth=2, label='Best E (red border)')
    fig.legend(handles=[full_patch, part_patch, best_patch],
               loc='lower right', fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.99, 0.02))

    fig.suptitle(
        '(b) Local Epoch E Comparison (lr = 1e-4, Fixed Total = 100 Local Epochs)\n'
        r'FL$_{\rm Skin}$ — Best Val F1 averaged over sites and seeds',
        fontsize=12, y=1.01
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    out = os.path.join(OUT_DIR, 'fig2_E_comparison.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Learning rate comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lr_comparison():
    lr_configs = [
        ('1e-4', '100epoch_1e-4_1', 3),
        ('1e-3', '100epoch_1e-3_1', 3),
        ('1e-2', '100epoch_1e-2_1', 1),
        ('1e-1', '100epoch_1e-1_1', 1),
    ]
    lr_labels  = ['1e-4', '1e-3', '1e-2', '1e-1']
    lr_colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    lr_ls      = ['-', '--', '-.', ':']

    # Representative methods with most complete data across lr values
    curve_methods = ['LocalTrain', 'FedAvg']
    bar_methods   = ['LocalTrain', 'FedAvg', 'FedNova', 'FedBN', 'FedPer']

    fig = plt.figure(figsize=(13, 9))
    gs  = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.38)

    # ── Row 1: training curves for LocalTrain and FedAvg ─────────────────────
    for col, method in enumerate(curve_methods):
        ax = fig.add_subplot(gs[0, col*2 : col*2+2])
        for (lr_str, setting, max_s), c, ls in zip(lr_configs, lr_colors, lr_ls):
            sd = os.path.join(GRID_BASE, setting)
            rounds, mean, std = load_method_curves(sd, method, max_seeds=max_s)
            if rounds is None:
                continue
            label = f'lr={lr_str}' + ('' if max_s >= 3 else ' †')
            ax.plot(rounds, mean, color=c, linestyle=ls, linewidth=1.8, label=label)
            if max_s >= 3:
                ax.fill_between(rounds, mean - std, mean + std, color=c, alpha=0.12)

        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Avg Val F1 (%)')
        ax.set_title(f'{method}\n(E=1, T=100)', fontsize=11, fontweight='bold',
                     color=COLORS.get(method, '#333'))
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 85)
        ax.legend(fontsize=8.5, loc='lower right' if method == 'LocalTrain' else 'lower right')

    # ── Row 2: bar chart — best val F1 per (method, lr) ──────────────────────
    ax_bar = fig.add_subplot(gs[1, :])

    x        = np.arange(len(bar_methods))
    n_lr     = len(lr_configs)
    width    = 0.18
    offsets  = np.linspace(-(n_lr-1)/2*width, (n_lr-1)/2*width, n_lr)

    for (lr_str, setting, max_s), offset, c, ls in zip(lr_configs, offsets, lr_colors, lr_ls):
        sd = os.path.join(GRID_BASE, setting)
        means, stds = [], []
        for method in bar_methods:
            m, s = load_best_val(sd, method, max_seeds=max_s)
            means.append(m if m is not None else np.nan)
            stds.append(s if s is not None else 0)

        label = f'lr={lr_str}' + ('' if max_s >= 3 else ' †')
        bars = ax_bar.bar(x + offset, means, width=width, yerr=stds, capsize=3,
                          color=c, alpha=0.85, label=label, edgecolor='white',
                          error_kw={'elinewidth': 1.0})

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_methods, fontsize=10)
    ax_bar.set_ylabel('Best Val F1 (%)')
    ax_bar.set_title('Best Val F1 across Learning Rates (E=1, T=100)', fontsize=11)
    ax_bar.set_ylim(0, 85)
    ax_bar.legend(fontsize=9, loc='upper right', ncol=4)
    ax_bar.text(0.01, 0.03, '† single seed (limited data)',
                transform=ax_bar.transAxes, fontsize=8, color='#555555')

    fig.suptitle(
        '(c) Learning Rate Selection\n'
        r'FL$_{\rm Skin}$ — E = 1, T = 100 rounds',
        fontsize=12, y=1.01
    )

    out = os.path.join(OUT_DIR, 'fig3_lr_comparison.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Combined summary (all 3 in one figure, paper-ready)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_combined():
    """3-panel horizontal figure for paper inclusion."""
    fig = plt.figure(figsize=(16, 4.8))
    gs  = fig.add_gridspec(1, 3, wspace=0.38)

    # ── Panel (a): Convergence curves ────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    setting_dir = os.path.join(GRID_BASE, '100epoch_1e-4_1')
    plot_methods = ['LocalTrain', 'FedAvg', 'FedNova', 'FedBN', 'FedPer', 'SioBN']

    for method in plot_methods:
        rounds, mean, std = load_method_curves(setting_dir, method)
        if rounds is None: continue
        c  = COLORS.get(method, '#999')
        ls = LINESTYLES.get(method, '-')
        ax_a.plot(rounds, mean, color=c, linestyle=ls, linewidth=1.6, label=method)
        ax_a.fill_between(rounds, mean-std, mean+std, color=c, alpha=0.10)

    ax_a.axvspan(50, 80, color='gold', alpha=0.18, zorder=0)
    ax_a.text(65, 73, 'Best\nwindow', ha='center', va='top', fontsize=7.5, color='#8B6914')
    ax_a.set_xlabel('Communication Round')
    ax_a.set_ylabel('Avg Val F1 (%)')
    ax_a.set_title('(a) T Determination\n(lr=1e-4, E=1)', fontsize=11)
    ax_a.set_xlim(1, 100); ax_a.set_ylim(0, 80)
    ax_a.legend(fontsize=7.5, ncol=1, loc='lower right', framealpha=0.85)

    # ── Panel (b): E comparison bar chart (avg over methods) ─────────────────
    ax_b = fig.add_subplot(gs[1])
    E_configs = [
        (1,  '100epoch_1e-4_1',  3),
        (2,  '100epoch_1e-4_2',  3),
        (5,  '100epoch_1e-4_5',  1),
        (10, '100epoch_1e-4_10', 3),
    ]
    compare_methods = ['LocalTrain','FedAvg','FedNova','FedBN','FedPer','SioBN',
                       'FedProx','FedRDN','FedRoD']
    E_means, E_stds, E_ns = [], [], []
    for E, setting, max_s in E_configs:
        sd = os.path.join(GRID_BASE, setting)
        vals = []
        for method in compare_methods:
            m, _ = load_best_val(sd, method, max_seeds=max_s)
            if m is not None: vals.append(m)
        E_means.append(np.mean(vals) if vals else np.nan)
        E_stds.append(np.std(vals) if vals else 0)
        E_ns.append(max_s)

    x = np.arange(4)
    bar_colors = ['#1f77b4' if n>=3 else '#aec7e8' for n in E_ns]
    bars = ax_b.bar(x, E_means, yerr=E_stds, capsize=5, width=0.55,
                    color=bar_colors, edgecolor='white',
                    error_kw={'elinewidth':1.3, 'ecolor':'#444'})

    # Highlight best
    best_i = int(np.nanargmax(E_means))
    bars[best_i].set_edgecolor('#d62728'); bars[best_i].set_linewidth(2.2)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(['E=1\n(T=100)','E=2\n(T=50)','E=5\n(T=20)','E=10\n(T=10)'],
                          fontsize=9)
    ax_b.set_ylabel('Avg Best Val F1 (%)\n(macro avg, 9 methods)')
    ax_b.set_title('(b) E Determination\n(lr=1e-4, fixed total=100 epochs)', fontsize=11)
    ax_b.set_ylim(max(0, min(v for v in E_means if not np.isnan(v))-6),
                  max(v for v in E_means if not np.isnan(v))+6)
    full_p = mpatches.Patch(color='#1f77b4', label='3 seeds')
    part_p = mpatches.Patch(color='#aec7e8', label='1 seed†')
    best_p = mpatches.Patch(fc='white', ec='#d62728', lw=2, label='Best E')
    ax_b.legend(handles=[full_p, part_p, best_p], fontsize=8, loc='lower left')

    # ── Panel (c): LR comparison — best val F1 for representative methods ─────
    ax_c = fig.add_subplot(gs[2])
    lr_configs = [
        ('1e-4', '100epoch_1e-4_1', 3),
        ('1e-3', '100epoch_1e-3_1', 3),
        ('1e-2', '100epoch_1e-2_1', 1),
        ('1e-1', '100epoch_1e-1_1', 1),
    ]
    lr_colors_c = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    lr_methods  = ['LocalTrain','FedAvg','FedBN','FedNova','FedPer']
    n_lr  = len(lr_configs)
    n_m   = len(lr_methods)
    x_c   = np.arange(n_m)
    width_c = 0.17
    offsets_c = np.linspace(-(n_lr-1)/2*width_c, (n_lr-1)/2*width_c, n_lr)

    for (lr_str, setting, max_s), off, c in zip(lr_configs, offsets_c, lr_colors_c):
        sd = os.path.join(GRID_BASE, setting)
        means_c, stds_c = [], []
        for method in lr_methods:
            m, s = load_best_val(sd, method, max_seeds=max_s)
            means_c.append(m if m is not None else np.nan)
            stds_c.append(s if s is not None else 0)
        lbl = f'lr={lr_str}' + ('' if max_s>=3 else '†')
        ax_c.bar(x_c + off, means_c, width=width_c, yerr=stds_c, capsize=3,
                 color=c, alpha=0.85, label=lbl, edgecolor='white',
                 error_kw={'elinewidth':1.0})

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(lr_methods, fontsize=9, rotation=15, ha='right')
    ax_c.set_ylabel('Best Val F1 (%)')
    ax_c.set_title('(c) LR Determination\n(E=1, T=100)', fontsize=11)
    ax_c.set_ylim(0, 85)
    ax_c.legend(fontsize=8, loc='upper right', ncol=2)
    ax_c.text(0.01, 0.01, '† single seed', transform=ax_c.transAxes,
              fontsize=7, color='#666')

    fig.suptitle(
        r'Parameter Selection for FL$_{\rm Skin}$ (ResNet50 ImageNet pretrained, Adam)',
        fontsize=13, y=1.02
    )
    fig.tight_layout()

    out = os.path.join(OUT_DIR, 'fig_param_selection_combined.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ─── main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating parameter selection plots...')
    plot_convergence()
    plot_E_comparison()
    plot_lr_comparison()
    plot_combined()
    print(f'\nAll figures saved to: {OUT_DIR}')
