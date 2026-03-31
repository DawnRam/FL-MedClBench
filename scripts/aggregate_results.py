"""aggregate_results.py — Aggregate all FL-MedClsBench results into comparison tables.

Usage (run from FL-MedClsBench/):
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results_dir results --out_dir results/summary

Reads every results/<exp_tag>/summary.json and produces:
  - comparison_best.csv   / comparison_last.csv  — per-method × per-center table
  - comparison_avg.csv    — macro-averaged across centers (main benchmark table)
  - comparison_table.txt  — formatted ASCII table for paper / terminal
  - comparison_plot.png   — heatmap of AUC / ACC / F1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

METHODS_ORDER = [
    'LocalTrain', 'FedAvg', 'FedProx', 'MOON', 'FedAWA',
    'FedNova', 'PN', 'FedRDN', 'FedLWS',
    'FedBN', 'SioBN', 'FedPer', 'FedRoD', 'Ditto',
]

CENTERS = ['Center1', 'Center2', 'Center3', 'Center4']

METRICS = ['acc', 'recall', 'prec', 'f1', 'auc']

_W = 100  # table width


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hline(char='─'):
    print(char * _W)

def _box(text):
    inner = _W - 2
    pad_total = inner - len(text)
    pl = pad_total // 2
    pr = pad_total - pl
    print('╔' + '═' * inner + '╗')
    print('║' + ' ' * pl + text + ' ' * pr + '║')
    print('╚' + '═' * inner + '╝')


def find_summary_files(results_dir: str):
    """Return dict: method_name -> summary dict."""
    results = {}
    for entry in sorted(Path(results_dir).iterdir()):
        if not entry.is_dir():
            continue
        summary_path = entry / 'summary.json'
        if not summary_path.exists():
            continue
        # Extract method name from dir name:  bench_FedBCa_ResNet50_N4_E5_{METHOD}
        parts = entry.name.split('_')
        method = parts[-1]
        with open(summary_path) as f:
            data = json.load(f)
        results[method] = data
    return results


def build_dataframe(summaries: dict, split: str = 'best') -> pd.DataFrame:
    """Build a DataFrame with rows=(method, center) and cols=metrics."""
    rows = []
    for method, data in summaries.items():
        center_data = data.get(split, {})
        for center in CENTERS:
            if center not in center_data:
                continue
            cd = center_data[center]
            rows.append({
                'method': method,
                'center': center,
                'acc_mean':    cd.get('acc_mean',    np.nan) * 100,
                'acc_std':     cd.get('acc_std',     np.nan) * 100,
                'recall_mean': cd.get('recall_mean', np.nan) * 100,
                'recall_std':  cd.get('recall_std',  np.nan) * 100,
                'prec_mean':   cd.get('prec_mean',   np.nan) * 100,
                'prec_std':    cd.get('prec_std',    np.nan) * 100,
                'f1_mean':     cd.get('f1_mean',     np.nan) * 100,
                'f1_std':      cd.get('f1_std',      np.nan) * 100,
                'auc_mean':    cd.get('auc_mean',    np.nan) * 100,
                'auc_std':     cd.get('auc_std',     np.nan) * 100,
            })
    return pd.DataFrame(rows)


def build_avg_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-average over centers per method."""
    rows = []
    for method in df['method'].unique():
        sub = df[df['method'] == method]
        row = {'method': method}
        for metric in METRICS:
            means = sub[f'{metric}_mean'].dropna().values
            stds  = sub[f'{metric}_std'].dropna().values
            if len(means) > 0:
                row[f'{metric}_mean'] = means.mean()
                row[f'{metric}_std']  = np.sqrt((stds ** 2).mean())
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std']  = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sort_by_method(df: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(METHODS_ORDER)}
    if 'method' in df.columns:
        df['_order'] = df['method'].map(lambda x: order.get(x, 999))
        df = df.sort_values('_order').drop(columns=['_order'])
    return df.reset_index(drop=True)


def fmt_cell(mean, std, bold=False):
    if np.isnan(mean):
        return '    —    '
    s = f'{mean:5.1f}±{std:4.1f}'
    return s


# ─────────────────────────────────────────────────────────────────────────────
# ASCII table printing
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(avg_df: pd.DataFrame, split: str):
    avg_df = sort_by_method(avg_df)

    _box(f' FL-MedClsBench — FedBCa — {split.upper()} (macro-avg over centers) ')
    print()

    col_w = 14
    metrics_disp = ['ACC (%)', 'Recall (%)', 'Precision (%)', 'F1 (%)', 'AUC (%)']
    metric_keys  = ['acc', 'recall', 'prec', 'f1', 'auc']

    # Header
    hdr  = f"  {'Method':<14}"
    for m in metrics_disp:
        hdr += f'  {m:^{col_w}}'
    print(hdr)
    _hline()

    # Find best per metric for highlighting
    best_mean = {}
    for mk in metric_keys:
        col = avg_df[f'{mk}_mean'].dropna()
        best_mean[mk] = col.max() if len(col) > 0 else None

    sep_after = {'LocalTrain': True, 'FedLWS': True}  # group separators

    for _, row in avg_df.iterrows():
        method = row['method']
        line = f"  {method:<14}"
        for mk in metric_keys:
            mn = row.get(f'{mk}_mean', np.nan)
            sd = row.get(f'{mk}_std', np.nan)
            cell = fmt_cell(mn, sd)
            # Mark best with *
            if best_mean[mk] is not None and not np.isnan(mn):
                if abs(mn - best_mean[mk]) < 0.05:
                    cell = cell + '*'
                else:
                    cell = cell + ' '
            line += f'  {cell:^{col_w}}'
        print(line)

        if sep_after.get(method, False):
            _hline('·')

    _hline()
    print('  * = best value for that metric')
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(avg_df: pd.DataFrame, out_path: str, split: str):
    avg_df = sort_by_method(avg_df)

    metrics_disp = ['ACC', 'Recall', 'Precision', 'F1', 'AUC']
    metric_keys  = ['acc', 'recall', 'prec', 'f1', 'auc']

    methods = avg_df['method'].tolist()
    n_methods = len(methods)
    n_metrics = len(metric_keys)

    data = np.zeros((n_methods, n_metrics))
    data_std = np.zeros((n_methods, n_metrics))
    for i, (_, row) in enumerate(avg_df.iterrows()):
        for j, mk in enumerate(metric_keys):
            data[i, j] = row.get(f'{mk}_mean', np.nan)
            data_std[i, j] = row.get(f'{mk}_std', np.nan)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(n_metrics * 2.4, max(n_methods * 0.55, 4)))
    fig.patch.set_facecolor('#f8f9fa')
    title = f'FL-MedClsBench  |  FedBCa  |  {split.title()} Results  (macro-avg over centers)'
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)

    cmap = plt.cm.RdYlGn

    for j, (ax, mk, md) in enumerate(zip(axes, metric_keys, metrics_disp)):
        col = data[:, j]
        vmin = np.nanmin(col) if not np.all(np.isnan(col)) else 0
        vmax = np.nanmax(col) if not np.all(np.isnan(col)) else 100

        for i, method in enumerate(methods):
            val = data[i, j]
            std = data_std[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((0, i), 1, 1,
                                           facecolor='#eeeeee', edgecolor='white'))
                ax.text(0.5, i + 0.5, 'N/A', ha='center', va='center', fontsize=7)
            else:
                norm_val = (val - vmin) / (vmax - vmin + 1e-8)
                color = cmap(norm_val)
                ax.add_patch(plt.Rectangle((0, i), 1, 1,
                                           facecolor=color, edgecolor='white', linewidth=0.5))
                # Text color: dark on light background, light on dark
                brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                txt_color = 'black' if brightness > 0.5 else 'white'
                ax.text(0.5, i + 0.5, f'{val:.1f}\n±{std:.1f}',
                        ha='center', va='center', fontsize=7,
                        color=txt_color, fontweight='bold')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_methods)
        ax.set_xticks([])
        ax.set_title(md, fontsize=10, fontweight='semibold', pad=6)
        ax.set_yticks(np.arange(n_methods) + 0.5)
        if j == 0:
            ax.set_yticklabels(methods, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved heatmap → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Per-center detail table
# ─────────────────────────────────────────────────────────────────────────────

def print_center_tables(df: pd.DataFrame, split: str):
    df = df.copy()
    df['_order'] = df['method'].map(
        lambda x: {m: i for i, m in enumerate(METHODS_ORDER)}.get(x, 999))
    df = df.sort_values(['_order', 'center']).drop(columns=['_order'])

    metrics_disp = ['ACC (%)', 'F1 (%)', 'AUC (%)']
    metric_keys  = ['acc', 'f1', 'auc']
    col_w = 13

    for center in CENTERS:
        sub = df[df['center'] == center]
        if sub.empty:
            continue
        print(f'\n  ── {center} ──')
        hdr = f"  {'Method':<14}"
        for m in metrics_disp:
            hdr += f'  {m:^{col_w}}'
        print(hdr)
        print('  ' + '─' * (len(hdr) - 2))
        for _, row in sub.iterrows():
            line = f"  {row['method']:<14}"
            for mk in metric_keys:
                mn = row.get(f'{mk}_mean', np.nan)
                sd = row.get(f'{mk}_std', np.nan)
                line += f'  {fmt_cell(mn, sd):^{col_w}}'
            print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--out_dir',     type=str, default='results/summary')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'\nSearching for results in: {os.path.abspath(args.results_dir)}')
    summaries = find_summary_files(args.results_dir)

    if not summaries:
        print('  No summary.json files found. Run experiments first.')
        sys.exit(0)

    found = sorted(summaries.keys())
    print(f'  Found {len(found)} experiments: {found}')
    print()

    for split in ['best', 'last']:
        df      = build_dataframe(summaries, split)
        avg_df  = build_avg_dataframe(df)

        # Save CSVs
        df_path  = os.path.join(args.out_dir, f'comparison_{split}.csv')
        avg_path = os.path.join(args.out_dir, f'comparison_avg_{split}.csv')
        df.to_csv(df_path,  index=False)
        avg_df.to_csv(avg_path, index=False)
        print(f'  Saved {df_path}')
        print(f'  Saved {avg_path}')

        # Print tables
        print_comparison_table(avg_df, split)
        print_center_tables(df, split)

        # Plot heatmap
        heatmap_path = os.path.join(args.out_dir, f'heatmap_{split}.png')
        plot_heatmap(avg_df, heatmap_path, split)

    print('\nDone.\n')


if __name__ == '__main__':
    main()
