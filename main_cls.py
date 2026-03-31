"""main_cls.py — FL-MedClsBench: Federated Learning Classification Benchmark.

Dataset:  FedBCa (4-center bladder cancer T2WI MRI, binary classification)
Methods:  LocalTrain, FedAvg, FedProx, MOON, FedAWA, FedNova, PN, FedRDN,
          FedLWS, FedBN, SioBN, FedPer, FedRoD, Ditto

Outputs per experiment (saved under results/<exp_tag>/):
  config.json           — full argument settings
  metrics_seed{N}.csv   — per-round: train_loss, val_acc/f1/auc, test_acc/f1/auc
  summary.json          — final mean±std across seeds
  curves_seed{N}.png    — loss & metric curves per seed
"""

import time
import argparse
import copy
import gc
import json
import os
import socket
import warnings
import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datasets import Data
from nodes import Node, SeedAverager
from server_funct import Server_update
from client_funct import Client_update
from utils import setup_seed, set_server_method, validate, FedRDNTransform, cosine_lr

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='FL-MedClsBench')

    # Data
    parser.add_argument('--data_path',   type=str, default='../FedBCa')
    parser.add_argument('--dataset',     type=str, default='FedBCa')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batchsize',   type=int, default=24)

    # Model
    parser.add_argument('--local_model', type=str, default='ResNet50',
                        help='ResNet50 | ResNet18 | Med3DCNN | ResNet3D10 | ResNet3D18')

    # FL
    parser.add_argument('--method',       type=str, default='FedAvg')
    parser.add_argument('--node_num',     type=int, default=4)
    parser.add_argument('--T',            type=int, default=500,
                        help='Communication rounds')
    parser.add_argument('--E',            type=int, default=5,
                        help='Local epochs per round')
    parser.add_argument('--select_ratio', type=float, default=1.0)

    # Optimisation
    parser.add_argument('--optimizer',     type=str,   default='adam')
    parser.add_argument('--lr',            type=float, default=0.0001)
    parser.add_argument('--local_wd_rate', type=float, default=5e-4)
    parser.add_argument('--momentum',      type=float, default=0.9)

    # Method-specific hyper-params
    parser.add_argument('--mu',          type=float, default=0.01,
                        help='FedProx / MOON / FedDYN / Ditto mu')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='MOON contrastive temperature τ')
    parser.add_argument('--lr_per',    type=float, default=0.0001,
                        help='Ditto personalised model LR')
    parser.add_argument('--beta',      type=float, default=1.0,
                        help='FedLWS beta')
    parser.add_argument('--min_tau',   type=float, default=0.0)
    parser.add_argument('--max_tau',   type=float, default=1.0)
    parser.add_argument('--server_epochs',    type=int,   default=100,
                        help='FedAWA inner optimisation steps')
    parser.add_argument('--server_interval',  type=int,   default=50)
    parser.add_argument('--server_optimizer', type=str,   default='sgd')
    parser.add_argument('--reg_distance',     type=str,   default='cos',
                        help='FedAWA distance: cos | euc')

    # Training strategy
    parser.add_argument('--pretrained',  action='store_true', default=False,
                        help='Use ImageNet pretrained ResNet50')
    parser.add_argument('--cosine_lr',   action='store_true', default=False,
                        help='Cosine LR annealing from lr to 0 over T rounds')

    # System
    parser.add_argument('--device',   type=str, default='0')
    parser.add_argument('--exp_name', type=str, default='exp')
    # Single-seed mode: run only one specific seed (for parallel launching)
    parser.add_argument('--seed', type=int, default=-1,
                        help='Run only this seed (0/1/2). -1 = run all seeds [0,1,2]')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Client selection helper
# ─────────────────────────────────────────────────────────────────────────────

def generate_select_list(node_num, ratio=1.0):
    if ratio >= 1.0:
        return list(range(node_num))
    k = max(1, int(ratio * node_num))
    return np.random.choice(node_num, k, replace=False).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Pretty logging helpers
# ─────────────────────────────────────────────────────────────────────────────

_W = 80  # console width

def _hline(char='─'):
    print(char * _W, flush=True)

def _box(text, char='═'):
    inner = _W - 2
    print('╔' + char * inner + '╗', flush=True)
    # Centre text
    pad_total = inner - len(text)
    pl = pad_total // 2
    pr = pad_total - pl
    print('║' + ' ' * pl + text + ' ' * pr + '║', flush=True)
    print('╚' + char * inner + '╝', flush=True)

def _section(title):
    side = ((_W - len(title) - 2) // 2)
    print('─' * side + ' ' + title + ' ' + '─' * side, flush=True)

def print_experiment_header(args):
    """Print formatted experiment configuration at startup."""
    print('\n', flush=True)
    _box(f' FL-MedClsBench Experiment ')
    print(flush=True)
    cfg_lines = [
        ('Method',      args.method),
        ('Model',       args.local_model),
        ('Dataset',     args.dataset),
        ('Data path',   args.data_path),
        ('Rounds (T)',  str(args.T)),
        ('Local epochs (E)', str(args.E)),
        ('Batch size',  str(args.batchsize)),
        ('Optimizer',   f'{args.optimizer}  lr={args.lr}'),
        ('Node num',    str(args.node_num)),
        ('GPU device',  args.device),
    ]
    col1 = max(len(k) for k, _ in cfg_lines) + 2
    for k, v in cfg_lines:
        print(f'  {k:<{col1}}: {v}', flush=True)
    print(flush=True)

def print_seed_header(seed, T, method, model):
    print(flush=True)
    _box(f' Seed {seed}  |  {method}  |  {model}  |  {T} rounds ', char='─')
    print(flush=True)

def _fmt_metrics(label, center, loss, acc, f1, auc, star=''):
    return (f'  {label:<6} {center:<10}{star:<2}'
            f'  loss={loss:7.4f}  acc={acc:6.2f}%'
            f'  f1={f1:6.2f}%  auc={auc:6.2f}%')

def print_round_header(rnd, T, elapsed_so_far, avg_round_time):
    eta = '--:--:--'
    if avg_round_time > 0 and rnd > 0:
        remaining = (T - rnd) * avg_round_time
        eta = str(datetime.timedelta(seconds=int(remaining)))
    ts  = datetime.datetime.now().strftime('%H:%M:%S')
    title = (f'Round {rnd+1:>4}/{T}  '
             f'[{ts}]  elapsed={str(datetime.timedelta(seconds=int(elapsed_so_far)))}  ETA={eta}')
    print(flush=True)
    _hline('─')
    # Centre the title
    pad = (_W - len(title)) // 2
    print(' ' * max(pad, 0) + title, flush=True)
    _hline('─')

def print_round_table(client_names, select_list,
                      train_losses, train_accs,
                      val_results, test_results, test_improved):
    """Single unified table: train + val + test per center per round."""
    # Column widths
    C = 10   # center name
    # header
    print(flush=True)
    hdr = (f"  {'Center':<{C}}  "
           f"{'TrLoss':>7}  {'TrACC%':>7}  │  "
           f"{'ValACC%':>7}  {'ValF1%':>6}  {'ValAUC%':>7}  │  "
           f"{'TstACC%':>7}  {'TstF1%':>6}  {'TstAUC%':>7}  {'':4}")
    sep = '  ' + '─' * (len(hdr) - 2)
    print(hdr, flush=True)
    print(sep, flush=True)
    for i in select_list:
        vl, va, vr, vp, vf, vauc = val_results[i]
        tl, ta, tr, tp, tf, tauc = test_results[i]
        star = ' ★' if test_improved.get(i, False) else '  '
        print(f"  {client_names[i]:<{C}}  "
              f"{train_losses[i]:7.4f}  {train_accs[i]:7.2f}%  │  "
              f"{va:7.2f}%  {vf:6.2f}%  {vauc:7.2f}%  │  "
              f"{ta:7.2f}%  {tf:6.2f}%  {tauc:7.2f}%  {star}", flush=True)
    print(sep, flush=True)

def print_round_summary(rnd_time):
    print(f"  {'─'*60}", flush=True)
    print(f"  Round time: {rnd_time:.1f}s", flush=True)

def print_seed_summary(seed, client_names, client_nodes):
    print(flush=True)
    _section(f'Seed {seed} — Final Results')
    print(flush=True)
    hdr = (f"  {'':8} {'Center':<10}  "
           f"{'ACC%':>7}  {'Recall%':>7}  {'Prec%':>7}  {'F1%':>7}  {'AUC%':>7}  {'Round':>6}")
    print(hdr, flush=True)
    print('  ' + '─' * (len(hdr) - 2), flush=True)

    print(f"  [Best val → Test]", flush=True)
    for i, name in enumerate(client_names):
        _, ba, br, bp, bf, bauc, be = client_nodes[i].recorder.log(is_log=False)
        print(f"  {'':8} {name:<10}  "
              f"{ba:7.2f}%  {br:7.2f}%  {bp:7.2f}%  {bf:7.2f}%  {bauc:7.2f}%  {be:6d}",
              flush=True)

    print(f"\n  [Last 5 rounds — Test]", flush=True)
    for i, name in enumerate(client_names):
        _, la, lr_, lp, lf, lauc = client_nodes[i].averager.log(is_log=False)
        print(f"  {'':8} {name:<10}  "
              f"{la:7.2f}%  {lr_:7.2f}%  {lp:7.2f}%  {lf:7.2f}%  {lauc:7.2f}%",
              flush=True)
    print(flush=True)

def print_final_results(method, client_names, best_avgs, last_avgs):
    print(flush=True)
    _box(f' FINAL RESULTS — {method} ')
    print(flush=True)
    hdr = (f"  {'Center':<10}  "
           f"{'ACC (mean±std)':^18}  {'F1 (mean±std)':^18}  {'AUC (mean±std)':^18}")
    sep = '  ' + '─' * (len(hdr) - 2)

    for tag, avgs in [('Best (val→test)', best_avgs), ('Last-5 test', last_avgs)]:
        print(f"  ── {tag} ──", flush=True)
        print(hdr, flush=True)
        print(sep, flush=True)
        for i, name in enumerate(client_names):
            s = avgs[i].log(is_log=False)
            # s = (m_acc, s_acc, m_rec, s_rec, m_prec, s_prec, m_f1, s_f1, m_auc, s_auc)
            ma, sa = s[0], s[1]
            mf, sf = s[6], s[7]
            mu, su = s[8], s[9]
            print(f"  {name:<10}  "
                  f"{ma:6.2f} ± {sa:5.2f}        "
                  f"{mf:6.2f} ± {sf:5.2f}        "
                  f"{mu:6.2f} ± {su:5.2f}", flush=True)
        print(flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config / CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_config(args, out_dir):
    cfg = {k: str(v) if not isinstance(v, (int, float, bool, str, list))
           else v for k, v in vars(args).items()}
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def append_metrics_row(csv_path, row: dict):
    df_new = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df_new.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, mode='a', header=False, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Beautiful curve plotting
# ─────────────────────────────────────────────────────────────────────────────

# Colorblind-friendly palette (Tableau-10)
_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
           '#59a14f', '#edc948', '#b07aa1', '#ff9da7']

_CENTER_COLORS = {
    'Center1':    _COLORS[0],
    'Center2':    _COLORS[1],
    'Center3':    _COLORS[2],
    'Center4':    _COLORS[3],
    'Derm7pt':    _COLORS[0],
    'HAM10000':   _COLORS[1],
    'ISIC_2019':  _COLORS[2],
    'PAD-UFES-20': _COLORS[3],
}


def _ema(arr, alpha=0.85):
    """Exponential moving average smoothing."""
    out = np.zeros_like(arr, dtype=float)
    if len(arr) == 0:
        return out
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * arr[i]
    return out


def plot_curves(csv_path, out_dir, seed, client_names, method, model,
                interim=False, dataset='FedBCa'):
    """Read per-round CSV and save loss + metric curves.

    interim=True  → lightweight save every PLOT_EVERY rounds (overwrites same file)
    interim=False → final high-quality save at end of seed
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if df.empty:
        return

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('ggplot')

    n_rounds_so_far = df['round'].max()
    status = f'Round {n_rounds_so_far} / —' if interim else f'Round {n_rounds_so_far} (final)'
    title  = f'{method}  |  {model}  |  Seed {seed}  |  {dataset}  [{status}]'

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)

    metric_specs = [
        ('train_loss', 'Train Loss',           False, None,      None),
        ('val_loss',   'Val Loss',             False, None,      None),
        ('val_acc',    'Val Accuracy (%)',      True,  0,         100),
        ('val_auc',    'Val AUC (%)',           True,  0,         100),
        ('test_acc',   'Test Accuracy (%)',     True,  0,         100),
        ('test_f1',    'Test F1 Score (%)',     True,  0,         100),
        ('test_auc',   'Test AUC (%)',          True,  0,         100),
    ]

    axes = [fig.add_subplot(2, 4, k + 1) for k in range(7)]
    for ax in axes:
        ax.set_facecolor('#ffffff')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cccccc')

    for ax, (col, title_ax, pct, ymin, ymax) in zip(axes, metric_specs):
        has_data = False
        for center in client_names:
            sub = df[df['center'] == center].sort_values('round')
            if col not in sub.columns or sub[col].isna().all():
                continue
            rounds = sub['round'].values
            vals   = sub[col].values.astype(float)
            color  = _CENTER_COLORS.get(center, '#888888')

            # Raw faint line
            ax.plot(rounds, vals, color=color, alpha=0.15, linewidth=0.7)
            # EMA-smoothed main line
            smooth = _ema(vals, alpha=0.90)
            ax.plot(rounds, smooth, color=color, linewidth=1.8,
                    label=center, solid_capstyle='round')
            # Endpoint dot + annotation of last smoothed value
            ax.scatter([rounds[-1]], [smooth[-1]], color=color,
                       s=25, zorder=5, clip_on=False)
            ax.annotate(f'{smooth[-1]:.1f}',
                        xy=(rounds[-1], smooth[-1]),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=6.5, color=color, va='center')
            has_data = True

        ax.set_title(title_ax, fontsize=10, fontweight='semibold', pad=5)
        ax.set_xlabel('Round', fontsize=8, labelpad=3)
        if pct:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))
        if ymin is not None:
            ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='both', labelsize=7.5)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.55)
        if has_data:
            leg = ax.legend(fontsize=7.5, framealpha=0.9, edgecolor='#cccccc',
                            loc='best', ncol=2, handlelength=1.2)
            leg.get_frame().set_linewidth(0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dpi = 100 if interim else 150
    out_path = os.path.join(out_dir, f'curves_seed{seed}.png')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    tag = '[interim]' if interim else '[final]'
    print(f'  Curves {tag} → {out_path}', flush=True)


def plot_curves_all_seeds(out_dir, client_names, seeds, method, model):
    """Overlay curves from all seeds for each metric (mean ± std band)."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('ggplot')

    # Collect per-seed data
    frames = {}
    for seed in seeds:
        csv_path = os.path.join(out_dir, f'metrics_seed{seed}.csv')
        if not os.path.exists(csv_path):
            continue
        frames[seed] = pd.read_csv(csv_path)

    if not frames:
        return

    metric_specs = [
        ('train_loss', 'Train Loss'),
        ('val_loss',   'Val Loss'),
        ('val_acc',    'Val ACC (%)'),
        ('val_auc',    'Val AUC (%)'),
        ('test_acc',   'Test ACC (%)'),
        ('test_f1',    'Test F1 (%)'),
        ('test_auc',   'Test AUC (%)'),
    ]

    n_metrics = len(metric_specs)
    n_centers = len(client_names)
    fig, axes = plt.subplots(n_centers, n_metrics,
                             figsize=(n_metrics * 3.5, n_centers * 3))
    if n_centers == 1:
        axes = axes[np.newaxis, :]
    if n_metrics == 1:
        axes = axes[:, np.newaxis]

    fig.patch.set_facecolor('#f8f9fa')
    title = f'{method}  |  {model}  |  All Seeds (mean ± std)'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for ci, center in enumerate(client_names):
        for mi, (col, col_title) in enumerate(metric_specs):
            ax = axes[ci][mi]
            ax.set_facecolor('#ffffff')

            # Collect values across seeds for this (center, metric)
            all_vals = []
            ref_rounds = None
            for seed, df in frames.items():
                sub = df[df['center'] == center].sort_values('round')
                if col not in sub.columns:
                    continue
                rounds = sub['round'].values
                vals   = _ema(sub[col].values.astype(float), alpha=0.90)
                all_vals.append(vals)
                if ref_rounds is None:
                    ref_rounds = rounds

            if not all_vals or ref_rounds is None:
                continue

            min_len   = min(len(v) for v in all_vals)
            all_arr   = np.array([v[:min_len] for v in all_vals])
            rounds_tr = ref_rounds[:min_len]

            mean_v = all_arr.mean(axis=0)
            std_v  = all_arr.std(axis=0)

            color = _CENTER_COLORS.get(center, '#4e79a7')
            ax.fill_between(rounds_tr, mean_v - std_v, mean_v + std_v,
                            alpha=0.20, color=color)
            ax.plot(rounds_tr, mean_v, color=color, linewidth=1.8,
                    label=center)

            if ci == 0:
                ax.set_title(col_title, fontsize=10, fontweight='semibold')
            if mi == 0:
                ax.set_ylabel(center, fontsize=9, fontweight='bold')
            ax.set_xlabel('Round', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'curves_all_seeds.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved all-seed curves → {out_path}', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(best_seed_averagers, last_seed_averagers, out_dir):
    summary = {'best': {}, 'last': {}}
    for averager, key in [(best_seed_averagers, 'best'),
                          (last_seed_averagers, 'last')]:
        for avg in averager:
            stats = avg.log(is_log=False)
            summary[key][avg.client_name] = {
                'acc_mean':    round(float(stats[0]), 4),
                'acc_std':     round(float(stats[1]), 4),
                'recall_mean': round(float(stats[2]), 4),
                'recall_std':  round(float(stats[3]), 4),
                'prec_mean':   round(float(stats[4]), 4),
                'prec_std':    round(float(stats[5]), 4),
                'f1_mean':     round(float(stats[6]), 4),
                'f1_std':      round(float(stats[7]), 4),
                'auc_mean':    round(float(stats[8]), 4),
                'auc_std':     round(float(stats[9]), 4),
            }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved summary → {os.path.join(out_dir, "summary.json")}',
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = get_args()

    # Dataset-specific settings
    if args.dataset == 'FLSkin':
        args.client_names = ['Derm7pt', 'HAM10000', 'ISIC_2019', 'PAD-UFES-20']
        args.num_classes  = 8
    else:  # FedBCa
        args.client_names = ['Center1', 'Center2', 'Center3', 'Center4']
        args.num_classes  = getattr(args, 'num_classes', 2)
    args.node_num = len(args.client_names)
    random_seeds  = [args.seed] if args.seed >= 0 else [0, 1, 2]

    # Method-specific defaults
    if args.method == 'FedProx':
        args.mu = 0.01

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # ── Experiment output directory ──────────────────────────────────────────
    # Structure: results/{dataset}/{exp_name}/{epochs}epoch_{lr}_{E}/{method}/
    # e.g. results/FLSkin/grid/100epoch_1e-4_1/FedAvg/
    total_epochs = args.T * args.E
    lr_str  = f'{args.lr:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
    setting = f'{total_epochs}epoch_{lr_str}_{args.E}'
    out_dir = os.path.join('results', args.dataset, args.exp_name,
                           setting, args.method)
    os.makedirs(out_dir, exist_ok=True)

    lr = args.lr  # keep original lr for multi-seed reset

    args = set_server_method(args)
    save_config(args, out_dir)

    print_experiment_header(args)
    print(f'  Output dir: {os.path.abspath(out_dir)}', flush=True)
    print(flush=True)

    # Seed-level averagers (best & last)
    best_seed_averagers = [SeedAverager(i, n)
                           for i, n in enumerate(args.client_names)]
    last_seed_averagers = [SeedAverager(i, n)
                           for i, n in enumerate(args.client_names)]

    # ─────────────────────────────────────────────────────────────────────────
    # Outer loop: random seeds
    # ─────────────────────────────────────────────────────────────────────────

    def _seed_complete(out_dir, seed, T, n_centers):
        csv_p = os.path.join(out_dir, f'metrics_seed{seed}.csv')
        if not os.path.exists(csv_p):
            return False
        try:
            df = pd.read_csv(csv_p)
            return df['round'].nunique() >= T
        except Exception:
            return False

    def _acquire_lock(lock_path):
        """Try to acquire a seed-level lock. Returns True if acquired."""
        if os.path.exists(lock_path):
            try:
                host, pid_str = open(lock_path).read().strip().split(':')
                pid = int(pid_str)
                if host == socket.gethostname():
                    try:
                        os.kill(pid, 0)
                        return False  # Process alive — lock held
                    except ProcessLookupError:
                        pass  # Stale lock from dead process
                else:
                    return False  # Different host — assume alive
            except Exception:
                pass  # Malformed lock — proceed
        with open(lock_path, 'w') as f:
            f.write(f'{socket.gethostname()}:{os.getpid()}')
        return True

    def _release_lock(lock_path):
        try:
            os.remove(lock_path)
        except Exception:
            pass

    def _save_checkpoint(ckpt_path, rnd, central_node, client_nodes, args,
                         best_val_acc, last5_test_cache, round_times):
        ckpt = {
            'round': rnd,
            'central_model': central_node.model.state_dict(),
            'best_val_acc': best_val_acc,
            'last5_test_cache': last5_test_cache,
            'round_times': round_times,
            'client_states': {},
        }
        if args.method == 'Scaffold' and hasattr(central_node, 'control'):
            ckpt['scaffold_central_control'] = central_node.control
        if hasattr(central_node, 'server_state'):
            ckpt['feddyn_server_state'] = central_node.server_state.state_dict()
        for i, node in client_nodes.items():
            cs = {
                'model': node.model.state_dict(),
                'optimizer': node.optimizer.state_dict(),
                'recorder': vars(node.recorder),
                'averager': (node.averager.loss, node.averager.acc, node.averager.recall,
                             node.averager.prec, node.averager.f1, node.averager.auc,
                             node.averager.num),
                'maxer': vars(node.maxer),
            }
            if args.method == 'Ditto' and hasattr(node, 'p_model'):
                cs['p_model'] = node.p_model.state_dict()
                cs['p_optimizer'] = node.p_optimizer.state_dict()
            if args.method == 'MOON' and hasattr(node, 'pre_model'):
                cs['pre_model'] = node.pre_model.state_dict()
            if args.method == 'Scaffold':
                cs['control'] = {k: v.cpu() for k, v in node.control.items()}
                cs['delta_control'] = {k: v.cpu() for k, v in node.delta_control.items()}
                cs['delta_y'] = {k: v.cpu() for k, v in node.delta_y.items()}
            if hasattr(node, 'old_grad'):
                cs['old_grad'] = node.old_grad.cpu()
            if hasattr(node, 'p_head'):
                cs['p_head'] = node.p_head.state_dict()
            ckpt['client_states'][i] = cs
        tmp = ckpt_path + '.tmp'
        torch.save(ckpt, tmp)
        os.replace(tmp, ckpt_path)

    def _load_checkpoint(ckpt_path, central_node, client_nodes, args, device):
        ckpt = torch.load(ckpt_path, map_location='cuda:0')
        central_node.model.load_state_dict(ckpt['central_model'])
        if 'scaffold_central_control' in ckpt:
            central_node.control = {k: v.cuda() for k, v in
                                    ckpt['scaffold_central_control'].items()}
        if 'feddyn_server_state' in ckpt:
            central_node.server_state.load_state_dict(ckpt['feddyn_server_state'])
        for i, node in client_nodes.items():
            cs = ckpt['client_states'][i]
            node.model.load_state_dict(cs['model'])
            node.optimizer.load_state_dict(cs['optimizer'])
            for k, v in cs['recorder'].items():
                if k not in ('client_id', 'client_name'):
                    setattr(node.recorder, k, v)
            (node.averager.loss, node.averager.acc, node.averager.recall,
             node.averager.prec, node.averager.f1, node.averager.auc,
             node.averager.num) = cs['averager']
            for k, v in cs['maxer'].items():
                if k not in ('client_id', 'client_name'):
                    setattr(node.maxer, k, v)
            if args.method == 'Ditto' and 'p_model' in cs:
                node.p_model.load_state_dict(cs['p_model'])
                node.p_optimizer.load_state_dict(cs['p_optimizer'])
            if args.method == 'MOON' and 'pre_model' in cs:
                node.pre_model.load_state_dict(cs['pre_model'])
            if args.method == 'Scaffold':
                node.control = {k: v.cuda() for k, v in cs['control'].items()}
                node.delta_control = {k: v.cuda() for k, v in cs['delta_control'].items()}
                node.delta_y = {k: v.cuda() for k, v in cs['delta_y'].items()}
            if 'old_grad' in cs:
                node.old_grad = cs['old_grad'].cuda()
            if 'p_head' in cs:
                node.p_head.load_state_dict(cs['p_head'])
        start_round = ckpt['round'] + 1
        best_val_acc = ckpt['best_val_acc']
        last5_test_cache = ckpt['last5_test_cache']
        round_times = ckpt['round_times']
        return start_round, best_val_acc, last5_test_cache, round_times

    exp_start = time.time()

    for random_seed in random_seeds:
        lock_path = os.path.join(out_dir, f'seed{random_seed}.lock')
        ckpt_path = os.path.join(out_dir, f'ckpt_seed{random_seed}.pt')

        if _seed_complete(out_dir, random_seed, args.T, args.node_num):
            print(f'  Seed {random_seed} already complete ({args.T} rounds × '
                  f'{args.node_num} centers) — skipping.', flush=True)
            _release_lock(lock_path)
            continue

        if not _acquire_lock(lock_path):
            print(f'  Seed {random_seed} locked by another process — skipping.', flush=True)
            continue

        gc.collect()
        torch.cuda.empty_cache()
        args.random_seed = random_seed
        args.lr = lr

        print_seed_header(random_seed, args.T, args.method, args.local_model)
        setup_seed(random_seed)
        args = set_server_method(args)

        csv_path = os.path.join(out_dir, f'metrics_seed{random_seed}.csv')
        ckpt_exists = os.path.exists(ckpt_path)

        if ckpt_exists:
            print(f'  [Resume] Checkpoint found — will resume after loading.', flush=True)
        elif os.path.exists(csv_path):
            # No checkpoint but partial CSV exists — delete it to avoid duplicate rows
            try:
                df_existing = pd.read_csv(csv_path)
                n_existing = len(df_existing)
                if 0 < n_existing < args.T * args.node_num:
                    rounds_done = df_existing['round'].nunique()
                    print(f'  [Restart] Partial CSV ({rounds_done}/{args.T} rounds), '
                          f'no checkpoint — deleting and restarting seed {random_seed}.', flush=True)
                    os.remove(csv_path)
            except Exception:
                os.remove(csv_path)

        select_lists = [generate_select_list(args.node_num, args.select_ratio)
                        for _ in range(args.T)]

        # ── Build data loaders ────────────────────────────────────────────────
        print('  Loading datasets...', flush=True)
        data = Data(args)

        sample_sizes = [len(data.train_loaders[i].dataset)
                        for i in range(args.node_num)]
        size_weights = [s / sum(sample_sizes) for s in sample_sizes]
        print(f"  Sample sizes: "
              f"{dict(zip(args.client_names, sample_sizes))}", flush=True)
        print(f"  Agg weights : "
              f"{[round(w, 4) for w in size_weights]}", flush=True)

        # ── Initialise nodes ──────────────────────────────────────────────────
        central_node = Node(args, -1, 'Server',
                            train_loader=None, val_loader=None, test_loader=None)
        client_nodes = {}
        for i in range(args.node_num):
            client_nodes[i] = Node(
                args, i, args.client_names[i],
                train_loader=data.train_loaders[i],
                val_loader=data.val_loaders[i],
                test_loader=data.test_loaders[i],
            )
            client_nodes[i].model.load_state_dict(
                copy.deepcopy(central_node.model.state_dict()))
            # Ditto: initialize personalized model from global model (not random weights)
            # Without this, p_model has different random weights → huge initial proximal term
            if args.method == 'Ditto' and hasattr(client_nodes[i], 'p_model'):
                client_nodes[i].p_model.load_state_dict(
                    copy.deepcopy(central_node.model.state_dict()))

        # ── FedRDN transforms ─────────────────────────────────────────────────
        if args.method == 'FedRDN':
            global_stats = [client_nodes[i].local_stats
                            for i in range(args.node_num)]
            for i in range(args.node_num):
                client_nodes[i].FedRDNTransform_train = FedRDNTransform(
                    local_stats=client_nodes[i].local_stats,
                    global_stats=global_stats, mode='train')
                client_nodes[i].FedRDNTransform_test = FedRDNTransform(
                    local_stats=client_nodes[i].local_stats,
                    global_stats=global_stats, mode='test')

        # ── Scaffold server control ───────────────────────────────────────────
        if args.method == 'Scaffold':
            central_node.control = {k: torch.zeros_like(v.data).cuda()
                                    for k, v in central_node.model.named_parameters()}

        best_val_acc     = {i: 0.0 for i in range(args.node_num)}
        last5_test_cache = {i: None for i in range(args.node_num)}
        start_round      = 0
        round_times      = []

        if ckpt_exists:
            try:
                start_round, best_val_acc, last5_test_cache, round_times = \
                    _load_checkpoint(ckpt_path, central_node, client_nodes,
                                     args, args.device)
                print(f'  [Resume] Loaded checkpoint — resuming from round {start_round}/{args.T}',
                      flush=True)
            except Exception as e:
                print(f'  [Resume] Checkpoint load failed ({e}) — restarting from round 0.',
                      flush=True)
                start_round = 0
                best_val_acc = {i: 0.0 for i in range(args.node_num)}
                last5_test_cache = {i: None for i in range(args.node_num)}
                round_times = []
                if os.path.exists(csv_path):
                    os.remove(csv_path)

        # ─────────────────────────────────────────────────────────────────────
        # Inner loop: communication rounds
        # ─────────────────────────────────────────────────────────────────────
        seed_start = time.time()

        for rnd in range(start_round, args.T):
            t0 = time.time()
            elapsed_so_far = time.time() - seed_start
            avg_round_time = np.mean(round_times) if round_times else 0

            print_round_header(rnd, args.T, elapsed_so_far, avg_round_time)

            # Cosine LR decay
            if args.cosine_lr:
                cur_lr = cosine_lr(lr, rnd, args.T)
                for node in client_nodes.values():
                    node.optimizer.param_groups[0]['lr'] = cur_lr

            select_list = select_lists[rnd]

            # 1. Local training
            client_nodes, train_losses, train_accs = Client_update(
                args, client_nodes, central_node, select_list)

            # 2. Server aggregation
            central_node, client_nodes = Server_update(
                args, central_node, client_nodes, select_list,
                size_weights, rnd)

            # 3. Validation + model selection + metric logging
            val_results  = {}
            test_results = {}
            test_improved = {}

            for i in select_list:
                center = args.client_names[i]
                vl, va, vr, vp, vf, vauc = validate(
                    args, client_nodes[i], client_nodes[i].val_loader)
                val_results[i] = (vl, va, vr, vp, vf, vauc)
                client_nodes[i].maxer.update(rnd, vl, va, vr, vp, vf, vauc)

                improved = va > best_val_acc[i]
                test_improved[i] = improved

                tl = ta = tr = tp = tf = tauc = 0.0
                if improved:
                    best_val_acc[i] = va
                    tl, ta, tr, tp, tf, tauc = validate(
                        args, client_nodes[i], client_nodes[i].test_loader)
                    client_nodes[i].recorder.update(rnd, tl, ta, tr, tp, tf, tauc)
                    last5_test_cache[i] = (tl, ta, tr, tp, tf, tauc)
                elif last5_test_cache[i] is not None:
                    tl, ta, tr, tp, tf, tauc = last5_test_cache[i]
                test_results[i] = (tl, ta, tr, tp, tf, tauc)

                # Last 5 rounds: always evaluate test fresh
                if rnd >= args.T - 5:
                    tl, ta, tr, tp, tf, tauc = validate(
                        args, client_nodes[i], client_nodes[i].test_loader)
                    client_nodes[i].averager.update(tl, ta, tr, tp, tf, tauc)
                    last5_test_cache[i] = (tl, ta, tr, tp, tf, tauc)
                    test_results[i] = (tl, ta, tr, tp, tf, tauc)

                # Append to CSV — loss: 4dp, percentage metrics: 2dp
                append_metrics_row(csv_path, {
                    'round':      rnd + 1,
                    'center':     center,
                    'train_loss': round(train_losses[i], 4),
                    'train_acc':  round(train_accs[i],   2),
                    'val_loss':   round(vl,   4),
                    'val_acc':    round(va,   2),
                    'val_rec':    round(vr,   2),
                    'val_prec':   round(vp,   2),
                    'val_f1':     round(vf,   2),
                    'val_auc':    round(vauc, 2),
                    'test_loss':  round(tl,   4),
                    'test_acc':   round(ta,   2),
                    'test_rec':   round(tr,   2),
                    'test_prec':  round(tp,   2),
                    'test_f1':    round(tf,   2),
                    'test_auc':   round(tauc, 2),
                })

            print_round_table(args.client_names, select_list,
                              train_losses, train_accs,
                              val_results, test_results, test_improved)

            rnd_time = time.time() - t0
            round_times.append(rnd_time)
            print_round_summary(rnd_time)

            # ── Save checkpoint after every round ──────────────────────────
            _save_checkpoint(ckpt_path, rnd, central_node, client_nodes, args,
                             best_val_acc, last5_test_cache, round_times)

            # ── Periodic curve save every PLOT_EVERY rounds ────────────────
            PLOT_EVERY = 50
            if (rnd + 1) % PLOT_EVERY == 0:
                plot_curves(csv_path, out_dir, random_seed,
                            args.client_names, args.method, args.local_model,
                            interim=True, dataset=args.dataset)

        # ── End of seed: remove checkpoint + lock, save curves ────────────────
        try:
            os.remove(ckpt_path)
        except FileNotFoundError:
            pass
        _release_lock(lock_path)
        print_seed_summary(random_seed, args.client_names, client_nodes)
        plot_curves(csv_path, out_dir, random_seed,
                    args.client_names, args.method, args.local_model,
                    interim=False, dataset=args.dataset)

        for i in range(args.node_num):
            _, ba, br, bp, bf, bauc, be = client_nodes[i].recorder.log(is_log=False)
            best_seed_averagers[i].update(ba, br, bp, bf, bauc)

        for i in range(args.node_num):
            _, la, lr_, lp, lf, lauc = client_nodes[i].averager.log(is_log=False)
            last_seed_averagers[i].update(la, lr_, lp, lf, lauc)

    # ─────────────────────────────────────────────────────────────────────────
    # Final aggregated results across seeds
    # ─────────────────────────────────────────────────────────────────────────
    save_summary(best_seed_averagers, last_seed_averagers, out_dir)
    plot_curves_all_seeds(out_dir, args.client_names, random_seeds,
                          args.method, args.local_model)

    print_final_results(args.method, args.client_names,
                        best_seed_averagers, last_seed_averagers)

    total_time = time.time() - exp_start
    print(f'\n  Total experiment time: '
          f'{str(datetime.timedelta(seconds=int(total_time)))}', flush=True)
    _hline('═')
