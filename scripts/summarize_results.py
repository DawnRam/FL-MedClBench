"""
summarize_results.py — Compute mean±std across seeds for completed methods.
Reports Best (val-AUC argmax → test metrics) and Last (avg of last 5 rounds).

Usage:
    python scripts/summarize_results.py --lr 1e-4 --E 1
"""
import argparse, os
import pandas as pd
import numpy as np

METRICS_TEST = ["test_acc", "test_rec", "test_prec", "test_f1", "test_auc"]
METRICS_VAL  = ["val_acc",  "val_rec",  "val_prec",  "val_f1",  "val_auc"]
METHODS = [
    "LocalTrain","FedAvg","FedProx","MOON","FedAWA","FedNova","PN","FedRDN",
    "FedLWS","FedBN","SioBN","FedPer","FedRoD","Ditto"
]
SEEDS = [0, 1, 2]
LAST_K = 5

def load_seed(path):
    df = pd.read_csv(path)
    # Average over centers per round
    return df.groupby("round")[METRICS_VAL + METRICS_TEST].mean()

def best_metrics(df):
    """Row with highest val_auc, return test metrics."""
    best_row = df.loc[df["val_auc"].idxmax()]
    return best_row[METRICS_TEST].values

def last_metrics(df, k=LAST_K):
    """Average test metrics over last k rounds."""
    return df[METRICS_TEST].iloc[-k:].mean().values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--E",  default="1", type=int)
    ap.add_argument("--base", default="results/FLSkin/grid")
    args = ap.parse_args()

    T = 100 // args.E
    tag = f"100epoch_{args.lr}_{args.E}"
    base = os.path.join(args.base, tag)

    print(f"\n{'='*90}")
    print(f"  FL_Skin Grid  |  lr={args.lr}  E={args.E}  T={T}")
    print(f"{'='*90}")
    hdr = f"{'Method':<12}  {'ACC':>10}  {'Recall':>10}  {'Prec':>10}  {'F1':>10}  {'AUC':>10}"
    sep = "-" * 90

    for mode in ("Best", "Last"):
        print(f"\n[{mode}]   (mean ± std across seeds)\n{sep}")
        print(hdr)
        print(sep)
        for method in METHODS:
            seed_vals = []
            missing = False
            for seed in SEEDS:
                fpath = os.path.join(base, method, f"metrics_seed{seed}.csv")
                if not os.path.isfile(fpath):
                    missing = True; break
                try:
                    df = load_seed(fpath)
                    expected_rows = T
                    if len(df) < expected_rows:
                        missing = True; break
                    if mode == "Best":
                        seed_vals.append(best_metrics(df))
                    else:
                        seed_vals.append(last_metrics(df))
                except Exception as e:
                    missing = True; break
            if missing:
                print(f"  {method:<12}  {'--- incomplete ---'}")
                continue
            arr = np.array(seed_vals)  # already in % (0-100)
            mu  = arr.mean(axis=0)
            std = arr.std(axis=0)
            row = f"  {method:<12}"
            for m, s in zip(mu, std):
                row += f"  {m:5.2f}±{s:4.2f}"
            print(row)
        print(sep)

if __name__ == "__main__":
    main()
