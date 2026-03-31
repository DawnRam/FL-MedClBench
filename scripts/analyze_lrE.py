"""
analyze_lrE.py — Summarize lr × E grid search results for LocalTrain and FedAvg.
Reports mean val-AUC and test-AUC across seeds, highlights best setting.

Usage:
    python scripts/analyze_lrE.py [--metric val_auc] [--base results/FLSkin/grid]
"""
import argparse, os
import pandas as pd
import numpy as np

METHODS  = ["LocalTrain", "FedAvg"]
LRS      = ["1e-4", "1e-3", "1e-2", "1e-1"]
ES       = [1, 2, 5, 10]
SEEDS    = [0, 1, 2]
LAST_K   = 5
N_CENTERS = 4

def load_seed(path, t):
    df = pd.read_csv(path)
    g  = df.groupby("round")[["val_auc","test_auc","val_acc","test_acc",
                               "val_f1","test_f1"]].mean()
    # must have all T rounds
    if len(g) < t:
        return None
    return g

def best_row(g):
    idx = g["val_auc"].idxmax()
    return g.loc[idx]

def last_row(g, k=LAST_K):
    return g.iloc[-k:].mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="val_auc",
                    help="selection metric for Best (default: val_auc)")
    ap.add_argument("--base", default="results/FLSkin/grid")
    args = ap.parse_args()

    print(f"\n{'='*80}")
    print(f"  FL_Skin  lr × E  Search Results")
    print(f"  Selection metric : {args.metric}")
    print(f"{'='*80}")

    records = []  # (method, lr, E, best_val_auc, best_test_auc, last_test_auc, n_seeds)

    for method in METHODS:
        print(f"\n── {method} ──────────────────────────────────────────────────────────")
        print(f"  {'lr':>6}  {'E':>3}  {'T':>4}  "
              f"{'BestValAUC':>12}  {'BestTestAUC':>12}  {'LastTestAUC':>12}  "
              f"{'BestTestACC':>12}  {'BestTestF1':>12}  seeds")
        print(f"  {'-'*90}")

        best_combo = None
        best_val   = -1

        for lr in LRS:
            for e in ES:
                t = 100 // e
                seed_best = []
                seed_last = []
                n_ok = 0
                for seed in SEEDS:
                    fpath = os.path.join(args.base,
                                         f"100epoch_{lr}_{e}", method,
                                         f"metrics_seed{seed}.csv")
                    if not os.path.isfile(fpath):
                        continue
                    g = load_seed(fpath, t)
                    if g is None:
                        continue
                    seed_best.append(best_row(g))
                    seed_last.append(last_row(g))
                    n_ok += 1

                if n_ok == 0:
                    print(f"  {lr:>6}  {e:>3}  {t:>4}  {'---':>12}  {'---':>12}  {'---':>12}  {'---':>12}  {'---':>12}  0/3")
                    continue

                best_arr = pd.DataFrame(seed_best)
                last_arr = pd.DataFrame(seed_last)

                bva  = best_arr["val_auc"].mean()
                bta  = best_arr["test_auc"].mean()
                lta  = last_arr["test_auc"].mean()
                btacc = best_arr["test_acc"].mean()
                btf1  = best_arr["test_f1"].mean()

                bva_std = best_arr["val_auc"].std() if n_ok>1 else 0
                bta_std = best_arr["test_auc"].std() if n_ok>1 else 0
                lta_std = last_arr["test_auc"].std() if n_ok>1 else 0

                flag = "  ← BEST" if bva > best_val else ""
                if bva > best_val:
                    best_val   = bva
                    best_combo = (lr, e)

                print(f"  {lr:>6}  {e:>3}  {t:>4}  "
                      f"{bva:6.2f}±{bva_std:4.2f}  "
                      f"{bta:6.2f}±{bta_std:4.2f}  "
                      f"{lta:6.2f}±{lta_std:4.2f}  "
                      f"{btacc:>12.2f}  {btf1:>12.2f}  {n_ok}/3"
                      f"{flag}")

                records.append({
                    "method": method, "lr": lr, "E": e,
                    "best_val_auc": bva, "best_test_auc": bta,
                    "last_test_auc": lta, "best_test_acc": btacc,
                    "n_seeds": n_ok
                })

        if best_combo:
            print(f"\n  ★  {method} best: lr={best_combo[0]}  E={best_combo[1]}")

    # Overall recommendation
    if records:
        df = pd.DataFrame(records)
        print(f"\n{'='*80}")
        print("  Overall Best Settings (by best_val_auc):")
        for method in METHODS:
            sub = df[df.method == method]
            if sub.empty: continue
            row = sub.loc[sub.best_val_auc.idxmax()]
            print(f"  {method:<14}  lr={row.lr}  E={int(row.E)}  "
                  f"val_AUC={row.best_val_auc:.2f}  test_AUC={row.best_test_auc:.2f}  "
                  f"({int(row.n_seeds)}/3 seeds)")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
