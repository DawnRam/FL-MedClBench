"""compare_methods_E1.py — All-method comparison at lr=1e-4, E=1, T=100."""
import pandas as pd, numpy as np, os

BASE    = "results/FLSkin/grid"
LR, E, T = "1e-4", 1, 100
LAST_K  = 5
CENTERS = ["Derm7pt","HAM10000","ISIC_2019","PAD-UFES-20"]
METRICS = ["val_auc","test_auc","test_acc","test_f1","test_rec","test_prec"]
METHODS = ["LocalTrain","FedAvg","FedProx","MOON","FedAWA","FedNova",
           "PN","FedRDN","FedLWS","FedBN","SioBN","FedPer","FedRoD","Ditto"]

def load(method, seed):
    f = f"{BASE}/100epoch_{LR}_{E}/{method}/metrics_seed{seed}.csv"
    if not os.path.isfile(f): return None
    try:
        df = pd.read_csv(f)
    except Exception:
        return None
    if "center" not in df.columns or "round" not in df.columns:
        # missing header — try re-reading with header from a known-good file
        return None
    # Deduplicate: multiple runs concatenated → keep last occurrence per (round, center)
    df = df.drop_duplicates(subset=["round","center"], keep="last")
    # Group by round, average over centers
    g = df.groupby("round", sort=True)[METRICS].mean()
    return g if len(g) >= T else None

def center_best(method, seed):
    f = f"{BASE}/100epoch_{LR}_{E}/{method}/metrics_seed{seed}.csv"
    if not os.path.isfile(f): return {}
    try:
        df = pd.read_csv(f)
    except Exception:
        return {}
    if "center" not in df.columns: return {}
    df = df.drop_duplicates(subset=["round","center"], keep="last")
    out = {}
    for c in CENTERS:
        cdf = df[df.center == c]
        if cdf.empty: continue
        out[c] = cdf.loc[cdf["val_auc"].idxmax(), "test_auc"]
    return out

def fmt(mu, std, n):
    return f"{mu:.2f}±{std:.2f}" if n > 1 else f"{mu:.2f}    "

results = []
for method in METHODS:
    bv, lv, cv = [], [], {c: [] for c in CENTERS}
    for seed in [0, 1, 2]:
        g = load(method, seed)
        if g is None: continue
        br = g["val_auc"].idxmax()
        bv.append(g.loc[br]); lv.append(g.iloc[-LAST_K:].mean())
        for c, v in center_best(method, seed).items(): cv[c].append(v)
    n = len(bv)
    if n == 0: continue
    ba = pd.DataFrame(bv); la = pd.DataFrame(lv)
    results.append({"method": method, "n": n,
        "best_auc":  ba["test_auc"].mean(), "best_auc_s": ba["test_auc"].std(ddof=0),
        "best_acc":  ba["test_acc"].mean(), "best_acc_s": ba["test_acc"].std(ddof=0),
        "best_f1":   ba["test_f1"].mean(),  "best_f1_s":  ba["test_f1"].std(ddof=0),
        "best_rec":  ba["test_rec"].mean(), "best_rec_s": ba["test_rec"].std(ddof=0),
        "last_auc":  la["test_auc"].mean(), "last_auc_s": la["test_auc"].std(ddof=0),
        "gap": ba["test_auc"].mean() - la["test_auc"].mean(),
        **{c: np.mean(vs) for c, vs in cv.items() if vs}})

df = pd.DataFrame(results).set_index("method")

W = 90
print("="*W)
print(f"  FL_Skin | lr=1e-4  E=1  T=100       (* = 仅 seed=0，无 std)")
print("="*W)

print(f"\n[Best]  val-AUC 最优轮 → test 指标 (mean ± std)")
print(f"  {'Method':<14}  {'AUC':>14}  {'ACC':>14}  {'F1':>14}  {'Recall':>14}  n")
print(f"  {'-'*82}")
for m, r in df.iterrows():
    n = int(r.n); tag = "*" if n == 1 else " "
    print(f"  {(m+tag):<15} {fmt(r.best_auc,r.best_auc_s,n):>14}  "
          f"{fmt(r.best_acc,r.best_acc_s,n):>14}  "
          f"{fmt(r.best_f1,r.best_f1_s,n):>14}  "
          f"{fmt(r.best_rec,r.best_rec_s,n):>14}  {n}/3")

print(f"\n[Last]  末尾 {LAST_K} 轮均值  &  收敛稳定性")
print(f"  {'Method':<14}  {'LastAUC':>14}  {'Best−Last':>10}")
print(f"  {'-'*46}")
for m, r in df.iterrows():
    n = int(r.n); tag = "*" if n == 1 else " "
    print(f"  {(m+tag):<15} {fmt(r.last_auc,r.last_auc_s,n):>14}  {r.gap:>+10.2f}")

print(f"\n[各中心 Best AUC (seeds 均值)]")
print(f"  {'Method':<14}  {'Derm7pt':>9}  {'HAM10000':>9}  {'ISIC_2019':>9}  {'PAD-UFES':>9}")
print(f"  {'-'*60}")
for m, r in df.iterrows():
    print(f"  {m:<15}" + "".join(
        f"  {r[c]:>9.2f}" if c in r.index and pd.notna(r[c]) else f"  {'---':>9}"
        for c in CENTERS))

print(f"\n{'='*W}")
print(f"  ★ Best-AUC 排名:")
print(f"  {'#':<4} {'Method':<14} {'BestAUC':>9}  {'BestF1':>8}  {'LastAUC':>9}  {'Best−Last':>10}")
print(f"  {'-'*62}")
for i, (m, r) in enumerate(df.sort_values("best_auc", ascending=False).iterrows(), 1):
    n = int(r.n); tag = "*" if n == 1 else ""
    bar = "▓" * int((r.best_auc - 50) / 3)
    print(f"  {i:<4} {(m+tag):<15} {r.best_auc:>9.2f}  {r.best_f1:>8.2f}  "
          f"{r.last_auc:>9.2f}  {r.gap:>+10.2f}  {bar}")
print("="*W)
