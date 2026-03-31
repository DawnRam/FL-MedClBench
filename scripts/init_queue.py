#!/usr/bin/env python3
"""init_queue.py — Initialise or refresh the file-based job queue.

Scans results/FLSkin/grid/ for completed seeds and populates:
  jobqueue/pending/   — jobs not yet started
  jobqueue/running/   — currently claimed jobs (managed by worker)
  jobqueue/done/      — verified-complete jobs
  jobqueue/failed/    — jobs that failed (worker moves here on error)

Each job file is named:  {method}__{lr}__{E}__{seed}
Content: JSON with all parameters needed to run the job.

Usage:
  python scripts/init_queue.py [--dry-run]
"""

import argparse
import json
import os
import sys

import pandas as pd

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_BASE = os.path.join(WORKDIR, "results", "FLSkin", "grid")
QUEUE_BASE   = os.path.join(WORKDIR, "jobqueue")

METHODS = ["LocalTrain","FedAvg","FedProx","MOON","FedAWA","FedNova","PN",
           "FedRDN","FedLWS","FedBN","SioBN","FedPer","FedRoD","Ditto"]
SEEDS   = [0, 1, 2]
GRID    = [
    ("1e-4", 1, 100), ("1e-4", 2, 50), ("1e-4", 5, 20), ("1e-4", 10, 10),
    ("1e-3", 1, 100), ("1e-3", 2, 50), ("1e-3", 5, 20), ("1e-3", 10, 10),
    ("1e-2", 1, 100), ("1e-2", 2, 50), ("1e-2", 5, 20), ("1e-2", 10, 10),
    ("1e-1", 1, 100), ("1e-1", 2, 50), ("1e-1", 5, 20), ("1e-1", 10, 10),
]


def _is_complete(out_dir, seed, T):
    csv_p = os.path.join(out_dir, f"metrics_seed{seed}.csv")
    if not os.path.exists(csv_p):
        return False
    try:
        df = pd.read_csv(csv_p)
        return df["round"].nunique() >= T
    except Exception:
        return False


def _is_locked(out_dir, seed):
    """Return True if seed is locked by a live process."""
    import socket
    lock_p = os.path.join(out_dir, f"seed{seed}.lock")
    if not os.path.exists(lock_p):
        return False
    try:
        host, pid_str = open(lock_p).read().strip().split(":")
        pid = int(pid_str)
        if host == socket.gethostname():
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
        else:
            return True  # Different host — assume alive
    except Exception:
        return False


def job_id(method, lr, E, seed):
    return f"{method}__{lr}__{E}__{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for subdir in ("pending", "running", "done", "failed"):
        os.makedirs(os.path.join(QUEUE_BASE, subdir), exist_ok=True)

    n_queued = n_done = n_running = n_skipped = 0

    for lr, E, T in GRID:
        setting_dir = os.path.join(RESULTS_BASE, f"{T}epoch_{lr}_{E}")
        for method in METHODS:
            out_dir = os.path.join(setting_dir, method)
            for seed in SEEDS:
                jid = job_id(method, lr, E, seed)
                done_p    = os.path.join(QUEUE_BASE, "done",    jid)
                running_p = os.path.join(QUEUE_BASE, "running", jid)
                pending_p = os.path.join(QUEUE_BASE, "pending", jid)

                # Always recheck: if CSV is now complete, move to done
                if _is_complete(out_dir, seed, T):
                    n_done += 1
                    if not os.path.exists(done_p):
                        if not args.dry_run:
                            payload = {"method": method, "lr": lr, "E": E,
                                       "T": T, "seed": seed}
                            with open(done_p, "w") as f:
                                json.dump(payload, f)
                        # Clean up from pending/running if it was there
                        for p in (pending_p, running_p):
                            if os.path.exists(p) and not args.dry_run:
                                os.remove(p)
                    continue

                # If already in running and process is live, leave it
                if os.path.exists(running_p) and _is_locked(out_dir, seed):
                    n_running += 1
                    n_skipped += 1
                    continue

                # Move stale running → pending
                if os.path.exists(running_p) and not args.dry_run:
                    os.replace(running_p, pending_p)

                # Queue as pending if not already
                if not os.path.exists(pending_p):
                    n_queued += 1
                    payload = {"method": method, "lr": lr, "E": E,
                               "T": T, "seed": seed}
                    if not args.dry_run:
                        with open(pending_p, "w") as f:
                            json.dump(payload, f)
                    print(f"  [QUEUE] {jid}")
                else:
                    n_skipped += 1

    print(f"\nQueue: {n_queued} new pending, {n_running} running, "
          f"{n_done} done, {n_skipped} skipped/existing")
    total = len(METHODS) * len(SEEDS) * len(GRID)
    print(f"Total: {n_done}/{total} complete ({100*n_done//total}%)")


if __name__ == "__main__":
    main()
