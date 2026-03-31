#!/bin/bash
# run_all_grid.sh — Run full grid: lr ∈ {1e-4,1e-3,1e-2,1e-1} × E ∈ {1,2,5,10}
# Runs each (lr,E) setting sequentially (all 8 GPUs per setting, 42 jobs each)
# Total: 16 settings × 42 jobs = 672 jobs

set -e
cd /nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
mkdir -p logs

LRS=(1e-4 1e-3 1e-2 1e-1)
ES=(1 2 5 10)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting full grid search (16 settings)"

for lr in "${LRS[@]}"; do
    for E in "${ES[@]}"; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Starting lr=${lr} E=${E} ==="
        bash scripts/run_skin_grid.sh --lr="${lr}" --E="${E}" --exp=grid
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Finished lr=${lr} E=${E} ==="
    done
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 16 grid settings completed!"
