#!/bin/bash
# Track B: GPUs 1,3 — lr=1e-3 all E → lr=1e-1 all E  (3 jobs/GPU)
set -e
cd /nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
mkdir -p logs

GPUS_ARG="1,3"
JPG=3   # jobs per GPU

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log "Track B started (GPUs $GPUS_ARG, ${JPG} jobs/GPU = $((2*JPG)) slots)"

for lr in 1e-3 1e-1; do
    for E in 1 2 5 10; do
        log "Track B: lr=${lr} E=${E}"
        bash scripts/run_skin_grid.sh --lr="${lr}" --E="${E}" --exp=grid \
            --gpus="${GPUS_ARG}" --jobs_per_gpu="${JPG}"
        log "Track B: done lr=${lr} E=${E}"
    done
done

log "Track B complete!"
