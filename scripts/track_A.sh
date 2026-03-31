#!/bin/bash
# Track A: GPUs 0,2,4,5,6 — lr=1e-4 all E → lr=1e-2 all E  (3 jobs/GPU)
set -e
cd /nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
mkdir -p logs

GPUS_ARG="0,2,4,5,6"
JPG=3   # jobs per GPU

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log "Track A started (GPUs $GPUS_ARG, ${JPG} jobs/GPU = $((5*JPG)) slots)"

# Wait until current Python jobs on GPUs 2,4,5 finish (Ditto/FedRoD style)
# Detect by watching GPU memory on GPUs 0,2,4,5,6 - skip, just start immediately
# since run_skin_grid.sh's resume logic will skip already-done seeds and the
# running jobs on 2,4,5 will just share GPU memory (24GB / 3 ≈ 8GB each = fine)

for E in 1 2 5 10; do
    log "Track A: lr=1e-4 E=${E}"
    bash scripts/run_skin_grid.sh --lr=1e-4 --E="${E}" --exp=grid \
        --gpus="${GPUS_ARG}" --jobs_per_gpu="${JPG}"
    log "Track A: done lr=1e-4 E=${E}"
done

for E in 1 2 5 10; do
    log "Track A: lr=1e-2 E=${E}"
    bash scripts/run_skin_grid.sh --lr=1e-2 --E="${E}" --exp=grid \
        --gpus="${GPUS_ARG}" --jobs_per_gpu="${JPG}"
    log "Track A: done lr=1e-2 E=${E}"
done

log "Track A complete!"
