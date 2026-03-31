#!/bin/bash
# Track C: GPUs 2,4,5,6 — 趁 E=1 收尾时并行跑短的 E=5/E=10 (lr=1e-4)
# 每卡只加 1 slot (已有1个E=1 job + 1个E=5/10 = 2 jobs ≈ 16-18GB, 安全)
set -e
cd /nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
mkdir -p logs

GPUS_ARG="2,4,5,6"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Track C started (GPUs $GPUS_ARG, 1 extra job/GPU = 4 extra slots)"

# E=10 先跑 (T=10 rounds, 最快)
log "Track C: lr=1e-4 E=10 (T=10)"
bash scripts/run_skin_grid.sh --lr=1e-4 --E=10 --exp=grid \
    --gpus="${GPUS_ARG}" --jobs_per_gpu=1
log "Track C: done lr=1e-4 E=10"

# E=5 (T=20 rounds)
log "Track C: lr=1e-4 E=5 (T=20)"
bash scripts/run_skin_grid.sh --lr=1e-4 --E=5 --exp=grid \
    --gpus="${GPUS_ARG}" --jobs_per_gpu=1
log "Track C: done lr=1e-4 E=5"

# E=2 (T=50 rounds, 部分已有结果)
log "Track C: lr=1e-4 E=2 (T=50)"
bash scripts/run_skin_grid.sh --lr=1e-4 --E=2 --exp=grid \
    --gpus="${GPUS_ARG}" --jobs_per_gpu=1
log "Track C: done lr=1e-4 E=2"

log "Track C complete!"
