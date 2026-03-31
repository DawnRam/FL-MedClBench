#!/bin/bash
# launch_workers.sh — Launch parallel workers across GPUs.
#
# Usage:
#   bash scripts/launch_workers.sh [--gpus=0,1,2,3,4,5,6] [--workers_per_gpu=3] [--exp=grid]
#
# Each GPU gets N concurrent workers; each worker claims one job at a time.
# Workers exit automatically when the queue is empty.

WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
GPUS=(0 1 2 3 4 5 6)
WORKERS_PER_GPU=3
EXP_NAME=grid

for arg in "$@"; do
    case $arg in
        --gpus=*)            IFS=',' read -ra GPUS <<< "${arg#--gpus=}" ;;
        --workers_per_gpu=*) WORKERS_PER_GPU="${arg#--workers_per_gpu=}" ;;
        --exp=*)             EXP_NAME="${arg#--exp=}" ;;
    esac
done

LOG_DIR="$WORKDIR/logs/launch_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"
cd "$WORKDIR"

total_workers=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
echo "========================================================"
echo "  Launch Workers"
echo "  GPUs       : ${GPUS[*]}"
echo "  Workers/GPU: $WORKERS_PER_GPU  (total: $total_workers)"
echo "  Exp        : $EXP_NAME"
echo "  Launch log : $LOG_DIR"
echo "========================================================"

# First, refresh the queue
echo ""
echo "Refreshing job queue..."
python3 scripts/init_queue.py
echo ""

pids=()
for gpu in "${GPUS[@]}"; do
    for ((w=0; w<WORKERS_PER_GPU; w++)); do
        wlog="$LOG_DIR/gpu${gpu}_w${w}.log"
        bash scripts/worker.sh --gpu="$gpu" --exp="$EXP_NAME" > "$wlog" 2>&1 &
        pid=$!
        pids+=($pid)
        echo "  Launched worker GPU=$gpu w=$w  PID=$pid  log=$wlog"
    done
done

echo ""
echo "  All $total_workers workers launched. Waiting for completion..."
echo ""

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "[$(date '+%H:%M:%S')] All workers finished."
python3 scripts/init_queue.py --dry-run
