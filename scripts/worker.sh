#!/bin/bash
# worker.sh — Atomic job worker for FL-MedClsBench grid search.
#
# Usage:
#   bash scripts/worker.sh --gpu=GPU_ID [--jobs=N] [--exp=EXP_NAME]
#
# Each worker atomically claims pending jobs, runs them, marks done/failed.
# Multiple workers can run on the same or different machines safely.
# Jobs are resumable (checkpoint system in main_cls.py).

PYTHON=/home/eechengyang/anaconda3/envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
QUEUE_BASE="$WORKDIR/jobqueue"
EXP_NAME=grid
GPU=0
MAX_JOBS=99999   # unlimited by default; set to limit jobs per worker invocation

for arg in "$@"; do
    case $arg in
        --gpu=*)      GPU="${arg#--gpu=}" ;;
        --exp=*)      EXP_NAME="${arg#--exp=}" ;;
        --jobs=*)     MAX_JOBS="${arg#--jobs=}" ;;
    esac
done

LOG_DIR="$WORKDIR/logs/worker_gpu${GPU}"
mkdir -p "$LOG_DIR"
cd "$WORKDIR"

echo "========================================================"
echo "  Worker  GPU=$GPU  exp=$EXP_NAME  host=$(hostname)"
echo "  Queue:  $QUEUE_BASE"
echo "  Log:    $LOG_DIR"
echo "========================================================"

jobs_run=0

while true; do
    # Claim a pending job atomically via mv
    job_file=""
    for f in "$QUEUE_BASE/pending"/*; do
        [[ -f "$f" ]] || continue
        jid=$(basename "$f")
        target="$QUEUE_BASE/running/$jid"
        if mv "$f" "$target" 2>/dev/null; then
            job_file="$target"
            break
        fi
    done

    if [[ -z "$job_file" ]]; then
        echo "[$(date '+%H:%M:%S')]  No pending jobs — exiting."
        break
    fi

    # Parse job JSON
    method=$(python3 -c "import json,sys; d=json.load(open('$job_file')); print(d['method'])")
    lr=$(python3     -c "import json,sys; d=json.load(open('$job_file')); print(d['lr'])")
    E=$(python3      -c "import json,sys; d=json.load(open('$job_file')); print(d['E'])")
    T=$(python3      -c "import json,sys; d=json.load(open('$job_file')); print(d['T'])")
    seed=$(python3   -c "import json,sys; d=json.load(open('$job_file')); print(d['seed'])")

    jid=$(basename "$job_file")
    logfile="$LOG_DIR/${jid}.log"

    echo "[$(date '+%H:%M:%S')]  START  GPU=$GPU  $jid"

    "$PYTHON" -u main_cls.py \
        --dataset     FLSkin     \
        --data_path   ../FL_Skin \
        --method      "$method"  \
        --local_model ResNet50   \
        --num_classes 8          \
        --T           "$T"       \
        --E           "$E"       \
        --lr          "$lr"      \
        --batchsize   64         \
        --optimizer   adam       \
        --pretrained             \
        --cosine_lr              \
        --device      "$GPU"     \
        --seed        "$seed"    \
        --exp_name    "$EXP_NAME" \
        > "$logfile" 2>&1
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        mv "$QUEUE_BASE/running/$jid" "$QUEUE_BASE/done/$jid"
        echo "[$(date '+%H:%M:%S')]  DONE   GPU=$GPU  $jid  (exit=0)"
    else
        mv "$QUEUE_BASE/running/$jid" "$QUEUE_BASE/failed/$jid"
        echo "[$(date '+%H:%M:%S')]  FAIL   GPU=$GPU  $jid  (exit=$exit_code)"
        echo "  Log: $logfile"
    fi

    ((jobs_run++)) || true
    if [[ $jobs_run -ge $MAX_JOBS ]]; then
        echo "[$(date '+%H:%M:%S')]  Reached MAX_JOBS=$MAX_JOBS — exiting."
        break
    fi
done

echo "[$(date '+%H:%M:%S')]  Worker done. Ran $jobs_run jobs."
