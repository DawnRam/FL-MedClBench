#!/bin/bash
# worker_filtered.sh — Worker that only claims jobs matching a filter pattern.
#
# Usage:
#   bash scripts/worker_filtered.sh --gpu=GPU_ID --filter=PATTERN [--exp=grid]
#
# FILTER is a substring match against the job ID (e.g. "__1e-4__1__")

PYTHON=/home/eechengyang/anaconda3/envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
QUEUE_BASE="$WORKDIR/jobqueue"
EXP_NAME=grid
GPU=0
FILTER=""

for arg in "$@"; do
    case $arg in
        --gpu=*)    GPU="${arg#--gpu=}" ;;
        --exp=*)    EXP_NAME="${arg#--exp=}" ;;
        --filter=*) FILTER="${arg#--filter=}" ;;
    esac
done

LOG_DIR="$WORKDIR/logs/worker_gpu${GPU}"
PIDFILE="$WORKDIR/jobqueue/workers.pid"
mkdir -p "$LOG_DIR"
cd "$WORKDIR"

# Register PID for clean shutdown via stop_workers.sh
echo "$$" >> "$PIDFILE"
trap "sed -i '/^$$$/d' '$PIDFILE'" EXIT

echo "[$(date '+%H:%M:%S')]  FilterWorker GPU=$GPU filter='$FILTER' PID=$$"

while true; do
    job_file=""
    for f in "$QUEUE_BASE/pending"/*; do
        [[ -f "$f" ]] || continue
        jid=$(basename "$f")
        # Apply filter
        [[ -n "$FILTER" && "$jid" != *"$FILTER"* ]] && continue
        target="$QUEUE_BASE/running/$jid"
        if mv "$f" "$target" 2>/dev/null; then
            job_file="$target"
            break
        fi
    done

    if [[ -z "$job_file" ]]; then
        echo "[$(date '+%H:%M:%S')]  No matching pending jobs — exiting."
        break
    fi

    method=$(python3 -c "import json; d=json.load(open('$job_file')); print(d['method'])")
    lr=$(python3     -c "import json; d=json.load(open('$job_file')); print(d['lr'])")
    E=$(python3      -c "import json; d=json.load(open('$job_file')); print(d['E'])")
    T=$(python3      -c "import json; d=json.load(open('$job_file')); print(d['T'])")
    seed=$(python3   -c "import json; d=json.load(open('$job_file')); print(d['seed'])")
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
        echo "[$(date '+%H:%M:%S')]  DONE   GPU=$GPU  $jid"
    else
        mv "$QUEUE_BASE/running/$jid" "$QUEUE_BASE/failed/$jid"
        echo "[$(date '+%H:%M:%S')]  FAIL   GPU=$GPU  $jid  (exit=$exit_code)"
    fi
done
