#!/bin/bash
# run_parallel_seeds.sh
#
# Launch all (method × seed) jobs across GPUs 0-7, strictly one job per GPU.
# Uses inline background launch (not command substitution) to correctly track PIDs.
#
# Usage:
#   bash scripts/run_parallel_seeds.sh          # run all 42 pairs
#   bash scripts/run_parallel_seeds.sh --dry-run # print what would run

METHODS=(LocalTrain FedAvg FedProx MOON FedAWA FedNova PN FedRDN FedLWS FedBN SioBN FedPer FedRoD Ditto)
SEEDS=(0 1 2)
GPUS=(0 1 2 3 4 5 6 7)

PYTHON=/home/eechengyang/anaconda3/envs/fedcls/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
LOG_DIR=$WORKDIR/logs/parallel
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] — commands will be printed but not executed"
fi

mkdir -p "$LOG_DIR"
cd "$WORKDIR"

# ─────────────────────────────────────────────────────────────────────────────
# Build full job queue
# ─────────────────────────────────────────────────────────────────────────────
declare -a QUEUE_METHOD=()
declare -a QUEUE_SEED=()
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        QUEUE_METHOD+=("$method")
        QUEUE_SEED+=("$seed")
    done
done

n_jobs=${#QUEUE_METHOD[@]}
echo "========================================================================"
echo "  FL-MedClsBench — Parallel Experiment Launcher"
echo "========================================================================"
echo "  Total jobs   : $n_jobs  (${#METHODS[@]} methods × ${#SEEDS[@]} seeds)"
echo "  GPUs         : ${GPUS[*]}"
echo "  Python       : $PYTHON"
echo "  Work dir     : $WORKDIR"
echo "  Log dir      : $LOG_DIR"
echo "========================================================================"
echo ""

if $DRY_RUN; then
    for i in $(seq 0 $((n_jobs - 1))); do
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        echo "  [DRY-RUN] GPU $gpu  ${QUEUE_METHOD[$i]}  seed=${QUEUE_SEED[$i]}"
    done
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# State tracking (indexed by GPU slot 0..7)
# ─────────────────────────────────────────────────────────────────────────────
declare -A GPU_PID    # gpu_index -> PID (empty = free)
declare -A GPU_LABEL  # gpu_index -> display label

job_idx=0

# Helper: launch one job on a GPU (sets GPU_PID and GPU_LABEL)
# NOTE: NOT a function — uses inline launch so $! is in parent shell
# Call via the loop below.

start_job() {
    local gpu=$1 method=$2 seed=$3
    local logfile="$LOG_DIR/${method}_seed${seed}.log"
    echo "[$(date '+%H:%M:%S')]  START  GPU $gpu  |  $method  seed=$seed  ->  $logfile"

    "$PYTHON" -u main_cls.py \
        --method      "$method"  \
        --local_model ResNet50   \
        --data_path   ../FedBCa  \
        --T           500        \
        --E           5          \
        --lr          0.0001     \
        --batchsize   24         \
        --optimizer   adam       \
        --device      "$gpu"     \
        --seed        "$seed"    \
        --exp_name    bench      \
        > "$logfile" 2>&1 &
    # $! is captured by caller immediately after this function returns
}

# ─────────────────────────────────────────────────────────────────────────────
# Initial fill: launch one job per GPU
# ─────────────────────────────────────────────────────────────────────────────
for gpu in "${GPUS[@]}"; do
    if [ $job_idx -ge $n_jobs ]; then break; fi
    method="${QUEUE_METHOD[$job_idx]}"
    seed="${QUEUE_SEED[$job_idx]}"
    start_job "$gpu" "$method" "$seed"
    GPU_PID[$gpu]=$!
    GPU_LABEL[$gpu]="${method} seed${seed}"
    ((job_idx++)) || true
done

echo ""
echo "  Initial jobs launched. Scheduler polling every 60s..."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler loop: poll every 60s, refill freed GPU slots
# ─────────────────────────────────────────────────────────────────────────────
while true; do
    sleep 60

    # Check each GPU slot
    for gpu in "${GPUS[@]}"; do
        pid="${GPU_PID[$gpu]:-}"
        [[ -z "$pid" ]] && continue

        if ! kill -0 "$pid" 2>/dev/null; then
            # Job finished (or crashed)
            label="${GPU_LABEL[$gpu]:-}"
            echo "[$(date '+%H:%M:%S')]  DONE   GPU $gpu  |  $label  (PID=$pid)"
            unset GPU_PID[$gpu]
            unset GPU_LABEL[$gpu]

            # Launch next queued job on this GPU
            if [ $job_idx -lt $n_jobs ]; then
                method="${QUEUE_METHOD[$job_idx]}"
                seed="${QUEUE_SEED[$job_idx]}"
                start_job "$gpu" "$method" "$seed"
                GPU_PID[$gpu]=$!
                GPU_LABEL[$gpu]="${method} seed${seed}"
                ((job_idx++)) || true
            fi
        fi
    done

    # Status line
    active=0
    status=""
    for gpu in "${GPUS[@]}"; do
        if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
            status+="  GPU${gpu}:${GPU_LABEL[$gpu]}"
            ((active++)) || true
        fi
    done
    remain=$((n_jobs - job_idx))
    echo "[$(date '+%H:%M:%S')]  Active=$active  Queue=$remain  |$status"

    # Done when queue empty and all GPUs idle
    all_idle=true
    for gpu in "${GPUS[@]}"; do
        [[ -n "${GPU_PID[$gpu]:-}" ]] && { all_idle=false; break; }
    done
    if $all_idle && [ $job_idx -ge $n_jobs ]; then
        break
    fi
done

echo ""
echo "[$(date '+%H:%M:%S')] ================================================================"
echo "[$(date '+%H:%M:%S')] All $n_jobs jobs completed."
echo "[$(date '+%H:%M:%S')] Results  -> $WORKDIR/results/"
echo "[$(date '+%H:%M:%S')] Logs     -> $LOG_DIR/"
echo "[$(date '+%H:%M:%S')] Summary  -> python scripts/aggregate_results.py"
echo "[$(date '+%H:%M:%S')] ================================================================"
