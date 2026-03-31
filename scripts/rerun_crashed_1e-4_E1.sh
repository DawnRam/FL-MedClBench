#!/bin/bash
# rerun_crashed_1e-4_E1.sh
# Reruns the 10 jobs that crashed during lr=1e-4 E=1 run.
# Crashed jobs: LocalTrain(0,1,2), FedAvg(0,1,2), FedProx(0,1,2), MOON(0)
# Run AFTER the main scheduler (run_skin_grid.sh lr=1e-4 E=1) has finished.

METHODS=(LocalTrain FedAvg FedProx)
EXTRA_METHOD=MOON
EXTRA_SEED=0
SEEDS=(0 1 2)
GPUS=(0 1 2 3 4 5 6 7)

PYTHON=/home/eechengyang/anaconda3/envs/fedcls/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
LOG_DIR=$WORKDIR/logs/grid_100epoch_1e-4_1

cd "$WORKDIR"

declare -a QUEUE_METHOD=()
declare -a QUEUE_SEED=()

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        QUEUE_METHOD+=("$method")
        QUEUE_SEED+=("$seed")
    done
done
# Add MOON seed0
QUEUE_METHOD+=("$EXTRA_METHOD")
QUEUE_SEED+=("$EXTRA_SEED")

n_jobs=${#QUEUE_METHOD[@]}
echo "========================================"
echo "  Rerun crashed jobs: lr=1e-4 E=1"
echo "  Jobs: $n_jobs"
echo "========================================"

start_job() {
    local gpu=$1 method=$2 seed=$3
    local logfile="$LOG_DIR/${method}_seed${seed}.log"
    echo "[$(date '+%H:%M:%S')]  START  GPU $gpu  |  $method  seed=$seed"

    "$PYTHON" -u main_cls.py \
        --dataset     FLSkin          \
        --data_path   ../FL_Skin      \
        --method      "$method"       \
        --local_model ResNet50        \
        --num_classes 8               \
        --T           100             \
        --E           1               \
        --lr          1e-4            \
        --batchsize   64              \
        --optimizer   adam            \
        --pretrained                  \
        --cosine_lr                   \
        --device      "$gpu"          \
        --seed        "$seed"         \
        --exp_name    grid            \
        > "$logfile" 2>&1 &
}

declare -A GPU_PID
declare -A GPU_LABEL
job_idx=0

for gpu in "${GPUS[@]}"; do
    if [ $job_idx -ge $n_jobs ]; then break; fi
    start_job "$gpu" "${QUEUE_METHOD[$job_idx]}" "${QUEUE_SEED[$job_idx]}"
    GPU_PID[$gpu]=$!
    GPU_LABEL[$gpu]="${QUEUE_METHOD[$job_idx]} seed${QUEUE_SEED[$job_idx]}"
    ((job_idx++)) || true
done

echo ""
echo "  Initial jobs launched. Polling every 60s..."
echo ""

while true; do
    sleep 60
    for gpu in "${GPUS[@]}"; do
        pid="${GPU_PID[$gpu]:-}"
        [[ -z "$pid" ]] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')]  DONE   GPU $gpu  |  ${GPU_LABEL[$gpu]}"
            unset GPU_PID[$gpu]; unset GPU_LABEL[$gpu]
            if [ $job_idx -lt $n_jobs ]; then
                start_job "$gpu" "${QUEUE_METHOD[$job_idx]}" "${QUEUE_SEED[$job_idx]}"
                GPU_PID[$gpu]=$!
                GPU_LABEL[$gpu]="${QUEUE_METHOD[$job_idx]} seed${QUEUE_SEED[$job_idx]}"
                ((job_idx++)) || true
            fi
        fi
    done
    active=0; status=""
    for gpu in "${GPUS[@]}"; do
        [[ -n "${GPU_PID[$gpu]:-}" ]] && { status+=" GPU${gpu}:${GPU_LABEL[$gpu]}"; ((active++)) || true; }
    done
    remain=$((n_jobs - job_idx))
    echo "[$(date '+%H:%M:%S')]  Active=$active  Queue=$remain |$status"
    all_idle=true
    for gpu in "${GPUS[@]}"; do
        [[ -n "${GPU_PID[$gpu]:-}" ]] && { all_idle=false; break; }
    done
    $all_idle && [ $job_idx -ge $n_jobs ] && break
done

echo ""
echo "[$(date '+%H:%M:%S')] Rerun complete: $n_jobs jobs done."
