#!/bin/bash
# run_skin_queued.sh
#
# Launches the remaining 18 queued FL_Skin jobs (FedLWS/FedBN/SioBN/FedPer/FedRoD/Ditto)
# as a second slot on each GPU, running in parallel with whatever is already there.
# Polls every 60s; as slots free up, dispatches the next job.

METHODS=(FedLWS FedBN SioBN FedPer FedRoD Ditto)
SEEDS=(0 1 2)
GPUS=(0 1 2 3 4 5 6 7)

PYTHON=/home/eechengyang/anaconda3/envs/fedcls/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
LOG_DIR=$WORKDIR/logs/skin

mkdir -p "$LOG_DIR"
cd "$WORKDIR"

declare -a QUEUE_METHOD=()
declare -a QUEUE_SEED=()
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        QUEUE_METHOD+=("$method")
        QUEUE_SEED+=("$seed")
    done
done

n_jobs=${#QUEUE_METHOD[@]}
echo "========================================================"
echo "  FL_Skin Queued Job Launcher (slot-2 on each GPU)"
echo "========================================================"
echo "  Jobs   : $n_jobs  (${#METHODS[@]} methods × ${#SEEDS[@]} seeds)"
echo "  GPUs   : ${GPUS[*]}"
echo "========================================================"

start_job() {
    local gpu=$1 method=$2 seed=$3
    local logfile="$LOG_DIR/${method}_seed${seed}.log"
    echo "[$(date '+%H:%M:%S')]  START  GPU $gpu  |  $method  seed=$seed  ->  $logfile"

    "$PYTHON" -u main_cls.py \
        --dataset     FLSkin          \
        --data_path   ../FL_Skin      \
        --method      "$method"       \
        --local_model ResNet50        \
        --num_classes 8               \
        --T           200             \
        --E           1               \
        --lr          0.0001          \
        --batchsize   64              \
        --optimizer   adam            \
        --device      "$gpu"          \
        --seed        "$seed"         \
        --exp_name    skin            \
        >> "$logfile" 2>&1 &
}

declare -A SLOT_PID
declare -A SLOT_LABEL

job_idx=0

# Fill one slot per GPU immediately
for gpu in "${GPUS[@]}"; do
    if [ $job_idx -ge $n_jobs ]; then break; fi
    method="${QUEUE_METHOD[$job_idx]}"
    seed="${QUEUE_SEED[$job_idx]}"
    start_job "$gpu" "$method" "$seed"
    SLOT_PID[$gpu]=$!
    SLOT_LABEL[$gpu]="${method} seed${seed}"
    ((job_idx++)) || true
done

echo ""
echo "  Initial 8 jobs launched. Polling every 60s..."
echo ""

while true; do
    sleep 60

    for gpu in "${GPUS[@]}"; do
        pid="${SLOT_PID[$gpu]:-}"
        [[ -z "$pid" ]] && continue

        if ! kill -0 "$pid" 2>/dev/null; then
            label="${SLOT_LABEL[$gpu]:-}"
            echo "[$(date '+%H:%M:%S')]  DONE   GPU $gpu  |  $label  (PID=$pid)"
            unset SLOT_PID[$gpu]
            unset SLOT_LABEL[$gpu]

            if [ $job_idx -lt $n_jobs ]; then
                method="${QUEUE_METHOD[$job_idx]}"
                seed="${QUEUE_SEED[$job_idx]}"
                start_job "$gpu" "$method" "$seed"
                SLOT_PID[$gpu]=$!
                SLOT_LABEL[$gpu]="${method} seed${seed}"
                ((job_idx++)) || true
            fi
        fi
    done

    active=0; status=""
    for gpu in "${GPUS[@]}"; do
        if [[ -n "${SLOT_PID[$gpu]:-}" ]]; then
            status+="  GPU${gpu}:${SLOT_LABEL[$gpu]}"
            ((active++)) || true
        fi
    done
    remain=$((n_jobs - job_idx))
    echo "[$(date '+%H:%M:%S')]  Slot2 Active=$active  Queue=$remain  |$status"

    all_idle=true
    for gpu in "${GPUS[@]}"; do
        [[ -n "${SLOT_PID[$gpu]:-}" ]] && { all_idle=false; break; }
    done
    if $all_idle && [ $job_idx -ge $n_jobs ]; then break; fi
done

echo ""
echo "[$(date '+%H:%M:%S')] All $n_jobs queued jobs completed."
