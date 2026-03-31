#!/bin/bash
# run_lrE_search.sh — Quick lr × E grid search for LocalTrain and FedAvg only.
# Priority: E=10 (T=10) → E=5 → E=2 → E=1 (fast jobs first).
# Skips already-completed jobs (checks metrics_seed*.csv line count).
#
# Usage: bash scripts/run_lrE_search.sh [--gpus=0,1,2,3,4,5,6] [--dry-run]

METHODS=(LocalTrain FedAvg)
SEEDS=(0)
LRS=(1e-4 1e-3 1e-2 1e-1)
ES=(10 5 2 1)       # fast → slow
N_CENTERS=4
GPUS=(0 1 2 3 4 5 6)
JOBS_PER_GPU=1

PYTHON=/nfs/scratch/eechengyang/conda_envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
EXP_NAME=grid
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --gpus=*)         IFS=',' read -ra GPUS <<< "${arg#--gpus=}" ;;
        --jobs_per_gpu=*) JOBS_PER_GPU="${arg#--jobs_per_gpu=}" ;;
        --dry-run)        DRY_RUN=true ;;
    esac
done

cd "$WORKDIR"

is_done() {
    local lr=$1 e=$2 method=$3 seed=$4
    local t=$((100 / e))
    local expected=$(( 1 + t * N_CENTERS ))
    local metrics="$WORKDIR/results/FLSkin/$EXP_NAME/100epoch_${lr}_${e}/${method}/metrics_seed${seed}.csv"
    [ -f "$metrics" ] && [ "$(wc -l < "$metrics")" -ge "$expected" ]
}

declare -a Q_LR Q_E Q_T Q_METHOD Q_SEED
skipped=0; queued=0

# Build queue: E=10→5→2→1, all lr, both methods, 3 seeds
for e in "${ES[@]}"; do
    for lr in "${LRS[@]}"; do
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if is_done "$lr" "$e" "$method" "$seed"; then
                    ((skipped++)) || true
                else
                    Q_LR+=("$lr"); Q_E+=("$e"); Q_T+=($((100/e)))
                    Q_METHOD+=("$method"); Q_SEED+=("$seed")
                    ((queued++)) || true
                fi
            done
        done
    done
done

n_slots=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

echo "========================================================================"
echo "  FL_Skin  lr×E Search  |  LocalTrain + FedAvg"
echo "========================================================================"
echo "  LRs        : ${LRS[*]}"
echo "  Es (order) : ${ES[*]}  (fast → slow)"
echo "  GPUs       : ${GPUS[*]}  (${JOBS_PER_GPU}/GPU → ${n_slots} slots)"
echo "  Skipped    : ${skipped} (already done)"
echo "  Queued     : ${queued} remaining jobs"
echo "========================================================================"

if $DRY_RUN; then
    for i in $(seq 0 $((queued-1))); do
        echo "  [DRY] GPU${GPUS[$((i%${#GPUS[@]}))]}  lr=${Q_LR[$i]} E=${Q_E[$i]} T=${Q_T[$i]}  ${Q_METHOD[$i]} seed=${Q_SEED[$i]}"
    done
    exit 0
fi

[ "$queued" -eq 0 ] && { echo "All done!"; exit 0; }

start_job() {
    local gpu=$1 lr=$2 e=$3 t=$4 method=$5 seed=$6
    local logdir="$WORKDIR/logs/grid_100epoch_${lr}_${e}"
    mkdir -p "$logdir"
    local logfile="$logdir/${method}_seed${seed}.log"
    echo "[$(date '+%H:%M:%S')]  START  GPU${gpu}  lr=${lr} E=${e} T=${t}  ${method}  seed=${seed}"
    "$PYTHON" -u main_cls.py \
        --dataset     FLSkin          \
        --data_path   ../FL_Skin      \
        --method      "$method"       \
        --local_model ResNet50        \
        --num_classes 8               \
        --T           "$t"            \
        --E           "$e"            \
        --lr          "$lr"           \
        --batchsize   64              \
        --optimizer   adam            \
        --pretrained                  \
        --cosine_lr                   \
        --device      "$gpu"          \
        --seed        "$seed"         \
        --exp_name    "$EXP_NAME"     \
        > "$logfile" 2>&1 &
}

declare -A SLOT_PID SLOT_LABEL SLOT_GPU
job_idx=0

for slot in $(seq 0 $((n_slots-1))); do
    [ $job_idx -ge $queued ] && break
    gpu=${GPUS[$((slot % ${#GPUS[@]}))]}
    SLOT_GPU[$slot]=$gpu
    start_job "$gpu" "${Q_LR[$job_idx]}" "${Q_E[$job_idx]}" "${Q_T[$job_idx]}" "${Q_METHOD[$job_idx]}" "${Q_SEED[$job_idx]}"
    SLOT_PID[$slot]=$!
    SLOT_LABEL[$slot]="lr=${Q_LR[$job_idx]} E=${Q_E[$job_idx]} ${Q_METHOD[$job_idx]} s${Q_SEED[$job_idx]}"
    ((job_idx++)) || true
done

echo ""
echo "  Initial ${n_slots} slots launched. Polling every 30s..."

while true; do
    sleep 30
    for slot in $(seq 0 $((n_slots-1))); do
        pid="${SLOT_PID[$slot]:-}"; [[ -z "$pid" ]] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu=${SLOT_GPU[$slot]}
            echo "[$(date '+%H:%M:%S')]  DONE  GPU${gpu}  |  ${SLOT_LABEL[$slot]}"
            unset SLOT_PID[$slot] SLOT_LABEL[$slot]
            if [ $job_idx -lt $queued ]; then
                start_job "$gpu" "${Q_LR[$job_idx]}" "${Q_E[$job_idx]}" "${Q_T[$job_idx]}" "${Q_METHOD[$job_idx]}" "${Q_SEED[$job_idx]}"
                SLOT_PID[$slot]=$!
                SLOT_LABEL[$slot]="lr=${Q_LR[$job_idx]} E=${Q_E[$job_idx]} ${Q_METHOD[$job_idx]} s${Q_SEED[$job_idx]}"
                ((job_idx++)) || true
            fi
        fi
    done
    active=0; status=""
    for slot in $(seq 0 $((n_slots-1))); do
        [[ -n "${SLOT_PID[$slot]:-}" ]] && { status+=" [G${SLOT_GPU[$slot]}:${SLOT_LABEL[$slot]}]"; ((active++)) || true; }
    done
    remain=$((queued - job_idx))
    echo "[$(date '+%H:%M:%S')]  Active=${active}  Queue=${remain}${status}"
    all_idle=true
    for slot in $(seq 0 $((n_slots-1))); do [[ -n "${SLOT_PID[$slot]:-}" ]] && { all_idle=false; break; }; done
    $all_idle && [ $job_idx -ge $queued ] && break
done

echo ""
echo "[$(date '+%H:%M:%S')] lr×E search complete! Run scripts/analyze_lrE.py to see results."
