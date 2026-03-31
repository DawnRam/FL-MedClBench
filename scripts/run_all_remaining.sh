#!/bin/bash
# run_all_remaining.sh — Launch ALL remaining grid search jobs across all (lr, E) settings.
# Skips jobs that already have complete metrics files.
# Grid: lr ∈ {1e-4,1e-3,1e-2,1e-1} × E ∈ {1,2,5,10} × 14 methods × 3 seeds = 672 total
#
# Usage: bash scripts/run_all_remaining.sh [--gpus=0,1,2,3,4,5,6] [--jobs_per_gpu=1] [--dry-run]

METHODS=(LocalTrain FedAvg FedProx MOON FedAWA FedNova PN FedRDN FedLWS FedBN SioBN FedPer FedRoD Ditto)
SEEDS=(0 1 2)
LRS=(1e-4 1e-3 1e-2 1e-1)
ES=(1 2 5 10)
GPUS=(0 1 2 3 4 5 6)
JOBS_PER_GPU=1

PYTHON=/nfs/scratch/eechengyang/conda_envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
EXP_NAME=grid
DRY_RUN=false
N_CENTERS=4   # FLSkin has 4 centers

for arg in "$@"; do
    case $arg in
        --gpus=*)         IFS=',' read -ra GPUS <<< "${arg#--gpus=}" ;;
        --jobs_per_gpu=*) JOBS_PER_GPU="${arg#--jobs_per_gpu=}" ;;
        --dry-run)        DRY_RUN=true ;;
    esac
done

cd "$WORKDIR"

# ── Build job queue (only unfinished jobs) ────────────────────────────────────
declare -a Q_LR Q_E Q_T Q_METHOD Q_SEED

is_done() {
    local lr=$1 e=$2 method=$3 seed=$4
    local t=$((100 / e))
    local expected_lines=$(( 1 + t * N_CENTERS ))
    local metrics="$WORKDIR/results/FLSkin/$EXP_NAME/${t}0epoch_${lr}_${e}/${method}/metrics_seed${seed}.csv"
    # Note: log dir uses total_epochs = t*e = 100, but result dir uses t+"0"? Let's check both
    # Actually result dir: results/FLSkin/{exp_name}/{T_ROUNDS}epoch_{LR}_{E}/{method}/
    local metrics2="$WORKDIR/results/FLSkin/$EXP_NAME/${t}epoch_${lr}_${e}/${method}/metrics_seed${seed}.csv"
    local actual_lines=0
    if [ -f "$metrics" ]; then
        actual_lines=$(wc -l < "$metrics")
    elif [ -f "$metrics2" ]; then
        actual_lines=$(wc -l < "$metrics2")
    fi
    [ "$actual_lines" -ge "$expected_lines" ]
}

# Discover actual result base dir pattern from existing results
# Pattern: results/FLSkin/grid/100epoch_1e-4_2/ → T=50epoch? No, 100epoch means total=100 (T*E)
# From script: LOG_DIR=$WORKDIR/logs/grid_${TOTAL_EPOCHS}epoch_${LR}_${E_LOCAL}
# TOTAL_EPOCHS=$((T_ROUNDS * E_LOCAL)) = 100
# Result dir: results/FLSkin/{exp_name}/{T_ROUNDS}epoch_{LR}_{E_LOCAL}/
# So for E=2 T=50: results/FLSkin/grid/50epoch_1e-4_2/ -- but we see "100epoch_1e-4_2"
# Let me re-check: the script sets LOG_DIR with TOTAL_EPOCHS but result dir uses T_ROUNDS
# Actually the main_cls.py saves to: results/{dataset}/{exp_name}/{T}epoch_{lr}_{E}/
# We see "100epoch_1e-4_2" for E=2,T=50... so it's total_epochs=100 in the dir name!

is_done2() {
    local lr=$1 e=$2 method=$3 seed=$4
    local t=$((100 / e))
    local total_epochs=100  # always 100 (T*E)
    local expected_lines=$(( 1 + t * N_CENTERS ))
    local metrics="$WORKDIR/results/FLSkin/$EXP_NAME/${total_epochs}epoch_${lr}_${e}/${method}/metrics_seed${seed}.csv"
    local actual_lines=0
    if [ -f "$metrics" ]; then
        actual_lines=$(wc -l < "$metrics")
    fi
    [ "$actual_lines" -ge "$expected_lines" ]
}

skipped=0
queued=0

for lr in "${LRS[@]}"; do
    for e in "${ES[@]}"; do
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if is_done2 "$lr" "$e" "$method" "$seed"; then
                    ((skipped++)) || true
                else
                    Q_LR+=("$lr")
                    Q_E+=("$e")
                    Q_T+=($((100 / e)))
                    Q_METHOD+=("$method")
                    Q_SEED+=("$seed")
                    ((queued++)) || true
                fi
            done
        done
    done
done

n_slots=$(( ${#GPUS[@]} * JOBS_PER_GPU ))
echo "========================================================================"
echo "  FL_Skin Grid — Run All Remaining"
echo "========================================================================"
echo "  GPUs       : ${GPUS[*]}  (${JOBS_PER_GPU}/GPU → ${n_slots} slots)"
echo "  Skipped    : ${skipped} (already done)"
echo "  Queued     : ${queued} remaining jobs"
echo "  Python     : $PYTHON"
echo "========================================================================"

if $DRY_RUN; then
    for i in $(seq 0 $((queued - 1))); do
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        echo "  [DRY] GPU${gpu}  lr=${Q_LR[$i]}  E=${Q_E[$i]}  T=${Q_T[$i]}  ${Q_METHOD[$i]}  seed=${Q_SEED[$i]}"
    done
    exit 0
fi

[ "$queued" -eq 0 ] && { echo "Nothing to do!"; exit 0; }

start_job() {
    local slot=$1 gpu=$2 lr=$3 e=$4 t=$5 method=$6 seed=$7
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

for slot in $(seq 0 $((n_slots - 1))); do
    [ $job_idx -ge $queued ] && break
    gpu=${GPUS[$((slot % ${#GPUS[@]}))]}
    SLOT_GPU[$slot]=$gpu
    start_job "$slot" "$gpu" "${Q_LR[$job_idx]}" "${Q_E[$job_idx]}" "${Q_T[$job_idx]}" "${Q_METHOD[$job_idx]}" "${Q_SEED[$job_idx]}"
    SLOT_PID[$slot]=$!
    SLOT_LABEL[$slot]="lr=${Q_LR[$job_idx]} E=${Q_E[$job_idx]} ${Q_METHOD[$job_idx]} s${Q_SEED[$job_idx]}"
    ((job_idx++)) || true
done

echo ""
echo "  Initial ${n_slots} slots launched. Polling every 60s..."
echo ""

while true; do
    sleep 60
    for slot in $(seq 0 $((n_slots - 1))); do
        pid="${SLOT_PID[$slot]:-}"
        [[ -z "$pid" ]] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu=${SLOT_GPU[$slot]}
            echo "[$(date '+%H:%M:%S')]  DONE  GPU${gpu}  |  ${SLOT_LABEL[$slot]}"
            unset SLOT_PID[$slot] SLOT_LABEL[$slot]
            if [ $job_idx -lt $queued ]; then
                start_job "$slot" "$gpu" "${Q_LR[$job_idx]}" "${Q_E[$job_idx]}" "${Q_T[$job_idx]}" "${Q_METHOD[$job_idx]}" "${Q_SEED[$job_idx]}"
                SLOT_PID[$slot]=$!
                SLOT_LABEL[$slot]="lr=${Q_LR[$job_idx]} E=${Q_E[$job_idx]} ${Q_METHOD[$job_idx]} s${Q_SEED[$job_idx]}"
                ((job_idx++)) || true
            fi
        fi
    done
    active=0; status=""
    for slot in $(seq 0 $((n_slots - 1))); do
        if [[ -n "${SLOT_PID[$slot]:-}" ]]; then
            status+=" [G${SLOT_GPU[$slot]}:${SLOT_LABEL[$slot]}]"
            ((active++)) || true
        fi
    done
    remain=$((queued - job_idx))
    echo "[$(date '+%H:%M:%S')]  Active=${active}  Queue=${remain}${status}"
    all_idle=true
    for slot in $(seq 0 $((n_slots - 1))); do
        [[ -n "${SLOT_PID[$slot]:-}" ]] && { all_idle=false; break; }
    done
    $all_idle && [ $job_idx -ge $queued ] && break
done

echo ""
echo "[$(date '+%H:%M:%S')] ALL ${queued} remaining jobs completed!"
