#!/bin/bash
# run_skin_grid.sh — Grid search over lr × E for FL_Skin
#
# Usage:
#   bash scripts/run_skin_grid.sh [--lr LR] [--E E] [--dry-run]
#
# Defaults: lr=1e-4, E=1 (first setting)
# Grid: lr ∈ {1e-1,1e-2,1e-3,1e-4}  E ∈ {1,2,5,10}
#   Each (lr,E) setting: 14 methods × 3 seeds = 42 jobs
#   T (comm rounds) = 100 / E  (total local epochs = 100 for all settings)
#
# Directory: results/FLSkin/{exp_name}/lr{lr}_E{E}/{method}/

METHODS=(LocalTrain FedAvg FedProx MOON FedAWA FedNova PN FedRDN FedLWS FedBN SioBN FedPer FedRoD Ditto)
SEEDS=(0 1 2)
GPUS=(0 1 2 3 4 5 6)
JOBS_PER_GPU=1

PYTHON=/home/eechengyang/anaconda3/envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
EXP_NAME=grid
DRY_RUN=false

# ── Parse arguments ──────────────────────────────────────────────────────────
LR=1e-4
E_LOCAL=1

for arg in "$@"; do
    case $arg in
        --lr=*)           LR="${arg#--lr=}" ;;
        --E=*)            E_LOCAL="${arg#--E=}" ;;
        --exp=*)          EXP_NAME="${arg#--exp=}" ;;
        --gpus=*)         IFS=',' read -ra GPUS <<< "${arg#--gpus=}" ;;
        --jobs_per_gpu=*) JOBS_PER_GPU="${arg#--jobs_per_gpu=}" ;;
        --dry-run)        DRY_RUN=true ;;
    esac
done

# T = 100 / E (total local epochs = 100)
T_ROUNDS=$((100 / E_LOCAL))

TOTAL_EPOCHS=$((T_ROUNDS * E_LOCAL))
LOG_DIR=$WORKDIR/logs/grid_${TOTAL_EPOCHS}epoch_${LR}_${E_LOCAL}
mkdir -p "$LOG_DIR"
cd "$WORKDIR"

# ── Job queue ─────────────────────────────────────────────────────────────────
declare -a QUEUE_METHOD=()
declare -a QUEUE_SEED=()
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        QUEUE_METHOD+=("$method")
        QUEUE_SEED+=("$seed")
    done
done

n_jobs=${#QUEUE_METHOD[@]}
n_slots=$(( ${#GPUS[@]} * JOBS_PER_GPU ))
echo "========================================================================"
echo "  FL_Skin Grid Launcher"
echo "========================================================================"
echo "  Setting    : lr=${LR}  E=${E_LOCAL}  T=${T_ROUNDS}  (total epochs=100)"
echo "  Pretrained : yes (ImageNet ResNet50)"
echo "  Cosine LR  : yes"
echo "  Methods    : ${#METHODS[@]}  Seeds: ${#SEEDS[@]}  Jobs: ${n_jobs}"
echo "  GPUs       : ${GPUS[*]}  (${JOBS_PER_GPU} job/GPU → ${n_slots} slots)"
echo "  Log dir    : ${LOG_DIR}"
echo "  Exp name   : ${EXP_NAME}"
echo "========================================================================"

if $DRY_RUN; then
    for i in $(seq 0 $((n_jobs - 1))); do
        gpu=${GPUS[$((i % ${#GPUS[@]}))]}
        echo "  [DRY] GPU$gpu  ${QUEUE_METHOD[$i]}  seed=${QUEUE_SEED[$i]}  T=$T_ROUNDS  E=$E_LOCAL  lr=$LR"
    done
    exit 0
fi

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
        --T           "$T_ROUNDS"     \
        --E           "$E_LOCAL"      \
        --lr          "$LR"           \
        --batchsize   64              \
        --optimizer   adam            \
        --pretrained                  \
        --cosine_lr                   \
        --device      "$gpu"          \
        --seed        "$seed"         \
        --exp_name    "$EXP_NAME"     \
        > "$logfile" 2>&1 &
}

# Use slot indices as keys (supports multiple jobs per GPU)
declare -A SLOT_PID
declare -A SLOT_LABEL
declare -A SLOT_GPU
job_idx=0

for slot in $(seq 0 $((n_slots - 1))); do
    if [ $job_idx -ge $n_jobs ]; then break; fi
    gpu=${GPUS[$((slot % ${#GPUS[@]}))]}
    SLOT_GPU[$slot]=$gpu
    start_job "$gpu" "${QUEUE_METHOD[$job_idx]}" "${QUEUE_SEED[$job_idx]}"
    SLOT_PID[$slot]=$!
    SLOT_LABEL[$slot]="${QUEUE_METHOD[$job_idx]} seed${QUEUE_SEED[$job_idx]}"
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
            echo "[$(date '+%H:%M:%S')]  DONE   GPU $gpu  |  ${SLOT_LABEL[$slot]}"
            unset SLOT_PID[$slot]; unset SLOT_LABEL[$slot]
            if [ $job_idx -lt $n_jobs ]; then
                start_job "$gpu" "${QUEUE_METHOD[$job_idx]}" "${QUEUE_SEED[$job_idx]}"
                SLOT_PID[$slot]=$!
                SLOT_LABEL[$slot]="${QUEUE_METHOD[$job_idx]} seed${QUEUE_SEED[$job_idx]}"
                ((job_idx++)) || true
            fi
        fi
    done
    active=0; status=""
    for slot in $(seq 0 $((n_slots - 1))); do
        if [[ -n "${SLOT_PID[$slot]:-}" ]]; then
            gpu=${SLOT_GPU[$slot]}
            status+=" G${gpu}:${SLOT_LABEL[$slot]}"
            ((active++)) || true
        fi
    done
    remain=$((n_jobs - job_idx))
    echo "[$(date '+%H:%M:%S')]  Active=$active  Queue=$remain |$status"
    all_idle=true
    for slot in $(seq 0 $((n_slots - 1))); do
        [[ -n "${SLOT_PID[$slot]:-}" ]] && { all_idle=false; break; }
    done
    $all_idle && [ $job_idx -ge $n_jobs ] && break
done

echo ""
echo "[$(date '+%H:%M:%S')] All $n_jobs jobs done: lr=$LR E=$E_LOCAL T=$T_ROUNDS"
echo "[$(date '+%H:%M:%S')] Results -> results/FLSkin/$EXP_NAME/${T_ROUNDS}epoch_${LR}_${E_LOCAL}/"
