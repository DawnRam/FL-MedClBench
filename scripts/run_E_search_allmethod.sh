#!/bin/bash
# run_E_search_allmethod.sh — lr=1e-4 × all 14 methods × E∈{1,2,5,10} × seed=0
# Skips already-completed jobs.

METHODS=(LocalTrain FedAvg FedProx MOON FedAWA FedNova PN FedRDN FedLWS FedBN SioBN FedPer FedRoD Ditto)
ES=(10 5 2 1)   # fast first
SEEDS=(0)
LR=1e-4
N_CENTERS=4
GPUS=(0 1 2 3 4 5 6)
JOBS_PER_GPU=2

PYTHON=/nfs/scratch/eechengyang/conda_envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
EXP_NAME=grid
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --jobs_per_gpu=*) JOBS_PER_GPU="${arg#--jobs_per_gpu=}" ;;
    esac
done

cd "$WORKDIR"

is_done() {
    local e=$1 method=$2 seed=$3
    local t=$((100/e))
    local expected=$((1 + t * N_CENTERS))
    local f="$WORKDIR/results/FLSkin/$EXP_NAME/100epoch_${LR}_${e}/${method}/metrics_seed${seed}.csv"
    [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]
}

declare -a Q_E Q_T Q_METHOD Q_SEED
skipped=0; queued=0

for e in "${ES[@]}"; do
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if is_done "$e" "$method" "$seed"; then
                ((skipped++)) || true
            else
                Q_E+=("$e"); Q_T+=($((100/e)))
                Q_METHOD+=("$method"); Q_SEED+=("$seed")
                ((queued++)) || true
            fi
        done
    done
done

n_slots=$(( ${#GPUS[@]} * JOBS_PER_GPU ))
echo "========================================================================"
echo "  All-Method E Search  |  lr=${LR}  E∈{1,2,5,10}  seed=0"
echo "========================================================================"
echo "  Methods    : ${#METHODS[@]}"
echo "  GPUs       : ${GPUS[*]}  (${JOBS_PER_GPU}/GPU → ${n_slots} slots)"
echo "  Skipped    : ${skipped}  Queued: ${queued}"
echo "========================================================================"

if $DRY_RUN; then
    for i in $(seq 0 $((queued-1))); do
        echo "  [DRY] E=${Q_E[$i]} T=${Q_T[$i]}  ${Q_METHOD[$i]}  seed=${Q_SEED[$i]}"
    done; exit 0
fi

[ "$queued" -eq 0 ] && { echo "All done!"; exit 0; }

start_job() {
    local gpu=$1 e=$2 t=$3 method=$4 seed=$5
    local logdir="$WORKDIR/logs/grid_100epoch_${LR}_${e}"
    mkdir -p "$logdir"
    echo "[$(date '+%H:%M:%S')]  START  GPU${gpu}  E=${e} T=${t}  ${method}  seed=${seed}"
    "$PYTHON" -u main_cls.py \
        --dataset FLSkin --data_path ../FL_Skin \
        --method "$method" --local_model ResNet50 --num_classes 8 \
        --T "$t" --E "$e" --lr "$LR" --batchsize 64 \
        --optimizer adam --pretrained --cosine_lr \
        --device "$gpu" --seed "$seed" --exp_name "$EXP_NAME" \
        > "$logdir/${method}_seed${seed}.log" 2>&1 &
}

declare -A SLOT_PID SLOT_LABEL SLOT_GPU
job_idx=0

for slot in $(seq 0 $((n_slots-1))); do
    [ $job_idx -ge $queued ] && break
    gpu=${GPUS[$((slot % ${#GPUS[@]}))]}
    SLOT_GPU[$slot]=$gpu
    start_job "$gpu" "${Q_E[$job_idx]}" "${Q_T[$job_idx]}" "${Q_METHOD[$job_idx]}" "${Q_SEED[$job_idx]}"
    SLOT_PID[$slot]=$!
    SLOT_LABEL[$slot]="E=${Q_E[$job_idx]} ${Q_METHOD[$job_idx]}"
    ((job_idx++)) || true
done

echo "  ${n_slots} slots launched. Polling every 30s..."

while true; do
    sleep 30
    for slot in $(seq 0 $((n_slots-1))); do
        pid="${SLOT_PID[$slot]:-}"; [[ -z "$pid" ]] && continue
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu=${SLOT_GPU[$slot]}
            echo "[$(date '+%H:%M:%S')]  DONE  GPU${gpu}  |  ${SLOT_LABEL[$slot]}"
            unset SLOT_PID[$slot] SLOT_LABEL[$slot]
            if [ $job_idx -lt $queued ]; then
                start_job "$gpu" "${Q_E[$job_idx]}" "${Q_T[$job_idx]}" "${Q_METHOD[$job_idx]}" "${Q_SEED[$job_idx]}"
                SLOT_PID[$slot]=$!
                SLOT_LABEL[$slot]="E=${Q_E[$job_idx]} ${Q_METHOD[$job_idx]}"
                ((job_idx++)) || true
            fi
        fi
    done
    active=0; status=""
    for slot in $(seq 0 $((n_slots-1))); do
        [[ -n "${SLOT_PID[$slot]:-}" ]] && { status+=" [G${SLOT_GPU[$slot]}:${SLOT_LABEL[$slot]}]"; ((active++)) || true; }
    done
    echo "[$(date '+%H:%M:%S')]  Active=${active}  Queue=$((queued-job_idx))${status}"
    all_idle=true
    for slot in $(seq 0 $((n_slots-1))); do [[ -n "${SLOT_PID[$slot]:-}" ]] && { all_idle=false; break; }; done
    $all_idle && [ $job_idx -ge $queued ] && break
done

echo "[$(date '+%H:%M:%S')] Done! Run: python scripts/analyze_E_allmethod.py"
