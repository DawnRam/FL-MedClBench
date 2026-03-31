#!/bin/bash
# run_pn_watch.sh — Wait for GPUs to free up, then run PN with bs=64.
# Requires 8 GB free (GPU used < 16576 MiB). One job per GPU slot.
# E_search was killed; as its jobs finish, slots open up for PN.

PYTHON=/nfs/scratch/eechengyang/conda_envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
LR=1e-4
N_CENTERS=4
BATCHSIZE=64
MEM_FREE_NEEDED=9000   # MiB — higher to avoid GPU6 when 2 big jobs co-reside

cd "$WORKDIR"

declare -a Q_METHOD Q_E Q_T Q_SEED

for seed in 0 1 2; do
    f="results/FLSkin/grid/100epoch_${LR}_1/PN/metrics_seed${seed}.csv"
    lk="results/FLSkin/grid/100epoch_${LR}_1/PN/seed${seed}.lock"
    expected=$((1 + 100 * N_CENTERS))
    if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]; then
        echo "SKIP E=1 PN seed=${seed} (done)"
    elif [ -f "$lk" ]; then
        echo "SKIP E=1 PN seed=${seed} (lock exists — already running)"
    else
        Q_METHOD+=(PN); Q_E+=(1); Q_T+=(100); Q_SEED+=("$seed")
    fi
done
for e in 2 5 10; do
    t=$((100/e))
    f="results/FLSkin/grid/100epoch_${LR}_${e}/PN/metrics_seed0.csv"
    lk="results/FLSkin/grid/100epoch_${LR}_${e}/PN/seed0.lock"
    expected=$((1 + t * N_CENTERS))
    if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]; then
        echo "SKIP E=${e} PN seed=0 (done)"
    elif [ -f "$lk" ]; then
        echo "SKIP E=${e} PN seed=0 (lock exists — already running)"
    else
        Q_METHOD+=(PN); Q_E+=("$e"); Q_T+=("$t"); Q_SEED+=(0)
    fi
done

f="results/FLSkin/grid/100epoch_${LR}_10/FedLWS/metrics_seed0.csv"
lk="results/FLSkin/grid/100epoch_${LR}_10/FedLWS/seed0.lock"
expected=$((1 + 10 * N_CENTERS))
if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]; then
    echo "SKIP FedLWS E=10 seed=0 (done)"
elif [ -f "$lk" ]; then
    echo "SKIP FedLWS E=10 seed=0 (lock exists — already running)"
else
    Q_METHOD+=(FedLWS); Q_E+=(10); Q_T+=(10); Q_SEED+=(0)
fi

queued=${#Q_METHOD[@]}
echo "==========================================================="
echo "  PN watcher  bs=${BATCHSIZE}  need ${MEM_FREE_NEEDED}MiB free"
echo "  ${queued} jobs. One job per GPU slot. Polling every 60s..."
echo "==========================================================="
[ "$queued" -eq 0 ] && { echo "Nothing to do!"; exit 0; }

declare -A GPU_PID

best_free_gpu() {
    local busy=""
    for gpu in "${!GPU_PID[@]}"; do
        pid="${GPU_PID[$gpu]}"
        if kill -0 "$pid" 2>/dev/null; then
            busy="$busy $gpu"
        else
            echo "[$(date '+%H:%M:%S')]  DONE  GPU${gpu}" >&2
            unset GPU_PID[$gpu]
        fi
    done
    nvidia-smi --query-gpu=index,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F'[, ]+' -v need=$MEM_FREE_NEEDED -v busy="$busy" \
        'BEGIN{split(busy,ba); for(i in ba) skip[ba[i]]=1}
         {free=$3-$2; if(free>=need && !skip[$1]) print free, $1}' | \
        sort -rn | head -1 | awk '{print $2}'
}

job_idx=0
while [ $job_idx -lt $queued ]; do
    gpu=$(best_free_gpu)
    if [ -n "$gpu" ]; then
        method="${Q_METHOD[$job_idx]}"
        e="${Q_E[$job_idx]}"
        t="${Q_T[$job_idx]}"
        seed="${Q_SEED[$job_idx]}"
        logdir="$WORKDIR/logs/grid_100epoch_${LR}_${e}"
        mkdir -p "$logdir"
        echo "[$(date '+%H:%M:%S')]  START  GPU${gpu}  ${method}  E=${e}  seed=${seed}  bs=${BATCHSIZE}"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON" -u main_cls.py \
            --dataset FLSkin --data_path ../FL_Skin \
            --method "$method" --local_model ResNet50 --num_classes 8 \
            --T "$t" --E "$e" --lr "$LR" --batchsize "$BATCHSIZE" \
            --optimizer adam --pretrained --cosine_lr \
            --device "$gpu" --seed "$seed" --exp_name grid \
            > "$logdir/${method}_seed${seed}.log" 2>&1 &
        pid=$!
        GPU_PID[$gpu]=$pid
        ((job_idx++)) || true
        echo "[$(date '+%H:%M:%S')]  PID=${pid} on GPU${gpu}. Remaining=$((queued-job_idx))"
        sleep 90  # let GPU memory fill before checking again
    else
        echo "[$(date '+%H:%M:%S')]  Waiting (slots=${#GPU_PID[@]}, remaining=$((queued-job_idx)))..."
        sleep 60
    fi
done

echo "[$(date '+%H:%M:%S')]  All ${queued} dispatched. Waiting..."
for gpu in "${!GPU_PID[@]}"; do
    wait "${GPU_PID[$gpu]}" 2>/dev/null || true
done
echo "[$(date '+%H:%M:%S')]  PN watcher done! Run compare_methods_E1.py to see updated results."
