#!/bin/bash
# run_pn_retry.sh — Rerun PN (all failed) after GPUs free up.
# Waits for a GPU with < 14000 MiB used before launching each job.

PYTHON=/nfs/scratch/eechengyang/conda_envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
LR=1e-4
N_CENTERS=4

cd "$WORKDIR"

declare -a Q_METHOD Q_E Q_T Q_SEED

for method in PN; do
    for seed in 0 1 2; do
        f="results/FLSkin/grid/100epoch_${LR}_1/${method}/metrics_seed${seed}.csv"
        expected=$((1 + 100 * N_CENTERS))
        if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]; then
            echo "SKIP E=1 ${method} seed=${seed}"
        else
            Q_METHOD+=("$method"); Q_E+=(1); Q_T+=(100); Q_SEED+=("$seed")
        fi
    done
    for e in 2 5 10; do
        t=$((100/e))
        f="results/FLSkin/grid/100epoch_${LR}_${e}/${method}/metrics_seed0.csv"
        expected=$((1 + t * N_CENTERS))
        if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]; then
            echo "SKIP E=${e} ${method} seed=0"
        else
            Q_METHOD+=("$method"); Q_E+=("$e"); Q_T+=("$t"); Q_SEED+=(0)
        fi
    done
done

# Also check FedLWS E=10 which failed
f="results/FLSkin/grid/100epoch_${LR}_10/FedLWS/metrics_seed0.csv"
expected=$((1 + 10 * N_CENTERS))
if ! ([ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]); then
    Q_METHOD+=(FedLWS); Q_E+=(10); Q_T+=(10); Q_SEED+=(0)
fi

queued=${#Q_METHOD[@]}
echo "======================================================="
echo "  PN retry + FedLWS E=10  (${queued} jobs)"
echo "  Waiting for GPU with < 14000 MiB used..."
echo "======================================================="
[ "$queued" -eq 0 ] && { echo "All done!"; exit 0; }

free_gpu() {
    nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F'[, ]+' '$2<14000{print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

declare -a RUNNING_PIDS=()

for ((i=0; i<queued; i++)); do
    method="${Q_METHOD[$i]}"
    e="${Q_E[$i]}"
    t="${Q_T[$i]}"
    seed="${Q_SEED[$i]}"

    while true; do
        # Reap finished
        new_pids=()
        for pid in "${RUNNING_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        RUNNING_PIDS=("${new_pids[@]}")

        gpu=$(free_gpu)
        if [ -n "$gpu" ]; then
            break
        fi
        echo "[$(date '+%H:%M:%S')]  Waiting for free GPU (active=${#RUNNING_PIDS[@]})..."
        sleep 60
    done

    logdir="$WORKDIR/logs/grid_100epoch_${LR}_${e}"
    mkdir -p "$logdir"
    echo "[$(date '+%H:%M:%S')]  START  GPU${gpu}  ${method}  E=${e} T=${t}  seed=${seed}"
    "$PYTHON" -u main_cls.py \
        --dataset FLSkin --data_path ../FL_Skin \
        --method "$method" --local_model ResNet50 --num_classes 8 \
        --T "$t" --E "$e" --lr "$LR" --batchsize 64 \
        --optimizer adam --pretrained --cosine_lr \
        --device "$gpu" --seed "$seed" --exp_name grid \
        > "$logdir/${method}_seed${seed}.log" 2>&1 &
    RUNNING_PIDS+=($!)
    sleep 30  # wait for memory to fill before checking next GPU
done

echo "[$(date '+%H:%M:%S')]  All jobs launched. Waiting..."
for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done
echo "[$(date '+%H:%M:%S')]  All PN retry runs complete!"
