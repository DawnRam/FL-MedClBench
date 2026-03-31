#!/bin/bash
# run_pn_fedlws_fixed.sh — Rerun PN and FedLWS with bug fixes.
# Covers: E=1 seeds{0,1,2}  +  E={2,5,10} seed=0  at lr=1e-4.
# Jobs are serialized: next job waits until a GPU has < 18 GB in use.

PYTHON=/nfs/scratch/eechengyang/conda_envs/llm/bin/python
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
LR=1e-4
N_CENTERS=4

cd "$WORKDIR"

# ── Build queue ──────────────────────────────────────────────────────────────
declare -a Q_METHOD Q_E Q_T Q_SEED

for method in PN FedLWS; do
    for seed in 0 1 2; do          # E=1, all seeds (highest priority)
        f="results/FLSkin/grid/100epoch_${LR}_1/${method}/metrics_seed${seed}.csv"
        expected=$((1 + 100 * N_CENTERS))
        if [ -f "$f" ] && [ "$(wc -l < "$f")" -ge "$expected" ]; then
            echo "SKIP E=1 ${method} seed=${seed}"
        else
            Q_METHOD+=("$method"); Q_E+=(1); Q_T+=(100); Q_SEED+=("$seed")
        fi
    done
    for e in 2 5 10; do            # E search, seed=0
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

queued=${#Q_METHOD[@]}
echo "========================================"
echo "  PN+FedLWS Fixed Rerun  (${queued} jobs)"
echo "========================================"
[ "$queued" -eq 0 ] && { echo "All done!"; exit 0; }

# ── GPU selector: pick GPU with most free memory (< 18000 MiB used) ─────────
free_gpu() {
    # Returns GPU index with least used memory among those under 18 GB
    nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F'[, ]+' '$2<18000{print $2, $1}' | sort -n | head -1 | awk '{print $2}'
}

# ── Launch jobs one at a time (wait for a free GPU before each) ──────────────
declare -a RUNNING_PIDS
RUNNING_PIDS=()

for ((i=0; i<queued; i++)); do
    method="${Q_METHOD[$i]}"
    e="${Q_E[$i]}"
    t="${Q_T[$i]}"
    seed="${Q_SEED[$i]}"

    # Wait for a GPU with free memory AND limit concurrent jobs to #GPUs
    while true; do
        # Reap finished jobs
        new_pids=()
        for pid in "${RUNNING_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        RUNNING_PIDS=("${new_pids[@]}")

        gpu=$(free_gpu)
        if [ -n "$gpu" ] && [ "${#RUNNING_PIDS[@]}" -lt 7 ]; then
            break
        fi
        echo "[$(date '+%H:%M:%S')]  Waiting (active=${#RUNNING_PIDS[@]}, free_gpu=${gpu:-none})..."
        sleep 30
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
    sleep 15  # let GPU memory fill before picking next GPU
done

echo "[$(date '+%H:%M:%S')]  All ${queued} jobs launched. Waiting for completion..."
for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done
echo "[$(date '+%H:%M:%S')]  All PN+FedLWS fixed runs complete!"
