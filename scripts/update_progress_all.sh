#!/bin/bash
# update_progress_all.sh — Scan result files and regenerate PROGRESS.md
# Detects completion by checking metrics_seed*.csv line count.

WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
RESULTS="$WORKDIR/results/FLSkin/grid"
LOGBASE="$WORKDIR/logs"
PROGRESS="$WORKDIR/PROGRESS.md"
N_CENTERS=4
METHODS=(LocalTrain FedAvg FedProx MOON FedAWA FedNova PN FedRDN FedLWS FedBN SioBN FedPer FedRoD Ditto)
SEEDS=(0 1 2)
LRS=(1e-4 1e-3 1e-2 1e-1)
ES=(1 2 5 10)

status_of() {
    local lr=$1 e=$2 method=$3 seed=$4
    local t=$((100 / e))
    local expected=$(( 1 + t * N_CENTERS ))
    local metrics="$RESULTS/100epoch_${lr}_${e}/${method}/metrics_seed${seed}.csv"
    local logfile="$LOGBASE/grid_100epoch_${lr}_${e}/${method}_seed${seed}.log"

    if [ -f "$metrics" ]; then
        local lines
        lines=$(wc -l < "$metrics")
        if [ "$lines" -ge "$expected" ]; then
            echo "✅"
            return
        fi
    fi

    # Check if process is running
    if [ -f "$logfile" ]; then
        local last_round
        last_round=$(grep -oP 'Round\s+\K[0-9]+(?=/[0-9]+)' "$logfile" 2>/dev/null | tail -1)
        if [ -n "$last_round" ]; then
            # Check if the process is actually running (via pid check on the python cmd)
            local is_running=false
            # Search for active python process matching this job
            if pgrep -f "main_cls.py.*--method.*${method}.*--seed.*${seed}.*--lr.*${lr}.*--E.*${e}" > /dev/null 2>&1; then
                is_running=true
            fi
            if $is_running; then
                echo "🔄${last_round}/$((100/e))"
            else
                echo "❌${last_round}/$((100/e))"
            fi
            return
        fi
        # Log exists but no rounds found (crashed at start)
        echo "❌0"
        return
    fi
    echo "⏳"
}

count_done() {
    local lr=$1 e=$2
    local count=0
    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local t=$((100 / e))
            local expected=$(( 1 + t * N_CENTERS ))
            local metrics="$RESULTS/100epoch_${lr}_${e}/${method}/metrics_seed${seed}.csv"
            if [ -f "$metrics" ] && [ "$(wc -l < "$metrics")" -ge "$expected" ]; then
                ((count++)) || true
            fi
        done
    done
    echo $count
}

# ── Check running processes ────────────────────────────────────────────────────
running_info() {
    local lr=$1 e=$2 method=$3 seed=$4
    local logfile="$LOGBASE/grid_100epoch_${lr}_${e}/${method}_seed${seed}.log"
    if [ ! -f "$logfile" ]; then echo "⏳"; return; fi
    local t=$((100 / e))
    local expected=$(( 1 + t * N_CENTERS ))
    local metrics="$RESULTS/100epoch_${lr}_${e}/${method}/metrics_seed${seed}.csv"
    if [ -f "$metrics" ] && [ "$(wc -l < "$metrics")" -ge "$expected" ]; then
        echo "✅"; return
    fi
    local last_round
    last_round=$(grep -oP 'Round\s+\K[0-9]+(?=/[0-9]+)' "$logfile" 2>/dev/null | tail -1)
    if [ -z "$last_round" ]; then echo "⏳"; return; fi
    # Is running?
    if pgrep -f "main_cls.*--seed.*${seed}.*--lr.*${lr}" >/dev/null 2>&1; then
        echo "🔄${last_round}/${t}"
    else
        echo "⏳${last_round}/${t}"
    fi
}

# ── Generate PROGRESS.md ───────────────────────────────────────────────────────
now=$(date '+%Y-%m-%d %H:%M:%S')

{
cat <<EOF
# FL_Skin Grid Search Progress

> 最后更新：${now}

**Grid**: lr ∈ {1e-1, 1e-2, 1e-3, 1e-4} × E ∈ {1, 2, 5, 10} = **16 settings**
**Per setting**: 14 methods × 3 seeds = **42 jobs**  |  **Node**: $(hostname)

---

EOF

for lr in "${LRS[@]}"; do
    echo "### lr=${lr}"
    echo ""
    echo "| E | T | Method | s0 | s1 | s2 | Done/42 |"
    echo "|---|---|--------|----|----|-----|---------|"
    for e in "${ES[@]}"; do
        t=$((100 / e))
        done_count=$(count_done "$lr" "$e")
        first_method=true
        for method in "${METHODS[@]}"; do
            s0=$(running_info "$lr" "$e" "$method" 0)
            s1=$(running_info "$lr" "$e" "$method" 1)
            s2=$(running_info "$lr" "$e" "$method" 2)
            if $first_method; then
                echo "| ${e} | ${t} | ${method} | ${s0} | ${s1} | ${s2} | ${done_count}/42 |"
                first_method=false
            else
                echo "| | | ${method} | ${s0} | ${s1} | ${s2} | |"
            fi
        done
        echo "| | | **合计** | | | | **${done_count}/42** |"
        echo ""
    done
done

# Overall summary
total_done=0
total_jobs=0
for lr in "${LRS[@]}"; do
    for e in "${ES[@]}"; do
        done=$(count_done "$lr" "$e")
        ((total_done += done)) || true
        ((total_jobs += 42)) || true
    done
done

cat <<EOF

---

## 总进度

**${total_done} / ${total_jobs}** jobs 完成 ($(( total_done * 100 / total_jobs ))%)

EOF
} > "$PROGRESS"

echo "PROGRESS.md updated: ${total_done}/${total_jobs} done"
