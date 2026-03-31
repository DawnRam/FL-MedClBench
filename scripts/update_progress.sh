#!/bin/bash
# update_progress.sh — 实时扫描所有 grid log，更新 PROGRESS.md
# Usage: nohup bash scripts/update_progress.sh > logs/progress_monitor.log 2>&1 &

WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
PROGRESS=$WORKDIR/PROGRESS.md
INTERVAL=120   # 每 2 分钟更新一次

METHODS=(LocalTrain FedAvg FedProx MOON FedAWA FedNova PN FedRDN FedLWS FedBN SioBN FedPer FedRoD Ditto)
SEEDS=(0 1 2)
LRS=(1e-4 1e-3 1e-2 1e-1)
ES=(1 2 5 10)

get_status() {
    local logfile=$1
    local T=$2
    [ ! -f "$logfile" ] && echo "⏳" && return
    local last_round
    last_round=$(grep -oP "Round\s+\K[0-9]+" "$logfile" 2>/dev/null | tail -1)
    [ -z "$last_round" ] && echo "⏳" && return
    if [ "$last_round" -ge "$T" ] 2>/dev/null; then
        echo "✅"
    else
        echo "🔄${last_round}/${T}"
    fi
}

while true; do
    {
    echo "# FL_Skin Grid Search Progress"
    echo ""
    echo "> 最后更新：$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "**Grid**: lr ∈ {1e-1, 1e-2, 1e-3, 1e-4} × E ∈ {1, 2, 5, 10} = **16 settings**"
    echo "**Per setting**: 14 methods × 3 seeds = **42 jobs**  |  **Node**: eez244"
    echo ""
    echo "---"
    echo ""
    echo "## 节点分工"
    echo ""
    echo "| 节点 | 负责 lr | 备注 |"
    echo "|------|---------|------|"
    echo "| **eez244** (本机) | **1e-4** | E=1(跑中+重跑) → 2 → 5 → 10 |"
    echo "| 节点 B | **1e-3** | E=1 → 2 → 5 → 10 |"
    echo "| 节点 C | **1e-2** | E=1 → 2 → 5 → 10 |"
    echo "| 节点 D | **1e-1** | E=1 → 2 → 5 → 10 |"
    echo ""
    echo "---"
    echo ""
    echo "## 各设定进度"
    echo ""

    for lr in "${LRS[@]}"; do
        echo "### lr=${lr}"
        echo ""
        echo "| E | T | Method | s0 | s1 | s2 | Done/42 |"
        echo "|---|---|--------|----|----|-----|---------|"

        for E in "${ES[@]}"; do
            T=$((100 / E))
            LOGDIR="$WORKDIR/logs/grid_$((T * E))epoch_${lr}_${E}"
            done_count=0
            total=42
            first=true
            for method in "${METHODS[@]}"; do
                s0=$(get_status "$LOGDIR/${method}_seed0.log" "$T")
                s1=$(get_status "$LOGDIR/${method}_seed1.log" "$T")
                s2=$(get_status "$LOGDIR/${method}_seed2.log" "$T")
                for s in $s0 $s1 $s2; do
                    [[ "$s" == "✅" ]] && ((done_count++)) || true
                done
                if $first; then
                    echo "| $E | $T | $method | $s0 | $s1 | $s2 | ${done_count}/${total} |"
                    first=false
                else
                    echo "| | | $method | $s0 | $s1 | $s2 | |"
                fi
            done
            echo "| | | **合计** | | | | **${done_count}/42** |"
            echo ""
        done
    done

    echo "---"
    echo ""
    echo "## 需要重跑的 jobs（lr=1e-4 E=1 崩溃）"
    echo ""
    echo "| Method | seed | 状态 |"
    echo "|--------|------|------|"
    RERUN_METHODS=(LocalTrain FedAvg FedProx)
    LOGDIR="$WORKDIR/logs/grid_100epoch_1e-4_1"
    for m in "${RERUN_METHODS[@]}"; do
        for s in 0 1 2; do
            last=$(grep -oP "Round\s+\K[0-9]+" "$LOGDIR/${m}_seed${s}.log" 2>/dev/null | tail -1)
            if [ "${last:-0}" -ge 100 ] 2>/dev/null; then
                echo "| $m | $s | ✅ 完成 |"
            else
                echo "| $m | $s | ❌ 需重跑 (${last:-0}/100r) |"
            fi
        done
    done
    for s in 0; do
        last=$(grep -oP "Round\s+\K[0-9]+" "$LOGDIR/MOON_seed${s}.log" 2>/dev/null | tail -1)
        if [ "${last:-0}" -ge 100 ] 2>/dev/null; then
            echo "| MOON | $s | ✅ 完成 |"
        else
            echo "| MOON | $s | ❌ 需重跑 (${last:-0}/100r) |"
        fi
    done
    echo ""
    echo "---"
    echo ""
    echo "## 其他节点启动命令"
    echo ""
    echo "\`\`\`bash"
    echo "cd /nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench"
    echo "# 节点 B (lr=1e-3): 按顺序"
    echo "nohup bash scripts/run_skin_grid.sh --lr=1e-3 --E=1  --exp=grid > logs/grid_launch_1e-3_E1.log  2>&1 &"
    echo "nohup bash scripts/run_skin_grid.sh --lr=1e-3 --E=2  --exp=grid > logs/grid_launch_1e-3_E2.log  2>&1 &"
    echo "nohup bash scripts/run_skin_grid.sh --lr=1e-3 --E=5  --exp=grid > logs/grid_launch_1e-3_E5.log  2>&1 &"
    echo "nohup bash scripts/run_skin_grid.sh --lr=1e-3 --E=10 --exp=grid > logs/grid_launch_1e-3_E10.log 2>&1 &"
    echo "# 节点 C (lr=1e-2) / 节点 D (lr=1e-1): 同上改 lr 值"
    echo "\`\`\`"

    } > "$PROGRESS"

    echo "[$(date '+%H:%M:%S')] PROGRESS.md updated"
    sleep "$INTERVAL"
done
