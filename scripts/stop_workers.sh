#!/bin/bash
# stop_workers.sh — Kill all workers and training jobs, reset running→pending.
WORKDIR=/nfs/scratch/eechengyang/Code/FedCL/FL-MedClsBench
PIDFILE="$WORKDIR/jobqueue/workers.pid"

if [[ -f "$PIDFILE" ]]; then
    echo "Killing tracked worker PIDs..."
    while read pid; do
        kill -9 "$pid" 2>/dev/null && echo "  killed worker $pid"
    done < "$PIDFILE"
    rm -f "$PIDFILE"
fi

# Also kill any remaining main_cls.py and worker scripts
kill -9 $(ps aux | grep -E "worker_filtered\.sh|worker\.sh|launch_workers\.sh" | grep -v grep | awk '{print $2}') 2>/dev/null
kill -9 $(ps aux | grep "main_cls.py" | grep -v grep | awk '{print $2}') 2>/dev/null
sleep 3

remaining=$(ps aux | grep "main_cls.py" | grep -v grep | wc -l)
echo "main_cls.py remaining: $remaining"

# Clean up
find "$WORKDIR/results/FLSkin/grid" -name "*.lock" -delete 2>/dev/null
for f in "$WORKDIR/jobqueue/running"/*; do
    [[ -f "$f" ]] && mv "$f" "$WORKDIR/jobqueue/pending/"
done
echo "Stale locks removed. Running jobs reset to pending."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
