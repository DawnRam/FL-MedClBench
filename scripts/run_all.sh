#!/bin/bash
# Run all FL-MedClsBench methods sequentially
# Usage: bash scripts/run_all.sh [GPU_ID]
GPU=${1:-0}
DATA="../../FedBCa"
MODEL="ResNet50"
T=500
E=5
LR=0.0001
BS=24

for METHOD in LocalTrain FedAvg FedProx MOON FedAWA FedNova PN FedRDN FedLWS FedBN SioBN FedPer FedRoD Ditto; do
    echo "========================================"
    echo "Running $METHOD"
    echo "========================================"
    python ../main_cls.py \
        --method $METHOD \
        --local_model $MODEL \
        --data_path $DATA \
        --T $T --E $E \
        --lr $LR \
        --batchsize $BS \
        --optimizer adam \
        --device $GPU \
        --exp_name "bench"
done
