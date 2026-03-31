#!/bin/bash
# Ditto
python ../main_cls.py \
    --method Ditto \
    --local_model ResNet50 \
    --data_path ../../FedBCa \
    --T 500 --E 5 \
    --lr 0.0001 \
    --batchsize 24 \
    --optimizer adam \
    --device ${1:-0} \
    --exp_name "bench"
