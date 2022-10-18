#!/bin/bash

# Open a terminal and run:
# bash submit_ovhai_job.sh <size> <model> <config>

SIZE=$1
MODEL=$2
CONFIG=$3

ovhai job run ovhcom/ai-training-pytorch:latest \
    --name common_voice_10_0_${SIZE}_${MODEL} \
    --flavor ai1-1-gpu \
    --gpu 1 \
    --volume common_voice_10_0_${SIZE}@BHS/:/workspace/common_voice_10_0_${SIZE}/:RO \
    --volume common_voice_10_0_scripts@BHS/:/workspace/common_voice_10_0_scripts/:RO \
    --volume common_voice_10_0_${SIZE}_output@BHS/:/workspace/common_voice_10_0_${SIZE}_output/:RW
    -- bash -c "/workspace/common_voice_10_0_scripts/run_experiment.sh ${SIZE} ${MODEL} ${CONFIG}"
