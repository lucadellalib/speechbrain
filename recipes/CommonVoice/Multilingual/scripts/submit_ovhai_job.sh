#!/bin/bash

# Open a terminal and run:
# ovhai data upload BHS common_voice_10_0_scripts run_experiment.sh
# bash submit_ovhai_job.sh <size> <model> <config>

SIZE=small
MODEL=whisper_hf
VARIANT=whisper-large
TAG=large_ft
CONFIG="$SIZE -m openai/${VARIANT}"
#CONFIG="$SIZE -m openai/${VARIANT} -t -p \"{'manifest_dir': '../data/common_voice_10_0_${SIZE}'}\" -c \"{'output_dir': 'results/multilingual/${SIZE}/${VARIANT}/1234'}\""
echo $CONFIG

ovhai job run ovhcom/ai-training-pytorch:latest \
    --name common_voice_10_0_${SIZE}_${MODEL}_${TAG} \
    --flavor ai1-1-gpu \
    --gpu 1 \
    --volume common_voice_10_0_${SIZE}@BHS/:/workspace/common_voice_10_0_${SIZE}/:RO \
    --volume common_voice_10_0_scripts@BHS/:/workspace/common_voice_10_0_scripts/:RW \
    --volume common_voice_10_0_${SIZE}_output@BHS/:/workspace/common_voice_10_0_${SIZE}_output/:RW
    #-- bash -c "chmod u+x /workspace/common_voice_10_0_scripts/run_experiment.sh && /workspace/common_voice_10_0_scripts/run_experiment.sh ${SIZE} ${MODEL} \"${CONFIG}\""
