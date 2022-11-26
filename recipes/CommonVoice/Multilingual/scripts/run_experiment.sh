#!/bin/bash

# Open a terminal and run:
# bash run_experiment.sh <size> <model> <config>

SIZE=$1
MODEL=$2
CONFIG=$3

cd /workspace
git clone https://github.com/lucadellalib/speechbrain.git
cd speechbrain
git checkout common-voice-multilingual
cd recipes/CommonVoice/Multilingual
conda env create -f environment.yaml
source activate multilingual-env
mkdir data
cd data
cp /workspace/common_voice_10_0_${SIZE}/common_voice_10_0_${SIZE}.tar.gz .
tar -xf common_voice_10_0_${SIZE}.tar.gz
cd ..
cd ${MODEL}
mkdir -p /workspace/common_voice_10_0_${SIZE}_output/results
ln -s /workspace/common_voice_10_0_${SIZE}_output/results results
python train_encoder_decoder.py ${CONFIG}
