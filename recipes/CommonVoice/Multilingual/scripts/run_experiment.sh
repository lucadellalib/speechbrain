#!/bin/bash

# Open a terminal and run:
# bash run_experiment.sh <size> <model> <config>

SIZE=$1
MODEL=$2
CONFIG=$3

git clone https://github.com/lucadellalib/speechbrain.git
cd speechbrain
git checkout multilingual-env
cd recipes/CommonVoice/Multilingual
conda env create -f environment.yaml
source activate multilingual-env
mkdir data
cp /workspace/common_voice_10_0_${SIZE}/common_voice_10_0_${SIZE}.tar.gz .
tar -xf common_voice_10_0_${SIZE}.tar.gz
cd ..
cd ${MODEL}
ln -s /workspace/common_voice_10_0_medium_output results
nohup python train.py hparams/${CONFIG}.yaml &
