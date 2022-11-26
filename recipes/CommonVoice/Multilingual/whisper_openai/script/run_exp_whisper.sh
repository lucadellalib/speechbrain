# Example: bash script/run2.sh  medium en  base 16 1
# set TEST t0 1 if you want to use pretraiend model and for finetuning set the value to 0
# For large whisper mdoel set the  BATCH_SIZE to 8
SIZE=$1
LANGUAGE=$2
MODEL=$3
BATCH_SIZE=$4
TEST=$5

tar -xf data/common_voice_10_0_$SIZE.tar.gz common_voice_10_0_$SIZE
git clone --branch  common-voice-multilingual --single-branch https://github.com/lucadellalib/speechbrain.git
cd speechbrain/recipes/CommonVoice/Multilingual/
conda env create -f environment.yaml
source activate multilingual-env
cd whisper_openai


if [ $TEST -eq 1 ]
then
    OUT_DIR=/workspace/output/common_voice_${SIZE}_whisper_${MODEL}_pretrained_${LANGUAGE}
    mkdir ${OUT_DIR}
    python whisper_finetune.py  --dataset_size=${SIZE}  --model_name=${MODEL} --dataset_dir=/workspace/common_voice_10_0  -l ${LANGUAGE}  --base_dir=${OUT_DIR}   --do_test  --test_manifest_with_references   --test_results_json_filepath=${OUT_DIR}/test_result.json   --test_batch_size=${BATCH_SIZE} 
else
    OUT_DIR=/workspace/output/common_voice_${SIZE}_whisper_${MODEL}_finetune_${LANGUAGE}
    mkdir ${OUT_DIR}
    python whisper_finetune.py  --dataset_size=${SIZE}  --model_name=${MODEL} --dataset_dir=/workspace/common_voice_10_0  -l ${LANGUAGE}  --base_dir=${OUT_DIR}  --do_train --do_test  --test_manifest_with_references --num_train_epochs=4  --test_results_json_filepath=${OUT_DIR}/test_result.json --experiment_directory=.${OUT_DIR} --batch_size=${BATCH_SIZE}  --dev_batch_size=${BATCH_SIZE}  --test_batch_size=${BATCH_SIZE} 
fi