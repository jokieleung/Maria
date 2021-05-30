# The name of experiment
GPUS=$1
NAME=$2

# Create dirs and make backup
output=snap/bert/$NAME
mkdir -p $output/src
cp -r vlm $output/src/
cp scripts/run_glue_epochs.bash $output/run_glue_epochs.bash
cp scripts/run_glue_at_epoch.bash $output/run_glue_at_epoch.bash 
cp $0 $output/run.bash

export TRAIN_FILE=data/wiki-cased/en.train.raw
export TEST_FILE=data/wiki-cased/en.valid.raw

# Pre-training
CUDA_VISIBLE_DEVICES=$1 unbuffer python vlm/run_vlm_distributed.py \
    --output_dir=$output \
	--overwrite_output_dir \
	--config_name=vlm/configs/bert-12L-768H.json \
	--tokenizer_name=bert-base-uncased \
    --model_type=bert \
	--block_size=126 \
	--per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
	--gradient_accumulation_steps=2 \
    --max_steps=200000 \
	--learning_rate=2e-4 \
	--weight_decay=0.01 \
	--warmup_steps=5000 \
    --mlm_probability 0.15 \
    --mlm_ratio 1.0 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --col_data \
    --split_sent \
    --do_voken_cls \
    --voken_labels all \
    --voken_dir snap/xmatching/bert_resnext/vokens \
    --voken_suffix vg_nococo \
    --mlm ${@:3} | tee $output/log.log

    #--fp16 \
	#--fp16_opt_level O2 \

# Wait for clearing the GPU cache
sleep 30
bash scripts/run_glue_epochs.bash $GPUS $output --snaps 4

