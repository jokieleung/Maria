
# ----------------------------------training ----------------------------------------
GPUS=$1
# The name of experiment
# NAME=$2

# Create dirs and make backup
# output=/data/t-zujieliang/xmatching_model/$NAME
# clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert
output=/data/t-zujieliang/xmatching_model/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert
mkdir -p $output/src/
cp -r xmatching $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$GPUS unbuffer python xmatching/main.py \
    --train-imgs mscoco_train,mscoco_nominival --valid-imgs mscoco_minival \
    --train-langs mscoco --valid-langs mscoco \
    --max-len 75 --dim 512 \
    --lang-layers 4,3,2,1 \
    --lang-pretrained --visn-pretrained \
    --num-workers 8 --batchSize 512 --optim adam --lr 1e-3 --epochs 25 \
    --nodes 1 --nr 0 \
    --visn resnext101_32x8d --lang bert \
    --output $output ${@:3} | tee $output/log.log
