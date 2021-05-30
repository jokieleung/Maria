# Benchmarking the cross-modal matching model with
#     1. Retrieval scores.
#     2. Voken Diversity w.r.t words in specific language corpus.
# Please run this after image_key_retrivel and tokenization. 
#    i.e., step 1 and step2 in readme.md

MODEL=$2
# MODELPATH=snap/xmatching/$MODEL
MODELPATH=/data/t-zujieliang/xmatching_model/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert
rm -rf $MODELPATH/analysis.log

# Retrieval scores
CUDA_VISIBLE_DEVICES=$1 unbuffer python vokenization/evaluate_retrieval.py \
    --load $MODELPATH \
    --image-sets coco_minival \
    | tee -a $MODELPATH/analysis.log

    # --image-sets coco_minival,cc_valid,coco_nominival,coco_train \
    
# Diversity
# Test diversity of vision-and-language (captioning) datasets
# CUDA_VISIBLE_DEVICES=$1 unbuffer python vokenization/evaluate_diversity.py \
#     --load $MODELPATH \
#     --image-sets vg_nococo \
#     --corpus coco_minival \
#     | tee -a $MODELPATH/analysis.log

# # --corpus coco_minival,cc_valid \

# # # Test diversity of pure-language corpus
# CUDA_VISIBLE_DEVICES=$1 unbuffer python vokenization/evaluate_diversity.py \
#     --load $MODELPATH \
#     --image-sets vg_nococo \
#     --corpus /data/t-zujieliang/wiki103-cased/wiki.valid.raw \
#     --maxsents 95000 \
#     | tee -a $MODELPATH/analysis.log
