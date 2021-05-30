CUDA_VISIBLE_DEVICES=0,1 python vokenization/extract_vision_keys.py \
    --image-sets open_images \
    --load-dir /data/t-zujieliang/xmatching_model/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert 
    # --batch-size 512
# CUDA_VISIBLE_DEVICES=0 python vokenization/extract_vision_keys.py \
#     --image-sets vg_nococo,coco_minival,coco_nominival,coco_train \
#     --load-dir /data/t-zujieliang/xmatching_model/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert \


# CUDA_VISIBLE_DEVICES=1 python vokenization/extract_vision_keys.py \
    # --image-sets coco_minival,coco_nominival,coco_train \
    # --load-dir /data/t-zujieliang/xmatching_model/sent-level_b1024_embed512_maxlen50_resnext_bert \




# CUDA_VISIBLE_DEVICES=$1 python vokenization/extract_vision_keys.py \
#     --image-sets vg_nococo,coco_minival,coco_nominival,coco_train,cc_valid \
#     --load-dir snap/xmatching/$2 
