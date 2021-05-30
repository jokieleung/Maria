CUDA_VISIBLE_DEVICES=0,1 python vokenization/extract_vision_keys.py \
    --image-sets open_images \
    --load-dir /data/t-zujieliang/xmatching_model/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert 
    --batch-size 512

