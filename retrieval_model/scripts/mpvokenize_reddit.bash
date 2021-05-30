GPU=0,1,2,3

# LOAD=snap/xmatching/$2
LOAD=/data/t-zujieliang/xmatching_model/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert
WIKI_DIR=/data/t-zujieliang/reddit
TOKENIZER=bert-base-uncased

for DATA_NAME in valid.pkl.txt test.pkl.txt train.pkl.txt
do 
    CUDA_VISIBLE_DEVICES=$GPU python vokenization/vokenize_corpus_mp.py \
        --load $LOAD \
        --batch-size 512 \
        --corpus=$WIKI_DIR/$DATA_NAME \
        --tokenizer-name $TOKENIZER \
        --image-sets open_images 
        

done

