


DATA_DIR=/data/t-zujieliang/reddit
TOKENIZER=bert-base-uncased
python tokenization/tokenize_dataset.py $DATA_DIR valid.pkl.txt $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR test.pkl.txt $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR train.pkl.txt $TOKENIZER



# DATA_DIR=/data/t-zujieliang/reddit_cc_3turns_data
# TOKENIZER=bert-base-uncased
# python tokenization/tokenize_dataset.py $DATA_DIR reddit_conversations.3turns.dev.txt $TOKENIZER
# python tokenization/tokenize_dataset.py $DATA_DIR reddit_conversations.3turns.test.txt $TOKENIZER
# python tokenization/tokenize_dataset.py $DATA_DIR reddit_conversations.3turns.train.txt $TOKENIZER
