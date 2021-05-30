


DATA_DIR=/data/t-zujieliang/reddit
TOKENIZER=bert-base-uncased
python tokenization/tokenize_dataset.py $DATA_DIR valid.pkl.txt $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR test.pkl.txt $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR train.pkl.txt $TOKENIZER

