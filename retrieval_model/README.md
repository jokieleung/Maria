# Text-to-Image Retrieval Model

The code for retrieval model is implemented as a fork of "[Vokenization](https://arxiv.org/pdf/2010.06775.pdf)". We learn a sentence-level cross-modal retrieval model from sentence-image aligned data (i.e., MS-COCO image captioning dataset)
In general, the main changes bwtween our retrieval model and the vokenizer are:

1. We do the sentence-level retrieval, a sentence-image matching score is computed, and the language representation here is the sentence embedding (the output for the first token [CLS]).
2. We modify the MLP hidden layer to 1024, and the embedding size for calculating the similarity is set to 512.


## Installation
```shell script
pip install -r requirements.txt
```

Require python 3.6 + (to support huggingface [transformers](https://github.com/huggingface/transformers)).


## Download Image and Captioning Data
1. Download MS COCO images:
    ```shell script
    # MS COCO (Train 13G, Valid 6G)
    mkdir -p data/mscoco
    wget http://images.cocodataset.org/zips/train2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/val2014.zip -P data/mscoco
    unzip data/mscoco/train2014.zip -d data/mscoco/images/ && rm data/mscoco/train2014.zip
    unzip data/mscoco/val2014.zip -d data/mscoco/images/ && rm data/mscoco/val2014.zip
    ```
   If you already have COCO image on disk. Save them as 
    ```
    data
      |-- mscoco
            |-- images
                 |-- train2014
                         |-- COCO_train2014_000000000009.jpg
                         |-- COCO_train2014_000000000025.jpg
                         |-- ......
                 |-- val2014
                         |-- COCO_val2014_000000000042.jpg
                         |-- ......
    ```

2. Download captions (split following the LXMERT project):
    ```shell script
    mkdir -p data/lxmert
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json -P data/lxmert/
    ```

## Training

The model is trained on MS COCO with pairwise hinge loss.

Running Commands:
```bash
# Run the cross-modal matching model with single-machine multi-processing distributed training
# "0,1" indicates using the GPUs 0 and 1,2,3.
# Speed: 20 min ~ 30 min / 1 Epoch, 25 Epochs by default.
bash scripts/run_xmatching.bash 0,1,2,3 
```
## Inference
Then we retrieve relevant images from the Open Image dataset using Reddit Corpus.

Coming Soon..