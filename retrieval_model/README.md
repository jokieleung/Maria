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
Here is a diagram of these processes and we next discuss them one-by-one:

```
Extracting Image Features-----> Benchmakring the Matching Models (Optional) --> Vokenization
Downloading Language Data --> Tokenization -->-->--/
```

### Downloading Reddit Data 

Please download the Reddit data from [here](https://drive.google.com/file/d/1kyfPcbwFwLm5tlaQ4on8e5K7zd-72ZyH/view).

### Pre-Processing Reddit Data 

For each case in the training data, you need to cat both `context` and `response` to a line, while in the val/ test data, you only need to use the `context` to retrieve the image. Then the processed data format will be like,

```
utt1 \t utt2 \t utt3 .... \t uttn
utt1 \t utt2 \t utt3 .... \t uttn
utt1 \t utt2 \t utt3 .... \t uttn
....
utt1 \t utt2 \t utt3 .... \t uttn
```

### Tokenization of Reddit Data

we convert the training file into

```
data 
 |-- reddit
        |-- train.pkl.txt.bert-base-uncased
        |-- train.pkl.txt.bert-base-uncased.hdf5
        |-- train.pkl.txt.bert-base-uncased.line
```

The txt file `train.pkl.txt.bert-base-uncased` saves the tokens and each line in this file is the tokens of a line 
in the original file,
The hdf5 file `train.pkl.txt.bert-base-uncased.hdf5` stores all the tokens continuously and use
`train.pkl.txt.bert-base-uncased.line` to index the starting token index of each line.
The ".line" file has `L+1` lines where `L` is the number of lines in the original files.
Each line has a range "line[i]" to "line[i+1]" in the hdf5 file.

Commands:

1. ```shell script
bash tokenization/tokenize_reddit_bert.bash 
   ```
   

### Extracting Image Features

The image pre-processing extracts the image features to build the keys in the retrieval process.

#### Download the Open Images

We will use the [Open Images](https://storage.googleapis.com/openimages/web/index.html) images as candidate images for retrievel.
Refer to [here](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) to download the images first. You can build the image index with the appropriate size (500,000 in our experiments) as needed。

If you already have Open Images dataset on disk. Save them as 

```
data
|-- open_images
    |-- images
         |-- 14928b4f367c217e.jpg
         |-- 289d643a8761aa83.jpg
         |-- ......
```

#### Build Universal Image Ids

We first build a list of universal image indexes with 
[vokenization/create_image_ids.py](vokenization/create_image_ids.py). 
It is used to unify the image ids in different experiments 
thus the feature array stored in hdf5 could be universally indexed.
The image ids are saved under a shared path `LOCAL_DIR` (default to `data/vokenization`)
 defined in [vokenization/common.py](vokenization/common.py).
The image ids are saved under `data/vokenization/images` with format `{IMAGE_SET}.ids`.
We will make sure that all the experiments agree with this meta info,
so that we would not get different indexing in different retrieval experiments.

> Note: The ids created by [create_image_ids.py](vokenization/create_image_ids.py) are only the order of the images.
> The actual images in the dictionary are provided by `extract_keys.bash`, thus is corresponding to the 
> `_paths.txt`, because the `extract_keys` will filter all broken images and non-existing images.

Commands:

```bash
# Step 1, Build image orders.
python vokenization/create_image_ids.py  
```

#### Extracting Image Features

Extract image features regarding the list built above, using code 
[vokenization/extract_vision_keys.py](vokenization/extract_vision_keys.py). 
The code will first read the image ids saved in `data/vokenization/images/{IMAGE_SET}.ids` and locate the images.
The features will be saved under `snap/xmatching/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert/keys/{IMAGE_SET}.hdf5`.

Commands:

```bash
# Step 2, Extract features. 
bash scripts/extract_keys.bash 
```

### The Retrieval Process

After all these steps, we could start to retrieve the relevant images of the Reddit corpus.
It would load the tokens saved in `dataset_name.tokenizer_name.hdf5` 
and uses the line-split information in `dataset_name.tokenzier_name.line`.

The retrieved images' image paths with be cached in `snap/xmatching/clsmlp_sent-level_b1024_embed512_maxlen75_resnext_bert/vokens/` by default. Then you can use the Visual Concept Detector to extract the relevant bbox features and object tags to continue the following steps.


Commands:

```shell script
# Note: mp is the abbreviation for "multi-processing"
bash scripts/mpvokenize_reddit.bash
```

> The script will call
> [vokenization/vokenize_corpus_mp.py](vokenization/vokenize_corpus_mp.py)
> to vokenize a corpus. 
> The vokenziation happens in [vokenization/vokenization.py](vokenization/vokenization.py) and
> it use [vokenization/indexing.py](vokenization/indexing.py) to do nearest neighbor search
> (based on [faiss](https://github.com/facebookresearch/faiss)).

### References

More details please refer to [Vokenization](https://github.com/airsplay/vokenization), many thanks for their work.