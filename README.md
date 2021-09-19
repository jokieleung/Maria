# Maria: A Visual Experience Powered Conversational Agent

This repository is the Pytorch implementation of our paper "[Maria: A Visual Experience Powered Conversational Agent](https://arxiv.org/abs/2105.13073)" in ACL 2021.

In this paper, we present Maria, a neural conversation agent powered by the visual world experiences which are retrieved from a large-scale image index. Maria consists of three flexible components, i.e., text-to-image retriever, visual concept detector and visual-knowledge-grounded response generator.

Coming soon!

## Summary

- [Maria: A Visual Experience Powered Conversational Agent](#maria-a-visual-experience-powered-conversational-agent)
  - [Summary](#summary)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Text-to-Image Retrieval Model](#text-to-image-retrieval-model)
    - [Bottom-up Detector Model](#bottom-up-detector-model)
    - [Dialog Generation Model](#dialog-generation-model)
  - [Citation](#citation)
  - [Acknowledgment](#acknowledgment)

## Dependencies

- python 3.7 

- pytorch 1.4.0

- Ubuntu 18.04

## Data

### Reddit Conversation Corpus
Please download the Reddit data from google drive [here](https://drive.google.com/file/d/1kyfPcbwFwLm5tlaQ4on8e5K7zd-72ZyH/view).

### Download the Open Images
We will use the [Open Images](https://storage.googleapis.com/openimages/web/index.html) images as candidate images for retrievel.
Refer to [here](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) to download the images first. You can build the image index with the appropriate size (500,000 in our experiments) as needed.

If you already have Open Images dataset on disk. Save them as 

```
data
|-- open_images
    |-- images
         |-- 14928b4f367c217e.jpg
         |-- 289d643a8761aa83.jpg
         |-- ......
```

## Usage

### Text-to-Image Retrieval Model

Please refer to [retrieval_model/README.md](retrieval_model/README.md)

### Bottom-up Detector Model

Please refer to [detector_model/README.md](detector_model/README.md)

### Dialog Generation Model

Please refer to [dialog_model/README.md](dialog_model/README.md)

## Citation

If you find this paper helps your research, please kindly consider citing our paper in your publications.

```BibTeX
@inproceedings{liang2021maria,
   title={Maria: A Visual Experience Powered Conversational Agent},
   author={Liang, Zujie and 
           Hu, Huang and 
           Xu, Can and 
           Tao, Chongyang and 
           Geng, Xiubo and 
           Chen, Yining and 
           Liang, Fan and 
           Jiang, Daxin},
   booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)},
   year={2021}
}
```

## Acknowledgment

Special thanks to the authors of [OSCAR][1], [vokenization][2], and [py-bottom-up-attention][3].


[1]: https://github.com/microsoft/OSCAR
[2]: https://github.com/airsplay/vokenization
[3]: https://github.com/airsplay/py-bottom-up-attention

