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
      - [Installation](#installation)
    - [Bottom-up Detector Model](#bottom-up-detector-model)
      - [Installation](#installation-1)
    - [Dialog Generation Model](#dialog-generation-model)
      - [Installation](#installation-2)
  - [Citaition](#citaition)
  - [Acknowledgment](#acknowledgment)

## Dependencies

- python 3.7 

- pytorch 1.4.0

- Ubuntu 18.04

## Usage

### Text-to-Image Retrieval Model

#### Installation

Please refer to [retrieval_model/README.md](retrieval_model/README.md)

### Bottom-up Detector Model

#### Installation

Please refer to [detector_model/README.md](detector_model/README.md)

### Dialog Generation Model

#### Installation

Please refer to [dialog_model/README.md](dialog_model/README.md)

## Citaition

If you find this paper helps your research, please kindly consider citing our paper in your publications.

```BibTeX
@inproceedings{liang2021maria,
   title={Maria: A Visual Experience Powered Conversational Agent},
   author={Liang, Zujie and Hu, Huang and Xu, Can and Chongyang, Tao and Geng, Xiubo and Chen, Danqi and Liang, Fan and Jiang, Daxin},
   booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)},
   year={2021}
}
```

## Acknowledgment

Special thanks to the authors of [OSCAR][1], [vokenization][2], and [py-bottom-up-attention][3].


[1]: https://github.com/microsoft/OSCAR
[2]: https://github.com/airsplay/vokenization
[3]: https://github.com/airsplay/py-bottom-up-attention

