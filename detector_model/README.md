# Bottom-up Attention Detector

The code for retrieval model is implemented as a fork of "[py-bottom-up-attention](https://github.com/airsplay/py-bottom-up-attention)". The object detector is pre-trained on Visual Genome and can extract fine-grained visual (bbox) features and correlated object tags.


## Installation
```
git clone https://github.com/airsplay/py-bottom-up-attention.git
cd py-bottom-up-attention

# Install python libraries
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron2
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install [--editable] .
```

## Feature Extraction Scripts for Open Images


1. For Open Images: [open images script](demo/demo_oi_for_reddit.py)

```shell script
python demo/demo_oi_for_reddit.py
```


## References

More details please refer to the original [README](py-bottom-up-attention_README.md)

