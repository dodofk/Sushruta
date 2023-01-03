<div align="center">

# IM Special Reswearch

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)

</div>

## Description

Code for run experiment of triplet classification 

## How to run

### Install dependencies

```bash
# clone project
git clone https://github.com/dodofk/IM_Project
cd IM_Project

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# Setup timm with dev version
pip install git+https://github.com/rwightman/pytorch-image-models
# if now work, run this
cd src/vendor/pytorch-image-models 
pip install -e .
```

### Dataset Setup
you should install requirements before download

```shell

# for downloading HeiChole Dataset
# it will download the preprocess dataset from downsample and resize image from the video
# the script might fail if download from drive is limited
bash scripts/download_cholect45.sh
```
if the above script is failed, you should download it from [link](https://drive.google.com/file/d/14o5HHzK7kSXQxeoOijiPpp_nHl4yer_t/view?usp=sharing) manually
```shell
# after download the file, you should put it under folder of IM_Project
# and run the following script
mv CholecT45_resize.tar.gz data
cd data 
tar -xvf CholecT45_resize.tar.gz
mv CholecT45_resize CholecT45
```



### Training
Train model with default configuration

For training in GPU, you might need to change configuration to fit in your GPU memory

ex: running experiment=triplet_attention_base
you can find configuration file in **/configs/experiment/triplet_attention_base.yaml**
```bash
# train on CPU (our proposed model)
python train.py experiment=triplet_attention_base_cpu

# train on GPU (our proposed model)
python train.py experiment=triplet_attention_base

# train on GPU (our baselien model)
python train.py experiment=cholec_base
```
