#!/usr/bin/env bash

conda create -n bert_hw python=3.10
conda activate bert_hw

conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia
pip install tqdm==4.66.1
pip install requests==2.31.0
pip install importlib-metadata==6.8.0
pip install filelock==3.9.0
pip install sklearn==0.0
pip install tokenizers==0.14.0
pip install explainaboard_client==0.1.4
