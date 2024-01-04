#!/bin/bash

# conda install -y mamba -c conda-forge

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/cu118 dgl

pip install easydict pyyaml pandas scikit-learn packaging