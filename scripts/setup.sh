#!/bin/bash

# conda install -y mamba -c conda-forge

mamba install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
mamba install -y -c dglteam/label/cu116 dgl

pip install easydict pyyaml pandas scikit-learn packaging