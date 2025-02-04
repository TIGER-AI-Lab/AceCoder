#!/bin/bash

# Create conda environment
conda create -n acecoder_data python=3.11
conda init
conda activate acecoder_data

# uncomment the following if you have CUDA 11.8
# export VLLM_VERSION=0.2.6
# export PYTHON_VERSION=311
# pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl

# pip uninstall torch -y
# pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# pip uninstall xformers -y
# pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118


# install packages
pip install -e .

## Intall easy open ai by Jiang Dong Fu
pip install git+https://github.com/jdf-prog/easy-openai
