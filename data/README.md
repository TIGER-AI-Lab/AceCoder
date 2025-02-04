# AceCode (Dev)

This repository contained Wyett's development code for the project AceCoder.

## Installation
I assume you have CUDA 12.1 and conda installed. With those, run the following command:

```bash
source setup.sh
```

You will be prompted some y/n options for installing packages, type y then enter in each instance <br />

Note if you have cuda 11.8, then you need to:
1. remove "vllm", "torch", and "xformers" from setup.py
2. uncomment the following code in setup.sh

```bash
export VLLM_VERSION=0.2.6
export PYTHON_VERSION=311
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl

pip uninstall torch -y
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip uninstall xformers -y
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

## install LLM-Blender
pip install git+https://github.com/yuchenlin/LLM-Blender.git
```

## Training Dataset
All the code related to training dataset creation can be found in ```training_dataset``` folder. When creating training dataset, we need to follow the following steps:
1. Source unstructured dataset (EVOL, OSS, and BigcodePythonFns)
2. Create MBPP-liked prompt and test cases using GPT-4o-mini
3. Create inferences of the prompts. ```training_dataset/inference_generated_prompts_helper.sh```
4. Evaluate on the inferences using the test cases generated. ```training_dataset/evaluate_inferenced_code_helper.sh```
5. Then consolidate dataset with inferences. ```training_dataset/consolidate_dataset.py```
6. Finally, you can create dataset of different purposes. The codes are found here: ```training_dataset/create_dataset/```

## Eval datasets
- [MBPP](https://huggingface.co/datasets/mbpp)
To inference all models on MBPP, run:
```
make mbpp
```