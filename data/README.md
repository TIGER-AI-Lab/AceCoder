# AceCode (Data Repository)
Welcome to the data directory for the AceCode project. In this folder, you can find scripts / code used to recreate the AceCode-87K and AceCodePair-300k.
**IMPORTANT: All instruction in this folder assumes your terminal is in the current folder (AceCoder/data/), please use ```cd data``` if you are not. We also use conda to manage our environment, so make sure you initialize to the correct interpreter:**

```bash
conda init
conda activate acecoder_data
```

## Installation
I assume you have **CUDA 12.1** and **conda** installed. With those, run the following command:

```bash
source setup.sh
```

You will be prompted some y/n options for installing packages, type y then enter in each instance.

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
```

## Dataset Curation
Follow the following steps closely to create AceCode-87K and AceCodePair-300K.

### Download datasets from hub
We will download the following datasets from huggingface and cache them locally:
- Bigcode Python Functions: [bigcode/stack-dedup-python-fns](https://huggingface.co/datasets/bigcode/stack-dedup-python-fns)
- OSS: [ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response)
- Evol Instruct: [ise-uiuc/Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K)

```bash
python training_dataset/bigcode_python_fns/preprocess.py
python training_dataset/evol/preprocess_evol.py
python training_dataset/oss/preprocess_oss.py
```

### Use GPT-4o-mini to convert seed code data into LeetCode-Like questions and test cases
First add the following environment variable to your shell (you can also add this to ~/.bashrc):
```bash
export OPENAI_API_KEYS="sk-your-openai-api-key"
export OPENAI_API_TYPE="OpenAI"
```

If you just want to sample a few questions to try it out, run (cost less than $1 USD):
```bash
python training_dataset/bigcode_python_fns/generate_test_cases.py --ct=50
python training_dataset/evol/generate_test_cases.py --ct=50
python training_dataset/oss/generate_test_cases.py --ct=50
```

If you want to fully recreate our dataset, run (this will cost you around $300 USD):
```bash
python training_dataset/bigcode_python_fns/generate_test_cases.py --ct=50000
python training_dataset/evol/generate_test_cases.py --ct=-1
python training_dataset/oss/generate_test_cases.py --ct=-1
```

### Creating Inferences for the generated leetcode-like prompts
Run the following to create inferences for the generated LeetCode-like prompts. This process is GPU heavy and you may want to set CUDA_VISIBLE_DEVICES if you do not wish to run the process on all of your gpus.
```bash
source training_dataset/inference_generated_prompts.sh
```

### Evaluate the inferenced code
**Note: this may drain up your CPU resources and it may also make unpredictable changes to your file system since we are executing generated code. You may want to run it in a docker for your safety.**

Run the following to compute the accuracies for the generated code:
```bash
source training_dataset/evaluate_inferenced_code.sh
```

### Consolidate the dataset
Run:
```bash
python data/training_dataset/consolidate_dataset.py
```

### Creating AceCode-98K
run:
```bash
python acecode_87K/generate_main_dataset.py
```

### Creating AceCodePair-300K
run:
```bash
source acecode_pair_300k/create_rm_dataset.sh
```