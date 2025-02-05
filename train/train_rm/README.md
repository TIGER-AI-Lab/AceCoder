## Installation

We use LLama-Factory off the shelf for our reward model training. Therefore, to install the environment, please refer to their repository. At the time of writing this page, the following scripts work for us:
```bash
conda create -n llamaFactory python=3.11
conda init
conda activate llamaFactory
pip install -e ".[torch,metrics]"
pip install deepspeed==0.15.4
pip install -U "huggingface_hub[cli]"
```

## Setup
Please complete the following steps:
1. Move the 3 files under configs into the llamafactory directory after you have cloned it.
2. Add the following two entries to `LLaMA-Factory/data/dataset_info.json`:
```json
"AceCodePair-300K": {
    "hf_hub_url": "TIGER-Lab/AceCodePair-300K",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },
"AceCodePair-QwenCoderIns32B": {
    "hf_hub_url": "TIGER-Lab/AceCodePair-QwenCoderIns32B",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
```

## Training
1. Change the `output_dir` field in the yaml files that you have copied for the desired model output path.
2. Run:
```bash
llamafactory-cli train train_qwen_coder_ins_2.5_{7/32}b.yaml
```
