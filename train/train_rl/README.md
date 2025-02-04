# Installtion
You need to first update the submodule by running the following command:
```bash
git submodule init
git submodule update
```

Then, you can install the required packages for OpenRLHF with the following command:
```bash
conda create -n 
cd OpenRLHF
pip install -e .[vllm]
pip install evalplus # requried for rule-based reward for code generation
```

## Data Preparation
- To get the AceCode-89K-hard that only keeps 25% of the examples that makes the RL training faster, run the following command:
```bash
python scripts/get_hard_data.py --dataset_path "TIGER-Lab/AceCode-89K" --output_path "./data/acecode_89k/acecode_89k.json" --only_keep_hard_examples True
```

## Reward model preparation
Since [AceCodeRM-7B](https://huggingface.co/TIGER-Lab/AceCodeRM-7B) is trained with LlamaFactory, the format might be different from the OpenRLHF RM format, but it's generally the same. The only difference is that the Llamafactory enabled the `bias=True` for the final linear layer, while OpenRLHF uses `bias=False`.

Two ways to use RM for RL training:
- Directly set `reward_pretrain="TIGER-Lab/AceCodeRM-7B"` in the RL training script and set `value_head_prefix="summary"` in the training script.
- Convert the RM to OpenRLHF format weights with the following command:
```bash
python scripts/change_lf_rm_to_openrlhf_rm.py --lf_rm_model_path "TIGER-Lab/AceCodeRM-7B" --openrlhf_rm_model_path "./models/AceCodeRM-7B-openrlhf" --push_to_hub False
```
Then, set `reward_pretrain="./models/AceCodeRM-7B-openrlhf"` in the RL training script and set `value_head_prefix="score"` in the training script.

(Note: the reason why we use LlamaFactory for training RM is historical reason. We have tried using OpenRLHF to train RM, and the performance is similar.)


### Training RL

please `export WANDB_API_KEY=your_wandb_api_key` before running the following scripts.

- with reward model
```bash
bash scripts/train_reinforce_ray.sh # reinforcement++
# and change the following variables in the script
# policy_pretrain="Your initial policy model"
# reward_pretrain="TIGER-Lab/AceCodeRM-7B"
# dataset_path="./data/acecode_89k/acecode_89k.json"
# run_name="Your run name"
```
- with rule-based reward (binary pass rate)
```bash
bash scripts/train_reinforce_ray_rule_rm.sh # reinforcement++
# and change the following variables in the script
# policy_pretrain="Your initial policy model"
# binary_reward=True 
# dataset_path="./data/acecode_89k/acecode_89k.json"
# run_name="Your run name"
```