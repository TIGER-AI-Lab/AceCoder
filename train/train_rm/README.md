## Installation

```bash
cd LLaMA-Factory
pip install -e .
```

## Training Reward Model
AceCodeRM is trained on [TIGER-Lab/AceCodePair-300K](https://huggingface.co/datasets/TIGER-Lab/AceCodePair-300K).

```bash
llamafactory train configs/train_acecoderm.yaml
```
