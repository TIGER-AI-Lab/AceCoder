import json

from datasets import Dataset

from training_dataset.constants import DATASET_LST
from utility.utility import load_jsonl

# Use this script to generate the "AceCode-89K dataset, which contains the questions, tests, inferences, etc."


def generate_entries(inf_model_name: str, huggingface_dataset_path: str):
    json_file_name = f"generated_datasets/dpo_{inf_model_name}_inf.json"
    with open(json_file_name, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        data[i]["id"] = i

    dataset = Dataset.from_list(data)
    dataset.push_to_hub(huggingface_dataset_path)


if __name__ == "__main__":
    inf_model_name = "qwen_coder_2.5"
    huggingface_dataset_path = "TIGER-Lab/AceCodePair-300K"
    generate_entries(
        inf_model_name=inf_model_name, huggingface_dataset_path=huggingface_dataset_path
    )
