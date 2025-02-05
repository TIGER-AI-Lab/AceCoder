import json

from utility.utility import save_jsonl

# Use this script to generate the "AceCode-89K dataset, which contains the questions, tests, inferences, etc."


def generate_entries(inf_model_name: str, save_path: str):
    json_file_name = f"generated_datasets/dpo_{inf_model_name}_inf.json"
    with open(json_file_name, "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        data[i]["id"] = i

    save_jsonl(save_path, data)


if __name__ == "__main__":
    inf_model_name = "qwen_coder_2.5"
    save_path = "AceCodePair-300K.jsonl"
    generate_entries(inf_model_name=inf_model_name, save_path=save_path)
