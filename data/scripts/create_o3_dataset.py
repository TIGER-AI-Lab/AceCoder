from training_dataset.constants import DATASET_LST
from utility.utility import load_jsonl
from datasets import Dataset
def create_o3_seed():
    out = []
    for dataset_name in DATASET_LST:
        file_path = f"training_dataset/{dataset_name}/data/v1.jsonl"[:1000000]
        lst = load_jsonl(file_path)
        print(f"{dataset_name}: {lst[0].keys()}")
        for i in range(len(lst)):
            lst[i]["id"] = f"{dataset_name}_{i}"
        out += lst

    dataset = Dataset.from_list(out)
    dataset.push_to_hub("CodeDPO/o3_seed_data")

if __name__ == "__main__":
    create_o3_seed()