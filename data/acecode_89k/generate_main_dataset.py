from training_dataset.constants import DATASET_LST
from utility.utility import load_jsonl, save_jsonl

# Use this script to generate the "AceCode-87K dataset, which contains the questions, tests, inferences, etc."


def generate_entries(oracle_model: str, save_path: str):
    out = []
    for dataset_name in DATASET_LST:
        jsonl_file_name = (
            f"training_dataset/{dataset_name}/data/v3_{oracle_model}.jsonl"
        )
        data = load_jsonl(jsonl_file_name)
        for entry in data:
            id = entry["id"]
            prompt = entry["prompt"]
            tests = entry["tests"]
            inferences = entry["inferences"]
            if prompt is None or len(tests) < 5:
                continue
            inferences_out = []
            for i, (inf, acc, inf_model) in enumerate(inferences):
                inferences_out.append(
                    {
                        "model_name": inf_model,
                        "completion_id": i,
                        "completion": inf,
                        "pass_rate": acc,
                    }
                )
            out.append(
                {
                    "id": id,
                    "source": dataset_name,
                    "question": prompt,
                    "test_cases": tests,
                    "inferences": inferences_out,
                    "context_messages": [{"content": prompt, "role": "user"}],
                }
            )

    save_jsonl(save_path, out)


if __name__ == "__main__":
    oracle_model = "qwen_coder_2.5_32b_greedy"
    save_path = "AceCode-87K.jsonl"
    generate_entries(oracle_model=oracle_model, save_path=save_path)
