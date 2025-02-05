import os
from typing import List

from tqdm import tqdm

from training_dataset.oss.oss_dataset import get_oss_dataset
from training_dataset.util import (get_python_code_from_string,
                                   remove_print_statements_from_python_program)
from utility.utility import load_jsonl, save_jsonl


def get_oss_programs(use_cache: bool = True) -> List[str]:
    """Step 1 of the process, extract python programs and instructions from the dataset. We only keep programs in function or class form."""
    file_name = "training_dataset/oss/data/v1.jsonl"
    if os.path.exists(file_name) and use_cache:
        out = load_jsonl(file_name)
        return [i["program"] for i in out]
    os.makedirs("training_dataset/oss/data/", exist_ok=True)
    data = get_oss_dataset()
    out = []
    idx = 0
    for i in tqdm(range(len(data))):
        if data[i]["lang"] != "python":
            continue  # only do python for now
        program = get_python_code_from_string(data[i]["response"])
        instruction = data[i]["instruction"]
        if len(program) == 0:
            # no python code found
            continue
        program = remove_print_statements_from_python_program(program)
        out.append({"id": idx, "instruction": instruction, "program": program})
        idx += 1
    save_jsonl(file_name, out)
    return [i["program"] for i in out]


if __name__ == "__main__":
    get_oss_programs()
