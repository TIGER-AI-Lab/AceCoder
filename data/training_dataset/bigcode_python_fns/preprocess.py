import os
from typing import List

from tqdm import tqdm

from training_dataset.bigcode_python_fns.dataset import \
    get_bigcode_python_fn_dataset
from training_dataset.util import remove_print_statements_from_python_program
from utility.utility import load_jsonl, save_jsonl


def get_bigcode_python_fns_programs(use_cache: bool = True) -> List[str]:
    """Step 1 of the process, extract python programs and instructions from the dataset. We only keep programs in function or class form."""
    file_name = "training_dataset/bigcode_python_fns/data/v1.jsonl"
    if os.path.exists(file_name) and use_cache:
        out = load_jsonl(file_name)
        return [i["program"] for i in out]
    os.makedirs("training_dataset/bigcode_python_fns/data/", exist_ok=True)
    data = get_bigcode_python_fn_dataset()
    out = []
    idx = 0
    for i in tqdm(range(len(data))):
        program = data[i]["content"]
        if len(program) <= 100:
            # too short
            continue
        program = remove_print_statements_from_python_program(program)
        out.append({"id": idx, "program": program})
        idx += 1
    save_jsonl(file_name, out)
    return [i["program"] for i in out]


if __name__ == "__main__":
    get_bigcode_python_fns_programs()
