import json
import os
from typing import Any, Dict, List


def load_jsonl(file_path: str) -> List[Dict[Any, Any]]:
    """load a .jsonl file. Return a List of dictionary, where each dictionary is a line in the file"""
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} Does not exist!!!!")
    with open(file_path, "r") as f:
        lst = f.readlines()
    lst = [json.loads(i) for i in lst]
    return lst


def get_python_code_from_string(input: str) -> str:
    """Basically find code wrapped in ```python ... ``` and return it. If none is found then will return the
    empty string"""
    left_index = input.find("```python")
    if left_index < 0:
        return ""
    input = input[left_index + 9 :]
    right_index = input.find("```")
    if right_index < 0:
        return ""
    input = input[:right_index]
    return input


def parse_incomplete_json(input: str) -> Any:
    """A helper function that will:
    1. try to parse the whole thing as json
    2. try to find json object wrapped in ```json ... ``` and parse it
    3. Try to see if the json is incomplete. if so then try to parse the incomplete json

    This will only work when we are missing ]} at the end, modify if you need it for other
    cases.
    """
    input = input.strip()
    left_idx = input.find("```json")
    if left_idx >= 0:
        input = input[left_idx + 7 :]
    right_idx = input.rfind("```")
    if right_idx >= 0:
        input = input[:right_idx]
    try:
        out = json.loads(input)
        return out
    except:
        pass

    # we now assume that the string is incomplete
    while len(input) > 0:
        try:
            data = json.loads(input + "]}")
            return data
        except json.decoder.JSONDecodeError:
            input = input[:-1]
    # we cannot parse this
    return {"question": None, "tests": None}


def remove_print_statements_from_python_program(input: str) -> str:
    lst = input.splitlines()
    lst = [i for i in lst if not i.strip().startswith("print")]
    return "\n".join(lst)


def print_data(file: str, idx: int = 0):
    data = load_jsonl(file)
    data = [row for row in data if row["id"] == idx][0]
    for key in data:
        print(f"----------------{key}:-------------------")
        if type(data[key]) == list:
            for i in data[key]:
                if type(i) == list:
                    # we omit the original inferences for easier print statements
                    for ii in i:
                        print(ii)
                    break
                else:
                    print(i)
            print(f"Contained {len(data[key])} items-----")
        else:
            print(data[key])


if __name__ == "__main__":
    print_data("training_dataset/oss/data/v2.jsonl", 22)
