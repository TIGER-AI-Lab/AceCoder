import json
import os
from typing import List

from training_dataset.constants import DATASET_LST, MODEL_LST
from utility.utility import load_jsonl

template_input_str = """Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python
"""


def remove_assert_statment(input_str: str):
    lst = input_str.splitlines()
    lst = [i for i in lst if not i.strip().startswith("assert")]
    return "\n".join(lst)


def get_mbpp_style_prompt(program1: str, program2: str, prompt: str, test: List[str]):
    """
    Create a prompt with the following style:
    Write a function to find the shared elements from the given two lists.
    assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
    """
    mbpp_prompt = f'"""\n{prompt}\n{test[0].strip()}\n"""\n'
    mbpp_prompt = prompt_final_post_process(mbpp_prompt)
    output = {
        "instruction": mbpp_prompt,
        "input": "",
        "chosen": template_input_str + program1 + "\n```",
        "rejected": template_input_str + program2 + "\n```",
    }
    return output


def humaneval_split_program(program):
    idx = program.find("def ")
    idx = program.find("\n", idx + 1)
    starting_code = program[:idx]
    ending_code = program[idx:]
    return starting_code, ending_code


def compare_starting_code(prog1: str, prog2: str):
    """See if two starting code are kind of similar, by splitting by lines"""
    prog1 = [line.strip() for line in prog1.splitlines()]
    prog2 = [line.strip() for line in prog2.splitlines()]
    prog1 = [i for i in prog1 if len(i) > 0]
    prog2 = [i for i in prog2 if len(i) > 0]
    if len(prog1) != len(prog2):
        return False
    for i in prog1:
        if i not in prog2:
            return False
    return True


def get_human_eval_style_prompt(
    program1: str, program2: str, prompt: str, test: List[str]
):
    """
        Create a prompt with the following style:
    from typing import List
    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        " Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        "

    """
    if "Class" in program1 or program1.count("def ") != 1:
        # we should use MBPP style for this prompt
        return None

    starting_code1, ending_code1 = humaneval_split_program(program1)
    starting_code2, ending_code2 = humaneval_split_program(program2)
    if not compare_starting_code(starting_code1, starting_code2):
        return None  # we use mbpp too

    humaneval_prompt = f'{starting_code1}\n    """{prompt}\n    >>> {test[0]}\n    >>> {test[1]}\n    """\n'
    humaneval_prompt = prompt_final_post_process(humaneval_prompt)
    output = {
        "instruction": humaneval_prompt,
        "input": template_input_str,
        "chosen": humaneval_prompt + ending_code1 + "\n```\n",
        "rejected": humaneval_prompt + ending_code2 + "\n```\n",
    }
    return output


def prompt_final_post_process(input_str: str) -> str:
    out = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
{input_str}
```
"""
    return out


def remove_start_header(input_str: str):
    START_HEADERS = ["<|start_header_id|>assistant<|end_header_id|>"]
    for i in START_HEADERS:
        if input_str.startswith(i):
            return input_str[len(i) :].strip()
    return input_str.strip()


def get_scaled_weight(acc1: float, acc2: float) -> int:
    diff = abs(acc1 - acc2)
    if diff <= 0.5:
        return 1
    elif diff <= 0.75:
        return 3
    else:
        return 10


def convert_dataset(
    dataset_lst: List[str],
    model_name: str = "cross_model",
    return_size: int = 1,
    scaled: bool = False,
):
    """Convert the dataset

    Parameter:
        model_name: the model name used to create the DPO dataset
        return_size: how many entries max can be generated from each question
    """
    out = []
    for dataset in dataset_lst:
        file_name = (
            f"training_dataset/{dataset}/data/dpo_{model_name}_{return_size}.jsonl"
        )
        lst = load_jsonl(file_name)
        mbpp_style_ct = 0
        human_eval_style_ct = 0
        for i in lst:
            # prog1 = remove_assert_statment(i["program_1"])
            # prog2 = remove_assert_statment(i["program_2"])
            prog1 = i["program_1"]
            prog2 = i["program_2"]
            prompt = get_mbpp_style_prompt(
                program1=remove_start_header(prog1),
                program2=remove_start_header(prog2),
                prompt=i["prompt"],
                test=i["tests"],
            )
            if scaled:
                prompt["scaled"] = get_scaled_weight(i["accuracy_1"], i["accuracy_2"])
            out.append(prompt)
            continue  # we just use mbpp for now
            if mbpp_style_ct < human_eval_style_ct:
                out.append(
                    get_mbpp_style_prompt(
                        program1=prog1,
                        program2=prog2,
                        prompt=i["prompt"],
                        test=i["tests"],
                    )
                )
                mbpp_style_ct += 1
            else:
                ram = get_human_eval_style_prompt(
                    program1=prog1,
                    program2=prog2,
                    prompt=i["prompt"],
                    test=i["tests"],
                )
                if ram is None:
                    out.append(
                        get_mbpp_style_prompt(
                            program1=prog1,
                            program2=prog2,
                            prompt=i["prompt"],
                            test=i["tests"],
                        )
                    )
                    mbpp_style_ct += 1
                else:
                    out.append(ram)
                    human_eval_style_ct += 1

    # print(f"MBPP style ct: {mbpp_style_ct}, humaneval style ct: {human_eval_style_ct}")
    os.makedirs("generated_datasets", exist_ok=True)
    out_file_name = f"generated_datasets/dpo_{model_name}_{return_size}.json"
    with open(out_file_name, "w") as f:
        f.write(json.dumps(out, indent=4))
    print(f"Finished Generating for {model_name}, {dataset_lst}")


if __name__ == "__main__":
    for model in list(MODEL_LST.keys()) + ["cross_models"]:
        for return_size in [
            "inf",
            # 5,
            # 1
        ]:
            convert_dataset(
                dataset_lst=DATASET_LST,
                model_name=model,
                return_size=return_size,
                scaled=False,
            )
