import json
import os
from typing import Dict, Iterable, List, Tuple

from inference.Constants import MODEL_PATH


def append_inference(dataset_name: str, model_name: str, lst: List[str]) -> None:
    """append to output file"""
    file_path = f"inferenced output/{dataset_name}/{model_name}.jsonl"
    dir_path = f"inferenced output/{dataset_name}/"
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "a") as f:
        for i in lst:
            f.write(i + "\n")


def get_saved_inference_index(dataset_name: str, model_name: str) -> int:
    """Check if previous inference has been done with regard to dataset and model. If so, then it will return the next id which will need to be inferenced
    If not, it will create the inference file"""
    file_path = f"inferenced output/{dataset_name}/{model_name}.jsonl"
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                lst = f.readlines()
                if len(lst) > 0:
                    dic = json.loads(lst[-1])
                    return dic["id"] + 1
                else:
                    return 0
        else:
            dir_path = f"inferenced output/{dataset_name}/"
            os.makedirs(dir_path, exist_ok=True)
            open(file_path, "a").close()
            return 0
    except:
        print(dataset_name)
        print(model_name)
        raise Exception(f"Failed to read {file_path}")


def load_saved_inference(dataset_name: str, model_name: str) -> Dict[int, List[str]]:
    """get the saved inference, output is a dictionary where the key is the index and the value is the response"""
    file_path = f"inferenced output/{dataset_name}/{model_name}.jsonl"
    if not os.path.exists(file_path):
        raise Exception(f"No saved inference between {dataset_name} + {model_name}")

    with open(file_path, "r") as f:
        lst = f.readlines()
    lst = [json.loads(i) for i in lst]
    out = {}
    for i in lst:
        if i["id"] in out:
            out[i["id"]].append(i["response"])
        else:
            out[i["id"]] = [i["response"]]
    return out


def load_processed_inference(
    dataset_name: str, model_name: str
) -> Dict[int, List[Tuple[str, float]]]:
    """get the processed inference with information such that output is a dictionary where:
    the key is the index and the value is a list of tuple in the following form: (code in string, accuracy)
    """
    file_path = f"inferenced output/{dataset_name}/processed/{model_name}.jsonl"
    if not os.path.exists(file_path):
        raise Exception(f"No processed inference between {dataset_name} + {model_name}")

    with open(file_path, "r") as f:
        lst = f.readlines()

    lst = [json.loads(i) for i in lst]
    out = {}
    for i in lst:
        if i["id"] in out:
            out[i["id"]].append((i["response"], i["accuracy"]))
        else:
            out[i["id"]] = [(i["response"], i["accuracy"])]
    return out


def print_inferenced_output(
    dataset_name: str, model_name: str, indices: int | Iterable[int] = range(10)
) -> None:
    """Print the inferenced output to the terminal,

    Parameter:
        dataset_name
        model_name
        indices: either an integer or a list of integers which you would like to print
    """

    inferences = load_saved_inference(dataset_name, model_name)
    if type(indices) == int:
        for sentence in inferences[indices]:
            print(sentence)
        return

    for i in indices:
        for sentence in inferences[i]:
            print(f"Index {i}:")
            print(sentence)


def get_huggingface_model_path(model_name: str, model_size: str) -> str:
    """Get the huggingface model path

    Parameter:
        model_name: a string such as "qwen_coder_2.5"
        model_size: a string such as "7b"
    """

    if model_name not in MODEL_PATH:
        raise Exception(f"{model_name} not in MODEL Constants")
    if model_size not in MODEL_PATH[model_name]:
        raise Exception(
            f"{model_size} not found for {model_name}, available ones are: {list(MODEL_PATH[model_name].keys())}"
        )
    return MODEL_PATH[model_name][model_size]


def get_suggested_inference_batch_size(model_size: str | float) -> int:
    """Get the suggested inference batchsize

    Parameter:
        model_size: an float such as 7 representing 7B parameters or a string such as '7b'
    """
    if type(model_size) == str:
        model_size = float(model_size[:-1])
    if model_size <= 10:
        return 64
    elif model_size <= 40:
        return 16
    elif model_size <= 80:
        return 4
    else:
        return 2
