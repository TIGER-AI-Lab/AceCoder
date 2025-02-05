import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


def _load_processed_data(
    key: str,
    dataset_name: str,
    model_name: str,
    test_set_name: str = "default",
) -> Dict[int, List[Any]]:
    """get the saved processed, output is a dictionary where the key is the index and the value is the list of whatever field requested"""
    file_path = (
        f"inferenced output/{dataset_name}/processed/{test_set_name}/{model_name}.jsonl"
    )
    if not os.path.exists(file_path):
        raise Exception(f"No saved inference between {dataset_name} + {model_name}")

    with open(file_path, "r") as f:
        lst = f.readlines()
    lst = [json.loads(i) for i in lst]
    out = defaultdict(list)
    for i in lst:
        out[i["id"]].append(i[key])
    return dict(out)


def load_processed_model_accuracy(
    dataset_name: str, model_name: str, test_set_name: str = "default"
) -> Dict[int, List[str]]:
    """get the saved processed, output is a dictionary where the key is the index and the value is the list of accuracy"""
    return _load_processed_data(
        key="accuracy",
        dataset_name=dataset_name,
        model_name=model_name,
        test_set_name=test_set_name,
    )


def load_processed_model_tests_status(
    dataset_name: str, model_name: str, test_set_name: str = "default"
) -> Dict[int, List[str]]:
    """get the saved processed, output is a dictionary where the key is the index and the value is the list of accuracy"""
    return _load_processed_data(
        key="test_case_status",
        dataset_name=dataset_name,
        model_name=model_name,
        test_set_name=test_set_name,
    )


def get_oracle_test_case_status(
    dataset_name: str, model_name: str, test_set_name: str = "default"
) -> Dict[int, List[float]]:
    """For rach test case, if any one inference passed, we will consider that test case as passed"""
    data = load_processed_model_tests_status(
        dataset_name=dataset_name, model_name=model_name, test_set_name=test_set_name
    )
    data = {k: np.max(np.array(v), axis=0).tolist() for k, v in data.items()}
    return data


def get_oracle_accuracy(
    dataset_name: str, model_name: str, test_set_name: str = "default"
) -> float:
    """Compute the accuracy if you randomly select from the answer set. Note, prior to running
    this function you should have a jsonl file in f"inferenced output/{dataset_name}/processed/{model_name}.jsonl"
    where each line is a json object with the following 3 fields: id, accuracy, and response.

    Parameter:
        dataset_name
        model_name

    Return:
        the oracle accuracy
    """
    accuracy_dict = load_processed_model_accuracy(
        dataset_name, model_name, test_set_name
    )
    max_acc_lst = [max(accuracy_dict[i]) for i in accuracy_dict]
    return sum(max_acc_lst) / len(max_acc_lst)


def get_random_select_accuracy(
    dataset_name: str,
    model_name: str,
    test_set_name: str = "default",
    sample_ct: int = 10,
) -> float:
    """Compute the accuracy if you randomly select from the answer set. Note, prior to running
    this function you should have a jsonl file in f"inferenced output/{dataset_name}/processed/{model_name}.jsonl"
    where each line is a json object with the following 3 fields: id, accuracy, and response.

    Parameter:
        dataset_name
        model_name
        sample_ct: an integer indicating how many time you would like the
            program to do random sampling.

    Return:
        the randomly selectred accuracy
    """
    if sample_ct <= 0:
        raise Exception(f"sample_ct must be at least one, {sample_ct} provided")
    accuracy_dict = load_processed_model_accuracy(
        dataset_name, model_name, test_set_name=test_set_name
    )
    max_acc_lst = [
        [random.choice(accuracy_dict[idx]) for i in range(sample_ct)]
        for idx in accuracy_dict
    ]
    max_acc_lst = [sum(i) / len(i) for i in max_acc_lst]
    return sum(max_acc_lst) / len(max_acc_lst)


def get_average_select_accuracy(
    dataset_name: str,
    model_name: str,
    test_set_name: str = "default",
    sample_ct: int = 10,
) -> float:
    """Compute the accuracy if you average from the answer set. Note, prior to running
    this function you should have a jsonl file in f"inferenced output/{dataset_name}/processed/{model_name}.jsonl"
    where each line is a json object with the following 3 fields: id, accuracy, and response.

    Parameter:
        dataset_name
        model_name
        sample_ct: an integer indicating how many time you would like the
            program to do random sampling.

    Return:
        the average accuracy
    """
    if sample_ct <= 0:
        raise Exception(f"sample_ct must be at least one, {sample_ct} provided")
    accuracy_dict = load_processed_model_accuracy(
        dataset_name, model_name, test_set_name=test_set_name
    )
    max_acc_lst = [
        sum(accuracy_dict[idx]) / len(accuracy_dict[idx]) for idx in accuracy_dict
    ]
    return sum(max_acc_lst) / len(max_acc_lst)


def get_greedy_accuracy(
    dataset_name: str, model_name: str, test_set_name: str = "default"
) -> float:
    """Compute the accuracy if you randomly select from the answer set. Note, prior to running
    this function you should have a jsonl file in f"inferenced output/{dataset_name}/processed/{model_name}.jsonl"
    where each line is a json object with the following 3 fields: id, accuracy, and response.
    Moreover, if you give a best of n model, then this function will return the accuracy of the first
    response for each question.

    Parameter:
        dataset_name
        model_name

    Return:
        the one shot accuracy
    """
    accuracy_dict = load_processed_model_accuracy(
        dataset_name, model_name, test_set_name=test_set_name
    )
    max_acc_lst = [accuracy_dict[idx][0] for idx in accuracy_dict]
    return sum(max_acc_lst) / len(max_acc_lst)
