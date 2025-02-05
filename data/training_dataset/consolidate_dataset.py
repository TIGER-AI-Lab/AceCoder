from collections import defaultdict
from types import NoneType

from inference.ComputeAccuracy import get_oracle_test_case_status
from training_dataset.constants import DATASET_LST, MODEL_LST
from utility.utility import MyTimer, load_jsonl, save_jsonl


def recursive_clean(obj):
    """Clean an object and remove all non-utf-8 characters"""
    if type(obj) in [int, float, bool, NoneType]:
        return obj
    if type(obj) == str:
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    elif type(obj) == list:
        return [recursive_clean(i) for i in obj]
    elif type(obj) == dict:
        return {recursive_clean(k): recursive_clean(v) for k, v in obj.items()}
    else:
        raise Exception(f"Unknown object type: {type(obj)}: {str(obj)[:300]}")


def consolidate_processed_data(
    dataset_name: str,
    ct: int = -1,
    oracle_model_name: str = "qwen_coder_2.5_32b_greedy",
    min_oracle_model_pass_case_requirement: int = 3,
):
    """Get accuracy of the provided solution from the dataset"""
    timer = MyTimer()
    # we now append each entry with all the inferenced answers and solution accuracy
    data = recursive_clean(load_jsonl(f"training_dataset/{dataset_name}/data/v2.jsonl"))
    test_case_status = get_oracle_test_case_status(
        dataset_name=dataset_name, model_name=oracle_model_name
    )
    if ct > 0:
        data = data[:ct]
    timer.print_runtime(f"{dataset_name} loading data and oracle answer")

    # adding the inferenced code as "inferences" to the dataset
    inferenced_program = {}
    for short_model_name, full_model_name in MODEL_LST.items():
        acc_file_path = f"inferenced output/{dataset_name}/processed/default/{full_model_name}.jsonl"
        acc_lst = recursive_clean(load_jsonl(acc_file_path))
        ram_dict = defaultdict(list)
        for acc_row in acc_lst:
            ram_dict[acc_row["id"]].append(acc_row)
        inferenced_program[short_model_name] = ram_dict
        timer.print_runtime(f"{dataset_name} loading {short_model_name}'s inference")

    out = []
    for i, row in enumerate(data):
        tmp_lst = []
        ground_truth_test_case = test_case_status[i]
        if sum(ground_truth_test_case) <= min_oracle_model_pass_case_requirement:
            continue  # less than the minimum requirement, so we just skipped
        for model_name in MODEL_LST:
            for roww in inferenced_program[model_name][i]:
                model_test_case = roww["test_case_status"]
                new_pass_rate = [
                    a * b for a, b in zip(ground_truth_test_case, model_test_case)
                ]
                acc = sum(new_pass_rate) / len(new_pass_rate)
                # tuple in the form: (code, accuracy, model name)
                tmp_lst.append((roww["response"], acc, model_name))
        tests = [
            test
            for test_idx, test in enumerate(row["tests"])
            if ground_truth_test_case[test_idx] == 1
        ]
        to_be_add_entry = {
            "id": f"{dataset_name}_{row['id']}",
            "prompt": row["gpt_question"],
            "tests": tests,
            "inferences": tmp_lst,
        }
        out.append(to_be_add_entry)
    timer.print_runtime(f"{dataset_name} creating dataset")

    jsonl_file_name = (
        f"training_dataset/{dataset_name}/data/v3_{oracle_model_name}.jsonl"
    )
    save_jsonl(jsonl_file_name, out)
    timer.print_runtime(f"{dataset_name} saving v3 data")


def consolidate_processed_data_without_oracle(
    dataset_name: str,
    min_test_case_requirement: int = 3,
    ct: int = -1,
):
    """Get accuracy of the provided solution from the dataset"""
    timer = MyTimer()
    # we now append each entry with all the inferenced answers and solution accuracy
    data = recursive_clean(load_jsonl(f"training_dataset/{dataset_name}/data/v2.jsonl"))
    if ct > 0:
        data = data[:ct]
    timer.print_runtime(f"{dataset_name} loading data and oracle answer")

    # adding the inferenced code as "inferences" to the dataset
    inferenced_program = {}
    for short_model_name, full_model_name in MODEL_LST.items():
        acc_file_path = f"inferenced output/{dataset_name}/processed/default/{full_model_name}.jsonl"
        acc_lst = recursive_clean(load_jsonl(acc_file_path))
        ram_dict = defaultdict(list)
        for acc_row in acc_lst:
            ram_dict[acc_row["id"]].append(acc_row)
        inferenced_program[short_model_name] = ram_dict
        timer.print_runtime(f"{dataset_name} loading {short_model_name}'s inference")

    out = []
    for i, row in enumerate(data):
        tmp_lst = []
        for model_name in MODEL_LST:
            for roww in inferenced_program[model_name][i]:
                model_test_case = roww["test_case_status"]
                if len(model_test_case) < min_test_case_requirement:
                    continue
                acc = sum(model_test_case) / len(model_test_case)
                # tuple in the form: (code, accuracy, model name)
                tmp_lst.append((roww["response"], acc, model_name))
        tests = row["tests"]
        to_be_add_entry = {
            "id": f"{dataset_name}_{row['id']}",
            "prompt": row["gpt_question"],
            "tests": tests,
            "inferences": tmp_lst,
        }
        out.append(to_be_add_entry)
    timer.print_runtime(f"{dataset_name} creating dataset")

    jsonl_file_name = f"training_dataset/{dataset_name}/data/v3_no_oracle.jsonl"
    save_jsonl(jsonl_file_name, out)
    timer.print_runtime(f"{dataset_name} saving v3 data")


if __name__ == "__main__":
    for dataset_name in DATASET_LST:
        consolidate_processed_data(
            dataset_name=dataset_name, oracle_model_name="qwen_coder_2.5_32b_greedy"
        )
