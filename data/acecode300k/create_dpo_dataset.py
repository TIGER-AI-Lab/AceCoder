from typing import Any, Dict, List, Tuple

from training_dataset.constants import DATASET_LST, MODEL_LST
from utility.utility import load_jsonl, save_jsonl


def create_cross_model_dataset(
    dataset_name: str, return_size: str, oracle_model_name: str
):
    """Create DPO dataset but the data can come from either any model"""
    dataset = []
    data = load_jsonl(
        f"training_dataset/{dataset_name}/data/v3_{oracle_model_name}.jsonl"
    )

    generated_lst = []  # this list tracks the number of generated entries
    for i in range(len(data)):
        # tuple in the form: (code, accuracy, model name)
        if data[i]["prompt"] is None:
            # we skip because it's None
            test_cases = []
        elif not data[i]["prompt"].isascii():
            # we skip because it contains non-ascii code
            test_cases = []
        else:
            test_cases = create_dataset_helper_1(
                data[i]["inferences"],
                data[i]["prompt"],
                data[i]["tests"],
                return_size=return_size,
            )
        dataset += test_cases
        # print(f"question {i} generated {len(test_cases)} entries")
        generated_lst.append(len(test_cases))
    no_entry_ct = len([i for i in generated_lst if i == 0])
    print(
        f"{dataset_name} - cross model - {return_size} - generated {sum(generated_lst)} entries, average yield is {sum(generated_lst) / len(data)}. {no_entry_ct} entries (or {no_entry_ct / len(data) * 100}%) produced no entires at all."
    )

    save_jsonl(
        f"training_dataset/{dataset_name}/data/dpo_cross_models_{return_size}.jsonl",
        dataset,
    )
    return generated_lst


def create_dataset_with_only_one_model(
    model_name: str, dataset_name: str, oracle_model_name: str, return_size: int = 1
):
    """Create DPO dataset but using questions from only 1 model"""
    dataset = []
    data = load_jsonl(
        f"training_dataset/{dataset_name}/data/v3_{oracle_model_name}.jsonl"
    )

    generated_lst = []  # this list tracks the number of generated entries
    for i in range(len(data)):
        if data[i]["prompt"] is None:
            # we skip because it's None
            test_cases = []
        elif not data[i]["prompt"].isascii():
            # we skip because it contains non-ascii code
            test_cases = []
        else:
            test_cases = create_dataset_helper_1(
                data[i]["inferences"],
                data[i]["prompt"],
                data[i]["tests"],
                specific_model_name=model_name,
                return_size=return_size,
            )
        dataset += test_cases
        # print(f"question {i} generated {len(test_cases)} entries")
        generated_lst.append(len(test_cases))
    no_entry_ct = len([i for i in generated_lst if i == 0])
    print(
        f"{dataset_name} - {model_name} - {return_size} - generated {sum(generated_lst)} entries, average yield is {sum(generated_lst) / len(data)}. {no_entry_ct} entries (or {no_entry_ct / len(data) * 100}%) produced no entires at all."
    )

    save_jsonl(
        f"training_dataset/{dataset_name}/data/dpo_{model_name}_{return_size}.jsonl",
        dataset,
    )
    return generated_lst


def create_dataset_helper_1(
    inferences: List[Tuple],
    prompt: Any,
    tests: Any,
    specific_model_name: str = None,
    return_size: int = 1,
) -> List[Dict]:
    """Create a dataset for 1 prompt, this is a helper function for the overall create_dataset function.

    Input:
        inferences: a list of tuple in the form: (code, accuracy, model name)
        prompt: the prompt for the question, will be appended to each test case
        tests: the tests for each question, will be appended to each test case
        specific_model_name: if you only want to generate test cases from 1 model

    output:
        A list of dictionary, each representing one question
    """
    inferences_by_model = {}
    for program, acc, model in inferences:
        if specific_model_name is not None:
            if model != specific_model_name:
                continue
        if model in inferences_by_model:
            inferences_by_model[model].append((program, acc, model))
        else:
            inferences_by_model[model] = [(program, acc, model)]

    model_lst = list(inferences_by_model.keys())
    output = []

    # we first generate test cases for each model
    for model in model_lst:
        output += create_dataset_helper_2(
            inferences_by_model[model], return_size=return_size
        )

    # we now generate cross-model test cases
    # if specific_model_name is None:
    #     for i in range(len(model_lst) - 1):
    #         for j in range(i + 1, len(model_lst)):
    #             tmp_inf = (
    #                 inferences_by_model[model_lst[i]]
    #                 + inferences_by_model[model_lst[j]]
    #             )
    #             output += create_dataset_helper_2(
    #                 tmp_inf, return_size=return_size, require_different_model=True
    #             )

    for i in output:
        i.update({"prompt": prompt, "tests": tests})

    return output


# The below are old code which we use the difference in accuracy to return the output. In a newer approached
# we will instead keep the preference choice constant
# def create_dataset_helper_2(
#     inferences: List[Tuple], return_size: int = 3, require_different_model: bool = False
# ) -> List[Dict]:
#     """Create a dataset for 1 prompt, this is a helper function for the overall create_dataset function.

#     Input:
#         tuple in the form: (code, accuracy, model name)
#         return_size: how many entries do you want the program to return
#         require_different_model: should the return models be from different dataset

#     output:
#         A list of dictionary, each representing one question
#     """
#     output = []
#     seen = set()
#     for j in range(len(inferences) - 1):
#         for k in range(j + 1, len(inferences)):
#             prog1, acc1, model1 = inferences[j]
#             prog2, acc2, model2 = inferences[k]
#             if require_different_model and model1 == model2:
#                 continue
#             if (
#                 min(acc1, acc2) > 0
#                 and max(acc1, acc2) >= 0.4
#                 and abs(acc2 - acc1) >= 0.2
#             ):
#                 if acc1 > acc2:
#                     prog_high = prog1
#                     acc_high = acc1
#                     model_high = model1
#                     prog_low = prog2
#                     acc_low = acc2
#                     model_low = model2
#                 else:
#                     prog_high = prog2
#                     acc_high = acc2
#                     model_high = model2
#                     prog_low = prog1
#                     acc_low = acc1
#                     model_low = model1
#                 ram = prog_high + prog_low
#                 if ram not in seen:
#                     # we prevent duplicates
#                     entry = {
#                         "program_1": prog_high,
#                         "program_2": prog_low,
#                         "winner": 1,
#                         "accuracy_1": acc_high,
#                         "accuracy_2": acc_low,
#                         "accuracy_difference": abs(acc_high - acc_low),
#                         "model_1": model_high,
#                         "model_2": model_low,
#                     }
#                     output.append(entry)
#                     seen.add(ram)
#     output.sort(key=lambda x: abs(x["accuracy_1"] - x["accuracy_2"]), reverse=True)
#     return output[:return_size]


def create_dataset_helper_2(
    inferences: List[Tuple], return_size: int = 3, require_different_model: bool = False
) -> List[Dict]:
    """Create a dataset for 1 prompt, this is a helper function for the overall create_dataset function.

    Input:
        tuple in the form: (code, accuracy, model name)
        return_size: how many entries do you want the program to return. If less than 1 then will return all entries
        require_different_model: should the return models be from different dataset

    output:
        A list of dictionary, each representing one question
    """
    output = []
    seen = set()
    inferences = sorted(inferences, key=lambda x: x[1], reverse=True)
    highest_acc = inferences[0][1]
    for j in range(len(inferences) - 1):
        for k in range(len(inferences) - 1, j, -1):
            prog1, acc1, model1 = inferences[j]
            prog2, acc2, model2 = inferences[k]
            if require_different_model and model1 == model2:
                continue
            # if acc1 < highest_acc:
            #     continue  # we only want highest accuracy
            if (
                max(acc1, acc2) >= 0.8
                and abs(acc2 - acc1) >= 0.4
                and min(acc1, acc2) > 0
            ):
                ram = prog1 + prog2
                if ram not in seen:
                    # we prevent duplicates
                    entry = {
                        "program_1": prog1,
                        "program_2": prog2,
                        "winner": 1,
                        "accuracy_1": acc1,
                        "accuracy_2": acc2,
                        "accuracy_difference": abs(acc1 - acc2),
                        "model_1": model1,
                        "model_2": model2,
                    }
                    output.append(entry)
                    seen.add(ram)
                    if return_size != "inf":
                        if len(output) >= return_size and return_size > 0:
                            return output
    return output


if __name__ == "__main__":
    # latest oracle setting: "qwen_coder_2.5_32b_greedy"
    oracle_model_name = "qwen_coder_2.5_32b_greedy"
    for dataset in DATASET_LST:
        # get_solution_accuracy(dataset_name=dataset)
        for return_size in [
            "inf",
            # 5,
            # 1,
        ]:
            create_cross_model_dataset(
                dataset_name=dataset,
                return_size=return_size,
                oracle_model_name=oracle_model_name,
            )
            for model in MODEL_LST.keys():
                create_dataset_with_only_one_model(
                    model_name=model,
                    dataset_name=dataset,
                    return_size=return_size,
                    oracle_model_name=oracle_model_name,
                )
