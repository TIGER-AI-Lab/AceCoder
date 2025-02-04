import fire

from inference.EvaluateInferencedCode import process_one_model
from inference.post_process_functions import eval_post_process
from utility.utility import load_jsonl


def codeblock_post_process(input_str: str) -> str:
    left_idx = input_str.find("```python")
    if left_idx < 0:
        return codeblock_post_process_2(input_str)
    out = input_str[left_idx + 9 :]
    right_idx = out.find("```")
    if right_idx < 0:
        return codeblock_post_process_2(input_str)
    out = out[:right_idx]
    out = out.strip()
    return out


def codeblock_post_process_2(input_str: str) -> str:
    left_idx = input_str.find("```")
    if left_idx < 0:
        return input_str
    out = input_str[left_idx + 3 :]
    right_idx = out.find("```")
    if right_idx < 0:
        return input_str
    out = out[:right_idx]
    out = out.strip()
    return out


def evaluate_inferenced_code(
    model_name: str,
    dataset_name: str,
    model_type: str,
    sampling_method: str = "best_of_n_top_p_sampling",
):
    print(
        f"Starting evaluation for {dataset_name} - {model_name} {model_type} {sampling_method}"
    )
    "Get the accuracy of the inferenced program"
    data = load_jsonl(f"training_dataset/{dataset_name}/data/v2.jsonl")
    tests = [
        (i["tests"] if i["tests"] is not None and len(i["tests"]) > 0 else [])
        for i in data
    ]
    # short_tests = [i[:3] for i in tests]
    # instructions = [
    #     i["gpt_question"] if (i["gpt_question"] is not None) else "ignore this question"
    #     for i in data
    # ]
    # prompts = get_deepseek_coder_prompts(instructions, short_tests)
    # prompts = {i: prompts[i] for i in range(len(prompts))}
    if sampling_method == "greedy":
        fast_algo = True
        max_execution_time = 0.8
    else:
        fast_algo = True
        max_execution_time = 0.2
    process_one_model(
        model_name=f"{model_name}_{model_type}_{sampling_method}",
        dataset_name=dataset_name,
        tests=tests,
        max_execution_time=max_execution_time,
        processing_func=eval_post_process,
        fast_algo=fast_algo,
    )


if __name__ == "__main__":
    fire.Fire(evaluate_inferenced_code)

# if __name__ == "__main__":
#     evaluate_inferenced_code(gpt_35=True, sampling_method="top_p_sampling")
#     # evaluate_inferenced_code(gpt_35=False, sampling_method="diverse_beam_search")
#     get_solution_accuracy(gpt_35=True, ct=2000)
#     # get_solution_accuracy(gpt_35=False)
