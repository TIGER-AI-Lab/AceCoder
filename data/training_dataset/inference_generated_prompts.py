from typing import Dict, List

import fire
from transformers import AutoTokenizer

from inference.InferenceModels import inference
from inference.Utility import get_huggingface_model_path
from utility.utility import load_jsonl

TEMPLATE_INPUT_PROMPT = """Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python"""


def get_untokenized_prompt(
    questions: List[str],
    tests: List[List[str]],
    test_ct: int = 2,
) -> List[List[Dict[str, str]]]:
    """Prepare prompts for deepseek coder

    Arguments:
    questions: a list of questions being asked for the LLM
    tests: a list of tests to be included in the prompt
    test_ct: how many test do you want to reveal to the LLM? default value is -1 which means show all
    """
    if test_ct >= 0:
        tests = [i[:test_ct] for i in tests]
    output = []
    for question, test in zip(questions, tests):
        test_prompt = "\n".join(test)
        question_prompt = f'"""\n{question}\n{test_prompt}\n"""'
        instruction_prompt = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
{question_prompt}
```
"""
        chat = [
            {"role": "user", "content": instruction_prompt},
            {"role": "assistant", "content": TEMPLATE_INPUT_PROMPT},
        ]

        output.append(chat)

    return output


def get_tokenized_prompt(
    model_path: str,
    questions: List[str],
    tests: List[List[str]],
    test_ct: int = 2,
) -> List[str]:
    """Prepare prompts for deepseek coder

    Arguments:
    questions: a list of questions being asked for the LLM
    tests: a list of tests to be included in the prompt
    test_ct: how many test do you want to reveal to the LLM? default value is -1 which means show all
    """
    chats = get_untokenized_prompt(questions=questions, tests=tests, test_ct=test_ct)
    output = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for chat in chats:
        try:
            token_ids = tokenizer.apply_chat_template(
                chat,
                continue_final_message=True,
                add_generation_prompt=False,
            )[
                :-1
            ]  # Remote EOS token
            output.append(token_ids)
        except Exception as e:
            print(e)
            print(f"error tokenizing: {chat}")
            output.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "please ignore this quesiton"}],
                    continue_final_message=True,
                    add_generation_prompt=False,
                )
            )

    return output


def create_inference(
    model_name: str,
    dataset_name: str,
    model_type: str,
    ct: int = -1,
    sampling_method: str = "best_of_n_top_p_sampling",
) -> None:
    print(
        f"starting inference: model name: {model_name} {model_type}, dataset name: {dataset_name}"
    )

    load_file_name = f"training_dataset/{dataset_name}/data/v2.jsonl"

    data = load_jsonl(load_file_name)
    if ct > 0:
        data = data[:ct]
    instructions = [
        i["gpt_question"] if (i["gpt_question"] is not None) else "ignore this question"
        for i in data
    ]
    tests = [
        i["tests"][:3] if (i["gpt_question"] is not None) else ["ignore this question"]
        for i in data
    ]
    huggingface_path = get_huggingface_model_path(
        model_name=model_name, model_size=model_type
    )
    input_token_ids = get_tokenized_prompt(
        model_path=huggingface_path, questions=instructions, tests=tests
    )

    inference(
        model_name=model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        sampling_method=sampling_method,
        input_token_ids=input_token_ids,
        inference_method="vllm",
    )


if __name__ == "__main__":
    fire.Fire(create_inference)
