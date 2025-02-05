import json
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from inference.Utility import append_inference, get_saved_inference_index
from utility.utility import MyTimer, chunking


def vllm_inference(
    huggingface_model_path: str,
    dataset_name: str,
    local_model_name: str,
    prompts: List[str] = None,
    input_token_ids: List = None,
    chunk_size: int = 200,
    sampling_params: Dict = None,
    trust_remote_code: bool = False,
    seperate_tokenizer_path: Optional[str] = None,
    additional_engine_param: Optional[Dict] = None,
) -> None:
    """Make inference on provided prompts using vllm, will use chunking and save inferenced answer locally.

    | Parameter:
        | huggingface_model_path: the model path copied from Hugging Face, ex: "deepseek-ai/deepseek-math-7b-base"
        | dataset_name: a string representing the dataset, this string will be part of a filename
        | prompts: a list of string, each is the prompt
        | local_model_name: a string indicating the model, this string will be used as a filename
        | input_token_ids: alternatively, you can pass in a List of input_token_ids that have already
            been tokenized by a tokenizer.
        | post_process_func: a function that will be used to process the output answer
        | chunk_size: the program will store every chunk_size inferenced prompts, the default 200 works well
            with one shot model inference for 7B models. You should decrease this number if you are inferencing
            on larger models / doing best of n inferences.
        | sampling_params: the sampling parameter passed for inferencing model, if given none then we will use
            {"max_tokens"=1024}
        | trust_remote_code: the same as the one defined in huggingface transformers
        | seperate_tokenizer_path: if you have a path that links to a tokenizer that you would like to use
        | additional_engine_param: additional parameter for the vLLM engine
    """
    starting_idx = get_saved_inference_index(dataset_name, local_model_name)
    if prompts:
        input_length = len(prompts)
    elif input_token_ids:
        input_length = len(input_token_ids)
    else:
        raise Exception("Both prompts and input token ids are None")
    if starting_idx >= input_length:
        print(
            f"Inference for {dataset_name} using {local_model_name} is already complete"
        )
        return
    print(
        f"starting vLLM inferencing for {dataset_name} using {local_model_name} with starting index {starting_idx}"
    )
    timer = MyTimer()
    if prompts:
        prompts = prompts[starting_idx:]
    if input_token_ids:
        input_token_ids = input_token_ids[starting_idx:]
    if sampling_params is None:
        sampling_params = {"max_tokens": 1024}
    elif "max_new_tokens" in sampling_params:
        tmp_val = sampling_params.pop("max_new_tokens")
        sampling_params["max_tokens"] = tmp_val
    elif "max_tokens" not in sampling_params:
        # default max new tokens
        sampling_params["max_tokens"] = 1024

    sampling_params = SamplingParams(**sampling_params)

    if not additional_engine_param:
        llm = LLM(
            model=huggingface_model_path,
            tokenizer=seperate_tokenizer_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=trust_remote_code,
            swap_space=24,
        )
    else:
        llm = LLM(
            model=huggingface_model_path,
            tokenizer=seperate_tokenizer_path,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=trust_remote_code,
            swap_space=24,
            **additional_engine_param,
        )
    timer.print_runtime(f"  getting model - {huggingface_model_path}")

    if prompts:
        text_prompt = True
        inputs = chunking(prompts, chunk_size)
    elif input_token_ids:
        text_prompt = False
        inputs = chunking(input_token_ids, chunk_size)

    for i, prompts in enumerate(tqdm(inputs, leave=True)):
        if text_prompt:
            outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
        else:
            outputs = llm.generate(
                prompt_token_ids=prompts, sampling_params=sampling_params
            )
        answers = []
        for idx, output in enumerate(outputs):
            for answer in output.outputs:
                answers.append(
                    json.dumps(
                        {
                            "id": i * chunk_size + idx + starting_idx,
                            "response": answer.text,
                        }
                    )
                )
        append_inference(dataset_name, local_model_name, answers)
