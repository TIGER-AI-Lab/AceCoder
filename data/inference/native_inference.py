import json
from typing import Callable, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from inference.Utility import append_inference, get_saved_inference_index
from utility.utility import MyTimer


def native_inference(
    huggingface_model_path: str,
    dataset_name: str,
    prompts: List[str],
    local_model_name: str,
    chunk_size: int = 20,
    sampling_params: Dict = None,
    generation_config: GenerationConfig = None,
    trust_remote_code: bool = False,
    seperate_tokenizer_path: Optional[str] = None,
) -> None:
    """Use the native huggingface generate function to make inferences. Mostly exist because vLLM does not support
    diverse beam search.

    | Parameter:
        | huggingface_model_path: the model path copied from Hugging Face, ex: "deepseek-ai/deepseek-math-7b-base"
        | dataset_name: a string representing the dataset, this string will be part of a filename
        | prompts: a list of string, each is the prompt
        | local_model_name: a string indicating the model, this string will be used as a filename
        | post_process_func: a function that will be used to process the output answer
        | chunk_size: the program will store every chunk_size inferenced prompts, the default 200 works well
            with one shot model inference for 7B models. You should decrease this number if you are inferencing
            on larger models / doing best of n inferences.
        | sampling_params: the sampling parameter passed for inferencing model, if given none then we will use
            {"max_tokens"=1024}
        | generation_config: a generation config object
        | trust_remote_code: the same as the one defined in huggingface transformers
    """
    starting_idx = get_saved_inference_index(dataset_name, local_model_name)
    if starting_idx >= len(prompts):
        print(
            f"Inference for {dataset_name} using {local_model_name} is already complete"
        )
        return
    print(
        f"starting inferencing for {dataset_name} using {local_model_name} with starting index {starting_idx}"
    )
    timer = MyTimer()
    device = torch.device("cuda")
    if sampling_params is None:
        sampling_params = {"max_new_tokens": 1024}
    else:
        if "max_tokens" in sampling_params:
            tmp_val = sampling_params.pop("max_tokens")
            sampling_params["max_new_tokens"] = tmp_val
        elif "max_new_tokens" not in sampling_params:
            # default max new tokens
            sampling_params["max_new_tokens"] = 1024

        if "n" in sampling_params:
            # we need to replace the name for best_of_n sampling
            tmp_val = sampling_params.pop("n")
            sampling_params["num_beams"] = tmp_val
            sampling_params["num_return_sequences"] = tmp_val

    prompts = prompts[starting_idx:]
    model = AutoModelForCausalLM.from_pretrained(
        huggingface_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if generation_config is not None:
        model.generation_config = generation_config

    if seperate_tokenizer_path is None:
        seperate_tokenizer_path = huggingface_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        seperate_tokenizer_path, trust_remote_code=trust_remote_code
    )

    timer.print_runtime("  loading models and datasets")

    output_lst = []
    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts)):
            encoding = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**encoding.to(device), **sampling_params)

            response_strs = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            response_strs = [
                i[(len(prompt)) :] if i.startswith(prompt) else i for i in response_strs
            ]
            output_lst += [
                json.dumps({"id": i + starting_idx, "response": strr})
                for strr in response_strs
            ]

            timer.print_runtime(f" inferenced {i} entries", reset_timer=False)
            if (i + 1) % chunk_size == 0:
                append_inference(dataset_name, local_model_name, output_lst)
                output_lst = []
    if len(output_lst) > 0:
        append_inference(dataset_name, local_model_name, output_lst)
