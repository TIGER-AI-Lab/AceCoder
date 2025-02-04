from typing import Dict, List, Optional

import torch

from inference.Constants import GENERIC_SAMPLING_PARAMETERS
from inference.native_inference import native_inference
from inference.Utility import (get_huggingface_model_path,
                               get_suggested_inference_batch_size)
from inference.vllm_inference import vllm_inference


def inference(
    model_name: str,
    model_type: str,
    dataset_name: str,
    sampling_method: str,
    prompts: List[str] = None,
    input_token_ids: List = None,
    chunk_size: int = -1,
    inference_method: str = "default",
    additional_sampling_params: Dict = {},
    custom_model_path: Optional[str] = None,
    seperate_tokenizer_path: Optional[str] = None,
) -> None:
    """this function inference a model against GSM8K using the vllm.

    | Parameter:
        | model_name: the model name, ex: f"{MAMMOTH}_cot"
        | model_type: which specific model do you want to use, ex: "7b"
        | dataset_name: the name of the dataset, ex: f"{GSM8K}"
        | sampling_method: should be a sampling method available in GENERIC_SAMPLING_PARAMETERS
            in Constants.py. Currently support: {greedy, best_of_n_diverse_beam_search,
            best_of_n_beam_search, best_of_n_top_p_sampling, best_of_n_top_k_sampling}.
        | prompts: the prompts that will be inferenced on, note that we will apply template on top of this prompt
        | local_model_name: a string indicating the model, this string will be used as a filename
        | chunk_size: the program will store every chunk_size inferenced prompts, if received <= 0, then will use the
            recommended chunk size stored in ModelConfigs
        | vllm: whether to use vllm or not
        | inference_method: currently support "default", "native", "vllm" and "accelerate"
        | additional_sampling_params: sampling parameter you would like to pass in in addition to the pre-configured one
        | custom_model_path: the path to the model if you have fine-tuned it
        | seperate_tokenizer_path: if there is a different path for tokenizer
    """

    sampling_params = GENERIC_SAMPLING_PARAMETERS[sampling_method]
    sampling_params.update(additional_sampling_params)

    if custom_model_path is None:
        huggingface_model_path = get_huggingface_model_path(
            model_name=model_name, model_size=model_type
        )
        local_model_name = f"{model_name}_{model_type}_{sampling_method}"
        trust_remote_code = True

        if chunk_size <= 0:
            chunk_size = get_suggested_inference_batch_size(model_size=model_type)
    else:
        # we are using custom model
        huggingface_model_path = custom_model_path
        local_model_name = f"{model_name}_{sampling_method}"
        trust_remote_code = False

        chunk_size = max(1, chunk_size)  # cannot be less than 1

    if model_name == "deepseek_coder" and model_type == "6.7b":
        additional_vllm_param = {"max_model_len": 40000}
    else:
        additional_vllm_param = None

    if "best_of_n" in sampling_method:
        # we have to reduce the chunk size accordingly to prevent errors
        n = max(sampling_params.get("n", 0), sampling_params.get("num_beams", 0))
        chunk_size = max(1, chunk_size // n)

    num_gpus = torch.cuda.device_count()
    chunk_size = chunk_size * num_gpus

    if "diverse_beam_search" in sampling_method:
        # vLLM does not support diverse beam search
        assert inference_method in {"default", "native"}
        inference_method = "native"

    if inference_method in {"default", "vllm"}:
        vllm_inference(
            huggingface_model_path=huggingface_model_path,
            dataset_name=dataset_name,
            prompts=prompts,
            input_token_ids=input_token_ids,
            local_model_name=local_model_name,
            chunk_size=max(1, chunk_size),
            sampling_params=sampling_params,
            trust_remote_code=trust_remote_code,
            seperate_tokenizer_path=seperate_tokenizer_path,
            additional_engine_param=additional_vllm_param,
        )
    else:
        # have not write the support yet
        assert input_token_ids is None
        native_inference(
            huggingface_model_path=huggingface_model_path,
            dataset_name=dataset_name,
            prompts=prompts,
            local_model_name=local_model_name,
            chunk_size=max(
                1, chunk_size
            ),  # native inference is slow so we reduce chunk size
            sampling_params=sampling_params,
            trust_remote_code=trust_remote_code,
            seperate_tokenizer_path=seperate_tokenizer_path,
        )
