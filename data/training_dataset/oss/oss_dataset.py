from inference.GetDatasets import get_dataset_from_huggingface

MODEL_LST = {
    # "deepseek_coder": "deepseek_coder_6.7b_best_of_n_top_p_sampling",
    # "codellama_instruct": "codellama_instruct_7b_best_of_n_top_p_sampling",
    # "code_qwen_v1.5": "code_qwen_v1.5_7b_best_of_n_top_p_sampling",
    "llama3_instruct": "llama3_instruct_8b_best_of_n_top_p_sampling",
    "mistral_instruct_v3": "mistral_instruct_v3_7b_best_of_n_top_p_sampling",
}


# https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K?row=0
def get_oss_dataset():
    data = get_dataset_from_huggingface(
        "ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response"
    )["train"]
    return data
