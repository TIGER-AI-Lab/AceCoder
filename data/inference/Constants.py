# Code models related constants
MBPP = "mbpp"
MBPP_LENGTH = 500

MBPPPLUS = "mbppplus"
MBPPPLUS_LENGTH = 378

HUMANEVAL = "humaneval"
HUMANEVAL_LENGTH = 164

HUMANEVALPLUS = "humanevalplus"
HUMANEVALPLUS_LENGTH = 164

LIVECODEBENCH = "livecodebench"
LIVECODEBENCH_LENGTH = 714

STABLE_CODE = "stable_code"
STARCODER2 = "starcoder2"
CODELLAMA = "codellama"
DEEPSEEK = "deepseek"
LEETCODE = "leetcode"
LLAMA = "llama"
CODEQWEN = "code_qwen"
WIZARDCODER = "wizardcoder"
MISTRAL = "mistral"
QWEN_CODER = "qwen_coder"
NXCODE = "nxcode"

# this is the generic sampling parameters that will be passed for different type of inferences
GENERIC_SAMPLING_PARAMETERS = {
    "greedy": {
        "temperature": 0,
    },
    "best_of_n_diverse_beam_search": {
        "num_beams": 16,
        "num_return_sequences": 8,
        "num_beam_groups": 8,
        "early_stopping": True,
        "diversity_penalty": 1.0,
    },
    "best_of_n_beam_search": {
        "n": 16,
        "temperature": 0,
        "use_beam_search": True,
        "early_stopping": True,
    },
    "best_of_n_top_p_sampling": {
        "n": 16,
        "top_p": 1.0,
        "temperature": 0.7,
    },
    "best_of_n_top_k_sampling": {
        "n": 16,
        "top_k": 50,
        "temperature": 0.7,
    },
}

MODEL_PATH = {
    "athena_v2": {"72b": "Nexusflow/Athene-V2-Chat"},
    f"{DEEPSEEK}_coder": {
        "1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "7b": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "33b": "deepseek-ai/deepseek-coder-33b-instruct",
    },
    f"{DEEPSEEK}_coder_v2": {"16b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"},
    f"{CODEQWEN}_v1.5": {
        "7b": "Qwen/CodeQwen1.5-7B-Chat",
    },
    f"{QWEN_CODER}_2.5": {
        "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    },
    f"{NXCODE}_cq_orpo": {
        "7b": "NTQAI/Nxcode-CQ-7B-orpo",
    },
    STARCODER2: {
        "3b": "bigcode/starcoder2-3b",
        "7b": "bigcode/starcoder2-7b",
        "15b": "bigcode/starcoder2-15b",
    },
    f"{MISTRAL}_instruct_v3": {
        "7b": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    f"{CODELLAMA}": {
        "7b": "codellama/CodeLlama-7b-hf",
        "13b": "codellama/CodeLlama-13b-hf",
        "34b": "codellama/CodeLlama-34b-hf",
        "70b": "codellama/CodeLlama-70b-hf",
    },
    f"{CODELLAMA}_python": {
        "7b": "codellama/CodeLlama-7b-Python-hf",
        "13b": "codellama/CodeLlama-13b-Python-hf",
        "34b": "codellama/CodeLlama-34b-Python-hf",
        "70b": "codellama/CodeLlama-70b-Python-hf",
    },
    f"{CODELLAMA}_instruct": {
        "7b": "codellama/CodeLlama-7b-Instruct-hf",
        "13b": "codellama/CodeLlama-13b-Instruct-hf",
        "34b": "codellama/CodeLlama-34b-Instruct-hf",
        "70b": "codellama/CodeLlama-70b-Instruct-hf",
    },
    f"{LLAMA}3_instruct": {
        "8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    },
    f"{LLAMA}3.1_instruct": {
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "meta-llama/Llama-3.1-70B-Instruct",
    },
}
