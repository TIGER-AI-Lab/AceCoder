def eval_post_process(input_str: str) -> str:
    markdowns = ["python", "markdown\n```python", "markdown\n```", "markdown"]
    for markdown in markdowns:
        if input_str.startswith(markdown):
            input_str = input_str[len(markdown) :]
            break
    end_tokens = ["```", "import unittest", "from unittest", "if __name__", "assert"]
    indices = [input_str.find(i) for i in end_tokens]
    indices = [i for i in indices if i > 0]
    if len(indices) > 0:
        return input_str[: min(indices)]
    return input_str


def deepseek_coder_post_process(input_str: str) -> str:
    left_idx = input_str.find("```python")
    if left_idx < 0:
        return deepseek_coder_post_process_2(input_str)
    out = input_str[left_idx + 9 :]
    right_idx = out.find("```")
    if right_idx < 0:
        return deepseek_coder_post_process_2(input_str)
    out = out[:right_idx]
    out = out.strip()
    return out


def deepseek_coder_post_process_2(input_str: str) -> str:
    right_idx = input_str.find("[DONE]")
    if right_idx < 0:
        return deepseek_coder_post_process_3(input_str)
    out = input_str[:right_idx]
    out = out.strip()
    return out


def deepseek_coder_post_process_3(input_str: str) -> str:
    right_idx = input_str.find("[END]")
    if right_idx < 0:
        return input_str
    out = input_str[:right_idx]
    out = out.strip()
    return out


def codellama_post_process(input_str: str) -> str:
    return '\n    """ ' + input_str + "\n    return result"


def codellama_instruct_post_process(input_str: str) -> str:
    left_idx = input_str.find("[PYTHON]")
    if left_idx < 0:
        return input_str
    out = input_str[left_idx + 8 :]
    right_idx = out.find("[/PYTHON]")
    if right_idx < 0:
        return input_str
    out = out[:right_idx]
    out = out.strip()
    return out


def starcoder2_post_process(input_str: str) -> str:
    out = input_str[3:]
    left_idx = out.find('"""')
    if left_idx < 0:
        return input_str
    out = input_str[left_idx + 6 :]
    right_idx = out.find('"""')
    if right_idx < 0:
        return input_str
    out = out[:right_idx]
    out = out.strip()
    return out
