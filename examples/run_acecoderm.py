"""pip install git+https://github.com/TIGER-AI-Lab/AceCoder"""
from acecoder import Qwen2ForCausalRM
from transformers import AutoTokenizer

model_path = "TIGER-Lab/AceCodeRM-7B"
model = Qwen2ForCausalRM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

question = """\
Given an array of numbers, write a function runningSum that returns an array where each element at index i is the sum of all elements from index 0 to i (inclusive).
For example:
Input: nums = [1,2,3,4]
Output: [1,3,6,10]
"""

program_with_3_errors = """\
def runningSum(nums):
    result = []
    current_sum = 0
    for i in range(1, len(nums)):
        result.append(nums[i])
        current_sum += nums[i]
    return result
"""

program_with_2_errors = """\
def runningSum(nums):
    result = []
    current_sum = 0
    for i in range(0, len(nums)):
        result.append(nums[i])
        current_sum += nums[i]
    return result
"""
    
program_with_1_errors = """\
def runningSum(nums):
    result = []
    current_sum = 0
    for i in range(0, len(nums)):
        result.append(current_sum)
        current_sum += nums[i]
    return result
"""
program_correct = """\
def runningSum(nums):
    result = []
    current_sum = 0
    for num in nums:
        current_sum += num
        result.append(current_sum)
    return result
"""

program_chats = [
    [
        {
            "content": question,
            "role": "user",
        },
        {
            "role": "assistant",
            "content": program
        }
    ] for program in [program_with_3_errors, program_with_2_errors, program_with_1_errors, program_correct]
]

input_tokens = tokenizer.apply_chat_template(
    program_chats,
    tokenize=True,
    return_dict=True,
    padding=True,
    return_tensors="pt",
).to(model.device)

_, _, values = model(
    **input_tokens,
    output_hidden_states=True,
    return_dict=True,
    use_cache=False,    
)
masks = input_tokens["attention_mask"]
rm_scores = values.gather(
    dim=-1, index=(masks.sum(dim=-1, keepdim=True) - 1)
) # find the last token (eos) in each sequence, a
rm_scores = rm_scores.squeeze()

print("RM Scores:", rm_scores)
print("Score of program with 3 errors:", rm_scores[0].item())
print("Score of program with 2 errors:", rm_scores[1].item())
print("Score of program with 1 errors:", rm_scores[2].item())
print("Score of correct program:", rm_scores[3].item())
"""
RM Scores: tensor([-20.5058,  -1.7867,   0.4395,  23.0689], device='cuda:0',
       grad_fn=<SqueezeBackward0>)
Score of program with 3 errors: -20.505754470825195
Score of program with 2 errors: -1.7866804599761963
Score of program with 1 errors: 0.43949759006500244
Score of correct program: 23.068859100341797
"""