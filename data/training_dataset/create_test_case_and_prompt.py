from typing import List, Optional

from easy_openai import openai_completions

PROMPT_TEMPLATE_RAW = """You are the latest and best bot aimed at transforming some code snippet into a leetcode style question. You will be provided with a prompt for writing code, along with a reference program that answers the question. Please complete the following for me:
1. Come up with a leetcode style question which consists of a well-defined problem. The generated question should meet the following criteria:
    a. The question is clear and understandable, with enough details to describe what the input and output are.
    b. The question should be solvable by only implementing 1 function instead of multiple functions or a class. Therefore, please avoid questions which require complicated pipelines.
    c. The question itself should not require any access to external resource or database.
    d. Feel free to use part of the original question if necessary. Moreover, please do not ask for runtime and space complexity analysis or any test cases in your response. 
2. Based on the modified question that you generated in part 1, you need to create around 20 test cases for this modified question. Each test case should be independent assert clauses. The parameters and expected output of each test case should all be constants, **without accessing any external resources**.

Here is the original question:
{instruction}

Here is the reference program that answers the question:
```python
{program}
```

Now give your modified question and generated test cases in the following json format: 
{"question": ..., "tests":["assert ...", "assert ..."]}.
"""

PROMPT_TEMPLATE_NO_INSTRUCTION = """You are the latest and best bot aimed at transforming some code snippet into a leetcode style question. You will be provided with a reference program. Please complete the following for me:
1. Come up with a leetcode style question which consists of a well-defined problem. The generated question should meet the following criteria:
    a. The question is clear and understandable, with enough details to describe what the input and output are.
    b. The question should be solvable by only implementing 1 function instead of multiple functions or a class. Therefore, please avoid questions which require complicated pipelines.
    c. The question itself should not require any access to external resource or database.
    d. Feel free to use part of the original question if necessary. Moreover, please do not ask for runtime and space complexity analysis or any test cases in your response. 
2. Based on the modified question that you generated in part 1, you need to create around 20 test cases for this modified question. Each test case should be independent assert clauses. The parameters and expected output of each test case should all be constants, **without accessing any external resources**.

Here is the reference program:
```python
{program}
```

Now give your modified question and generated test cases in the following json format: 
{"question": ..., "tests":["assert ...", "assert ..."]}.
"""


def create_test_cases_using_gpt(
    programs: List[str],
    instructions: Optional[List[str]] = None,
    use_cache: bool = True,
    return_price: bool = False,
) -> None:
    """Use this program to create tests cases for raw (badly formatted) questions and source code. Ex: Evol"""
    if instructions:
        chatmls = [
            [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people with python coding tasks.",
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE_RAW.replace("{program}", prompt).replace(
                        "{instruction}", instruction
                    ),
                },
            ]
            for prompt, instruction in zip(programs, instructions)
        ]
    else:
        chatmls = [
            [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people with python coding tasks.",
                },
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE_NO_INSTRUCTION.replace(
                        "{program}", prompt
                    ),
                },
            ]
            for prompt in programs
        ]
    output = openai_completions(
        chatmls,
        model_name="gpt-4o-mini",
        use_cache=use_cache,
        return_json=True,
        temperature=0.7,
        max_tokens=8192,
    )
    if return_price:
        total_price = sum(output["price_per_example"])
        return output["completions"], total_price
    else:
        return output["completions"]
