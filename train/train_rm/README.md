## Installation

```bash
cd LLaMA-Factory
```


# Dataset
- AceCode-87K
    - `Id` (str): Unique identifier for each question
    - `source` (str): which dataset
    - `question` (str): the question
    - `test_cases` (List[str]): test cases for the question
    - `inferences` (List[dict]):
        - `model_name` (str): the model name
        - `completion_id` (int): the completion id
        - `completion` (str): the completion
        - `pass_rate` (float): the pass rate
        - `test_results` (List[int]): the test results
    - `context_messages` (List[List[dict]]): context messages for the question
        - `content` (str): the content of the message
        - `role` (str): the role of the message

(TODO)
- AceCode-Pair-300K:
    - `Id` (str): Unique identifier for each question

