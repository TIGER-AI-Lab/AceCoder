import fire
import json5
import json
import datasets
import regex as re
from pathlib import Path

def clean_text_for_utf8(text):
    """
    Clean text by removing or replacing invalid Unicode characters for UTF-8 encoding.
    
    Args:
        text (str): Input text that may contain invalid Unicode characters
        
    Returns:
        str: Cleaned text that can be safely encoded in UTF-8
    """
    try:
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        pass
    # Method 1: Replace surrogate pairs with replacement character
    cleaned = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'replace')
    print(f"Cleaning text with invalid Unicode characters, cleaned: {cleaned}")
    
    # Method 2: Remove surrogate pairs entirely
    # cleaned = ''.join(char for char in text if not 0xD800 <= ord(char) <= 0xDFFF)
    
    # Method 3: Encode with error handling
    # cleaned = text.encode('utf-8', 'ignore').decode('utf-8')
    
    return cleaned


def filter_gpt_response(item):
    if '"question"' in item['gpt_response'] and '"tests"' in item['gpt_response']:
        return True
    return False

def load_single_line_str(string):
    """
    The string may be invalid like "A"B"C", I and want to change it into ABC
    """
    string = string.strip('\r')
    try:
        return json.loads(f'"{string}"')
    except:
        pass
    
    try:
        return json.loads('"' + string)
    except:
        pass
    
    try:
        return json.loads(string + '"')
    except:
        pass
    
    try:
        result = json.loads(string)
        assert isinstance(result, str)
    except:
        pass
    
    final_string = ""
    in_str = False
    last_idx = 0
    i = 0
    while i < len(string):
        if string[i] == '"':
            in_str = not in_str
            final_string += string[last_idx:i]
            last_idx = i + 1
            
        i += 1
    final_string += string[last_idx:]
    try:
        return json.loads(f'"{final_string}"')
    except Exception as e:
        pass
        # if "Invalid \\escape" not in str(e) or "Unterminated string starting" not in str(e):
        #     print(str(e)) 
    
    try:
        final_string = re.sub(r'(?<!\\)\\(?=([^nrt\\]|$))', r"\\\\", final_string)
        return json.loads(f'"{final_string}"')
    except:
        # print(string)
        # print('-----')
        # print(final_string)
        raise
    
def load_str(string):
    last_idx = 0
    final_string = ""
    while True:
        next_newline = string.find("\n", last_idx)
        if next_newline == -1:
            final_string += load_single_line_str(string[last_idx:])
            break
        final_string += load_single_line_str(string[last_idx:next_newline]) + '\n'
        last_idx = next_newline + 1
    return final_string

def parse_gpt_response(item):
    response = item["gpt_response"]
    left, right = response.find("```"), response.rfind("```")
    response = response[left:right]
    response = response.strip('` \n')
    if response.startswith("json"):
        response = response[4:]
    error = None
    try:
        raw_tests = None
        json_data = json5.loads(response)
        new_question = json_data['question']
        tests = json_data['tests']
    except Exception as e:
        error = ""
        new_question = re.search(r'(?<="question": )"(.|\n)+?"(?=(\n|(,\n)))', response)
        new_question = new_question.group() if new_question else None
        try:
            new_question = load_str(new_question)
        except Exception as e:
            error += "\n" + str(e)
            new_question = None
        raw_tests = re.search(r'"tests": (\[(.|\n)+?\])\n', response)
        raw_tests = raw_tests.group(1) if raw_tests else None
        if raw_tests:
            try:
                # remove comments for json object
                raw_tests = re.sub(r'//[\w ]*\n', "\n", raw_tests)
                tests = json.loads(raw_tests)
                assert isinstance(tests, list)
            except ValueError:
                try:
                    # load test as list of string and merge into a single test case
                    test = ""
                    for line in raw_tests.split("\n"):
                        if not line.strip('[] '):
                            continue
                        start = line.find('"')
                        end = line.rfind('"')
                        assert start != -1 and end != -1
                        line = line[start+1:end]
                        test += line + "\n"
                    tests = [test]
                except Exception as e:
                    error += "\n" + str(e)
                    tests = None
        else:
            tests = None
    if new_question and not (isinstance(new_question, str) and new_question.strip()):
        new_question = None
        error = (error or "") + "\n" + "new_question is not a string or empty"
    assert new_question is None or isinstance(new_question, str), f"new_question: {new_question}"
    assert raw_tests is None or isinstance(raw_tests, str), f"raw_tests: {raw_tests}"
    
    if tests:
        if not (isinstance(tests, list) and all(isinstance(test, str) for test in tests)):
            tests = None
            error = (error or "") + "\n" + "tests is not a list of string"
    assert tests is None or isinstance(tests, list), f"tests: {tests}"
    assert all(isinstance(test, str) for test in tests) if tests else True, f"tests: {tests}"
    assert error is None or isinstance(error, str), f"error: {error}"
    item['ori_question'] = item['instruction']
    item['ori_program'] = item['program']
    
    try:
        new_question.encode('utf-8') if new_question else None
    except Exception as e:
        error = (error or "") + "\n" + str(e)
        new_question = None
    
    try:
        [x.encode('utf-8') for x in tests] if tests else None
    except Exception as e:
        error = (error or "") + "\n" + str(e)
        tests = None
        raw_tests = None
    
    item['question'] = new_question
    item['tests'] = tests
    item['raw_tests'] = raw_tests
    # item['question'] = clean_text_for_utf8(new_question) if new_question else None
    # item['raw_tests'] = clean_text_for_utf8(raw_tests) if raw_tests else None
    # item['tests'] = [clean_text_for_utf8(x) for x in tests] if tests else None
    item['error'] = error
    return item
    
def main(
    dataset_path="CodeDPO/AceCoderV2-mini-raw",
    output_dir="data/acecoder_v2_mini",
    to_upload_repo="CodeDPO/AceCoderV2-mini-processed",
    num_proc=32
):
    dataset = datasets.load_dataset(dataset_path, split='train')
    output_file = Path(output_dir) / "dataset.json"
    
    # dataset = dataset.select(range(1000))
    
    print(dataset)
    print(f"Start filtering items where gpt response does not contain question or tests")
    dataset = dataset.filter(filter_gpt_response)
    print(f"Filtered dataset: {dataset}")
    
    print(f"Start parsing gpt response")
    dataset = dataset.map(parse_gpt_response, num_proc=num_proc, remove_columns=["instruction", "program"])
    
    print(f"Num errors: {len([item for item in dataset if item['error']])}/{len(dataset)}")
    print(f"Num valid questions: {len([item for item in dataset if item['question']])}/{len(dataset)}")
    print(f"Num valid tests: {len([item for item in dataset if item['tests']])}/{len(dataset)}")
    
    def filter_valid(item):
        return item['question'] and item['tests'] and not item['error']
    dataset = dataset.filter(filter_valid)
    print(f"Final dataset: {dataset}")
    dataset = dataset.remove_columns(["gpt_response", "error", "raw_tests"])
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump([x for x in dataset], f, indent=4)
    print(f"Saved to {output_file}")
    
    if to_upload_repo:
        dataset.push_to_hub(to_upload_repo, split='train')
        print(f"Pushed to {to_upload_repo}")

if __name__ == "__main__":
    fire.Fire(main)
    
    
"""
python scripts/process_v2_data.py --dataset_path CodeDPO/AceCoderV2-mini --output_dir data/acecoder_v2_mini --to_upload_repo CodeDPO/AceCoderV2-mini-processed
python scripts/process_v2_data.py --dataset_path CodeDPO/AceCoderV2-150K-raw --output_dir data/acecoder_v2_150k --to_upload_repo CodeDPO/AceCoderV2-150K-processed
"""