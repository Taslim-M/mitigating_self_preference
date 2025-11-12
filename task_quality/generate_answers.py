import json
import re
from tqdm import tqdm
import time
from math import exp
from dotenv import load_dotenv
load_dotenv()

from together import Together
import os
import asyncio
from tqdm import tqdm
from together import AsyncTogether
from openai import OpenAI


import nest_asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


together_client = Together()


def format_model_name_together(model_name):
    if model_name.startswith("Meta-Llama"):
        return f"meta-llama/{model_name}"
    elif model_name.startswith("Qwen"):
        return f"Qwen/{model_name}"
    elif model_name.startswith("DeepSeek"):
        return f"deepseek-ai/{model_name}"
    elif model_name.startswith("Llama"):
        return f"meta-llama/{model_name}"
    elif model_name.startswith("Mistral"):
        return f"mistralai/{model_name}"
    elif model_name.startswith("Mixtral"):
        return f"mistralai/{model_name}"
    elif model_name.startswith("Kimi"):
        return f"moonshotai/{model_name}"
    elif model_name.startswith("gpt"):
        return f"openai/{model_name}"
    elif model_name.startswith("GLM"):
        return f"zai-org/{model_name}"
    else:
        return model_name  # Return as is if no specific match is found



def fix_json_response(response: str) -> dict:
    """
    Fixes common JSON formatting issues in a string response.
    
    Args:
        response (str): The response string from ChatGPT.
        
    Returns:
        dict: The JSON-compatible dictionary.
    """
    # Attempt to parse the JSON without any modifications
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass  # If it fails, continue with the processing steps
    
    # Remove markdown JSON code fences and the `json` keyword
    response = re.sub(r'```json\n|```|json', '', response)
    
    # Replace non-standard quotes with standard double quotes
    response = response.replace('“', '"').replace('”', '"')
    
    # Replace invalid fractions with their approximate decimal equivalents
    response = re.sub(r'(\d+)/(\d+)', lambda m: str(float(m.group(1)) / float(m.group(2))), response)
    
    # Strip leading and trailing whitespace
    response = response.strip()
    
    # Attempt to find JSON object or array within the string
    match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', response)
    
    if match:
        cleaned_string = match.group(0)
    else:
        # If no JSON object or array is found, assume the whole response needs fixing
        cleaned_string = response
    
    # Count the number of opening and closing braces
    open_curly = cleaned_string.count('{')
    close_curly = cleaned_string.count('}')
    open_square = cleaned_string.count('[')
    close_square = cleaned_string.count(']')
    
    # Attempt to add enclosing brackets if missing
    if open_curly == 1 and close_curly == 0:
        cleaned_string += '}'
    elif close_curly == 1 and open_curly == 0:
        cleaned_string = '{' + cleaned_string
    elif open_square == 1 and close_square == 0:
        cleaned_string += ']'
    elif close_square == 1 and open_square == 0:
        cleaned_string = '[' + cleaned_string

    # Handle case where both opening and closing brackets are missing
    if open_curly == 0 and close_curly == 0 and open_square == 0 and close_square == 0:
        cleaned_string = '{' + cleaned_string + '}'
    
    # Attempt to fix common issues and parse the JSON
    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError:
        # Handle common issues
        cleaned_string = cleaned_string.replace("'", '"')  # Replace single quotes with double quotes
        cleaned_string = cleaned_string.replace("\n", " ")  # Remove newlines
        cleaned_string = cleaned_string.replace("\t", " ")  # Remove tabs

        try:
            return json.loads(cleaned_string)
        except json.JSONDecodeError:
            try:
                wrapped_string = f"[{cleaned_string}]"
                return json.loads(wrapped_string)
            except json.JSONDecodeError:
                raise ValueError("Unable to fix JSON response")
            
            
def extract_output_label(question_text, answer):
    """
    Extracts the correct multiple-choice label (A, B, C, D) based on the given answer.
    """
    pattern = r"\((A|B|C|D)\)\s(.+)"
    matches = re.findall(pattern, question_text)

    for label, option in matches:
        if option.strip() == answer.strip():
            return label
    return None  # Return None if no match is found

def process_jsonl_with_labels(input_file):
    processed_records = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            record = json.loads(line.strip())

            if 'input' in record:
                parts = record['input'].split('\n\n\n', 1)
                record['questions'] = parts[0]
                record['text'] = parts[1] if len(parts) > 1 else ""
                del record['input']  # Remove original input field if needed
            
            if 'questions' in record and 'output' in record:
                record['output_label'] = extract_output_label(record['questions'], record['output'])

            processed_records.append(record)

    return processed_records

def extract_braces_content(s):
    match = re.search(r'\{(.*?)\}', s, re.DOTALL)
    return match.group(0) if match else ""

# Prompt
ANSWER_SELECTION_PROMPT_TEMPLATE = """You are an AI system designed to analyze questions and their corresponding text passages to determine the correct multiple-choice answer.

### System Output Format:
Respond in **JSON format** with:
- `"output_label"`: The correct answer label (e.g., "A", "B", "C", "D").
- `"reason"`: A short explanation of why this answer is correct.

### Task:
Given the **question** and the **text** below, pick the correct answer by identifying the most relevant information in the text. Your response must only include the output label and a brief justification.

#### Question:
{questions}

#### Text:
{text}

### Expected Response Format:
```
{{
  "output_label": "A",
  "reason": "The text states that the event lasted 10 hours, which corresponds to option B."
}}"""


#
nest_asyncio.apply()

async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# Limit number of concurrent API calls
semaphore = asyncio.Semaphore(20)  # <-- Set to 10, 20, or 50 depending on your system and Together API limits

failed_records_id = []
failed_records = []

async def process_record(record, model_name):
    async with semaphore:  # <--- Throttle concurrent calls
        questions = record.get("questions", "")
        text = record.get("text", "")
        
        if not questions or not text:
            return None  # Skip invalid records

        prompt = ANSWER_SELECTION_PROMPT_TEMPLATE.format(questions=questions, text=text)
        exact_model = format_model_name_together(model_name)

        try:
            response = await async_client.chat.completions.create(
                model=exact_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            api_response = response.choices[0].message.content
            api_response = extract_braces_content(api_response)

            key_name_output = model_name + "_output_label"
            key_name_reason = model_name + "_reason"

            response_json = fix_json_response(api_response)
            record[key_name_output] = response_json.get("output_label")
            record[key_name_reason] = response_json.get("reason")

            return record

        except Exception as e:
            failed_records.append(str(e))
            failed_records_id.append(record.get("id"))
            return None

async def generate_answer_selection_quality_async(model_name, start_index, end_index, processed_data):
    tasks = [
        process_record(record, model_name) for record in processed_data[start_index:end_index]
    ]
    results = []
    for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing Records"):
        result = await future
        if result:
            results.append(result)
    return results, failed_records_id, failed_records

# sample usage

# async def main():
# results, failed_ids, failed_content = asyncio.run(generate_answer_selection_quality_async("Llama-4-Scout-17B-16E-Instruct", 0, len(responses), responses))

# if __name__ == "__main__":
#     asyncio.run(main())