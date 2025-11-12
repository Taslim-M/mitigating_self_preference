import re
import asyncio
import json

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

import io
import tokenize

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

import random
random.seed(23)

import sys
import argparse
from pathlib import Path
from typing import List

# Initialize Together client
together_client = Together()

#UTILS

def format_model_name_together(model_name):
    if model_name.startswith("Meta-Llama"):
        return f"meta-llama/{model_name}"
    elif model_name.startswith("Qwen"):
        return f"Qwen/{model_name}"
    elif model_name.startswith("DeepSeek"):
        return f"deepseek-ai/{model_name}"
    elif model_name.startswith("Llama"):
        return f"meta-llama/{model_name}"
    else:
        return model_name  # Return as is if no specific match is found
    
# Prompt
CODE_PARAPHRASER_PROMPT_TEMPLATE = """You are a helpful assistant that helps rewrites coding solutions in python. 
Given the coding solution, rewrite it in a style that changes the structural or idiomatic style of the code. The logic of the code should remain the same. 
You are only allowed to change the code, not the comment. The resulting code should generate the same output as the original code.

### System Output Format:
Respond in **JSON format** with:
- `"new_solution"`: The re-written coding solution.

### code:
{code}

### Expected Response Format:
```
{{
  "new_solution": "the re-written coding solution",
}}
"""




async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# Limit number of concurrent API calls
semaphore = asyncio.Semaphore(20)  

async def paraphrase_code(code, paraphraser='Qwen3-Coder-480B-A35B-Instruct-FP8'):
    async with semaphore:  # Throttle concurrent calls
        prompt = CODE_PARAPHRASER_PROMPT_TEMPLATE.format(
            code=code
        )
        # use the paraphraser model to generate the paraphrased answer
        exact_model = format_model_name_together(paraphraser)
        
        try:
            response = await async_client.chat.completions.create(
                model=exact_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
                
            )
            api_response = response.choices[0].message.content
            # print(api_response)
            response_json = json.loads(api_response)
            
            return response_json.get("new_solution") 
        
        except Exception as e:
            print(f"Failed to paraphrase: {e}")
            return None


async def process_incorrect_records(records):
    tasks = []

    async def process_single_record(record):
        if (record.get('assistent_1_is_correct') == False and 
            record.get('assistent_2_is_correct') == True and
            not record.get('assistent_1_answer_restyle_qwen3')):

            print('processing record', record.get('unique_id'))
            original_code = record.get('assistent_1_answer')
            paraphrased = await paraphrase_code(original_code)

            if paraphrased:
                record['assistent_1_answer_restyle_qwen3'] = paraphrased
            else:
                print(f"Failed to paraphrase record {record.get('unique_id')}")

        return record

    # Create a task for each record
    for record in records:
        task = asyncio.create_task(process_single_record(record))
        tasks.append(task)

    # Run all tasks concurrently (honoring semaphore in paraphrase_code)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed with: {result}")

    return results

# Example usage:
# with open('data/llama-3.1-70b_gpt-3.5-turbo_eval.jsonl') as f:
#     records = [json.loads(line) for line in f]
#     print(len(records))
# records = await process_incorrect_records(records)