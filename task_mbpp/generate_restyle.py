import os
import json
import asyncio
import random

from tqdm.asyncio import tqdm_asyncio

from common.clients import async_together, format_model_name_together, paraphrase_semaphore
from common.prompts import CODE_PARAPHRASER_PROMPT_TEMPLATE

random.seed(23)

# Single shared client
_client = async_together()

async def paraphrase_code(code: str, paraphraser: str = 'Qwen3-Coder-480B-A35B-Instruct-FP8'):
    async with paraphrase_semaphore:
        prompt = CODE_PARAPHRASER_PROMPT_TEMPLATE.format(code=code)
        exact_model = format_model_name_together(paraphraser)
        try:
            response = await _client.chat.completions.create(
                model=exact_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            api_response = response.choices[0].message.content
            data = json.loads(api_response)
            return data.get("new_solution")
        except Exception as e:
            print(f"Failed to paraphrase: {e}")
            return None


async def process_incorrect_records(records):
    async def handle(record):
        if (record.get('assistent_1_is_correct') is False and
            record.get('assistent_2_is_correct') is True and
            not record.get('assistent_1_answer_restyle_qwen3')):
            print('processing record', record.get('unique_id'))
            original = record.get('assistent_1_answer')
            paraphrased = await paraphrase_code(original)
            if paraphrased:
                record['assistent_1_answer_restyle_qwen3'] = paraphrased
            else:
                print(f"Failed to paraphrase record {record.get('unique_id')}")
        return record

    tasks = [asyncio.create_task(handle(r)) for r in records]
    done = []
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Restyling"):
        res = await fut
        done.append(res)
    return done

# Example usage (kept as comment only):
# with open('data/llama-3.1-70b_gpt-3.5-turbo_eval.jsonl') as f:
#     records = [json.loads(line) for line in f]
# records = await process_incorrect_records(records)
