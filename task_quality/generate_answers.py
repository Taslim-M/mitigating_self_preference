import asyncio
from typing import List, Tuple

from tqdm.asyncio import tqdm_asyncio

from common.clients import async_together, format_model_name_together, paraphrase_semaphore as semaphore
from common.prompts import ANSWER_SELECTION_PROMPT_TEMPLATE
from common.json_utils import fix_json_response, extract_braces_content

_client = async_together()

failed_records_id: List[str] = []
failed_records: List[str] = []

def extract_output_label(question_text: str, answer: str):
    import re
    pattern = r"\((A|B|C|D)\)\s(.+)"
    matches = re.findall(pattern, question_text)
    for label, option in matches:
        if option.strip() == answer.strip():
            return label
    return None

def process_jsonl_with_labels(input_file: str):
    import json
    processed = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            record = json.loads(line.strip())
            if 'input' in record:
                parts = record['input'].split('\n\n\n', 1)
                record['questions'] = parts[0]
                record['text'] = parts[1] if len(parts) > 1 else ""
                del record['input']
            if 'questions' in record and 'output' in record:
                record['output_label'] = extract_output_label(record['questions'], record['output'])
            processed.append(record)
    return processed


async def process_record(record: dict, model_name: str):
    async with semaphore:
        questions = record.get("questions", "")
        text = record.get("text", "")
        if not questions or not text:
            return None
        prompt = ANSWER_SELECTION_PROMPT_TEMPLATE.format(questions=questions, text=text)
        exact = format_model_name_together(model_name)
        try:
            resp = await _client.chat.completions.create(
                model=exact,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            api = resp.choices[0].message.content
            api = extract_braces_content(api)

            key_label = model_name + "_output_label"
            key_reason = model_name + "_reason"

            data = fix_json_response(api)
            record[key_label] = data.get("output_label")
            record[key_reason] = data.get("reason")
            return record
        except Exception as e:
            failed_records.append(str(e))
            failed_records_id.append(record.get("id"))
            return None


async def generate_answer_selection_quality_async(model_name: str, start_index: int, end_index: int, processed_data: list) -> Tuple[list, list, list]:
    tasks = [process_record(r, model_name) for r in processed_data[start_index:end_index]]
    results = []
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing Records"):
        res = await fut
        if res:
            results.append(res)
    return results, failed_records_id, failed_records

# sample usage :
# results, failed_ids, failed_content = asyncio.run(
#     generate_answer_selection_quality_async("Llama-4-Scout-17B-16E-Instruct", 0, len(responses), responses)
# )
