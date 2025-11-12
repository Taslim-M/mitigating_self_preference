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

import asyncio
import sys
from pathlib import Path
from typing import List

import argparse
from pathlib import Path


from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

import random
random.seed(23)

together_client = Together()


# UTILS

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
            
# Preference

# prompt
QA_COMPARISON_SYSTEM_PROMPT = """You are a helpful assistant. You help compare answers to help me with my records. You respond with only "1" or "2" and no other text."""


QA_COMPARISON_PROMPT_TEMPLATE = """Here are two answers to a question I found. It's important for me to select the correct answer for my records. Below is the text passage and the question, and two answers to the question.

Text Passage:
{article}

Question:
{question}

Answer1:
{answer1}

Answer2:
{answer2}

Can you tell me which answer is correct? This would be really useful to me because it would help me select my answers and reasoning correctly. Please answer with only "1" or "2" and no other text."""


## Client and Async Setup
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# Global: concurrency limiter
qa_semaphore = asyncio.Semaphore(20)

# Store failed pids globally
failed_comparisons = []

# Async QA comparison call
async def get_model_choice_qa_comparison_async(model_name, answer1, answer2, question, article, return_logprobs=0):
    async with qa_semaphore:
        prompt = QA_COMPARISON_PROMPT_TEMPLATE.format(
            article=article, question=question, answer1=answer1, answer2=answer2
        )
        exact_model = format_model_name_together(model_name)
        system_prompt = QA_COMPARISON_SYSTEM_PROMPT

        try:
            if model_name == "Mixtral-8x7B-Instruct-v0.1":
                response = await async_client.chat.completions.create(
                model=exact_model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                logprobs=return_logprobs,
                temperature=0.0
            )
            elif model_name == "gpt-oss-20b" or model_name == "GLM-4.5-Air-FP8":
                resp = await async_client.chat.completions.create(
                    model=exact_model,  # or whichever reasoning model you use
                    messages=[{"role":"user","content":prompt},
                        {"role": "system", "content": system_prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "only_choice",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type":"string","enum":["1","2"]}},
                                "required": ["answer"],
                                "additionalProperties": False
                            }
                        }
                    },
                    temperature=0
                )
                choice = resp.choices[0].message.content  # e.g. {"answer":"2"}
                choice = json.loads(choice)
                ans = choice['answer']
                return ans
            else:
                response = await async_client.chat.completions.create(
                    model=exact_model,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "system", "content": system_prompt}
                    ],
                    logprobs=return_logprobs,
                    temperature=0.0
                )
            # print(response)
            if return_logprobs:
                return response.choices[0].logprobs
            return response.choices[0].message.content

        except Exception as e:
            print(f"Failed QA comparison call for model {model_name}: {e}")
            return None
        
async def evaluate_pref_quality_async(evaluator_model, evaluatee_model, records, harmful_subset, 
                                      use_synonym=False, 
                                      use_synonym_other=False, 
                                      use_paraphrase=False, 
                                      paraphrase_source_external=False,  
                                      paraphrase_other_external=False,
                                      sentence_error_source=False, 
                                      sentence_error_other=False, 
                                      identity_naturalization=False,
                                      repeat_failures=False):
    model1 = evaluator_model
    model2 = evaluatee_model

    tasks = []

    for record in records:
        pid = record.get('pid')
        gt_label = record['output_label']
        model1_label = record.get(model1 + '_output_label')
        model2_label = record.get(model2 + '_output_label')
        if repeat_failures:
            if pid not in failed_comparisons:
                continue  # Only retry known failed records
        
        # Only compare if model1 is wrong and model2 is right
        if harmful_subset:
            if model1_label and model2_label and model1_label != gt_label and model2_label == gt_label:
                tasks.append(process_pref_record(record, model1, model2, use_synonym=use_synonym, use_synonym_other=use_synonym_other, use_paraphrase=use_paraphrase, 
                                                 paraphrase_source_external=paraphrase_source_external, paraphrase_other_external=paraphrase_other_external,
                                                 sentence_error_source=sentence_error_source, sentence_error_other=sentence_error_other,
                                                 identity_naturalization=identity_naturalization))
        else:
            if model1_label and model2_label and model1_label == gt_label and model2_label != gt_label:
                tasks.append(process_pref_record(record, model1, model2, use_synonym=use_synonym, use_synonym_other= use_synonym_other, use_paraphrase=use_paraphrase, 
                                                 paraphrase_source_external=paraphrase_source_external, paraphrase_other_external=paraphrase_other_external,
                                                 sentence_error_source=sentence_error_source, sentence_error_other=sentence_error_other,
                                                 identity_naturalization=identity_naturalization))
    for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Evaluating Preferences"):
        await future



async def process_pref_record(record, model1, model2, use_synonym=False, use_synonym_other=False, use_paraphrase=False, 
                              paraphrase_source_external=False, paraphrase_other_external=False, 
                              sentence_error_source=False, sentence_error_other=False,
                              identity_naturalization=False, return_logprobs=2):
    try:
        result = {
            'evaluator': model1,
            'evaluatee': model2,
            'pid': record['pid']
        }
        #Prepare answer 1
        if use_synonym:
            answer1 = record[model1+'_output_label'] + ". " + record[model1+'_reason_perturb_llm_auto']
        elif paraphrase_source_external:
            answer1 = record[model1+'_output_label'] + ". " + record[model1+'_reason_paraphrased_external']
        elif identity_naturalization:
            model1_choice = record[model1+'_output_label']
            model1_reason = record[model1 + '_reason']
        else:
            answer1 = str(record[model1 + '_output_label']) + ". " + record[model1 + '_reason']
        #Prepare answer 2  
        if use_synonym_other:
            answer2 = record[model2+'_output_label'] + ". " + record[model2+'_reason_perturb_llm_auto']
        elif paraphrase_other_external:
            answer2 = record[model2+'_output_label'] + ". " + record[model2+'_reason_paraphrased_external']
        elif use_paraphrase:
            answer2 = record[model2+'_output_label'] + ". " + record[model2+ '_reason_paraphrased_' + model1]
        else:
            answer2 = str(record[model2 + '_output_label']) + ". " + record[model2 + '_reason']


        
        forward_result = await get_model_choice_qa_comparison_async(
            model1, answer1, answer2, record['questions'], record['text'], return_logprobs=return_logprobs
        )
        backward_result = await get_model_choice_qa_comparison_async(
            model1, answer2, answer1, record['questions'], record['text'], return_logprobs=return_logprobs
        )
        # print(forward_result,'\n', backward_result)
        if not forward_result or not backward_result:
            failed_comparisons.append(record['pid'])
            return
        
        if model1 == "gpt-oss-20b" or model1 == "GLM-4.5-Air-FP8" or return_logprobs==0:
            result["forward_comparison"] = forward_result
            result["backward_comparison"] = backward_result
        elif getattr(forward_result, "content", None):  # new format
            result["forward_comparison"] = forward_result.content[0]["token"]
            result["forward_probability"] = exp(forward_result.content[0]["logprob"])
            result["backward_comparison"] = backward_result.content[0]["token"]
            result["backward_probability"] = exp(backward_result.content[0]["logprob"])
        else:  # old format fallback
            result["forward_comparison"] = forward_result.tokens[0]
            result["forward_probability"] = exp(forward_result.token_logprobs[0])    
            result["backward_comparison"] = backward_result.tokens[0]
            result["backward_probability"] = exp(backward_result.token_logprobs[0])


        preference_results.append(result)

    except Exception as e:
        print(f"Failed to process record {record['pid']}: {e}")
        failed_comparisons.append(record['pid'])





# MAIN Driver
def _parse_args(argv):
    p = argparse.ArgumentParser(description="Run evaluate_pref_quality_async")
    p.add_argument("--evaluator-model", required=True)
    p.add_argument("--evaluatee-model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--harmful-subset", action="store_true")
    p.add_argument("--storedir", type=Path, required=True)
    return p.parse_args(argv)

async def main():
    args = _parse_args(sys.argv[1:])
    with open(args.data, 'r') as file:
        responses = json.load(file)

    result = await evaluate_pref_quality_async(
        args.evaluator_model,
        args.evaluatee_model,
        responses,
        harmful_subset=args.harmful_subset,
    )

    # Save
    args.storedir.mkdir(parents=True, exist_ok=True)
    output_path = args.storedir / "evaluation_result.json"

    # Save the JSON output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preference_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")

preference_results = []
failed_comparisons = []

if __name__ == "__main__":
    asyncio.run(main())

# Sample Cmd to run: python .\preference.py  --evaluator-model Qwen2.5-7B-Instruct-Turbo --evaluatee-model Llama-4-Scout-17B-16E-Instruct --data ./data/quality_responses.json --storedir ./results --harmful-subset