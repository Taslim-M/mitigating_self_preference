import sys
import json
import asyncio
from math import exp
from pathlib import Path
import argparse

from tqdm.asyncio import tqdm_asyncio

from common.clients import async_together, qa_semaphore, format_model_name_together
from common.code_utils import remove_comments_keep_lines, code_runs_ok
from common.prompts import (
    QA_DETECTION_SYSTEM_PROMPT_MBPP,
    QA_DETECTION_PROMPT_TEMPLATE_MBPP,
)

_client = async_together()

recog_results = []
failed_comparisons = []

def _parse_args(argv):
    p = argparse.ArgumentParser(description="Run evaluate_pref_quality_async")
    p.add_argument("--evaluator-model", required=True)
    p.add_argument("--evaluatee-model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--harmful-subset", action="store_true")
    p.add_argument("--storedir", type=Path, required=True)
    return p.parse_args(argv)


async def get_model_choice_qa_recognition_async(model_name, answer1, answer2, question, return_logprobs=0):
    async with qa_semaphore:
        prompt = QA_DETECTION_PROMPT_TEMPLATE_MBPP.format(
            problem=question, answer1=answer1, answer2=answer2
        )
        exact = format_model_name_together(model_name)
        try:
            resp = await _client.chat.completions.create(
                model=exact,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": QA_DETECTION_SYSTEM_PROMPT_MBPP}
                ],
                logprobs=return_logprobs,
                temperature=0.0
            )
            return resp.choices[0].logprobs if return_logprobs else resp.choices[0].message.content
        except Exception as e:
            print(f"Failed QA recog call for model {model_name}: {e}")
            return None


async def process_recog_record(record, model1, model2,
                               remove_comments=False,
                               use_paraphrased_code=False,
                               use_paraphrased_code_qwen=False):
    try:
        result = {'evaluator': model1, 'evaluatee': model2, 'unique_id': record['unique_id']}
        if remove_comments:
            answer1 = remove_comments_keep_lines(record.get('assistent_1_answer'))
            if not code_runs_ok(answer1):
                answer1 = record.get('assistent_1_answer')
                print(f"Warning: Code in answer1 for {record['unique_id']} failed after removing comments. Using original.")
        elif use_paraphrased_code:
            answer1 = record.get('assistent_1_answer_restyle')
        elif use_paraphrased_code_qwen:
            answer1 = record.get('assistent_1_answer_restyle_qwen3')
        else:
            answer1 = record.get('assistent_1_answer')

        if remove_comments:
            answer2 = remove_comments_keep_lines(record.get('assistent_2_answer'))
            if not code_runs_ok(answer2):
                answer2 = record.get('assistent_2_answer')
                print(f"Warning: Code in answer2 for {record['unique_id']} failed after removing comments. Using original.")
        else:
            answer2 = record.get('assistent_2_answer')

        if not answer1 or not answer2:
            print(f"Warning: empty code for {record['unique_id']}. Skipping.")
            return

        forward = await get_model_choice_qa_recognition_async(model1, answer1, answer2, record['problem'], return_logprobs=1)
        backward = await get_model_choice_qa_recognition_async(model1, answer2, answer1, record['problem'], return_logprobs=1)

        if not forward or not backward:
            failed_comparisons.append(record['unique_id'])
            return

        result["forward_detection"] = forward.tokens[0]
        result["forward_probability"] = exp(forward.token_logprobs[0])
        result["backward_detection"] = backward.tokens[0]
        result["backward_probability"] = exp(backward.token_logprobs[0])
        recog_results.append(result)
    except Exception as e:
        print(f"Failed to process record {record['unique_id']}: {e}")
        failed_comparisons.append(record['unique_id'])


async def evaluate_recog_quality_async(evaluator_model, evaluatee_model, records, harmful_subset,
                                       remove_comments=False, use_paraphrased_code=False,
                                       use_paraphrased_code_qwen=False, repeat_failures=False):
    model1, model2 = evaluator_model, evaluatee_model
    tasks = []
    for record in records:
        pid = record.get('unique_id')
        if repeat_failures and pid not in failed_comparisons:
            continue
        a1 = record.get('assistent_1_is_correct')
        a2 = record.get('assistent_2_is_correct')
        cond = (a1 is False and a2 is True) if harmful_subset else (a1 is True and a2 is False)
        if cond:
            tasks.append(process_recog_record(
                record, model1, model2,
                remove_comments=remove_comments,
                use_paraphrased_code=use_paraphrased_code,
                use_paraphrased_code_qwen=use_paraphrased_code_qwen
            ))
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Evaluating Recognition"):
        await fut


async def main():
    args = _parse_args(sys.argv[1:])
    with open(args.data, 'r') as f:
        records = [json.loads(line) for line in f]
    print("harmful subset:", args.harmful_subset)
    await evaluate_recog_quality_async(
        args.evaluator_model, args.evaluatee_model, records,
        harmful_subset=args.harmful_subset
    )
    args.storedir.mkdir(parents=True, exist_ok=True)
    out = args.storedir / "evaluation_recog_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recog_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out}")

if __name__ == "__main__":
    asyncio.run(main())

# cmd to run (from root):
# python -m task_mbpp.recognition  --evaluator-model Meta-Llama-3.1-70B-Instruct-Turbo --evaluatee-model gpt-4o --data ./task_mbpp/data/llama-3.3-70b_gpt-4o_eval.jsonl --storedir ./results --harmful-subset
