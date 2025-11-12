import sys
import json
import asyncio
from math import exp
from pathlib import Path
import argparse

from tqdm.asyncio import tqdm_asyncio

from common.clients import async_together, qa_semaphore, format_model_name_together
from common.prompts import (
    QA_DETECTION_SYSTEM_PROMPT_QUALITY,
    QA_DETECTION_PROMPT_TEMPLATE_QUALITY,
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


async def get_model_choice_qa_recognition_async(model_name, answer1, answer2, question, article, return_logprobs=0):
    async with qa_semaphore:
        prompt = QA_DETECTION_PROMPT_TEMPLATE_QUALITY.format(
            article=article, question=question, answer1=answer1, answer2=answer2
        )
        exact = format_model_name_together(model_name)
        try:
            if model_name in ('Mixtral-8x7B-Instruct-v0.1',):
                resp = await _client.chat.completions.create(
                    model=exact,
                    messages=[{"role": "user", "content": prompt}],
                    logprobs=return_logprobs,
                    temperature=0.0
                )
            elif model_name in ("gpt-oss-20b", "GLM-4.5-Air-FP8"):
                resp = await _client.chat.completions.create(
                    model=exact,
                    messages=[{"role":"user","content":prompt},
                              {"role": "system", "content": QA_DETECTION_SYSTEM_PROMPT_QUALITY}],
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
                choice = json.loads(resp.choices[0].message.content)
                return choice['answer']
            else:
                resp = await _client.chat.completions.create(
                    model=exact,
                    messages=[{"role": "user", "content": prompt},
                              {"role": "system", "content": QA_DETECTION_SYSTEM_PROMPT_QUALITY}],
                    logprobs=return_logprobs,
                    temperature=0.0
                )
            return resp.choices[0].logprobs if return_logprobs else resp.choices[0].message.content
        except Exception as e:
            print(f"Failed QA recog call for model {model_name}: {e}")
            return None


async def process_recog_record(record, model1, model2,
                               use_synonym=False, use_paraphrase=False,
                               paraphrase_source_external=False,
                               cross_reason=False, same_answer_dupe=False, return_logprobs=1):
    try:
        result = {'evaluator': model1, 'evaluatee': model2, 'pid': record['pid']}
        # Prepare answer1
        if use_synonym:
            answer1 = record[model1+'_output_label'] + ". " + record[model1+'_reason_perturb_llm_auto']
        elif paraphrase_source_external:
            answer1 = record[model1+'_output_label'] + ". " + record[model1+'_reason_paraphrased_external']
        else:
            answer1 = record[model1 + '_output_label'] + ". " + record[model1 + '_reason']
        # Prepare answer2
        if use_paraphrase:
            answer2 = record[model2+'_output_label'] + ". " + record[model2+ '_reason_paraphrased_' + model1]
        else:
            answer2 = record[model2 + '_output_label'] + ". " + record[model2 + '_reason']
        # Cross-reason (preserve your variants)
        if cross_reason:
            m1 = record.get(model1 + '_output_label')
            m2 = record.get(model2 + '_output_label')
            answer1 = record[model1 + '_output_label'] + ". " + record[model2+ '_Dupe_reason_output_label_'+ str(m1)]
            answer2 = record[model2 + '_output_label'] + ". " + record[model1+ '_Dupe_reason_output_label_'+ str(m2)]
        if same_answer_dupe:
            m2 = record.get(model2 + '_output_label')
            answer1 = record[model2 + '_output_label'] + ". " + record[model1+ '_Dupe_reason_output_label_'+ str(m2)]
            answer2 = record[model2 + '_output_label'] + ". " + record[model2 + '_reason']

        forward = await get_model_choice_qa_recognition_async(
            model1, answer1, answer2, record['questions'], record['text'], return_logprobs=return_logprobs
        )
        backward = await get_model_choice_qa_recognition_async(
            model1, answer2, answer1, record['questions'], record['text'], return_logprobs=return_logprobs
        )

        if not forward or not backward:
            failed_comparisons.append(record['pid'])
            return

        if model1 in ("gpt-oss-20b", "GLM-4.5-Air-FP8") or return_logprobs == 0:
            result["forward_detection"] = forward
            result["backward_detection"] = backward
        elif getattr(forward, "content", None):
            result["forward_detection"] = forward.content[0]["token"]
            result["forward_probability"] = exp(forward.content[0]["logprob"])
            result["backward_detection"] = backward.content[0]["token"]
            result["backward_probability"] = exp(backward.content[0]["logprob"])
        else:
            result["forward_detection"] = forward.tokens[0]
            result["forward_probability"] = exp(forward.token_logprobs[0])
            result["backward_detection"] = backward.tokens[0]
            result["backward_probability"] = exp(backward.token_logprobs[0])

        recog_results.append(result)
    except Exception as e:
        print(f"Failed to process record {record['pid']}: {e}")
        failed_comparisons.append(record['pid'])


async def evaluate_recog_quality_async(evaluator_model, evaluatee_model, records, harmful_subset,
                                       use_synonym=False, use_paraphrase=False,
                                       paraphrase_source_external=False, cross_reason=False,
                                       same_answer_dupe=False, repeat_failures= False):
    model1, model2 = evaluator_model, evaluatee_model
    tasks = []
    for record in records:
        pid = record.get('pid')
        if repeat_failures and pid not in failed_comparisons:
            continue
        gt = record['output_label']
        m1 = record.get(model1 + '_output_label')
        m2 = record.get(model2 + '_output_label')
        cond = (m1 and m2 and m1 != gt and m2 == gt) if harmful_subset else (m1 and m2 and m1 == gt and m2 != gt)
        if cond:
            tasks.append(process_recog_record(
                record, model1, model2,
                use_synonym=use_synonym, use_paraphrase=use_paraphrase,
                paraphrase_source_external=paraphrase_source_external,
                cross_reason=cross_reason, same_answer_dupe=same_answer_dupe
            ))
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Evaluating Recognition"):
        await fut


async def main():
    args = _parse_args(sys.argv[1:])
    with open(args.data, 'r') as f:
        responses = json.load(f)
    await evaluate_recog_quality_async(
        args.evaluator_model, args.evaluatee_model, responses,
        harmful_subset=args.harmful_subset
    )
    args.storedir.mkdir(parents=True, exist_ok=True)
    out = args.storedir / "evaluation_recog_quality_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(recog_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out}")

if __name__ == "__main__":
    asyncio.run(main())

# Sample:
# python -m task_quality.recognition  --evaluator-model Qwen2.5-7B-Instruct-Turbo --evaluatee-model Llama-4-Scout-17B-16E-Instruct --data ./task_quality/data/quality_responses.json --storedir ./results --harmful-subset
