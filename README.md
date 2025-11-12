# Mitigating Self-Preference by Authorship Obfuscation

This repository accompanies our research **“Mitigating Self-Preference by Authorship Obfuscation”**.  

It provides reproducible scripts for evaluating **recognition** and **preference** behaviors of large language models (LLMs) across tasks such as **MBPP (code)** and **QuALITY (text QA)**, as well as utilities.

---

## 🧩 Repository Structure

mitigating_self_preference/
├─ common/ # Shared utilities (clients, prompts, JSON helpers, etc.)
├─ task_mbpp/ # Code-based MBPP experiments
│ ├─ generate_restyle.py
│ ├─ preference.py
│ └─ recognition.py
├─ task_quality/ # Text-based QA experiments
│ ├─ generate_answers.py
│ ├─ preference.py
│ └─ recognition.py
└─ requirements.txt

Each `preference.py` and `recognition.py` file acts as a **main driver** for its respective experiment.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Taslim-M/mitigating_self_preference.git
cd mitigating_self_preference
```

### 2. Create and activate a virtual environment (recommended)

```
python -m venv venv
# Windows
venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -U pip
pip install -r requirements.txt
```

### 4. Add your API key
Create a .env file in the project root:
TOGETHER_API_KEY=your_together_api_key_here

## Dataset
All experiment scripts expect .json or .jsonl input files following the schema used in the paper.

You can download our dataset from Hugging Face:

Hugging Face Dataset Link ← [(VIEW)](https://huggingface.co/datasets/taslimmahbub/mitigating-self-preference)

Place your data files inside:

```
task_quality/data/
task_mbpp/data/
```

## Running Experiments
Example: Quality Task (Preference Evaluation)

```
python -m task_quality.preference \
  --evaluator-model Qwen2.5-7B-Instruct-Turbo \
  --evaluatee-model Llama-4-Scout-17B-16E-Instruct \
  --data ./task_quality/data/quality_responses.json \
  --storedir ./results \
  --harmful-subset
```

Example: Quality Task (Recognition Evaluation)
```
python -m task_quality.recognition \
  --evaluator-model Qwen2.5-7B-Instruct-Turbo \
  --evaluatee-model Llama-4-Scout-17B-16E-Instruct \
  --data ./task_quality/data/quality_responses.json \
  --storedir ./results \
  --harmful-subset
```

### Citation
If you use this codebase in your research, please cite our paper:

@inproceedings{tas2025mitigating,
  title={Mitigating Self-Preference by Authorship Obfuscation},
  author={},
  year={2026},
  url={}
}