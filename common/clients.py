import os
import asyncio
from dotenv import load_dotenv
from together import AsyncTogether

load_dotenv()

# single shared AsyncTogether client
_async_together = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# global concurrency defaults 
QA_CONCURRENCY = int(os.environ.get("QA_CONCURRENCY", "10"))
PARAPHRASE_CONCURRENCY = int(os.environ.get("PARAPHRASE_CONCURRENCY", "10"))

qa_semaphore = asyncio.Semaphore(QA_CONCURRENCY)
paraphrase_semaphore = asyncio.Semaphore(PARAPHRASE_CONCURRENCY)

def async_together() -> AsyncTogether:
    """Return the shared AsyncTogether client."""
    return _async_together


def format_model_name_together(model_name: str) -> str:
    """
    Normalizes friendly model names to Together-hosted slugs.
    """
    if model_name.startswith("Meta-Llama") or model_name.startswith("Llama"):
        return f"meta-llama/{model_name}"
    if model_name.startswith("Qwen"):
        return f"Qwen/{model_name}"
    if model_name.startswith("DeepSeek"):
        return f"deepseek-ai/{model_name}"
    if model_name.startswith("Mistral") or model_name.startswith("Mixtral"):
        return f"mistralai/{model_name}"
    if model_name.startswith("Kimi"):
        return f"moonshotai/{model_name}"
    if model_name.startswith("gpt"):
        return f"openai/{model_name}"
    if model_name.startswith("GLM"):
        return f"zai-org/{model_name}"
    return model_name
