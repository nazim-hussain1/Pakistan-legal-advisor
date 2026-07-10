import time

from openai import OpenAI

from config import Config

if not Config.OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

client = OpenAI(
    base_url=Config.OPENROUTER_BASE_URL,
    api_key=Config.OPENROUTER_API_KEY,
)


def call_llm_with_retry(system_msg: str, user_prompt: str, max_retries: int = 2,
                         base_delay: float = 2.0, model: str = None) -> str:
    """
    Calls the LLM with automatic retry on rate-limit (429) errors.
    Uses exponential backoff: 2s, 4s, 8s... up to max_retries attempts.
    Raises the last exception if all retries are exhausted.
    """
    model = model or Config.MODEL_NAME
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=Config.MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate-limited" in err_str.lower() or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"[RETRY] Rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise last_error
    raise last_error
