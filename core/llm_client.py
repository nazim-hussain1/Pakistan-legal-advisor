import time

from openai import OpenAI

from app.config import Config

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

def call_llm_with_fallback(system_msg: str, user_prompt: str) -> tuple:
    """
    Tries the primary model first (with its own built-in retry-on-429
    logic). If the primary fails for ANY reason after its retries are
    exhausted — rate limit, timeout, insufficient credits, provider
    outage, bad gateway — automatically falls back to a second free
    OpenRouter model before giving up entirely.

    Returns (answer_text, model_name_actually_used) so callers/logs
    can tell which model produced the response.
    """
    try:
        answer = call_llm_with_retry(
            system_msg, user_prompt,
            max_retries=2, base_delay=2.0,
            model=Config.MODEL_NAME
        )
        return answer, Config.MODEL_NAME
    except Exception as primary_error:
        print(f"[FALLBACK] Primary model '{Config.MODEL_NAME}' failed: {primary_error}")
        print(f"[FALLBACK] Trying fallback model '{Config.FALLBACK_MODEL_NAME}'...")
        try:
            answer = call_llm_with_retry(
                system_msg, user_prompt,
                max_retries=1, base_delay=2.0,
                model=Config.FALLBACK_MODEL_NAME
            )
            print(f"[FALLBACK] Fallback model succeeded")
            return answer, Config.FALLBACK_MODEL_NAME
        except Exception as fallback_error:
            print(f"[FALLBACK] Fallback model also failed: {fallback_error}")
            raise fallback_error