import time

from google import genai

from app.config import Config

if not Config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(
    api_key=Config.GEMINI_API_KEY
)


def call_llm_with_retry(
    system_msg: str,
    user_prompt: str,
    max_retries: int = 2,
    base_delay: float = 2.0,
    model: str = None,
):
    model = model or Config.MODEL_NAME

    prompt = f"""
System:
{system_msg}

User:
{user_prompt}
"""

    last_error = None

    for attempt in range(max_retries + 1):

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            return response.text.strip()

        except Exception as e:

            last_error = e

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"[Retry] {delay}s")
                time.sleep(delay)
            else:
                raise last_error

def call_llm_with_fallback(system_msg: str, user_prompt: str) -> tuple:
    """
    Tries the primary model first (with its own built-in retry-on-429
    logic). If the primary fails for ANY reason after its retries are
    exhausted — rate limit, timeout, insufficient credits, provider
    outage, bad gateway — automatically falls back to a second free
    Gemini model before giving up entirely.

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