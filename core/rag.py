import traceback

from core import memory
from app.config import Config
from core.language_detection import detect_language
from core.smalltalk import check_smalltalk
from core.translation import translate_roman_urdu_query, translate_urdu_script_query
from core.retrieval import hybrid_retrieve, rerank, assemble_context
from core.prompts import build_prompt
from core.llm_client import call_llm_with_retry

def rag_query(query: str) -> tuple:
    """Returns (answer_string, detected_language_string). Also updates
    the per-browser conversation memory (see memory.py) on every
    successful exchange."""
    try:
        print(f"\n[QUERY] {query}")
        lang = detect_language(query)
        print(f"[LANG]  Detected: {lang}")

        smalltalk = check_smalltalk(query, lang)
        if smalltalk:
            print("[SMALLTALK] Matched — returning canned response")
            answer, resp_lang = smalltalk
            memory.push_turn("user", query)
            memory.push_turn("assistant", answer)
            return answer, resp_lang

        candidates = hybrid_retrieve(query, lang=lang, k=Config.TOP_K)
        print(f"[RETRIEVE] {len(candidates)} candidates found")
        if not candidates:
            no_result = {
                "roman_urdu": "Is sawal ka jawab dataset mein nahi mila. Meherbani kar ke alag alfaz mein puchiye.",
                "ur":         "اس سوال کا جواب ڈیٹاسیٹ میں نہیں ملا۔ براہ کرم مختلف الفاظ میں پوچھیں۔",
            }
            answer = no_result.get(lang, "No relevant legal provisions found for this query.")
            memory.push_turn("user", query)
            memory.push_turn("assistant", answer)
            return answer, lang

        if lang == "roman_urdu":
            rerank_query = translate_roman_urdu_query(query)
        elif lang == "ur":
            rerank_query = translate_urdu_script_query(query)
        else:
            rerank_query = query

        top_chunks = rerank(rerank_query, candidates, top_n=Config.RERANK_TOP)
        print(f"[RERANK] {len(top_chunks)} chunks selected")
        context = assemble_context(top_chunks)
        print(f"[CONTEXT] {len(context):,} chars sent to LLM | lang={lang}")

        history = memory.get_history()
        system_msg, user_prompt = build_prompt(query, context, lang, history=history)

        try:
            answer, model_used = call_llm_with_fallback(system_msg, user_prompt)
            print(f"[RESPONSE] {len(answer)} chars | lang={lang} | model={model_used}")
            memory.push_turn("user", query)
            memory.push_turn("assistant", answer)
            return answer, lang
        except Exception as llm_error:
            print(f"[ERROR] LLM call failed after retries:\n{traceback.format_exc()}")
            err_str = str(llm_error)
            is_rate_limit = "429" in err_str or "rate-limited" in err_str.lower() or "rate_limit" in err_str.lower()
            friendly_messages = {
                "roman_urdu": (
                    "Is waqt sawaal ka jawab dene mein masla ho raha hai kyun ke AI model par "
                    "zyada load hai. Meherbani kar ke thodi dair baad dobara koshish karein."
                    if is_rate_limit else
                    "Is sawaal ka jawab dete waqt kuch masla pesh aa gaya. Meherbani kar ke dobara koshish karein."
                ),
                "ur": (
                    "اس وقت جواب دینے میں مسئلہ ہو رہا ہے کیونکہ AI ماڈل پر زیادہ لوڈ ہے۔ "
                    "براہ کرم تھوڑی دیر بعد دوبارہ کوشش کریں۔"
                    if is_rate_limit else
                    "اس سوال کا جواب دیتے وقت کچھ مسئلہ پیش آ گیا۔ براہ کرم دوبارہ کوشش کریں۔"
                ),
                "en": (
                    "The legal assistant is currently experiencing high demand. "
                    "Please wait a moment and try again."
                    if is_rate_limit else
                    "Something went wrong while generating a response. Please try again."
                ),
            }
            # NOTE: we deliberately do not push a failed exchange into
            # conversation memory — a transient error shouldn't pollute
            # the context used for the next follow-up question.
            return friendly_messages.get(lang, friendly_messages["en"]), lang
    except Exception as e:
        print(f"[ERROR] RAG query failed:\n{traceback.format_exc()}")
        return f"System error: {str(e) or 'Unknown error'}", "en"
