_HISTORY_HEADERS = {
    "roman_urdu": "PICHLI GUFTAGU (sirf follow-up sawaal samajhne ke liye — isay qanooni source na samjhein):",
    "ur":         "پچھلی گفتگو (صرف سیاق و سباق سمجھنے کے لیے — اسے قانونی ماخذ نہ سمجھیں):",
    "en":         "PREVIOUS CONVERSATION (for context only — do not treat this as a legal source):",
}


def _history_block(history: list, lang: str) -> str:
    """Formats recent conversation turns into a short block that is
    inserted before CONTEXT in the prompt. Returns "" if there is no
    history yet (first message of a chat)."""
    if not history:
        return ""
    lines = []
    for turn in history:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    convo = "\n".join(lines)
    header = _HISTORY_HEADERS.get(lang, _HISTORY_HEADERS["en"])
    return f"{header}\n{convo}\n\n"


def build_prompt(query: str, context: str, lang: str, history: list = None) -> tuple:
    history_block = _history_block(history or [], lang)

    if lang == "roman_urdu":
        system_msg = (
            "Aap ek strict Pakistani qanooni assistant hain. "
            "Aap SIRF neeche diye gaye CONTEXT se jawab dete hain. "
            "HAMESHA Roman Urdu mein jawab do — "
            "matlab Urdu ko Latin haroof mein likho jaise 'Article 10 kehta hai...'. "
            "Kabhi bhi Urdu script (Arabic characters) mat use karo. "
            "Agar context mein jawab nahi hai to SIRF likho: "
            "'Is sawal ka jawab dataset mein maujood nahi hai.' "
            "Kabhi bhi apni taraf se kuch mat banao. "
            "PICHLI GUFTAGU sirf follow-up sawalat samajhne ke liye hai, "
            "usay qanooni maloomat ka source mat banao."
        )
        user_prompt = f"""Aap ek Pakistani qanooni chatbot hain. Neeche diye gaye CONTEXT ki madad se sawal ka jawab Roman Urdu mein dijiye.

ZAROORI QAWAID:
1. Sirf CONTEXT mein diya gaya information use karo — bahar ki koi knowledge nahi.
2. Relevant Article ya Section number zaroor batao agar context mein visible hai.
3. Agar CONTEXT mein jawab nahi hai to likho: "Is sawal ka jawab dataset mein maujood nahi hai."
4. PICHLI GUFTAGU sirf follow-up sawaal (jaise "iske exceptions kya hain?") samajhne ke liye use karo, legal fact ke tor par nahi.
5. Jawab is tarah do:
   **Qanooni Bunyad** → Konsa qanoon ya article lagoo hota hai
   **Mutaalliq Provision** → Context se exact provision kya kehti hai
   **Natija** → Khulasa kya nikalta hai

{history_block}CONTEXT:
{context}

SAWAL:
{query}

JAWAB (Roman Urdu mein likho, koi Urdu script nahi):"""

    elif lang == "ur":
        system_msg = (
            "آپ ایک سخت پاکستانی قانونی معاون ہیں۔ "
            "آپ صرف فراہم کردہ سیاق و سباق سے جواب دیتے ہیں۔ "
            "ہمیشہ اردو میں جواب دیں۔ "
            "اگر جواب موجود نہیں تو لکھیں: 'یہ معلومات ڈیٹاسیٹ میں موجود نہیں۔' "
            "کبھی بھی اپنی طرف سے کچھ نہ بنائیں۔ "
            "پچھلی گفتگو صرف فالو اپ سوالات سمجھنے کے لیے ہے، اسے قانونی ماخذ نہ سمجھیں۔"
        )
        user_prompt = f"""آپ ایک پاکستانی قانونی چیٹ بوٹ ہیں۔ نیچے دیے گئے سیاق و سباق کی مدد سے سوال کا جواب اردو میں دیجیے۔

لازمی قواعد:
1. صرف CONTEXT میں دی گئی معلومات استعمال کریں۔
2. متعلقہ آرٹیکل یا سیکشن نمبر ضرور بتائیں۔
3. اگر CONTEXT میں جواب نہیں تو لکھیں: "یہ معلومات ڈیٹاسیٹ میں موجود نہیں۔"
4. پچھلی گفتگو کو صرف فالو اپ سوال سمجھنے کے لیے استعمال کریں، قانونی حقیقت کے طور پر نہیں۔
5. جواب اس طرح دیں:
   **قانونی بنیاد** ← کون سا قانون یا آرٹیکل لاگو ہوتا ہے
   **متعلقہ شق** ← سیاق سے عین شق کیا کہتی ہے
   **نتیجہ** ← خلاصہ

{history_block}CONTEXT:
{context}

سوال:
{query}

جواب:"""

    else:
        system_msg = (
            "You are a strict Pakistani legal retrieval assistant. "
            "Never fabricate legal information. "
            "Answer ONLY from the provided context. "
            "The PREVIOUS CONVERSATION block, if present, is only for understanding "
            "follow-up questions — never treat it as a source of legal information."
        )
        user_prompt = f"""You are a Pakistani legal chatbot. Answer using ONLY the context below.

RULES:
1. Only use information explicitly present in the CONTEXT.
2. Quote the relevant Article or Section number if visible in the context.
3. If the context does not contain sufficient information, state:
   "The provided legal provisions do not directly address this query."
4. Do not add general legal knowledge not present in the context.
5. Use PREVIOUS CONVERSATION only to resolve follow-up references (e.g. "what about its exceptions?") — never as a legal source.
6. Structure: **Legal Basis** → **Applicable Provision** → **Conclusion**

{history_block}CONTEXT:
{context}

QUERY:
{query}

ANSWER:"""

    return system_msg, user_prompt
