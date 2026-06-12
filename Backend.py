import os
import re
import warnings

warnings.filterwarnings("ignore")

# ── Safe imports ─────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    print("[OK] rank_bm25 loaded")
except ImportError:
    raise ImportError("Run: pip install rank-bm25")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("[OK] sentence_transformers loaded")
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

try:
    import faiss
    print("[OK] faiss loaded")
except ImportError:
    raise ImportError("Run: pip install faiss-cpu")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

app = Flask(__name__, template_folder='Templates')

@app.route('/favicon.png')
def favicon():
    import pathlib
    path = pathlib.Path(__file__).parent / 'templates' / 'favicon.png'
    print(f"[FAVICON] Looking at: {path} | exists: {path.exists()}")
    return send_from_directory(str(path.parent), path.name, mimetype='image/png')

# ── Config ───────────────────────────────────────────────
file_path   = file_path = "fyp_cleaned_dataset.csv"
INDEX_FILE  = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
TOP_K       = 20
RERANK_TOP  = 5
MAX_TOKENS  = 900
MIN_SCORE   = 0.18   # slightly lower threshold = wider recall

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
MODEL_NAME = "gpt-oss-120b"


# ═══════════════════════════════════════════════════════════
# ROMAN URDU DETECTION
# ═══════════════════════════════════════════════════════════
ROMAN_URDU_KEYWORDS = {
    "mujhe","mein","mai","main","hum","aap","tum","woh","yeh","ye",
    "is","us","unka","unke","unki","apna","apne","apni",
    "hai","hain","tha","thi","the","hoga","hogi","honge",
    "karo","karna","karta","karti","karte","kar","kiya","ki",
    "ho","hua","hui","hue","ja","jao","jana","gaya","gayi",
    "milta","milti","milte","mile","batao","bata","puchna",
    "chahiye","chahta","chahti","chahte","sakta","sakti","sakte",
    "lagta","lagti","lagte","lena","dena","deta","deti",
    "kya","kyun","kaise","kab","kahan","kaun","kitna","kitne","kitne",
    "qanoon","adalat","haq","huqooq","zamin","zameen","mulk",
    "sarkar","hakumat","police","arrest","giraftari","muqadma",
    "waqil","judge","faisla","saza","jaidad","maal",
    "talaq","nikah","shadi","warasat","wirasat","merath",
    "constitution","parliament","assembly","vote",
    "election","intikhaab","ilzam","jurm","gunah",
    "aur","ya","lekin","magar","phir","bhi","hi","to",
    "ke","ka","se","par","pe","tak","wala","wali","wale",
    "nahi","nahin","mat","na",
    "batao","samjhao","bataiye","samjhaiye",
    "matlab","yani","yaani","iska","uska",
    "zaroor","bilkul","theek","sahi","galat",
    # time-related (key for the failing query)
    "ghante","ghanta","minute","din","raat","waqt","muddat",
    "baad","pehle","jald","jaldi","abhi","kab","kitni","kitne",
    # court-related
    "pesh","saamne","magistrate","judge","court","adalat",
    "sunwai","faisla","hukum","order","bail","remand",
    # arrest / detention specific
    "pakad","pakda","band","qaid","hirasaat","custody",
    "chhod","riha","release","tafrish","information","rights",
}

def detect_roman_urdu(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return False
    matches = sum(1 for t in tokens if t in ROMAN_URDU_KEYWORDS)
    if matches >= 2:
        return True
    if len(tokens) >= 4 and (matches / len(tokens)) >= 0.15:
        return True
    return False

def detect_language(text: str) -> str:
    if re.search(r'[\u0600-\u06FF]', text):
        return "ur"
    if detect_roman_urdu(text):
        return "roman_urdu"
    try:
        return detect(text)
    except Exception:
        return "en"


# ── Load dataset ─────────────────────────────────────────
def read_csv_dataset(path):
    print(f"Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path).fillna("")
    combined = df.astype(str).agg(" ".join, axis=1)
    result = "\n".join(combined.tolist())
    print(f"[OK] Dataset loaded: {len(result):,} characters")
    return result

def preprocess_text(text):
    text = re.sub(r"\bPage \d+\b", "", text)
    text = re.sub(r"_{3,}", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()

def create_legal_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=300,
        separators=[
            "\nArticle ", "\nPart ", "\nChapter ",
            "\n\n", "\n   (", "\n  (", "\n (",
            "\n", " ", ""
        ]
    )
    chunks = splitter.split_text(text)
    print(f"[OK] Created {len(chunks)} chunks")
    return chunks

try:
    dataset_text = read_csv_dataset(file_path)
    cleaned_text = preprocess_text(dataset_text)
    chunks_list  = create_legal_chunks(cleaned_text)
except Exception as e:
    print(f"[FATAL] Dataset loading failed: {e}")
    raise

print("Loading embedding model...")
try:
    embedder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    print("[OK] Embedding model loaded")
except Exception as e:
    print(f"[WARN] Primary model failed: {e}. Falling back...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("[OK] Fallback embedding model loaded")

print("Loading reranker model...")
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    USE_RERANKER = True
    print("[OK] Reranker loaded")
except Exception as e:
    print(f"[WARN] Reranker failed: {e}")
    USE_RERANKER = False

def build_faiss(chunks):
    print(f"Encoding {len(chunks)} chunks...")
    embeddings = embedder.encode(
        chunks, batch_size=16, convert_to_numpy=True, show_progress_bar=True
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    faiss.write_index(idx, INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks, dtype=object))
    print(f"[OK] FAISS index built: {len(chunks)} vectors")
    return idx

try:
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading existing FAISS index...")
        index  = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
        print(f"[OK] FAISS index loaded: {len(chunks)} chunks")
    else:
        print("No existing index. Building...")
        index  = build_faiss(chunks_list)
        chunks = chunks_list
except Exception as e:
    print(f"[WARN] Index load failed ({e}). Rebuilding...")
    for f in [INDEX_FILE, CHUNKS_FILE]:
        if os.path.exists(f): os.remove(f)
    index  = build_faiss(chunks_list)
    chunks = chunks_list

print("Building BM25 index...")
try:
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25     = BM25Okapi(tokenized_chunks)
    USE_BM25 = True
    print("[OK] BM25 index ready")
except Exception as e:
    print(f"[WARN] BM25 failed: {e}")
    USE_BM25 = False


# ═══════════════════════════════════════════════════════════
# ROMAN URDU → ENGLISH QUERY TRANSLATION
# This is the KEY fix for the failing query.
# Before retrieval, Roman Urdu queries are translated to their
# English legal equivalents so FAISS can find the right chunks.
# ═══════════════════════════════════════════════════════════

# Maps Roman Urdu phrases/words → English legal terms for FAISS retrieval
ROMAN_URDU_TO_ENGLISH = {
    # Arrest / detention
    "giraftari":            "arrest",
    "giraftaar":            "arrested",
    "hirasaat":             "custody detention",
    "qaid":                 "imprisonment custody",
    "remand":               "remand detention",
    "bail":                 "bail release",
    "riha":                 "release discharged",

    # Time expressions mapped to legal context
    "ghante mein":          "within hours",
    "kitne ghante":         "how many hours within period",
    "24 ghante":            "twenty four hours 24 hours",
    "ghante":               "hours period time",
    "muddat":               "period duration time limit",
    "waqt":                 "time period",

    # Court appearances
    "magistrate ke saamne pesh": "produced before magistrate",
    "pesh karna":           "produced before present",
    "saamne pesh":          "produced before appear",
    "magistrate":           "magistrate court",
    "sunwai":               "hearing proceeding",
    "faisla":               "judgment order decision",
    "hukum":                "order direction",

    # Rights
    "haq":                  "right entitlement",
    "huqooq":               "rights fundamental rights",
    "azadi":                "freedom liberty",
    "insaf":                "justice fair trial",
    "waqil":                "advocate lawyer legal practitioner",

    # Property / land
    "zamin":                "land property immovable",
    "zameen":               "land property immovable",
    "jaidad":               "property assets estate",
    "muawza":               "compensation payment",
    "zer qabd":             "possession acquisition",

    # Family law
    "talaq":                "divorce dissolution marriage",
    "nikah":                "marriage matrimonial",
    "warasat":              "inheritance succession",
    "wirasat":              "inheritance succession",
    "merath":               "inheritance legal heirs",

    # Parliament / government
    "hakumat":              "government federal government",
    "sarkar":               "government state",
    "parliament":           "parliament majlis-e-shoora national assembly",
    "assembly":             "assembly legislature provincial assembly",
    "vote":                 "vote election franchise",
    "intikhaab":            "election electoral",

    # Crime / punishment
    "jurm":                 "offence crime criminal",
    "gunah":                "offence crime",
    "saza":                 "punishment sentence penalty",
    "ilzam":                "charge accusation",
    "muqadma":              "case legal proceedings lawsuit",
    "adalat":               "court tribunal judicature",
    "qanoon":               "law statute legal provision",

    # General legal
    "shikayat":             "complaint petition grievance",
    "darkhwast":            "application petition request",
    "appeal":               "appeal appellate",
    "nazar sani":           "review revision",
    "ittila":               "information notice intimation",
    "wajah":                "ground reason cause",
    "saboot":               "evidence proof",
    "gawah":                "witness testimony",
}

def translate_roman_urdu_query(query: str) -> str:
    """
    Converts Roman Urdu query into an English-enriched query
    suitable for FAISS + BM25 retrieval against an English legal dataset.
    
    Strategy: append English equivalents — do NOT remove original words
    so the LLM still sees the original question for answer generation.
    """
    q_lower = query.lower()
    english_expansions = []

    # Multi-word phrases first (longer phrases take priority)
    sorted_mappings = sorted(
        ROMAN_URDU_TO_ENGLISH.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    matched_positions = set()
    for roman_phrase, english_eq in sorted_mappings:
        idx = q_lower.find(roman_phrase)
        if idx != -1:
            # Check no overlap with already matched positions
            positions = set(range(idx, idx + len(roman_phrase)))
            if not positions.intersection(matched_positions):
                matched_positions.update(positions)
                english_expansions.append(english_eq)

    if english_expansions:
        expanded = query + " " + " ".join(english_expansions)
        print(f"[TRANSLATE] Roman Urdu expansion: {' | '.join(english_expansions[:5])}")
        return expanded

    return query


# ── Legal synonym expansion (English) ────────────────────
LEGAL_SYNONYMS = {
    "land":         ["property", "immovable property", "acquisition"],
    "compensation": ["payment", "compulsory acquisition", "indemnity"],
    "arrest":       ["detention", "custody", "safeguards", "article 10"],
    "detention":    ["arrest", "custody", "preventive detention", "article 10"],
    "hours":        ["twenty-four hours", "24 hours", "period", "produced before magistrate"],
    "magistrate":   ["produced before magistrate", "24 hours", "article 10", "custody"],
    "produced":     ["magistrate", "24 hours", "arrest", "article 10"],
    "freedom":      ["fundamental rights", "liberty"],
    "acquire":      ["compulsory acquisition", "take possession"],
    "parliament":   ["majlis-e-shoora", "national assembly", "senate"],
    "court":        ["judicature", "high court", "supreme court"],
    "equality":     ["equal protection", "non-discrimination", "article 25"],
    "education":    ["right to education", "article 25a"],
    "religion":     ["freedom of religion", "article 20"],
    "speech":       ["freedom of speech", "article 19"],
    "fair trial":   ["article 10a", "due process", "right to fair trial"],
    "bail":         ["bail", "release", "detention", "article 10"],
    "inheritance":  ["succession", "legal heirs", "estate", "property"],
    "divorce":      ["dissolution of marriage", "family law", "matrimonial"],
    "marriage":     ["nikah", "matrimonial", "family law"],
    "property":     ["immovable property", "acquisition", "article 24"],
    "punishment":   ["sentence", "penalty", "offence", "criminal"],
    "election":     ["electoral", "franchise", "voting rights", "article 51"],
    "president":    ["head of state", "article 41", "article 48"],
    "prime minister": ["chief executive", "article 91", "cabinet"],
}

def expand_query(query: str) -> str:
    expanded = query
    q_lower  = query.lower()
    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in q_lower:
            expanded += " " + " ".join(synonyms[:2])
    art_match = re.search(r'article\s+(\d+[A-Za-z]*)', query, re.IGNORECASE)
    if art_match:
        expanded += f" Article {art_match.group(1)} constitution Pakistan law"
    return expanded


# ── Hybrid retrieval ─────────────────────────────────────
def hybrid_retrieve(query: str, lang: str, k: int = TOP_K) -> list:
    """
    For Roman Urdu queries: first translate to English, then retrieve.
    For all queries: apply synonym expansion, then hybrid BM25 + vector search.
    """
    # Step 1: Translate Roman Urdu → English enriched query
    if lang == "roman_urdu":
        retrieval_query = translate_roman_urdu_query(query)
    else:
        retrieval_query = query

    # Step 2: Apply English legal synonym expansion
    expanded = expand_query(retrieval_query)
    print(f"[EXPANDED] {expanded[:120]}...")

    # Step 3: Vector retrieval
    q_vec = embedder.encode([expanded], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    vec_scores, vec_indices = index.search(q_vec, min(k * 2, len(chunks)))

    rrf_scores = {}
    valid_vec_indices = []
    for rank, (score, idx) in enumerate(zip(vec_scores[0], vec_indices[0])):
        if idx < len(chunks) and score >= MIN_SCORE:
            valid_vec_indices.append(int(idx))
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (60 + rank + 1)

    # Step 4: BM25 retrieval
    if USE_BM25:
        try:
            bm25_scores = bm25.get_scores(expanded.lower().split())
            bm25_top_k  = np.argsort(bm25_scores)[::-1][:k * 2]
            for rank, idx in enumerate(bm25_top_k):
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (60 + rank + 1)
        except Exception as e:
            print(f"[WARN] BM25 retrieval error: {e}")

    if not rrf_scores:
        return [chunks[i] for i in valid_vec_indices[:k] if i < len(chunks)]

    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [chunks[i] for i in top_indices if i < len(chunks)]


# ── Reranking ────────────────────────────────────────────
def rerank(query: str, candidates: list, top_n: int = RERANK_TOP) -> list:
    if not candidates:
        return []
    if not USE_RERANKER or len(candidates) <= top_n:
        return candidates[:top_n]
    try:
        pairs  = [(query, c) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [text for _, text in ranked[:top_n]]
    except Exception as e:
        print(f"[WARN] Reranker error: {e}. Using top-{top_n} without reranking.")
        return candidates[:top_n]

def assemble_context(top_chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Provision {i+1}]\n{chunk}"
        for i, chunk in enumerate(top_chunks)
    )


# ═══════════════════════════════════════════════════════════
# LANGUAGE-AWARE PROMPT BUILDER
# ═══════════════════════════════════════════════════════════

def build_prompt(query: str, context: str, lang: str) -> tuple:
    """
    Returns (system_message, user_prompt) tuned for detected language.
    roman_urdu → respond in Roman Urdu (Latin script)
    ur         → respond in Urdu script
    en / other → respond in English
    """

    if lang == "roman_urdu":
        system_msg = (
            "Aap ek strict Pakistani qanooni assistant hain. "
            "Aap SIRF neeche diye gaye CONTEXT se jawab dete hain. "
            "HAMESHA Roman Urdu mein jawab do — "
            "matlab Urdu ko Latin haroof mein likho jaise 'Article 10 kehta hai...'. "
            "Kabhi bhi Urdu script (Arabic characters) mat use karo. "
            "Agar context mein jawab nahi hai to SIRF likho: "
            "'Is sawal ka jawab dataset mein maujood nahi hai.' "
            "Kabhi bhi apni taraf se kuch mat banao."
        )
        user_prompt = f"""Aap ek Pakistani qanooni chatbot hain. Neeche diye gaye CONTEXT ki madad se sawal ka jawab Roman Urdu mein dijiye.

ZAROORI QAWAID:
1. Sirf CONTEXT mein diya gaya information use karo — bahar ki koi knowledge nahi.
2. Relevant Article ya Section number zaroor batao agar context mein visible hai.
3. Agar CONTEXT mein jawab nahi hai to likho: "Is sawal ka jawab dataset mein maujood nahi hai."
4. Jawab is tarah do:
   **Qanooni Bunyad** → Konsa qanoon ya article lagoo hota hai
   **Mutaalliq Provision** → Context se exact provision kya kehti hai
   **Natija** → Khulasa kya nikalta hai

CONTEXT:
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
            "کبھی بھی اپنی طرف سے کچھ نہ بنائیں۔"
        )
        user_prompt = f"""آپ ایک پاکستانی قانونی چیٹ بوٹ ہیں۔ نیچے دیے گئے سیاق و سباق کی مدد سے سوال کا جواب اردو میں دیجیے۔

لازمی قواعد:
1. صرف CONTEXT میں دی گئی معلومات استعمال کریں۔
2. متعلقہ آرٹیکل یا سیکشن نمبر ضرور بتائیں۔
3. اگر CONTEXT میں جواب نہیں تو لکھیں: "یہ معلومات ڈیٹاسیٹ میں موجود نہیں۔"
4. جواب اس طرح دیں:
   **قانونی بنیاد** ← کون سا قانون یا آرٹیکل لاگو ہوتا ہے
   **متعلقہ شق** ← سیاق سے عین شق کیا کہتی ہے
   **نتیجہ** ← خلاصہ

CONTEXT:
{context}

سوال:
{query}

جواب:"""

    else:
        system_msg = (
            "You are a strict Pakistani legal retrieval assistant. "
            "Never fabricate legal information. "
            "Answer ONLY from the provided context."
        )
        user_prompt = f"""You are a Pakistani legal chatbot. Answer using ONLY the context below.

RULES:
1. Only use information explicitly present in the CONTEXT.
2. Quote the relevant Article or Section number if visible in the context.
3. If the context does not contain sufficient information, state:
   "The provided legal provisions do not directly address this query."
4. Do not add general legal knowledge not present in the context.
5. Structure: **Legal Basis** → **Applicable Provision** → **Conclusion**

CONTEXT:
{context}

QUERY:
{query}

ANSWER:"""

    return system_msg, user_prompt


# ── RAG core ─────────────────────────────────────────────
def rag_query(query: str) -> tuple:
    """Returns (answer_string, detected_language_string)."""
    try:
        print(f"\n[QUERY] {query}")

        lang = detect_language(query)
        print(f"[LANG]  Detected: {lang}")

        # Pass lang to retrieval so Roman Urdu queries get translated
        candidates = hybrid_retrieve(query, lang=lang, k=TOP_K)
        print(f"[RETRIEVE] {len(candidates)} candidates found")

        if not candidates:
            no_result = {
                "roman_urdu": "Is sawal ka jawab dataset mein nahi mila. Meherbani kar ke alag alfaz mein puchiye.",
                "ur":         "اس سوال کا جواب ڈیٹاسیٹ میں نہیں ملا۔ براہ کرم مختلف الفاظ میں پوچھیں۔",
            }
            return no_result.get(lang, "No relevant legal provisions found for this query."), lang

        # For reranking, use English-translated query for better cross-encoder scoring
        rerank_query = translate_roman_urdu_query(query) if lang == "roman_urdu" else query
        top_chunks = rerank(rerank_query, candidates, top_n=RERANK_TOP)
        print(f"[RERANK] {len(top_chunks)} chunks selected")

        context = assemble_context(top_chunks)
        print(f"[CONTEXT] {len(context):,} chars sent to LLM | lang={lang}")

        system_msg, user_prompt = build_prompt(query, context, lang)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS
        )

        answer = response.choices[0].message.content.strip()
        print(f"[RESPONSE] {len(answer)} chars | lang={lang}")
        return answer, lang

    except Exception as e:
        import traceback
        print(f"[ERROR] RAG query failed:\n{traceback.format_exc()}")
        return f"System error: {str(e) or 'Unknown error — check terminal for full traceback'}", "en"


# ── Routes ───────────────────────────────────────────────
@app.route("/")
def home():
    try:
        return render_template("Frontend.html")
    except Exception as e:
        return f"<h1>App is running!</h1><p>Template error: {str(e)}</p>", 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400
        answer, lang = rag_query(message)
        return jsonify({"reply": answer, "language": lang})
    except Exception as e:
        import traceback
        print(f"[ERROR] Chat route:\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/ping")
def ping():
    return "ok", 200

@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "chunks":     len(chunks),
        "bm25":       USE_BM25,
        "reranker":   USE_RERANKER,
        "index_type": type(index).__name__
    })

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    print("[READY] Flask is starting on port 7860...")
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        use_reloader=False,
        threaded=True
    )