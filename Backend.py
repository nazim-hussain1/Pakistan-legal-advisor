import os
import re
import random
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

@app.route("/")
def home():
    return render_template("Frontend.html")

# ── Config ───────────────────────────────────────────────
file_path   = "fyp_cleaned_dataset.csv"
INDEX_FILE  = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
TOP_K       = 20
RERANK_TOP  = 5
MAX_TOKENS  = 900
MIN_SCORE   = 0.16   # slightly lower threshold = wider recall

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")


# ═══════════════════════════════════════════════════════════
# GREETING & SMALL-TALK DETECTION
# Handles greetings, creator questions, capability queries —
# all returned instantly without touching the RAG pipeline.
# ═══════════════════════════════════════════════════════════

# ── English greeting patterns ────────────────────────────
GREETING_PATTERNS_EN = re.compile(
    r"^\s*(hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening|night|day)|"
    r"salaam|salam|assalam|assalamu|what'?s?\s*up|sup|yo|hiya|heya|namaste|"
    r"how\s*are\s*(you|u)|how'?s?\s*(it\s*going|everything|life)|"
    r"hope\s*you'?re?\s*(well|good|fine)|nice\s*to\s*(meet|see)\s*you|"
    r"pleased\s*to\s*(meet|see)\s*you)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

# ── Roman Urdu greeting patterns ─────────────────────────
GREETING_PATTERNS_RU = re.compile(
    r"^\s*(salam|salaam|assalam|assalamu\s*alaikum|walaikum|wslm|aoa|"
    r"adab|sat\s*sri\s*akal|haan\s*bhai|kya\s*haal|kaise\s*ho|kaisi\s*ho|"
    r"kya\s*haal\s*(hai|hain)|aap\s*kaise\s*(hain|ho)|ap\s*kaise\s*(hain|ho)|"
    r"theek\s*ho|sab\s*theek|sub\s*theek|kya\s*hal|kiya\s*hal|"
    r"namaste|namaskar|jai\s*hind|hello\s*bhai|hi\s*bhai|"
    r"haan\s*ji|ji\s*haan|ji\s*han|bohat\s*(aacha|acha|khush)|"
    r"marhaba|mubarik|mubaarak)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

# ── Urdu script greeting patterns ────────────────────────
GREETING_PATTERNS_UR = re.compile(
    r"^\s*(السلام|سلام|آداب|ہیلو|ہائے|صبح\s*بخیر|شام\s*بخیر|کیسے\s*ہیں|کیا\s*حال)\s*.*$"
)

# ── Creator / identity question patterns ─────────────────
CREATOR_PATTERNS = re.compile(
    r"(who\s*(made|built|created|developed|designed|coded|programmed|trained|wrote)\s*(you|this|u)|"
    r"who'?s?\s*your\s*(creator|developer|maker|author|owner|builder)|"
    r"who\s*(are\s*you|r\s*u)|what\s*are\s*you|tell\s*me\s*about\s*yourself|"
    r"introduce\s*yourself|your\s*(name|identity|origin)|"
    r"kis\s*ne\s*(banaya|banaaya|likha|develop|create)|"
    r"tumhe\s*kis\s*ne\s*(banaya|banaaya)|aap\s*ko\s*kis\s*ne\s*(banaya|banai)|"
    r"apna\s*(naam|parichay|taruf)\s*(batao|bataiye|do)|"
    r"tum\s*(kaun|kon)\s*(ho|hain)|aap\s*(kaun|kon)\s*(ho|hain)|"
    r"ap\s*(kaun|kon)\s*(hain|ho)|kaun\s*(ho|hain)\s*(tum|aap|ap)|"
    r"tumhara\s*naam|aap\s*ka\s*naam|apka\s*naam|"
    r"آپ\s*کون|آپ\s*کا\s*نام|کس\s*نے\s*بنایا)",
    re.IGNORECASE
)

# ── Capability / help patterns ───────────────────────────
CAPABILITY_PATTERNS = re.compile(
    r"(what\s*can\s*(you|u)\s*do|how\s*can\s*(you|u)\s*help|"
    r"what\s*(do\s*you|are\s*you\s*able\s*to)\s*(do|know|cover|handle)|"
    r"help\s*me\s*with|i\s*need\s*(help|assistance)|"
    r"kya\s*kar\s*sakte\s*(ho|hain)|aap\s*kya\s*(kar|bata)\s*sakte|"
    r"ap\s*kya\s*kar\s*sakte|mujhe\s*madad\s*chahiye|"
    r"kaise\s*madad\s*karoge|tumse\s*kya\s*puchh\s*sakta|"
    r"start|begin|shuru|kaise\s*shuru|kahan\s*se\s*shuru|"
    r"آپ\s*کیا\s*کر\s*سکتے)",
    re.IGNORECASE
)

# ── Thanks patterns ───────────────────────────────────────
THANKS_PATTERNS = re.compile(
    r"^\s*(thanks|thank\s*you|thank\s*u|thx|ty|shukriya|shukriyah|meherbani|"
    r"bohat\s*shukriya|bahut\s*shukriya|bahut\s*meherbani|jazak\s*allah|"
    r"جزاک\s*اللہ|شکریہ|بہت\s*شکریہ)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

# ── Farewell patterns ─────────────────────────────────────
FAREWELL_PATTERNS = re.compile(
    r"^\s*(bye|goodbye|good\s*bye|see\s*you|later|take\s*care|"
    r"khuda\s*hafiz|allah\s*hafiz|alvida|phir\s*milenge|"
    r"خدا\s*حافظ|اللہ\s*حافظ|الوداع)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

# ── Responses ─────────────────────────────────────────────

GREETING_RESPONSES_EN = [
    "Hello! 👋 Welcome to the **Pakistan Legal Advisor**. I'm here to help you navigate Pakistani law — the Constitution, criminal law, property rights, family law, and more. What legal question can I assist you with today?",
    "Hi there! Great to have you here. I'm your AI-powered legal assistant specializing in Pakistani law. Ask me anything about the Constitution of Pakistan, your fundamental rights, legal procedures, or any statute. How can I help?",
    "Hello and welcome! ⚖️ I'm the Pakistan Legal Advisor — an intelligent chatbot trained on verified Pakistani legal provisions. Feel free to ask in **English, Urdu, or Roman Urdu**. What would you like to know?",
    "Hey! Good to see you. I'm here to make Pakistani law accessible to everyone. Whether it's about fundamental rights, court procedures, property, family law, or the Constitution — just ask away. What's on your mind?",
]

GREETING_RESPONSES_RU = [
    "Salam! 👋 Pakistan Legal Advisor mein aapka khair maqdam hai. Main aapko Pakistani qanoon ke baare mein madad karne ke liye yahan hoon — Constitution, criminal law, family law, property rights, aur bohat kuch. Aaj kya jaanna chahte hain?",
    "Assalam u Alaikum! Khoosh amdeed. Main ek AI-powered legal chatbot hoon jo Pakistani qanoon mein mahir hai. Aap mujhse **English, Urdu, ya Roman Urdu** mein pooch sakte hain. Kya sawal hai aapka?",
    "Hello ji! ⚖️ Pakistan Legal Advisor mein aapka swagat hai. Fundamental rights, court procedures, property, ya Constitution — kuch bhi poochhiye, main haazir hoon. Kaise madad kar sakta hoon?",
    "Salam ji! Mujhe khushi hai ke aap aaye. Pakistani qanoon ke baare mein koi bhi sawaal poochhiye — main verified qanooni malumaat se jawab doonga. Batayein, kya jaanna chahte hain?",
]

GREETING_RESPONSES_UR = [
    "السلام علیکم! 👋 پاکستان لیگل ایڈوائزر میں خوش آمدید۔ میں پاکستانی قانون کے بارے میں آپ کی مدد کے لیے حاضر ہوں۔ آج کیا جاننا چاہتے ہیں؟",
    "ہیلو! ⚖️ میں ایک AI قانونی معاون ہوں جو پاکستانی قانون میں ماہر ہے۔ آئین، بنیادی حقوق، عدالتی طریقہ کار — کچھ بھی پوچھیں۔",
]

CREATOR_RESPONSE_EN = """I'm the **Pakistan Legal Advisor** — an AI-powered legal chatbot built to make Pakistani law accessible to everyone.

**Developer:** Nazim Hussain
**Institution:** Quaid-e-Awam University of Engineering, Science & Technology (QUEST), Nawabshah
**Program:** BS Artificial Intelligence, Final Year Project — 2026

Nazim built me with a clear mission: to bridge the gap between complex legal statutes and everyday citizens, students, and legal professionals across Pakistan. I leverage a hybrid Retrieval-Augmented Generation (RAG) system — combining FAISS vector search, BM25 sparse retrieval, multilingual sentence embeddings, and Cross-Encoder reranking — powered by a large language model via OpenRouter.

Every answer I give is grounded in verified Pakistani legal provisions, including the **Constitution of Pakistan (2025 Edition)** and key legislative documents.

⚖️ **How can I assist you today?**"""

CREATOR_RESPONSE_RU = """Main **Pakistan Legal Advisor** hoon — ek AI-powered legal chatbot jo Pakistani qanoon ko sab ke liye asaan banana ke liye banaya gaya hai.

**Developer:** Nazim Hussain
**University:** Quaid-e-Awam University of Engineering, Science & Technology (QUEST), Nawabshah
**Degree:** BS Artificial Intelligence, Final Year Project — 2026

Nazim ne mujhe ek maqsad ke saath banaya: Pakistani qanoon ko aam logon, students, aur legal professionals ke liye qabil-e-faham banana. Main verified Pakistani qanooni documents — khaas tor par **Pakistan ka Aain (2025 Edition)** — se jawab deta hoon.

⚖️ **Aaj kaise madad kar sakta hoon aapki?**"""

CREATOR_RESPONSE_UR = """میں **پاکستان لیگل ایڈوائزر** ہوں — ایک AI قانونی چیٹ بوٹ جو پاکستانی قانون کو سب کے لیے قابلِ رسائی بنانے کے لیے بنایا گیا ہے۔

**ڈویلپر:** نظیم حسین
**یونیورسٹی:** قائد عوام یونیورسٹی آف انجینئرنگ، سائنس اینڈ ٹیکنالوجی (QUEST)، نوابشاہ
**پروگرام:** بی ایس آرٹیفیشل انٹیلیجنس، فائنل ایئر پروجیکٹ — 2026

⚖️ **آج میں آپ کی کیا مدد کر سکتا ہوں؟**"""

CAPABILITY_RESPONSE_EN = """Great question! Here's what I can help you with:

**⚖️ Constitutional Law**
Articles, fundamental rights, amendment history, federal vs. provincial powers.

**🏛️ Criminal Law & Procedure**
Arrest rights, detention rules, bail, remand, trial procedures, offences & penalties.

**🏠 Property & Land Law**
Acquisition, compensation, inheritance, tenancy, and property disputes.

**👨‍👩‍👧 Family Law**
Marriage (Nikah), divorce (Talaq), child custody, inheritance (Wirasat), maintenance.

**🗳️ Elections & Parliament**
Electoral laws, disqualification, National Assembly, Senate, Provincial Assemblies.

**📜 Administrative & Service Law**
Government service rules, public service commissions, tribunals.

**🌐 Languages Supported**
English · اردو (Urdu Script) · Roman Urdu

Just type your question naturally — I'll do my best to find the right legal provision for you!"""

CAPABILITY_RESPONSE_RU = """Bilkul! Main yeh sab kuch kar sakta hoon:

**⚖️ Aain (Constitution)**
Fundamental rights, articles, amendments, federal aur provincial qawaneen.

**🏛️ Criminal Law**
Giraftari ke huqooq, detention, bail, remand, muqadma, sazaen.

**🏠 Property & Zameen**
Acquisition, muawza, warasat, kiraya, zameen ke jhagray.

**👨‍👩‍👧 Family Law**
Nikah, talaq, bachon ki custody, nafaqa, wirasat.

**🗳️ Elections & Parliament**
Electoral qawaneen, disqualification, National Assembly, Senate.

**🌐 Supported Languages**
English · اردو · Roman Urdu

Bas apna sawal seedha likhein — main best possible qanooni jawab doonga!"""

THANKS_RESPONSES_EN = [
    "You're welcome! ⚖️ Feel free to ask another legal question anytime.",
    "Happy to help! If you have more questions about Pakistani law, just ask.",
    "Glad I could assist! Don't hesitate to reach out with any other legal queries.",
]

THANKS_RESPONSES_RU = [
    "Koi baat nahi! ⚖️ Koi aur qanooni sawaal ho to zaroor poochhiyega.",
    "Khushi hui madad karke! Aur koi bhi sawaal ho to bataiye.",
    "Shukriya aapka! Pakistani qanoon ke baare mein kuch bhi poochhna ho to hamesha haazir hoon.",
]

FAREWELL_RESPONSES_EN = [
    "Goodbye! ⚖️ Come back anytime you need legal guidance. Take care!",
    "Khuda Hafiz! It was a pleasure assisting you. Feel free to return for any legal queries.",
    "See you! Remember, the Pakistan Legal Advisor is always here when you need it. Goodbye!",
]

FAREWELL_RESPONSES_RU = [
    "Allah Hafiz! ⚖️ Jab bhi qanooni madad chahiye, wapas aayein.",
    "Khuda Hafiz! Bohat khushi hui aapse baat karke. Phir milenge!",
    "Alvida! Koi bhi qanooni sawaal ho to kabhi bhi wapas aayein.",
]


def check_smalltalk(text: str, lang: str):
    """
    Returns (response_string, lang) if this is a greeting/small-talk query,
    otherwise returns None. Fast path — no RAG needed.
    """
    t = text.strip()

    # ── Greetings ─────────────────────────────────────────
    if GREETING_PATTERNS_EN.match(t):
        if lang == "roman_urdu":
            return random.choice(GREETING_RESPONSES_RU), "roman_urdu"
        if lang == "ur":
            return random.choice(GREETING_RESPONSES_UR), "ur"
        return random.choice(GREETING_RESPONSES_EN), "en"

    if GREETING_PATTERNS_RU.match(t):
        return random.choice(GREETING_RESPONSES_RU), "roman_urdu"

    if GREETING_PATTERNS_UR.match(t):
        return random.choice(GREETING_RESPONSES_UR), "ur"

    # ── Creator / identity ─────────────────────────────────
    if CREATOR_PATTERNS.search(t):
        if lang == "roman_urdu":
            return CREATOR_RESPONSE_RU, "roman_urdu"
        if lang == "ur":
            return CREATOR_RESPONSE_UR, "ur"
        return CREATOR_RESPONSE_EN, "en"

    # ── Capabilities / help ────────────────────────────────
    if CAPABILITY_PATTERNS.search(t):
        if lang == "roman_urdu":
            return CAPABILITY_RESPONSE_RU, "roman_urdu"
        return CAPABILITY_RESPONSE_EN, lang

    # ── Thanks ─────────────────────────────────────────────
    if THANKS_PATTERNS.match(t):
        if lang in ("roman_urdu",):
            return random.choice(THANKS_RESPONSES_RU), "roman_urdu"
        return random.choice(THANKS_RESPONSES_EN), lang

    # ── Farewell ───────────────────────────────────────────
    if FAREWELL_PATTERNS.match(t):
        if lang in ("roman_urdu",):
            return random.choice(FAREWELL_RESPONSES_RU), "roman_urdu"
        return random.choice(FAREWELL_RESPONSES_EN), lang

    return None


# ═══════════════════════════════════════════════════════════
# ROMAN URDU DETECTION  (expanded vocabulary)
# ═══════════════════════════════════════════════════════════
ROMAN_URDU_KEYWORDS = {
    # Pronouns & personal
    "mujhe","mujh","mein","mai","main","hum","aap","ap","tum","woh","yeh","ye",
    "is","us","unka","unke","unki","apna","apne","apni","tera","teri","mera","meri",
    "hamara","hamare","hamari","inki","inke","inka","unka",

    # Verbs (common)
    "hai","hain","tha","thi","the","hoga","hogi","honge","hote","hoti","hota",
    "karo","karna","karta","karti","karte","kar","kiya","ki","karo","karen",
    "ho","hua","hui","hue","ja","jao","jana","gaya","gayi","gaye",
    "milta","milti","milte","mile","batao","bata","puchna","puchho","pucho",
    "chahiye","chahta","chahti","chahte","sakta","sakti","sakte","sakein",
    "lagta","lagti","lagte","lena","dena","deta","deti","dete","lete","leti",
    "raha","rahi","rahe","rakha","rakhi","rakhe","rakhna",
    "aana","aao","aaye","aaya","aayi","ata","aati","aate",
    "padhna","likhna","samajhna","samjhao","samjhiye","bataiye",
    "poochna","poochho","maango","maangna","lena","lene",
    "dekhna","dekho","suno","bolna","bolo","boliye","kehna","kaho",

    # Question words
    "kya","kyun","kaise","kab","kahan","kaun","kon","kitna","kitne","kitni",
    "kuch","koi","sab","sirf","hi","bhi","to","phir",

    # Legal terms
    "qanoon","adalat","haq","huqooq","zamin","zameen","mulk","desh",
    "sarkar","hakumat","police","arrest","giraftari","muqadma","maamla",
    "waqil","advocate","judge","faisla","saza","jaidad","maal","milkiyat",
    "talaq","nikah","shadi","warasat","wirasat","merath","wirsa",
    "constitution","parliament","assembly","vote","intikhaab",
    "ilzam","jurm","gunah","mutaghazzi","mujrim","be-gunah",
    "bail","zamanat","remand","hirasaat","qaid","rehayi",
    "nafaqa","alaiment","hirasat","wardship","custody",
    "zulm","insaf","mazalim","shikayat","darkhwast","iltimas",
    "appeal","sunwai","peshi","pesh","samaan","saboot","gawah",
    "kharidar","bechne","kiraya","ijara","mukaan","ghar","plot",
    "rishwat","corruption","faraib","dhoka","cheating",
    "accident","haadsa","zakhmi","nuksan","muawza",
    "naukri","mulazim","tankhwa","salary","pension","service",
    "tax","mehsool","jagir","malikana","ownership",
    "firqa","mazhab","religion","mosque","masjid","church","mandir",
    "azadi","khawateen","bachay","bache","buzurg","beemar",
    "election","naib","nazim","councillor","MPA","MNA","PM","CM",

    # Time / duration
    "ghante","ghanta","minute","din","raat","waqt","muddat","arsa",
    "baad","pehle","jald","jaldi","abhi","foran","turant","jab","jab tak",
    "kitni","kitne","muddat","mein","tak","se",

    # Connectors & modifiers
    "aur","ya","lekin","magar","phir","bhi","hi","to","par","pe",
    "ke","ka","se","tak","wala","wali","wale","nahi","nahin","mat","na",
    "bilkul","zaroor","zaruri","lazim","wajib","jaiz","najaiz",
    "theek","sahi","galat","durust","ghair","illegal","legal",
    "zyada","kam","bohat","thoda","kafi","poora","aadha",

    # Common phrases (standalone detection)
    "matlab","yani","yaani","iska","uska","matlb","yaane",
    "batao","samjhao","bataiye","samjhaiye","bata","samjha",
    "please","meherbani","kripya","shukria","shukriya",
}

def detect_roman_urdu(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return False
    matches = sum(1 for t in tokens if t in ROMAN_URDU_KEYWORDS)
    # More sensitive: 1 strong match in short queries, or ratio in longer queries
    if len(tokens) <= 3 and matches >= 1:
        return True
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
# ROMAN URDU → ENGLISH QUERY TRANSLATION  (greatly expanded)
# Maps Roman Urdu legal phrases to English equivalents so
# FAISS + BM25 can retrieve the right constitutional chunks.
# ═══════════════════════════════════════════════════════════

ROMAN_URDU_TO_ENGLISH = {
    # ── Arrest / detention ────────────────────────────────
    "giraftari":                "arrest",
    "giraftaar":                "arrested detained",
    "giraftar":                 "arrested detained",
    "hirasaat":                 "custody detention",
    "hirasat":                  "custody detention",
    "qaid":                     "imprisonment custody",
    "nazar band":               "detained house arrest",
    "band karna":               "detention imprisonment",
    "pakad liya":               "arrested detained",
    "pakad":                    "arrest apprehension",
    "harasat":                  "custody detention",
    "remand":                   "remand detention custody",
    "bail":                     "bail release",
    "zamanat":                  "bail surety",
    "riha":                     "release discharged freed",
    "chhorna":                  "release discharge",
    "rehayi":                   "release discharge bail",

    # ── Time expressions ──────────────────────────────────
    "ghante mein":              "within hours time period",
    "kitne ghante":             "how many hours within period",
    "24 ghante":                "twenty four hours 24 hours period",
    "48 ghante":                "forty eight hours 48 hours period",
    "ghante":                   "hours period time",
    "din mein":                 "within days period",
    "kitne din":                "how many days period",
    "muddat":                   "period duration time limit",
    "arsa":                     "period duration",
    "waqt":                     "time period limit",
    "jald":                     "soon immediate expeditious",
    "foran":                    "immediate forthwith",

    # ── Court / judicial ──────────────────────────────────
    "magistrate ke saamne pesh": "produced before magistrate court",
    "pesh karna":               "produced before appear court",
    "saamne pesh":              "produced before appear",
    "peshi":                    "appearance hearing court",
    "magistrate":               "magistrate court",
    "adalat":                   "court tribunal judicature",
    "sunwai":                   "hearing proceeding trial",
    "faisla":                   "judgment order decision",
    "hukum":                    "order direction decree",
    "appeal":                   "appeal appellate review",
    "nazar sani":               "review revision",
    "ijlas":                    "session sitting",
    "maqadma":                  "case proceedings lawsuit",
    "muqadma":                  "case legal proceedings lawsuit",
    "maamla":                   "matter case issue",
    "inquiry":                  "inquiry investigation",
    "tahqiqat":                 "investigation inquiry",
    "gawah":                    "witness testimony",
    "saboot":                   "evidence proof",
    "iqrar":                    "confession admission",
    "bayan":                    "statement testimony",
    "waqil":                    "advocate lawyer legal practitioner",
    "attorney":                 "attorney advocate",
    "judge":                    "judge justice court",
    "supreme court":            "supreme court chief justice",
    "high court":               "high court justice",

    # ── Fundamental rights ────────────────────────────────
    "haq":                      "right entitlement fundamental right",
    "huqooq":                   "rights fundamental rights",
    "azadi":                    "freedom liberty",
    "hurriyat":                 "freedom liberty rights",
    "insaf":                    "justice fair trial due process",
    "barabar":                  "equality equal rights",
    "tabheez":                  "discrimination equal protection",
    "boli ki azadi":            "freedom of speech expression",
    "mazhab ki azadi":          "freedom of religion",
    "ijtima ki azadi":          "freedom of assembly",
    "insan ki izzat":           "dignity of man inviolable",
    "free speech":              "freedom of speech expression article 19",

    # ── Property / land ───────────────────────────────────
    "zamin":                    "land property immovable",
    "zameen":                   "land property immovable",
    "jaidad":                   "property assets estate",
    "milkiyat":                 "ownership property right",
    "mukaan":                   "house property dwelling",
    "plot":                     "plot land property",
    "muawza":                   "compensation payment indemnity",
    "zer qabd":                 "possession acquisition",
    "qabza":                    "possession occupation",
    "kiraya":                   "rent tenancy lease",
    "ijara":                    "lease tenancy",
    "bechi":                    "sale transfer property",
    "khareed":                  "purchase acquisition",
    "usurp":                    "dispossess possession",

    # ── Family law ────────────────────────────────────────
    "talaq":                    "divorce dissolution marriage",
    "talaaq":                   "divorce dissolution marriage",
    "nikah":                    "marriage matrimonial contract",
    "shadi":                    "marriage matrimonial",
    "warasat":                  "inheritance succession",
    "wirasat":                  "inheritance succession",
    "merath":                   "inheritance legal heirs estate",
    "wirsa":                    "inheritance estate succession",
    "nafaqa":                   "maintenance alimony financial support",
    "guardianship":             "guardianship custody minor",
    "hirasat bachay":           "child custody guardianship",
    "bache ki custody":         "child custody guardianship",
    "doodh pilane":             "breastfeeding child rights",
    "mehr":                     "dower mahr marriage payment",
    "iddat":                    "iddah waiting period divorce",

    # ── Parliament / government ───────────────────────────
    "hakumat":                  "government federal government",
    "sarkar":                   "government state",
    "parliament":               "parliament majlis-e-shoora national assembly",
    "assembly":                 "assembly legislature provincial assembly",
    "senate":                   "senate upper house parliament",
    "vote":                     "vote election franchise",
    "intikhaab":                "election electoral franchise",
    "wazir-e-azam":             "prime minister chief executive",
    "PM":                       "prime minister",
    "CM":                       "chief minister province",
    "governor":                 "governor province",
    "president":                "president head of state",
    "federal":                  "federal government centre",
    "provincial":               "provincial government province",
    "naib":                     "deputy vice",
    "nazim":                    "local government head",

    # ── Crime / punishment ────────────────────────────────
    "jurm":                     "offence crime criminal",
    "gunah":                    "offence crime",
    "saza":                     "punishment sentence penalty",
    "ilzam":                    "charge accusation allegation",
    "mujrim":                   "criminal accused convict",
    "be-gunah":                 "innocent not guilty acquittal",
    "mutaghazzi":               "aggrieved complainant",
    "rishwat":                  "bribery corruption",
    "faraib":                   "fraud deceit",
    "dhoka":                    "fraud cheating",
    "zulm":                     "oppression injustice",
    "khatarnak":                "dangerous hazardous",
    "qatl":                     "murder homicide",
    "chor":                     "theft larceny",
    "chori":                    "theft larceny",
    "rape":                     "rape sexual assault zina",
    "hamla":                    "assault attack",
    "FIR":                      "FIR first information report",

    # ── Elections ─────────────────────────────────────────
    "MNA":                      "member national assembly",
    "MPA":                      "member provincial assembly",
    "election commission":      "election commission chief election commissioner",
    "polling":                  "polling voting election",
    "ballot":                   "ballot vote election",
    "candidate":                "candidate nomination election",

    # ── Employment / service ──────────────────────────────
    "naukri":                   "employment service job",
    "mulazim":                  "employee servant service",
    "tankhwa":                  "salary remuneration pay",
    "pension":                  "pension retirement benefit",
    "barkhargi":                "dismissal removal service",
    "suspend":                  "suspension service",
    "taraqqi":                  "promotion service",
    "contract":                 "contract employment",

    # ── Tax / finance ─────────────────────────────────────
    "tax":                      "tax levy duty",
    "mehsool":                  "tax revenue",
    "zakaat":                   "zakat religious tax",
    "ushr":                     "ushr agricultural tax",
    "NFC":                      "national finance commission",
    "budget":                   "budget annual budget statement",

    # ── Education ─────────────────────────────────────────
    "taleem":                   "education right to education",
    "school":                   "school educational institution",
    "university":               "university higher education",
    "free education":           "free compulsory education article 25A",

    # ── Health / environment ──────────────────────────────
    "sehat":                    "health medical",
    "mareeZ":                   "patient sick ill",
    "hospital":                 "hospital medical institution",
    "saaf mahaul":              "clean environment article 9A",
    "aaab o hawa":              "environment clean sustainable",

    # ── General legal terms ───────────────────────────────
    "shikayat":                 "complaint petition grievance",
    "darkhwast":                "application petition request",
    "iltimas":                  "petition request",
    "writ":                     "writ petition high court",
    "ittila":                   "information notice intimation",
    "wajah":                    "ground reason cause",
    "shart":                    "condition restriction",
    "paband":                   "restriction limitation",
    "ijazat":                   "permission leave license",
    "roko":                     "injunction restraint stop",
    "ban":                      "ban prohibition",
    "nafiz":                    "enforcement implementation",
    "qanoon ki roo se":         "according to law legal provision",
    "aain":                     "constitution constitutional",
    "article":                  "article provision constitutional",
    "section":                  "section provision law",
    "qanoon ki kitaab":         "statute law act",
    "IPC":                      "Pakistan penal code criminal law",
    "CrPC":                     "code of criminal procedure",
    "CPC":                      "code of civil procedure",
}

def translate_roman_urdu_query(query: str) -> str:
    """
    Converts Roman Urdu query into an English-enriched query
    suitable for FAISS + BM25 retrieval against an English legal dataset.
    Strategy: append English equivalents — preserve originals so LLM sees full context.
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
            positions = set(range(idx, idx + len(roman_phrase)))
            if not positions.intersection(matched_positions):
                matched_positions.update(positions)
                english_expansions.append(english_eq)

    if english_expansions:
        expanded = query + " " + " ".join(english_expansions)
        print(f"[TRANSLATE] Roman Urdu expansion: {' | '.join(english_expansions[:6])}")
        return expanded

    return query


# ── Legal synonym expansion (English) ────────────────────
LEGAL_SYNONYMS = {
    "land":           ["property", "immovable property", "acquisition", "article 24"],
    "compensation":   ["payment", "compulsory acquisition", "indemnity"],
    "arrest":         ["detention", "custody", "safeguards", "article 10"],
    "detention":      ["arrest", "custody", "preventive detention", "article 10"],
    "hours":          ["twenty-four hours", "24 hours", "period", "produced before magistrate"],
    "magistrate":     ["produced before magistrate", "24 hours", "article 10", "custody"],
    "produced":       ["magistrate", "24 hours", "arrest", "article 10"],
    "freedom":        ["fundamental rights", "liberty"],
    "acquire":        ["compulsory acquisition", "take possession"],
    "parliament":     ["majlis-e-shoora", "national assembly", "senate"],
    "court":          ["judicature", "high court", "supreme court"],
    "equality":       ["equal protection", "non-discrimination", "article 25"],
    "education":      ["right to education", "article 25a", "free compulsory"],
    "religion":       ["freedom of religion", "article 20"],
    "speech":         ["freedom of speech", "article 19"],
    "fair trial":     ["article 10a", "due process", "right to fair trial"],
    "bail":           ["bail", "release", "detention", "article 10"],
    "inheritance":    ["succession", "legal heirs", "estate", "property"],
    "divorce":        ["dissolution of marriage", "family law", "matrimonial"],
    "marriage":       ["nikah", "matrimonial", "family law"],
    "property":       ["immovable property", "acquisition", "article 24"],
    "punishment":     ["sentence", "penalty", "offence", "criminal"],
    "election":       ["electoral", "franchise", "voting rights", "article 51"],
    "president":      ["head of state", "article 41", "article 48"],
    "prime minister": ["chief executive", "article 91", "cabinet"],
    "environment":    ["clean environment", "article 9A", "sustainable"],
    "torture":        ["dignity", "article 14", "no torture"],
    "slavery":        ["forced labour", "article 11", "prohibited"],
    "assembly":       ["freedom of assembly", "article 16", "peaceful"],
    "association":    ["freedom of association", "article 17", "political party"],
    "information":    ["right to information", "article 19A", "public importance"],
    "trade":          ["freedom of trade", "article 18", "business profession"],
    "movement":       ["freedom of movement", "article 15", "reside settle"],
    "discrimination": ["non-discrimination", "article 25", "article 26", "article 27"],
    "language":       ["national language", "article 251", "urdu", "provincial language"],
    "emergency":      ["proclamation of emergency", "article 232", "article 233"],
    "amendment":      ["constitution amendment", "article 238", "article 239"],
}

def expand_query(query: str) -> str:
    expanded = query
    q_lower  = query.lower()
    added = set()
    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in q_lower and term not in added:
            expanded += " " + " ".join(synonyms[:2])
            added.add(term)
    art_match = re.search(r'article\s+(\d+[A-Za-z]*)', query, re.IGNORECASE)
    if art_match:
        expanded += f" Article {art_match.group(1)} constitution Pakistan law provision"
    return expanded


# ── Hybrid retrieval ─────────────────────────────────────
def hybrid_retrieve(query: str, lang: str, k: int = TOP_K) -> list:
    """
    For Roman Urdu queries: translate to English first, then retrieve.
    For all queries: synonym expansion + hybrid BM25 + vector search.
    """
    if lang == "roman_urdu":
        retrieval_query = translate_roman_urdu_query(query)
    else:
        retrieval_query = query

    expanded = expand_query(retrieval_query)
    print(f"[EXPANDED] {expanded[:120]}...")

    q_vec = embedder.encode([expanded], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    vec_scores, vec_indices = index.search(q_vec, min(k * 2, len(chunks)))

    rrf_scores = {}
    valid_vec_indices = []
    for rank, (score, idx) in enumerate(zip(vec_scores[0], vec_indices[0])):
        if idx < len(chunks) and score >= MIN_SCORE:
            valid_vec_indices.append(int(idx))
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (60 + rank + 1)

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
    """Returns (system_message, user_prompt) tuned for detected language."""

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

        # ── Fast path: small-talk / greetings / meta queries ──
        smalltalk = check_smalltalk(query, lang)
        if smalltalk:
            print(f"[SMALLTALK] Matched — returning canned response")
            return smalltalk  # already a (text, lang) tuple

        # ── RAG path ──────────────────────────────────────────
        candidates = hybrid_retrieve(query, lang=lang, k=TOP_K)
        print(f"[RETRIEVE] {len(candidates)} candidates found")

        if not candidates:
            no_result = {
                "roman_urdu": "Is sawal ka jawab dataset mein nahi mila. Meherbani kar ke alag alfaz mein puchiye.",
                "ur":         "اس سوال کا جواب ڈیٹاسیٹ میں نہیں ملا۔ براہ کرم مختلف الفاظ میں پوچھیں۔",
            }
            return no_result.get(lang, "No relevant legal provisions found for this query."), lang

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
    print("\n" + "="*60)
    print("  Pakistan Legal RAG — Roman Urdu + English + Urdu")
    print("  Developer: Nazim Hussain | QUEST University, Nawabshah")
    print("="*60)
    print(f"  Chunks   : {len(chunks)}")
    print(f"  BM25     : {USE_BM25}")
    print(f"  Reranker : {USE_RERANKER}")
    print(f"  Model    : {MODEL_NAME}")
    print("="*60 + "\n")
    print("[READY] Flask is starting on port 7860...")
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        use_reloader=False,
        threaded=True
    )