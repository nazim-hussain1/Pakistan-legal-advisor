import os
import re
import random
import warnings
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from functools import wraps

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

from flask import (
    Flask, jsonify, render_template, request,
    session, redirect, url_for, abort
)
from flask_sqlalchemy import SQLAlchemy
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

# ── Google OAuth ──────────────────────────────────────────
try:
    from authlib.integrations.flask_client import OAuth
    OAUTH_AVAILABLE = True
    print("[OK] authlib loaded")
except ImportError:
    OAUTH_AVAILABLE = False
    print("[WARN] authlib not installed — Google OAuth disabled. Run: pip install authlib")

load_dotenv()

# ═══════════════════════════════════════════════════════════
# APP & DB SETUP
# ═══════════════════════════════════════════════════════════

app = Flask(__name__, template_folder='Templates')

app.config['PREFERRED_URL_SCHEME'] = 'https'

from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Secret key — generate a strong one or set via env
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
app.permanent_session_lifetime = timedelta(days=30)

# SQLite database (file-based, zero configuration)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = (
    os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'pla_users.db')}")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ═══════════════════════════════════════════════════════════
# DATABASE MODELS
# ═══════════════════════════════════════════════════════════

class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=True)   # NULL for OAuth users
    google_id     = db.Column(db.String(200), unique=True, nullable=True)
    avatar_url    = db.Column(db.String(500), nullable=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    last_login    = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    chats         = db.relationship("ChatSession", backref="user", lazy=True, cascade="all, delete-orphan")
    settings      = db.relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")

    def set_password(self, password: str):
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        self.password_hash = f"{salt}:{hashed}"

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        try:
            salt, hashed = self.password_hash.split(":", 1)
            return hashlib.sha256(f"{salt}{password}".encode()).hexdigest() == hashed
        except Exception:
            return False

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "email":      self.email,
            "avatar_url": self.avatar_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class ChatSession(db.Model):
    __tablename__ = "chat_sessions"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    session_id = db.Column(db.String(64), unique=True, nullable=False, default=lambda: secrets.token_hex(16))
    title      = db.Column(db.String(200), default="New Chat")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages   = db.relationship("ChatMessage", backref="chat_session", lazy=True,
                                  cascade="all, delete-orphan", order_by="ChatMessage.created_at")

    def to_dict(self, include_messages=False):
        d = {
            "id":         self.id,
            "session_id": self.session_id,
            "title":      self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages),
        }
        if include_messages:
            d["messages"] = [m.to_dict() for m in self.messages]
        return d


class ChatMessage(db.Model):
    __tablename__ = "chat_messages"
    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False)
    role       = db.Column(db.String(10), nullable=False)   # "user" | "assistant"
    content    = db.Column(db.Text, nullable=False)
    language   = db.Column(db.String(20), default="en")
    status     = db.Column(db.String(20), default="ok")     # "ok" | "error"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id":         self.id,
            "role":       self.role,
            "content":    self.content,
            "language":   self.language,
            "status":     self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserSettings(db.Model):
    __tablename__ = "user_settings"
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    theme           = db.Column(db.String(10), default="dark")          # "dark" | "light"
    language_pref   = db.Column(db.String(10), default="en")            # "en" | "ur"
    font_size       = db.Column(db.Integer, default=14)
    compact_mode    = db.Column(db.Boolean, default=False)
    markdown_render = db.Column(db.Boolean, default=True)
    typing_anim     = db.Column(db.Boolean, default=True)
    save_history    = db.Column(db.Boolean, default=True)
    analytics       = db.Column(db.Boolean, default=False)
    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "theme":           self.theme,
            "language_pref":   self.language_pref,
            "font_size":       self.font_size,
            "compact_mode":    self.compact_mode,
            "markdown_render": self.markdown_render,
            "typing_anim":     self.typing_anim,
            "save_history":    self.save_history,
            "analytics":       self.analytics,
        }


# ═══════════════════════════════════════════════════════════
# GOOGLE OAUTH SETUP
# ═══════════════════════════════════════════════════════════

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

if OAUTH_AVAILABLE and GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth = OAuth(app)
    google = oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    GOOGLE_OAUTH_ENABLED = True
    print("[OK] Google OAuth configured")
else:
    GOOGLE_OAUTH_ENABLED = False
    print("[WARN] Google OAuth not configured — set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env")


# ═══════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════

def login_required(f):
    """Decorator — returns 401 JSON if not logged in."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required", "code": "UNAUTHENTICATED"}), 401
        return f(*args, **kwargs)
    return decorated


def get_current_user() -> "User | None":
    uid = session.get("user_id")
    if not uid:
        return None
    return db.session.get(User, uid)


def _login_user(user: User):
    """Write user into Flask session."""
    session.permanent = True
    session["user_id"]    = user.id
    session["user_name"]  = user.name
    session["user_email"] = user.email
    user.last_login = datetime.utcnow()
    db.session.commit()


def _ensure_settings(user: User):
    """Create default settings row if missing."""
    if not user.settings:
        db.session.add(UserSettings(user_id=user.id))
        db.session.commit()


# ═══════════════════════════════════════════════════════════
# RAG CONFIG
# ═══════════════════════════════════════════════════════════

file_path   = "fyp_cleaned_dataset.csv"
INDEX_FILE  = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
TOP_K       = 20
RERANK_TOP  = 5
MAX_TOKENS  = 900
MIN_SCORE   = 0.16

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")


# ═══════════════════════════════════════════════════════════
# GREETING & SMALL-TALK DETECTION
# ═══════════════════════════════════════════════════════════

GREETING_PATTERNS_EN = re.compile(
    r"^\s*(hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening|night|day)|"
    r"salaam|salam|assalam|assalamu|what'?s?\s*up|sup|yo|hiya|heya|namaste|"
    r"how\s*are\s*(you|u)|how'?s?\s*(it\s*going|everything|life)|"
    r"hope\s*you'?re?\s*(well|good|fine)|nice\s*to\s*(meet|see)\s*you|"
    r"pleased\s*to\s*(meet|see)\s*you)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

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

GREETING_PATTERNS_UR = re.compile(
    r"^\s*(السلام|سلام|آداب|ہیلو|ہائے|صبح\s*بخیر|شام\s*بخیر|کیسے\s*ہیں|کیا\s*حال)\s*.*$"
)

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

THANKS_PATTERNS = re.compile(
    r"^\s*(thanks|thank\s*you|thank\s*u|thx|ty|shukriya|shukriyah|meherbani|"
    r"bohat\s*shukriya|bahut\s*shukriya|bahut\s*meherbani|jazak\s*allah|"
    r"جزاک\s*اللہ|شکریہ|بہت\s*شکریہ)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

FAREWELL_PATTERNS = re.compile(
    r"^\s*(bye|goodbye|good\s*bye|see\s*you|later|take\s*care|"
    r"khuda\s*hafiz|allah\s*hafiz|alvida|phir\s*milenge|"
    r"خدا\s*حافظ|اللہ\s*حافظ|الوداع)\s*[!?.,]*\s*$",
    re.IGNORECASE
)

# ── Canned responses (identical to original) ─────────────

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
    t = text.strip()
    if GREETING_PATTERNS_EN.match(t):
        if lang == "roman_urdu": return random.choice(GREETING_RESPONSES_RU), "roman_urdu"
        if lang == "ur":         return random.choice(GREETING_RESPONSES_UR), "ur"
        return random.choice(GREETING_RESPONSES_EN), "en"
    if GREETING_PATTERNS_RU.match(t):
        return random.choice(GREETING_RESPONSES_RU), "roman_urdu"
    if GREETING_PATTERNS_UR.match(t):
        return random.choice(GREETING_RESPONSES_UR), "ur"
    if CREATOR_PATTERNS.search(t):
        if lang == "roman_urdu": return CREATOR_RESPONSE_RU, "roman_urdu"
        if lang == "ur":         return CREATOR_RESPONSE_UR, "ur"
        return CREATOR_RESPONSE_EN, "en"
    if CAPABILITY_PATTERNS.search(t):
        if lang == "roman_urdu": return CAPABILITY_RESPONSE_RU, "roman_urdu"
        return CAPABILITY_RESPONSE_EN, lang
    if THANKS_PATTERNS.match(t):
        if lang == "roman_urdu": return random.choice(THANKS_RESPONSES_RU), "roman_urdu"
        return random.choice(THANKS_RESPONSES_EN), lang
    if FAREWELL_PATTERNS.match(t):
        if lang == "roman_urdu": return random.choice(FAREWELL_RESPONSES_RU), "roman_urdu"
        return random.choice(FAREWELL_RESPONSES_EN), lang
    return None


# ═══════════════════════════════════════════════════════════
# ROMAN URDU DETECTION
# ═══════════════════════════════════════════════════════════

ROMAN_URDU_KEYWORDS = {
    "mujhe","mujh","mein","mai","main","hum","aap","ap","tum","woh","yeh","ye",
    "is","us","unka","unke","unki","apna","apne","apni","tera","teri","mera","meri",
    "hamara","hamare","hamari","inki","inke","inka","unka",
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
    "kya","kyun","kaise","kab","kahan","kaun","kon","kitna","kitne","kitni",
    "kuch","koi","sab","sirf","hi","bhi","to","phir",
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
    "ghante","ghanta","minute","din","raat","waqt","muddat","arsa",
    "baad","pehle","jald","jaldi","abhi","foran","turant","jab","jab tak",
    "kitni","kitne","muddat","mein","tak","se",
    "aur","ya","lekin","magar","phir","bhi","hi","to","par","pe",
    "ke","ka","se","tak","wala","wali","wale","nahi","nahin","mat","na",
    "bilkul","zaroor","zaruri","lazim","wajib","jaiz","najaiz",
    "theek","sahi","galat","durust","ghair","illegal","legal",
    "zyada","kam","bohat","thoda","kafi","poora","aadha",
    "matlab","yani","yaani","iska","uska","matlb","yaane",
    "batao","samjhao","bataiye","samjhaiye","bata","samjha",
    "please","meherbani","kripya","shukria","shukriya",
}

def detect_roman_urdu(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens: return False
    matches = sum(1 for t in tokens if t in ROMAN_URDU_KEYWORDS)
    if len(tokens) <= 3 and matches >= 1: return True
    if matches >= 2: return True
    if len(tokens) >= 4 and (matches / len(tokens)) >= 0.15: return True
    return False

def detect_language(text: str) -> str:
    if re.search(r'[\u0600-\u06FF]', text): return "ur"
    if detect_roman_urdu(text): return "roman_urdu"
    try: return detect(text)
    except Exception: return "en"


# ═══════════════════════════════════════════════════════════
# DATASET & VECTOR INDEX
# ═══════════════════════════════════════════════════════════

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
        chunk_size=1400, chunk_overlap=300,
        separators=["\nArticle ", "\nPart ", "\nChapter ",
                    "\n\n", "\n   (", "\n  (", "\n (",
                    "\n", " ", ""]
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
# ═══════════════════════════════════════════════════════════

ROMAN_URDU_TO_ENGLISH = {
    "giraftari": "arrest", "giraftaar": "arrested detained",
    "giraftar": "arrested detained", "hirasaat": "custody detention",
    "hirasat": "custody detention", "qaid": "imprisonment custody",
    "nazar band": "detained house arrest", "band karna": "detention imprisonment",
    "pakad liya": "arrested detained", "pakad": "arrest apprehension",
    "harasat": "custody detention", "remand": "remand detention custody",
    "bail": "bail release", "zamanat": "bail surety",
    "riha": "release discharged freed", "chhorna": "release discharge",
    "rehayi": "release discharge bail",
    "ghante mein": "within hours time period",
    "kitne ghante": "how many hours within period",
    "24 ghante": "twenty four hours 24 hours period",
    "48 ghante": "forty eight hours 48 hours period",
    "ghante": "hours period time", "din mein": "within days period",
    "kitne din": "how many days period", "muddat": "period duration time limit",
    "arsa": "period duration", "waqt": "time period limit",
    "magistrate ke saamne pesh": "produced before magistrate court",
    "pesh karna": "produced before appear court",
    "saamne pesh": "produced before appear", "peshi": "appearance hearing court",
    "magistrate": "magistrate court", "adalat": "court tribunal judicature",
    "sunwai": "hearing proceeding trial", "faisla": "judgment order decision",
    "hukum": "order direction decree", "appeal": "appeal appellate review",
    "nazar sani": "review revision", "ijlas": "session sitting",
    "maqadma": "case proceedings lawsuit", "muqadma": "case legal proceedings lawsuit",
    "maamla": "matter case issue", "inquiry": "inquiry investigation",
    "tahqiqat": "investigation inquiry", "gawah": "witness testimony",
    "saboot": "evidence proof", "iqrar": "confession admission",
    "bayan": "statement testimony", "waqil": "advocate lawyer legal practitioner",
    "judge": "judge justice court", "supreme court": "supreme court chief justice",
    "high court": "high court justice",
    "haq": "right entitlement fundamental right",
    "huqooq": "rights fundamental rights", "azadi": "freedom liberty",
    "hurriyat": "freedom liberty rights", "insaf": "justice fair trial due process",
    "barabar": "equality equal rights", "tabheez": "discrimination equal protection",
    "boli ki azadi": "freedom of speech expression",
    "mazhab ki azadi": "freedom of religion",
    "ijtima ki azadi": "freedom of assembly",
    "insan ki izzat": "dignity of man inviolable",
    "free speech": "freedom of speech expression article 19",
    "zamin": "land property immovable", "zameen": "land property immovable",
    "jaidad": "property assets estate", "milkiyat": "ownership property right",
    "mukaan": "house property dwelling", "plot": "plot land property",
    "muawza": "compensation payment indemnity", "zer qabd": "possession acquisition",
    "qabza": "possession occupation", "kiraya": "rent tenancy lease",
    "ijara": "lease tenancy", "bechi": "sale transfer property",
    "khareed": "purchase acquisition",
    "talaq": "divorce dissolution marriage", "talaaq": "divorce dissolution marriage",
    "nikah": "marriage matrimonial contract", "shadi": "marriage matrimonial",
    "warasat": "inheritance succession", "wirasat": "inheritance succession",
    "merath": "inheritance legal heirs estate", "wirsa": "inheritance estate succession",
    "nafaqa": "maintenance alimony financial support",
    "hirasat bachay": "child custody guardianship",
    "bache ki custody": "child custody guardianship",
    "mehr": "dower mahr marriage payment", "iddat": "iddah waiting period divorce",
    "hakumat": "government federal government", "sarkar": "government state",
    "parliament": "parliament majlis-e-shoora national assembly",
    "assembly": "assembly legislature provincial assembly",
    "senate": "senate upper house parliament",
    "vote": "vote election franchise", "intikhaab": "election electoral franchise",
    "wazir-e-azam": "prime minister chief executive", "PM": "prime minister",
    "CM": "chief minister province", "governor": "governor province",
    "president": "president head of state",
    "jurm": "offence crime criminal", "gunah": "offence crime",
    "saza": "punishment sentence penalty", "ilzam": "charge accusation allegation",
    "mujrim": "criminal accused convict", "be-gunah": "innocent not guilty acquittal",
    "rishwat": "bribery corruption", "faraib": "fraud deceit",
    "qatl": "murder homicide", "chori": "theft larceny", "FIR": "FIR first information report",
    "naukri": "employment service job", "mulazim": "employee servant service",
    "tankhwa": "salary remuneration pay", "pension": "pension retirement benefit",
    "tax": "tax levy duty", "mehsool": "tax revenue",
    "taleem": "education right to education",
    "free education": "free compulsory education article 25A",
    "aain": "constitution constitutional", "article": "article provision constitutional",
    "section": "section provision law", "qanoon ki roo se": "according to law legal provision",
}

def translate_roman_urdu_query(query: str) -> str:
    q_lower = query.lower()
    english_expansions = []
    sorted_mappings = sorted(ROMAN_URDU_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True)
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


# ═══════════════════════════════════════════════════════════
# HYBRID RETRIEVAL + RERANKING
# ═══════════════════════════════════════════════════════════

def hybrid_retrieve(query: str, lang: str, k: int = TOP_K) -> list:
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


def rerank(query: str, candidates: list, top_n: int = RERANK_TOP) -> list:
    if not candidates: return []
    if not USE_RERANKER or len(candidates) <= top_n: return candidates[:top_n]
    try:
        pairs  = [(query, c) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [text for _, text in ranked[:top_n]]
    except Exception as e:
        print(f"[WARN] Reranker error: {e}")
        return candidates[:top_n]

def assemble_context(top_chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Provision {i+1}]\n{chunk}" for i, chunk in enumerate(top_chunks)
    )


# ═══════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════

def build_prompt(query: str, context: str, lang: str) -> tuple:
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


# ═══════════════════════════════════════════════════════════
# RAG CORE
# ═══════════════════════════════════════════════════════════

def rag_query(query: str) -> tuple:
    """Returns (answer_string, detected_language_string)."""
    try:
        print(f"\n[QUERY] {query}")
        lang = detect_language(query)
        print(f"[LANG]  Detected: {lang}")
        smalltalk = check_smalltalk(query, lang)
        if smalltalk:
            print("[SMALLTALK] Matched — returning canned response")
            return smalltalk
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
        return f"System error: {str(e) or 'Unknown error'}", "en"


# ═══════════════════════════════════════════════════════════
# ── ROUTES ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("Frontend.html")


# ── Core chat ────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400

        answer, lang = rag_query(message)

        # Persist to DB if user is logged in and has save_history on
        user = get_current_user()
        if user:
            _ensure_settings(user)
            if user.settings.save_history:
                chat_session_id = data.get("session_id")
                chat_sess = None
                if chat_session_id:
                    chat_sess = ChatSession.query.filter_by(
                        session_id=chat_session_id, user_id=user.id
                    ).first()
                if not chat_sess:
                    chat_sess = ChatSession(
                        user_id=user.id,
                        title=message[:60] + ("…" if len(message) > 60 else "")
                    )
                    db.session.add(chat_sess)
                    db.session.flush()

                db.session.add(ChatMessage(
                    session_id=chat_sess.id, role="user",
                    content=message, language=lang
                ))
                db.session.add(ChatMessage(
                    session_id=chat_sess.id, role="assistant",
                    content=answer, language=lang, status="ok"
                ))
                chat_sess.updated_at = datetime.utcnow()
                db.session.commit()

                return jsonify({
                    "reply":      answer,
                    "language":   lang,
                    "session_id": chat_sess.session_id,
                })

        return jsonify({"reply": answer, "language": lang})

    except Exception as e:
        import traceback
        print(f"[ERROR] Chat route:\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ── Auth — Email/Password ────────────────────────────────

@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.flush()
    db.session.add(UserSettings(user_id=user.id))
    db.session.commit()

    _login_user(user)
    return jsonify({"message": "Account created successfully", "user": user.to_dict()}), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    _login_user(user)
    _ensure_settings(user)
    return jsonify({"message": "Login successful", "user": user.to_dict()})


@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})


@app.route("/auth/me", methods=["GET"])
def me():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not authenticated", "code": "UNAUTHENTICATED"}), 401
    _ensure_settings(user)
    return jsonify({
        "user":     user.to_dict(),
        "settings": user.settings.to_dict() if user.settings else {}
    })


# ── Auth — Google OAuth ───────────────────────────────────

@app.route("/auth/google")
def google_login():
    if not GOOGLE_OAUTH_ENABLED:
        return jsonify({"error": "Google OAuth is not configured on this server"}), 503
    redirect_uri = "https://nazhussain-pakistan-legal-advisor.hf.space/auth/google/callback"
    return google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    if not GOOGLE_OAUTH_ENABLED:
        return redirect("/?error=oauth_not_configured")
    try:
        token     = google.authorize_access_token()
        user_info = token.get("userinfo")
        if not user_info:
            import httpx
            resp      = httpx.get("https://openidconnect.googleapis.com/v1/userinfo",
                                  headers={"Authorization": f"Bearer {token['access_token']}"})
            user_info = resp.json()

        google_id  = user_info["sub"]
        email      = user_info.get("email", "").lower()
        name       = user_info.get("name", email.split("@")[0])
        avatar_url = user_info.get("picture", "")

        # Look up or create the user
        user = User.query.filter_by(google_id=google_id).first()
        if not user:
            user = User.query.filter_by(email=email).first()
        if not user:
            user = User(name=name, email=email, google_id=google_id, avatar_url=avatar_url)
            db.session.add(user)
            db.session.flush()
            db.session.add(UserSettings(user_id=user.id))
        else:
            user.google_id  = google_id
            user.avatar_url = avatar_url
            if not user.settings:
                db.session.add(UserSettings(user_id=user.id))
        db.session.commit()

        _login_user(user)
        return redirect("/?login=success")

    except Exception as e:
        import traceback
        print(f"[ERROR] Google OAuth callback:\n{traceback.format_exc()}")
        return redirect(f"/?error=oauth_failed&msg={str(e)[:80]}")


# ── Chat history ─────────────────────────────────────────

@app.route("/history", methods=["GET"])
@login_required
def get_history():
    user = get_current_user()
    sessions = (ChatSession.query
                .filter_by(user_id=user.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(100)
                .all())
    return jsonify({"sessions": [s.to_dict() for s in sessions]})


@app.route("/history/<session_id>", methods=["GET"])
@login_required
def get_session(session_id):
    user = get_current_user()
    chat_sess = ChatSession.query.filter_by(
        session_id=session_id, user_id=user.id
    ).first()
    if not chat_sess:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"session": chat_sess.to_dict(include_messages=True)})


@app.route("/history/<session_id>", methods=["DELETE"])
@login_required
def delete_session(session_id):
    user = get_current_user()
    chat_sess = ChatSession.query.filter_by(
        session_id=session_id, user_id=user.id
    ).first()
    if not chat_sess:
        return jsonify({"error": "Session not found"}), 404
    db.session.delete(chat_sess)
    db.session.commit()
    return jsonify({"message": "Session deleted"})


@app.route("/history/clear", methods=["DELETE"])
@login_required
def clear_history():
    user = get_current_user()
    ChatSession.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    return jsonify({"message": "All history cleared"})


# ── Settings ─────────────────────────────────────────────

@app.route("/settings", methods=["GET"])
@login_required
def get_settings():
    user = get_current_user()
    _ensure_settings(user)
    return jsonify({"settings": user.settings.to_dict()})


@app.route("/settings", methods=["PUT"])
@login_required
def update_settings():
    user = get_current_user()
    _ensure_settings(user)
    data = request.get_json(silent=True) or {}
    s = user.settings

    if "theme"           in data: s.theme           = data["theme"]
    if "language_pref"   in data: s.language_pref   = data["language_pref"]
    if "font_size"       in data: s.font_size        = int(data["font_size"])
    if "compact_mode"    in data: s.compact_mode     = bool(data["compact_mode"])
    if "markdown_render" in data: s.markdown_render  = bool(data["markdown_render"])
    if "typing_anim"     in data: s.typing_anim      = bool(data["typing_anim"])
    if "save_history"    in data: s.save_history     = bool(data["save_history"])
    if "analytics"       in data: s.analytics        = bool(data["analytics"])

    db.session.commit()
    return jsonify({"message": "Settings saved", "settings": s.to_dict()})


# ── Account ──────────────────────────────────────────────

@app.route("/account", methods=["PUT"])
@login_required
def update_account():
    user = get_current_user()
    data = request.get_json(silent=True) or {}
    if "name"  in data: user.name  = data["name"].strip()
    if "email" in data:
        new_email = data["email"].strip().lower()
        existing  = User.query.filter_by(email=new_email).first()
        if existing and existing.id != user.id:
            return jsonify({"error": "Email already in use"}), 409
        user.email = new_email
    if "password" in data:
        if len(data["password"]) < 8:
            return jsonify({"error": "Password must be at least 8 characters"}), 400
        user.set_password(data["password"])
    db.session.commit()
    return jsonify({"message": "Profile updated", "user": user.to_dict()})


@app.route("/account", methods=["DELETE"])
@login_required
def delete_account():
    user = get_current_user()
    db.session.delete(user)
    db.session.commit()
    session.clear()
    return jsonify({"message": "Account deleted"})


# ── Health / ping ────────────────────────────────────────

@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/health")
def health():
    return jsonify({
        "status":          "ok",
        "chunks":          len(chunks),
        "bm25":            USE_BM25,
        "reranker":        USE_RERANKER,
        "index_type":      type(index).__name__,
        "google_oauth":    GOOGLE_OAUTH_ENABLED,
        "db":              "sqlite",
    })


# ═══════════════════════════════════════════════════════════
# STARTUP — create DB tables, then run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("[OK] Database tables ready")

    print("\n" + "="*60)
    print("  Pakistan Legal RAG — Roman Urdu + English + Urdu")
    print("  Developer: Nazim Hussain | QUEST University, Nawabshah")
    print("="*60)
    print(f"  Chunks        : {len(chunks)}")
    print(f"  BM25          : {USE_BM25}")
    print(f"  Reranker      : {USE_RERANKER}")
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Google OAuth  : {GOOGLE_OAUTH_ENABLED}")
    print(f"  Auth DB       : pla_users.db (SQLite)")
    print("="*60 + "\n")
    print("[READY] Flask is starting on port 7860...")
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        use_reloader=False,
        threaded=True
    )