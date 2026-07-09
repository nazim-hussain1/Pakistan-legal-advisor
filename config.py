"""
Central configuration for the Pakistan Legal Advisor backend.

All tunable constants (RAG parameters, model names, DB URI, etc.) live
here so the rest of the codebase never hard-codes them.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ── Flask / sessions ─────────────────────────────────
    PREFERRED_URL_SCHEME = "https"
    PERMANENT_SESSION_LIFETIME_DAYS = 30

    # ── Database ──────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'pla_users.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── RAG / retrieval ───────────────────────────────────
    DATASET_PATH = "fyp_cleaned_dataset.csv"
    INDEX_FILE   = "faiss_index.bin"
    CHUNKS_FILE  = "chunks.npy"
    TOP_K        = 20      # candidates pulled from FAISS / BM25 before fusion
    RERANK_TOP   = 5       # final passages sent to the LLM
    MAX_TOKENS   = 900
    MIN_SCORE    = 0.16    # min FAISS inner-product score to admit a candidate
    RRF_K        = 60      # Reciprocal Rank Fusion constant

    EMBEDDING_MODEL          = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    EMBEDDING_MODEL_FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL            = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM (OpenRouter) ──────────────────────────────────
    OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL_NAME           = os.getenv("MODEL_NAME", "gpt-oss-120b")

    # ── Conversation memory ───────────────────────────────
    # Number of previous messages (user + assistant turns combined)
    # kept as short-term context for follow-up questions.
    MAX_MEMORY_MESSAGES   = 6
    MEMORY_TRUNCATE_CHARS = 500   # cap per stored message to keep the session cookie small
