import os
from dotenv import load_dotenv

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))   # app folder
BASE_DIR = os.path.dirname(APP_DIR)                    # project root
DATA_DIR = os.path.join(BASE_DIR, "data")


class Config:
    # ── Flask / sessions ─────────────────────────────────
    PREFERRED_URL_SCHEME = "https"
    PERMANENT_SESSION_LIFETIME_DAYS = 30

    # ── Database ──────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(DATA_DIR, 'pla_users.db')}"
)
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ── RAG / retrieval ───────────────────────────────────
    DATASET_PATH = os.path.join(DATA_DIR, "fyp_cleaned_dataset.csv")
    INDEX_FILE   = os.path.join(DATA_DIR, "faiss_index.bin")
    CHUNKS_FILE  = os.path.join(DATA_DIR, "chunks.npy")
    TOP_K        = 20      # candidates pulled from FAISS / BM25 before fusion
    RERANK_TOP   = 5       # final passages sent to the LLM
    MAX_TOKENS   = 900
    MIN_SCORE    = 0.16    # min FAISS inner-product score to admit a candidate
    RRF_K        = 60      # Reciprocal Rank Fusion constant

    EMBEDDING_MODEL          = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    EMBEDDING_MODEL_FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL            = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM  ──────────────────────────────────
    GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
    MODEL_NAME           = os.getenv("MODEL_NAME", "gemini-flash-lite-latest")
    FALLBACK_MODEL_NAME  = os.getenv("FALLBACK_MODEL_NAME", "gemini-2.0-flash")

    # ── Conversation memory ───────────────────────────────
    # Number of previous messages (user + assistant turns combined)
    # kept as short-term context for follow-up questions.
    MAX_MEMORY_MESSAGES   = 6
    MEMORY_TRUNCATE_CHARS = 500   # cap per stored message to keep the session cookie small
