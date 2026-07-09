"""
Pakistan Legal Advisor — Flask application entry point.

This file is intentionally thin: it wires together the app factory,
database, and blueprints. All RAG/query logic lives in dedicated
modules (retrieval.py, rag.py, prompts.py, translation.py,
language_detection.py, smalltalk.py, memory.py, llm_client.py) and all
routes live in their own blueprint files (auth.py, chat_routes.py,
user_routes.py).

Google OAuth has been removed entirely — email/password is the only
authentication method.
"""
import os
import secrets
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

from flask import Flask, render_template
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from extensions import db

# Loads the dataset, builds/loads the FAISS + BM25 indices, and loads
# the embedding + reranker models. Imported explicitly (and early) so
# all heavy startup work happens once, before Flask starts accepting
# requests — mirrors the original monolithic script's behavior.
import retrieval


def create_app() -> Flask:
    app = Flask(__name__, template_folder="Templates")
    app.config["PREFERRED_URL_SCHEME"] = "https"
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    # Session signing key — prefers the HF Space secret APP_SECRET_KEY,
    # falls back to FLASK_SECRET_KEY, then a random per-process key.
    app.secret_key = (
        os.getenv("APP_SECRET_KEY")
        or os.getenv("FLASK_SECRET_KEY")
        or secrets.token_hex(32)
    )
    app.permanent_session_lifetime = timedelta(days=Config.PERMANENT_SESSION_LIFETIME_DAYS)

    app.config["SQLALCHEMY_DATABASE_URI"] = Config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)

    # Blueprints
    from auth import auth_bp
    from chat_routes import chat_bp
    from user_routes import user_bp

    app.register_blueprint(auth_bp)   # /auth/*
    app.register_blueprint(chat_bp)   # /chat, /chat/new
    app.register_blueprint(user_bp)   # /history*, /settings, /account

    @app.route("/")
    def home():
        return render_template("Frontend.html")

    @app.route("/ping")
    def ping():
        return "ok", 200

    @app.route("/health")
    def health():
        return {
            "status":     "ok",
            "chunks":     len(retrieval.chunks),
            "bm25":       retrieval.USE_BM25,
            "reranker":   retrieval.USE_RERANKER,
            "index_type": type(retrieval.index).__name__,
            "db":         "sqlite",
        }

    return app


app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("[OK] Database tables ready")

    print("\n" + "=" * 60)
    print("  Pakistan Legal RAG — Roman Urdu + English + Urdu")
    print("  Developer: Nazim Hussain | QUEST University, Nawabshah")
    print("=" * 60)
    print(f"  Chunks        : {len(retrieval.chunks)}")
    print(f"  BM25          : {retrieval.USE_BM25}")
    print(f"  Reranker      : {retrieval.USE_RERANKER}")
    print(f"  Model         : {Config.MODEL_NAME}")
    print(f"  Auth DB       : pla_users.db (SQLite)")
    print(f"  Memory        : last {Config.MAX_MEMORY_MESSAGES} messages per browser session")
    print("=" * 60 + "\n")
    print("[READY] Flask is starting on port 7860...")
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        use_reloader=False,
        threaded=True
    )
