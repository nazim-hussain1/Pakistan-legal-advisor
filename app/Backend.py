import os
import secrets
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

from flask import Flask, render_template
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import Config
from app.extensions import db

import core.retrieval as retrieval


def create_app() -> Flask:
    template_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates"
    )
    app = Flask(__name__, template_folder=template_dir)
    app.config["PREFERRED_URL_SCHEME"] = "https"
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    app.secret_key = (
        os.getenv("APP_SECRET_KEY")
        or os.getenv("FLASK_SECRET_KEY")
        or secrets.token_hex(32)
    )
    app.permanent_session_lifetime = timedelta(days=Config.PERMANENT_SESSION_LIFETIME_DAYS)

    app.config["SQLALCHEMY_DATABASE_URI"] = Config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    db.init_app(app)

    from app.auth import auth_bp
    from app.chat_routes import chat_bp
    from app.user_routes import user_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(user_bp)

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