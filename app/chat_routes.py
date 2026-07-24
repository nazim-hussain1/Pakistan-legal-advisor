from datetime import datetime

from flask import Blueprint, jsonify, request

from core import memory
from app.extensions import db
from app.models import ChatSession, ChatMessage
from app.auth import get_current_user, _ensure_settings
from core.rag import rag_query

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
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


@chat_bp.route("/chat/new", methods=["POST"])
def new_chat():
    """Clears the short-term conversation memory (see memory.py) so a
    freshly started chat doesn't inherit context from the previous one.
    Wire this up to the frontend's "New Chat" button."""
    memory.reset_memory()
    return jsonify({"message": "Conversation memory cleared"})
