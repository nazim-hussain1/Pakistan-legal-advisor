from flask import Blueprint, jsonify, request

from extensions import db
from models import ChatSession, User
from auth import login_required, get_current_user, _ensure_settings

user_bp = Blueprint("user", __name__)


# ═══════════════════════════════════════════════════════════
# CHAT HISTORY
# ═══════════════════════════════════════════════════════════

@user_bp.route("/history", methods=["GET"])
@login_required
def get_history():
    user = get_current_user()
    sessions = (ChatSession.query
                .filter_by(user_id=user.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(100)
                .all())
    return jsonify({"sessions": [s.to_dict() for s in sessions]})


@user_bp.route("/history/<session_id>", methods=["GET"])
@login_required
def get_session(session_id):
    user = get_current_user()
    chat_sess = ChatSession.query.filter_by(
        session_id=session_id, user_id=user.id
    ).first()
    if not chat_sess:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"session": chat_sess.to_dict(include_messages=True)})


@user_bp.route("/history/<session_id>", methods=["DELETE"])
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


@user_bp.route("/history/clear", methods=["DELETE"])
@login_required
def clear_history():
    user = get_current_user()
    ChatSession.query.filter_by(user_id=user.id).delete()
    db.session.commit()
    return jsonify({"message": "All history cleared"})


# ═══════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════

@user_bp.route("/settings", methods=["GET"])
@login_required
def get_settings():
    user = get_current_user()
    _ensure_settings(user)
    return jsonify({"settings": user.settings.to_dict()})


@user_bp.route("/settings", methods=["PUT"])
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


# ═══════════════════════════════════════════════════════════
# ACCOUNT
# ═══════════════════════════════════════════════════════════

@user_bp.route("/account", methods=["PUT"])
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


@user_bp.route("/account", methods=["DELETE"])
@login_required
def delete_account():
    user = get_current_user()
    db.session.delete(user)
    db.session.commit()
    from flask import session
    session.clear()
    return jsonify({"message": "Account deleted"})
