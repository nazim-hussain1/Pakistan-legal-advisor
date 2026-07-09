"""
Authentication: email/password registration, login, logout, and the
current-user helper. Google OAuth has been removed entirely — this is
the only auth path in the application now.
"""
from datetime import datetime
from functools import wraps

from flask import Blueprint, jsonify, request, session

from extensions import db
from models import User, UserSettings

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


# ═══════════════════════════════════════════════════════════
# HELPERS
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
# ROUTES — Email / Password
# ═══════════════════════════════════════════════════════════

@auth_bp.route("/register", methods=["POST"])
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


@auth_bp.route("/login", methods=["POST"])
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


@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})


@auth_bp.route("/me", methods=["GET"])
def me():
    user = get_current_user()
    if not user:
        return jsonify({"error": "Not authenticated", "code": "UNAUTHENTICATED"}), 401
    _ensure_settings(user)
    return jsonify({
        "user":     user.to_dict(),
        "settings": user.settings.to_dict() if user.settings else {}
    })
