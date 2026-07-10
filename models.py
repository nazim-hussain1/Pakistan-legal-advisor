import secrets
import hashlib
from datetime import datetime

from extensions import db


class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    last_login    = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    chats    = db.relationship("ChatSession", backref="user", lazy=True, cascade="all, delete-orphan")
    settings = db.relationship("UserSettings", backref="user", uselist=False, cascade="all, delete-orphan")

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

    messages = db.relationship("ChatMessage", backref="chat_session", lazy=True,
                                cascade="all, delete-orphan", order_by="ChatMessage.created_at")

    def to_dict(self, include_messages=False):
        d = {
            "id":            self.id,
            "session_id":    self.session_id,
            "title":         self.title,
            "created_at":    self.created_at.isoformat() if self.created_at else None,
            "updated_at":    self.updated_at.isoformat() if self.updated_at else None,
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
