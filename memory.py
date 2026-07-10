from flask import session

from config import Config


def _trim(text: str) -> str:
    """Cap a single message's length before it goes into the cookie."""
    if len(text) <= Config.MEMORY_TRUNCATE_CHARS:
        return text
    return text[:Config.MEMORY_TRUNCATE_CHARS].rstrip() + "…"


def push_turn(role: str, content: str):
    """Append one message (role='user'|'assistant') to session memory,
    keeping only the most recent MAX_MEMORY_MESSAGES entries."""
    history = session.get("conversation_memory", [])
    history.append({"role": role, "content": _trim(content)})
    history = history[-Config.MAX_MEMORY_MESSAGES:]
    session["conversation_memory"] = history
    session.modified = True


def get_history() -> list:
    """Return stored turns, oldest first. Each item is
    {"role": "user"|"assistant", "content": str}."""
    return session.get("conversation_memory", [])


def reset_memory():
    """Clear conversation memory — call this when the user starts a
    new chat (wired to POST /chat/new)."""
    session["conversation_memory"] = []
    session.modified = True
