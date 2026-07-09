"""
Lightweight per-browser conversation memory.

Stores the last MAX_MEMORY_MESSAGES (default 6, i.e. 3 user+assistant
exchanges) turns in the signed Flask session cookie so multi-turn
follow-up questions ("what about its exceptions?", "and in Urdu
script?") can be answered with the right context — without requiring a
database round-trip or a logged-in user.

This works identically for anonymous visitors and authenticated users,
since it rides on Flask's session cookie rather than the ChatSession /
ChatMessage database tables (which remain unchanged and are only used
for the persistent "Chat History" page for logged-in users).

IMPORTANT: memory is for conversational continuity only. The RAG
prompt (see prompts.py) explicitly instructs the LLM to draw all legal
facts exclusively from the retrieved CONTEXT — never from prior
conversation turns — so this cannot introduce hallucinated legal
claims carried over from earlier in the chat.
"""
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
