"""
Shared extension instances.

Kept in their own module (rather than inside Backend.py) so models.py,
auth.py, chat_routes.py, and user_routes.py can all import `db` without
creating circular imports with the Flask app factory.
"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
