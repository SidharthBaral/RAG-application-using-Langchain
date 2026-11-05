import sqlite3
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


class SQLiteChatMemory:
    """Manages persistent multi-session chat history using SQLite."""

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates the messages table if it does not exist."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id);"
            )
            conn.commit()

    def add_message(self, session_id: str, role: str, message: str):
        """Appends a new turn to the session history."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?);",
                (session_id, role, message),
            )
            conn.commit()

    def get_messages(self, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
        """Retrieves the last N messages for a session in chronological order."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, message FROM chat_history
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?;
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
            # Reverse rows to return chronological order (oldest to newest)
            return [{"role": row["role"], "message": row["message"]} for row in reversed(rows)]

    def get_formatted_history(self, session_id: str, limit: int = 6) -> str:
        """Formats the last N messages as a text block for the LLM prompt."""
        messages = self.get_messages(session_id=session_id, limit=limit)
        if not messages:
            return "No previous conversation history."

        formatted_lines = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['message']}"
            for m in messages
        ]
        return "\n".join(formatted_lines)

    def clear_session(self, session_id: str):
        """Deletes all messages for a specific session."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?;", (session_id,))
            conn.commit()