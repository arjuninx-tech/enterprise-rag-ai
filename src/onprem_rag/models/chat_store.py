"""
chat_store.py
─────────────
SQLite-backed persistence for chat sessions and messages.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from collections.abc import Iterator
from onprem_rag.config import ROOT

DB_PATH = ROOT / "data" / "chats.db"


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    """Yield a configured connection and always release its file handle."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL DEFAULT 'New Chat',
                model      TEXT NOT NULL DEFAULT 'qwen2.5:7b',
                agent_profile_id TEXT NOT NULL DEFAULT 'default',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        chat_cols = [r[1] for r in conn.execute("PRAGMA table_info(chats)").fetchall()]
        if "agent_profile_id" not in chat_cols:
            conn.execute(
                "ALTER TABLE chats ADD COLUMN agent_profile_id "
                "TEXT NOT NULL DEFAULT 'default'"
            )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         TEXT PRIMARY KEY,
                chat_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                sources    TEXT,
                speed_info TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id)")
        # Migration: add attachments column if it doesn't exist yet
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "attachments" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attachments (
                id         TEXT PRIMARY KEY,
                chat_id    TEXT NOT NULL,
                filename   TEXT NOT NULL,
                content    TEXT NOT NULL,
                truncated  INTEGER NOT NULL DEFAULT 0,
                size_kb    REAL,
                chars      INTEGER,
                created_at REAL NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attach_chat ON attachments(chat_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_profiles (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                description   TEXT NOT NULL DEFAULT '',
                instructions  TEXT NOT NULL,
                fallback      TEXT NOT NULL,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()


# ── Chats ─────────────────────────────────────────────────────────────────────

def create_chat(
    name: str = "New Chat",
    model: str = "qwen2.5:7b",
    agent_profile_id: str = "default",
) -> dict:
    chat_id = str(uuid.uuid4())
    now = time.time()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO chats "
            "(id, name, model, agent_profile_id, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (chat_id, name, model, agent_profile_id, now, now),
        )
        conn.commit()
    return {
        "id": chat_id,
        "name": name,
        "model": model,
        "agent_profile_id": agent_profile_id,
        "created_at": now,
        "updated_at": now,
    }


def get_chats() -> list:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    return dict(row) if row else None


def update_chat(
    chat_id: str,
    name: str | None = None,
    model: str | None = None,
    agent_profile_id: str | None = None,
):
    now = time.time()
    with _conn() as conn:
        if name is not None:
            conn.execute("UPDATE chats SET name=?, updated_at=? WHERE id=?", (name, now, chat_id))
        if model is not None:
            conn.execute("UPDATE chats SET model=?, updated_at=? WHERE id=?", (model, now, chat_id))
        if agent_profile_id is not None:
            conn.execute(
                "UPDATE chats SET agent_profile_id=?, updated_at=? WHERE id=?",
                (agent_profile_id, now, chat_id),
            )
        conn.commit()


def delete_chat(chat_id: str):
    with _conn() as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
        conn.commit()


# ── Messages ──────────────────────────────────────────────────────────────────

def add_message(
    chat_id: str,
    role: str,
    content: str,
    sources: list | None = None,
    speed_info: dict | None = None,
    attachments: list | None = None,
) -> dict:
    msg_id = str(uuid.uuid4())
    now = time.time()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, sources, speed_info, created_at, attachments) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                msg_id,
                chat_id,
                role,
                content,
                json.dumps(sources) if sources else None,
                json.dumps(speed_info) if speed_info else None,
                now,
                json.dumps(attachments) if attachments else None,
            ),
        )
        conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
        conn.commit()
    return {
        "id": msg_id,
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "speed_info": speed_info,
        "attachments": attachments or [],
        "created_at": now,
    }


def get_messages(chat_id: str) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
    result = []
    for r in rows:
        msg = dict(r)
        msg["sources"]     = json.loads(r["sources"])     if r["sources"]     else []
        msg["speed_info"]  = json.loads(r["speed_info"])  if r["speed_info"]  else None
        msg["attachments"] = json.loads(r["attachments"]) if r["attachments"] else []
        result.append(msg)
    return result


# ── Attachments ───────────────────────────────────────────────────────────────

def save_attachment(
    chat_id: str,
    filename: str,
    content: str,
    truncated: bool,
    size_kb: float,
    chars: int,
) -> None:
    """Upsert an attachment record (replace if same filename exists for this chat)."""
    now = time.time()
    attach_id = str(uuid.uuid4())
    with _conn() as conn:
        # Delete existing entry with same filename for this chat first
        conn.execute(
            "DELETE FROM attachments WHERE chat_id=? AND filename=?",
            (chat_id, filename),
        )
        conn.execute(
            "INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?)",
            (attach_id, chat_id, filename, content,
             1 if truncated else 0, size_kb, chars, now),
        )
        conn.commit()


def load_attachments(chat_id: str) -> list[dict]:
    """Return all attachment rows (including content) for a chat."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT filename, content, truncated, size_kb, chars "
            "FROM attachments WHERE chat_id=? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
    return [
        {
            "name":      r["filename"],
            "content":   r["content"],
            "truncated": bool(r["truncated"]),
            "size_kb":   r["size_kb"],
            "chars":     r["chars"],
        }
        for r in rows
    ]


def delete_attachment(chat_id: str, filename: str) -> None:
    with _conn() as conn:
        conn.execute(
            "DELETE FROM attachments WHERE chat_id=? AND filename=?",
            (chat_id, filename),
        )
        conn.commit()


def clear_attachments(chat_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM attachments WHERE chat_id=?", (chat_id,))
        conn.commit()
