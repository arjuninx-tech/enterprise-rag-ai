"""SQLite persistence and validation for configurable agent profiles."""

import time
import uuid

from onprem_rag.models.chat_store import _conn
from onprem_rag.services.prompts import DEFAULT_FALLBACK, DEFAULT_PROFILE_INSTRUCTIONS

DEFAULT_PROFILE_ID = "default"
MAX_NAME_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 500
MAX_INSTRUCTIONS_LENGTH = 12_000
MAX_FALLBACK_LENGTH = 500


def default_profile() -> dict:
    """Return the immutable built-in profile."""
    return {
        "id": DEFAULT_PROFILE_ID,
        "name": "General Document Assistant",
        "description": "Grounded question answering, summarization, comparison, and extraction.",
        "instructions": DEFAULT_PROFILE_INSTRUCTIONS,
        "fallback": DEFAULT_FALLBACK,
        "built_in": True,
        "created_at": None,
        "updated_at": None,
    }


def _validate(name: str, description: str, instructions: str, fallback: str) -> tuple:
    name = name.strip()
    description = description.strip()
    instructions = instructions.strip()
    fallback = fallback.strip()

    if not name or len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"Profile name must contain 1-{MAX_NAME_LENGTH} characters")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"Description cannot exceed {MAX_DESCRIPTION_LENGTH} characters")
    if not instructions or len(instructions) > MAX_INSTRUCTIONS_LENGTH:
        raise ValueError(
            f"Instructions must contain 1-{MAX_INSTRUCTIONS_LENGTH} characters"
        )
    if not fallback or len(fallback) > MAX_FALLBACK_LENGTH:
        raise ValueError(f"Fallback must contain 1-{MAX_FALLBACK_LENGTH} characters")
    return name, description, instructions, fallback


def list_profiles() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_profiles ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [default_profile(), *[{**dict(row), "built_in": False} for row in rows]]


def get_profile(profile_id: str) -> dict:
    if not profile_id or profile_id == DEFAULT_PROFILE_ID:
        return default_profile()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM agent_profiles WHERE id=?", (profile_id,)
        ).fetchone()
    return {**dict(row), "built_in": False} if row else default_profile()


def get_default_profile_id() -> str:
    """Return the profile used for newly created chats."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key='default_agent_profile_id'"
        ).fetchone()
    profile_id = row["value"] if row else DEFAULT_PROFILE_ID
    return get_profile(profile_id)["id"]


def set_default_profile_id(profile_id: str) -> str:
    """Persist the profile used for newly created chats."""
    resolved_id = get_profile(profile_id)["id"]
    if resolved_id != profile_id:
        raise ValueError("Agent profile not found")
    with _conn() as conn:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("default_agent_profile_id", resolved_id),
        )
        conn.commit()
    return resolved_id


def create_profile(
    name: str,
    description: str,
    instructions: str,
    fallback: str,
) -> dict:
    name, description, instructions, fallback = _validate(
        name, description, instructions, fallback
    )
    profile_id = str(uuid.uuid4())
    now = time.time()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO agent_profiles VALUES (?,?,?,?,?,?,?)",
            (profile_id, name, description, instructions, fallback, now, now),
        )
        conn.commit()
    return get_profile(profile_id)


def update_profile(
    profile_id: str,
    name: str,
    description: str,
    instructions: str,
    fallback: str,
) -> dict:
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("The built-in profile cannot be modified")
    name, description, instructions, fallback = _validate(
        name, description, instructions, fallback
    )
    with _conn() as conn:
        cursor = conn.execute(
            "UPDATE agent_profiles SET name=?, description=?, instructions=?, "
            "fallback=?, updated_at=? WHERE id=?",
            (name, description, instructions, fallback, time.time(), profile_id),
        )
        if cursor.rowcount == 0:
            raise ValueError("Agent profile not found")
        conn.commit()
    return get_profile(profile_id)


def delete_profile(profile_id: str) -> None:
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("The built-in profile cannot be deleted")
    with _conn() as conn:
        conn.execute(
            "UPDATE app_settings SET value=? "
            "WHERE key='default_agent_profile_id' AND value=?",
            (DEFAULT_PROFILE_ID, profile_id),
        )
        conn.execute(
            "UPDATE chats SET agent_profile_id=? WHERE agent_profile_id=?",
            (DEFAULT_PROFILE_ID, profile_id),
        )
        conn.execute("DELETE FROM agent_profiles WHERE id=?", (profile_id,))
        conn.commit()
