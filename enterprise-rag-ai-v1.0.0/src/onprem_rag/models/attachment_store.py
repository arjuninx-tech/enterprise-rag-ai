"""
attachment_store.py
─────────────
Per-chat document store — persisted to SQLite, cached in RAM.

Documents attached in a chat session are extracted to text, saved to the
database, and loaded back automatically when the app restarts.
"""

from pathlib import Path

from onprem_rag.models.chat_store import (
    save_attachment,
    load_attachments,
    delete_attachment,
    clear_attachments,
)

# ── In-memory cache ────────────────────────────────────────────────────────────
# { chat_id: [ { name, content, size_kb, truncated, chars } ] }
# None means "not yet loaded from DB for this chat_id"
_cache: dict[str, list[dict]] = {}

# Maximum characters per document included in the LLM prompt.
MAX_CHARS_PER_DOC = 8_000


# ── Internal helpers ───────────────────────────────────────────────────────────

def _ensure_loaded(chat_id: str) -> None:
    """Load from DB into cache if this chat hasn't been accessed yet."""
    if chat_id not in _cache:
        _cache[chat_id] = load_attachments(chat_id)


def _meta(entry: dict) -> dict:
    """Strip the 'content' key before sending metadata to the frontend."""
    return {k: v for k, v in entry.items() if k != "content"}


# ── Public API ─────────────────────────────────────────────────────────────────

def attach_document(chat_id: str, file_path: str, display_name: str | None = None) -> dict:
    """
    Extract text from *file_path*, persist it, and cache it for *chat_id*.
    If a document with the same filename is already attached it is replaced.

    Returns a metadata dict (no 'content' key) safe to send to the frontend.
    """
    from onprem_rag.services.document_loader import load_document  # lazy import — avoids circular

    path = Path(file_path)
    raw = load_document(path)["text"]
    truncated = len(raw) > MAX_CHARS_PER_DOC
    content   = raw[:MAX_CHARS_PER_DOC] if truncated else raw

    stored_name = Path(display_name).name if display_name else path.name
    entry: dict = {
        "name":      stored_name,
        "content":   content,
        "truncated": truncated,
        "size_kb":   round(path.stat().st_size / 1024, 1),
        "chars":     len(content),
    }

    # Persist to SQLite
    save_attachment(
        chat_id,
        filename=entry["name"],
        content=entry["content"],
        truncated=entry["truncated"],
        size_kb=entry["size_kb"],
        chars=entry["chars"],
    )

    # Update in-memory cache
    _ensure_loaded(chat_id)
    _cache[chat_id] = [d for d in _cache[chat_id] if d["name"] != stored_name]
    _cache[chat_id].append(entry)

    return _meta(entry)


def get_documents(chat_id: str) -> list[dict]:
    """Return metadata list for all documents attached to *chat_id*."""
    _ensure_loaded(chat_id)
    return [_meta(d) for d in _cache[chat_id]]


def get_document_context(chat_id: str) -> str:
    """
    Return a formatted text block for injection into the LLM prompt.
    Returns an empty string when no documents are attached.
    """
    _ensure_loaded(chat_id)
    docs = _cache.get(chat_id, [])
    if not docs:
        return ""

    parts: list[str] = []
    for doc in docs:
        note = "  [first 8 000 chars — document was truncated]" if doc.get("truncated") else ""
        parts.append(
            f"=== Attached Document: {doc['name']}{note} ===\n"
            f"{doc['content']}\n"
            f"=== End: {doc['name']} ==="
        )
    return "\n\n".join(parts)


def remove_document(chat_id: str, filename: str) -> None:
    delete_attachment(chat_id, filename)
    if chat_id in _cache:
        _cache[chat_id] = [d for d in _cache[chat_id] if d["name"] != filename]


def clear_documents(chat_id: str) -> None:
    clear_attachments(chat_id)
    _cache.pop(chat_id, None)
