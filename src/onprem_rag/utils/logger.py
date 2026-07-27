"""
logger.py
─────────
Module 9.6 — Logging Module
Logs user questions, answers, sources, timestamps, and errors
to logs/questions.log as defined in FR-08.
"""

import json
import traceback
from datetime import datetime
from onprem_rag.config import LOG_PATH


def _ensure_log_dir() -> None:
    """Create the logs directory if it does not exist."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_question(
    question: str,
    answer: str,
    sources: list[dict],
    found: bool,
    error: str | None = None,
) -> None:
    """
    Append a log entry to logs/questions.log.

    Args:
        question:  The user's question.
        answer:    The assistant's answer (or NOT_FOUND_RESPONSE).
        sources:   List of source dicts [{'document_name': ..., 'page': ...}].
        found:     True if relevant context was found; False otherwise.
        error:     Optional error message if something went wrong.
    """
    _ensure_log_dir()

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer_found": found,
        "sources": sources,
        "answer": answer,
    }
    if error:
        entry["error"] = error

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Logger] Warning: could not write to log file: {e}")


def log_error(context: str, exc: Exception) -> None:
    """
    Log a system error (e.g. Ollama not running, ingestion failure).

    Args:
        context: Short description of where the error occurred.
        exc:     The exception object.
    """
    _ensure_log_dir()

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "error_context": context,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
