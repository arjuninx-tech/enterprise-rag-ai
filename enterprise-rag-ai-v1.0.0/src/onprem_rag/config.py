"""
config.py
─────────
Central configuration for the On-Prem RAG Assistant.
All settings loaded from .env (falls back to defaults if .env is absent).

Frozen mode (PyInstaller):
  - App code lives in sys._MEIPASS (extracted bundle, read-only)
  - User data (ChromaDB, chats.db, documents) lives in:
        Windows  → %APPDATA%\\On-Prem RAG Assistant
        macOS    → ~/Library/Application Support/On-Prem RAG Assistant
        Linux    → ~/.local/share/On-Prem RAG Assistant
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Locate roots ──────────────────────────────────────────────────────────────

def _user_data_dir() -> Path:
    """Return a writable per-user directory for app data."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "On-Prem RAG Assistant"


if getattr(sys, "frozen", False):
    # Running as a PyInstaller bundle
    # BUNDLE_DIR  → sys._MEIPASS  (extracted code; read-only at runtime)
    # ROOT        → user-writable data directory
    BUNDLE_DIR: Path = Path(sys._MEIPASS)          # type: ignore[attr-defined]
    ROOT: Path       = _user_data_dir()
    ROOT.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")                     # optional user override
else:
    # Normal development run
    ROOT        = Path(__file__).resolve().parent.parent.parent
    BUNDLE_DIR  = ROOT
    load_dotenv(ROOT / ".env")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str       = os.getenv("LLM_MODEL",       "qwen2.5:7b")
EMBED_MODEL: str     = os.getenv("EMBED_MODEL",      "nomic-embed-text")
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_PATH:   Path = ROOT / os.getenv("CHROMA_PATH",   "data/vector_db")
DOCS_PATH:     Path = ROOT / os.getenv("DOCS_PATH",     "data/documents")
METADATA_PATH: Path = ROOT / os.getenv("METADATA_PATH", "data/metadata/metadata.json")
LOG_PATH:      Path = ROOT / os.getenv("LOG_PATH",      "logs/questions.log")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION: str = "knowledge_documents"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE:    int = int(os.getenv("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "3"))
SUMMARY_TOP_K: int = int(os.getenv("SUMMARY_TOP_K", "12"))
MAX_COSINE_DISTANCE: float = float(os.getenv("MAX_COSINE_DISTANCE", "0.45"))

# ── Supported file extensions ─────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".docx", ".txt", ".md", ".markdown")
