"""
ollama_manager.py
─────────────────
Handles automatic startup of Ollama and model availability checks.
- If models are already present locally: starts immediately, no download.
- If a model is missing: pulls it automatically (requires internet once).
"""

import subprocess
import time
import sys

import ollama

from onprem_rag.config import OLLAMA_BASE_URL, LLM_MODEL, EMBED_MODEL


def _is_ollama_running() -> bool:
    try:
        ollama.Client(host=OLLAMA_BASE_URL).list()
        return True
    except Exception:
        return False


def _start_ollama() -> tuple:
    """Start Ollama as a background process. Returns (success, message)."""
    try:
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
    except FileNotFoundError:
        return False, "Ollama not found. Install it from https://ollama.com then restart."
    except Exception as e:
        return False, f"Failed to start Ollama: {e}"

    for _ in range(15):
        time.sleep(1)
        if _is_ollama_running():
            return True, "Ollama started."

    return False, (
        "Ollama did not respond within 15 seconds. "
        "Try running 'ollama serve' manually in a terminal."
    )


def _get_pulled_model_names() -> list:
    """Return list of model name strings currently available locally."""
    try:
        models = ollama.Client(host=OLLAMA_BASE_URL).list()
        result = []
        for m in models.get("models", []):
            if isinstance(m, dict):
                result.append(m.get("name", ""))
            else:
                result.append(getattr(m, "model", getattr(m, "name", "")))
        return result
    except Exception:
        return []


def _model_is_available(model: str, pulled: list) -> bool:
    """
    Check if a model exists locally.
    Matches on base name + tag so 'qwen2.5:7b' matches 'qwen2.5:7b'
    and won't false-match 'qwen2.5:14b'.
    """
    base = model.split(":")[0].lower()
    tag = model.split(":")[1].lower() if ":" in model else ""
    for p in pulled:
        p_lower = p.lower()
        if base in p_lower and (not tag or tag in p_lower):
            return True
    return False


def _pull_model(model: str, status_callback=None) -> tuple:
    """Pull a model via ollama pull. Returns (success, message)."""
    if status_callback:
        status_callback(f"Downloading '{model}'... (this may take several minutes on first run)")

    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        if result.returncode == 0:
            return True, f"'{model}' downloaded successfully."
        else:
            return False, f"Failed to download '{model}': {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, f"Download timed out for '{model}'. Run manually: ollama pull {model}"
    except FileNotFoundError:
        return False, "Ollama not found. Install from https://ollama.com"
    except Exception as e:
        return False, f"Error downloading '{model}': {e}"


def is_model_available(model: str) -> bool:
    """Return True if the given model is already pulled locally."""
    return _model_is_available(model, _get_pulled_model_names())


def pull_model_now(model: str, status_callback=None) -> tuple:
    """Public wrapper to pull a single model. Returns (success, message)."""
    return _pull_model(model, status_callback=status_callback)


def ensure_ollama_ready(status_callback=None) -> tuple:
    """
    Startup sequence:
      1. Start Ollama if it is not already running.
      2. Check which required models are present locally.
      3. Skip models that are already downloaded.
      4. Pull only the missing ones (auto-download on first run).

    Returns:
        (ready: bool, message: str)
    """
    def _log(msg: str):
        if status_callback:
            status_callback(msg)
        print(f"[Ollama] {msg}")

    # Step 1 — ensure Ollama process is running
    if _is_ollama_running():
        _log("Ollama is running.")
    else:
        _log("Starting Ollama...")
        started, msg = _start_ollama()
        if not started:
            return False, msg
        _log(msg)

    # Step 2 — check which models are already present
    pulled = _get_pulled_model_names()
    required = [LLM_MODEL, EMBED_MODEL]
    missing = [m for m in required if not _model_is_available(m, pulled)]

    if not missing:
        _log(f"All models ready: {', '.join(required)}")
        return True, f"Ready. Models: {', '.join(required)}"

    # Step 3 — pull only missing models
    already = [m for m in required if m not in missing]
    if already:
        _log(f"Already installed: {', '.join(already)}")
    _log(f"Missing models: {', '.join(missing)} — downloading now...")

    for model in missing:
        ok, msg = _pull_model(model, status_callback=_log)
        if not ok:
            return False, (
                f"{msg}\n\n"
                f"Pull it manually when connected to the internet:\n"
                f"  ollama pull {model}"
            )
        _log(msg)

    return True, f"Ready. All models available: {', '.join(required)}"
