"""
desktop.py
──────────
PyWebView desktop application entry point.
Exposes a Python API class to the HTML/JS frontend via window.pywebview.api.
Streaming is achieved by calling window.evaluate_js() from background threads.
"""

import json
import shutil
import sys
import time
import threading
from pathlib import Path

import webview

from onprem_rag import __version__
from onprem_rag.config import DOCS_PATH, EMBED_MODEL, OLLAMA_BASE_URL
from onprem_rag.models.agent_profile_store import (
    create_profile,
    delete_profile,
    get_default_profile_id,
    get_profile,
    list_profiles,
    set_default_profile_id,
    update_profile,
)
from onprem_rag.models.chat_store import (
    add_message,
    create_chat,
    delete_chat,
    get_chat,
    get_chats,
    get_messages,
    init_db,
    update_chat,
)
from onprem_rag.models.attachment_store import (
    attach_document,
    get_documents,
    get_document_context,
    remove_document,
    clear_documents,
)
from onprem_rag.services.ingest import (
    clear_knowledge_base,
    delete_knowledge_document,
    get_source_documents,
    rebuild_index,
)
from onprem_rag.services.ollama_manager import is_model_available, pull_model_now
from onprem_rag.services.rag_engine import get_kb_stats, retrieve, ask_stream

if getattr(sys, "frozen", False):
    # In a PyInstaller bundle __file__ is inside the read-only _MEIPASS temp dir.
    UI_HTML = Path(sys._MEIPASS) / "onprem_rag" / "ui" / "index.html"  # type: ignore[attr-defined]
else:
    UI_HTML = Path(__file__).parent / "ui" / "index.html"
APP_ICON = UI_HTML.parent / "favicon.ico"

MODELS = {
    "Qwen 2.5 — 7B":  "qwen2.5:7b",
    "Qwen 3 — 8B":    "qwen3:8b",
    "Gemma 3 — 4B":   "gemma3:4b",
    "Llama 3.1 — 8B": "llama3.1:8b",
}


class Api:
    """All public methods are callable from JS via window.pywebview.api.<method>()"""

    def __init__(self):
        self._win = None

    def _set_window(self, win):
        self._win = win

    def _js(self, code: str):
        """Execute JavaScript in the UI window (safe to call from any thread)."""
        if self._win:
            try:
                self._win.evaluate_js(code)
            except Exception:
                pass

    def _emit(self, event: str, payload):
        """Dispatch a named event to the UI."""
        self._js(f"window._emit({json.dumps(event)}, {json.dumps(payload)})")

    # ── Chats ─────────────────────────────────────────────────────────────────

    def get_chats(self):
        return get_chats()

    def create_chat(
        self,
        model: str = "qwen2.5:7b",
        agent_profile_id: str = "default",
    ) -> dict:
        return create_chat(model=model, agent_profile_id=agent_profile_id)

    def rename_chat(self, chat_id: str, name: str) -> dict:
        update_chat(chat_id, name=name.strip() or "New Chat")
        return {"ok": True}

    def set_chat_model(self, chat_id: str, model: str) -> dict:
        update_chat(chat_id, model=model)
        return {"ok": True}

    def set_chat_agent_profile(self, chat_id: str, profile_id: str) -> dict:
        profile = get_profile(profile_id)
        update_chat(chat_id, agent_profile_id=profile["id"])
        return {"ok": True, "profile": profile}

    def delete_chat(self, chat_id: str) -> dict:
        delete_chat(chat_id)
        return {"ok": True}

    def get_messages(self, chat_id: str) -> list:
        return get_messages(chat_id)

    # Agent profiles

    def get_agent_profiles(self) -> list:
        return list_profiles()

    def create_agent_profile(
        self,
        name: str,
        description: str,
        instructions: str,
        fallback: str,
    ) -> dict:
        return create_profile(name, description, instructions, fallback)

    def update_agent_profile(
        self,
        profile_id: str,
        name: str,
        description: str,
        instructions: str,
        fallback: str,
    ) -> dict:
        return update_profile(profile_id, name, description, instructions, fallback)

    def delete_agent_profile(self, profile_id: str) -> dict:
        delete_profile(profile_id)
        return {"ok": True}

    def get_default_agent_profile(self) -> dict:
        return {"profile_id": get_default_profile_id()}

    def set_default_agent_profile(self, profile_id: str) -> dict:
        return {"profile_id": set_default_profile_id(profile_id)}

    # ── Models ────────────────────────────────────────────────────────────────

    def get_models(self) -> list:
        return [
            {"label": label, "id": mid, "available": is_model_available(mid)}
            for label, mid in MODELS.items()
        ]

    def pull_model(self, model_id: str) -> dict:
        def _pull():
            def cb(msg):
                self._emit("pull:progress", {"model": model_id, "message": msg})
            ok, msg = pull_model_now(model_id, status_callback=cb)
            self._emit("pull:done", {"model": model_id, "ok": ok, "message": msg})
        threading.Thread(target=_pull, daemon=True).start()
        return {"status": "pulling"}

    # ── Messaging ─────────────────────────────────────────────────────────────

    def send_message(self, chat_id: str, content: str, model: str, msg_id: str, attachments: list = None) -> dict:
        chat = get_chat(chat_id)
        if not chat:
            return {"error": "Chat not found"}
        # Snapshot the profile before starting the worker so a concurrent edit
        # cannot change the behavior of an answer already being generated.
        agent_profile = get_profile(chat.get("agent_profile_id", "default"))

        def _stream():
            try:
                # Gather any in-memory documents attached to this chat
                attached_ctx = get_document_context(chat_id)
                has_attachments = bool(attached_ctx)

                # Step 1 — retrieve relevant KB chunks
                #   (skip if we have attachments AND the KB is empty — avoids
                #    "knowledge base empty" error when user is working with
                #    uploaded files only)
                self._emit("step", {"id": msg_id, "step": "retrieve", "status": "loading"})
                ret = retrieve(content)

                if ret.get("error"):
                    self._emit("stream:error", {"id": msg_id, "error": ret["error"]})
                    return

                chunk_count = len(ret.get("chunks", []))
                self._emit("step", {
                    "id": msg_id, "step": "retrieve", "status": "done",
                    "count": chunk_count,
                })

                # Step 2 — generate answer (stream tokens)
                self._emit("step", {"id": msg_id, "step": "generate", "status": "loading"})

                # Pass KB chunks only when found; pass None to ask_stream when
                # collection is empty AND attachments are present (ask_stream
                # will skip its own KB lookup in that case).
                if ret.get("collection_empty") and has_attachments:
                    chunks = None   # attachments cover context; skip KB error
                else:
                    chunks = ret.get("chunks") if not ret.get("collection_empty") else None

                gen = ask_stream(
                    content,
                    chunks=chunks,
                    model=model,
                    attached_context=attached_ctx,
                    agent_profile=agent_profile,
                )

                for token in gen:
                    self._emit("stream:token", {"id": msg_id, "token": token})

                result = gen.result or {}
                # Persist both turns before announcing completion. A browser
                # refresh immediately after stream:done must see the complete
                # conversation rather than a partially written turn.
                add_message(chat_id, "user", content, attachments=attachments or [])
                add_message(
                    chat_id, "assistant",
                    result.get("answer", ""),
                    sources=result.get("sources", []),
                    speed_info={
                        "model":        result.get("model"),
                        "gen_time":     result.get("gen_time"),
                        "token_count":  result.get("token_count"),
                        "tokens_per_sec": result.get("tokens_per_sec"),
                        # Retain the effective profile with the answer for local
                        # auditability even if that profile is edited later.
                        "agent_profile": agent_profile,
                    },
                )

                # Auto-name chat from first message
                if chat["name"] == "New Chat":
                    auto_name = content[:50].strip().rstrip("?.")
                    update_chat(chat_id, name=auto_name)
                    self._emit("chat:renamed", {"id": chat_id, "name": auto_name})

                self._emit("step", {"id": msg_id, "step": "generate", "status": "done"})
                self._emit("stream:done", {"id": msg_id, "result": result})

            except Exception as exc:
                self._emit("stream:error", {"id": msg_id, "error": str(exc)})

        threading.Thread(target=_stream, daemon=True).start()
        return {"status": "streaming"}

    # ── Chat-attached documents (in-memory, session only) ─────────────────────

    def attach_file_to_chat(self, chat_id: str) -> dict:
        """Open a file dialog and attach the chosen file to this chat session."""
        files = self._win.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("PDF Files (*.pdf)", "Word Documents (*.docx)", "Text Files (*.txt)", "All Files (*.*)"),
        )
        if not files:
            return {"ok": False, "cancelled": True}
        try:
            meta = attach_document(chat_id, files[0])
            return {"ok": True, "file": meta}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def get_attached_files(self, chat_id: str) -> list:
        return get_documents(chat_id)

    def remove_attached_file(self, chat_id: str, filename: str) -> dict:
        remove_document(chat_id, filename)
        return {"ok": True}

    def clear_attached_files(self, chat_id: str) -> dict:
        clear_documents(chat_id)
        return {"ok": True}

    # ── Knowledge Base ────────────────────────────────────────────────────────

    def get_kb_stats(self) -> dict:
        return get_kb_stats()

    def delete_kb_document(self, filename: str) -> dict:
        return delete_knowledge_document(filename)

    def clear_kb(self) -> dict:
        return clear_knowledge_base()

    def rebuild_kb(self) -> dict:
        if not get_source_documents():
            return {
                "status": "blocked",
                "error": (
                    "Upload at least one supported document before rebuilding "
                    "the knowledge base."
                ),
            }

        def _rebuild():
            def on_msg(msg):
                self._emit("kb:progress", {"message": msg})
            def on_chunks(done, total):
                self._emit("kb:chunks", {"done": done, "total": total})
            try:
                result = rebuild_index(
                    progress_callback=on_msg,
                    chunk_progress_callback=on_chunks,
                )
                self._emit("kb:done", result)
            except Exception as exc:
                self._emit("kb:error", {"error": str(exc)})
        threading.Thread(target=_rebuild, daemon=True).start()
        return {"status": "rebuilding"}

    def upload_documents(self) -> dict:
        """Open a native file dialog and copy selected files into the docs folder."""
        files = self._win.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("PDF Files (*.pdf)", "Word Documents (*.docx)", "Text Files (*.txt)"),
        )
        if not files:
            return {"files": [], "failed": [], "cancelled": True}
        DOCS_PATH.mkdir(parents=True, exist_ok=True)
        copied = []
        failed = []
        for src in files:
            self._emit(
                "kb:upload-status",
                {"file": Path(src).name, "status": "queued"},
            )
        for src in files:
            filename = Path(src).name
            self._emit(
                "kb:upload-status",
                {"file": filename, "status": "uploading"},
            )
            try:
                dest = DOCS_PATH / filename
                shutil.copy2(src, dest)
                copied.append(filename)
                self._emit(
                    "kb:upload-status",
                    {"file": filename, "status": "uploaded"},
                )
            except Exception as exc:
                failed.append({"file": filename, "error": str(exc)})
                self._emit(
                    "kb:upload-status",
                    {"file": filename, "status": "failed", "error": str(exc)},
                )
        return {"files": copied, "failed": failed, "cancelled": False}

    # ── Ollama ────────────────────────────────────────────────────────────────

    def get_ollama_status(self) -> dict:
        try:
            import ollama
            from onprem_rag.config import OLLAMA_BASE_URL
            ollama.Client(host=OLLAMA_BASE_URL).list()
            return {"running": True}
        except Exception:
            return {"running": False}

    def get_app_info(self) -> dict:
        return {
            "name": "On-Prem RAG Assistant",
            "version": __version__,
            "repository": "https://github.com/arjuninx-tech/enterprise-rag-ai",
            "third_party_notices": (
                "https://github.com/arjuninx-tech/enterprise-rag-ai/"
                "blob/main/THIRD_PARTY_NOTICES.md"
            ),
            "license": "MIT",
            "runtime": "Ollama",
            "ollama_endpoint": OLLAMA_BASE_URL,
            "embedding_model": EMBED_MODEL,
        }


# ── Loading screen shown immediately on launch ────────────────────────────────

_LOADING_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#212121;display:flex;align-items:center;
     justify-content:center;height:100vh;
     font-family:'Segoe UI',system-ui,sans-serif;color:#e0e0e0}
.wrap{text-align:center}
h1{font-size:26px;font-weight:600;margin-bottom:28px;letter-spacing:-.3px}
.ring{width:46px;height:46px;border:3px solid #333;border-top-color:#6366f1;
      border-radius:50%;animation:spin .75s linear infinite;margin:0 auto}
@keyframes spin{to{transform:rotate(360deg)}}
p{margin-top:22px;color:#555;font-size:13px}
</style></head>
<body><div class="wrap">
  <h1>On-Prem RAG Assistant</h1>
  <div class="ring"></div>
  <p>Starting up&hellip;</p>
</div></body></html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def _seed_bundled_kb():
    """
    On the first frozen launch, copy the bundled read-only vector_db seed
    from sys._MEIPASS into the user's writable AppData directory so
    ChromaDB can open and write to it normally.
    Skipped if the AppData KB already exists or there is no bundled seed.
    """
    if not getattr(sys, "frozen", False):
        return
    from onprem_rag.config import BUNDLE_DIR, CHROMA_PATH
    if CHROMA_PATH.exists():
        return                                      # already seeded
    seed = BUNDLE_DIR / "data" / "vector_db"
    if not seed.exists():
        return                                      # no KB bundled at build time
    import shutil
    CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(seed), str(CHROMA_PATH))


def main():
    api = Api()

    # Create the window immediately with an inline loading screen.
    # This appears within ~1 second while heavy init runs in the background.
    win = webview.create_window(
        title="On-Prem RAG Assistant",
        html=_LOADING_HTML,
        js_api=api,
        width=1280,
        height=820,
        min_size=(900, 620),
        background_color="#212121",
        text_select=True,
    )
    api._set_window(win)

    def _on_start():
        """Runs in a background thread after the webview window appears."""
        _seed_bundled_kb()
        init_db()
        UI_HTML.parent.mkdir(parents=True, exist_ok=True)
        time.sleep(0.1)
        win.load_url(UI_HTML.as_uri())

    webview.start(_on_start, debug=False, icon=str(APP_ICON))


if __name__ == "__main__":
    main()
