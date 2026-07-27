"""Local Flask server for the browser-based document assistant."""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
from pathlib import Path
from uuid import UUID

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from onprem_rag.config import SUPPORTED_EXTENSIONS
from onprem_rag.desktop import Api
from onprem_rag.models.chat_store import init_db

UI_DIR = Path(__file__).parent / "ui"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

flask_app = Flask(__name__)
flask_app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_sse_lock = threading.Lock()
_sse_queues: dict[str, queue.Queue] = {}
_web_apis: dict[str, Api] = {}

# Client IDs isolate asynchronous browser events. They are routing identifiers,
# not authentication credentials; network mode remains restricted to trusted LANs.

def _valid_client_id(value: str) -> str:
    """Return a canonical UUID or an empty string for invalid session identifiers."""
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        return ""


def _web_emit(client_id: str, event: str, payload) -> None:
    """Push an event only to the browser session that initiated the work."""
    data = json.dumps({"event": event, "payload": payload})
    with _sse_lock:
        client_queue = _sse_queues.get(client_id)
        if client_queue is not None:
            try:
                client_queue.put_nowait(data)
            except queue.Full:
                pass


def _remove_sse_client(client_id: str, client_queue: queue.Queue) -> None:
    """Remove a connection only when it still owns the client's active queue."""
    with _sse_lock:
        if _sse_queues.get(client_id) is client_queue:
            _sse_queues.pop(client_id, None)
            _web_apis.pop(client_id, None)


class _MockWin:
    """Window adapter for API methods shared with desktop mode."""

    def evaluate_js(self, code: str) -> None:
        pass

    def create_file_dialog(self, *args, **kwargs):
        return None


def _api_for(client_id: str) -> Api:
    """Return an API bridge whose asynchronous events are session-scoped."""
    with _sse_lock:
        api = _web_apis.get(client_id)
        if api is None:
            api = Api()
            api._set_window(_MockWin())
            api._emit = lambda event, payload: _web_emit(client_id, event, payload)
            _web_apis[client_id] = api
        return api


def _supported_upload(file) -> tuple[str, str]:
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if not original_name or suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return original_name, suffix


@flask_app.errorhandler(413)
def upload_too_large(_error):
    max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({"ok": False, "error": f"Upload exceeds the {max_mb} MB limit"}), 413


@flask_app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@flask_app.route("/app-icon.svg")
def app_icon():
    return send_from_directory(UI_DIR, "app-icon.svg")


@flask_app.route("/favicon.ico")
def favicon():
    return send_from_directory(UI_DIR, "favicon.ico", mimetype="image/vnd.microsoft.icon")


@flask_app.route("/api/events")
def sse():
    """Deliver real-time events to one browser session."""
    client_id = _valid_client_id(request.args.get("client_id", ""))
    if not client_id:
        return jsonify({"error": "A valid client_id is required"}), 400

    client_queue: queue.Queue = queue.Queue(maxsize=200)
    with _sse_lock:
        _sse_queues[client_id] = client_queue

    def generate():
        try:
            while True:
                try:
                    data = client_queue.get(timeout=20)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    yield 'data: {"event":"ping","payload":{}}\n\n'
        finally:
            # A reconnect may already have installed a replacement queue for the
            # same browser session. An older connection must not remove it.
            _remove_sse_client(client_id, client_queue)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@flask_app.route("/api/upload_attachment", methods=["POST"])
def upload_attachment():
    """Receive a supported file and attach its extracted text to a chat."""
    chat_id = request.form.get("chat_id", "")
    uploaded_file = request.files.get("file")
    if not uploaded_file or not chat_id:
        return jsonify({"ok": False, "error": "Missing file or chat_id"}), 400

    try:
        original_name, suffix = _supported_upload(uploaded_file)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 415

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as file_handle:
            uploaded_file.save(file_handle)

        from onprem_rag.models.attachment_store import attach_document

        meta = attach_document(chat_id, tmp_path, display_name=original_name)
        return jsonify({"ok": True, "file": meta})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@flask_app.route("/api/upload_kb_document", methods=["POST"])
def upload_kb_document():
    """Upload a supported document into the local knowledge-base folder."""
    from onprem_rag.config import DOCS_PATH

    uploaded_file = request.files.get("file")
    if not uploaded_file:
        return jsonify({"ok": False, "error": "No file provided"}), 400

    try:
        original_name, _suffix = _supported_upload(uploaded_file)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 415

    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    uploaded_file.save(str(DOCS_PATH / original_name))
    return jsonify({"ok": True, "name": original_name})


_BROWSER_OVERRIDE = {"attach_file_to_chat", "upload_documents"}
# The browser bridge must stay deny-by-default. Adding a public Api method does
# not expose it remotely unless it is explicitly reviewed and listed here.
_WEB_API_METHODS = {
    "get_chats",
    "create_chat",
    "rename_chat",
    "set_chat_model",
    "set_chat_agent_profile",
    "delete_chat",
    "get_messages",
    "get_models",
    "get_agent_profiles",
    "create_agent_profile",
    "update_agent_profile",
    "delete_agent_profile",
    "get_default_agent_profile",
    "set_default_agent_profile",
    "pull_model",
    "send_message",
    "get_attached_files",
    "remove_attached_file",
    "clear_attached_files",
    "get_kb_stats",
    "delete_kb_document",
    "clear_kb",
    "rebuild_kb",
    "get_ollama_status",
    "get_app_info",
}


@flask_app.route("/api/<method>", methods=["POST"])
def api_call(method: str):
    """Map an allowlisted browser request to the shared application API."""
    if method not in _WEB_API_METHODS and method not in _BROWSER_OVERRIDE:
        return jsonify({"error": "Not allowed"}), 403

    if method in _BROWSER_OVERRIDE:
        return jsonify(
            {
                "ok": False,
                "error": "Use the browser upload button instead",
                "__browser_upload": True,
            }
        )

    client_id = _valid_client_id(request.headers.get("X-Client-ID", ""))
    if not client_id:
        return jsonify({"error": "A valid X-Client-ID header is required"}), 400

    body = request.get_json(force=True, silent=True) or []
    method_fn = getattr(_api_for(client_id), method)
    try:
        if isinstance(body, list):
            result = method_fn(*body)
        elif isinstance(body, dict):
            result = method_fn(**body)
        else:
            result = method_fn()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


def main():
    init_db()
    allow_network = os.getenv("ALLOW_NETWORK_ACCESS", "").lower() in {"1", "true", "yes"}
    host = "0.0.0.0" if allow_network else "127.0.0.1"

    print("\n  On-Prem RAG Assistant - Web Mode")
    print("  http://localhost:5000")
    if allow_network:
        print("  WARNING: LAN mode has no authentication. Use only on a trusted network.")
    print()

    flask_app.run(host=host, port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
