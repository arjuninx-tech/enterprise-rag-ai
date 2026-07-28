from io import BytesIO
import queue
from unittest import TestCase
from uuid import uuid4

from onprem_rag import server
from onprem_rag.server import flask_app


class ServerBoundaryTests(TestCase):
    def setUp(self):
        self.client = flask_app.test_client()

    def test_web_api_rejects_non_allowlisted_method(self):
        response = self.client.post(
            "/api/_set_window",
            json=[],
            headers={"X-Client-ID": str(uuid4())},
        )
        self.assertEqual(response.status_code, 403)

    def test_web_api_requires_client_id(self):
        response = self.client.post("/api/get_chats", json=[])
        self.assertEqual(response.status_code, 400)

    def test_upload_rejects_unsupported_file(self):
        response = self.client.post(
            "/api/upload_attachment",
            data={
                "chat_id": "example",
                "file": (BytesIO(b"content"), "payload.exe"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 415)

    def test_browser_brand_assets_are_served(self):
        icon_response = self.client.get("/app-icon.svg")
        favicon_response = self.client.get("/favicon.ico")
        try:
            self.assertEqual(icon_response.status_code, 200)
            self.assertEqual(favicon_response.status_code, 200)
        finally:
            icon_response.close()
            favicon_response.close()

    def test_markdown_renderer_escapes_untrusted_model_output(self):
        response = self.client.get("/")
        try:
            html = response.get_data(as_text=True)
            renderer = html.split("function renderMd(text)", 1)[1].split(
                "// \u2500\u2500 Theme", 1
            )[0]

            self.assertIn("let t = String(text)", renderer)
            self.assertIn("t = esc(t)", renderer)
            self.assertLess(
                renderer.index("t = esc(t)"),
                renderer.index(".replace(/\\*\\*"),
            )
            self.assertIn(".replace(/</g, '&lt;')", renderer)
            self.assertIn(".replace(/\"/g, '&quot;')", renderer)
            self.assertIn(".replace(/'/g, '&#039;')", renderer)
        finally:
            response.close()

    def test_stale_sse_cleanup_preserves_reconnected_client(self):
        client_id = str(uuid4())
        stale_queue = queue.Queue()
        replacement_queue = queue.Queue()
        api_marker = object()
        with server._sse_lock:
            server._sse_queues[client_id] = replacement_queue
            server._web_apis[client_id] = api_marker

        try:
            server._remove_sse_client(client_id, stale_queue)

            self.assertIs(server._sse_queues[client_id], replacement_queue)
            self.assertIs(server._web_apis[client_id], api_marker)

            server._remove_sse_client(client_id, replacement_queue)
            self.assertNotIn(client_id, server._sse_queues)
            self.assertNotIn(client_id, server._web_apis)
        finally:
            with server._sse_lock:
                server._sse_queues.pop(client_id, None)
                server._web_apis.pop(client_id, None)
