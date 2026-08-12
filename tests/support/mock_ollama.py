"""Test-support stub Ollama server for `container_smoke.py` (Design.md §7.5).

Standard-library only. Binds `("0.0.0.0", 0)` explicitly (never relies on an
implicit default) and exposes `/mock/ping` as an independent host-gateway
reachability probe endpoint, separate from the app-level `/api/generate`
stub.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def do_GET(self):  # noqa: N802 - stdlib signature
        if self.path == "/mock/ping":
            body = b"pong"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802 - stdlib signature
        if self.path == "/api/generate":
            body = json.dumps({"response": "mock-ollama-fixed-answer", "done": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def start_mock_ollama_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
    return server
