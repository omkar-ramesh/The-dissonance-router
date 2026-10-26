"""Tiny stdlib HTTP server for the dev dashboard. No dependencies, idles at ~0% CPU.

    python -m src.devdash          # then open http://127.0.0.1:8765
"""
import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src import config
from src.devdash.store import active_run, list_runs, load_run, read_log, set_stop

PAGE = Path(__file__).with_name("index.html")
SUITES = ("all", "pytest", "scenarios")
PORT = int(os.getenv("DEVDASH_PORT", "8765"))

_proc: "subprocess.Popen | None" = None
_lock = threading.Lock()


def start_run(suite: str) -> bool:
    """Launch the runner as a subprocess. False if one is already running."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return False
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        _proc = subprocess.Popen(
            [sys.executable, "-m", "src.devdash.run", "--suite", suite],
            cwd=config.ROOT_DIR, creationflags=flags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # polling would flood the terminal
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload).encode())

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/runs":
            self._json(200, list_runs())
        elif path.startswith("/api/runs/") and path.endswith("/log"):
            start = parse_qs(urlparse(self.path).query).get("from", ["0"])[0]
            log = read_log(path[len("/api/runs/"):-len("/log")], int(start) if start.isdigit() else 0)
            self._json(200, log) if log else self._json(404, {"error": "not found"})
        elif path.startswith("/api/runs/"):
            run = load_run(path[len("/api/runs/"):])
            # None also covers a file caught mid-write on Windows; the client just retries.
            self._json(200, run) if run else self._json(404, {"error": "not found"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/api/start", "/api/control"):
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = {}
        if self.path == "/api/control":
            action = body.get("action")
            run_id = active_run()
            if action not in ("stop", "resume"):
                return self._json(400, {"error": "action must be 'stop' or 'resume'"})
            if not run_id or not set_stop(run_id, action == "stop"):
                return self._json(409, {"error": "no run in progress"})
            return self._json(202, {"run": run_id, "action": action})
        suite = body.get("suite", "all")
        if suite not in SUITES:
            return self._json(400, {"error": f"suite must be one of {SUITES}"})
        if start_run(suite):
            self._json(202, {"started": suite})
        else:
            self._json(409, {"error": "a run is already in progress"})


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Dev dashboard on {url}  (Ctrl+C to stop)")
    if not os.getenv("DEVDASH_NO_BROWSER"):
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
