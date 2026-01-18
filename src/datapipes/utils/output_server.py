from __future__ import annotations
import threading, time, json, socket, html as _html
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import quote
from typing import Any, Callable, Optional
import logging
logger = logging.getLogger(__name__)
from blake3 import blake3
import base64

_outputs: _OutputServer
_server_started = False

def get_server(block: bool=True) -> "_OutputServer":
    global _server_started
    global _outputs
    if not _server_started:
        # print("starting new server")
        _outputs = _OutputServer(interval_ms=250)
        _outputs.start()
        if block:
            while not _outputs.is_running():
                time.sleep(0.05)
        _server_started = True
    # print(f"{_outputs.is_running() = }")
    return _outputs

def set_output(key: str, content_html: str):
    key = _url_escape(key)
    server = get_server()
    server.set_content(key, content_html)
    # print(f"{server.url}/{key}")

def show_output(key: str, onconnect: Callable[[str], None]=lambda key_url: print(f"To see output, open: {key_url}")):
    key = _url_escape(key)
    server = get_server()
    key_url = f"{server.url}/{key}"
    onconnect(key_url)






def _connect_sse(request, server, key: str):
    # SSE stream: each message payload MUST be HTML
    request.send_response(200)
    request.send_header("Content-Type", "text/event-stream; charset=utf-8")
    request.send_header("Cache-Control", "no-store")
    request.send_header("Connection", "keep-alive")
    # If you ever access from another origin, uncomment:
    # self.send_header("Access-Control-Allow-Origin", "*")
    request.end_headers()

    # Helpful for proxies: initial comment
    try:
        request.wfile.write(b": connected\n\n")
        request.wfile.flush()
    except Exception:
        return

    try:
        current_content: str = ""
        while True:
            payload_html: str = server.get_content(key)

            # if payload_html != current_content:
            #     current_content = payload_html
            _send_sse_html(request=request, html_payload=payload_html)

            time.sleep(server.interval_ms / 1000.0)
    except (BrokenPipeError, ConnectionResetError) as ex:
        # Client disconnected
        logger.debug(ex)
        print(ex)
        return
    except Exception as ex:
        # Any other error: just drop connection
        logger.debug(ex)
        print(ex)
        return
    return

def _send_sse_html(request, html_payload: str):
    """
    SSE format: one event separated by blank line.
    data: <line1>
    data: <line2>
    ...
    \n
    """
    # IMPORTANT: SSE "data:" lines cannot contain raw newlines.
    # So we split into lines and send multiple data: lines.
    lines = html_payload.splitlines() or [""]
    out = []
    for line in lines:
        out.append(f"data: {line}\n")
    out.append("\n")  # end of event
    blob = "".join(out).encode("utf-8")

    request.wfile.write(blob)
    request.wfile.flush()

def _url_escape(content) -> str:
    if isinstance(content, Path):
        # content = Path(content).as_posix().replace("/", "_").replace(".", "_")
        # content = base64.b16encode(content.as_posix().encode("utf-8"))[2:-1]
        # content = "path"
        raw = str(content).encode("utf-8")
        hasher = blake3()
        hasher.update(raw)
        content = hasher.hexdigest(length=16)
        # content = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return quote(str(content))


class _OutputServer:
    """
    Minimal local web server with live sse-updated html content.
    """
    def __init__(self, host="127.0.0.1", port=0, interval_ms=200, title="Datapipes outputs"):
        self.host = host
        self.port = port
        self.interval_ms = int(interval_ms)
        self.title = title

        self.content: dict[str, str] = {}
        # self.clients: dict[str, set[BaseHTTPRequestHandler]] = {}

        self._lock = threading.Lock()
        self._server = None
        self._thread = None

        

    def set_content(self, key: str, value: str):
        with self._lock:
            self.content[_url_escape(key)] = value

    def get_content(self, key: str):
        with self._lock:
            return self.content.get(key, "404")

    # def connect_client(self, key: str, request: BaseHTTPRequestHandler):
    #     if key not in self.clients.keys():
    #         self.clients[key] = set()
        
    #     self.clients[key].add(request)

    # def connect_client(self, key: str, request: BaseHTTPRequestHandler):
    #     if key in self.clients.keys():
    #         self.clients[key].remove(request)
        
        

    @property
    def url(self):
        if not self._server:
            return None
        h, p = self._server.server_address
        return f"http://{h}:{p}"

    def get(self):
        with self._lock:
            return dict(self._state)

    def run_in_thread(self, fn, *args, daemon=True, **kwargs):
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=daemon)
        t.start()
        return t

    def is_running(self) -> bool:
        return (
            self._server is not None
            and self._thread is not None
            and self._thread.is_alive()
        )

    def start(self):
        if self._server is not None:
            return self.url

        if self.port == 0:
            self.port = self._pick_free_port(self.host)

        parent = self
        # index_html = self._make_index_html(self.title, "")

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                print(self.path)
                parts = self.path[1:].split("/")
                print(f"{parts}")
                if parts[0] == "events":
                    if len(parts) > 0 and parts[0]:
                        key = parts[1]
                        _connect_sse(request=self, server=parent, key=key)
                
                if len(parts) > 0 and parts[0]:
                    key = parts[0]
                    body = parent._make_index_html("Datapipes output", key=key).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                self.send_response(404)
                self.end_headers()

            
            def log_message(self, format, *args):
                pass  # quiet

        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def stop(self):
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    @staticmethod
    def _pick_free_port(host):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]

    @staticmethod
    def _make_index_html(title, key: str):
        # Page connects to SSE and replaces body.innerHTML with incoming HTML payload
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }}
  </style>
</head>
<body>
  <div>Connecting…</div>

<script>
(function() {{
  const es = new EventSource("/events/{key}");

  es.onmessage = (evt) => {{
    // Payload MUST be HTML
    document.body.innerHTML = evt.data;
  }};

  es.onerror = (err) => {{
    // Browser will auto-reconnect
    console.error("SSE error (will retry):", err);
  }};
}})();
</script>
</body>
</html>
"""

