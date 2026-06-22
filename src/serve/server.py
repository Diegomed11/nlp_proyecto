"""Servicio HTTP del baseline con la stdlib (§10): http.server, cero dependencias web.

En producción corre con numpy como única dependencia no-stdlib. Inferencia en µs.

    python -m src.serve.server         # http://127.0.0.1:8000/predict
    curl -s -X POST localhost:8000/predict -d '{"title":"csr_matrix bug","body":"..."}'
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.serve.predict import BaselinePredictor

PRED = BaselinePredictor()  # carga el artefacto una sola vez


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "classes": len(PRED.classes)})
        else:
            self._json(404, {"error": "use POST /predict"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        preds = PRED.predict(data.get("title", ""), data.get("body", ""), data.get("threshold"))
        self._json(200, {"modules": [{"module": c, "confidence": p} for c, p in preds]})

    def log_message(self, *args) -> None:  # silencio
        pass


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    sys.stderr.write(f"baseline TF-IDF+logreg en http://{host}:{port}/predict\n")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
