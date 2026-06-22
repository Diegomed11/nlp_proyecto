"""Cliente mínimo de la API REST de GitHub 

Filosofía del proyecto  cero dependencias para la extracción. Maneja
autenticación por token, paginación vía la cabecera `Link`, y rate limits
(primario y secundario) respetando las cabeceras `X-RateLimit-*` y `Retry-After`.

Uso:
    client = GitHubClient()                      # toma GITHUB_TOKEN del entorno
    data, hdrs = client.request("/repos/scipy/scipy/issues", {"state": "all"})
    for page in client.paginate("/repos/scipy/scipy/issues", {"state": "all"}):
        ...                                       # page = lista de la página actual
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_ROOT = "https://api.github.com"
USER_AGENT = "issue-triage-nlp (proyecto educativo de NLP)"
API_VERSION = "2022-11-28"


class GitHubError(RuntimeError):
    """Error no recuperable de la API (4xx que no sea rate limit, 5xx agotado)."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Carga pares KEY=VALUE de un archivo .env al entorno, sin imprimir valores.

    Reemplaza a python-dotenv con stdlib (filosofía minimalista). No sobreescribe
    variables ya presentes en el entorno. El token nunca se loguea.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _parse_next_link(link_header: str | None) -> str | None:
    """Extrae la URL `rel="next"` de la cabecera Link, o None si no hay más páginas.

    Formato: '<https://api.github.com/...&page=2>; rel="next", <...>; rel="last"'
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        url = segments[0].strip().lstrip("<").rstrip(">")
        for seg in segments[1:]:
            if seg.strip() == 'rel="next"':
                return url
    return None


class GitHubClient:
    """Cliente GET de la API de GitHub con retry de rate limit y paginación."""

    def __init__(
        self,
        token: str | None = None,
        *,
        api_root: str = API_ROOT,
        max_retries: int = 5,
        verbose: bool = True,
    ) -> None:
        load_dotenv()  # toma GITHUB_TOKEN de un .env si existe (sin imprimirlo)
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.api_root = api_root.rstrip("/")
        self.max_retries = max_retries
        self.verbose = verbose


    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[github] {msg}", file=sys.stderr)

    def _sleep_for_rate_limit(self, headers, *, attempt: int) -> None:
        """Duerme lo necesario ante un rate limit (primario o secundario)."""
        retry_after = headers.get("Retry-After")
        remaining = headers.get("X-RateLimit-Remaining")
        if retry_after is not None:  # rate limit secundario / abuse detection
            wait = int(retry_after) + 1
        elif remaining == "0":  # rate limit primario: esperar al reset
            reset = int(headers.get("X-RateLimit-Reset", "0"))
            wait = max(1, reset - int(time.time()) + 1)
        else:  # backoff exponencial para 5xx u otros transitorios
            wait = min(60, 2 ** attempt)
        self._log(f"rate limit / transitorio — durmiendo {wait}s (intento {attempt})")
        time.sleep(wait)


    def request(self, path: str, params: dict | None = None) -> tuple[object, dict]:
        """Una petición GET. Devuelve `(json_decodificado, headers)`.

        Reintenta automáticamente ante rate limits y errores 5xx transitorios.
        """
        url = path if path.startswith("http") else self.api_root + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self._headers())

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read().decode("utf-8")
                    headers = {k: v for k, v in resp.headers.items()}
                    # Throttle proactivo: si quedan pocas llamadas, frena.
                    remaining = headers.get("X-RateLimit-Remaining")
                    if remaining is not None and remaining.isdigit() and int(remaining) <= 1:
                        self._sleep_for_rate_limit(headers, attempt=attempt)
                    return (json.loads(body) if body else None), headers
            except urllib.error.HTTPError as exc:
                headers = {k: v for k, v in exc.headers.items()} if exc.headers else {}
                if exc.code in (403, 429) and (
                    headers.get("Retry-After") is not None
                    or headers.get("X-RateLimit-Remaining") == "0"
                ):
                    self._sleep_for_rate_limit(headers, attempt=attempt)
                    continue
                if exc.code >= 500:  # error de servidor: reintentar con backoff
                    self._sleep_for_rate_limit(headers, attempt=attempt)
                    continue
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise GitHubError(f"HTTP {exc.code} en {url}: {detail}") from exc
            except urllib.error.URLError as exc:  # red caída, DNS, timeout
                if attempt == self.max_retries - 1:
                    raise GitHubError(f"Error de red en {url}: {exc.reason}") from exc
                self._log(f"error de red ({exc.reason}) — reintentando")
                time.sleep(min(60, 2 ** attempt))

        raise GitHubError(f"Agotados {self.max_retries} reintentos en {url}")

    def paginate(self, path: str, params: dict | None = None):
        """Itera sobre todas las páginas siguiendo la cabecera `Link`.

        Produce el cuerpo de cada página (típicamente una lista). El llamador
        decide cómo acumular / cachear.
        """
        params = dict(params or {})
        params.setdefault("per_page", 100)
        url: str | None = self.api_root + path + "?" + urllib.parse.urlencode(params)
        page_num = 1
        while url:
            data, headers = self.request(url)
            yield page_num, data
            url = _parse_next_link(headers.get("Link"))
            page_num += 1

    def rate_limit(self) -> dict:
        """Estado actual del rate limit (no consume cuota del core)."""
        data, _ = self.request("/rate_limit")
        return data
