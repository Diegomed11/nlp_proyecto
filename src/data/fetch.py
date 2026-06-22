"""Extracción de issues + PRs de un repo y caché del JSON crudo 

Paginación **por cursor** (`after`/`before` vía el header `Link`): la API REST de
GitHub corta la paginación por `page` en la página 100 (10.000 items) para
datasets grandes y exige cursores. Se ordena por fecha de creación ascendente.

Resumible (mitigación de rate limits, §16): tras cada página se persiste el
cursor `next` en `_state.json`. Si la extracción se interrumpe, en la siguiente
corrida se reanuda desde ese cursor sin re-bajar lo ya cacheado.

Nota: el endpoint `/issues` devuelve issues Y pull requests mezclados; los PRs
traen la clave `pull_request`. Se guardan ambos crudos; la separación y curación
viven en un paso posterior (`project_issue`).

CLI:
    python -m src.data.fetch scipy/scipy
    python -m src.data.fetch pymc-devs/pymc --max-pages 2   # prueba rápida
    python -m src.data.fetch scipy/scipy --force            # ignora caché/estado
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from pathlib import Path

from .github_client import GitHubClient, GitHubError, _parse_next_link

DEFAULT_OUT = Path("data/raw")


def slugify(repo: str) -> str:
    """'scipy/scipy' -> 'scipy__scipy' (nombre de carpeta seguro)."""
    return repo.replace("/", "__")


def is_pull_request(item: dict) -> bool:
    return "pull_request" in item


def project_issue(item: dict) -> dict:
    """Proyecta un issue/PR crudo a los campos relevantes (§5).

    Curación mínima reutilizable; la propagación de etiqueta de módulo y los
    splits temporales se hacen en fases posteriores.
    """
    return {
        "repo_number": item.get("number"),
        "is_pr": is_pull_request(item),
        "title": item.get("title"),
        "body": item.get("body"),
        "labels": [lbl["name"] for lbl in item.get("labels", []) if isinstance(lbl, dict)],
        "state": item.get("state"),
        "created_at": item.get("created_at"),
        "closed_at": item.get("closed_at"),
        "comments": item.get("comments"),
        "reactions": (item.get("reactions") or {}).get("total_count"),
        "author": (item.get("user") or {}).get("login"),
        "assignees": [a["login"] for a in item.get("assignees", []) if isinstance(a, dict)],
        "milestone": (item.get("milestone") or {}).get("title"),
        "html_url": item.get("html_url"),
    }


def _load_state(pages_dir: Path) -> dict:
    f = pages_dir / "_state.json"
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(pages_dir: Path, state: dict) -> None:
    (pages_dir / "_state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )


def _count_cached(pages_dir: Path) -> tuple[int, int]:
    """Cuenta (issues, prs) sobre todas las páginas cacheadas en disco."""
    n_issues = n_prs = 0
    for pf in sorted(pages_dir.glob("page*.json")):
        try:
            items = json.loads(pf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for it in items:
            if is_pull_request(it):
                n_prs += 1
            else:
                n_issues += 1
    return n_issues, n_prs


def _summary(repo: str, pages_dir: Path, done: bool) -> dict:
    n_issues, n_prs = _count_cached(pages_dir)
    summary = {
        "repo": repo,
        "pages": len(list(pages_dir.glob("page*.json"))),
        "issues": n_issues,
        "pull_requests": n_prs,
        "complete": done,
        "cache_dir": str(pages_dir),
    }
    (pages_dir.parent / "issues_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def fetch_issues(
    client: GitHubClient,
    repo: str,
    out_dir: Path = DEFAULT_OUT,
    *,
    state: str = "all",
    max_pages: int | None = None,
    force: bool = False,
) -> dict:
    """Pagina issues+PRs de `repo` por cursor y cachea cada página cruda.

    Resumible vía `_state.json` (cursor `next` persistido). `max_pages` limita las
    páginas *nuevas* de esta corrida (para pruebas). `force=True` reinicia desde
    el principio ignorando el estado. Páginas en `<slug>/issues/pageNNNN.json`.
    """
    pages_dir = out_dir / slugify(repo) / "issues"
    pages_dir.mkdir(parents=True, exist_ok=True)

    base_params = {"state": state, "sort": "created", "direction": "asc", "per_page": 100}
    first_url = (
        client.api_root + f"/repos/{repo}/issues?" + urllib.parse.urlencode(base_params)
    )

    saved = {} if force else _load_state(pages_dir)
    if saved.get("done"):
        print(
            f"[fetch] {repo}: ya completo ({saved.get('pages')} págs). "
            "--force para refrescar.",
            file=sys.stderr,
        )
        return _summary(repo, pages_dir, done=True)

    page_num = saved.get("pages", 0)
    url = saved.get("next_url") or first_url
    if page_num:
        print(f"[fetch] {repo}: reanudando desde la página {page_num + 1}", file=sys.stderr)

    new_pages = 0
    done = False
    while url:
        items, headers = client.request(url)
        if not items:
            done = True
            _save_state(pages_dir, {"pages": page_num, "next_url": None, "done": True})
            break
        page_num += 1
        new_pages += 1
        (pages_dir / f"page{page_num:04d}.json").write_text(
            json.dumps(items, ensure_ascii=False), encoding="utf-8"
        )
        next_url = _parse_next_link(headers.get("Link"))
        _save_state(
            pages_dir,
            {"pages": page_num, "next_url": next_url, "done": next_url is None},
        )
        n_pr = sum(is_pull_request(it) for it in items)
        print(
            f"[fetch] {repo} pág {page_num}: +{len(items)} "
            f"(issues={len(items) - n_pr}, prs={n_pr})",
            file=sys.stderr,
        )
        if next_url is None:  # última página
            done = True
            break
        if max_pages is not None and new_pages >= max_pages:
            print(f"[fetch] límite de {max_pages} páginas nuevas alcanzado", file=sys.stderr)
            break
        url = next_url

    return _summary(repo, pages_dir, done=done)


def _force_utf8_stdio() -> None:
    """Evita que la consola de Windows (cp1252) destroce acentos/símbolos."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description="Extrae issues+PRs de un repo de GitHub.")
    parser.add_argument("repo", help="owner/repo, p. ej. scipy/scipy")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dir de salida (data/raw)")
    parser.add_argument("--state", default="all", choices=["all", "open", "closed"])
    parser.add_argument("--max-pages", type=int, default=None, help="tope de páginas (prueba)")
    parser.add_argument("--force", action="store_true", help="ignora el caché y re-baja todo")
    args = parser.parse_args(argv)

    client = GitHubClient()
    if not client.token:
        print(
            "[aviso] sin GITHUB_TOKEN: rate limit de 60 req/h (≈6000 issues máx).\n"
            "        Para extracción completa: export GITHUB_TOKEN=ghp_xxx",
            file=sys.stderr,
        )

    try:
        summary = fetch_issues(
            client,
            args.repo,
            args.out,
            state=args.state,
            max_pages=args.max_pages,
            force=args.force,
        )
    except GitHubError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
