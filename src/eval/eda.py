"""EDA sobre los datos crudos extraídos (notebook 01 del plan).

Cuantifica lo que informa las decisiones de las fases siguientes:
- **Split temporal** (§5): issues por año → ¿hay suficiente data 2025+ para eval?
- **Weak supervision** (§5, tarea 2): ¿qué % de issues ya trae una etiqueta de
  módulo directa (p. ej. `scipy.sparse`)? Eso es ground truth barato.
- **Tokenizer** (§7, tarea 3): prevalencia de tracebacks, bloques de código,
  identificadores y `#refs` → justifica la decisión señal-vs-ruido del traceback.

Solo stdlib. CLI:
    python -m src.eval.eda scipy/scipy
    python -m src.eval.eda pymc-devs/pymc
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.extract.fetch import DEFAULT_OUT, is_pull_request, slugify

# Patrones de características textuales (presencia, no conteo exhaustivo).
RE_CODE_FENCE = re.compile(r"```")
RE_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
RE_ISSUE_REF = re.compile(r"#\d+")
RE_INLINE_CODE = re.compile(r"`[^`\n]+`")
# Etiqueta de módulo estilo "scipy.sparse" / "pymc.distributions".
RE_MODULE_LABEL = re.compile(r"^([a-z_][\w-]*)\.(\w+)")


def iter_items(repo: str, raw_dir: Path = DEFAULT_OUT):
    """Itera todos los issues+PRs cacheados de un repo (orden de página)."""
    pages_dir = raw_dir / slugify(repo) / "issues"
    for pf in sorted(pages_dir.glob("page*.json")):
        try:
            for it in json.loads(pf.read_text(encoding="utf-8")):
                yield it
        except (json.JSONDecodeError, OSError):
            continue


def _quantiles(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "max": 0}
    s = sorted(values)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"min": s[0], "median": q(0.5), "p90": q(0.9), "max": s[-1]}


def analyze_repo(repo: str, raw_dir: Path = DEFAULT_OUT) -> dict:
    n_issues = n_prs = 0
    state_counter = Counter()
    year_counter = Counter()           # issues (no PRs) por año de creación
    label_counter = Counter()
    module_label_counter = Counter()
    issues_with_module_label = 0
    feats = Counter()                  # traceback / code_fence / issue_ref / inline_code
    body_chars: list[int] = []
    body_lines: list[int] = []
    n_issue_with_body = 0

    for it in iter_items(repo, raw_dir):
        is_pr = is_pull_request(it)
        n_prs += is_pr
        n_issues += not is_pr
        state_counter[it.get("state", "?")] += 1

        labels = [l["name"] for l in it.get("labels", []) if isinstance(l, dict)]
        for name in labels:
            label_counter[name] += 1
            m = RE_MODULE_LABEL.match(name)
            if m:
                module_label_counter[m.group(0)] += 1

        if is_pr:
            continue  # el resto del análisis es sobre issues reales

        created = it.get("created_at") or ""
        if len(created) >= 4 and created[:4].isdigit():
            year_counter[created[:4]] += 1
        if any(RE_MODULE_LABEL.match(n) for n in labels):
            issues_with_module_label += 1

        body = it.get("body") or ""
        if body.strip():
            n_issue_with_body += 1
            body_chars.append(len(body))
            body_lines.append(body.count("\n") + 1)
        if RE_TRACEBACK.search(body):
            feats["traceback"] += 1
        if RE_CODE_FENCE.search(body):
            feats["code_fence"] += 1
        if RE_ISSUE_REF.search(body):
            feats["issue_ref"] += 1
        if RE_INLINE_CODE.search(body):
            feats["inline_code"] += 1

    pct = lambda n: round(100 * n / n_issues, 1) if n_issues else 0.0
    report = {
        "repo": repo,
        "totals": {"items": n_issues + n_prs, "issues": n_issues, "pull_requests": n_prs},
        "state": dict(state_counter),
        "issues_by_year": dict(sorted(year_counter.items())),
        "temporal_split_estimate": {
            "train_<=2024": sum(v for y, v in year_counter.items() if y <= "2024"),
            "eval_2025+": sum(v for y, v in year_counter.items() if y >= "2025"),
        },
        "labels": {
            "distinct": len(label_counter),
            "top20": label_counter.most_common(20),
            "module_like_top20": module_label_counter.most_common(20),
            "issues_with_module_label": issues_with_module_label,
            "issues_with_module_label_pct": pct(issues_with_module_label),
        },
        "issue_text_features_pct": {
            "has_body": pct(n_issue_with_body),
            "traceback": pct(feats["traceback"]),
            "code_fence": pct(feats["code_fence"]),
            "issue_ref": pct(feats["issue_ref"]),
            "inline_code": pct(feats["inline_code"]),
        },
        "issue_body_length": {"chars": _quantiles(body_chars), "lines": _quantiles(body_lines)},
    }
    return report


def render(report: dict) -> str:
    t = report["totals"]
    out = [
        f"\n=== EDA · {report['repo']} ===",
        f"Items: {t['items']}  (issues={t['issues']}, PRs={t['pull_requests']})",
        f"Estado: {report['state']}",
        f"Issues por año: {report['issues_by_year']}",
        f"Split temporal estimado: {report['temporal_split_estimate']}",
        "",
        f"Labels distintos: {report['labels']['distinct']}",
        f"Issues con label de módulo directo: {report['labels']['issues_with_module_label']} "
        f"({report['labels']['issues_with_module_label_pct']}%)",
        f"Módulos (top): {report['labels']['module_like_top20'][:10]}",
        "",
        "Características del texto del issue (% de issues):",
        f"  {report['issue_text_features_pct']}",
        f"Longitud body (chars): {report['issue_body_length']['chars']}",
        f"Longitud body (líneas): {report['issue_body_length']['lines']}",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    repo = (argv or sys.argv[1:])[0]
    report = analyze_repo(repo)
    print(render(report))
    out = DEFAULT_OUT.parent / "processed"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"eda_{slugify(repo)}.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eda] guardado en {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
