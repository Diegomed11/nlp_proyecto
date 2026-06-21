"""Construye el dataset curado de SciPy para clasificación de módulo (tarea núcleo).

Decisiones de curación (se documentan en data/datasheet.md):
- **Solo issues** (no PRs): el sistema clasifica issues entrantes.
- **Solo issues con ≥1 label `scipy.X`** (ground truth directo de los maintainers,
  ~70% de cobertura, consistente todos los años). Sin weak supervision: el alcance
  es enfocado a SciPy con etiqueta directa.
- **Etiqueta = subpaquete nivel `scipy.X`**: `scipy.sparse.linalg` → `sparse`. Es el
  nivel de granularidad útil para triaje (enrutar al área correcta). **Multi-label**.
- **Split temporal** (§5): train ≤2024, eval 2025+. No aleatorio: simula predecir el
  futuro y no filtra vocabulario de releases entre train y test.

CLI:
    python -m src.dataset                 # genera data/processed/scipy/{train,eval}.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.extract.fetch import DEFAULT_OUT, is_pull_request, slugify

PROCESSED = DEFAULT_OUT.parent / "processed"
REPO = "scipy/scipy"
TRAIN_MAX_YEAR = 2024  # train ≤2024, eval ≥2025

# Captura el subpaquete de un label de módulo: "scipy.sparse.linalg" -> "sparse".
MODULE_RE = re.compile(r"^scipy\.([A-Za-z_]\w*)")


def issue_modules(item: dict) -> list[str]:
    """Módulos `scipy.X` (subpaquete) de un issue, deduplicados y ordenados."""
    mods = set()
    for lbl in item.get("labels", []):
        if not isinstance(lbl, dict):
            continue
        m = MODULE_RE.match(lbl.get("name", ""))
        if m:
            mods.add(m.group(1))
    return sorted(mods)


def build_records(repo: str = REPO) -> list[dict]:
    """Issues con ≥1 módulo, proyectados a registros curados."""
    records = []
    for it in _iter_pages(repo):
        if is_pull_request(it):
            continue
        mods = issue_modules(it)
        if not mods:
            continue
        created = it.get("created_at") or ""
        records.append(
            {
                "number": it.get("number"),
                "title": it.get("title") or "",
                "body": it.get("body") or "",
                "modules": mods,
                "created_at": created,
                "year": int(created[:4]) if created[:4].isdigit() else None,
                "state": it.get("state"),
                "html_url": it.get("html_url"),
            }
        )
    return records


def _iter_pages(repo: str):
    pages_dir = DEFAULT_OUT / slugify(repo) / "issues"
    for pf in sorted(pages_dir.glob("page*.json")):
        try:
            yield from json.loads(pf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue


def temporal_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    train = [r for r in records if r["year"] is not None and r["year"] <= TRAIN_MAX_YEAR]
    eval_ = [r for r in records if r["year"] is not None and r["year"] > TRAIN_MAX_YEAR]
    return train, eval_


def _class_dist(records: list[dict]) -> dict:
    c = Counter()
    for r in records:
        for m in r["modules"]:
            c[m] += 1
    return dict(c.most_common())


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    records = build_records()
    train, eval_ = temporal_split(records)
    vocab = sorted({m for r in records for m in r["modules"]})

    out = PROCESSED / "scipy"
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(eval_, out / "eval.jsonl")

    multilabel = sum(len(r["modules"]) > 1 for r in records)
    meta = {
        "repo": REPO,
        "task": "module classification (multi-label)",
        "label_granularity": "scipy.X subpackage",
        "split": "temporal (train<=2024, eval>=2025)",
        "n_total": len(records),
        "n_train": len(train),
        "n_eval": len(eval_),
        "n_classes": len(vocab),
        "classes": vocab,
        "multilabel_issues": multilabel,
        "class_dist_train": _class_dist(train),
        "class_dist_eval": _class_dist(eval_),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dataset curado SciPy → {out}")
    print(f"  total={len(records)}  train(≤2024)={len(train)}  eval(2025+)={len(eval_)}")
    print(f"  clases={len(vocab)}  multi-label={multilabel} ({100*multilabel/len(records):.1f}%)")
    print(f"  vocab={vocab}")
    print("\n  distribución train:", meta["class_dist_train"])
    print("\n  distribución eval :", meta["class_dist_eval"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
