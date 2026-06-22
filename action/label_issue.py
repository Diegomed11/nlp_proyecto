"""Predice el módulo de un issue desde variables de entorno (para la GitHub Action).

Lee ISSUE_TITLE / ISSUE_BODY, predice con el baseline (numpy), y propone etiquetas. Solo
auto-etiqueta con confianza alta (§2: el sistema es asistente, no automatización total —
lo dudoso se deja al humano). Escribe `labels=[...]` en GITHUB_OUTPUT.

    ISSUE_TITLE="..." ISSUE_BODY="..." python -m action.label_issue
"""

from __future__ import annotations

import json
import os
import sys

from src.serve.predict import BaselinePredictor


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    title = os.environ.get("ISSUE_TITLE", "")
    body = os.environ.get("ISSUE_BODY", "")
    thr = float(os.environ.get("CONF_THRESHOLD", "0.9"))

    ranked = BaselinePredictor().predict(title, body, threshold=0.0)   # todos, ordenados
    auto = [c for c, p in ranked if p >= thr]                          # solo alta confianza
    labels = [f"scipy.{c}" for c in auto]

    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"labels={json.dumps(labels)}\n")

    print(json.dumps({
        "top_suggestions": [{"module": c, "confidence": p} for c, p in ranked[:3]],
        "auto_labels": labels,
        "threshold": thr,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
