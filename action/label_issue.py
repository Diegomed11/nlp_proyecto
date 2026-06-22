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

    info = BaselinePredictor().explain(title, body)
    ranked = info["modules"]                                 # [(módulo, prob), ...] ordenado
    labels = [f"scipy.{c}" for c, p in ranked if p >= thr]   # auto-etiqueta solo alta confianza

    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"labels={json.dumps(labels)}\n")

    print(json.dumps({
        "top_suggestions": [{"module": c, "confidence": round(p, 4)} for c, p in ranked[:3]],
        "auto_labels": labels,
        "key_terms": info["key_terms"],
        "location": info["location"],
        "frames": info["frames"],
        "threshold": thr,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
