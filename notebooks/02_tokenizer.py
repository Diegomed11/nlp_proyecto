"""Demo y validación del tokenizer técnico (Fase 1, notebook 02 del plan).

Genera: (1) tabla de decisiones con ejemplos antes/después, (2) un caso real con
traceback mostrando los 3 modos, (3) cobertura del detector de tracebacks en el
corpus. Correr:  python -m notebooks.02_tokenizer_demo   (o `python notebooks/...`).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess.tokenizer import Tokenizer, TokenizerConfig, _traceback_spans  # noqa: E402

TRAIN = ROOT / "data" / "processed" / "scipy" / "train.jsonl"
MARKER = "Traceback (most recent call last)"


def iter_train():
    with TRAIN.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def decision_cases() -> None:
    tk = Tokenizer()
    cases = [
        ("versión vs número", "numpy 1.2.3 and scipy v2.4.0 but pi is 3.14, n=100"),
        ("path vs identificador", "crash in scipy/sparse/_compressed.py:42 in scipy.sparse.linalg.spsolve"),
        ("identificadores", "csr_matrix with complex dtype and pm.sample()"),
        ("hash y dirección", "broke at commit a3f9e21d99, object at 0x7f3abc12"),
        ("ref y mención", "same as #1234, cc @rgommers and @tylerjereddy"),
    ]
    print("=" * 70, "\nTABLA DE DECISIONES (antes → después)\n" + "=" * 70)
    for name, txt in cases:
        print(f"\n[{name}]\n  IN : {txt}\n  OUT: {tk.tokenize(txt)}")


def traceback_case() -> None:
    print("\n" + "=" * 70, "\nCASO REAL CON TRACEBACK — los 3 modos\n" + "=" * 70)
    sample = next(
        (r for r in iter_train() if MARKER in (r.get("body") or "") and len(r["body"]) < 2500),
        None,
    )
    if not sample:
        print("  (sin ejemplo)"); return
    text = (sample["title"] or "") + "\n" + (sample["body"] or "")
    seg = Counter(k for k, _ in Tokenizer().segment(text))
    print(f"issue #{sample['number']}  módulos={sample['modules']}")
    print("segmentación:", dict(seg))
    for mode in ("keep", "drop", "special"):
        n = len(Tokenizer(TokenizerConfig(traceback=mode)).tokenize(text))
        print(f"   traceback={mode:8s}: {n} tokens")


def traceback_coverage() -> None:
    print("\n" + "=" * 70, "\nCOBERTURA E IMPACTO DEL TRACEBACK (train)\n" + "=" * 70)
    tk = Tokenizer()
    keep = Tokenizer(TokenizerConfig(traceback="keep"))
    drop = Tokenizer(TokenizerConfig(traceback="drop"))
    total = segmented = 0
    removed_frac = []
    for r in iter_train():
        b = r.get("body") or ""
        if MARKER not in b:
            continue
        total += 1
        full = (r["title"] or "") + "\n" + b
        if any(k == "traceback" for k, _ in tk.segment(full)):
            segmented += 1
        nk = len(keep.tokenize(full))
        nd = len(drop.tokenize(full))
        if nk:
            removed_frac.append((nk - nd) / nk)
    pct = lambda x: f"{100 * x / total:.0f}%" if total else "—"
    avg_removed = 100 * sum(removed_frac) / len(removed_frac) if removed_frac else 0
    print(f"issues con marcador de traceback: {total}")
    print(f"  segmentados con span 'traceback' (fenced+bare): {segmented} ({pct(segmented)})")
    print(f"  tokens removidos en promedio por traceback=drop: {avg_removed:.0f}%")
    print("  → el traceback es señal candidata Y ruido voluminoso: por eso se ablaciona")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    decision_cases()
    traceback_case()
    traceback_coverage()
