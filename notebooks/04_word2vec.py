"""Demo de word2vec (Fase 3, notebook 04): vecinos de dominio.

    python notebooks/04_word2vec.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import tokenized_split  # noqa: E402
from src.preprocess.tokenizer import Tokenizer  # noqa: E402
from src.representations.word2vec import Word2Vec  # noqa: E402

QUERIES = ["sparse", "optimize", "matrix", "convergence", "fft",
           "interpolate", "minimize", "memory", "eigenvalues", "nan"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = tokenized_split("train", Tokenizer(), "default")
    t0 = time.time()
    w2v = Word2Vec(dim=100, window=5, negatives=5, min_count=10, epochs=5, seed=0).fit(corpus)
    print(f"entrenado en {time.time() - t0:.0f}s\n")

    for q in QUERIES:
        nbrs = w2v.most_similar(q, topn=8)
        if nbrs:
            print(f"{q:14s} → " + ", ".join(f"{w}({s:.2f})" for w, s in nbrs))
        else:
            print(f"{q:14s} → (OOV)")


if __name__ == "__main__":
    main()
