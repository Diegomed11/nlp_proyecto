"""Inferencia del baseline desde el artefacto serializado (§10). Solo numpy + stdlib.

    from src.serve.predict import BaselinePredictor
    p = BaselinePredictor()
    p.predict("csr_matrix multiplication wrong with complex dtype", "...")
    # -> [('sparse', 0.94), ('linalg', 0.21)]

CLI (para la GitHub Action): lee título y cuerpo, imprime JSON de módulos sobre el umbral.
    python -m src.serve.predict --title "..." --body "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.preprocess.tokenizer import Tokenizer, TokenizerConfig
from src.representations import TfidfVectorizer

ARTIFACT = Path("artifact/baseline")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class BaselinePredictor:
    def __init__(self, artifact_dir: Path = ARTIFACT) -> None:
        itos = json.loads((artifact_dir / "vocab.json").read_text(encoding="utf-8"))
        params = np.load(artifact_dir / "params.npz")
        self.meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
        self.classes = self.meta["classes"]
        self.W, self.b = params["W"], params["b"]
        self.tokenizer = Tokenizer(TokenizerConfig(split_dotted=self.meta["split_dotted"]))
        self.vec = TfidfVectorizer(min_df=self.meta["min_df"], sublinear_tf=self.meta["sublinear_tf"])
        self.vec.counts.vocabulary_ = {feat: i for i, feat in enumerate(itos)}
        self.vec.idf_ = params["idf"]

    def predict(self, title: str, body: str = "", threshold: float | None = None) -> list[tuple[str, float]]:
        thr = self.meta["threshold"] if threshold is None else threshold
        tokens = self.tokenizer.tokenize((title or "") + "\n" + (body or ""))
        X = self.vec.transform([tokens]).row_l2_normalize()
        probs = _sigmoid(X.dot(self.W) + self.b)[0]
        ranked = sorted(((self.classes[i], float(probs[i])) for i in range(len(self.classes))),
                        key=lambda kv: -kv[1])
        above = [(c, round(p, 4)) for c, p in ranked if p >= thr]
        # si nada supera el umbral, devuelve el más probable (mejor sugerencia)
        return above or [(ranked[0][0], round(ranked[0][1], 4))]


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--body", default="")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args(argv)
    preds = BaselinePredictor().predict(args.title, args.body, args.threshold)
    print(json.dumps({"modules": [{"module": c, "confidence": p} for c, p in preds]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
