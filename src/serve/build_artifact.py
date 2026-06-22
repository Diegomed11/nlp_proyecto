"""Entrena el baseline TF-IDF+logreg y serializa un artefacto liviano (§10, §11).

El artefacto (vocab + idf + pesos) pesa ~MB y se sirve con numpy puro. Es la evidencia de
criterio de ingeniería: producción puede correr el baseline con una sola dependencia.

    python -m src.serve.build_artifact
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.dataset import load_classes, load_split, labels_matrix, tokenized_split
from src.models import MultiLabelLogReg
from src.preprocess.tokenizer import Tokenizer, TokenizerConfig
from src.representations import TfidfVectorizer

ARTIFACT = Path("artifact/baseline")
SPLIT_DOTTED = True
MIN_DF = 5
SUBLINEAR = True
THRESHOLD = 0.5


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    classes = load_classes()
    # entrena con TODO lo etiquetado (producción): train + eval
    records = load_split("train") + load_split("eval")
    tk = Tokenizer(TokenizerConfig(split_dotted=SPLIT_DOTTED))
    tok_tr = tokenized_split("train", tk, "split") + tokenized_split("eval", tk, "split")
    Y = labels_matrix(records, classes)

    vec = TfidfVectorizer(min_df=MIN_DF, sublinear_tf=SUBLINEAR)
    X = vec.fit_transform(tok_tr).row_l2_normalize()
    clf = MultiLabelLogReg(lr=0.05, n_epochs=150, l2=1e-5, class_weight=True, pos_weight_cap=10).fit(X, Y)

    ARTIFACT.mkdir(parents=True, exist_ok=True)
    itos = [None] * len(vec.vocabulary_)
    for feat, idx in vec.vocabulary_.items():
        itos[idx] = feat
    (ARTIFACT / "vocab.json").write_text(json.dumps(itos, ensure_ascii=False), encoding="utf-8")
    np.savez(ARTIFACT / "params.npz", idf=vec.idf_, W=clf.W, b=clf.b)
    (ARTIFACT / "meta.json").write_text(json.dumps({
        "classes": classes,
        "split_dotted": SPLIT_DOTTED,
        "sublinear_tf": SUBLINEAR,
        "min_df": MIN_DF,
        "threshold": THRESHOLD,
        "n_features": len(itos),
        "n_train": len(records),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    size_kb = sum(f.stat().st_size for f in ARTIFACT.iterdir()) / 1024
    print(f"artefacto en {ARTIFACT}  ·  {len(itos)} features  ·  {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
