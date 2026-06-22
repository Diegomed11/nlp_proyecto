"""Baselines tontos que el NLP debe superar (§9.2).

Si el modelo no le gana a estos, no está aprendiendo nada útil:
- **Majority**: predice siempre el módulo más común (`stats`). Mide el desbalance.
- **Keyword**: predice el módulo `X` si su nombre aparece en el texto (`scipy.sparse`,
  `sparse`…). Exploit barato pero fuerte: el dominio nombra su propio módulo. La pregunta
  del proyecto es cuánto le gana el modelo a esto.
"""

from __future__ import annotations

import numpy as np


class MajorityBaseline:
    def fit(self, Y: np.ndarray) -> "MajorityBaseline":
        self.top_ = int(Y.sum(axis=0).argmax())
        self.K_ = Y.shape[1]
        return self

    def predict(self, n_docs: int) -> np.ndarray:
        P = np.zeros((n_docs, self.K_), dtype=np.int64)
        P[:, self.top_] = 1
        return P


class KeywordBaseline:
    def __init__(self, classes: list[str]) -> None:
        self.classes = classes

    def _hit(self, name: str, tokens: set[str]) -> bool:
        # el nombre del módulo como token exacto o como componente de uno con puntos
        return any(
            tok == name or tok.endswith("." + name) or tok.startswith(name + ".")
            or ("." + name + ".") in tok
            for tok in tokens
        )

    def predict(self, docs_tokens: list[list[str]]) -> np.ndarray:
        P = np.zeros((len(docs_tokens), len(self.classes)), dtype=np.int64)
        for i, toks in enumerate(docs_tokens):
            s = set(toks)
            for c, name in enumerate(self.classes):
                if self._hit(name, s):
                    P[i, c] = 1
        return P
