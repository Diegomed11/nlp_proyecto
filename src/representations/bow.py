"""Bag-of-Words: matriz documento-término de conteos, from scratch (§7).

El modelo de bolsa pierde el orden (lo recupera parcialmente `ngram_range`). Aprende el
vocabulario en `fit` (con filtro `min_df` y tope `max_features`) y vectoriza en
`transform`. Devuelve una matriz CSR propia.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .matrix import CSR
from .ngrams import word_ngrams


class CountVectorizer:
    """Vectorizador de conteos. `fit`/`transform`/`fit_transform` sobre docs ya tokenizados."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 3,
        max_features: int | None = None,
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_: dict[str, int] = {}
        self.df_: np.ndarray = np.array([])      # doc-frequency por feature (orden del vocab)
        self.n_docs_fit_: int = 0

    def fit(self, docs: list[list[str]]) -> "CountVectorizer":
        df: Counter = Counter()
        for toks in docs:
            for feat in set(word_ngrams(toks, self.ngram_range)):
                df[feat] += 1
        items = [(f, c) for f, c in df.items() if c >= self.min_df]
        # orden estable: por frecuencia desc, luego alfabético (reproducible)
        items.sort(key=lambda fc: (-fc[1], fc[0]))
        if self.max_features is not None:
            items = items[: self.max_features]
        self.vocabulary_ = {f: i for i, (f, _) in enumerate(items)}
        self.df_ = np.array([c for _, c in items], dtype=np.float64)
        self.n_docs_fit_ = len(docs)
        return self

    def transform(self, docs: list[list[str]]) -> CSR:
        vocab = self.vocabulary_
        rows: list[Counter] = []
        for toks in docs:
            counts: Counter = Counter()
            for feat in word_ngrams(toks, self.ngram_range):
                j = vocab.get(feat)
                if j is not None:
                    counts[j] += 1
            rows.append(counts)
        return CSR.from_rows(rows, len(vocab))

    def fit_transform(self, docs: list[list[str]]) -> CSR:
        return self.fit(docs).transform(docs)
