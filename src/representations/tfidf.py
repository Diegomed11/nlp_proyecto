"""TF-IDF from scratch (§7): IDF como medida de información.

Una palabra que aparece en casi todos los issues (`error`, `python`) informa poco; una
que aparece en pocos (`gammaln`, `csr_matrix`) discrimina el módulo. IDF formaliza eso.
Se usa el IDF suavizado (estilo sklearn) y normalización L2 por fila para que documentos
largos y cortos sean comparables.

    idf(t) = ln((1 + n) / (1 + df(t))) + 1
"""

from __future__ import annotations

import numpy as np

from .bow import CountVectorizer
from .matrix import CSR


class TfidfVectorizer:
    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 3,
        max_features: int | None = None,
        sublinear_tf: bool = True,
    ) -> None:
        self.counts = CountVectorizer(ngram_range, min_df, max_features)
        self.sublinear_tf = sublinear_tf  # tf -> 1+log(tf): atenúa repeticiones
        self.idf_: np.ndarray = np.array([])

    def fit(self, docs: list[list[str]]) -> "TfidfVectorizer":
        self.counts.fit(docs)
        n = self.counts.n_docs_fit_
        self.idf_ = np.log((1.0 + n) / (1.0 + self.counts.df_)) + 1.0
        return self

    def transform(self, docs: list[list[str]]) -> CSR:
        X = self.counts.transform(docs)              # conteos
        tf = np.where(X.data > 0, 1.0 + np.log(X.data), 0.0) if self.sublinear_tf else X.data
        weighted = tf * self.idf_[X.indices]         # tf * idf
        return CSR(weighted, X.indices, X.indptr, X.shape).row_l2_normalize()

    def fit_transform(self, docs: list[list[str]]) -> CSR:
        return self.fit(docs).transform(docs)

    @property
    def vocabulary_(self) -> dict[str, int]:
        return self.counts.vocabulary_
