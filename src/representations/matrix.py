"""Matriz dispersa CSR mínima, numpy puro (sin scipy — coherente con §8/§14).

El corpus es disperso: cada issue activa decenas de features de un vocabulario de
decenas de miles. Una matriz densa sería ~GB; CSR guarda solo los no-ceros. Implementa
los dos productos que necesita la regresión logística:
    X @ W      (forward)   y   X.T @ G   (gradiente)
vectorizados con `np.bincount` (acumulación dispersa en C, rápida).
"""

from __future__ import annotations

from collections import Counter

import numpy as np


class CSR:
    """Compressed Sparse Row: (data, indices, indptr) + shape (n_rows, n_cols)."""

    def __init__(self, data, indices, indptr, shape: tuple[int, int]) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.shape = shape
        # fila de cada no-cero (para los productos dispersos)
        self._rows = np.repeat(np.arange(shape[0]), np.diff(self.indptr))

    @classmethod
    def from_rows(cls, rows: list[Counter | dict], n_cols: int) -> "CSR":
        """Construye desde una lista de filas `{col_index: valor}`."""
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for feats in rows:
            for j, v in feats.items():
                indices.append(j)
                data.append(v)
            indptr.append(len(data))
        return cls(data, indices, indptr, (len(rows), n_cols))

    @property
    def nnz(self) -> int:
        return len(self.data)

    def dot(self, W: np.ndarray) -> np.ndarray:
        """X @ W, con W denso (n_cols × k) → (n_rows × k).

        Acumula por clase con gathers 1D (pico de memoria O(nnz), no O(nnz·k)).
        """
        k = W.shape[1]
        out = np.empty((self.shape[0], k))
        for c in range(k):
            wc = W[self.indices, c] * self.data              # (nnz,)
            out[:, c] = np.bincount(self._rows, weights=wc, minlength=self.shape[0])
        return out

    def T_dot(self, G: np.ndarray) -> np.ndarray:
        """X.T @ G, con G denso (n_rows × k) → (n_cols × k)."""
        k = G.shape[1]
        out = np.empty((self.shape[1], k))
        for c in range(k):
            gc = G[self._rows, c] * self.data                # (nnz,)
            out[:, c] = np.bincount(self.indices, weights=gc, minlength=self.shape[1])
        return out

    def row_l2_normalize(self) -> "CSR":
        """Devuelve una copia con cada fila normalizada a norma L2 = 1."""
        sq = self.data ** 2
        row_norm = np.sqrt(np.bincount(self._rows, weights=sq, minlength=self.shape[0]))
        row_norm[row_norm == 0] = 1.0
        return CSR(self.data / row_norm[self._rows], self.indices, self.indptr, self.shape)
