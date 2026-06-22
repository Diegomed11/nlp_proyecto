"""word2vec from scratch (§7): hipótesis distribucional, skip-gram + negative sampling.

"Una palabra se conoce por la compañía que tiene." Entrena embeddings densos prediciendo
el contexto de cada palabra. En el dominio, palabras que comparten contexto caen juntas
(`sparse`~`csr`, `optimize`~`minimize`, `fft`~`fourier`) — eso es lo que TF-IDF no captura.

Vectorizado en numpy (motor de álgebra; la lógica —pares, neg-sampling, gradiente SGNS—
es propia). Init estándar SGNS: W_in uniforme pequeño, W_out en cero.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Word2Vec:
    def __init__(
        self,
        dim: int = 100,
        window: int = 5,
        negatives: int = 5,
        min_count: int = 5,
        epochs: int = 5,
        lr: float = 0.025,
        subsample: float = 1e-3,
        batch_size: int = 4096,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.window = window
        self.negatives = negatives
        self.min_count = min_count
        self.epochs = epochs
        self.lr = lr
        self.subsample = subsample
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    # -- vocabulario ------------------------------------------------------
    def build_vocab(self, corpus: list[list[str]]) -> None:
        from collections import Counter

        counts: Counter = Counter()
        for sent in corpus:
            counts.update(sent)
        items = [(w, c) for w, c in counts.items() if c >= self.min_count]
        items.sort(key=lambda wc: (-wc[1], wc[0]))
        self.itos = [w for w, _ in items]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.freq = np.array([c for _, c in items], dtype=np.float64)
        V = len(self.itos)
        # tabla de negative sampling ~ freq^0.75
        p = self.freq ** 0.75
        self.neg_p = p / p.sum()
        # prob de descarte por subsampling (Mikolov)
        total = self.freq.sum()
        f = self.freq / total
        self.keep_p = np.minimum(1.0, np.sqrt(self.subsample / f) + self.subsample / f)
        self.W_in = (self.rng.uniform(-0.5, 0.5, (V, self.dim)) / self.dim)
        self.W_out = np.zeros((V, self.dim))

    # -- generación de pares (vectorizada por offset) ---------------------
    def _pairs(self, corpus: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        centers: list[np.ndarray] = []
        contexts: list[np.ndarray] = []
        for sent in corpus:
            ids = [self.stoi[w] for w in sent if w in self.stoi]
            if len(ids) < 2:
                continue
            ids = np.array(ids)
            keep = self.rng.random(len(ids)) < self.keep_p[ids]   # subsampling
            ids = ids[keep]
            L = len(ids)
            if L < 2:
                continue
            for d in range(1, self.window + 1):
                if d >= L:
                    break
                centers.append(ids[:-d]); contexts.append(ids[d:])   # hacia adelante
                centers.append(ids[d:]); contexts.append(ids[:-d])   # hacia atrás
        if not centers:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.concatenate(centers), np.concatenate(contexts)

    # -- entrenamiento ----------------------------------------------------
    def fit(self, corpus: list[list[str]], verbose: bool = True) -> "Word2Vec":
        self.build_vocab(corpus)
        c_all, o_all = self._pairs(corpus)
        n = len(c_all)
        if verbose:
            print(f"vocab={len(self.itos)}  pares={n}")
        k, d, V = self.negatives, self.dim, len(self.itos)
        for epoch in range(self.epochs):
            perm = self.rng.permutation(n)
            c_all, o_all = c_all[perm], o_all[perm]
            lr = self.lr * (1 - epoch / max(self.epochs, 1) * 0.9)  # decay lineal
            for start in range(0, n, self.batch_size):
                c = c_all[start : start + self.batch_size]
                o = o_all[start : start + self.batch_size]
                B = len(c)
                neg = self.rng.choice(V, size=(B, k), p=self.neg_p)
                vc, vo, vneg = self.W_in[c], self.W_out[o], self.W_out[neg]
                g_pos = _sigmoid((vc * vo).sum(1)) - 1.0                # (B,)
                s_neg = _sigmoid(np.einsum("bd,bkd->bk", vc, vneg))     # (B,k)
                grad_vc = g_pos[:, None] * vo + np.einsum("bk,bkd->bd", s_neg, vneg)
                grad_vo = g_pos[:, None] * vc
                grad_vneg = s_neg[:, :, None] * vc[:, None, :]
                np.add.at(self.W_in, c, -lr * grad_vc)
                np.add.at(self.W_out, o, -lr * grad_vo)
                np.add.at(self.W_out, neg.reshape(-1), (-lr * grad_vneg).reshape(-1, d))
            if verbose:
                print(f"  epoch {epoch + 1}/{self.epochs}  lr={lr:.4f}")
        # normaliza para coseno
        self.vectors = self.W_in / (np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-9)
        return self

    # -- consulta ---------------------------------------------------------
    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        if word not in self.stoi:
            return []
        v = self.vectors[self.stoi[word]]
        sims = self.vectors @ v
        order = np.argsort(-sims)
        return [(self.itos[i], float(sims[i])) for i in order if i != self.stoi[word]][:topn]
