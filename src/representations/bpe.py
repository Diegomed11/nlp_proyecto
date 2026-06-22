"""Byte-Pair Encoding from scratch (§7): subpalabras reutilizables para resolver OOV.

Los identificadores (`csr_matrix`, `__init__`, `gammaln`, `pm.sample`) revientan el
vocabulario y son casi siempre OOV para BoW/TF-IDF. BPE los descompone en subpalabras
reutilizables aprendidas del corpus: aprende `_matrix`, `np`, `__`, `scipy`… La tabla de
merges es la evidencia de que se entiende el mecanismo de subpalabras (el mismo que usan
los tokenizers subword modernos para no quedarse sin vocabulario).

Algoritmo (Sennrich et al., 2016): cada palabra = secuencia de caracteres + `</w>`; se
cuenta el par adyacente más frecuente del corpus y se fusiona; se repite N veces.
"""

from __future__ import annotations

from collections import Counter

END = "</w>"


class BPE:
    def __init__(self, num_merges: int = 1000, verbose: bool = False) -> None:
        self.num_merges = num_merges
        self.verbose = verbose
        self.merges: list[tuple[str, str]] = []     # en orden de aprendizaje
        self.ranks: dict[tuple[str, str], int] = {}

    @staticmethod
    def _pairs(vocab: dict[tuple, int]) -> Counter:
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _apply_merge(vocab: dict[tuple, int], a: str, b: str) -> dict[tuple, int]:
        merged = a + b
        out: dict[tuple, int] = {}
        for word, freq in vocab.items():
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            out[tuple(new_word)] = freq
        return out

    def fit(self, word_freqs: dict[str, int]) -> "BPE":
        vocab = {tuple(w) + (END,): f for w, f in word_freqs.items() if w}
        for step in range(self.num_merges):
            pairs = self._pairs(vocab)
            if not pairs:
                break
            # desempate determinista (frecuencia, luego el par) para reproducibilidad
            (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            vocab = self._apply_merge(vocab, a, b)
            self.merges.append((a, b))
            if self.verbose and step < 40:
                print(f"  merge {step:3d}: {a!r}+{b!r} -> {a + b!r}")
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self

    def encode(self, word: str) -> list[str]:
        """Aplica los merges aprendidos (por rango) a una palabra → subpalabras."""
        if not word:
            return []
        symbols = list(word) + [END]
        while len(symbols) > 1:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(symbols) - 1):
                r = self.ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            symbols[best_i : best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        return symbols
