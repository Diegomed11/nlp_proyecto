"""N-gramas de palabras: capturan el orden local que el modelo de bolsa pierde (§7).

Ejemplo firma del plan: "does not converge" (trigrama) lleva información que los
unigramas {does, not, converge} no: la negación. `ngram_range=(1,2)` añade bigramas.
"""

from __future__ import annotations

NGRAM_SEP = " "  # "does not" como un solo feature


def word_ngrams(tokens: list[str], ngram_range: tuple[int, int] = (1, 1)) -> list[str]:
    """Genera los n-gramas de `tokens` para n en [lo, hi]."""
    lo, hi = ngram_range
    if lo < 1:
        raise ValueError("ngram_range debe empezar en >=1")
    out: list[str] = []
    n = len(tokens)
    for k in range(lo, hi + 1):
        if k == 1:
            out.extend(tokens)
        else:
            for i in range(n - k + 1):
                out.append(NGRAM_SEP.join(tokens[i : i + k]))
    return out
