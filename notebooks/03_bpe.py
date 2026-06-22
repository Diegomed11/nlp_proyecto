"""Demo de BPE (Fase 3, notebook 03): tabla de merges + descomposición de OOV.

    python notebooks/03_bpe.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import tokenized_split  # noqa: E402
from src.preprocess.tokenizer import Tokenizer  # noqa: E402
from src.representations.bpe import BPE  # noqa: E402

SPECIAL = re.compile(r"^<[A-Z_]+>$")


def word_freqs(token_lists: list[list[str]]) -> Counter:
    c: Counter = Counter()
    for toks in token_lists:
        for t in toks:
            if not SPECIAL.match(t):   # los <TOKENS> especiales no se subdividen
                c[t] += 1
    return c


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    # identificadores enteros (split_dotted=False) para que BPE aprenda sus subpalabras
    toks = tokenized_split("train", Tokenizer(), "default")
    wf = word_freqs(toks)
    print(f"corpus: {sum(wf.values())} tokens, {len(wf)} palabras únicas")

    print("\nPrimeros 40 merges aprendidos:")
    bpe = BPE(num_merges=800, verbose=True).fit(wf)

    print("\nDescomposición de identificadores (OOV para BoW):")
    for w in ["csr_matrix", "__init__", "gammaln", "spsolve", "nanmean",
              "scipy.sparse", "convergencewarning", "lobpcg", "qmc"]:
        print(f"  {w:22s} -> {bpe.encode(w)}")


if __name__ == "__main__":
    main()
