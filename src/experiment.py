"""Experimentos clásicos (Fase 2, notebook 04): baselines vs BoW/TF-IDF/n-grams + logreg.

Todo bajo el split TEMPORAL (train ≤2024, eval 2025+). Responde a §9.2: ¿le gana el NLP
a los baselines tontos, y cuánto aporta IDF, el orden local (n-gramas) y splitear los
identificadores con punto?

Config honesta: umbral fijo 0.5 (sin tunear en test); el desbalance se maneja con pesos
de clase acotados (cap=10), no ajustando el umbral sobre la evaluación.

    python -m src.experiment
"""

from __future__ import annotations

import sys
import time

import numpy as np

from src.dataset import load_classes, load_split, labels_matrix, tokenized_split
from src.eval import metrics
from src.eval.baselines import KeywordBaseline, MajorityBaseline
from src.models import MultiLabelLogReg
from src.preprocess.tokenizer import Tokenizer, TokenizerConfig
from src.representations import CountVectorizer, TfidfVectorizer

EPOCHS = 150
CAP = 10.0


def _logreg() -> MultiLabelLogReg:
    return MultiLabelLogReg(lr=0.05, n_epochs=EPOCHS, l2=1e-5, class_weight=True, pos_weight_cap=CAP)


def _print_table(rows: list[tuple[str, dict]]) -> None:
    cols = ["macro_f1", "macro_f1_supported", "micro_f1", "subset_acc", "hamming_loss"]
    labels = {"macro_f1": "macroF1", "macro_f1_supported": "macroF1sup", "micro_f1": "microF1",
              "subset_acc": "subset", "hamming_loss": "hamming"}
    head = f"{'modelo':<22}" + "".join(f"{labels[c]:>11}" for c in cols) + f"{'feats':>9}"
    print("\n" + head + "\n" + "-" * len(head))
    for name, m in rows:
        print(f"{name:<22}" + "".join(f"{m.get(c, 0):>11}" for c in cols) + f"{m.get('feats', ''):>9}")


def run() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    classes = load_classes()
    tr, ev = load_split("train"), load_split("eval")
    Ytr, Yev = labels_matrix(tr, classes), labels_matrix(ev, classes)
    support_mask = Yev.sum(axis=0) > 0
    print(f"train={len(tr)}  eval={len(ev)}  clases={len(classes)}  "
          f"(con soporte en eval: {int(support_mask.sum())})")

    # Tokens: con y sin split de identificadores con punto (cacheados)
    t0 = time.time()
    tok = tokenized_split("train", Tokenizer(TokenizerConfig(split_dotted=True)), "split")
    tev = tokenized_split("eval", Tokenizer(TokenizerConfig(split_dotted=True)), "split")
    tok_ns = tokenized_split("train", Tokenizer(), "default")
    tev_ns = tokenized_split("eval", Tokenizer(), "default")
    print(f"tokens listos en {time.time() - t0:.1f}s")

    rows: list[tuple[str, dict]] = []

    # --- baselines ---
    maj = MajorityBaseline().fit(Ytr)
    rows.append(("majority", metrics.summary(Yev, maj.predict(len(ev)), support_mask)))
    rows.append(("keyword", metrics.summary(Yev, KeywordBaseline(classes).predict(tev), support_mask)))

    # --- representaciones + logreg ---
    experiments = [
        ("BoW+logreg", CountVectorizer(min_df=5), tok, tev),
        ("TFIDF(1)+logreg", TfidfVectorizer(min_df=5), tok, tev),
        ("TFIDF(1,2)+logreg", TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=50000), tok, tev),
        ("TFIDF(1) sin-split", TfidfVectorizer(min_df=5), tok_ns, tev_ns),  # ablación
    ]
    best = None
    for name, vec, a, b in experiments:
        t0 = time.time()
        Xtr = vec.fit_transform(a).row_l2_normalize()
        Xev = vec.transform(b).row_l2_normalize()
        clf = _logreg().fit(Xtr, Ytr)
        Yp = clf.predict(Xev, threshold=0.5)
        m = metrics.summary(Yev, Yp, support_mask)
        m["feats"] = Xtr.shape[1]
        rows.append((name, m))
        print(f"  [{name}] {time.time() - t0:.0f}s")
        if name == "TFIDF(1)+logreg":
            best = (name, Yp)

    _print_table(rows)
    print("\n(referencia: el keyword es un baseline muy fuerte porque los issues nombran su módulo)")

    if best is not None:
        name, Yp = best
        print(f"\nPor clase — {name} (eval):\n")
        print(metrics.per_class_table(Yev, Yp, classes))


if __name__ == "__main__":
    run()
