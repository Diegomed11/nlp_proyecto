"""Análisis de errores lingüístico: dónde y por qué falla cada representación.

Pregunta clave: el keyword gana cuando el módulo se nombra. ¿Qué pasa cuando NO se nombra
(issues descritos por síntomas)? Ahí debería notarse el valor de una representación que
generaliza (TF-IDF) frente al match literal. Particiona eval en "nombre presente" vs
"ausente" y compara keyword vs TF-IDF+logreg.

    python -m src.eval.error_analysis
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from src.data.dataset import load_classes, load_split, labels_matrix, tokenized_split
from src.eval import metrics
from src.eval.baselines import KeywordBaseline
from src.models import MultiLabelLogReg
from src.preprocess.tokenizer import Tokenizer, TokenizerConfig
from src.representations import TfidfVectorizer


def name_present_mask(records, kw: KeywordBaseline, tok_ev) -> np.ndarray:
    """True si ALGÚN módulo verdadero del issue aparece por nombre en su texto."""
    out = np.zeros(len(records), dtype=bool)
    for i, (rec, toks) in enumerate(zip(records, tok_ev)):
        s = set(toks)
        out[i] = any(kw._hit(m, s) for m in rec["modules"])
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    classes = load_classes()
    tr, ev = load_split("train"), load_split("eval")
    Ytr, Yev = labels_matrix(tr, classes), labels_matrix(ev, classes)
    smask = Yev.sum(0) > 0

    tk = Tokenizer(TokenizerConfig(split_dotted=True))
    tok_tr, tok_ev = tokenized_split("train", tk, "split"), tokenized_split("eval", tk, "split")

    # --- predicciones ---
    kw = KeywordBaseline(classes)
    P_kw = kw.predict(tok_ev)

    vec = TfidfVectorizer(min_df=5)
    Xtr = vec.fit_transform(tok_tr).row_l2_normalize()
    Xev = vec.transform(tok_ev).row_l2_normalize()
    clf = MultiLabelLogReg(lr=0.05, n_epochs=150, l2=1e-5, class_weight=True, pos_weight_cap=10).fit(Xtr, Ytr)
    P_tfidf = clf.predict(Xev, 0.5)

    # --- partición por presencia del nombre del módulo ---
    present = name_present_mask(ev, kw, tok_ev)
    absent = ~present
    print(f"eval={len(ev)}  nombre presente={present.sum()}  ausente={absent.sum()}\n")
    print(f"{'subset':<18}{'keyword':>10}{'TF-IDF':>10}   (micro-F1)")
    for name, m in [("todo", np.ones(len(ev), bool)), ("nombre presente", present), ("nombre AUSENTE", absent)]:
        row = [metrics.micro_f1(Yev[m], P[m]) for P in (P_kw, P_tfidf)]
        print(f"{name:<18}" + "".join(f"{v:>10.3f}" for v in row))

    # --- confusión de módulos (issues single-label) ---
    print("\nTop confusiones de módulo (TF-IDF, issues de 1 módulo):")
    conf = Counter()
    for i in range(len(ev)):
        if Yev[i].sum() != 1:
            continue
        true = classes[int(np.argmax(Yev[i]))]
        for j in np.where(P_tfidf[i] == 1)[0]:
            if classes[j] != true:
                conf[(true, classes[j])] += 1
    for (t, p), c in conf.most_common(8):
        print(f"  {t:>12} → {p:<12} ×{c}")

    # --- ejemplos: nombre AUSENTE, TF-IDF acierta donde el keyword no puede ---
    print("\nEjemplos (nombre AUSENTE · keyword no puede · TF-IDF acierta):")
    shown = 0
    for i in np.where(absent)[0]:
        if (P_tfidf[i] == Yev[i]).all() and not (P_kw[i] == Yev[i]).all():
            true = [classes[j] for j in np.where(Yev[i] == 1)[0]]
            print(f"  #{ev[i]['number']} «{(ev[i]['title'] or '')[:68]}»  → {true}")
            shown += 1
            if shown >= 4:
                break


if __name__ == "__main__":
    main()
