"""Métricas multi-label. Todas desde los conteos por clase, numpy puro.

- **F1 macro**: media de F1 por clase. Importa porque hay módulos raros (un modelo que
  solo acierta `stats` saca buen micro pero pésimo macro).
- **F1 micro**: F1 global agregando TP/FP/FN. Refleja el rendimiento ponderado por volumen.
- **Hamming loss**: fracción de etiquetas (doc×clase) mal puestas.
- **Subset accuracy**: fracción de issues con el conjunto de módulos EXACTO.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _counts(Yt: np.ndarray, Yp: np.ndarray):
    tp = ((Yt == 1) & (Yp == 1)).sum(axis=0).astype(float)
    fp = ((Yt == 0) & (Yp == 1)).sum(axis=0).astype(float)
    fn = ((Yt == 1) & (Yp == 0)).sum(axis=0).astype(float)
    return tp, fp, fn


def prf_per_class(Yt: np.ndarray, Yp: np.ndarray):
    tp, fp, fn = _counts(Yt, Yp)
    p = tp / np.maximum(tp + fp, EPS)
    r = tp / np.maximum(tp + fn, EPS)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, EPS), 0.0)
    return p, r, f1


def macro_f1(Yt: np.ndarray, Yp: np.ndarray, mask: np.ndarray | None = None) -> float:
    _, _, f1 = prf_per_class(Yt, Yp)
    if mask is not None:
        f1 = f1[mask]
    return float(f1.mean()) if len(f1) else 0.0


def micro_f1(Yt: np.ndarray, Yp: np.ndarray) -> float:
    tp, fp, fn = _counts(Yt, Yp)
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    p = TP / max(TP + FP, EPS)
    r = TP / max(TP + FN, EPS)
    return float(2 * p * r / max(p + r, EPS))


def hamming_loss(Yt: np.ndarray, Yp: np.ndarray) -> float:
    return float(np.mean(Yt != Yp))


def subset_accuracy(Yt: np.ndarray, Yp: np.ndarray) -> float:
    return float(np.mean((Yt == Yp).all(axis=1)))


def summary(Yt: np.ndarray, Yp: np.ndarray, support_mask: np.ndarray | None = None) -> dict:
    """Métricas agregadas. `support_mask` limita el macro a clases con soporte en eval."""
    return {
        "macro_f1": round(macro_f1(Yt, Yp), 4),
        "macro_f1_supported": round(macro_f1(Yt, Yp, support_mask), 4) if support_mask is not None else None,
        "micro_f1": round(micro_f1(Yt, Yp), 4),
        "hamming_loss": round(hamming_loss(Yt, Yp), 4),
        "subset_acc": round(subset_accuracy(Yt, Yp), 4),
    }


def per_class_table(Yt: np.ndarray, Yp: np.ndarray, classes: list[str]) -> str:
    p, r, f1 = prf_per_class(Yt, Yp)
    support = Yt.sum(axis=0).astype(int)
    order = np.argsort(-support)
    lines = [f"{'clase':<14}{'P':>6}{'R':>6}{'F1':>6}{'sup':>6}"]
    for i in order:
        lines.append(f"{classes[i]:<14}{p[i]:6.2f}{r[i]:6.2f}{f1[i]:6.2f}{support[i]:6d}")
    return "\n".join(lines)
