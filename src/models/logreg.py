"""Regresión logística multi-label .

One-vs-rest: K clasificadores binarios que comparten la matriz de features, como una
sola matriz de pesos W (n_features × K) + sesgo b. Sigmoide por clase, pérdida de
entropía cruzada binaria, regularización L2, descenso de gradiente full-batch sobre la
matriz dispersa CSR. `class_weight` sube el peso de los positivos (clases raras).

numpy es solo el motor de álgebra; la lógica (forward, BCE, gradiente, CSR) es propia.
"""

from __future__ import annotations

import numpy as np

from src.representations.matrix import CSR


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoide numéricamente estable."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class MultiLabelLogReg:
    """Logreg multi-label con Adam (converge rápido) y pesos de clase acotados.

    Adam (lr≈0.05) en vez de GD full-batch (que sub-converge: las features casi
    perfectas como `scipy.stats`→stats requieren pesos grandes que GD tarda miles de
    épocas en alcanzar). `class_weight` con tope `pos_weight_cap` ayuda a las clases
    raras sin que sus pesos enormes (datasets ~1848×) inflen los falsos positivos.
    """

    def __init__(
        self,
        l2: float = 1e-5,
        lr: float = 0.05,
        n_epochs: int = 200,
        class_weight: bool = False,
        pos_weight_cap: float = 20.0,
        optimizer: str = "adam",
        verbose: bool = False,
    ) -> None:
        self.l2 = l2
        self.lr = lr
        self.n_epochs = n_epochs
        self.class_weight = class_weight
        self.pos_weight_cap = pos_weight_cap
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X: CSR, Y: np.ndarray) -> "MultiLabelLogReg":
        n, n_features = X.shape
        K = Y.shape[1]
        Y = Y.astype(np.float64)
        self.W = np.zeros((n_features, K))
        self.b = np.zeros(K)

        if self.class_weight:
            pos = Y.sum(axis=0)
            pw = np.where(pos > 0, (n - pos) / np.maximum(pos, 1.0), 1.0)
            self.pos_w_ = np.minimum(pw, self.pos_weight_cap)
        else:
            self.pos_w_ = np.ones(K)

        # estado de Adam
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, self.n_epochs + 1):
            P = _sigmoid(X.dot(self.W) + self.b)
            w = np.where(Y == 1.0, self.pos_w_[None, :], 1.0)   # peso por positivo
            R = w * (P - Y)                                     # residual (n × K)
            dW = X.T_dot(R) / n + self.l2 * self.W
            db = R.mean(axis=0)

            if self.optimizer == "adam":
                mW = b1 * mW + (1 - b1) * dW
                vW = b2 * vW + (1 - b2) * dW * dW
                mb = b1 * mb + (1 - b1) * db
                vb = b2 * vb + (1 - b2) * db * db
                mWh, vWh = mW / (1 - b1 ** epoch), vW / (1 - b2 ** epoch)
                mbh, vbh = mb / (1 - b1 ** epoch), vb / (1 - b2 ** epoch)
                self.W -= self.lr * mWh / (np.sqrt(vWh) + eps)
                self.b -= self.lr * mbh / (np.sqrt(vbh) + eps)
            else:  # descenso de gradiente simple
                self.W -= self.lr * dW
                self.b -= self.lr * db

            if self.verbose and (epoch % 50 == 0 or epoch == self.n_epochs):
                le = 1e-12
                loss = -np.mean(w * (Y * np.log(P + le) + (1 - Y) * np.log(1 - P + le)))
                print(f"  epoch {epoch:4d}  loss={loss:.4f}")
        return self

    def predict_proba(self, X: CSR) -> np.ndarray:
        return _sigmoid(X.dot(self.W) + self.b)

    def predict(self, X: CSR, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
