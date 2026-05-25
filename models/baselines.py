"""
STAGE-9: Baseline modeller — k-NN, SVM, LSTM, hafif TFT.

Tüm modeller q={0.1, 0.5, 0.9} quantile çıktı üretir.

API:
    KNNQuantile                                — k komşu quantile regressor
    SVMQuantile                                — Nystroem + LinearSVR + split-conformal
    LSTMQuantile                               — 2 katmanlı LSTM, MPS/CPU
    LightTFTQuantile                           — LSTM + attention + GRN, MPS/CPU
    train_baseline(name, X, y, ...)            → fitted model
    predict_baseline(name, model, X)           → dict[str, np.ndarray]
    evaluate_baseline(name, model, X, y)       → dict[str, float]
    train_all_baselines(X_tr, y_tr, X_te, y_te) → dict

Kurallar:
    - Serileştirme: joblib (pickle yasak)
    - Cihaz: MPS öncelikli (Apple M4), fallback CPU
    - Walk-forward mantığı: zaman sıralı, seq_len adım geri bakış
    - pytorch_forecasting gerektirmez; hafif TFT sıfırdan implement edildi
"""

import logging
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

log = logging.getLogger(__name__)

# ── Sabitler ───────────────────────────────────────────────────────────────────

QUANTILES:  tuple[float, ...] = (0.1, 0.5, 0.9)
SEQ_LEN:    int   = 12     # 12 × 5 dk = 1 saat geri bakış
BATCH_SIZE: int   = 512
EPOCHS:     int   = 30
LR:         float = 1e-3
HIDDEN:     int   = 64
D_MODEL:    int   = 64
N_HEAD:     int   = 4
PATIENCE:   int   = 5      # erken durdurma sabrı
KNN_K:      int   = 10
SVM_COMP:   int   = 500
CALIB_FRAC: float = 0.20   # SVMQuantile: son %20 kalibrasyon seti


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tabular (n, f) → dizi (n−seq_len, seq_len, f).
    Hedef y[seq_len:] — dizi i, X[i:i+seq_len] girdisiyle y[i+seq_len-1]'i tahmin eder.
    """
    n = len(X)
    xs = np.stack([X[i: i + seq_len] for i in range(n - seq_len)])
    return xs, y[seq_len:]


def _pb(y: torch.Tensor, pred: torch.Tensor, q: float) -> torch.Tensor:
    """Gözlem başına pinball kaybı ortalaması."""
    r = y - pred
    return torch.where(r >= 0, q * r, (q - 1) * r).mean()


def _pb_total(y: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """Üç quantile pinball toplamı.  y: (B,)  preds: (B, 3)."""
    return sum(_pb(y, preds[:, i], q) for i, q in enumerate(QUANTILES))


# ── k-NN Quantile ──────────────────────────────────────────────────────────────

class KNNQuantile:
    """
    k komşunun y değerlerinden quantile alır (proper quantile k-NN).
    sklearn.neighbors.KNeighborsRegressor kneighbors() ile komşu indisler alınır.
    """

    def __init__(self, n_neighbors: int = KNN_K) -> None:
        self.n_neighbors = n_neighbors
        self._knn  = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        self._y_tr: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNQuantile":
        self._knn.fit(X, y)
        self._y_tr = np.asarray(y, dtype=np.float32)
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[str, np.ndarray]:
        _, idx = self._knn.kneighbors(X)          # (n, k)
        ny = self._y_tr[idx]                       # (n, k)
        return {
            "q01": np.quantile(ny, 0.1, axis=1),
            "q05": np.quantile(ny, 0.5, axis=1),
            "q09": np.quantile(ny, 0.9, axis=1),
        }


# ── SVM Quantile ───────────────────────────────────────────────────────────────

class SVMQuantile:
    """
    Nystroem RBF kernel yaklaşıklığı + LinearSVR medyan tahmini.
    q=0.1 / q=0.9 bantları eğitim setinin son %20'sinin artıklarından
    split-conformal yaklaşımıyla hesaplanır.
    """

    def __init__(
        self,
        n_components: int   = SVM_COMP,
        calib_frac:   float = CALIB_FRAC,
    ) -> None:
        self.n_components = n_components
        self.calib_frac   = calib_frac
        self._pipe:    Pipeline | None = None
        self._dq_low:  float = 0.0
        self._dq_high: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMQuantile":
        n_cal    = max(1, int(len(X) * self.calib_frac))
        X_fit, y_fit = X[:-n_cal], y[:-n_cal]
        X_cal, y_cal = X[-n_cal:], y[-n_cal:]

        self._pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("ny",  Nystroem(kernel="rbf", n_components=self.n_components, random_state=42)),
            ("svr", LinearSVR(C=1.0, max_iter=3000, random_state=42)),
        ])
        self._pipe.fit(X_fit, y_fit)

        residuals      = y_cal - self._pipe.predict(X_cal)
        self._dq_low   = float(np.quantile(residuals, 0.1))
        self._dq_high  = float(np.quantile(residuals, 0.9))
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[str, np.ndarray]:
        mid = self._pipe.predict(X)
        return {
            "q01": mid + self._dq_low,
            "q05": mid.copy(),
            "q09": mid + self._dq_high,
        }


# ── LSTM ───────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])    # son adım → (B, 3)


class LSTMQuantile:
    """2 katmanlı LSTM, pinball loss, erken durdurma, MPS/CPU."""

    def __init__(
        self,
        seq_len:    int   = SEQ_LEN,
        hidden:     int   = HIDDEN,
        epochs:     int   = EPOCHS,
        batch_size: int   = BATCH_SIZE,
        lr:         float = LR,
        patience:   int   = PATIENCE,
    ) -> None:
        self.seq_len    = seq_len
        self.hidden     = hidden
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self._net: _LSTMNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMQuantile":
        Xs, ys  = make_sequences(X, y, self.seq_len)
        Xt = torch.tensor(Xs, dtype=torch.float32).to(device)
        yt = torch.tensor(ys, dtype=torch.float32).to(device)
        loader  = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=False)

        self._net = _LSTMNet(Xs.shape[2], self.hidden).to(device)
        opt       = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        best, wait = float("inf"), 0

        for ep in range(self.epochs):
            self._net.train()
            ep_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = _pb_total(yb, self._net(xb))
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            ep_loss /= len(loader)

            if ep_loss < best - 1e-6:
                best, wait = ep_loss, 0
            else:
                wait += 1
                if wait >= self.patience:
                    log.info("LSTM erken durdurma: epoch=%d loss=%.4f", ep + 1, ep_loss)
                    break
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Xs, _ = make_sequences(X, np.zeros(len(X)), self.seq_len)
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(Xs, dtype=torch.float32).to(device)).cpu().numpy()
        return {"q01": out[:, 0], "q05": out[:, 1], "q09": out[:, 2]}


# ── Hafif TFT ──────────────────────────────────────────────────────────────────

class _GRN(nn.Module):
    """Gated Residual Network — TFT'nin çekirdek bloğu (Lim et al. 2021)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d, d)
        self.lin2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.lin1(x))
        h = self.lin2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class _LightTFTNet(nn.Module):
    """
    Hafif TFT: doğrusal gömme → LSTM encoder → multi-head attention → GRN → 3 çıktı.
    pytorch_forecasting gerektirmez.
    """

    def __init__(self, n_feat: int, d_model: int = D_MODEL, nhead: int = N_HEAD) -> None:
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model)
        self.lstm  = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.grn   = _GRN(d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)                        # (B, T, d)
        h, _ = self.lstm(h)                      # (B, T, d)
        att, _ = self.attn(h, h, h)              # (B, T, d)
        h = self.norm(h + att)                   # skip + attention
        h = self.grn(h[:, -1, :])               # son adım + GRN
        return self.head(h)                      # (B, 3)


class LightTFTQuantile:
    """Hafif TFT: LSTM encoder + interpretable multi-head attention + GRN."""

    def __init__(
        self,
        seq_len:    int   = SEQ_LEN,
        d_model:    int   = D_MODEL,
        nhead:      int   = N_HEAD,
        epochs:     int   = EPOCHS,
        batch_size: int   = BATCH_SIZE,
        lr:         float = LR,
        patience:   int   = PATIENCE,
    ) -> None:
        self.seq_len    = seq_len
        self.d_model    = d_model
        self.nhead      = nhead
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self._net: _LightTFTNet | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightTFTQuantile":
        Xs, ys  = make_sequences(X, y, self.seq_len)
        Xt = torch.tensor(Xs, dtype=torch.float32).to(device)
        yt = torch.tensor(ys, dtype=torch.float32).to(device)
        loader  = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=False)

        self._net = _LightTFTNet(Xs.shape[2], self.d_model, self.nhead).to(device)
        opt       = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        best, wait = float("inf"), 0

        for ep in range(self.epochs):
            self._net.train()
            ep_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = _pb_total(yb, self._net(xb))
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            ep_loss /= len(loader)

            if ep_loss < best - 1e-6:
                best, wait = ep_loss, 0
            else:
                wait += 1
                if wait >= self.patience:
                    log.info("TFT erken durdurma: epoch=%d loss=%.4f", ep + 1, ep_loss)
                    break
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Xs, _ = make_sequences(X, np.zeros(len(X)), self.seq_len)
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(Xs, dtype=torch.float32).to(device)).cpu().numpy()
        return {"q01": out[:, 0], "q05": out[:, 1], "q09": out[:, 2]}


# ── Metrik ────────────────────────────────────────────────────────────────────

def evaluate_quantiles(
    y_true: np.ndarray,
    preds:  dict[str, np.ndarray],
) -> dict[str, float]:
    """pinball_q01/q05/q09, crps, mae, rmse, coverage."""
    y = np.asarray(y_true, dtype=np.float64)
    q01, q05, q09 = preds["q01"], preds["q05"], preds["q09"]

    def pb(pred: np.ndarray, q: float) -> float:
        r = y - pred
        return float(np.mean(np.where(r >= 0, q * r, (q - 1) * r)))

    pb01, pb05, pb09 = pb(q01, 0.1), pb(q05, 0.5), pb(q09, 0.9)
    return {
        "pinball_q01": pb01,
        "pinball_q05": pb05,
        "pinball_q09": pb09,
        "crps":        (pb01 + pb05 + pb09) / 3,
        "mae":         float(np.mean(np.abs(y - q05))),
        "rmse":        float(np.sqrt(np.mean((y - q05) ** 2))),
        "coverage":    float(np.mean((y >= q01) & (y <= q09))),
    }


# ── Unified API ───────────────────────────────────────────────────────────────

BASELINE_REGISTRY: dict[str, type] = {
    "knn":  KNNQuantile,
    "svm":  SVMQuantile,
    "lstm": LSTMQuantile,
    "tft":  LightTFTQuantile,
}


def train_baseline(
    name:           str,
    X_train:        np.ndarray | pd.DataFrame,
    y_train:        np.ndarray | pd.Series,
    checkpoint_dir: str = "models/checkpoints",
    **kwargs:       Any,
) -> Any:
    """
    Baseline modeli eğit ve diske kaydet.

    sklearn modelleri (knn, svm) → joblib.
    PyTorch modelleri (lstm, tft) → model nesnesi joblib ile (state_dict dahil).

    Returns:
        Fitted model nesnesi.
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Bilinmeyen baseline: '{name}'. Seçenekler: {list(BASELINE_REGISTRY)}")

    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    t0    = time.time()
    model = BASELINE_REGISTRY[name](**kwargs)
    model.fit(X, y)
    elapsed = time.time() - t0
    log.info("%s eğitildi | süre=%.1fs", name, elapsed)

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"baseline_{name}.joblib"
    joblib.dump(model, path)
    log.info("Kaydedildi: %s", path)
    return model


def predict_baseline(
    name:  str,
    model: Any,
    X:     np.ndarray | pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Baseline modelden quantile tahminleri al."""
    return model.predict_quantiles(np.asarray(X, dtype=np.float32))


def evaluate_baseline(
    name:   str,
    model:  Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Baseline modeli değerlendir.
    Dizi modelleri (lstm, tft) için y_test otomatik olarak seq_len kadar öne hizalanır.
    """
    X = np.asarray(X_test, dtype=np.float32)
    y = np.asarray(y_test, dtype=np.float32)
    preds = model.predict_quantiles(X)

    seq_len = getattr(model, "seq_len", 0)
    if seq_len:
        y = y[seq_len:]

    return evaluate_quantiles(y, preds)


def train_all_baselines(
    X_train:        np.ndarray | pd.DataFrame,
    y_train:        np.ndarray | pd.Series,
    X_test:         np.ndarray | pd.DataFrame,
    y_test:         np.ndarray | pd.Series,
    checkpoint_dir: str = "models/checkpoints",
) -> dict[str, dict]:
    """
    4 baseline modeli sırayla eğit ve değerlendir.

    Returns:
        {model_name: {"model": ..., "metrics": {...}, "train_time": float}}
    """
    results: dict[str, dict] = {}
    for name in BASELINE_REGISTRY:
        t0      = time.time()
        model   = train_baseline(name, X_train, y_train, checkpoint_dir)
        elapsed = time.time() - t0
        metrics = evaluate_baseline(name, model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics, "train_time": elapsed}
        log.info(
            "%s | crps=%.4f mae=%.4f coverage=%.3f | süre=%.1fs",
            name, metrics["crps"], metrics["mae"], metrics["coverage"], elapsed,
        )
    return results
