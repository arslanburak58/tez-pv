"""
CQR — Conformalized Quantile Regression.

Referans: Romano, Patterson & Candès (2019). NeurIPS.

API (simetrik):
    compute_cqr_offset(y_cal, q01_cal, q09_cal, alpha)           → float
    apply_cqr_correction(preds, offset)                           → dict

API (asimetrik):
    compute_cqr_asymmetric(y_cal, q01_cal, q05_cal, q09_cal, alpha) → (float, float)
    apply_cqr_asymmetric(preds, off_low, off_up)                  → dict

API (locally scaled):
    compute_cqr_locally_scaled(y_cal, q01_cal, q05_cal, q09_cal, alpha) → float
    apply_locally_scaled(preds, k_scale)                          → dict
"""

import numpy as np


def compute_cqr_offset(
    y_cal: np.ndarray,
    q01_cal: np.ndarray,
    q09_cal: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """
    Calibration set üzerinde CQR conformity score'larından offset hesapla.

    Conformity score: max(q01 - y, y - q09)
      - y band içindeyse negatif (iyi)
      - y band dışındaysa pozitif mesafe kadar

    Test setinde [q01 - offset, q09 + offset] uygulanınca
    marginal coverage garantisi: 1 - alpha.

    Parameters
    ----------
    y_cal   : gerçek değerler (calibration)
    q01_cal : alt quantile tahminleri (calibration)
    q09_cal : üst quantile tahminleri (calibration)
    alpha   : hata oranı; 0.20 → %80 coverage hedefi

    Returns
    -------
    float — scalar offset (negatif olabilir → bant zaten yeterli geniş)
    """
    y     = np.asarray(y_cal,   dtype=np.float64)
    q01   = np.asarray(q01_cal, dtype=np.float64)
    q09   = np.asarray(q09_cal, dtype=np.float64)

    scores = np.maximum(q01 - y, y - q09)
    n      = len(scores)
    k      = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    offset = float(np.sort(scores)[k - 1])
    return offset


def apply_cqr_correction(
    preds: dict[str, np.ndarray],
    offset: float,
) -> dict[str, np.ndarray]:
    """
    Test tahminlerine simetrik CQR offset uygula.
    q05 dokunulmaz; q01 aşağı, q09 yukarı kayar.
    """
    return {
        "q01": preds["q01"] - offset,
        "q05": preds["q05"],
        "q09": preds["q09"] + offset,
    }


# ── Asimetrik CQR ─────────────────────────────────────────────────────────────

def compute_cqr_asymmetric(
    y_cal: np.ndarray,
    q01_cal: np.ndarray,
    q05_cal: np.ndarray,
    q09_cal: np.ndarray,
    alpha: float = 0.20,
) -> tuple[float, float]:
    """
    Asimetrik CQR: alt ve üst tarafa ayrı offset.

    Alt taraf: y < q01 (taşma); üst taraf: y > q09 (taşma).
    Her tarafa alpha/2 risk payı ayrılır → toplam 1-alpha coverage.

    Returns
    -------
    (offset_lower, offset_upper) — her iki değer ≥ 0
    """
    y   = np.asarray(y_cal,   dtype=np.float64)
    q01 = np.asarray(q01_cal, dtype=np.float64)
    q09 = np.asarray(q09_cal, dtype=np.float64)

    lower_scores = np.maximum(q01 - y, 0)   # alt taşma mesafesi
    upper_scores = np.maximum(y - q09, 0)   # üst taşma mesafesi
    n = len(y)
    k = min(int(np.ceil((n + 1) * (1 - alpha / 2))), n)
    off_low = float(np.sort(lower_scores)[k - 1])
    off_up  = float(np.sort(upper_scores)[k - 1])
    return off_low, off_up


def apply_cqr_asymmetric(
    preds: dict[str, np.ndarray],
    off_low: float,
    off_up: float,
) -> dict[str, np.ndarray]:
    """Asimetrik CQR: q01 aşağı off_low, q09 yukarı off_up kaydır."""
    return {
        "q01": preds["q01"] - off_low,
        "q05": preds["q05"],
        "q09": preds["q09"] + off_up,
    }


# ── Locally Scaled CQR ────────────────────────────────────────────────────────

def compute_cqr_locally_scaled(
    y_cal: np.ndarray,
    q01_cal: np.ndarray,
    q05_cal: np.ndarray,
    q09_cal: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """
    Locally scaled CQR: normalize edilmiş skor ile bant ölçeklenir.

    Dar bantta küçük mutlak offset, geniş bantta büyük mutlak offset.

    Returns
    -------
    k_scale — test'te:
        new_q01 = q05 - k_scale*(q05 - q01)
        new_q09 = q05 + k_scale*(q09 - q05)
    """
    y   = np.asarray(y_cal,   dtype=np.float64)
    q01 = np.asarray(q01_cal, dtype=np.float64)
    q05 = np.asarray(q05_cal, dtype=np.float64)
    q09 = np.asarray(q09_cal, dtype=np.float64)

    half_lower = np.maximum(q05 - q01, 1e-6)
    half_upper = np.maximum(q09 - q05, 1e-6)
    lower_norm = np.maximum((q01 - y) / half_lower, 0)
    upper_norm = np.maximum((y - q09) / half_upper, 0)
    scores = np.maximum(lower_norm, upper_norm)

    n     = len(scores)
    k_idx = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    k_scale = 1.0 + float(np.sort(scores)[k_idx - 1])
    return k_scale


def apply_locally_scaled(
    preds: dict[str, np.ndarray],
    k_scale: float,
) -> dict[str, np.ndarray]:
    """new_q01 = q05 - k*(q05-q01), new_q09 = q05 + k*(q09-q05)."""
    q01, q05, q09 = preds["q01"], preds["q05"], preds["q09"]
    return {
        "q01": q05 - k_scale * (q05 - q01),
        "q05": q05,
        "q09": q05 + k_scale * (q09 - q05),
    }
