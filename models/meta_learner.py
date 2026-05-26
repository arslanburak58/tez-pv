"""
STAGE-6: Ridge × 3 meta-öğrenici — 9 OOF + 3 missingness flag = 12 özellik.

Hipotez: is_X_missing flag'leri meta-katmana eklenince CRPS istatistiksel anlamlı
         düşer (Diebold-Mariano testi, STAGE-10).

API:
    enrich_x_meta(X_meta, flags)              → pd.DataFrame (n_oof × 12)
    train_meta_learner(q, X_meta_12, y)       → fitted Ridge
    train_all_meta_learners(X_meta_12, y)     → dict[str, Ridge]
    predict_intervals(models, X_meta_12)      → dict[str, np.ndarray]
    coverage_score(y_true, y_lower, y_upper)  → float  (hedef ~0.80)
    compare_baseline(models, X_meta_12, y)    → dict

Kurallar:
    - LightGBM predict → DataFrame (sütun adları korunur)
    - Serileştirme joblib (pickle yasak)
"""

import logging
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.base_learners import META_COLS, QUANTILES, _col_name, pinball_loss

random.seed(42)
np.random.seed(42)

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

FLAG_COLS: list[str] = [
    "is_G_missing",
    "is_Tamb_missing",
    "is_RH_missing",
]

META_IN_COLS: list[str] = META_COLS + FLAG_COLS  # 12 özellik (9 OOF + 3 flag)

DEFAULT_ALPHA: float = 1.0


def _meta_key(q: float) -> str:
    return f"meta_q{int(round(q * 10)):02d}"  # meta_q01 | meta_q05 | meta_q09


# ── enrich_x_meta ─────────────────────────────────────────────────────────────

def enrich_x_meta(
    X_meta: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    """
    9 OOF sütununa 4 missingness flag ekle → 13 özellikli X_meta.

    Args:
        X_meta: (n_oof × 9) — build_x_meta() çıktısı; orijinal index korunmuş
        flags:  (n × 4)    — missingness flags; index X_meta ile hizalanabilir

    Returns:
        pd.DataFrame (n_oof × 13) — sütunlar META_IN_COLS sırasında
    """
    missing_oof = set(META_COLS) - set(X_meta.columns)
    if missing_oof:
        raise ValueError(f"X_meta'da eksik OOF sütunları: {missing_oof}")
    missing_flags = set(FLAG_COLS) - set(flags.columns)
    if missing_flags:
        raise ValueError(f"flags'da eksik sütunlar: {missing_flags}")

    flags_aligned = flags.reindex(X_meta.index)[FLAG_COLS].fillna(0).astype(int)
    result = pd.concat([X_meta[META_COLS], flags_aligned], axis=1)
    result.columns = META_IN_COLS
    return result


# ── train_meta_learner ─────────────────────────────────────────────────────────

def train_meta_learner(
    q: float,
    X_meta_13: pd.DataFrame,
    y: pd.Series | np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Ridge:
    """
    Tek bir quantile için Ridge meta-öğrenici eğit.

    Taban modeller quantile'ı zaten kodladığından meta-katmanda
    doğrusal harmanlama (squared-loss Ridge) yeterlidir.
    Alpha STAGE-7'de Optuna ile aranacak.
    """
    if q not in QUANTILES:
        raise ValueError(f"q={q} geçersiz. Seçenekler: {QUANTILES}")

    model = Ridge(alpha=alpha)
    model.fit(X_meta_13, np.asarray(y))
    log.info("Meta-öğrenici eğitildi | q=%.1f | alpha=%.4f", q, alpha)
    return model


# ── train_all_meta_learners ────────────────────────────────────────────────────

def train_all_meta_learners(
    X_meta_13: pd.DataFrame,
    y: pd.Series | np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    checkpoint_dir: str | None = None,
) -> dict[str, Ridge]:
    """
    3 Ridge meta-öğrenicisini eğit (q=0.1 / 0.5 / 0.9).

    Returns:
        dict — anahtar: "meta_q01" | "meta_q05" | "meta_q09"
    """
    models: dict[str, Ridge] = {}
    for q in QUANTILES:
        key = _meta_key(q)
        model = train_meta_learner(q, X_meta_13, y, alpha=alpha)
        models[key] = model
        if checkpoint_dir:
            path = f"{checkpoint_dir}/{key}.joblib"
            joblib.dump(model, path)
            log.info("Kaydedildi: %s", path)
    return models


# ── predict_intervals ──────────────────────────────────────────────────────────

def predict_intervals(
    models: dict[str, Ridge],
    X_meta_13: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    3 quantile tahmini üret.

    X_meta_13 DataFrame olarak geçirilir; Ridge sütun adlarına
    ihtiyaç duymaz ama tutarlılık için tip korunur.

    Returns:
        dict — anahtar: "meta_q01" | "meta_q05" | "meta_q09"
    """
    return {key: model.predict(X_meta_13) for key, model in models.items()}


# ── coverage_score ─────────────────────────────────────────────────────────────

def coverage_score(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    %10–%90 bant kapsama oranı.

    Nominal kapsama = 0.80 (q=0.9 bandı - q=0.1 bandı).
    İyi kalibre model: 0.75 ≤ coverage ≤ 0.85.

    Returns:
        float — [0, 1]
    """
    y = np.asarray(y_true)
    return float(np.mean((y >= y_lower) & (y <= y_upper)))


# ── compare_baseline ───────────────────────────────────────────────────────────

def compare_baseline(
    meta_models: dict[str, Ridge],
    X_meta_13: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """
    Stacked meta-model vs tek LightGBM-quantile baseline karşılaştır.

    Baseline: X_meta_13 içindeki lgbm_q01 / lgbm_q05 / lgbm_q09 OOF tahminleri.
    Tamamlandı ölçütü: stacked modelin her quantile'da ≥ %5 pinball iyileşmesi.

    Returns:
        dict:
            stacked:     {meta_key: pinball_score}
            baseline:    {meta_key: pinball_score}
            improvement: {meta_key: % iyileşme (pozitif = iyi)}
    """
    y = np.asarray(y_true)

    q_map: dict[str, tuple[float, str]] = {
        _meta_key(0.1): (0.1, "lgbm_q01"),
        _meta_key(0.5): (0.5, "lgbm_q05"),
        _meta_key(0.9): (0.9, "lgbm_q09"),
    }

    stacked_scores: dict[str, float] = {}
    baseline_scores: dict[str, float] = {}

    for meta_key, (q, lgbm_col) in q_map.items():
        stacked_pred = meta_models[meta_key].predict(X_meta_13)
        stacked_scores[meta_key] = pinball_loss(y, stacked_pred, q)

        baseline_pred = X_meta_13[lgbm_col].to_numpy()
        baseline_scores[meta_key] = pinball_loss(y, baseline_pred, q)

    improvement: dict[str, float] = {}
    for key in stacked_scores:
        base = baseline_scores[key]
        imp = (base - stacked_scores[key]) / base * 100 if base > 0 else 0.0
        improvement[key] = imp
        log.info(
            "%s | stacked=%.4f | baseline=%.4f | iyileşme=%.1f%%",
            key, stacked_scores[key], base, imp,
        )

    return {
        "stacked": stacked_scores,
        "baseline": baseline_scores,
        "improvement": improvement,
    }
