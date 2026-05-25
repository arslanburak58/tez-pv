"""
STAGE-5: 9 taban öğrenici — LightGBM × 3 + CatBoost × 3 + XGBoost × 3 (quantile).

API:
    train_base_learner(algo, q, X_train, y_train) → fitted model
    make_oof_predictions(algo, q, X_train, y_train) → np.ndarray (n_train,)
    build_x_meta(X_train, y_train) → X_meta (n_oof × 9), oof_scores dict

Kurallar:
    - OOF için TimeSeriesSplit kullanılır (random K-Fold yasak)
    - Serileştirme joblib (pickle yasak)
    - XGBoost custom pinball objective — grad/hess elle yazılmış
"""

import logging
import random
from typing import Any, Protocol

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

random.seed(42)
np.random.seed(42)

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
ALGOS: tuple[str, ...] = ("lgbm", "catboost", "xgboost")

OOF_N_SPLITS: int = 5
OOF_GAP: int = 24  # timestep cinsinden (DKASC 5-dk: 24 × 5 dk = 2 saat)

# Sütun adı kalıbı: {algo}_q{int(q*10):02d}  örn. lgbm_q01, catboost_q05
def _col_name(algo: str, q: float) -> str:
    return f"{algo}_q{int(round(q * 10)):02d}"

META_COLS: list[str] = [_col_name(a, q) for a in ALGOS for q in QUANTILES]

# ── Varsayılan hiperparametreler (STAGE-7'de Optuna optimize edecek) ──────────

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "lgbm": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "catboost": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "subsample": 0.8,
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "nthread": -1,
    },
}

# ── Pinball loss (değerlendirme — eğitim için değil) ──────────────────────────

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Ortalama pinball (quantile) loss. Düşük = iyi."""
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, q * residual, (q - 1) * residual)))

# ── XGBoost custom objective ───────────────────────────────────────────────────

def _make_pinball_objective(q: float):
    """Quantile q için XGBoost gradient/hessian fonksiyonu döndürür."""
    def objective(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        residual = y_true - y_pred
        grad = np.where(residual >= 0, -q, 1.0 - q)
        hess = np.ones_like(y_pred)
        return grad, hess
    return objective

# ── Protokol: tüm fitted model'ların .predict() arayüzü ──────────────────────

class PredictorProtocol(Protocol):
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...

# XGBoost fonksiyonel API'yi sarmak için basit wrapper
class _XGBWrapper:
    def __init__(self, booster: xgb.Booster) -> None:
        self._booster = booster

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self._booster.predict(xgb.DMatrix(X))

# ── train_base_learner ────────────────────────────────────────────────────────

def train_base_learner(
    algo: str,
    q: float,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """
    Tek bir (algo, q) çiftini eğit ve fitted model döndür.

    Args:
        algo:    "lgbm" | "catboost" | "xgboost"
        q:       0.1 | 0.5 | 0.9
        X_train: öznitelik matrisi
        y_train: hedef değişken
        params:  None → DEFAULT_PARAMS[algo] kullanılır

    Returns:
        Fitted model (.predict() arayüzüyle)
    """
    if algo not in ALGOS:
        raise ValueError(f"algo '{algo}' geçersiz. Seçenekler: {ALGOS}")
    if q not in QUANTILES:
        raise ValueError(f"q={q} geçersiz. Seçenekler: {QUANTILES}")

    p = {**DEFAULT_PARAMS[algo], **(params or {})}

    if algo == "lgbm":
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **p)
        model.fit(X_train, y_train)

    elif algo == "catboost":
        model = CatBoostRegressor(loss_function=f"Quantile:alpha={q}", **p)
        model.fit(X_train, y_train)

    else:  # xgboost
        xgb_params = {
            "objective": "reg:squarederror",  # custom objective override edilecek
            **{k: v for k, v in p.items() if k not in ("n_estimators", "seed", "nthread")},
            "seed": p.get("seed", 42),
            "nthread": p.get("nthread", -1),
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        booster = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=p.get("n_estimators", 500),
            obj=_make_pinball_objective(q),
            verbose_eval=False,
        )
        model = _XGBWrapper(booster)

    return model

# ── OOF tahminleri ────────────────────────────────────────────────────────────

def make_oof_predictions(
    algo: str,
    q: float,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    n_splits: int = OOF_N_SPLITS,
    gap: int = OOF_GAP,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    TimeSeriesSplit ile out-of-fold tahminleri üret.

    İlk fold'un eğitim kısmına karşılık gelen satırlar NaN kalır
    (sonradan build_x_meta içinde uyumlu satırlar seçilir).

    Returns:
        np.ndarray (n_train,) — OOF tahminleri, başta NaN olabilir
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    n = len(X)

    oof = np.full(n, np.nan)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
        model = train_base_learner(algo, q, X[tr_idx], y[tr_idx], params=params)
        oof[vl_idx] = model.predict(X[vl_idx])
        score = pinball_loss(y[vl_idx], oof[vl_idx], q)
        log.info("OOF | %s q=%.1f | fold %d | pinball=%.4f", algo, q, fold, score)

    return oof

# ── X_meta matrisi ────────────────────────────────────────────────────────────

def build_x_meta(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    n_splits: int = OOF_N_SPLITS,
    gap: int = OOF_GAP,
    params_override: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Tüm 9 modelin OOF tahminlerini X_meta matrisinde topla.

    Returns:
        X_meta:     pd.DataFrame (n_oof × 9) — NaN içeren başlangıç satırları çıkarılmış
        oof_scores: dict — her (algo, q) için ortalama pinball skoru
    """
    y_arr = np.asarray(y_train)
    oof_dict: dict[str, np.ndarray] = {}
    oof_scores: dict[str, float] = {}

    for algo in ALGOS:
        for q in QUANTILES:
            col = _col_name(algo, q)
            log.info("OOF üretiliyor: %s", col)
            p = (params_override or {}).get(algo)
            oof = make_oof_predictions(algo, q, X_train, y_train, n_splits, gap, params=p)
            oof_dict[col] = oof

            # NaN olmayan satırlar üzerinde skor hesapla
            valid = ~np.isnan(oof)
            if valid.any():
                oof_scores[col] = pinball_loss(y_arr[valid], oof[valid], q)

    X_meta = pd.DataFrame(oof_dict, columns=META_COLS)

    # NaN içeren satırları at (TimeSeriesSplit'in ilk eğitim bloğu)
    valid_rows = X_meta.notna().all(axis=1)
    X_meta = X_meta[valid_rows].copy()
    y_meta = y_arr[valid_rows.values]

    log.info(
        "X_meta hazır | şekil: %s | NaN atılan satır: %d",
        X_meta.shape,
        (~valid_rows).sum(),
    )

    return X_meta, oof_scores

# ── Tüm 9 modeli eğit ve kaydet ───────────────────────────────────────────────

def train_all_base_learners(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    checkpoint_dir: str = "models/checkpoints",
    params_override: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    9 modeli eğit, joblib ile kaydet, model sözlüğü döndür.

    Returns:
        dict — anahtar: "{algo}_q{quantile}", değer: fitted model
    """
    models: dict[str, Any] = {}
    for algo in ALGOS:
        for q in QUANTILES:
            col = _col_name(algo, q)
            log.info("Eğitiliyor: %s", col)
            p = (params_override or {}).get(algo)
            model = train_base_learner(algo, q, X_train, y_train, params=p)
            models[col] = model
            path = f"{checkpoint_dir}/{col}.joblib"
            joblib.dump(model, path)
            log.info("Kaydedildi: %s", path)
    return models
