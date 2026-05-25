"""Unit tests for models/base_learners.py — STAGE-5"""

import random

import numpy as np
import pandas as pd
import pytest

random.seed(42)
np.random.seed(42)

from models.base_learners import (
    META_COLS,
    QUANTILES,
    _col_name,
    build_x_meta,
    make_oof_predictions,
    pinball_loss,
    train_base_learner,
    train_all_base_learners,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _synthetic(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.uniform(0, 1, (n, 5)),
        columns=[f"f{i}" for i in range(5)],
    )
    y = pd.Series(rng.uniform(0, 100, n), name="target")
    return X, y


# ── pinball_loss ───────────────────────────────────────────────────────────────

def test_pinball_symmetric_at_median() -> None:
    y = np.array([10.0, 20.0])
    y_pred_low = np.array([5.0, 15.0])   # underprediction
    y_pred_high = np.array([15.0, 25.0])  # overprediction
    # At q=0.5 pinball is symmetric
    assert abs(pinball_loss(y, y_pred_low, 0.5) - pinball_loss(y, y_pred_high, 0.5)) < 1e-10


def test_pinball_perfect_prediction_zero() -> None:
    y = np.array([10.0, 20.0, 30.0])
    assert pinball_loss(y, y, 0.1) == pytest.approx(0.0)
    assert pinball_loss(y, y, 0.5) == pytest.approx(0.0)
    assert pinball_loss(y, y, 0.9) == pytest.approx(0.0)


def test_pinball_nonnegative() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, 100)
    y_pred = rng.uniform(0, 100, 100)
    for q in QUANTILES:
        assert pinball_loss(y, y_pred, q) >= 0.0


def test_pinball_asymmetric() -> None:
    y = np.array([10.0])
    # underprediction at q=0.9: penalized 0.9 × residual
    assert pinball_loss(y, np.array([5.0]), 0.9) == pytest.approx(0.9 * 5.0)
    # overprediction at q=0.9: penalized 0.1 × |residual|
    assert pinball_loss(y, np.array([15.0]), 0.9) == pytest.approx(0.1 * 5.0)


# ── _col_name ──────────────────────────────────────────────────────────────────

def test_col_name_format() -> None:
    assert _col_name("lgbm", 0.1) == "lgbm_q01"
    assert _col_name("catboost", 0.5) == "catboost_q05"
    assert _col_name("xgboost", 0.9) == "xgboost_q09"


def test_meta_cols_length() -> None:
    assert len(META_COLS) == 9  # 3 algos × 3 quantiles


# ── train_base_learner ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("algo", ["lgbm", "catboost", "xgboost"])
@pytest.mark.parametrize("q", QUANTILES)
def test_train_returns_predictor(algo: str, q: float) -> None:
    X, y = _synthetic(n=200)
    model = train_base_learner(algo, q, X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)


@pytest.mark.parametrize("algo", ["lgbm", "catboost", "xgboost"])
def test_train_output_finite(algo: str) -> None:
    X, y = _synthetic(n=200)
    model = train_base_learner(algo, 0.5, X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all(), f"{algo}: non-finite predictions"


def test_train_invalid_algo_raises() -> None:
    X, y = _synthetic(n=50)
    with pytest.raises(ValueError, match="geçersiz"):
        train_base_learner("random_forest", 0.5, X, y)


def test_train_invalid_q_raises() -> None:
    X, y = _synthetic(n=50)
    with pytest.raises(ValueError, match="geçersiz"):
        train_base_learner("lgbm", 0.3, X, y)


def test_train_accepts_numpy_arrays() -> None:
    X, y = _synthetic(n=200)
    model = train_base_learner("lgbm", 0.5, X.values, y.values)
    preds = model.predict(X.values)
    assert preds.shape == (200,)


# ── make_oof_predictions ───────────────────────────────────────────────────────

@pytest.mark.parametrize("algo", ["lgbm", "catboost", "xgboost"])
def test_oof_shape(algo: str) -> None:
    X, y = _synthetic(n=300)
    oof = make_oof_predictions(algo, 0.5, X, y, n_splits=3, gap=5)
    assert oof.shape == (300,)


@pytest.mark.parametrize("algo", ["lgbm", "catboost", "xgboost"])
def test_oof_has_leading_nans(algo: str) -> None:
    X, y = _synthetic(n=300)
    oof = make_oof_predictions(algo, 0.5, X, y, n_splits=3, gap=5)
    # First fold's train block → NaN; last fold → no NaN
    assert np.isnan(oof).any(), f"{algo}: expected some NaN from first train block"
    assert np.isfinite(oof[-50:]).all(), f"{algo}: last rows should be finite"


def test_oof_non_nan_finite() -> None:
    X, y = _synthetic(n=300)
    oof = make_oof_predictions("lgbm", 0.5, X, y, n_splits=3, gap=5)
    valid = oof[~np.isnan(oof)]
    assert np.isfinite(valid).all()


# ── build_x_meta ───────────────────────────────────────────────────────────────

def test_build_x_meta_shape() -> None:
    X, y = _synthetic(n=400)
    X_meta, scores = build_x_meta(X, y, n_splits=3, gap=5)
    assert X_meta.shape[1] == 9
    assert len(X_meta) < len(X)   # NaN rows dropped
    assert len(X_meta) > 0


def test_build_x_meta_no_nans() -> None:
    X, y = _synthetic(n=400)
    X_meta, _ = build_x_meta(X, y, n_splits=3, gap=5)
    assert not X_meta.isna().any().any()


def test_build_x_meta_columns() -> None:
    X, y = _synthetic(n=400)
    X_meta, _ = build_x_meta(X, y, n_splits=3, gap=5)
    assert list(X_meta.columns) == META_COLS


def test_build_x_meta_scores_keys() -> None:
    X, y = _synthetic(n=400)
    _, scores = build_x_meta(X, y, n_splits=3, gap=5)
    assert set(scores.keys()) == set(META_COLS)


def test_build_x_meta_scores_nonnegative() -> None:
    X, y = _synthetic(n=400)
    _, scores = build_x_meta(X, y, n_splits=3, gap=5)
    for col, s in scores.items():
        assert s >= 0.0, f"{col}: negative pinball score {s}"


# ── train_all_base_learners ────────────────────────────────────────────────────

def test_train_all_returns_9_models(tmp_path: "pytest.TempPathFactory") -> None:
    X, y = _synthetic(n=200)
    models = train_all_base_learners(X, y, checkpoint_dir=str(tmp_path))
    assert len(models) == 9
    assert set(models.keys()) == set(META_COLS)


def test_train_all_saves_joblib(tmp_path: "pytest.TempPathFactory") -> None:
    X, y = _synthetic(n=200)
    train_all_base_learners(X, y, checkpoint_dir=str(tmp_path))
    saved = list(tmp_path.glob("*.joblib"))
    assert len(saved) == 9
