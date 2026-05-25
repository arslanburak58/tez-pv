"""Unit tests for models/meta_learner.py — STAGE-6"""

import random

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

random.seed(42)
np.random.seed(42)

from models.base_learners import META_COLS, build_x_meta
from models.meta_learner import (
    FLAG_COLS,
    META_IN_COLS,
    _meta_key,
    compare_baseline,
    coverage_score,
    enrich_x_meta,
    predict_intervals,
    train_all_meta_learners,
    train_meta_learner,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _synthetic(n: int = 400, seed: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """X_train (with flags), y_train, flags_df"""
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    X = pd.DataFrame(rng.uniform(0, 1, (n, 5)), columns=[f"f{i}" for i in range(5)], index=idx)
    y = pd.Series(rng.uniform(0, 100, n), name="target", index=idx)
    flags = pd.DataFrame(
        {col: rng.integers(0, 2, n) for col in FLAG_COLS},
        index=idx,
    )
    return X, y, flags


def _x_meta_and_flags(
    n: int = 400, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Returns (X_meta_9, y_meta, flags_meta) ready for enrich_x_meta."""
    X, y, flags = _synthetic(n=n, seed=seed)
    X_meta, _ = build_x_meta(X, y, n_splits=3, gap=5)
    y_meta = y.iloc[X_meta.index]
    flags_meta = flags.loc[X_meta.index]
    return X_meta, y_meta, flags_meta


# ── _meta_key ──────────────────────────────────────────────────────────────────

def test_meta_key_format() -> None:
    assert _meta_key(0.1) == "meta_q01"
    assert _meta_key(0.5) == "meta_q05"
    assert _meta_key(0.9) == "meta_q09"


# ── META_IN_COLS ───────────────────────────────────────────────────────────────

def test_meta_in_cols_length() -> None:
    assert len(META_IN_COLS) == 13  # 9 OOF + 4 flags


def test_meta_in_cols_order() -> None:
    assert META_IN_COLS[:9] == META_COLS
    assert META_IN_COLS[9:] == FLAG_COLS


# ── enrich_x_meta ─────────────────────────────────────────────────────────────

def test_enrich_shape() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    assert X13.shape[1] == 13
    assert X13.shape[0] == len(X_meta)


def test_enrich_columns() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    assert list(X13.columns) == META_IN_COLS


def test_enrich_no_nans() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    assert not X13.isna().any().any()


def test_enrich_flags_binary() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    for col in FLAG_COLS:
        assert X13[col].isin([0, 1]).all(), f"{col} non-binary"


def test_enrich_missing_oof_col_raises() -> None:
    X_meta, _, flags_meta = _x_meta_and_flags()
    bad = X_meta.drop(columns=["lgbm_q01"])
    with pytest.raises(ValueError, match="eksik OOF"):
        enrich_x_meta(bad, flags_meta)


def test_enrich_missing_flag_col_raises() -> None:
    X_meta, _, flags_meta = _x_meta_and_flags()
    bad_flags = flags_meta.drop(columns=["is_G_missing"])
    with pytest.raises(ValueError, match="eksik sütunlar"):
        enrich_x_meta(X_meta, bad_flags)


def test_enrich_index_alignment() -> None:
    """flags sütunu X_meta'nın dışındaki satırları içerse bile hizalanmalı."""
    X_meta, _, flags_meta = _x_meta_and_flags(n=400)
    # flags'e X_meta.index dışı satırlar ekle
    extra = pd.DataFrame(
        {col: [0] * 10 for col in FLAG_COLS},
        index=range(10000, 10010),
    )
    flags_big = pd.concat([flags_meta, extra])
    X13 = enrich_x_meta(X_meta, flags_big)
    assert len(X13) == len(X_meta)


# ── train_meta_learner ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_train_meta_learner_returns_ridge(q: float) -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    model = train_meta_learner(q, X13, y_meta)
    assert isinstance(model, Ridge)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_train_meta_learner_predict_shape(q: float) -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    model = train_meta_learner(q, X13, y_meta)
    preds = model.predict(X13)
    assert preds.shape == (len(X13),)


def test_train_meta_learner_invalid_q_raises() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    with pytest.raises(ValueError, match="geçersiz"):
        train_meta_learner(0.3, X13, y_meta)


# ── train_all_meta_learners ────────────────────────────────────────────────────

def test_train_all_returns_3_models() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    assert len(models) == 3


def test_train_all_keys() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    assert set(models.keys()) == {"meta_q01", "meta_q05", "meta_q09"}


def test_train_all_saves_joblib(tmp_path) -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    train_all_meta_learners(X13, y_meta, checkpoint_dir=str(tmp_path))
    saved = list(tmp_path.glob("*.joblib"))
    assert len(saved) == 3


# ── predict_intervals ──────────────────────────────────────────────────────────

def test_predict_intervals_keys() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    preds = predict_intervals(models, X13)
    assert set(preds.keys()) == {"meta_q01", "meta_q05", "meta_q09"}


def test_predict_intervals_shape() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    preds = predict_intervals(models, X13)
    for key, arr in preds.items():
        assert arr.shape == (len(X13),), f"{key}: wrong shape"


def test_predict_intervals_finite() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    preds = predict_intervals(models, X13)
    for key, arr in preds.items():
        assert np.isfinite(arr).all(), f"{key}: non-finite predictions"


def test_predict_input_is_dataframe() -> None:
    """predict_intervals DataFrame almalı — tip kaybı olmamalı."""
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    # DataFrame geçirilince hata vermemeli
    preds = predict_intervals(models, X13)
    assert isinstance(X13, pd.DataFrame)  # X13 hâlâ DataFrame
    assert all(isinstance(v, np.ndarray) for v in preds.values())


# ── coverage_score ─────────────────────────────────────────────────────────────

def test_coverage_perfect() -> None:
    y = np.array([5.0, 10.0, 15.0])
    assert coverage_score(y, np.zeros(3), np.full(3, 100.0)) == pytest.approx(1.0)


def test_coverage_none() -> None:
    y = np.array([5.0, 10.0, 15.0])
    assert coverage_score(y, np.full(3, 50.0), np.full(3, 100.0)) == pytest.approx(0.0)


def test_coverage_range() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, 500)
    lo = rng.uniform(0, 50, 500)
    hi = rng.uniform(50, 100, 500)
    cov = coverage_score(y, lo, hi)
    assert 0.0 <= cov <= 1.0


def test_coverage_accepts_series() -> None:
    y = pd.Series([5.0, 10.0, 15.0])
    cov = coverage_score(y, np.zeros(3), np.full(3, 100.0))
    assert cov == pytest.approx(1.0)


# ── compare_baseline ───────────────────────────────────────────────────────────

def test_compare_baseline_keys() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    result = compare_baseline(models, X13, y_meta)
    assert set(result.keys()) == {"stacked", "baseline", "improvement"}


def test_compare_baseline_score_keys() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    result = compare_baseline(models, X13, y_meta)
    for section in ("stacked", "baseline", "improvement"):
        assert set(result[section].keys()) == {"meta_q01", "meta_q05", "meta_q09"}


def test_compare_baseline_scores_nonnegative() -> None:
    X_meta, y_meta, flags_meta = _x_meta_and_flags()
    X13 = enrich_x_meta(X_meta, flags_meta)
    models = train_all_meta_learners(X13, y_meta)
    result = compare_baseline(models, X13, y_meta)
    for key, s in result["stacked"].items():
        assert s >= 0.0, f"{key}: negative pinball"
    for key, s in result["baseline"].items():
        assert s >= 0.0, f"{key}: negative pinball"
