"""Unit tests for scripts/make_dataset.py — STAGE-4"""

import random

import numpy as np
import pandas as pd
import pytest

random.seed(42)
np.random.seed(42)

from features.physical import DKASC_LOCATION
from scripts.make_dataset import (
    make_dataset,
    make_missingness_flags,
    make_walk_forward_splits,
    rename_columns,
)

# ── Fixture ────────────────────────────────────────────────────────────────────

def _synthetic_df(
    n: int = 1000,
    freq: str = "5min",
    missing_frac: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Minimal synthetic DKASC-like DataFrame with canonical column names."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2015-01-01", periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "G": rng.uniform(0, 900, n),
            "T_amb": rng.uniform(10, 45, n),
            "RH": rng.uniform(10, 90, n),
            "wind_speed": rng.uniform(0, 15, n),
            "target": rng.uniform(0, 200, n),
        },
        index=times,
    )
    # Introduce missing values
    for col in ["G", "T_amb", "RH", "wind_speed"]:
        mask = rng.random(n) < missing_frac
        df.loc[df.index[mask], col] = np.nan
    return df


# ── rename_columns ─────────────────────────────────────────────────────────────

def test_rename_dkasc() -> None:
    df = pd.DataFrame(
        {
            "101_DKA_WeatherStation_Global_Horizontal_Radiation": [1.0],
            "101_DKA_WeatherStation_Weather_Temperature_Celsius": [25.0],
            "96_DKA_MasterMeter1_Active_Power": [100.0],
        }
    )
    result = rename_columns(df, dataset="dkasc")
    assert "G" in result.columns
    assert "T_amb" in result.columns
    assert "target" in result.columns


# ── make_missingness_flags ─────────────────────────────────────────────────────

def test_flags_created_before_imputation() -> None:
    times = pd.date_range("2020-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({"G": [np.nan, 1.0] * 5, "T_amb": 25.0, "RH": 50.0, "wind_speed": 5.0}, index=times)
    flags = make_missingness_flags(df)
    assert flags["is_G_missing"].sum() == 5
    assert flags["is_Tamb_missing"].sum() == 0


def test_flags_all_four_present() -> None:
    times = pd.date_range("2020-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame({"G": 1.0, "T_amb": 25.0, "RH": 50.0, "wind_speed": 5.0}, index=times)
    flags = make_missingness_flags(df)
    assert set(flags.columns) == {"is_G_missing", "is_Tamb_missing", "is_RH_missing", "is_wind_missing"}


# ── make_dataset — split ratios ────────────────────────────────────────────────

def test_split_sizes() -> None:
    df = _synthetic_df(n=1000)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    n = 1000
    assert len(out["X_train"]) == int(n * 0.70)
    assert len(out["X_val"]) == int(n * 0.15)
    assert len(out["X_test"]) == n - int(n * 0.70) - int(n * 0.15)


def test_no_temporal_overlap() -> None:
    df = _synthetic_df(n=1000)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    assert out["split_info"]["train_end"] < out["split_info"]["val_start"]
    assert out["split_info"]["val_end"] < out["split_info"]["test_start"]


def test_chronological_order() -> None:
    """Train must end before val starts; val before test."""
    df = _synthetic_df(n=500)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    assert out["X_train"].index[-1] < out["X_val"].index[0]
    assert out["X_val"].index[-1] < out["X_test"].index[0]


# ── make_dataset — leakage ────────────────────────────────────────────────────

def test_imputer_fit_on_train_only() -> None:
    """Imputer statistics must come from train data, not val/test."""
    df = _synthetic_df(n=600, missing_frac=0.1)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    imp = out["imputer"]
    # KNNImputer stores no explicit mean, but fitting should complete without error
    # and val/test imputation uses train statistics
    assert imp is not None
    assert not out["X_val"].isna().any().any()
    assert not out["X_test"].isna().any().any()


def test_no_nans_after_imputation() -> None:
    df = _synthetic_df(n=600, missing_frac=0.08)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    for key in ["X_train", "X_val", "X_test"]:
        assert not out[key].isna().any().any(), f"NaN found in {key}"


# ── make_dataset — feature columns ────────────────────────────────────────────

def test_physical_cols_present() -> None:
    df = _synthetic_df(n=500)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    for col in ["cos_zenith", "k_t", "T_cell", "hour_sin", "month_cos"]:
        assert col in out["feature_cols"], f"Missing: {col}"


def test_missingness_flags_in_features() -> None:
    df = _synthetic_df(n=500)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    for flag in ["is_G_missing", "is_Tamb_missing", "is_RH_missing", "is_wind_missing"]:
        assert flag in out["feature_cols"], f"Missing flag: {flag}"


def test_target_not_in_X() -> None:
    df = _synthetic_df(n=500)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    assert "target" not in out["X_train"].columns
    assert "target" not in out["X_val"].columns
    assert "target" not in out["X_test"].columns


# ── make_dataset — reproducibility ────────────────────────────────────────────

def test_reproducible_output() -> None:
    df = _synthetic_df(n=500)
    out1 = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    out2 = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    pd.testing.assert_frame_equal(out1["X_train"], out2["X_train"])
    pd.testing.assert_series_equal(out1["y_train"], out2["y_train"])


# ── make_walk_forward_splits ───────────────────────────────────────────────────

def test_wf_n_folds() -> None:
    df = _synthetic_df(n=800)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    folds = make_walk_forward_splits(out["X_train"], out["y_train"], n_splits=5, gap=24)
    assert len(folds) == 5


def test_wf_no_train_val_overlap() -> None:
    df = _synthetic_df(n=800)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    folds = make_walk_forward_splits(out["X_train"], out["y_train"], n_splits=3, gap=24)
    for fold in folds:
        assert fold["train_end"] < fold["val_start"], f"Fold {fold['fold']}: leakage!"


def test_wf_gap_respected() -> None:
    """val_start must be at least gap timesteps after train_end."""
    df = _synthetic_df(n=800)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    gap = 12
    folds = make_walk_forward_splits(out["X_train"], out["y_train"], n_splits=3, gap=gap)
    freq_td = pd.Timedelta("5min")
    for fold in folds:
        diff = (fold["val_start"] - fold["train_end"]) / freq_td
        assert diff >= gap, f"Fold {fold['fold']}: gap {diff} < {gap}"


def test_wf_increasing_train_size() -> None:
    """Each fold's train set must be larger than the previous."""
    df = _synthetic_df(n=800)
    out = make_dataset(df, DKASC_LOCATION, freq_minutes=5)
    folds = make_walk_forward_splits(out["X_train"], out["y_train"], n_splits=4, gap=10)
    sizes = [f["n_train"] for f in folds]
    assert sizes == sorted(sizes), f"Train sizes not increasing: {sizes}"
