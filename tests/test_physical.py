"""Unit tests for features/physical.py — STAGE-3.9"""

import math
import random

import numpy as np
import pandas as pd
import pytest

random.seed(42)
np.random.seed(42)

from features.physical import (
    DKASC_LOCATION,
    NOCT,
    PHYSICAL_FEATURE_COLS,
    build_physical_features,
    build_pipeline,
    compute_cell_temp,
    compute_clearness_index,
    compute_cos_zenith,
    compute_cyclic_features,
    compute_hour_angle,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _daytime_times() -> pd.DatetimeIndex:
    # 12:00 Darwin (UTC+9:30) = 02:30 UTC → solar noon at Alice Springs
    return pd.date_range("2020-06-21 02:30:00", periods=4, freq="15min", tz="UTC")


def _nighttime_times() -> pd.DatetimeIndex:
    # 20:00 Darwin = 10:30 UTC → night
    return pd.date_range("2020-06-21 10:30:00", periods=4, freq="15min", tz="UTC")


def _df(times: pd.DatetimeIndex, G: float = 600.0, T_amb: float = 25.0) -> pd.DataFrame:
    return pd.DataFrame({"G": G, "T_amb": T_amb}, index=times)


# ── cos_zenith ─────────────────────────────────────────────────────────────────

def test_cos_zenith_positive_at_noon() -> None:
    # June 21 = winter solstice in southern hemisphere; zenith ~47° → cos ~0.68
    result = compute_cos_zenith(_daytime_times(), DKASC_LOCATION)
    assert result.max() > 0.5


def test_cos_zenith_zero_at_night() -> None:
    result = compute_cos_zenith(_nighttime_times(), DKASC_LOCATION)
    assert (result == 0.0).all()


def test_cos_zenith_clipped_nonnegative() -> None:
    times = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
    result = compute_cos_zenith(times, DKASC_LOCATION)
    assert (result >= 0.0).all()


# ── hour_angle ────────────────────────────────────────────────────────────────

def test_hour_angle_near_zero_at_solar_noon() -> None:
    # Single point at approximate solar noon UTC for Alice Springs
    times = pd.DatetimeIndex(["2020-06-21 02:30:00"], tz="UTC")
    result = compute_hour_angle(times, DKASC_LOCATION)
    assert abs(result.iloc[0]) < 15.0  # within ±1 hour of solar noon


def test_hour_angle_returns_series() -> None:
    result = compute_hour_angle(_daytime_times(), DKASC_LOCATION)
    assert isinstance(result, pd.Series)
    assert result.name == "hour_angle"


# ── k_t ───────────────────────────────────────────────────────────────────────

def test_k_t_zero_when_G_zero() -> None:
    times = _daytime_times()
    kt = compute_clearness_index(pd.Series(0.0, index=times), times, DKASC_LOCATION)
    assert (kt == 0.0).all()


def test_k_t_zero_at_night() -> None:
    times = _nighttime_times()
    # Even with non-zero G, nighttime k_t must be 0
    kt = compute_clearness_index(pd.Series(100.0, index=times), times, DKASC_LOCATION)
    assert (kt == 0.0).all()


def test_k_t_bounds() -> None:
    times = _daytime_times()
    kt = compute_clearness_index(pd.Series(800.0, index=times), times, DKASC_LOCATION)
    assert (kt >= 0.0).all()
    assert (kt <= 1.5).all()


# ── T_cell ────────────────────────────────────────────────────────────────────

def test_cell_temp_formula_at_800Wm2() -> None:
    """At G=800 W/m², T_cell = T_amb + (NOCT - 20)."""
    times = _daytime_times()
    G = pd.Series(800.0, index=times)
    T_amb = pd.Series(25.0, index=times)
    expected = 25.0 + (NOCT - 20.0)  # G*(NOCT-20)/800 = NOCT-20 when G=800
    np.testing.assert_allclose(compute_cell_temp(T_amb, G).values, expected, rtol=1e-9)


def test_cell_temp_equals_tamb_at_zero_irradiance() -> None:
    times = _daytime_times()
    G = pd.Series(0.0, index=times)
    T_amb = pd.Series(30.0, index=times)
    np.testing.assert_allclose(compute_cell_temp(T_amb, G).values, 30.0, rtol=1e-9)


def test_cell_temp_custom_noct() -> None:
    times = _daytime_times()
    G = pd.Series(800.0, index=times)
    T_amb = pd.Series(20.0, index=times)
    noct = 48.0
    expected = 20.0 + 800.0 * (48.0 - 20.0) / 800.0  # = 48.0
    np.testing.assert_allclose(compute_cell_temp(T_amb, G, noct=noct).values, expected, rtol=1e-9)


# ── Cyclic features ────────────────────────────────────────────────────────────

def test_cyclic_at_midnight() -> None:
    times = pd.DatetimeIndex(["2020-01-01 00:00:00"], tz="UTC")
    df = compute_cyclic_features(times)
    np.testing.assert_allclose(df["hour_sin"].iloc[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(df["hour_cos"].iloc[0], 1.0, atol=1e-10)


def test_cyclic_at_6am() -> None:
    times = pd.DatetimeIndex(["2020-01-01 06:00:00"], tz="UTC")
    df = compute_cyclic_features(times)
    expected = math.sin(2 * math.pi * 6 / 24)  # sin(π/2) = 1.0
    np.testing.assert_allclose(df["hour_sin"].iloc[0], expected, atol=1e-10)


def test_cyclic_at_12pm() -> None:
    times = pd.DatetimeIndex(["2020-01-01 12:00:00"], tz="UTC")
    df = compute_cyclic_features(times)
    np.testing.assert_allclose(df["hour_sin"].iloc[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(df["hour_cos"].iloc[0], -1.0, atol=1e-10)


def test_cyclic_bounds() -> None:
    times = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
    df = compute_cyclic_features(times)
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert df[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_cyclic_columns() -> None:
    df = compute_cyclic_features(_daytime_times())
    assert set(df.columns) == {"hour_sin", "hour_cos", "month_sin", "month_cos"}


# ── build_physical_features ───────────────────────────────────────────────────

def test_build_contains_all_physical_cols() -> None:
    times = _daytime_times()
    result = build_physical_features(_df(times), DKASC_LOCATION)
    for col in PHYSICAL_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_build_no_nans_in_physical_cols() -> None:
    times = _daytime_times()
    result = build_physical_features(_df(times), DKASC_LOCATION)
    assert not result[PHYSICAL_FEATURE_COLS].isna().any().any()


def test_build_preserves_original_cols() -> None:
    times = _daytime_times()
    df = _df(times)
    result = build_physical_features(df, DKASC_LOCATION)
    assert "G" in result.columns and "T_amb" in result.columns


def test_build_raises_without_datetime_index() -> None:
    df = pd.DataFrame({"G": [1.0], "T_amb": [20.0]})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        build_physical_features(df, DKASC_LOCATION)


def test_build_raises_missing_g_col() -> None:
    times = _daytime_times()
    df = pd.DataFrame({"T_amb": 25.0}, index=times)
    with pytest.raises(ValueError, match="'G'"):
        build_physical_features(df, DKASC_LOCATION)


# ── Pipeline — scaler fit on train only ───────────────────────────────────────

def test_pipeline_fit_transform_shape() -> None:
    times_train = _daytime_times()
    times_test = pd.date_range("2020-07-01 02:30", periods=3, freq="15min", tz="UTC")
    pipe = build_pipeline(DKASC_LOCATION)
    pipe.fit(_df(times_train, G=700.0, T_amb=30.0))
    result = pipe.transform(_df(times_test, G=400.0, T_amb=20.0))
    assert result.shape[0] == 3


def test_pipeline_no_nans() -> None:
    times_train = _daytime_times()
    times_test = pd.date_range("2020-07-01 02:30", periods=3, freq="15min", tz="UTC")
    pipe = build_pipeline(DKASC_LOCATION)
    pipe.fit(_df(times_train))
    result = pipe.transform(_df(times_test))
    assert not np.isnan(result).any()


def test_pipeline_scaler_learned_from_train() -> None:
    times_train = _daytime_times()
    pipe = build_pipeline(DKASC_LOCATION)
    pipe.fit(_df(times_train, G=1000.0, T_amb=40.0))
    scaler = pipe.named_steps["scaler"]
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None
