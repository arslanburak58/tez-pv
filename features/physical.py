"""Physical feature engineering pipeline using pvlib."""

import math
import random
from typing import Any

import numpy as np
import pandas as pd
import pvlib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────

NOCT: float = 46.0  # nominal operating cell temperature [°C]

DKASC_LOCATION: dict[str, Any] = {
    "latitude": -23.762,
    "longitude": 133.875,
    "altitude": 546.0,
    "tz": "Australia/Darwin",
}

# DKASC raw → canonical column mapping
DKASC_COL_MAP: dict[str, str] = {
    "101_DKA_WeatherStation_Global_Horizontal_Radiation": "G",
    "101_DKA_WeatherStation_Weather_Temperature_Celsius": "T_amb",
    "101_DKA_WeatherStation_Weather_Relative_Humidity": "RH",
    "101_DKA_WeatherStation_Wind_Speed": "wind_speed",
    "96_DKA_MasterMeter1_Active_Power": "target",
}

# PVOD raw → canonical column mapping
PVOD_COL_MAP: dict[str, str] = {
    "lmd_totalirrad": "G",
    "lmd_temperature": "T_amb",
    "nwp_humidity": "RH",
    "lmd_windspeed": "wind_speed",
    "power": "target",
}

# ── Internal helper ────────────────────────────────────────────────────────────

def _make_location(cfg: dict[str, Any]) -> pvlib.location.Location:
    return pvlib.location.Location(
        latitude=cfg["latitude"],
        longitude=cfg["longitude"],
        tz=cfg["tz"],
        altitude=float(cfg.get("altitude", 0.0)),
    )


def _solar_position(times: pd.DatetimeIndex, location: dict[str, Any]) -> pd.DataFrame:
    return _make_location(location).get_solarposition(times)


def _to_utc_hours(times: pd.DatetimeIndex) -> pd.Series:
    utc = times.tz_convert("UTC") if times.tz is not None else times
    return pd.Series(
        utc.hour + utc.minute / 60.0 + utc.second / 3600.0,
        index=times,
    )


# ── Feature functions ──────────────────────────────────────────────────────────

def compute_cos_zenith(times: pd.DatetimeIndex, location: dict[str, Any]) -> pd.Series:
    """cos(apparent_zenith) clipped to [0, 1]. Nighttime → 0."""
    sp = _solar_position(times, location)
    return np.cos(np.deg2rad(sp["apparent_zenith"])).clip(lower=0.0).rename("cos_zenith")


def compute_hour_angle(times: pd.DatetimeIndex, location: dict[str, Any]) -> pd.Series:
    """Solar hour angle in degrees. 0° at solar noon, negative AM, positive PM."""
    sp = _solar_position(times, location)
    eot = sp["equation_of_time"]          # minutes
    lon = float(location["longitude"])
    utc_h = _to_utc_hours(times)
    solar_time = utc_h + (eot + 4.0 * lon) / 60.0
    return ((solar_time - 12.0) * 15.0).rename("hour_angle")


def compute_air_mass(times: pd.DatetimeIndex, location: dict[str, Any]) -> pd.Series:
    """Kasten-Young relative air mass. Nighttime (zenith > 90°) → 0."""
    sp = _solar_position(times, location)
    am = pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"], model="kastenyoung1989")
    return am.fillna(0.0).clip(lower=0.0).rename("air_mass")


def compute_clearness_index(
    G: pd.Series,
    times: pd.DatetimeIndex,
    location: dict[str, Any],
) -> pd.Series:
    """k_t = G / G_0 (extraterrestrial horizontal irradiance). Nighttime → 0."""
    sp = _solar_position(times, location)
    cos_z = np.cos(np.deg2rad(sp["apparent_zenith"])).clip(lower=0.0)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    G_0 = (dni_extra * cos_z).clip(lower=1e-6)
    k_t = (G / G_0).clip(lower=0.0, upper=1.5)
    return k_t.where(cos_z > 0.01, other=0.0).rename("k_t")


def compute_cell_temp(T_amb: pd.Series, G: pd.Series, noct: float = NOCT) -> pd.Series:
    """Ross model: T_cell = T_amb + G * (NOCT - 20) / 800."""
    return (T_amb + G * (noct - 20.0) / 800.0).rename("T_cell")


def compute_cyclic_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic sin/cos encoding of clock hour and calendar month."""
    hour = times.hour + times.minute / 60.0
    month = times.month
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2.0 * math.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * math.pi * hour / 24.0),
            "month_sin": np.sin(2.0 * math.pi * (month - 1) / 12.0),
            "month_cos": np.cos(2.0 * math.pi * (month - 1) / 12.0),
        },
        index=times,
    )


# ── Main builder ───────────────────────────────────────────────────────────────

PHYSICAL_FEATURE_COLS: list[str] = [
    "cos_zenith",
    "hour_angle",
    "air_mass",
    "k_t",
    "T_cell",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


def build_physical_features(
    df: pd.DataFrame,
    location: dict[str, Any],
    g_col: str = "G",
    t_amb_col: str = "T_amb",
    noct: float = NOCT,
) -> pd.DataFrame:
    """
    Append pvlib-derived physical features to df.

    df must have a DatetimeIndex and columns g_col, t_amb_col.
    Returns a copy with additional columns listed in PHYSICAL_FEATURE_COLS.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be pd.DatetimeIndex")
    if g_col not in df.columns:
        raise ValueError(f"Column '{g_col}' not found. Available: {list(df.columns)}")
    if t_amb_col not in df.columns:
        raise ValueError(f"Column '{t_amb_col}' not found. Available: {list(df.columns)}")

    times = df.index
    G = df[g_col]
    T_amb = df[t_amb_col]

    out = df.copy()
    out["cos_zenith"] = compute_cos_zenith(times, location)
    out["hour_angle"] = compute_hour_angle(times, location)
    out["air_mass"] = compute_air_mass(times, location)
    out["k_t"] = compute_clearness_index(G, times, location)
    out["T_cell"] = compute_cell_temp(T_amb, G, noct)

    return pd.concat([out, compute_cyclic_features(times)], axis=1)


# ── Sklearn transformer ────────────────────────────────────────────────────────

class PhysicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer wrapping build_physical_features."""

    def __init__(
        self,
        location: dict[str, Any] = DKASC_LOCATION,
        g_col: str = "G",
        t_amb_col: str = "T_amb",
        noct: float = NOCT,
    ) -> None:
        self.location = location
        self.g_col = g_col
        self.t_amb_col = t_amb_col
        self.noct = noct

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PhysicalFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_physical_features(
            X,
            location=self.location,
            g_col=self.g_col,
            t_amb_col=self.t_amb_col,
            noct=self.noct,
        )


def build_pipeline(
    location: dict[str, Any] = DKASC_LOCATION,
    g_col: str = "G",
    t_amb_col: str = "T_amb",
    noct: float = NOCT,
) -> Pipeline:
    """Return Pipeline: physical features → StandardScaler. Fit scaler on train only."""
    return Pipeline(
        [
            ("physical", PhysicalFeatureTransformer(location, g_col, t_amb_col, noct)),
            ("scaler", StandardScaler()),
        ]
    )
