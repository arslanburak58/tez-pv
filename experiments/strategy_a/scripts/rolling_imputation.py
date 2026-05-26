"""Rolling same-hour imputation — Strategy A experiment.

Leakage-free: only uses past observations (window_end = idx - 1 second).
"""
from __future__ import annotations

import pandas as pd


def rolling_same_hour_imputation(
    df: pd.DataFrame,
    target_cols: list[str],
    window_days: int = 7,
    fallback_days: int = 30,
) -> pd.DataFrame:
    df_filled = df.copy()
    for col in target_cols:
        missing_mask = df_filled[col].isna()
        if not missing_mask.any():
            continue
        for idx in df_filled.index[missing_mask]:
            hour = idx.hour
            window_start = idx - pd.Timedelta(days=window_days)
            window_end = idx - pd.Timedelta(seconds=1)
            window_data = df_filled.loc[
                (df_filled.index >= window_start) & (df_filled.index <= window_end)
                & (df_filled.index.hour == hour), col
            ].dropna()
            if len(window_data) > 0:
                df_filled.at[idx, col] = window_data.mean()
                continue
            window_start = idx - pd.Timedelta(days=fallback_days)
            window_data = df_filled.loc[
                (df_filled.index >= window_start) & (df_filled.index <= window_end)
                & (df_filled.index.hour == hour), col
            ].dropna()
            if len(window_data) > 0:
                df_filled.at[idx, col] = window_data.mean()
            else:
                df_filled.at[idx, col] = df_filled[col].mean()
    return df_filled
