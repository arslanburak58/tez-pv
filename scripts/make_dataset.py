"""
STAGE-4: Kronolojik veri bölme + imputasyon + Walk-Forward iskeleti.

Kullanım:
    from scripts.make_dataset import make_dataset, make_walk_forward_splits

Kurallar:
    - shuffle=False (kronolojik sıra korunur)
    - KNNImputer SADECE train setinde fit edilir
    - Missingness flags imputasyondan ÖNCE yaratılır
    - Scaler bu modülde yok; model pipeline'ında olacak (STAGE-5)
"""

import logging
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

from features.physical import DKASC_COL_MAP, PVOD_COL_MAP, build_physical_features

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
# test_ratio = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

INTERP_GAP_HOURS: float = 3.0   # gaps ≤ this → linear interpolation
KNN_NEIGHBORS: int = 5

IMPUTER_STRATEGIES: tuple[str, ...] = ("ffill", "median", "knn")
WF_GAP: int = 24                 # Walk-Forward gap (timesteps, not hours)
WF_N_SPLITS: int = 5

SENSOR_COLS: list[str] = ["G", "T_amb", "RH", "wind_speed"]
MISSINGNESS_FLAG_COLS: list[str] = [
    "is_G_missing",
    "is_Tamb_missing",
    "is_RH_missing",
    "is_wind_missing",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def rename_columns(df: pd.DataFrame, dataset: str = "dkasc") -> pd.DataFrame:
    """Rename raw columns to canonical names (G, T_amb, RH, wind_speed, target)."""
    col_map = DKASC_COL_MAP if dataset == "dkasc" else PVOD_COL_MAP
    available = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=available)


def make_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for each sensor column. Created from RAW data before imputation."""
    flags = pd.DataFrame(index=df.index)
    col_flag_map = {
        "G": "is_G_missing",
        "T_amb": "is_Tamb_missing",
        "RH": "is_RH_missing",
        "wind_speed": "is_wind_missing",
    }
    for col, flag in col_flag_map.items():
        flags[flag] = df[col].isna().astype(int) if col in df.columns else 0
    return flags


def _gap_lengths(series: pd.Series) -> pd.Series:
    """Return NaN run-lengths at each NaN position (in timesteps)."""
    is_nan = series.isna()
    cumsum = (~is_nan).cumsum()
    return is_nan.groupby(cumsum).transform("sum") * is_nan


def _interpolate_short_gaps(
    df: pd.DataFrame,
    cols: list[str],
    freq_minutes: int,
    max_gap_hours: float,
) -> pd.DataFrame:
    """Linear interpolation for gaps ≤ max_gap_hours. Longer gaps left as NaN."""
    max_gap_steps = int(max_gap_hours * 60 / freq_minutes)
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        gap_len = _gap_lengths(df[col])
        short_gap_mask = (gap_len > 0) & (gap_len <= max_gap_steps)
        if short_gap_mask.any():
            interpolated = df[col].interpolate(method="linear")
            df.loc[short_gap_mask, col] = interpolated[short_gap_mask]
    return df


def _log_split_info(label: str, df: pd.DataFrame) -> None:
    log.info(
        "%s: %s → %s  (%d rows)",
        label,
        df.index[0].isoformat(),
        df.index[-1].isoformat(),
        len(df),
    )


# ── Main function ──────────────────────────────────────────────────────────────

def make_dataset(
    df: pd.DataFrame,
    location: dict[str, Any],
    target_col: str = "target",
    freq_minutes: int = 5,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    interp_gap_hours: float = INTERP_GAP_HOURS,
    n_neighbors: int = KNN_NEIGHBORS,
    imputer_path: str | None = None,
    imputer_strategy: str = "ffill",
) -> dict[str, Any]:
    """
    Ham DataFrame'den (X_train, y_train, X_val, y_val, X_test, y_test) üret.

    Adımlar (leakage-safe):
      1. Missingness flags → ham veriden
      2. Kronolojik 70/15/15 bölme
      3. Kısa boşluk interpolasyonu (her split ayrı ayrı)
      4. İmputation (imputer_strategy: "ffill" | "median" | "knn")
         - "ffill": ileri/geri doldurma — zaman serisi için varsayılan, O(n)
         - "median": train medyanı → val/test'e transform, O(n)
         - "knn": KNNImputer(n_neighbors) — doğru ama O(n²), büyük veride yavaş
      5. Fiziksel öznitelikler → imputasyondan sonra (NaN kalmaz)
      6. Scaler yok — model pipeline'ında olacak (STAGE-5)

    Returns:
        dict ile anahtarlar:
            X_train, y_train, X_val, y_val, X_test, y_test  (pd.DataFrame/Series)
            imputer        (fitted KNNImputer)
            split_info     (dict: sınır tarihleri + satır sayıları)
            feature_cols   (list[str])
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be pd.DatetimeIndex")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in df")

    df = df.sort_index()

    # ── 0. Target NaN satırlarını düşür ───────────────────────────────────────
    # Hedef değeri olmayan satırlar eğitim için kullanılamaz.
    nan_target = df[target_col].isna()
    if nan_target.any():
        log.warning(
            "Target NaN: %d satır düşürüldü (%% %.1f)",
            nan_target.sum(), nan_target.mean() * 100,
        )
        df = df[~nan_target].copy()

    # ── 1. Missingness flags (raw veriden, imputasyondan önce) ─────────────────
    flags = make_missingness_flags(df)

    # ── 2. Kronolojik 70/15/15 bölme (ham veri üzerinde) ──────────────────────
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_raw = df.iloc[:n_train].copy()
    val_raw = df.iloc[n_train : n_train + n_val].copy()
    test_raw = df.iloc[n_train + n_val :].copy()

    _log_split_info("TRAIN", train_raw)
    _log_split_info("VAL  ", val_raw)
    _log_split_info("TEST ", test_raw)

    split_info = {
        "train_start": train_raw.index[0],
        "train_end": train_raw.index[-1],
        "val_start": val_raw.index[0],
        "val_end": val_raw.index[-1],
        "test_start": test_raw.index[0],
        "test_end": test_raw.index[-1],
        "n_train": len(train_raw),
        "n_val": len(val_raw),
        "n_test": len(test_raw),
    }

    # ── 3. Her split içinde kısa boşluk interpolasyonu (sensör kolonları) ─────
    sensor_cols_avail = [c for c in SENSOR_COLS if c in df.columns]
    train_raw = _interpolate_short_gaps(train_raw, sensor_cols_avail, freq_minutes, interp_gap_hours)
    val_raw = _interpolate_short_gaps(val_raw, sensor_cols_avail, freq_minutes, interp_gap_hours)
    test_raw = _interpolate_short_gaps(test_raw, sensor_cols_avail, freq_minutes, interp_gap_hours)

    # ── 4. İmputation — imputer_strategy seçimine göre ────────────────────────
    # Tamamen NaN olan kolonlar imputable listesine alınmaz;
    # missingness flag zaten 1, tree modelleri NaN'ı tolere eder.
    if imputer_strategy not in IMPUTER_STRATEGIES:
        raise ValueError(
            f"imputer_strategy='{imputer_strategy}' geçersiz. "
            f"Seçenekler: {IMPUTER_STRATEGIES}"
        )
    imputable = [c for c in sensor_cols_avail if not train_raw[c].isna().all()]
    skipped   = [c for c in sensor_cols_avail if c not in imputable]
    if skipped:
        log.warning("Tamamı NaN, impute edilmedi: %s", skipped)

    imputer: Any = None

    if imputer_strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        if imputable:
            train_raw[imputable] = imputer.fit_transform(train_raw[imputable])
            val_raw[imputable]   = imputer.transform(val_raw[imputable])
            test_raw[imputable]  = imputer.transform(test_raw[imputable])

    elif imputer_strategy == "median":
        imputer = SimpleImputer(strategy="median")
        if imputable:
            train_raw[imputable] = imputer.fit_transform(train_raw[imputable])
            val_raw[imputable]   = imputer.transform(val_raw[imputable])
            test_raw[imputable]  = imputer.transform(test_raw[imputable])

    else:  # "ffill" — ileri/geri doldurma, durumsuz (stateless)
        for split in (train_raw, val_raw, test_raw):
            if imputable:
                split[imputable] = split[imputable].ffill().bfill()
            # Hâlâ NaN kalan sütunlar (split başından itibaren NaN): 0 ile doldur
            remaining_nan = [c for c in imputable if split[c].isna().any()]
            if remaining_nan:
                split[remaining_nan] = split[remaining_nan].fillna(0.0)
                log.warning("ffill sonrası 0 dolduruldu: %s", remaining_nan)

    if imputer_path and imputer is not None:
        joblib.dump(imputer, imputer_path)
        log.info("Imputer kaydedildi: %s", imputer_path)

    log.info("İmputation tamamlandı | strateji=%s", imputer_strategy)

    # ── 5. Fiziksel öznitelikler (imputasyondan SONRA — NaN kalmaz) ───────────
    train_feat = build_physical_features(train_raw, location, g_col="G", t_amb_col="T_amb")
    val_feat = build_physical_features(val_raw, location, g_col="G", t_amb_col="T_amb")
    test_feat = build_physical_features(test_raw, location, g_col="G", t_amb_col="T_amb")

    # ── 6. Missingness flags ekle ──────────────────────────────────────────────
    train_flags = flags.iloc[:n_train]
    val_flags = flags.iloc[n_train : n_train + n_val]
    test_flags = flags.iloc[n_train + n_val :]

    train_feat = pd.concat([train_feat, train_flags], axis=1)
    val_feat = pd.concat([val_feat, val_flags], axis=1)
    test_feat = pd.concat([test_feat, test_flags], axis=1)

    # ── 7. X / y ayır ─────────────────────────────────────────────────────────
    feature_cols = [c for c in train_feat.columns if c != target_col]

    X_train = train_feat[feature_cols]
    X_val = val_feat[feature_cols]
    X_test = test_feat[feature_cols]

    y_train = train_feat[target_col]
    y_val = val_feat[target_col]
    y_test = test_feat[target_col]

    log.info(
        "make_dataset tamamlandı | features: %d | train: %d | val: %d | test: %d",
        len(feature_cols),
        len(X_train),
        len(X_val),
        len(X_test),
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "imputer": imputer,
        "split_info": split_info,
        "feature_cols": feature_cols,
    }


# ── Walk-Forward iskeleti ──────────────────────────────────────────────────────

def make_walk_forward_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = WF_N_SPLITS,
    gap: int = WF_GAP,
) -> list[dict[str, Any]]:
    """
    TimeSeriesSplit(gap=gap) ile Walk-Forward fold'ları üret.

    gap=24: her fold'un test başlangıcı, train bitişinden 24 timestep sonra başlar
    (DKASC 5-dk veride gap=24 → 2 saatlik boşluk; ayarlamak gerekirse kwarg olarak geç).

    Returns:
        list of dicts: her dict'te
            fold, X_tr, y_tr, X_vl, y_vl, train_end, val_start
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    folds: list[dict[str, Any]] = []

    X_arr = X_train.values
    y_arr = y_train.values

    for fold_idx, (tr_idx, vl_idx) in enumerate(tscv.split(X_arr)):
        fold_data: dict[str, Any] = {
            "fold": fold_idx,
            "X_tr": X_train.iloc[tr_idx],
            "y_tr": y_train.iloc[tr_idx],
            "X_vl": X_train.iloc[vl_idx],
            "y_vl": y_train.iloc[vl_idx],
            "train_end": X_train.index[tr_idx[-1]],
            "val_start": X_train.index[vl_idx[0]],
            "n_train": len(tr_idx),
            "n_val": len(vl_idx),
        }
        folds.append(fold_data)
        log.info(
            "Fold %d | train end: %s | val start: %s | n_train: %d | n_val: %d",
            fold_idx,
            fold_data["train_end"].isoformat(),
            fold_data["val_start"].isoformat(),
            fold_data["n_train"],
            fold_data["n_val"],
        )

    return folds
