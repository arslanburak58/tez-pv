"""
Strategy A izole pipeline: rolling same-hour imputation vs ffill (main v7).

Hipotez: ffill → stale G=0 değerleri → base model over-predict → meta agresif
düzeltme öğreniyor (büyük negatif flag koef). Rolling imputation bu patolojiyi
gidererek daha semantik flag katsayıları ve gerçek CRPS iyileşmesi sağlar mı?

Çıktılar (main'e DOKUNULMAZ):
  experiments/strategy_a/data/dataset_sa.joblib
  experiments/strategy_a/models/base_models_sa.joblib
  experiments/strategy_a/data/x_meta_sa.joblib
  experiments/strategy_a/models/meta_models_sa.joblib
  experiments/strategy_a/models/meta_models_robust_sa.joblib
  experiments/strategy_a/data/stage8_results_sa.joblib
  experiments/strategy_a/figures/

Çalıştırma:
  PYTHONPATH=. tez-env/bin/python experiments/strategy_a/scripts/run_pipeline_strategy_a.py
  PYTHONPATH=. tez-env/bin/python experiments/strategy_a/scripts/run_pipeline_strategy_a.py --force
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# ── Proje kökü ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # rolling_imputation modülü

from rolling_imputation import rolling_same_hour_imputation

from features.physical import DKASC_LOCATION, build_physical_features
from models.base_learners import (
    ALGOS, META_COLS, QUANTILES as BASE_QUANTILES,
    _col_name, build_x_meta, pinball_loss, train_all_base_learners,
)
from models.meta_learner import (
    FLAG_COLS, QuantileLinear, QuantileLinearBounded,
    QUANTILES as META_QUANTILES, _meta_key, _q_cols,
    enrich_x_meta, predict_intervals, train_all_meta_learners,
)
from models.baselines import evaluate_quantiles
from evaluation.comparison import enforce_monotonicity, normalize_stacked_preds
from evaluation.cqr import apply_locally_scaled
from evaluation.robustness import diebold_mariano_test

# ── Çıktı dizinleri ─────────────────────────────────────────────────────────────

EXPR_DIR   = ROOT / "experiments" / "strategy_a"
DATA_DIR   = EXPR_DIR / "data"
MODEL_DIR  = EXPR_DIR / "models"
FIG_DIR    = EXPR_DIR / "figures"

for d in (DATA_DIR, MODEL_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXPR_DIR / "run_pipeline_sa.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Sabitler ───────────────────────────────────────────────────────────────────

TRAIN_RATIO:         float = 0.70
VAL_RATIO:           float = 0.15
SENSOR_COLS:         list[str] = ["G", "T_amb", "RH"]
CONTEXT_DAYS:        int   = 30    # rolling imputation context window for val/test
NOCT:                float = 46.0
ALPHA:               float = 1.0
FLAG_BOUND:          float = 1.0
DAYLIGHT_THRESHOLD:  float = 0.087
CQR_K:               float = 2.0
BURST_INTENSITY:     float = 0.30
SENSOR_RATE:         float = 0.30
FREQ_MINUTES:        int   = 5
STEPS_PER_HOUR:      int   = 60 // FREQ_MINUTES   # = 12 for 5-min data

BEST_PARAMS_PATH = ROOT / "models" / "checkpoints" / "best_params.json"

SENSOR_TO_FLAG: dict[str, str] = {
    "G":     "is_G_missing",
    "T_amb": "is_Tamb_missing",
    "RH":    "is_RH_missing",
}

# ── Yardımcılar ────────────────────────────────────────────────────────────────

def _elapsed(label: str, t0: float) -> None:
    log.info("  ✓ %s | %.1f s", label, time.time() - t0)


def _skip(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        log.info("  ↩ Atlanıyor (mevcut): %s", path.name)
        return True
    return False


def _load_best_params() -> dict[str, dict]:
    if not BEST_PARAMS_PATH.exists():
        log.warning("  best_params.json bulunamadı — DEFAULT_PARAMS kullanılacak")
        return {}
    with open(BEST_PARAMS_PATH) as f:
        bp = json.load(f)
    return {k: v for k, v in bp.items() if not k.startswith("_") and k != "ridge_alpha"}


def _rolling_impute_with_context(
    raw_split: pd.DataFrame,
    context_df: pd.DataFrame | None,
    cols: list[str],
) -> pd.DataFrame:
    """
    context_df varsa (val/test için): context + raw_split concat, impute, context'i düşür.
    context_df yoksa (train için): raw_split'e direkt uygula.
    """
    if context_df is None:
        return rolling_same_hour_imputation(raw_split, cols)

    combined = pd.concat([context_df[cols], raw_split[cols]])
    filled   = rolling_same_hour_imputation(combined, cols)
    split_filled = raw_split.copy()
    split_filled[cols] = filled.loc[raw_split.index, cols].values
    return split_filled


def _recompute_derived(df: pd.DataFrame, location: dict) -> pd.DataFrame:
    """G veya T_amb değiştiyse T_cell, k_t, cos_zenith yeniden hesapla."""
    df = df.copy()
    times = df.index
    from features.physical import (
        compute_cell_temp, compute_clearness_index, compute_cos_zenith,
    )
    if "T_cell" in df.columns:
        df["T_cell"] = compute_cell_temp(df["T_amb"], df["G"])
    if "k_t" in df.columns:
        df["k_t"] = compute_clearness_index(df["G"], times, location)
    return df


def _make_burst_mask(n: int, burst_steps: int, intensity: float, rng) -> np.ndarray:
    target  = int(n * intensity)
    mask    = np.zeros(n, dtype=bool)
    starts  = list(range(0, n - burst_steps + 1, burst_steps))
    rng.shuffle(starts)
    covered = 0
    for s in starts:
        if covered >= target:
            break
        end = min(s + burst_steps, n)
        mask[s:end] = True
        covered += end - s
    return mask


# ── ADIM 1: Ham veri yükle ────────────────────────────────────────────────────

def step1_load_raw(force: bool = False) -> pd.DataFrame:
    cache = ROOT / "data" / "processed" / "dkasc_raw.joblib"
    if not cache.exists():
        raise FileNotFoundError(
            f"Ham veri bulunamadı: {cache}\n"
            "Önce main pipeline'da step1_load_dkasc() çalıştır."
        )
    log.info("=== ADIM 1: Ham veri yükleniyor (read-only) ===")
    df = joblib.load(cache)
    log.info("  Satır: %d | Sütunlar: %s", len(df), list(df.columns))
    return df


# ── ADIM 2: Dataset — rolling imputation ─────────────────────────────────────

def step2_make_dataset_rolling(
    df: pd.DataFrame, force: bool = False
) -> dict:
    out_path = DATA_DIR / "dataset_sa.joblib"
    if _skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    log.info("=== ADIM 2: make_dataset — rolling imputation ===")

    df = df.sort_index()

    # Hedef NaN satırlarını düşür
    nan_tgt = df["target"].isna()
    if nan_tgt.any():
        log.warning("  Target NaN düşürüldü: %d satır", nan_tgt.sum())
        df = df[~nan_tgt].copy()

    # 1. Missingness flags (raw NaN'dan, imputasyondan önce)
    flags = pd.DataFrame({
        "is_G_missing":    df["G"].isna().astype(int),
        "is_Tamb_missing": df["T_amb"].isna().astype(int),
        "is_RH_missing":   df["RH"].isna().astype(int),
    }, index=df.index)

    # 2. Kronolojik 70/15/15 bölme
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_raw = df.iloc[:n_train].copy()
    val_raw   = df.iloc[n_train : n_train + n_val].copy()
    test_raw  = df.iloc[n_train + n_val :].copy()

    log.info(
        "  Train: %d (%s→%s) | Val: %d | Test: %d",
        len(train_raw), train_raw.index[0].date(), train_raw.index[-1].date(),
        len(val_raw), len(test_raw),
    )

    # 3. Rolling imputation (sensor sütunları)
    avail = [c for c in SENSOR_COLS if c in df.columns]
    log.info("  Rolling imputation — train...")
    train_filled = _rolling_impute_with_context(train_raw, None, avail)

    context_for_val = train_filled.iloc[-CONTEXT_DAYS * 24 * STEPS_PER_HOUR :]
    log.info("  Rolling imputation — val (context: %d satır)...", len(context_for_val))
    val_filled = _rolling_impute_with_context(val_raw, context_for_val, avail)

    context_for_test = val_filled.iloc[-CONTEXT_DAYS * 24 * STEPS_PER_HOUR :]
    log.info("  Rolling imputation — test (context: %d satır)...", len(context_for_test))
    test_filled = _rolling_impute_with_context(test_raw, context_for_test, avail)

    # Kalan NaN → 0 (nadir: split başından itibaren eksik)
    for split in (train_filled, val_filled, test_filled):
        remaining = [c for c in avail if split[c].isna().any()]
        if remaining:
            split[remaining] = split[remaining].fillna(0.0)
            log.warning("  Kalan NaN sıfırlandı: %s", remaining)

    # 4. Fiziksel öznitelikler
    log.info("  Fiziksel öznitelikler üretiliyor...")
    train_feat = build_physical_features(train_filled, DKASC_LOCATION)
    val_feat   = build_physical_features(val_filled,   DKASC_LOCATION)
    test_feat  = build_physical_features(test_filled,  DKASC_LOCATION)

    # 5. Missingness flags ekle
    train_feat = pd.concat([train_feat, flags.iloc[:n_train]], axis=1)
    val_feat   = pd.concat([val_feat,   flags.iloc[n_train : n_train + n_val]], axis=1)
    test_feat  = pd.concat([test_feat,  flags.iloc[n_train + n_val :]], axis=1)

    feature_cols = [c for c in train_feat.columns if c != "target"]

    dataset = {
        "X_train": train_feat[feature_cols],
        "y_train": train_feat["target"],
        "X_val":   val_feat[feature_cols],
        "y_val":   val_feat["target"],
        "X_test":  test_feat[feature_cols],
        "y_test":  test_feat["target"],
        "feature_cols": feature_cols,
        # STAGE-8 rolling recovery için impute edilmiş sensor sütunları
        "val_sensor_tail":  val_filled[avail].iloc[-CONTEXT_DAYS * 24 * STEPS_PER_HOUR :],
        "test_sensor_raw":  test_raw[avail],  # orijinal NaN'ler + STAGE-8 korupsiyon context'i
    }

    joblib.dump(dataset, out_path)
    log.info(
        "  Kaydedildi: %s | features: %d", out_path.name, len(feature_cols)
    )
    _elapsed("make_dataset_rolling", t0)
    return dataset


# ── ADIM 3: Taban modeller (OOF + final) ────────────────────────────────────

def step3_base_learners(dataset: dict, force: bool = False) -> tuple[dict, pd.DataFrame]:
    base_path   = MODEL_DIR  / "base_models_sa.joblib"
    x_meta_path = DATA_DIR   / "x_meta_sa.joblib"

    if base_path.exists() and x_meta_path.exists() and not force:
        log.info("  ↩ Atlanıyor: base_models_sa.joblib + x_meta_sa.joblib")
        return joblib.load(base_path), joblib.load(x_meta_path)[0]

    t0 = time.time()
    log.info("=== ADIM 3: 9 taban model — OOF (5 fold) + final eğitim ===")

    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    best_params = _load_best_params()

    log.info("  OOF tahminleri...")
    X_meta, oof_scores = build_x_meta(X_train, y_train, params_override=best_params)
    log.info("  OOF bitti | X_meta: %s", X_meta.shape)

    log.info("  Final taban modeller...")
    base_models = train_all_base_learners(
        X_train, y_train,
        checkpoint_dir=str(MODEL_DIR),
        params_override=best_params,
    )

    joblib.dump(base_models, base_path)
    joblib.dump((X_meta, oof_scores), x_meta_path)
    _elapsed("Taban modeller", t0)
    return base_models, X_meta


# ── ADIM 4a: Clean meta-öğreniciler ─────────────────────────────────────────

def step4a_clean_meta(dataset: dict, X_meta: pd.DataFrame, force: bool = False) -> dict:
    out_path = MODEL_DIR / "meta_models_sa.joblib"
    if _skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    log.info("=== ADIM 4a: Clean meta-öğreniciler ===")

    flags_train  = dataset["X_train"][FLAG_COLS]
    y_meta       = dataset["y_train"].reindex(X_meta.index)
    X_meta_12    = enrich_x_meta(X_meta, flags_train)

    meta_models  = train_all_meta_learners(X_meta_12, y_meta, alpha=ALPHA,
                                           checkpoint_dir=str(MODEL_DIR))
    joblib.dump(meta_models, out_path)
    _elapsed("Clean meta", t0)
    return meta_models


# ── ADIM 4b: Robust meta (v7-eşdeğer, rolling imputation ile) ───────────────

def step4b_robust_meta(
    dataset: dict, base_models: dict, force: bool = False
) -> dict:
    out_path = MODEL_DIR / "meta_models_robust_sa.joblib"
    if _skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    log.info("=== ADIM 4b: Robust meta (rolling imputation augmentation) ===")

    X_val   = dataset["X_val"].copy()
    y_val   = dataset["y_val"]
    n_val   = len(X_val)

    # Clean OOF X_meta
    X_meta_clean = joblib.load(DATA_DIR / "x_meta_sa.joblib")[0]
    flags_clean  = dataset["X_train"][FLAG_COLS]
    X_meta_clean_12 = enrich_x_meta(X_meta_clean, flags_clean)
    y_clean = dataset["y_train"].reindex(X_meta_clean_12.index)

    # Val sensor tail for rolling context
    val_sensor_tail  = dataset["val_sensor_tail"]
    avail_sensors    = [c for c in SENSOR_COLS if c in X_val.columns]

    def _corrupt_rolling_enrich(
        X: pd.DataFrame, sensor: str, row_mask: np.ndarray
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Corrupt → rolling impute (with val tail context) → base preds → X12."""
        # Build context + corrupted sensor series
        corrupted_sensor = X[avail_sensors].copy()
        corrupted_sensor.loc[X.index[row_mask], sensor] = np.nan

        context_combined = pd.concat([val_sensor_tail, corrupted_sensor])
        filled_combined  = rolling_same_hour_imputation(
            context_combined, [sensor]
        )
        X_c = X.copy()
        X_c[sensor] = filled_combined.loc[X.index, sensor].values

        # Recompute T_cell / k_t if sensor affects them
        if sensor in ("G", "T_amb"):
            X_c = _recompute_derived(X_c, DKASC_LOCATION)

        # Missingness flags
        flags_c = pd.DataFrame(
            {fc: np.zeros(n_val, dtype=int) for fc in FLAG_COLS},
            index=X_c.index,
        )
        flags_c[SENSOR_TO_FLAG[sensor]] = row_mask.astype(int)

        bp  = {c: np.asarray(base_models[c].predict(X_c), dtype=np.float64)
               for c in META_COLS}
        Xm  = pd.DataFrame(bp, index=X_c.index)
        return enrich_x_meta(Xm, flags_c), y_val.loc[X_c.index]

    # --- Random single-sensor batch (seed 99) ---
    log.info("  Augmentation — random single-sensor (rolling)...")
    rng5     = np.random.default_rng(99)
    rates5   = rng5.uniform(0.10, 0.50, n_val)
    corrupt5 = rng5.random(n_val) < rates5
    sidx5    = rng5.integers(0, 3, n_val)

    rnd_batches_x12: list[pd.DataFrame] = []
    rnd_batches_y:   list[pd.Series]    = []
    for i, sensor in enumerate(SENSOR_COLS):
        if sensor not in X_val.columns:
            continue
        mask = corrupt5 & (sidx5 == i)
        if mask.sum() == 0:
            continue
        x12_b, y_b = _corrupt_rolling_enrich(X_val, sensor, mask)
        rnd_batches_x12.append(x12_b)
        rnd_batches_y.append(y_b)
        log.info("    rnd %s: %d satır", sensor, mask.sum())

    # --- Burst single-sensor batch ---
    BURST_CONFIGS = [
        ("G",     1, 0.04),
        ("G",     6, 0.04),
        ("G",    24, 0.04),
        ("T_amb", 6, 0.04),
        ("RH",    1, 0.04),
    ]
    log.info("  Augmentation — burst single-sensor (rolling)...")
    burst_batches_x12: list[pd.DataFrame] = []
    burst_batches_y:   list[pd.Series]    = []
    for sensor_b, dur_h, intensity_b in BURST_CONFIGS:
        if sensor_b not in X_val.columns:
            continue
        bsteps = dur_h * STEPS_PER_HOUR
        bmask  = _make_burst_mask(n_val, bsteps, intensity_b, rng5)
        if bmask.sum() == 0:
            continue
        x12_b, y_b = _corrupt_rolling_enrich(X_val, sensor_b, bmask)
        burst_batches_x12.append(x12_b)
        burst_batches_y.append(y_b)
        log.info("    burst %s %dh: %d satır", sensor_b, dur_h, bmask.sum())

    X_combined = pd.concat(
        [X_meta_clean_12] + rnd_batches_x12 + burst_batches_x12,
        ignore_index=True,
    )
    y_combined = pd.concat(
        [y_clean] + rnd_batches_y + burst_batches_y,
        ignore_index=True,
    )
    log.info("  Augmented dataset: %s", X_combined.shape)

    # Train QuantileLinearBounded (same as v7)
    robust_models: dict[str, QuantileLinear] = {}
    for q in META_QUANTILES:
        key  = _meta_key(q)
        cols = _q_cols(q)
        Xq   = X_combined[cols].to_numpy()
        m    = QuantileLinearBounded(quantile=q, alpha=ALPHA,
                                     flag_bound=FLAG_BOUND, n_flag_features=3)
        m.fit(Xq, np.asarray(y_combined))
        robust_models[key] = m
        fp = [cols.index(fc) for fc in FLAG_COLS]
        log.info(
            "  Eğitildi | %s | q09_G=%.4f q01_G=%.4f",
            key, m.coef_[fp[0]],
            (robust_models[_meta_key(0.1)].coef_[fp[0]]
             if q == 0.9 else m.coef_[fp[0]]),
        )

    # Re-log all flag coefs clearly
    for q in META_QUANTILES:
        key = _meta_key(q)
        cols = _q_cols(q)
        fp = [cols.index(fc) for fc in FLAG_COLS]
        log.info(
            "  SA robust %s coefs: G=%.4f T=%.4f RH=%.4f",
            key,
            robust_models[key].coef_[fp[0]],
            robust_models[key].coef_[fp[1]],
            robust_models[key].coef_[fp[2]],
        )

    joblib.dump(robust_models, out_path)
    _elapsed("Robust meta", t0)
    return robust_models


# ── ADIM 5: STAGE-8 — 9 senaryo × rolling imputation recovery ───────────────

def step5_stage8(dataset: dict, base_models: dict, robust_models: dict,
                 force: bool = False) -> list[dict]:
    out_path = DATA_DIR / "stage8_results_sa.joblib"
    if _skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    log.info("=== ADIM 5: STAGE-8 robustness × rolling imputation recovery ===")

    X_test = dataset["X_test"].copy()
    y_arr  = np.asarray(dataset["y_test"], dtype=np.float64)
    n      = len(X_test)

    dl_mask = X_test["cos_zenith"].to_numpy(dtype=np.float64) > DAYLIGHT_THRESHOLD
    log.info("  X_test: %s  |  daylight: %d / %d", X_test.shape,
             dl_mask.sum(), len(dl_mask))

    # Sensor context for rolling recovery (imputed val tail)
    val_sensor_tail = dataset["val_sensor_tail"]
    avail_sensors   = [c for c in SENSOR_COLS if c in X_test.columns]

    def _corrupt_rolling_test(
        sensor: str, row_mask: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Test setinde sensor corrupt et → rolling imputation (val_tail context) →
        fiziksel öznitelikleri güncelle. (X_corrupted, flags_df) döndür.
        """
        X_c = X_test.copy()

        # Sensor serisini corrupt et
        corrupted = X_c[avail_sensors].copy()
        corrupted.loc[X_c.index[row_mask], sensor] = np.nan

        # Context ekleyerek rolling impute
        context_combined = pd.concat([val_sensor_tail, corrupted])
        filled_combined  = rolling_same_hour_imputation(
            context_combined, [sensor]
        )
        X_c[sensor] = filled_combined.loc[X_c.index, sensor].values

        # Fiziksel öznitelikleri güncelle
        if sensor in ("G", "T_amb"):
            X_c = _recompute_derived(X_c, DKASC_LOCATION)

        # Flags
        existing = {fc: np.zeros(n, dtype=int) for fc in FLAG_COLS}
        existing[SENSOR_TO_FLAG[sensor]] = row_mask.astype(int)
        flags_df = pd.DataFrame(existing, index=X_c.index)

        return X_c, flags_df

    def _predict(X_c: pd.DataFrame, flags_df: pd.DataFrame,
                 use_flags: bool) -> dict[str, np.ndarray]:
        bp   = {c: np.asarray(base_models[c].predict(X_c), dtype=np.float64)
                for c in META_COLS}
        Xm   = pd.DataFrame(bp, index=X_c.index)
        zf   = pd.DataFrame({c: np.zeros(n, dtype=int) for c in FLAG_COLS},
                             index=X_c.index)
        x12  = enrich_x_meta(Xm, flags_df if use_flags else zf)
        raw  = normalize_stacked_preds(predict_intervals(robust_models, x12))
        raw  = enforce_monotonicity(raw)
        raw  = enforce_monotonicity(apply_locally_scaled(raw, CQR_K))
        return raw

    def _crps_per_obs(y: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
        def pb(pred: np.ndarray, q: float) -> np.ndarray:
            r = y - pred
            return np.where(r >= 0, q * r, (q - 1) * r)
        return (pb(p["q01"], 0.1) + pb(p["q05"], 0.5) + pb(p["q09"], 0.9)) / 3.0

    # Senaryo tanımları (aynı seed sequence olarak main'le kıyaslanabilir)
    rng = np.random.default_rng(42)
    SCENARIOS: list[dict] = []

    for rate in (0.10, 0.20, 0.30, 0.50):
        mask = rng.random(n) < rate
        SCENARIOS.append({"name": f"random_G_{int(rate*100):02d}pct",
                          "label": f"Rnd G %{int(rate*100)}", "sensor": "G", "mask": mask})

    for hours in (1, 6, 24):
        bsteps = hours * STEPS_PER_HOUR
        bmask  = _make_burst_mask(n, bsteps, BURST_INTENSITY, rng)
        SCENARIOS.append({"name": f"burst_G_{hours}h",
                          "label": f"Burst G {hours}h", "sensor": "G", "mask": bmask})

    for sensor in ("T_amb", "RH"):
        mask = rng.random(n) < SENSOR_RATE
        SCENARIOS.append({"name": f"sensor_{sensor}_30pct",
                          "label": f"Rnd {sensor} %30", "sensor": sensor, "mask": mask})

    results: list[dict] = []
    for i, sc in enumerate(SCENARIOS):
        t_sc = time.time()
        log.info("[%d/%d] %s ...", i + 1, len(SCENARIOS), sc["name"])

        X_c, flags_df = _corrupt_rolling_test(sc["sensor"], sc["mask"])
        p_with = _predict(X_c, flags_df, use_flags=True)
        p_zero = _predict(X_c, flags_df, use_flags=False)

        # Daylight filtresi
        nr = len(p_with["q01"])
        m  = dl_mask[:nr]
        y_dl = y_arr[:nr][m]
        pw   = {k: v[m] for k, v in p_with.items()}
        pz   = {k: v[m] for k, v in p_zero.items()}

        m_w = evaluate_quantiles(y_dl, pw)
        m_z = evaluate_quantiles(y_dl, pz)

        crps_w  = _crps_per_obs(y_dl, pw)
        crps_z  = _crps_per_obs(y_dl, pz)
        dm      = diebold_mariano_test(crps_z, crps_w)

        delta_pct = (m_w["crps"] - m_z["crps"]) / m_z["crps"] * 100 \
                    if m_z["crps"] > 0 else float("nan")

        results.append({
            "name":       sc["name"],
            "label":      sc["label"],
            "corrupt_pct": float(sc["mask"].mean()) * 100,
            "crps_with":  m_w["crps"],
            "crps_zero":  m_z["crps"],
            "delta_pct":  delta_pct,
            "cov_with":   float(np.mean((y_dl >= pw["q01"]) & (y_dl <= pw["q09"]))),
            "cov_zero":   float(np.mean((y_dl >= pz["q01"]) & (y_dl <= pz["q09"]))),
            "mae_with":   m_w["mae"],
            "pb01_with":  m_w["pinball_q01"],
            "pb09_with":  m_w["pinball_q09"],
            "dm_stat":    dm["dm_stat"],
            "dm_p_raw":   dm["p_value"],
            "crps_w_obs": crps_w,
            "crps_z_obs": crps_z,
        })
        log.info("  %.1fs | CRPS_w=%.4f CRPS_z=%.4f Δ=%.2f%% DM_p=%.2e",
                 time.time() - t_sc, m_w["crps"], m_z["crps"], delta_pct, dm["p_value"])

    # Holm-Bonferroni
    p_raw   = np.array([r["dm_p_raw"] for r in results])
    n_tests = len(p_raw)
    order   = np.argsort(p_raw)
    p_adj   = np.ones(n_tests)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, p_raw[idx] * (n_tests - rank))
    for i in range(1, n_tests):
        p_adj[order[i]] = max(p_adj[order[i]], p_adj[order[i - 1]])
    for i, r in enumerate(results):
        r["dm_p_adj"]  = float(p_adj[i])
        r["significant"] = bool(p_adj[i] < 0.05)

    joblib.dump(results, out_path)
    _elapsed("STAGE-8 SA", t0)
    return results


# ── Çıktı: flag katsayı tablosu ──────────────────────────────────────────────

def print_flag_coefs(
    meta_sa: dict, robust_sa: dict, tag: str = "SA"
) -> None:
    print("\n" + "═" * 80)
    print(f"FLAG KATSAYILARı — Strategy A ({tag})")
    print("═" * 80)
    print(f"{'Model':<20}  {'q09_G':>9}  {'q09_T':>9}  {'q09_RH':>9}  "
          f"{'q01_G':>9}  {'q01_T':>9}  {'q01_RH':>9}")
    print("─" * 80)
    for label, mm in [("clean_meta", meta_sa), ("robust_meta", robust_sa)]:
        c09 = [mm["meta_q09"].coef_[_q_cols(0.9).index(fc)] for fc in FLAG_COLS]
        c01 = [mm["meta_q01"].coef_[_q_cols(0.1).index(fc)] for fc in FLAG_COLS]
        print(f"{label:<20}  {c09[0]:>9.4f}  {c09[1]:>9.4f}  {c09[2]:>9.4f}  "
              f"{c01[0]:>9.4f}  {c01[1]:>9.4f}  {c01[2]:>9.4f}")
    print("═" * 80)


# ── Çıktı: STAGE-8 master tablosu ────────────────────────────────────────────

def print_stage8_table(results: list[dict], title: str = "Strategy A") -> None:
    print("\n" + "═" * 110)
    print(f"STAGE-8 — {title}  (daylight, CQR k=2.0, rolling imputation)")
    print("═" * 110)
    print(f"{'Senaryo':<22}  {'Corrupt%':>8}  {'CRPS_w':>9}  {'CRPS_0':>9}  "
          f"{'ΔCRPS%':>8}  {'Cov_w':>7}  {'DM_p_adj':>10}  {'Anlam':>6}")
    print("─" * 110)
    for r in results:
        sig = "✓" if r["significant"] else " "
        print(f"{r['label']:<22}  {r['corrupt_pct']:>7.1f}%  "
              f"{r['crps_with']:>9.4f}  {r['crps_zero']:>9.4f}  "
              f"{r['delta_pct']:>+8.2f}%  {r['cov_with']:>7.4f}  "
              f"{r['dm_p_adj']:>10.2e}  {sig:>6}")
    print("═" * 110)

    sig_count  = sum(1 for r in results if r["significant"])
    mean_delta = float(np.mean([r["delta_pct"] for r in results]))
    neg_count  = sum(1 for r in results if r["delta_pct"] < 0)
    print(f"\nAnlamlı: {sig_count}/{len(results)}  |  "
          f"Ort. ΔCRPS: {mean_delta:+.2f}%  |  "
          f"Flags iyileştirdi: {neg_count}/{len(results)}")


# ── Görsel: CRPS bar karşılaştırma ──────────────────────────────────────────

def make_crps_bar(results: list[dict]) -> None:
    labels    = [r["label"]     for r in results]
    crps_with = [r["crps_with"] for r in results]
    crps_zero = [r["crps_zero"] for r in results]
    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, crps_with, width, label="with flags", color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, crps_zero, width, label="zero flags", color="#FF7043", alpha=0.85)

    for i, r in enumerate(results):
        if r["significant"]:
            ymax = max(crps_with[i], crps_zero[i])
            ax.text(i, ymax + 0.008, "*", ha="center", va="bottom", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("CRPS (↓ daha iyi)")
    ax.set_title("Strategy A — CRPS: with flags vs zero flags  (* p<0.05 Holm-Bonferroni)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "sa_crps_bar.png", dpi=150)
    fig.savefig(FIG_DIR / "sa_crps_bar.pdf")
    plt.close(fig)
    log.info("Kaydedildi: %s", FIG_DIR / "sa_crps_bar.png")


# ── Ana akış ───────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    t_total = time.time()
    log.info("=" * 70)
    log.info("STRATEGY A PIPELINE BAŞLADI  (rolling same-hour imputation)")
    log.info("force=%s", force)
    log.info("=" * 70)

    df      = step1_load_raw(force)
    dataset = step2_make_dataset_rolling(df, force)
    del df

    base_models, X_meta = step3_base_learners(dataset, force)
    meta_sa             = step4a_clean_meta(dataset, X_meta, force)
    robust_sa           = step4b_robust_meta(dataset, base_models, force)
    results             = step5_stage8(dataset, base_models, robust_sa, force)

    print_flag_coefs(meta_sa, robust_sa)
    print_stage8_table(results)
    make_crps_bar(results)

    log.info("=" * 70)
    log.info("PIPELINE TAMAMLANDI | Toplam: %.1f dk", (time.time() - t_total) / 60)
    log.info("Çıktılar: %s", EXPR_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy A — rolling imputation pipeline")
    parser.add_argument("--force", action="store_true", help="Mevcut dosyaları yenile")
    args = parser.parse_args()
    main(force=args.force)
