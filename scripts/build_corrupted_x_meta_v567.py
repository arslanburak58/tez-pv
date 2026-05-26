"""
v5 / v6 / v7 meta modelleri — tek paket.

v5: Burst-augmented (random %30 + burst %20 + clean %50), alpha=1.0, QuantileLinear
v6: v4 augmentation + QuantileLinearBounded (flag_bound=1.0)
v7: v5 augmentation + QuantileLinearBounded (flag_bound=1.0)

Master tablo: flag coef + smoke test (Rnd G/T/RH %30, STAGE-8 seed sequence).
"""

import logging
import math
import random

import joblib
import numpy as np
import pandas as pd
import scipy.stats

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

from models.base_learners import META_COLS
from models.meta_learner import (
    FLAG_COLS, QuantileLinear, QuantileLinearBounded,
    QUANTILES, _meta_key, _q_cols,
    enrich_x_meta, predict_intervals,
)
from models.baselines import evaluate_quantiles
from evaluation.comparison import enforce_monotonicity, normalize_stacked_preds
from evaluation.cqr import apply_locally_scaled

NOCT: float = 46.0
ALPHA: float = 1.0
FLAG_BOUND: float = 1.0
DAYLIGHT_THRESHOLD: float = 0.087
CQR_K: float = 2.0
STEPS_PER_HOUR: int = 12        # DKASC 5 dk/satır
SENSOR_TO_FLAG: dict[str, str] = {
    "G": "is_G_missing", "T_amb": "is_Tamb_missing", "RH": "is_RH_missing",
}
SENSORS: list[str] = ["G", "T_amb", "RH"]


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _tcell(X: pd.DataFrame) -> pd.DataFrame:
    if all(c in X.columns for c in ("T_cell", "T_amb", "G")):
        X = X.copy()
        X["T_cell"] = X["T_amb"] + X["G"] * (NOCT - 20.0) / 800.0
    return X


def _corrupt_single(X_val: pd.DataFrame, sensor: str,
                    row_mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tek sensör corrupt et, (X_corrupted, flags_df) döndür."""
    X_c = X_val.copy()
    idx = X_c.index[row_mask]
    X_c.loc[idx, sensor] = np.nan
    X_c[sensor] = X_c[sensor].ffill().bfill()
    if sensor in ("G", "T_amb"):
        X_c = _tcell(X_c)
    flags = pd.DataFrame({fc: np.zeros(len(X_c), dtype=int) for fc in FLAG_COLS},
                         index=X_c.index)
    flags[SENSOR_TO_FLAG[sensor]] = row_mask.astype(int)
    return X_c, flags


def _corrupt_to_x12(bm: dict, X_c: pd.DataFrame,
                    flags_df: pd.DataFrame) -> pd.DataFrame:
    preds = {c: np.asarray(bm[c].predict(X_c), dtype=np.float64) for c in META_COLS}
    Xm = pd.DataFrame(preds, index=X_c.index)
    return enrich_x_meta(Xm, flags_df)


def _make_burst_mask(n: int, burst_steps: int,
                     intensity: float, rng) -> np.ndarray:
    target = int(n * intensity)
    mask = np.zeros(n, dtype=bool)
    starts = list(range(0, n - burst_steps + 1, burst_steps))
    rng.shuffle(starts)
    covered = 0
    for s in starts:
        if covered >= target:
            break
        end = min(s + burst_steps, n)
        mask[s:end] = True
        covered += end - s
    return mask


def _train_all(X12: pd.DataFrame, y: pd.Series | np.ndarray,
               alpha: float, bounded: bool = False,
               flag_bound: float = 1.0) -> dict[str, QuantileLinear]:
    """3 meta-model eğit (standart veya bounded)."""
    models: dict[str, QuantileLinear] = {}
    for q in QUANTILES:
        key = _meta_key(q)
        cols = _q_cols(q)
        X_q = X12[cols].to_numpy()
        if bounded:
            model: QuantileLinear = QuantileLinearBounded(
                quantile=q, alpha=alpha, flag_bound=flag_bound, n_flag_features=3
            )
        else:
            model = QuantileLinear(quantile=q, alpha=alpha)
        model.fit(X_q, np.asarray(y))
        models[key] = model
        log.info("Eğitildi | key=%s | bounded=%s | q09_G=%.4f",
                 key, bounded,
                 model.coef_[[_q_cols(q).index("is_G_missing")]][0])
    return models


# ── 1. Yükle ──────────────────────────────────────────────────────────────────

log.info("Yükleniyor...")
ds  = joblib.load("data/processed/dataset.joblib")
bm  = joblib.load("data/processed/base_models.joblib")

xm_raw      = joblib.load("data/processed/x_meta.joblib")
xm_clean, _ = xm_raw if isinstance(xm_raw, tuple) else (xm_raw, None)

X_val   = ds["X_val"].copy()
y_val   = ds["y_val"]
X_train = ds["X_train"]
y_train = ds["y_train"]
n_val   = len(X_val)

log.info("X_val: %s  |  clean OOF: %s", X_val.shape, xm_clean.shape)

# Clean OOF
flags_clean   = X_train[FLAG_COLS].copy()
X_meta_clean  = enrich_x_meta(xm_clean, flags_clean)
y_meta_clean  = y_train.loc[X_meta_clean.index]

# ── 2. v4 corrupted_x_meta (mevcut strateji, re-build) ────────────────────────

log.info("v4 augmentation oluşturuluyor (random tek-sensör)...")
rng4 = np.random.default_rng(42)
rates4       = rng4.uniform(0.10, 0.50, n_val)
corrupt4_mask = rng4.random(n_val) < rates4
sensor4_idx  = rng4.integers(0, 3, n_val)

G_m4    = corrupt4_mask & (sensor4_idx == 0)
Tamb_m4 = corrupt4_mask & (sensor4_idx == 1)
RH_m4   = corrupt4_mask & (sensor4_idx == 2)

X_c4 = X_val.copy()
for sensor, mask in zip(SENSORS, [G_m4, Tamb_m4, RH_m4]):
    if sensor not in X_c4.columns:
        continue
    X_c4.loc[X_c4.index[mask], sensor] = np.nan
    X_c4[sensor] = X_c4[sensor].ffill().bfill()
    X_c4[SENSOR_TO_FLAG[sensor]] = mask.astype(int)
if "T_cell" in X_c4.columns:
    X_c4["T_cell"] = X_c4["T_amb"] + X_c4["G"] * (NOCT - 20.0) / 800.0

preds4 = {c: np.asarray(bm[c].predict(X_c4), dtype=np.float64) for c in META_COLS}
Xmc4   = enrich_x_meta(pd.DataFrame(preds4, index=X_c4.index),
                        X_c4[[FLAG_COLS[0], FLAG_COLS[1], FLAG_COLS[2]]].copy())
y4 = y_val.loc[Xmc4.index]

X_combined_v4 = pd.concat([X_meta_clean, Xmc4], ignore_index=True)
y_combined_v4  = pd.concat([y_meta_clean,  y4],  ignore_index=True)
log.info("v4 combined: %s", X_combined_v4.shape)

# ── 3. v5 corrupted_x_meta (random %30 + burst %20) ──────────────────────────

log.info("v5 augmentation oluşturuluyor (random + burst)...")
rng5 = np.random.default_rng(99)  # farklı seed, v4'ten bağımsız

# --- Random batch: %30 satır ---
rates5       = rng5.uniform(0.10, 0.50, n_val)
corrupt5_mask = rng5.random(n_val) < rates5
sensor5_idx  = rng5.integers(0, 3, n_val)

G_m5    = corrupt5_mask & (sensor5_idx == 0)
Tamb_m5 = corrupt5_mask & (sensor5_idx == 1)
RH_m5   = corrupt5_mask & (sensor5_idx == 2)

X_c5r = X_val.copy()
for sensor, mask in zip(SENSORS, [G_m5, Tamb_m5, RH_m5]):
    if sensor not in X_c5r.columns:
        continue
    X_c5r.loc[X_c5r.index[mask], sensor] = np.nan
    X_c5r[sensor] = X_c5r[sensor].ffill().bfill()
    X_c5r[SENSOR_TO_FLAG[sensor]] = mask.astype(int)
if "T_cell" in X_c5r.columns:
    X_c5r["T_cell"] = X_c5r["T_amb"] + X_c5r["G"] * (NOCT - 20.0) / 800.0

preds5r = {c: np.asarray(bm[c].predict(X_c5r), dtype=np.float64) for c in META_COLS}
Xmc5r   = enrich_x_meta(pd.DataFrame(preds5r, index=X_c5r.index),
                          X_c5r[[FLAG_COLS[0], FLAG_COLS[1], FLAG_COLS[2]]].copy())

# --- Burst batch: 5 scenario, her biri ~%4 satır ---
BURST_CONFIGS = [
    ("G",    1,  0.04),  # G 1h burst, %4 satır
    ("G",    6,  0.04),  # G 6h burst, %4 satır
    ("G",   24,  0.04),  # G 24h burst, %4 satır
    ("T_amb", 6, 0.04),  # T_amb 6h burst
    ("RH",   1,  0.04),  # RH 1h burst
]

burst_batches_x12: list[pd.DataFrame] = []
burst_batches_y:   list[pd.Series]    = []

for sensor_b, dur_h, intensity_b in BURST_CONFIGS:
    bsteps = dur_h * STEPS_PER_HOUR
    bmask  = _make_burst_mask(n_val, bsteps, intensity_b, rng5)
    if bmask.sum() == 0:
        continue
    X_cb, flags_b = _corrupt_single(X_val, sensor_b, bmask)
    Xm12b = _corrupt_to_x12(bm, X_cb, flags_b)
    burst_batches_x12.append(Xm12b)
    burst_batches_y.append(y_val.loc[Xm12b.index])
    log.info("  Burst %s %dh: %d satır corrupt", sensor_b, dur_h, bmask.sum())

X_combined_v5 = pd.concat(
    [X_meta_clean, Xmc5r] + burst_batches_x12,
    ignore_index=True,
)
y_combined_v5 = pd.concat(
    [y_meta_clean, y_val.loc[Xmc5r.index]] + burst_batches_y,
    ignore_index=True,
)
log.info("v5 combined: %s", X_combined_v5.shape)

# ── 4. Modelleri eğit (v5, v6, v7) ───────────────────────────────────────────

log.info("=== v5: v5 aug, standart QuantileLinear ===")
models_v5 = _train_all(X_combined_v5, y_combined_v5, alpha=ALPHA, bounded=False)

log.info("=== v6: v4 aug, QuantileLinearBounded (flag_bound=%.1f) ===", FLAG_BOUND)
models_v6 = _train_all(X_combined_v4, y_combined_v4, alpha=ALPHA, bounded=True, flag_bound=FLAG_BOUND)

log.info("=== v7: v5 aug, QuantileLinearBounded (flag_bound=%.1f) ===", FLAG_BOUND)
models_v7 = _train_all(X_combined_v5, y_combined_v5, alpha=ALPHA, bounded=True, flag_bound=FLAG_BOUND)

# Kaydet
joblib.dump(models_v5, "data/processed/meta_models_robust_v5.joblib")
joblib.dump(models_v6, "data/processed/meta_models_robust_v6.joblib")
joblib.dump(models_v7, "data/processed/meta_models_robust_v7.joblib")
log.info("v5/v6/v7 kaydedildi.")

# ── 5. Flag katsayı tablosu ───────────────────────────────────────────────────

models_v2 = joblib.load("data/processed/meta_models_robust_v2.joblib")
models_v4 = joblib.load("data/processed/meta_models_robust_v4.joblib")

all_models = {"v2": models_v2, "v4": models_v4,
              "v5": models_v5, "v6": models_v6, "v7": models_v7}

print("\n" + "═"*80)
print("FLAG COEFFİCİENTLER — q09 ve q01")
print("═"*80)
print(f"{'Meta':<6}  {'q09_G':>9}  {'q09_T':>9}  {'q09_RH':>9}  {'q09_L2':>8}"
      f"  {'q01_G':>9}  {'q01_T':>9}  {'q01_RH':>9}  {'q01_L2':>8}")
print("─"*80)

for tag, mm in all_models.items():
    fp09 = [_q_cols(0.9).index(fc) for fc in FLAG_COLS]
    fp01 = [_q_cols(0.1).index(fc) for fc in FLAG_COLS]
    c09  = mm["meta_q09"].coef_[fp09]
    c01  = mm["meta_q01"].coef_[fp01]
    print(f"{tag:<6}  {c09[0]:>9.4f}  {c09[1]:>9.4f}  {c09[2]:>9.4f}  "
          f"{np.linalg.norm(c09):>8.4f}  "
          f"{c01[0]:>9.4f}  {c01[1]:>9.4f}  {c01[2]:>9.4f}  "
          f"{np.linalg.norm(c01):>8.4f}")

# ── 6. Smoke test — STAGE-8 seed sequence ─────────────────────────────────────

X_test = ds["X_test"].copy()
y_arr  = np.asarray(ds["y_test"], dtype=np.float64)
dl     = X_test["cos_zenith"].to_numpy() > DAYLIGHT_THRESHOLD
n_test = len(X_test)

# STAGE-8 rng sequence: Rnd G %10, %20, %30(hedef), %50, Burst 1h, 6h, 24h, T_amb, RH
rng_s8 = np.random.default_rng(42)
_ = rng_s8.random(n_test) < 0.10
_ = rng_s8.random(n_test) < 0.20
mask_G30   = rng_s8.random(n_test) < 0.30  # 3. çağrı
_ = rng_s8.random(n_test) < 0.50
for h in (1, 6, 24):
    bs = h * STEPS_PER_HOUR
    _ = _make_burst_mask(n_test, bs, 0.30, rng_s8)
mask_T30 = rng_s8.random(n_test) < 0.30
mask_RH30 = rng_s8.random(n_test) < 0.30

SMOKE_SCENARIOS = [
    ("Rnd G  %30", "G",     mask_G30),
    ("Rnd T  %30", "T_amb", mask_T30),
    ("Rnd RH %30", "RH",    mask_RH30),
]


def _predict_score(mm: dict, X_c: pd.DataFrame, flags_df: pd.DataFrame,
                   use_flags: bool) -> float:
    bp = {c: np.asarray(bm[c].predict(X_c), dtype=np.float64) for c in META_COLS}
    Xm = pd.DataFrame(bp, index=X_c.index)
    zf = pd.DataFrame({fc: np.zeros(n_test, dtype=int) for fc in FLAG_COLS}, index=X_c.index)
    x12 = enrich_x_meta(Xm, flags_df if use_flags else zf)
    raw = normalize_stacked_preds(predict_intervals(mm, x12))
    raw = enforce_monotonicity(raw)
    raw = enforce_monotonicity(apply_locally_scaled(raw, CQR_K))
    nr = len(raw["q01"]); m = dl[:nr]
    yd = y_arr[:nr][m]; pd_ = {k: v[m] for k, v in raw.items()}
    return evaluate_quantiles(yd, pd_)["crps"]


print("\n" + "═"*90)
print("SMOKE TEST — STAGE-8 seed sequence, daylight + CQR(k=2)")
print("(+% = flags bozdu  |  -% = flags iyileştirdi)")
print("═"*90)
print(f"\n{'Meta':<6}  ", end="")
for sc_label, _, _ in SMOKE_SCENARIOS:
    print(f"  {sc_label:<14}", end="")
print()
print(f"{'':6}  ", end="")
for _ in SMOKE_SCENARIOS:
    print(f"  {'CRPS_w':>7} {'CRPS_0':>7} {'Δ%':>7}", end="")
print()
print("─"*90)

best_row: dict[str, float] = {}

for tag, mm in all_models.items():
    row_vals: list[float] = []
    print(f"{tag:<6}  ", end="")
    for sc_label, sensor, row_mask in SMOKE_SCENARIOS:
        X_c, flags_df = _corrupt_single(X_test, sensor, row_mask)
        cw = _predict_score(mm, X_c, flags_df, True)
        c0 = _predict_score(mm, X_c, flags_df, False)
        dp = (cw - c0) / c0 * 100
        row_vals.append(dp)
        direction = "✓" if dp < 0 else "✗"
        print(f"  {cw:>7.4f} {c0:>7.4f} {dp:>+6.1f}%{direction}", end="")
    print(f"  avg={np.mean(row_vals):>+5.1f}%")
    best_row[tag] = float(np.mean(row_vals))

print("─"*90)
best_tag = min(best_row, key=lambda k: best_row[k])
print(f"\n>>> En iyi varyant: {best_tag} (ortalama Δ={best_row[best_tag]:+.2f}%)")
print("(Beklenen: negatif = flags ortalamada CRPS iyileştirdi)")
print("═"*90)
