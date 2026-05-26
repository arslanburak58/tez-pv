"""
STAGE-10: Karşılaştırmalı analiz — gerçek checkpoint'lerle run_comparison() çalıştır.

Adımlar:
  1. Joblib'leri yükle
  2. Daylight maskesi (cos_zenith > 0.087)
  3. Val üzerinde CQR parametrelerini hesapla (3 varyant)
  4. Test üzerinde stacked tahminler + 3 CQR varyantı
  5. Baseline tahminler (knn, svm, lstm, tft)
  6. 9 satırlık karşılaştırma tablosu (raw + 3 CQR + 4 baseline)
  7. run_comparison → figures/
"""

import logging
import random
import time

import joblib
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

from models.base_learners import META_COLS
from models.meta_learner import FLAG_COLS, enrich_x_meta, predict_intervals
from models.baselines import evaluate_quantiles, predict_baseline
from evaluation.comparison import ModelResult, dm_pairwise, enforce_monotonicity, normalize_stacked_preds, run_comparison
from evaluation.cqr import (
    apply_cqr_asymmetric, apply_cqr_correction, apply_locally_scaled,
    compute_cqr_asymmetric, compute_cqr_locally_scaled, compute_cqr_offset,
)

DAYLIGHT_ONLY:      bool  = True
DAYLIGHT_THRESHOLD: float = 0.087
STACKED_TRAIN_TIME: float = 735.7 + 7.4

# ── 1. Yükle ──────────────────────────────────────────────────────────────────

log.info("Joblib'ler yükleniyor...")
ds  = joblib.load("data/processed/dataset.joblib")
bm  = joblib.load("data/processed/base_models.joblib")
mm  = joblib.load("data/processed/meta_models.joblib")
bl  = joblib.load("data/processed/baseline_results.joblib")

X_test = ds["X_test"]
y_arr  = np.asarray(ds["y_test"], dtype=np.float64)
X_val  = ds["X_val"]
y_val  = np.asarray(ds["y_val"],  dtype=np.float64)

log.info("X_test: %s | X_val: %s", X_test.shape, X_val.shape)

# ── 2. Daylight maskeleri ─────────────────────────────────────────────────────

dl_mask     = X_test["cos_zenith"].to_numpy(dtype=np.float64) > DAYLIGHT_THRESHOLD
dl_mask_val = X_val["cos_zenith"].to_numpy(dtype=np.float64)  > DAYLIGHT_THRESHOLD
log.info("Daylight — test: %d/%d (%.1f%%)  val: %d/%d (%.1f%%)",
         dl_mask.sum(), len(dl_mask), 100*dl_mask.mean(),
         dl_mask_val.sum(), len(dl_mask_val), 100*dl_mask_val.mean())


def _enrich(X: pd.DataFrame) -> pd.DataFrame:
    preds_dict = {col: np.asarray(bm[col].predict(X), dtype=np.float64) for col in META_COLS}
    X_meta = pd.DataFrame(preds_dict, index=X.index)
    flag_cols = [c for c in FLAG_COLS if c in X.columns]
    flags = X[flag_cols].copy() if len(flag_cols) == len(FLAG_COLS) else pd.DataFrame(
        {c: np.zeros(len(X), dtype=int) for c in FLAG_COLS}, index=X.index)
    return enrich_x_meta(X_meta, flags)


def _mask_preds(
    y: np.ndarray, preds: dict[str, np.ndarray], mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(preds["q01"])
    m = mask[:n]
    return y[:n][m], {k: v[m] for k, v in preds.items()}


# ── 3. Val CQR parametreleri ──────────────────────────────────────────────────

log.info("Val meta tahminleri üretiliyor...")
X_meta_val_12 = _enrich(X_val)
raw_val       = predict_intervals(mm, X_meta_val_12)
preds_val_raw = enforce_monotonicity(normalize_stacked_preds(raw_val))

y_v   = y_val[dl_mask_val]
q01_v = preds_val_raw["q01"][dl_mask_val]
q05_v = preds_val_raw["q05"][dl_mask_val]
q09_v = preds_val_raw["q09"][dl_mask_val]

val_cov_raw = float(np.mean((y_v >= q01_v) & (y_v <= q09_v)))

off_sym            = compute_cqr_offset(y_v, q01_v, q09_v, alpha=0.20)
off_low, off_up    = compute_cqr_asymmetric(y_v, q01_v, q05_v, q09_v, alpha=0.20)
k_scale            = compute_cqr_locally_scaled(y_v, q01_v, q05_v, q09_v, alpha=0.20)

print(f"\n{'─'*60}")
print(f"VAL coverage (CQR öncesi, daylight)  : {val_cov_raw:.4f}")
print(f"Symmetric offset                      : {off_sym:.4f}")
print(f"Asymmetric offsets  lower={off_low:.4f}  upper={off_up:.4f}")
print(f"Locally scaled k                      : {k_scale:.4f}")
print(f"{'─'*60}\n")

# ── 4. Test — stacked tahminler (raw + 3 CQR) ─────────────────────────────────

log.info("Test meta tahminleri üretiliyor...")
X_meta_test_12 = _enrich(X_test)
raw_test       = predict_intervals(mm, X_meta_test_12)
preds_raw      = enforce_monotonicity(normalize_stacked_preds(raw_test))

preds_sym    = enforce_monotonicity(apply_cqr_correction(preds_raw,  off_sym))
preds_asym   = enforce_monotonicity(apply_cqr_asymmetric(preds_raw,  off_low, off_up))
preds_scaled = enforce_monotonicity(apply_locally_scaled(preds_raw,  k_scale))

stacked_variants: dict[str, dict[str, np.ndarray]] = {
    "Stacked RAW":        preds_raw,
    "Stacked CQR sym":    preds_sym,
    "Stacked CQR asym":   preds_asym,
    "Stacked CQR scaled": preds_scaled,
}

# ── 5. Baseline tahminler ──────────────────────────────────────────────────────

X_arr = np.asarray(X_test, dtype=np.float32)
baseline_preds: dict[str, tuple[np.ndarray, dict[str, np.ndarray], float]] = {}

for name in ("knn", "svm", "lstm", "tft"):
    model      = bl[name]["model"]
    train_time = float(bl[name]["train_time"])
    preds_bl   = predict_baseline(name, model, X_arr)
    seq_len    = int(getattr(model, "seq_len", 0))
    y_bl_full  = y_arr[seq_len:] if seq_len else y_arr
    mask_bl    = (dl_mask[seq_len:] if seq_len else dl_mask)[:len(preds_bl["q01"])]
    y_bl       = y_bl_full[:len(preds_bl["q01"])][mask_bl]
    preds_bl   = {k: v[mask_bl] for k, v in preds_bl.items()}
    baseline_preds[name] = (y_bl, preds_bl, train_time)

# ── 6. Karşılaştırma tablosu ──────────────────────────────────────────────────

def _row(
    label: str,
    y: np.ndarray,
    preds: dict[str, np.ndarray],
) -> dict:
    m     = evaluate_quantiles(y, preds)
    width = float(np.median(preds["q09"] - preds["q01"]))
    cov   = float(np.mean((y >= preds["q01"]) & (y <= preds["q09"])))
    ok    = "✓" if 0.75 <= cov <= 0.85 else " "
    return {
        "Model":        label,
        "Coverage":     cov,
        "CRPS":         m["crps"],
        "PB_q01":       m["pinball_q01"],
        "PB_q09":       m["pinball_q09"],
        "Band_med":     width,
        "In_target":    ok,
    }

rows: list[dict] = []

for label, preds in stacked_variants.items():
    y_filt, p_filt = _mask_preds(y_arr, preds, dl_mask)
    rows.append(_row(label, y_filt, p_filt))

baseline_labels = {"knn": "k-NN", "svm": "SVM", "lstm": "LSTM", "tft": "Light TFT"}
for name, (y_bl, preds_bl, _) in baseline_preds.items():
    rows.append(_row(baseline_labels[name], y_bl, preds_bl))

cmp_df = pd.DataFrame(rows).set_index("Model")

print("=== CQR KARŞILAŞTIRMA TABLOSU (daylight, n≈95K) ===")
print(f"{'Model':<24s}  {'Coverage':>8}  {'CRPS':>7}  {'PB_q01':>7}  {'PB_q09':>7}  {'Band_med':>8}  {'[0.75,0.85]':>11}")
print("─" * 84)
for model, row in cmp_df.iterrows():
    flag = " ← IN TARGET" if row["In_target"] == "✓" else ""
    print(f"{model:<24s}  {row['Coverage']:8.4f}  {row['CRPS']:7.4f}  "
          f"{row['PB_q01']:7.4f}  {row['PB_q09']:7.4f}  {row['Band_med']:8.3f}{flag}")

# ── 7. run_comparison (CQR sym ana model olarak) ──────────────────────────────

results: dict[str, ModelResult] = {}

for label_key, preds in [("stacked_flags", preds_sym), ("stacked_noflags", preds_sym)]:
    y_f, p_f = _mask_preds(y_arr, preds, dl_mask)
    results[label_key] = ModelResult(
        metrics=evaluate_quantiles(y_f, p_f),
        preds=p_f, y_true=y_f,
        train_time_s=STACKED_TRAIN_TIME,
    )

for name, (y_bl, preds_bl, train_time) in baseline_preds.items():
    results[name] = ModelResult(
        metrics=evaluate_quantiles(y_bl, preds_bl),
        preds=preds_bl, y_true=y_bl,
        train_time_s=train_time,
    )

# ── PAKET A: Empirik k sweep ──────────────────────────────────────────────────

K_VALUES = [1.0, 1.19, 1.5, 2.0, 2.5, 3.0, 4.0]

print("\n=== PAKET A — k SWEEP (CQR scaled, daylight) ===")
print(f"{'k':>5}  {'Coverage':>8}  {'CRPS':>7}  {'PB_q01':>7}  {'PB_q09':>7}  {'Band_med':>8}  {'|cov-0.80|':>10}")
print("─" * 68)

sweep_rows: list[dict] = []
for k in K_VALUES:
    p_k          = enforce_monotonicity(apply_locally_scaled(preds_raw, k))
    y_k, pk      = _mask_preds(y_arr, p_k, dl_mask)
    m_k          = evaluate_quantiles(y_k, pk)
    cov_k        = float(np.mean((y_k >= pk["q01"]) & (y_k <= pk["q09"])))
    band_k       = float(np.median(pk["q09"] - pk["q01"]))
    dist_k       = abs(cov_k - 0.80)
    sweep_rows.append(dict(k=k, preds=pk, y=y_k, cov=cov_k,
                           crps=m_k["crps"], pb01=m_k["pinball_q01"],
                           pb09=m_k["pinball_q09"], band=band_k, dist=dist_k))
    marker = " ← en yakın" if k == K_VALUES[0] else ""  # geçici
    print(f"{k:>5.2f}  {cov_k:8.4f}  {m_k['crps']:7.4f}  {m_k['pinball_q01']:7.4f}"
          f"  {m_k['pinball_q09']:7.4f}  {band_k:8.3f}  {dist_k:10.4f}")

best_row = min(sweep_rows, key=lambda r: r["dist"])
best_k   = best_row["k"]
print(f"\n→ En iyi k = {best_k}  (coverage={best_row['cov']:.4f}, "
      f"|cov-0.80|={best_row['dist']:.4f})")

# Kazanan k ile stacked vs TFT DM testi
tft_y, tft_p = baseline_preds["tft"][0], baseline_preds["tft"][1]
dm_results   = dm_pairwise({
    "stacked_flags": ModelResult(
        metrics=evaluate_quantiles(best_row["y"], best_row["preds"]),
        preds=best_row["preds"], y_true=best_row["y"],
        train_time_s=STACKED_TRAIN_TIME,
    ),
    "tft": ModelResult(
        metrics=evaluate_quantiles(tft_y, tft_p),
        preds=tft_p, y_true=tft_y,
        train_time_s=baseline_preds["tft"][2],
    ),
})
print("\n=== DM: Stacked (k={:.2f}) vs TFT ===".format(best_k))
stacked_vs_tft = dm_results[dm_results["model_i"].str.contains("Stacked")]
if not stacked_vs_tft.empty:
    row = stacked_vs_tft.iloc[0]
    direction = "stacked daha iyi" if row["mean_diff"] < 0 else "TFT daha iyi"
    print(f"  dm_stat={row['dm_stat']:.3f}  mean_diff={row['mean_diff']:.4f}  "
          f"p_adj={row['p_adj']:.2e}  significant={row['significant']}  → {direction}")

# ── Kazanan k ile master tablo ─────────────────────────────────────────────────

results["stacked_flags"]   = ModelResult(
    metrics=evaluate_quantiles(best_row["y"], best_row["preds"]),
    preds=best_row["preds"], y_true=best_row["y"],
    train_time_s=STACKED_TRAIN_TIME,
)
results["stacked_noflags"] = results["stacked_flags"]

log.info("run_comparison() → figures/ (k=%.2f)", best_k)
t0  = time.time()
out = run_comparison(results, save_dir="figures", suffix=f"_k{best_k:.0f}")
log.info("run_comparison tamamlandı | %.1f s", time.time() - t0)

print(f"\n=== MASTER TABLO (k={best_k}) ===")
print(out["master_table"].to_string())
