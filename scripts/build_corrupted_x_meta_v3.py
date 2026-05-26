"""
v3: Daylight-only corruption-aware meta training.

Fark v2'den:
- X_val üzerinde cos_zenith > 0.087 filtresi uygulanır → sadece gündüz satırlar
- Clean OOF (X_train) üzerinde de daylight filtresi
- Her şey aynı (seed=42, 3 bağımsız Bernoulli(0.30), alpha=0.5)

Hipotez: Gece satırlar meta eğitimini kirletiyorsa (trivial flag=0 → model gece
dinamiklerini öğreniyor), daylight-only eğitim flag katsayılarını düzeltir.

Smoke test: Rnd G %30, seed=42.
"""

import logging
import random

import joblib
import numpy as np
import pandas as pd
import scipy.stats

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

from models.base_learners import META_COLS, ALGOS, QUANTILES
from models.meta_learner import FLAG_COLS, enrich_x_meta, predict_intervals, train_all_meta_learners
from models.baselines import evaluate_quantiles
from evaluation.comparison import enforce_monotonicity, normalize_stacked_preds
from evaluation.cqr import apply_locally_scaled
from models.meta_learner import _q_cols

NOCT: float = 46.0
BERNOULLI_P: float = 0.30
ALPHA: float = 0.5
DAYLIGHT_THRESHOLD: float = 0.087
CQR_K: float = 2.0
OUT_PATH: str = "data/processed/meta_models_robust_v3.joblib"

SENSOR_TO_FLAG: dict[str, str] = {
    "G":     "is_G_missing",
    "T_amb": "is_Tamb_missing",
    "RH":    "is_RH_missing",
}


def _compute_cell_temp(T_amb: pd.Series, G: pd.Series) -> pd.Series:
    return (T_amb + G * (NOCT - 20.0) / 800.0).rename("T_cell")


# ── 1. Yükle ──────────────────────────────────────────────────────────────────

log.info("Yükleniyor...")
ds  = joblib.load("data/processed/dataset.joblib")
bm  = joblib.load("data/processed/base_models.joblib")

xm_raw      = joblib.load("data/processed/x_meta.joblib")
xm_clean, _ = xm_raw if isinstance(xm_raw, tuple) else (xm_raw, None)

X_val  = ds["X_val"].copy()
y_val  = ds["y_val"]
X_train = ds["X_train"]
y_train = ds["y_train"]

# ── 2. Daylight filtresi ──────────────────────────────────────────────────────

dl_val   = X_val["cos_zenith"].to_numpy() > DAYLIGHT_THRESHOLD
dl_train = X_train["cos_zenith"].to_numpy() > DAYLIGHT_THRESHOLD

X_val_dl = X_val[dl_val]
y_val_dl = y_val[dl_val]

log.info("X_val daylight: %d/%d (%.1f%%)", dl_val.sum(), len(X_val), 100*dl_val.mean())
log.info("X_train daylight: %d/%d (%.1f%%)", dl_train.sum(), len(X_train), 100*dl_train.mean())

# OOF x_meta: X_train index ile hizalı → daylight satırları seç
X_train_dl_idx = X_train.index[dl_train]
xm_clean_dl = xm_clean.loc[xm_clean.index.intersection(X_train_dl_idx)]
y_meta_clean_dl = y_train.loc[xm_clean_dl.index]
log.info("Clean OOF daylight: %d rows", len(xm_clean_dl))

# ── 3. Corrupted X_val_dl ─────────────────────────────────────────────────────

rng = np.random.default_rng(42)
n   = len(X_val_dl)

X_corrupt = X_val_dl.copy()

sensor_masks: dict[str, np.ndarray] = {}
for sensor in SENSOR_TO_FLAG:
    if sensor not in X_corrupt.columns:
        log.warning("%s sütunu yok — atlandı", sensor)
        continue
    mask = rng.random(n) < BERNOULLI_P
    sensor_masks[sensor] = mask
    log.info("  %s: %d/%d (%.1f%%) corrupt", sensor, mask.sum(), n, 100*mask.mean())

for sensor, mask in sensor_masks.items():
    X_corrupt.loc[X_corrupt.index[mask], sensor] = np.nan
    X_corrupt[sensor] = X_corrupt[sensor].ffill().bfill()
    X_corrupt[SENSOR_TO_FLAG[sensor]] = mask.astype(int)

if ("G" in sensor_masks or "T_amb" in sensor_masks) and "T_cell" in X_corrupt.columns:
    X_corrupt["T_cell"] = _compute_cell_temp(X_corrupt["T_amb"], X_corrupt["G"])
    log.info("T_cell yeniden hesaplandı")

flag_sum = sum(sensor_masks[s].astype(int) for s in sensor_masks)
log.info("En az 1 sensör corrupt: %.1f%%  |  2+ sensör: %.1f%%",
         100*(flag_sum >= 1).mean(), 100*(flag_sum >= 2).mean())

# ── 4. Corrupted base predictions ────────────────────────────────────────────

log.info("9 base model corrupted X_val_dl üzerinde tahmin yapıyor...")
corrupted_preds: dict[str, np.ndarray] = {}
for col in META_COLS:
    corrupted_preds[col] = np.asarray(bm[col].predict(X_corrupt), dtype=np.float64)

X_meta_corrupt_df = pd.DataFrame(corrupted_preds, index=X_corrupt.index)
flags_corrupt = X_corrupt[[FLAG_COLS[0], FLAG_COLS[1], FLAG_COLS[2]]].copy()
X_meta_corrupt_12 = enrich_x_meta(X_meta_corrupt_df, flags_corrupt)

# ── 5. Clean OOF daylight ─────────────────────────────────────────────────────

flags_clean_dl = X_train.loc[xm_clean_dl.index][FLAG_COLS].copy()
X_meta_clean_12 = enrich_x_meta(xm_clean_dl, flags_clean_dl)

# ── 6. Concatenate ────────────────────────────────────────────────────────────

y_corrupt_dl = y_val_dl.loc[X_meta_corrupt_12.index]

X_combined = pd.concat([X_meta_clean_12, X_meta_corrupt_12], ignore_index=True)
y_combined  = pd.concat([y_meta_clean_dl, y_corrupt_dl],   ignore_index=True)

log.info("Birleşik (daylight only): %s  (clean=%d + corrupt=%d)",
         X_combined.shape, len(X_meta_clean_12), len(X_meta_corrupt_12))

for fc in FLAG_COLS:
    log.info("  %s aktif: %.3f", fc, float(X_combined[fc].mean()))

# ── 7. QuantileLinear v3 ──────────────────────────────────────────────────────

log.info("QuantileLinear v3 × 3 eğitiliyor (alpha=%.2f, daylight-only)...", ALPHA)
models_v3 = train_all_meta_learners(X_combined, y_combined, alpha=ALPHA, checkpoint_dir=None)
log.info("Eğitim tamamlandı.")

# ── 8. Flag katsayı raporu ───────────────────────────────────────────────────

print("\n=== v3 FLAG COEFFİCİENTLER (daylight-only) ===")
print(f"{'Model':<12}  {'is_G_miss':>10}  {'is_Tamb_miss':>12}  {'is_RH_miss':>10}  {'L2 norm':>8}")
print("─" * 60)
for key, model in models_v3.items():
    q        = int(key[-2:]) / 10.0
    cols     = _q_cols(q)
    flag_pos = [cols.index(fc) for fc in FLAG_COLS]
    fc_vals  = model.coef_[flag_pos]
    l2_norm  = float(np.linalg.norm(fc_vals))
    print(f"{key:<12}  {fc_vals[0]:>10.6f}  {fc_vals[1]:>12.6f}  {fc_vals[2]:>10.6f}  {l2_norm:>8.4f}")

# v2 ile karşılaştır
models_v2 = joblib.load("data/processed/meta_models_robust_v2.joblib")
print("\n=== v2 vs v3 FLAG COEF L2 NORM ===")
print(f"{'Model':<12}  {'v2 norm':>10}  {'v3 norm':>10}  {'fark':>10}")
print("─" * 46)
for key in ("meta_q01", "meta_q05", "meta_q09"):
    q    = int(key[-2:]) / 10.0
    cols = _q_cols(q)
    fp   = [cols.index(fc) for fc in FLAG_COLS]
    n2   = float(np.linalg.norm(models_v2[key].coef_[fp]))
    n3   = float(np.linalg.norm(models_v3[key].coef_[fp]))
    print(f"{key:<12}  {n2:>10.6f}  {n3:>10.6f}  {n3-n2:>+10.6f}")

# ── 9. Smoke test (Rnd G %30, seed=42, X_test) ───────────────────────────────

log.info("Smoke test: Rnd G %30, seed=42...")

X_test = ds["X_test"].copy()
y_arr  = np.asarray(ds["y_test"], dtype=np.float64)
dl_test = X_test["cos_zenith"].to_numpy() > DAYLIGHT_THRESHOLD

rng2 = np.random.default_rng(42)
miss_mask = rng2.random(len(X_test)) < 0.30
X_smoke = X_test.copy()
X_smoke.loc[X_smoke.index[miss_mask], "G"] = np.nan
X_smoke["G"] = X_smoke["G"].ffill().bfill()
if "T_cell" in X_smoke.columns:
    X_smoke["T_cell"] = _compute_cell_temp(X_smoke["T_amb"], X_smoke["G"])
X_smoke["is_G_missing"] = miss_mask.astype(int)

smoke_preds: dict[str, np.ndarray] = {}
for col in META_COLS:
    smoke_preds[col] = np.asarray(bm[col].predict(X_smoke), dtype=np.float64)

X_meta_smoke = pd.DataFrame(smoke_preds, index=X_smoke.index)
flags_real = X_smoke[FLAG_COLS].copy()
flags_zero = pd.DataFrame({c: np.zeros(len(X_smoke), dtype=int) for c in FLAG_COLS},
                           index=X_smoke.index)

X_meta_real = enrich_x_meta(X_meta_smoke, flags_real)
X_meta_zero = enrich_x_meta(X_meta_smoke, flags_zero)


def _preds_dl(models: dict, X_12: pd.DataFrame, y: np.ndarray,
              dl: np.ndarray, k: float) -> tuple[np.ndarray, dict]:
    raw = normalize_stacked_preds(predict_intervals(models, X_12))
    raw = enforce_monotonicity(raw)
    raw = enforce_monotonicity(apply_locally_scaled(raw, k))
    n   = len(raw["q01"])
    m   = dl[:n]
    return y[:n][m], {kk: vv[m] for kk, vv in raw.items()}


# v3 smoke
y_dl_w,  p_with    = _preds_dl(models_v3, X_meta_real, y_arr, dl_test, CQR_K)
y_dl_wo, p_without = _preds_dl(models_v3, X_meta_zero, y_arr, dl_test, CQR_K)

m_with    = evaluate_quantiles(y_dl_w,  p_with)
m_without = evaluate_quantiles(y_dl_wo, p_without)
delta_crps = m_with["crps"] - m_without["crps"]
delta_pct  = delta_crps / m_without["crps"] * 100 if m_without["crps"] > 0 else float("nan")

# v2 smoke (same data)
y_dl_w2,  p_with2    = _preds_dl(models_v2, X_meta_real, y_arr, dl_test, CQR_K)
y_dl_wo2, p_without2 = _preds_dl(models_v2, X_meta_zero, y_arr, dl_test, CQR_K)

m_with2    = evaluate_quantiles(y_dl_w2,  p_with2)
m_without2 = evaluate_quantiles(y_dl_wo2, p_without2)
delta2_crps = m_with2["crps"] - m_without2["crps"]
delta2_pct  = delta2_crps / m_without2["crps"] * 100 if m_without2["crps"] > 0 else float("nan")

# DM test v3
def _dm(y: np.ndarray, pi: dict, pj: dict) -> tuple[float, float]:
    def _pb(p: np.ndarray, q: float) -> np.ndarray:
        r = y - p; return np.where(r >= 0, q*r, (q-1)*r)
    def _crps(p: dict) -> np.ndarray:
        return (_pb(p["q01"],0.1) + _pb(p["q05"],0.5) + _pb(p["q09"],0.9)) / 3.0
    d = _crps(pi) - _crps(pj)
    n = len(d); v = float(np.var(d, ddof=1) / n)
    stat = float(d.mean()) / np.sqrt(v) if v > 0 else 0.0
    return stat, float(2*(1 - scipy.stats.t.cdf(abs(stat), df=n-1)))

dm3_s, dm3_p = _dm(y_dl_w, p_with, p_without)
dm2_s, dm2_p = _dm(y_dl_w2, p_with2, p_without2)

cov3_w  = float(np.mean((y_dl_w  >= p_with["q01"])  & (y_dl_w  <= p_with["q09"])))
cov3_wo = float(np.mean((y_dl_wo >= p_without["q01"]) & (y_dl_wo <= p_without["q09"])))
cov2_w  = float(np.mean((y_dl_w2  >= p_with2["q01"])  & (y_dl_w2  <= p_with2["q09"])))
cov2_wo = float(np.mean((y_dl_wo2 >= p_without2["q01"]) & (y_dl_wo2 <= p_without2["q09"])))

print("\n" + "═" * 68)
print("SMOKE TEST KARŞILAŞTIRMA — Rnd G %30, seed=42, daylight+CQR(k=2)")
print("═" * 68)
print(f"\n{'Model':<8}  {'CRPS_w':>8}  {'CRPS_0':>8}  {'ΔCRPS%':>8}  {'Cov_w':>7}  {'Cov_0':>7}  {'DM_p':>10}")
print("─" * 68)
print(f"{'v2':<8}  {m_with2['crps']:>8.4f}  {m_without2['crps']:>8.4f}  {delta2_pct:>+7.2f}%  "
      f"{cov2_w:>7.4f}  {cov2_wo:>7.4f}  {dm2_p:>10.2e}")
print(f"{'v3':<8}  {m_with['crps']:>8.4f}  {m_without['crps']:>8.4f}  {delta_pct:>+7.2f}%  "
      f"{cov3_w:>7.4f}  {cov3_wo:>7.4f}  {dm3_p:>10.2e}")
print("═" * 68)

if delta_crps < 0:
    print(f"\n>>> v3 BAŞARILI: Flags CRPS'i {abs(delta_pct):.2f}% iyileştirdi")
    print(f"    Gece contamination v2'nin sorunuydu → v3 ile giderildi")
else:
    improvement = delta2_pct - delta_pct
    print(f"\n>>> v3 hâlâ +{delta_pct:.2f}% (flags bozuyor)")
    print(f"    v2'ye göre {improvement:+.2f}pp fark — gece contamination kısmi etkiydi")
    print(f"    Asıl sorun: multi-sensor eğitim vs single-sensor test dağılım uyumsuzluğu")

# ── 10. Kaydet ────────────────────────────────────────────────────────────────

joblib.dump(models_v3, OUT_PATH)
log.info("Kaydedildi: %s", OUT_PATH)
print(f"\nKaydedildi: {OUT_PATH}")
