"""
STAGE-8 smoke test — tek senaryo, hızlı doğrulama.

Senaryo: %30 random missingness, yalnızca G sütununda.
Karşılaştırma: meta_models_robust (flag'li) vs aynı model ama zero-flags.
"""

import logging
import random

import joblib
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

import scipy.stats

from models.base_learners import META_COLS
from models.meta_learner import FLAG_COLS, enrich_x_meta, predict_intervals
from models.baselines import evaluate_quantiles
from evaluation.comparison import enforce_monotonicity, normalize_stacked_preds
from evaluation.cqr import apply_locally_scaled


def _dm_test(
    y: np.ndarray,
    preds_i: dict[str, np.ndarray],
    preds_j: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    """Diebold-Mariano testi (HLN düzeltmeli). Döndürür: (dm_stat, p_value, mean_diff)."""
    def _pb(pred: np.ndarray, q: float) -> np.ndarray:
        r = y - pred
        return np.where(r >= 0, q * r, (q - 1) * r)
    def _crps_obs(p: dict) -> np.ndarray:
        return (_pb(p["q01"], 0.1) + _pb(p["q05"], 0.5) + _pb(p["q09"], 0.9)) / 3.0

    d      = _crps_obs(preds_i) - _crps_obs(preds_j)
    n      = len(d)
    mean_d = float(d.mean())
    # HLN: var(d) / n  düzeltmesi
    var_d  = float(np.var(d, ddof=1) / n)
    dm_stat = mean_d / np.sqrt(var_d) if var_d > 0 else 0.0
    p_val   = 2 * (1 - scipy.stats.t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_val), mean_d

DAYLIGHT_THRESHOLD: float = 0.087
MISSING_PROB:       float = 0.30
CQR_K:              float = 2.0

# ── 1. Yükle ──────────────────────────────────────────────────────────────────

log.info("Yükleniyor...")
ds      = joblib.load("data/processed/dataset.joblib")
bm      = joblib.load("data/processed/base_models.joblib")
mm_rob  = joblib.load("data/processed/meta_models_robust.joblib")

X_test  = ds["X_test"].copy()
y_arr   = np.asarray(ds["y_test"], dtype=np.float64)

log.info("X_test: %s", X_test.shape)

# ── 2. Daylight maskesi ────────────────────────────────────────────────────────

dl_mask = X_test["cos_zenith"].to_numpy(dtype=np.float64) > DAYLIGHT_THRESHOLD
log.info("Daylight: %d / %d  (%.1f%%)", dl_mask.sum(), len(dl_mask), 100*dl_mask.mean())

# ── 3. Senaryo: %30 G eksikliği ───────────────────────────────────────────────

rng = np.random.default_rng(42)
miss_mask = rng.random(len(X_test)) < MISSING_PROB

X_corrupted = X_test.copy()
X_corrupted.loc[miss_mask, "G"] = np.nan
X_corrupted["G"] = X_corrupted["G"].ffill().bfill()   # ffill, ardından bfill uç güvencesi
X_corrupted["is_G_missing"] = miss_mask.astype(int)

log.info("Senaryo G-missingness: %.1f%% satır eksik", 100 * miss_mask.mean())

# ── 4. Base model tahminleri (corrupted X üzerinde) ───────────────────────────

meta_preds: dict[str, np.ndarray] = {}
for col in META_COLS:
    meta_preds[col] = np.asarray(bm[col].predict(X_corrupted), dtype=np.float64)
X_meta_test = pd.DataFrame(meta_preds, index=X_corrupted.index)

# ── 5. İki varyant: gerçek flags vs zero flags ─────────────────────────────────

flags_real = X_corrupted[FLAG_COLS].copy()                  # is_G_missing=mask, diğerleri=0
flags_zero = pd.DataFrame(
    {c: np.zeros(len(X_corrupted), dtype=int) for c in FLAG_COLS},
    index=X_corrupted.index,
)

X_meta_real = enrich_x_meta(X_meta_test, flags_real)   # 12 sütun, gerçek flag
X_meta_zero = enrich_x_meta(X_meta_test, flags_zero)   # 12 sütun, flag=0

# meta_models_robust ile tahmin — farkı yaratacak olan flag katsayıları
raw_with    = normalize_stacked_preds(predict_intervals(mm_rob, X_meta_real))
raw_without = normalize_stacked_preds(predict_intervals(mm_rob, X_meta_zero))

preds_with    = enforce_monotonicity(raw_with)
preds_without = enforce_monotonicity(raw_without)

# ── 6. CQR k=2.0 + daylight maskesi ──────────────────────────────────────────

def _dl_filter(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(preds["q01"])
    m = mask[:n]
    return y[:n][m], {k: v[m] for k, v in preds.items()}

preds_with_cqr    = enforce_monotonicity(apply_locally_scaled(preds_with,    CQR_K))
preds_without_cqr = enforce_monotonicity(apply_locally_scaled(preds_without, CQR_K))

y_dl, pw_dl  = _dl_filter(y_arr, preds_with_cqr,    dl_mask)
_,    pwo_dl = _dl_filter(y_arr, preds_without_cqr, dl_mask)

# ── 7. Metrikler ──────────────────────────────────────────────────────────────

m_with    = evaluate_quantiles(y_dl, pw_dl)
m_without = evaluate_quantiles(y_dl, pwo_dl)

crps_with    = m_with["crps"]
crps_without = m_without["crps"]
delta_crps   = crps_with - crps_without
delta_pct    = delta_crps / crps_without * 100 if crps_without > 0 else float("nan")
direction    = "flags iyileştirdi" if delta_crps < 0 else "flags bozdu / etkisiz"

cov_with    = float(np.mean((y_dl >= pw_dl["q01"])  & (y_dl <= pw_dl["q09"])))
cov_without = float(np.mean((y_dl >= pwo_dl["q01"]) & (y_dl <= pwo_dl["q09"])))

# ── 8. DM testi ───────────────────────────────────────────────────────────────

dm_stat, dm_p_adj, dm_mean_diff = _dm_test(y_dl, pw_dl, pwo_dl)
dm_significant = dm_p_adj < 0.05

# ── 9. Rapor ──────────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
print("STAGE-8 SMOKE TEST — %30 G missingness, daylight+CQR(k=2)")
print("═" * 60)

print(f"\nSenaryo     : G sütununda %{100*MISSING_PROB:.0f} missingness ({miss_mask.sum():,} satır)")
print(f"Daylight n  : {dl_mask.sum():,}")

print(f"\n{'Metrik':<22}  {'with_flags':>12}  {'zero_flags':>12}  {'Δ (with-without)':>18}")
print("─" * 68)
print(f"{'CRPS':<22}  {crps_with:>12.4f}  {crps_without:>12.4f}  {delta_crps:>+18.4f}")
print(f"{'MAE':<22}  {m_with['mae']:>12.4f}  {m_without['mae']:>12.4f}  "
      f"{m_with['mae']-m_without['mae']:>+18.4f}")
print(f"{'Coverage':<22}  {cov_with:>12.4f}  {cov_without:>12.4f}  "
      f"{cov_with-cov_without:>+18.4f}")
print(f"{'Pinball_q01':<22}  {m_with['pinball_q01']:>12.4f}  {m_without['pinball_q01']:>12.4f}  "
      f"{m_with['pinball_q01']-m_without['pinball_q01']:>+18.4f}")
print(f"{'Pinball_q09':<22}  {m_with['pinball_q09']:>12.4f}  {m_without['pinball_q09']:>12.4f}  "
      f"{m_with['pinball_q09']-m_without['pinball_q09']:>+18.4f}")

print(f"\nΔCRPS / CRPS_without : {delta_pct:+.2f}%  →  {direction}")
print(f"DM mean_diff         : {dm_mean_diff:+.6f}")
print(f"DM p_adj             : {dm_p_adj:.2e}  (significant={dm_significant})")

print("\n── YARGI ──")
abs_pct = abs(delta_pct)
if abs_pct > 5 and delta_crps < 0:
    verdict = "FLAG ETKİSİ ANLAMLI — flags CRPS'yi >%5 iyileştirdi ✓"
elif abs_pct < 2:
    verdict = "Flag etkisi ihmal edilebilir (<%%2 fark)"
elif delta_crps < 0:
    verdict = f"Marjinal iyileşme (%{abs_pct:.1f}, %2-%5 arası)"
else:
    verdict = f"Flag etkisi negatif veya marjinal (%{abs_pct:.1f}, yön bozuk)"
print(f"  {verdict}")
print("═" * 60)
