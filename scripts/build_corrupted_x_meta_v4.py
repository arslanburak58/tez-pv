"""
v4: Single-sensor, variable-rate corruption-aware meta training.

Strateji (v2/v3 uyumsuzluğunu giderir):
  - Her satır için bağımsız: rate ~ Uniform(0.10, 0.50), sonra Bernoulli(rate)
  - Corruption varsa: G / T_amb / RH arasından rastgele TEK sensör
  - Sonuç: sum=0 ≈ %70, sum=1 ≈ %30, sum=2+ = %0
  - Test dağılımıyla (Rnd G %30 = tek sensör) eşleşir

v2 sorunu: 3 bağımsız Bernoulli → %21.5 multi-sensor → q09 flag coef -5.32 yanlış öğrendi
v4 fix:    tek-sensörlü → flag katsayıları doğru semantiği öğrenir
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

from models.base_learners import META_COLS
from models.meta_learner import FLAG_COLS, enrich_x_meta, predict_intervals, train_all_meta_learners
from models.baselines import evaluate_quantiles
from evaluation.comparison import enforce_monotonicity, normalize_stacked_preds
from evaluation.cqr import apply_locally_scaled
from models.meta_learner import _q_cols

NOCT: float = 46.0
ALPHA: float = 0.5
RATE_LOW: float = 0.10
RATE_HIGH: float = 0.50
DAYLIGHT_THRESHOLD: float = 0.087
CQR_K: float = 2.0
OUT_PATH: str = "data/processed/meta_models_robust_v4.joblib"

SENSORS: list[str] = ["G", "T_amb", "RH"]
SENSOR_TO_FLAG: dict[str, str] = {
    "G": "is_G_missing", "T_amb": "is_Tamb_missing", "RH": "is_RH_missing",
}


def _compute_cell_temp(T_amb: pd.Series, G: pd.Series) -> pd.Series:
    return (T_amb + G * (NOCT - 20.0) / 800.0).rename("T_cell")


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

log.info("X_val: %s  |  clean x_meta (OOF): %s", X_val.shape, xm_clean.shape)

# ── 2. Single-sensor, variable-rate corruption ────────────────────────────────

rng = np.random.default_rng(42)
n   = len(X_val)

# Her satır için: rate ~ Uniform(0.10, 0.50) → Bernoulli(rate)
rates        = rng.uniform(RATE_LOW, RATE_HIGH, n)
corrupt_mask = rng.random(n) < rates          # bool: bu satır corrupt mu?
sensor_idx   = rng.integers(0, 3, n)          # 0=G, 1=T_amb, 2=RH

# Sensör bazında maskeler (sadece corrupt satırlar, tek sensör)
G_mask    = corrupt_mask & (sensor_idx == 0)
Tamb_mask = corrupt_mask & (sensor_idx == 1)
RH_mask   = corrupt_mask & (sensor_idx == 2)

sensor_masks = {"G": G_mask, "T_amb": Tamb_mask, "RH": RH_mask}

for s, m in sensor_masks.items():
    log.info("  %s: %d/%d (%.1f%%) corrupt", s, m.sum(), n, 100*m.mean())

total_corrupt = corrupt_mask.sum()
log.info("Toplam corrupt: %d/%d (%.1f%%)", total_corrupt, n, 100*corrupt_mask.mean())

# ── 3. Korupsiyon uygula ──────────────────────────────────────────────────────

X_corrupt = X_val.copy()

for sensor, mask in sensor_masks.items():
    if sensor not in X_corrupt.columns:
        continue
    X_corrupt.loc[X_corrupt.index[mask], sensor] = np.nan
    X_corrupt[sensor] = X_corrupt[sensor].ffill().bfill()
    X_corrupt[SENSOR_TO_FLAG[sensor]] = mask.astype(int)

if ("G" in sensor_masks or "T_amb" in sensor_masks) and "T_cell" in X_corrupt.columns:
    X_corrupt["T_cell"] = _compute_cell_temp(X_corrupt["T_amb"], X_corrupt["G"])
    log.info("T_cell yeniden hesaplandı")

# Flag dağılımı kontrolü
flag_sum = sum(X_corrupt[SENSOR_TO_FLAG[s]].to_numpy() for s in SENSORS)
log.info("Flag dağılımı: sum=0=%.1f%%  sum=1=%.1f%%  sum=2+=%.1f%%",
         100*(flag_sum==0).mean(), 100*(flag_sum==1).mean(), 100*(flag_sum>=2).mean())

# ── 4. Corrupted base predictions ────────────────────────────────────────────

log.info("9 base model corrupted X_val üzerinde tahmin yapıyor...")
corrupted_preds: dict[str, np.ndarray] = {}
for col in META_COLS:
    corrupted_preds[col] = np.asarray(bm[col].predict(X_corrupt), dtype=np.float64)

X_meta_corrupt_df = pd.DataFrame(corrupted_preds, index=X_corrupt.index)
flags_corrupt     = X_corrupt[[FLAG_COLS[0], FLAG_COLS[1], FLAG_COLS[2]]].copy()
X_meta_corrupt_12 = enrich_x_meta(X_meta_corrupt_df, flags_corrupt)

# ── 5. Clean OOF hazırla ──────────────────────────────────────────────────────

flags_clean      = X_train[FLAG_COLS].copy()
X_meta_clean_12  = enrich_x_meta(xm_clean, flags_clean)
y_meta_clean     = y_train.loc[X_meta_clean_12.index]

# ── 6. Concatenate ────────────────────────────────────────────────────────────

y_corrupt = y_val.loc[X_meta_corrupt_12.index]

X_combined = pd.concat([X_meta_clean_12, X_meta_corrupt_12], ignore_index=True)
y_combined  = pd.concat([y_meta_clean,   y_corrupt],          ignore_index=True)

log.info("Birleşik veri: %s  (clean=%d + corrupt=%d)",
         X_combined.shape, len(X_meta_clean_12), len(X_meta_corrupt_12))

# ── 7. QuantileLinear v4 ──────────────────────────────────────────────────────

log.info("QuantileLinear v4 × 3 eğitiliyor (alpha=%.2f)...", ALPHA)
models_v4 = train_all_meta_learners(X_combined, y_combined, alpha=ALPHA, checkpoint_dir=None)
log.info("Eğitim tamamlandı.")

# ── 8. Flag katsayı raporu ───────────────────────────────────────────────────

print("\n=== FLAG DAĞILIMI KONTROLÜ ===")
fc_sum = sum(X_meta_corrupt_12[fc].to_numpy() for fc in FLAG_COLS)
for v in range(3):
    cnt = int((fc_sum == v).sum())
    print(f"  sum={v}: {cnt:>8,} ({100*cnt/len(fc_sum):.1f}%)")
cnt2 = int((fc_sum >= 2).sum())
print(f"  sum=2+: {cnt2:>7,} ({100*cnt2/len(fc_sum):.1f}%)  ← beklenen: %0")

models_v2 = joblib.load("data/processed/meta_models_robust_v2.joblib")

print("\n=== FLAG COEFFİCİENTLER ===")
print(f"{'Model':<12}  {'v2_G':>9}  {'v2_T':>9}  {'v2_RH':>9}  {'v2_L2':>7}"
      f"  {'v4_G':>9}  {'v4_T':>9}  {'v4_RH':>9}  {'v4_L2':>7}")
print("─" * 84)
for key in ("meta_q01", "meta_q05", "meta_q09"):
    q    = int(key[-2:]) / 10.0
    cols = _q_cols(q)
    fp   = [cols.index(fc) for fc in FLAG_COLS]
    c2   = models_v2[key].coef_[fp]
    c4   = models_v4[key].coef_[fp]
    print(f"{key:<12}  {c2[0]:>9.4f}  {c2[1]:>9.4f}  {c2[2]:>9.4f}  {np.linalg.norm(c2):>7.4f}"
          f"  {c4[0]:>9.4f}  {c4[1]:>9.4f}  {c4[2]:>9.4f}  {np.linalg.norm(c4):>7.4f}")

# ── 9. Smoke test — Rnd G %30, seed=42 ──────────────────────────────────────

log.info("Smoke test: Rnd G %30, seed=42...")

X_test = ds["X_test"].copy()
y_arr  = np.asarray(ds["y_test"], dtype=np.float64)
dl     = X_test["cos_zenith"].to_numpy() > DAYLIGHT_THRESHOLD

rng2 = np.random.default_rng(42)
miss = rng2.random(len(X_test)) < 0.30
X_smoke = X_test.copy()
X_smoke.loc[X_smoke.index[miss], "G"] = np.nan
X_smoke["G"] = X_smoke["G"].ffill().bfill()
if "T_cell" in X_smoke.columns:
    X_smoke["T_cell"] = _compute_cell_temp(X_smoke["T_amb"], X_smoke["G"])
X_smoke["is_G_missing"] = miss.astype(int)

bp: dict[str, np.ndarray] = {}
for col in META_COLS:
    bp[col] = np.asarray(bm[col].predict(X_smoke), dtype=np.float64)
Xm = pd.DataFrame(bp, index=X_smoke.index)

flags_real = X_smoke[FLAG_COLS].copy()
flags_zero = pd.DataFrame({c: np.zeros(len(X_smoke), dtype=int) for c in FLAG_COLS},
                           index=X_smoke.index)


def _preds_dl(models: dict, X_12: pd.DataFrame, y: np.ndarray,
              dl_m: np.ndarray, k: float) -> tuple[np.ndarray, dict]:
    raw = normalize_stacked_preds(predict_intervals(models, X_12))
    raw = enforce_monotonicity(raw)
    raw = enforce_monotonicity(apply_locally_scaled(raw, k))
    n_p = len(raw["q01"]); m = dl_m[:n_p]
    return y[:n_p][m], {kk: vv[m] for kk, vv in raw.items()}


def _dm(y: np.ndarray, pi: dict, pj: dict) -> tuple[float, float]:
    def pb(p: np.ndarray, q: float) -> np.ndarray:
        r = y - p; return np.where(r >= 0, q*r, (q-1)*r)
    def crps(p: dict) -> np.ndarray:
        return (pb(p["q01"],0.1) + pb(p["q05"],0.5) + pb(p["q09"],0.9)) / 3.0
    d = crps(pi) - crps(pj)
    n = len(d); v = float(np.var(d, ddof=1)/n)
    s = float(d.mean()) / np.sqrt(v) if v > 0 else 0.0
    return s, float(2*(1-scipy.stats.t.cdf(abs(s), df=n-1)))


X_real = enrich_x_meta(Xm, flags_real)
X_zero = enrich_x_meta(Xm, flags_zero)

print("\n" + "═"*70)
print("SMOKE TEST — Rnd G %30, seed=42, daylight+CQR(k=2)")
print("═"*70)
print(f"\n{'Model':<8}  {'CRPS_w':>8}  {'CRPS_0':>8}  {'ΔCRPS%':>8}  {'Cov_w':>7}  {'DM_p':>10}")
print("─"*70)

for tag, mm in [("v2", models_v2), ("v4", models_v4)]:
    yw, pw = _preds_dl(mm, X_real, y_arr, dl, CQR_K)
    yo, po = _preds_dl(mm, X_zero, y_arr, dl, CQR_K)
    mw = evaluate_quantiles(yw, pw); mo = evaluate_quantiles(yo, po)
    dp = (mw["crps"]-mo["crps"])/mo["crps"]*100
    cov = float(np.mean((yw >= pw["q01"]) & (yw <= pw["q09"])))
    _, dm_p = _dm(yw, pw, po)
    direction = "✓ iyileştirdi" if dp < 0 else "✗ bozdu"
    print(f"{tag:<8}  {mw['crps']:>8.4f}  {mo['crps']:>8.4f}  {dp:>+7.2f}%  {cov:>7.4f}  {dm_p:>10.2e}  {direction}")

print("═"*70)

# ── 10. Kaydet ────────────────────────────────────────────────────────────────

joblib.dump(models_v4, OUT_PATH)
log.info("Kaydedildi: %s", OUT_PATH)
print(f"\nKaydedildi: {OUT_PATH}")
