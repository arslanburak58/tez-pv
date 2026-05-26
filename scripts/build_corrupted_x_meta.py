"""
PAKET B v2: Corrupted raw features → Base predictions → Robust meta retrain.

Sorun (v1): Yalnızca flag sütunlarını 0→1 yaparak augment etmek işe yaramadı.
            Base model tahminleri değişmediği için flag katsayıları ~0.003 kaldı.

Çözüm (v2): Ham öznitelikler (G, T_amb, RH) NaN → ffill olarak corrupt edilir,
            base modeller bu corrupted X üzerinde tahmin yapar.
            Meta-model, "flag=1 → base tahmin sapması" ilişkisini öğrenir.
            Beklenen flag coef normu: ~0.5–2.0 (v1: ~0.003).

Adımlar:
  1. X_val (ham, OOF dışı) yükle → korupsiyon kaynağı
  2. Her sensör (G, T_amb, RH) için bağımsız Bernoulli(0.30)
  3. NaN → ffill/bfill impute; T_cell yeniden hesapla
  4. 9 base modeli corrupted X üzerinde çalıştır → corrupted preds
  5. corrupted_x_meta_12 = 9 pred + 3 gerçek flag
  6. Clean x_meta (OOF) + corrupted_x_meta concatenate et
  7. QuantileLinear yeniden eğit → meta_models_robust_v2.joblib
  8. Flag katsayılarını raporla
  9. Smoke test (v2 modeli ile) çalıştır
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

from models.base_learners import META_COLS, ALGOS, QUANTILES
from models.meta_learner import FLAG_COLS, enrich_x_meta, predict_intervals, train_all_meta_learners

NOCT: float = 46.0  # Ross model sabit


def _compute_cell_temp(T_amb: pd.Series, G: pd.Series) -> pd.Series:
    return (T_amb + G * (NOCT - 20.0) / 800.0).rename("T_cell")


BERNOULLI_P: float = 0.30
ALPHA: float = 0.5          # v1'den düşük → flag coef'lerin büyümesine izin ver
OUT_PATH: str = "data/processed/meta_models_robust_v2.joblib"

# Sensör sütunu → flag sütunu eşlemesi
SENSOR_TO_FLAG: dict[str, str] = {
    "G":     "is_G_missing",
    "T_amb": "is_Tamb_missing",
    "RH":    "is_RH_missing",
}

# ── 1. Yükle ──────────────────────────────────────────────────────────────────

log.info("Yükleniyor...")
ds  = joblib.load("data/processed/dataset.joblib")
bm  = joblib.load("data/processed/base_models.joblib")

xm_raw        = joblib.load("data/processed/x_meta.joblib")
xm_clean, _   = xm_raw if isinstance(xm_raw, tuple) else (xm_raw, None)

X_val  = ds["X_val"].copy()
y_val  = ds["y_val"]
X_train = ds["X_train"]
y_train = ds["y_train"]

log.info("X_val: %s  |  clean x_meta (OOF): %s", X_val.shape, xm_clean.shape)

# ── 2. X_val üzerinde bağımsız Bernoulli korupsiyonu ──────────────────────────

rng = np.random.default_rng(42)
n   = len(X_val)

X_corrupt = X_val.copy()

# Her sensör için bağımsız flag maskesi
sensor_masks: dict[str, np.ndarray] = {}
for sensor in SENSOR_TO_FLAG:
    if sensor not in X_corrupt.columns:
        log.warning("%s sütunu X_val'de yok — atlandı", sensor)
        continue
    mask = rng.random(n) < BERNOULLI_P
    sensor_masks[sensor] = mask
    log.info("  %s: %d / %d satır (%.1f%%) corrupt edildi",
             sensor, mask.sum(), n, 100 * mask.mean())

# Önce ham sütunları NaN yap, sonra ffill/bfill impute
for sensor, mask in sensor_masks.items():
    X_corrupt.loc[mask, sensor] = np.nan
    X_corrupt[sensor] = X_corrupt[sensor].ffill().bfill()
    flag_col = SENSOR_TO_FLAG[sensor]
    if flag_col in X_corrupt.columns:
        X_corrupt[flag_col] = X_corrupt[flag_col].copy()
    X_corrupt[flag_col] = mask.astype(int)

# T_cell yeniden hesapla (G veya T_amb değiştiyse)
if ("G" in sensor_masks or "T_amb" in sensor_masks) and "T_cell" in X_corrupt.columns:
    X_corrupt["T_cell"] = _compute_cell_temp(X_corrupt["T_amb"], X_corrupt["G"])
    log.info("T_cell yeniden hesaplandı (G/T_amb corrupt sonrası)")

# İstatistik
any_corrupt = np.zeros(n, dtype=bool)
for mask in sensor_masks.values():
    any_corrupt |= mask
multi_corrupt = np.zeros(n, dtype=bool)
flag_sum = np.zeros(n, dtype=int)
for mask in sensor_masks.values():
    flag_sum += mask.astype(int)
multi_corrupt = flag_sum > 1

log.info("En az 1 sensör corrupt: %.1f%%  |  2+ sensör: %.1f%%",
         100 * any_corrupt.mean(), 100 * multi_corrupt.mean())

# ── 3. 9 base modeli corrupted X üzerinde çalıştır ────────────────────────────

log.info("9 base model, corrupted X_val üzerinde tahmin yapıyor...")
corrupted_preds: dict[str, np.ndarray] = {}
for col in META_COLS:
    corrupted_preds[col] = np.asarray(bm[col].predict(X_corrupt), dtype=np.float64)
    log.info("  %s: mean=%.4f  std=%.4f", col,
             corrupted_preds[col].mean(), corrupted_preds[col].std())

# ── 4. corrupted_x_meta_12 oluştur ────────────────────────────────────────────

X_meta_corrupt_df = pd.DataFrame(corrupted_preds, index=X_corrupt.index)
flags_corrupt     = X_corrupt[[FLAG_COLS[0], FLAG_COLS[1], FLAG_COLS[2]]].copy()
X_meta_corrupt_12 = enrich_x_meta(X_meta_corrupt_df, flags_corrupt)

log.info("corrupted_x_meta_12: %s  |  flag aktif oranı: is_G=%.2f  is_T=%.2f  is_RH=%.2f",
         X_meta_corrupt_12.shape,
         flags_corrupt["is_G_missing"].mean(),
         flags_corrupt["is_Tamb_missing"].mean(),
         flags_corrupt["is_RH_missing"].mean())

# ── 5. Clean x_meta (OOF) hazırla ────────────────────────────────────────────

flags_clean  = X_train[FLAG_COLS].copy()
X_meta_clean_12 = enrich_x_meta(xm_clean, flags_clean)
y_meta_clean = y_train.loc[X_meta_clean_12.index]

log.info("Clean x_meta_12: %s  |  y_meta_clean: %d", X_meta_clean_12.shape, len(y_meta_clean))

# ── 6. Concatenate ────────────────────────────────────────────────────────────

y_corrupt = y_val.loc[X_meta_corrupt_12.index]

X_combined = pd.concat([X_meta_clean_12, X_meta_corrupt_12], ignore_index=True)
y_combined = pd.concat([y_meta_clean, y_corrupt], ignore_index=True)

log.info("Birleşik veri: %s  (clean=%d + corrupt=%d)",
         X_combined.shape, len(X_meta_clean_12), len(X_meta_corrupt_12))

# Flag oranı birleşik sette
for fc in FLAG_COLS:
    rate = float(X_combined[fc].mean())
    log.info("  %s aktif: %.3f (%.1f%%)", fc, rate, 100 * rate)

# ── 7. QuantileLinear yeniden eğit ────────────────────────────────────────────

log.info("QuantileLinear × 3 eğitiliyor (alpha=%.2f, birleşik veri)...", ALPHA)
models_v2 = train_all_meta_learners(X_combined, y_combined, alpha=ALPHA,
                                    checkpoint_dir=None)
log.info("Eğitim tamamlandı.")

# ── 8. Train coverage raporu ──────────────────────────────────────────────────

from models.baselines import evaluate_quantiles
from evaluation.comparison import enforce_monotonicity, normalize_stacked_preds

preds_train = predict_intervals(models_v2, X_combined)
y_np  = np.asarray(y_combined)
p01, p05, p09 = preds_train["meta_q01"], preds_train["meta_q05"], preds_train["meta_q09"]
cov_train  = float(np.mean((y_np >= p01) & (y_np <= p09)))
mono_train = float(np.mean((p01 <= p05) & (p05 <= p09)))
print(f"\nTrain coverage (birleşik, %80 hedef): {cov_train:.4f}  |  monotonluk: {mono_train:.4f}")

# ── 9. Flag coefficient raporu ────────────────────────────────────────────────

from models.meta_learner import _q_cols

print("\n=== FLAG COEFFİCİENTLER (robust_v2) ===")
print(f"{'Model':<12}  {'is_G_miss':>10}  {'is_Tamb_miss':>12}  {'is_RH_miss':>10}  "
      f"{'L2 norm':>8}  {'sıfır_mi':>8}")
print("─" * 70)

all_zero = True
for key, model in models_v2.items():
    q        = int(key[-2:]) / 10.0
    cols     = _q_cols(q)
    flag_pos = [cols.index(fc) for fc in FLAG_COLS]
    fc_vals  = model.coef_[flag_pos]
    l2_norm  = float(np.linalg.norm(fc_vals))
    is_zero  = np.allclose(fc_vals, 0, atol=1e-4)
    if not is_zero:
        all_zero = False
    print(f"{key:<12}  {fc_vals[0]:>10.6f}  {fc_vals[1]:>12.6f}  "
          f"{fc_vals[2]:>10.6f}  {l2_norm:>8.4f}  {'EVET' if is_zero else 'HAYIR':>8}")

print()
if all_zero:
    print("UYARI: Tüm flag katsayıları hâlâ sıfır — corrupted base predictions etkisiz.")
else:
    print("BAŞARILI: En az bir flag katsayısı anlamlı.")

# v1 ile karşılaştırma
import os
if os.path.exists("data/processed/meta_models_robust.joblib"):
    models_v1 = joblib.load("data/processed/meta_models_robust.joblib")
    print("\n=== v1 (naive aug) vs v2 (corrupted preds) FLAG COEF L2 NORM ===")
    print(f"{'Model':<12}  {'v1 norm':>10}  {'v2 norm':>10}  {'fark':>10}")
    print("─" * 46)
    for key in ("meta_q01", "meta_q05", "meta_q09"):
        q        = int(key[-2:]) / 10.0
        cols     = _q_cols(q)
        flag_pos = [cols.index(fc) for fc in FLAG_COLS]
        n1 = float(np.linalg.norm(models_v1[key].coef_[flag_pos]))
        n2 = float(np.linalg.norm(models_v2[key].coef_[flag_pos]))
        print(f"{key:<12}  {n1:>10.6f}  {n2:>10.6f}  {n2-n1:>+10.6f}")

# ── 10. Hızlı smoke: corrupted X_val üzerinde with vs without flags ───────────

import scipy.stats

log.info("Hızlı smoke test: %30 G missingness, corrupted preds ile...")

SMOKE_MISSING_PROB: float = 0.30
SMOKE_CQR_K:       float = 2.0
DAYLIGHT_THRESHOLD: float = 0.087

ds_test  = ds
X_test   = ds_test["X_test"].copy()
y_arr    = np.asarray(ds_test["y_test"], dtype=np.float64)
dl_mask  = X_test["cos_zenith"].to_numpy(dtype=np.float64) > DAYLIGHT_THRESHOLD

rng2 = np.random.default_rng(99)
miss_mask = rng2.random(len(X_test)) < SMOKE_MISSING_PROB
X_smoke = X_test.copy()
X_smoke.loc[miss_mask, "G"] = np.nan
X_smoke["G"] = X_smoke["G"].ffill().bfill()
if "T_cell" in X_smoke.columns:
    X_smoke["T_cell"] = _compute_cell_temp(X_smoke["T_amb"], X_smoke["G"])
X_smoke["is_G_missing"] = miss_mask.astype(int)

smoke_preds: dict[str, np.ndarray] = {}
for col in META_COLS:
    smoke_preds[col] = np.asarray(bm[col].predict(X_smoke), dtype=np.float64)
X_meta_smoke = pd.DataFrame(smoke_preds, index=X_smoke.index)

from models.meta_learner import FLAG_COLS as FC
flags_real  = X_smoke[FC].copy()
flags_zero  = pd.DataFrame({c: np.zeros(len(X_smoke), dtype=int) for c in FC},
                            index=X_smoke.index)

X_meta_real = enrich_x_meta(X_meta_smoke, flags_real)
X_meta_zero = enrich_x_meta(X_meta_smoke, flags_zero)

from evaluation.comparison import normalize_stacked_preds
from evaluation.cqr import apply_locally_scaled

def _preds_dl(models, X_12, y, mask, k):
    raw = normalize_stacked_preds(predict_intervals(models, X_12))
    raw = enforce_monotonicity(raw)
    raw = enforce_monotonicity(apply_locally_scaled(raw, k))
    n   = len(raw["q01"])
    m   = mask[:n]
    return y[:n][m], {kk: vv[m] for kk, vv in raw.items()}

y_dl_w,  p_with    = _preds_dl(models_v2, X_meta_real, y_arr, dl_mask, SMOKE_CQR_K)
y_dl_wo, p_without = _preds_dl(models_v2, X_meta_zero, y_arr, dl_mask, SMOKE_CQR_K)

m_with    = evaluate_quantiles(y_dl_w,  p_with)
m_without = evaluate_quantiles(y_dl_wo, p_without)
delta_crps  = m_with["crps"] - m_without["crps"]
delta_pct   = delta_crps / m_without["crps"] * 100 if m_without["crps"] > 0 else float("nan")
direction   = "flags iyileştirdi ✓" if delta_crps < 0 else "flags bozdu / etkisiz"

def _dm_stat(y, pi, pj):
    def _pb(p, q): r = y - p; return np.where(r >= 0, q*r, (q-1)*r)
    def _crps(p): return (_pb(p["q01"],0.1) + _pb(p["q05"],0.5) + _pb(p["q09"],0.9)) / 3.0
    d = _crps(pi) - _crps(pj)
    n = len(d); v = float(np.var(d, ddof=1) / n)
    stat = float(d.mean()) / np.sqrt(v) if v > 0 else 0.0
    return stat, float(2*(1 - scipy.stats.t.cdf(abs(stat), df=n-1))), float(d.mean())

dm_s, dm_p, dm_d = _dm_stat(y_dl_w, p_with, p_without)

print("\n" + "═" * 62)
print("SMOKE TEST v2 — %30 G missingness, daylight+CQR(k=2), v2 meta")
print("═" * 62)
print(f"\n{'Metrik':<22}  {'with_flags':>12}  {'zero_flags':>12}  {'Δ':>10}")
print("─" * 62)
print(f"{'CRPS':<22}  {m_with['crps']:>12.4f}  {m_without['crps']:>12.4f}  {delta_crps:>+10.4f}")
print(f"{'MAE':<22}  {m_with['mae']:>12.4f}  {m_without['mae']:>12.4f}  "
      f"{m_with['mae']-m_without['mae']:>+10.4f}")
cov_w  = float(np.mean((y_dl_w  >= p_with["q01"])  & (y_dl_w  <= p_with["q09"])))
cov_wo = float(np.mean((y_dl_wo >= p_without["q01"]) & (y_dl_wo <= p_without["q09"])))
print(f"{'Coverage':<22}  {cov_w:>12.4f}  {cov_wo:>12.4f}  {cov_w-cov_wo:>+10.4f}")
print(f"\nΔCRPS%: {delta_pct:+.2f}%  →  {direction}")
print(f"DM: stat={dm_s:+.3f}  p={dm_p:.2e}  mean_diff={dm_d:+.6f}  "
      f"significant={dm_p < 0.05}")
print("─" * 62)

abs_pct = abs(delta_pct)
if abs_pct > 5 and delta_crps < 0:
    verdict = "FLAG ETKİSİ ANLAMLI — CRPS >%5 iyileşti ✓"
elif abs_pct < 2:
    verdict = "Flag etkisi ihmal edilebilir (<2% fark) ✗"
elif delta_crps < 0:
    verdict = f"Marjinal iyileşme (%{abs_pct:.1f}, 2-5% arası)"
else:
    verdict = f"Flag etkisi negatif (%{abs_pct:.1f})"
print(f"YARGI: {verdict}")
print("═" * 62)

# ── 11. Kaydet ────────────────────────────────────────────────────────────────

joblib.dump(models_v2, OUT_PATH)
log.info("Kaydedildi: %s", OUT_PATH)
print(f"\nKaydedildi: {OUT_PATH}")
