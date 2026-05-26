"""
PAKET B: Augmented missingness ile meta retrain.

Mantık:
  - Train OOF tahminlerine dokunma (base modeller clean veriyle tahmin yaptı).
  - Eğitim satırlarının %30'una rastgele 1 sensör flag'i aktif et.
  - Meta-learner "flag=1 iken base tahminlerine ne kadar güveneceğini" öğrensin.
  - Sonuç: meta_models_robust.joblib — flag coef'leri sıfır OLMAMALI.
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

from models.meta_learner import FLAG_COLS, META_IN_COLS, enrich_x_meta, train_all_meta_learners, _q_cols
from models.base_learners import META_COLS, ALGOS, QUANTILES, _col_name

BERNOULLI_P:  float = 0.30   # her sensör için bağımsız eksiklik olasılığı
ALPHA:        float = 1.0
OUT_PATH:     str   = "data/processed/meta_models_robust.joblib"

# ── 1. Yükle ──────────────────────────────────────────────────────────────────

log.info("Yükleniyor...")
xm_raw    = joblib.load("data/processed/x_meta.joblib")
xm_df, _  = xm_raw if isinstance(xm_raw, tuple) else (xm_raw, None)

ds        = joblib.load("data/processed/dataset.joblib")
X_train   = ds["X_train"]
y_train   = ds["y_train"]

log.info("x_meta: %s  X_train: %s", xm_df.shape, X_train.shape)

# ── 2. Temiz X_meta_12 ────────────────────────────────────────────────────────

flags_clean = X_train[FLAG_COLS].copy()
X_meta_12   = enrich_x_meta(xm_df, flags_clean)
y_meta      = y_train.loc[X_meta_12.index]

log.info("X_meta_12 (clean): %s  y_meta: %d", X_meta_12.shape, len(y_meta))

# ── 3. Augmented missingness (bağımsız Bernoulli per sensör) ─────────────────

rng = np.random.default_rng(42)
n   = len(X_meta_12)

# Her sensör için bağımsız Bernoulli(p) — satır başına 0, 1, 2 veya 3 flag aktif
X_aug = X_meta_12.copy()
for fc in FLAG_COLS:
    mask = rng.random(n) < BERNOULLI_P          # ~%30 satır True
    col_idx = X_aug.columns.get_loc(fc)
    X_aug.iloc[mask, col_idx] = 1               # clean=0 → augmented=1

# İstatistik
for fc in FLAG_COLS:
    orig = int(X_meta_12[fc].sum())
    augm = int(X_aug[fc].sum())
    log.info("  %s: clean=%d  augmented=%d  (+%d)", fc, orig, augm, augm - orig)

any_flag_rate = float((X_aug[FLAG_COLS] > 0).any(axis=1).mean())
multi_flag_rate = float((X_aug[FLAG_COLS].sum(axis=1) > 1).mean())
log.info("En az 1 flag aktif: %.3f  |  2+ flag aktif: %.3f",
         any_flag_rate, multi_flag_rate)

# ── 4. Retrain ────────────────────────────────────────────────────────────────

log.info("QuantileLinear × 3 eğitiliyor (augmented flags)...")
models_robust = train_all_meta_learners(X_aug, y_meta, alpha=ALPHA,
                                        checkpoint_dir=None)
log.info("Eğitim tamamlandı.")

# ── 5. Train coverage raporu ─────────────────────────────────────────────────

from models.meta_learner import predict_intervals
preds_train = predict_intervals(models_robust, X_aug)
y_np = np.asarray(y_meta)
p01, p05, p09 = preds_train["meta_q01"], preds_train["meta_q05"], preds_train["meta_q09"]
cov_train  = float(np.mean((y_np >= p01) & (y_np <= p09)))
mono_train = float(np.mean((p01 <= p05) & (p05 <= p09)))
print(f"\nTrain coverage (augmented): {cov_train:.4f}  |  monotonluk: {mono_train:.4f}")

# ── 6. Flag coefficient raporu ────────────────────────────────────────────────

print("\n=== FLAG COEFFİCİENTLER (robust meta) ===")
print(f"{'Model':<12}  {'is_G_miss':>10}  {'is_Tamb_miss':>12}  {'is_RH_miss':>10}  {'sıfır_mi':>8}")
print("─" * 62)

all_zero = True
for key, model in models_robust.items():
    q        = int(key[-2:]) / 10.0
    cols     = _q_cols(q)                      # 6 özellik: 3 OOF + 3 flag
    flag_pos = [cols.index(fc) for fc in FLAG_COLS]
    fc_vals  = model.coef_[flag_pos]
    is_zero  = np.allclose(fc_vals, 0, atol=1e-8)
    if not is_zero:
        all_zero = False
    print(f"{key:<12}  {fc_vals[0]:>10.6f}  {fc_vals[1]:>12.6f}  "
          f"{fc_vals[2]:>10.6f}  {'EVET' if is_zero else 'HAYIR':>8}")

print()
if all_zero:
    print("UYARI: Tüm flag katsayıları hâlâ sıfır — augmentation etkisiz.")
else:
    print("BAŞARILI: En az bir flag katsayısı sıfırdan farklı.")

# Temiz vs robust karşılaştırma
log.info("Temiz meta modeli de yükleniyor (karşılaştırma için)...")
models_clean = joblib.load("data/processed/meta_models.joblib")
print("\n=== TEMİZ vs ROBUST KARŞILAŞTIRMA ===")
print(f"{'Model':<12}  {'Clean flag coef norm':>20}  {'Robust flag coef norm':>22}")
print("─" * 58)
for key in ("meta_q01", "meta_q05", "meta_q09"):
    q        = int(key[-2:]) / 10.0
    cols     = _q_cols(q)
    flag_pos = [cols.index(fc) for fc in FLAG_COLS]
    c_norm   = float(np.linalg.norm(models_clean[key].coef_[flag_pos]))
    r_norm   = float(np.linalg.norm(models_robust[key].coef_[flag_pos]))
    print(f"{key:<12}  {c_norm:>20.6f}  {r_norm:>22.6f}")

# ── 7. Kaydet ─────────────────────────────────────────────────────────────────

joblib.dump(models_robust, OUT_PATH)
log.info("Kaydedildi: %s", OUT_PATH)
print(f"\nKaydedildi: {OUT_PATH}")
