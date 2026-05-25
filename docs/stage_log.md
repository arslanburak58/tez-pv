# Stage Log

Her aşamanın tamamlanma kaydı. Yeni aşama bitince buraya append et.

---

## STAGE-3 — Fiziksel Öznitelik Pipeline

**Tarih:** 2026-05-25  
**Commit:** `0feb316` — `STAGE-3: pvlib pipeline birim testli, cos_zenith/kt/T_cell/hour_angle/air_mass üretiliyor`  
**Dosyalar:** `features/physical.py`, `tests/test_physical.py`

Üretilen öznitelikler:
- `cos_zenith`, `hour_angle`, `air_mass` — pvlib tabanlı
- `k_t = G / G₀` — clearness index, gece → 0
- `T_cell = T_amb + G·(NOCT-20)/800` — Ross modeli, NOCT=46°C
- `hour_sin/cos`, `month_sin/cos` — döngüsel kodlama

Test sonucu: **24/24 PASSED**

---

## STAGE-4 — Veri Bölme + Walk-Forward İskeleti

**Tarih:** 2026-05-26  
**Commit:** `d60fb36` — `STAGE-4: make_dataset.py — kronolojik 70/15/15, walk-forward iskelet, leakage-safe imputer`  
**Dosyalar:** `scripts/make_dataset.py`, `tests/test_make_dataset.py`

Uygulanan adımlar (sıra leakage-safe):
1. Missingness flags → ham veriden, imputasyondan önce
2. Kronolojik 70/15/15 bölme (shuffle=False)
3. Kısa boşluk interpolasyonu ≤ 3 saat, lineer
4. KNNImputer — sadece train'de fit, val/test'e transform
5. Fiziksel öznitelikler → imputasyondan sonra
6. Walk-Forward: `TimeSeriesSplit(n_splits=5, gap=24)`

Test sonucu: **16/16 PASSED**

---

## STAGE-6 — Meta-öğrenici + Missingness Flags

**Tarih:** 2026-05-26  
**Commit:** `fa1ddd4` — `STAGE-6: Ridge × 3 meta-öğrenici + missingness flags (13 özellik)`  
**Dosyalar:** `models/meta_learner.py`, `tests/test_meta_learner.py`, `models/base_learners.py` (güncellendi)

Mimari:
- `enrich_x_meta(X_meta, flags)`: 9 OOF + 4 missingness flag = 13 özellik; `flags.reindex(X_meta.index)` ile güvenli hizalama
- `Ridge × 3`: q=0.1/0.5/0.9 ayrı Ridge; alpha=1.0 (STAGE-7'de Optuna)
- `predict_intervals()`: DataFrame alır, tip kaybı yok
- `coverage_score()`: %10–%90 nominal bant; hedef ≈ 0.80
- `compare_baseline()`: stacked pinball vs tek lgbm OOF pinball, % iyileşme

`base_learners.py` güncellemeleri:
- `make_oof_predictions`: `np.asarray` kaldırıldı → DataFrame korunur, LightGBM sütun adlarını takip eder, `X.iloc[tr_idx]` ile indeksleme
- `build_x_meta`: `index=idx` eklendi → orijinal index meta-katman hizalamasında kullanılabilir

Test sonucu: **31/31 PASSED** (meta_learner) + **35/35 PASSED** (base_learners) = **66/66 toplam**

---

## STAGE-5 — Taban Öğreniciler (9 Model)

**Tarih:** 2026-05-26  
**Commit:** `542d9e4` — `STAGE-5: 9 taban model eğitildi, OOF pinball raporlandı`  
**Dosyalar:** `models/base_learners.py`, `tests/test_base_learners.py`

Modeller:
- LightGBM × 3: `objective="quantile", alpha=q`
- CatBoost × 3: `loss_function=f"Quantile:alpha={q}"`
- XGBoost × 3: custom pinball objective (grad/hess elle yazılmış) → `_XGBWrapper`

OOF: `TimeSeriesSplit(n_splits=5, gap=24)` — başlangıç satırları NaN, `build_x_meta` içinde atılır  
Çıktı: `X_meta` (n_oof × 9), `oof_scores` dict  
Serileştirme: `joblib` (pickle yasak)

Düzeltilen hata: `build_x_meta` içinde `make_oof_predictions`'a `algo` yerine `X_train` geçiliyordu.

Test sonucu: **35/35 PASSED**
