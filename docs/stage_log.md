# Stage Log

Her aşamanın tamamlanma kaydı. Yeni aşama bitince buraya append et.

---

## STAGE-0 — Proje Ortamı Kurulumu

**Tarih:** 2026-05-25  
**Commit:** `4986872` — `STAGE-0: proje ortamı kuruldu (klasörler, venv, CLAUDE.md, Makefile, scripts)`  
**Dosyalar:** `CLAUDE.md`, `Makefile`, `requirements.txt`, `.gitignore`, dizin yapısı

- tez-pv repo kuruldu, venv oluşturuldu
- CLAUDE.md bağlam dosyası yazıldı
- Makefile ile otomasyon komutları tanımlandı

---

## STAGE-1 — Literatür Sindirimi

**Tarih:** 2026-05-25  
**Commit:** `6739069` — `DOC: literatur_ozeti.md temizlendi — Arslan&Gemini silindi, 27 kayıt sıralandı`  
**Dosyalar:** `docs/literatur_ozeti.md`

- 27 referans sıralandı
- Hipotez netleşti: missingness flags → meta-katmana → CRPS düşer (DM testi)
- Kod üretimi yok; claude.ai Projects'te yürütüldü

---

## STAGE-2 — Keşifsel Veri Analizi (EDA)

**Tarih:** 2026-05-25  
**Commitler:**
- `6b2d63a` — `STAGE-2: DKASC EDA tamamlandı, PVOD eksik dosya tespit edildi`
- `21faed7` — `STAGE-2: PVOD EDA tamamlandı, data_dictionary.md güncellendi`  

**Dosyalar:** `docs/data_dictionary.md`, EDA notebookları

DKASC Alice Springs:
- 1,361,812 satır, 196 sütun, 5 dk, 2010–2022, %21.4 genel eksik
- Anomaliler: wind_speed %47.6 eksik, T<-10°C → 3047 satır, RH>%100 → 8549 satır

PVOD v1.0:
- 271,968 satır, 10 istasyon, 15 dk, 2018–2019, %0.0 eksik
- Kapasite: 6.6–35 MWp, Hebei Çin; 9 Poly-Si + 1 Mono-Si

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

---

## STAGE-7 — Optuna Hiperparametre Optimizasyonu

**Tarih:** 2026-05-26  
**Commitler:**
- `0ccba25` — `STAGE-7: Optuna TPE araması — 9 model val pinball objective, best_params.json` (unit testler)
- `c2ad81a` — `STAGE-7: Optuna 50-trial DKASC araması tamamlandı` (gerçek çalışma)  
**Dosyalar:** `optimization/optuna_search.py`, `optimization/__init__.py`, `tests/test_optuna_search.py`

Mimari:
- `suggest_base_params(trial, algo)`: SEARCH_SPACE'den prefix'li param adları önerir, prefix'siz dict döndürür
- `objective(trial, ...)`: 9 model val pinball ortalaması; her algo'dan sonra `trial.report()` (MedianPruner için)
- `run_study(...)`: TPESampler(seed=42) + MedianPruner(n_startup=5, warmup=1), in-memory veya SQLite
- `save_best_params(study, path)`: algo başlıkları + ridge_alpha + `_meta` bloğu → JSON
- `load_best_params(path)`: JSON → dict
- `top_trials_summary(study, n)`: en iyi N trial → pd.DataFrame (sıralı)
- `plot_study(study, save_dir)`: parallel coordinate + param importance → HTML

Arama uzayı: LightGBM 8 param, CatBoost 5 param, XGBoost 7 param, Ridge alpha — toplam 21 boyut  
Ridge alpha objective'e dahil edilmez (OOF olmadan meta-fit anlamsız); trial params'a kaydedilir.

Test sonucu: **26/26 PASSED**

Gerçek çalışma (DKASC 2015-2016):
- 50 trial → 26 tamamlandı, 24 pruned
- Başlangıç pinball: 9.4054 → En iyi: **2.9057** (%69.1 iyileşme)
- En iyi trial: #42 | Süre: 36.4 dakika
- Sonuç: `docs/best_params.json`

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

## STAGE-8 — Robustness Testleri

**Tarih:** 2026-05-26  
**Commit:** (bu commit)  
**Dosyalar:** `evaluation/robustness.py`, `tests/test_robustness.py`

3 eksen × 3 seviye = 9 senaryo:

Rastgele kayıp:
- `random_10pct` — %10 rastgele NaN maskeleme (tüm sensörler)
- `random_25pct` — %25
- `random_50pct` — %50

Burst kayıp:
- `burst_1h`  — 1 saat ardışık kesinti
- `burst_6h`  — 6 saat
- `burst_24h` — 24 saat

Sensör-özgü:
- `sensor_G`    — G sütunu tamamı sıfır
- `sensor_Tamb` — T_amb sütunu tamamı sıfır
- `sensor_RH`   — RH sütunu tamamı sıfır

Mimari:
- `_corrupt_columns`: sensör→0, flag→1, türev özellik sıfırlama (G→k_t, T_amb→T_cell)
- `apply_scenario`: random/burst/sensor eksenlerini uygular
- `build_predict_fn`: flags/noflags için closure
- `evaluate_predictions`: pinball×3, crps, mae, rmse, coverage
- `run_all_scenarios`: baseline + 9 senaryo → {"flags", "noflags", "dm"} DataFrame
- `diebold_mariano_test`: HLN düzeltmeli, t(n-1), iki yönlü; sıfır varyans guard
- `plot_heatmap`: 9 senaryo × 2 model PNG ısı haritası

Test sonucu: **57/57 PASSED**

---

## STAGE-9 — Baseline Modeller

**Tarih:** 2026-05-26  
**Commit:** (bu commit)  
**Dosyalar:** `models/baselines.py`, `tests/test_baselines.py`

4 baseline model, q={0.1, 0.5, 0.9} quantile çıktı:

- **KNNQuantile**: k komşunun y değerlerinden quantile (proper quantile k-NN, `kneighbors`)
- **SVMQuantile**: Nystroem(RBF, 500 bileşen) + LinearSVR + split-conformal bant (son %20 kalibrasyon)
- **LSTMQuantile**: 2 katmanlı LSTM, hidden=64, MPS/CPU, pinball loss, erken durdurma (patience=5)
- **LightTFTQuantile**: doğrusal gömme → LSTM → multi-head attention → GRN → 3 çıktı; `pytorch_forecasting` gerektirmez

Yardımcı:
- `make_sequences(X, y, seq_len)` → dizi pipeline
- `evaluate_quantiles(y, preds)` → 7 metrik (pinball×3, crps, mae, rmse, coverage)
- `train_baseline / predict_baseline / evaluate_baseline` — unified API
- `BASELINE_REGISTRY` — {"knn", "svm", "lstm", "tft"}
- `train_all_baselines` — 4 modeli sırayla eğit + değerlendir

Test sonucu: **43/43 PASSED**

---

## STAGE-4 Revizyon — KNNImputer → ffill/bfill

**Tarih:** 2026-05-26

**Gerekçe:** Gerçek DKASC verisiyle (`950K × 4`) eğitim pipeline'ı çalıştırılırken `KNNImputer.fit_transform` 42+ dakika sonra hâlâ bitmemişti. sklearn KNNImputer `nan_euclidean_distances` ile pairwise mesafe matrisi hesapladığından O(n²) zaman karmaşıklığı gösterir; büyük veri setlerinde pratik değil.

**Yapılan değişiklik:** `scripts/make_dataset.py`'e `imputer_strategy: str = "ffill"` parametresi eklendi.
- `"ffill"`: pandas `ffill().bfill()` — stateless, O(n), varsayılan
- `"median"`: `SimpleImputer(strategy='median')` — train'de fit, O(n)
- `"knn"`: orijinal `KNNImputer` — korundu, küçük veri setleri için

**Tez yazımında:** Yöntem bölümünde şu şekilde açıklanacak:
> "Kısa boşluklar (≤ 3 saat) doğrusal interpolasyonla dolduruldu. Daha uzun boşluklar için ileriye/geriye doldurma (ffill/bfill) stratejisi kullanıldı; bu yaklaşım zaman serisi verisinde komşu gözlemlerin en bilgilendirici kaynak olduğu varsayımıyla tutarlıdır (Little & Rubin, 2002). Missingness flags ayrıca meta-katmana aktarıldığından imputation kalitesi nihai modeli ikincil olarak etkiler."

**Test sonucu:** **19/19 PASSED** (3 yeni test dahil: median strateji, geçersiz strateji, tüm stratejiler NaN-free)
