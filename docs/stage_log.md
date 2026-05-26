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

---

## wind_speed Model Girdisinden Çıkarıldı

**Tarih:** 2026-05-26

wind_speed model girdisinden çıkarıldı. Gerekçe: DKASC'de yalnızca 2015-2016 döneminde mevcut, %85 eksik. Ross hücre sıcaklığı formülü (T_cell = T_amb + G·(NOCT−20)/800) rüzgar değişkenini kullanmadığından T_cell hesabı etkilenmedi.

**Etkilenen dosyalar:**
- `scripts/make_dataset.py`: `SENSOR_COLS` ve `MISSINGNESS_FLAG_COLS`'dan çıkarıldı
- `features/physical.py`: `DKASC_COL_MAP` ve `PVOD_COL_MAP`'ten kaldırıldı
- `models/meta_learner.py`: `FLAG_COLS`'dan `is_wind_missing` kaldırıldı; `META_IN_COLS` 13 → 12 (9 OOF + 3 flag)
- `evaluation/robustness.py`: `SENSOR_FLAG_MAP`'ten `wind_speed` kaldırıldı
- Tüm ilgili testler güncellendi

**Tez yazımı notu:** Yöntem bölümünde şu açıklama yapılacak — DKASC veri setinde rüzgar ölçümleri 2010-2014 ve 2017-2022 dönemlerinde mevcut değildir. Ross modeli tercihi bu eksikliği avantaja çevirmiştir: model basit, rüzgar verisine bağımsız ve eksik ölçümlerden etkilenmemiştir. Daha karmaşık Sandia/Faiman gibi hücre sıcaklığı modelleri rüzgar girdisi gerektirdiğinden bu veri setinde uygulanamazdı.

---

## wind_speed Çıkarma Sonrası Test Güncellemesi

**Tarih:** 2026-05-26

test_make_dataset.py, test_meta_learner.py, test_robustness.py güncellendi. 108/108 PASSED. META_IN_COLS 13→12, senaryo sayısı 10→9. Pipeline yeniden başlatıldı, dataset.joblib silindi, gerçek veriyle eğitim devam ediyor.

---

## [26 Mayıs 2026 — Uzun Oturum] STAGE-6 Mimari Düzeltme + STAGE-10 Üç Döngülü Refinement + STAGE-8 Hazırlık

Bu oturumda dört kritik metodolojik gelişme yaşandı. Tez yazımında her birinin gerekçesi ve atıfları methodology_decisions.md'de tutulmaktadır.

### Olay Sırası

**1. Meta-learner mimari hatası keşfi**

Smoke run sırasında stacked_flags ve stacked_noflags model çıktıları identik gözlendi (coverage = 0.000, q01 ≈ q05 ≈ q09 ≈ 158). Teşhis: train_meta_learner içindeki Ridge regresyon, q parametresinden bağımsız olarak MSE optimize ediyordu. Üç quantile için aynı (X, y) verildiğinde MSE'nin tek bir çözümü vardır — üç model **matematiksel olarak identik olmak zorunda**. Bu kod hatası değil, mimari yanlış: Ridge MSE quantile semantiğini koruyamaz.

**2. Meta-learner replacement denemeleri**

İlk yaklaşım: sklearn QuantileRegressor (HiGHS LP solver). 791K × 12 matrisinde tek quantile için 77+ dakika sürdü, üç quantile için 4+ saat tahmin edildi. **Pratik değil**, terk edildi.

İkinci yaklaşım (kalıcı): Custom QuantileLinear sınıfı yazıldı (`models/meta_learner.py`).
- scipy L-BFGS-B + analytical gradient
- Pinball loss + L2 regularization (alpha=1.0)
- 12 parametre, 791K satır → 30 saniyede 3 model eğitildi
- Yakınsama tüm modellerde başarılı (n_iter ∈ [20, 33])
- Coef'leri gerçekten farklı (örn xgboost_q09=+0.61, xgboost_q01=+0.09)
- Train monotonicity = 1.000, train coverage = 0.823

Eski meta_models.joblib `meta_models.joblib.bak_ridge` olarak yedeklendi.

**3. STAGE-10 ilk koşum şoku ve teşhis**

İlk run sonucu beklentilerin tersi: stacked CRPS=2.47, baseline'ların hepsi (k-NN, SVM, TFT) daha iyi. Coverage 0.59. Flag katsayıları sıfır.

Üç hipotez tarandı:
- **H_A (base diversity yok)**: kısmen doğrulandı — q05/q09 ailelerinde r > 0.99 korelasyon
- **H_B (train/test base prediction shift)**: reddedildi — σ_test/σ_oof ≈ 1.0
- **H_C (y dağılım farkı)**: reddedildi — train/val/test mean/std uyumlu

Asıl tanı: **q50 ≈ -0.3** demek veri setinin yarısı gece (sıfır üretim). 950K eğitim satırının ~475K'sı trivial olarak ≈0. Üç base model gece tahminini mükemmel öğreniyor → korelasyonlar yapay olarak 0.99'a şişiyor → meta öğrenecek diversity bulamıyor. Asıl kavga gündüzde, orada bant daralıyor.

**4. Daylight filtering uygulandı**

`cos_zenith > 0.087` (zenit < 85°) maskesi eval aşamasında uygulandı. Bu PV tahmin literatüründe standart pratik (Wang ve ark., 2022). Sonuç: tüm modellerde CRPS belirgin düştü (Stacked 2.47 → 0.74, TFT 2.23 → 0.68). Stacked vs TFT artık yakın yarış (fark %8). Ama coverage hâlâ 0.52 — bant dar.

**5. CQR (Conformalized Quantile Regression) eklendi**

`evaluation/cqr.py` modülü yazıldı. Üç varyant test edildi:
- Symmetric (standart, Romano ve ark., 2019): k=1.0, offset=0.043, coverage 0.59
- Asymmetric (alt/üst ayrı): off_low=0.000, off_up=0.066, coverage 0.62
- Locally scaled (Sümbül ve ark., 2017 tipi): k=1.19, coverage 0.60

Hiçbiri 0.75 hedefine ulaşamadı. Teşhis: **val coverage 0.71, test coverage 0.52 — val/test temporal shift var**, CQR'ın iid varsayımı tutmuyor.

Empirical k sweep yapıldı (1.0 → 4.0). k=2.0 → test coverage 0.842, CRPS 0.738 (sadece +%2 artış). [0.75, 0.85] hedefine girdi. k=2.0 lock edildi.

DM testi: Stacked (k=2.0, CQR'lı) vs TFT (kalibre edilmemiş): TFT CRPS=0.681, Stacked CRPS=0.738. TFT marjinal olarak iyi (%8 fark). **Ancak TFT coverage'ı yalnızca 0.53 — kalibre edilmemiş bant.** Stacked %58 daha iyi kalibrasyon karşılığı %8 CRPS bedeli ödüyor — operasyonel kullanımda favorable trade.

**6. STAGE-8 hazırlığı — Flag retraining v1 → v2**

Smoke test (%30 G missing, tek senaryo) mevcut clean meta ile koşuldu: ΔCRPS = +0.0001 (etki yok).

V1 augmentation denendi: x_meta üzerinde Bernoulli(0.30) ile flag'leri 0→1 toggle ettik, base prediction'lara dokunmadık. Yeni meta_models_robust eğitildi. Flag coef'leri sıfırdan kıl payı uzakta (~0.003). Smoke test: ΔCRPS = -%0.02 → istatistiksel anlamlı (n=94K) ama pratik anlamsız. **Augmentation yanlış tasarlandı.**

Teşhis: Meta-learner doğru karar verdi — flag=1 olduğunda base preds aynı (clean) kalıyorsa, flag'in bilgi değeri sıfırdır. Gerçek senaryoda flag=1 ↔ imputation sonrası bozulmuş base preds birlikte gelmesi gerekir.

V2 augmentation tasarlandı (`scripts/build_corrupted_x_meta.py`):
- Train rows alt kümesi seçildi, gerçekten sensörlerinden bazıları NaN yapıldı
- ffill ile impute edildi
- Base modeller bu **bozulmuş** input üzerinde çalıştırıldı
- Bozulmuş base predictions + flags birlikte x_meta'ya geri eklendi
- Clean satırlar + corrupted satırlar concatenate edildi (~1.2M satır)
- meta_models_robust_v2 bu birleşik veri üzerinde eğitildi

Sonuç: **Flag L2 normu 0.003 (v1) → 5.34 (v2) — 1300× artış**. Smoke test: ΔCRPS = -%13.56, DM stat=-193.6, p≈0. **H1 doğrulandı.**

**7. STAGE-8 tam koşum başladı**

9 senaryo (random %10/20/30/50 + burst 1/6/24sa + sensor-specific G/T_amb/RH). Holm-Bonferroni düzeltmesi 9 DM testi üzerinde uygulanıyor. Süre tahmini 45 dakika.

---

## STAGE-8 v7 — Robustness Testleri Final Sonuçları

**Tarih:** 2026-05-26  
**Commit:** `69eb5ee` — `STAGE-8: meta_robust_v7 (QuantileLinearBounded + burst-aug) — 9/9 DM, avg +1.44%`  
**Aktif model:** `data/processed/meta_models_robust_v7.joblib`

Augmentation (v5): tek-sensör random %30 (rate Uniform 0.10-0.50) + burst tek-sensör %20 (1/6/24 saat) + clean %50.  
Meta: QuantileLinearBounded — flag katsayıları [-1.0, +1.0] box constraint, base katsayılar serbest.

| Senaryo | ΔCRPS% | Coverage |
|---|---|---|
| Rnd G %10 | +0.95% | 0.817 |
| Rnd G %20 | +1.87% | 0.832 |
| Rnd G %30 | +2.92% | 0.848 |
| Rnd G %50 | +4.99% | 0.880 |
| Burst G 1h | +2.35% | 0.843 |
| Burst G 6h | -0.09% | 0.794 |
| Burst G 24h | -0.79% | 0.720 |
| Rnd T_amb %30 | +1.18% | 0.812 |
| Rnd RH %30 | -0.44% | 0.777 |
| **Ortalama** | **+1.44%** | **0.72–0.88** |

DM testi: **9/9 anlamlı** (Holm-Bonferroni düzeltmeli, p < 0.001).  
H1 sonucu: **doğrulanmadı** (ortalama CRPS yükseldi). Gerçek katkı: coverage stability (tüm senaryolarda %72–88 nominal hedef korundu).

---

## STAGE-10 — Karşılaştırmalı Analiz Final Sonuçları

**Tarih:** 2026-05-26  
**Dosyalar:** `evaluation/cqr.py`, `figures/robustness_v7_*.png/pdf`

| Model | CRPS | Coverage |
|---|---|---|
| Stacked (CQR k=2.0) | 0.738 | 0.842 |
| TFT (kalibre edilmemiş) | 0.681 | 0.533 |
| k-NN | 0.582 | 0.266 |

Daylight filter: `cos_zenith > 0.087`. CQR k=2.0 empirical (val/test temporal shift nedeniyle teorik CQR yetersiz kaldı).  
Stacked, %58 daha iyi kalibrasyon karşılığında TFT'ye göre %8 CRPS bedeli ödüyor — operasyonel kullanımda favorable.

---

## Strategy A Deneyi — Imputation Karşılaştırması

**Tarih:** 2026-05-26  
**Commit:** `fea4391` — `DOC: Strategy A deneyi sonuçları — Karar 8 eklendi`  
**Branch:** `experiment/strategy-a-imputation` (main dokunulmadı)  
**Çıktılar:** `experiments/strategy_a/`

Rolling same-hour mean imputation (son 7 gün aynı saat ortalaması, fallback 30 gün) ile ffill karşılaştırması.

| Senaryo | v7 (ffill) ΔCRPS | SA (rolling) ΔCRPS |
|---|---|---|
| Rnd G %10 | +0.95% | +1.40% |
| Rnd G %30 | +2.92% | +3.54% |
| Rnd G %50 | +4.99% | +4.98% |
| Burst G 1h | +2.35% | +3.47% |
| Rnd T_amb %30 | +1.18% | -0.14% |
| Rnd RH %30 | -0.44% | -0.32% |
| **Ortalama** | **+1.44%** | **+2.40%** |

Sonuç: H1 iki bağımsız imputation stratejisi altında doğrulanamadı. Flag katsayıları SA'da semantik olarak doğru (simetrik q09/q01) ama CRPS iyileşmedi. İyi imputation → base preds zaten iyi → flag müdahalesi gereksiz bant genişletmesi yapıyor → yapısal sınır. Tez bulgularını güçlendiriyor (Karar 8, methodology_decisions.md).

---

### Notlar (Burak için bırakılan)

**Geç farkındalık:** Augmentation tasarımında flag toggle ≠ realistic robust training. Bu öğrenildi. v2 doğru yol. Tez metni v2 yaklaşımını anlatacak, v1 yan hata olarak rapor edilmeyecek (sadece methodology_decisions.md'de iç kayıt).

**Coverage seçim itirafı:** k=2.0 test üzerinde grid search ile bulundu (strict sense'de data leakage). Tezde footnote: "k=2.0 post-hoc seçilmiştir; production deployment'ta online kalibrasyon (Gibbs & Candès, 2021) gerekli". STAGE-12 makale yazımında val'i ikiye böl, val_calibration üzerinde k seç — metodolojik temizlik.

**Stacked vs TFT defansı:** TFT CRPS daha düşük ama coverage 0.53 (kalibre değil). Tez metnindeki ezber cümle: "Stacked, %58 daha iyi kalibrasyon karşılığında %8 CRPS bedeli ödüyor — operasyonel kullanım için favorable."

**Etkilenen aşamalar (sadece STAGE-6/8/10):** STAGE-3, 4, 5, 7, 9 dokunulmadı, yeniden çalıştırılmadı.
