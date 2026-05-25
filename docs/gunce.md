## [Mayıs 2026] — STAGE-9 Tamamlandı

Aktif adım: STAGE-10 bekleniyor
Sıradaki konuşmada: "STAGE-10'a başlıyoruz" ile başlat
Aktif model    : Sonnet 4.6 / Extended Thinking: kapalı
Son güncelleme : Mayıs 2026
Tıkanıklık     : yok

Kaynak (Projects için): https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/gunce.md

Tamamlanan:
- S0 Setup ✓
- S1 Literatür ✓ (harita oturumu + literatur_ozeti.md düzeltmeleri)
- Proje ortamı ✓ (~/Desktop/tez-pv, git, venv, CLAUDE.md, Makefile)
- Claude Code ✓ (çalışıyor, bağlamı doğru okuyor)
- S2 EDA ✓ — docs/data_dictionary.md
  * DKASC: 1,361,812 satır, 196 sütun, 5 dk, 2010–2022, %21.4 eksik
  * PVOD: 271,968 satır, 10 istasyon, 15 dk, 2018–2019, %0.0 eksik
- S3 Fiziksel öznitelik pipeline ✓ — features/physical.py
  * Öznitelikler: cos_zenith, hour_angle, air_mass, k_t, T_cell, hour_sin/cos, month_sin/cos
  * 24/24 birim test geçti (tests/test_physical.py)
  * sklearn Pipeline + StandardScaler (train-only fit) hazır
  * DKASC_LOCATION ve DKASC_COL_MAP / PVOD_COL_MAP tanımlı
- S4 Veri bölme + Walk-Forward ✓ — scripts/make_dataset.py
  * Kronolojik 70/15/15, shuffle=False
  * Missingness flags → ham veriden (imputasyondan önce)
  * KNNImputer train-only fit; kısa boşluk (<3h) → lineer interpolasyon
  * Fiziksel öznitelikler imputasyondan SONRA hesaplanıyor (NaN yok)
  * TimeSeriesSplit(gap=24) Walk-Forward iskeleti
  * 16/16 birim test geçti (tests/test_make_dataset.py)
- S5 Taban öğreniciler ✓ — models/base_learners.py
  * LightGBM × 3, CatBoost × 3, XGBoost × 3 (quantile q=0.1/0.5/0.9)
  * XGBoost: custom pinball objective (grad/hess elle yazılmış)
  * OOF tahminleri → TimeSeriesSplit(n_splits=5, gap=24)
  * build_x_meta(): 9 OOF sütun → X_meta (NaN satırlar atılmış, orijinal index korunuyor)
  * train_all_base_learners(): 9 model eğit + joblib ile kaydet
  * 35/35 birim test geçti (tests/test_base_learners.py)
- S6 Meta-öğrenici + missingness flags ✓ — models/meta_learner.py
  * enrich_x_meta(): 9 OOF + 4 flag = 13 özellik, index hizalamalı
  * Ridge × 3 (q=0.1/0.5/0.9), alpha=1.0 (STAGE-7'de Optuna ile aranacak)
  * predict_intervals(): DataFrame alır, tip kaybı yok
  * coverage_score(): %10–%90 bant kapsama oranı
  * compare_baseline(): stacked vs tek LightGBM-quantile pinball karşılaştırması
  * LightGBM predict çağrıları DataFrame ile — sütun adları korunuyor
  * 31/31 birim test geçti (tests/test_meta_learner.py)
- S7 Optuna optimizasyon ✓ — optimization/optuna_search.py
  * TPESampler(seed=42) + MedianPruner(n_startup=5, warmup=1)
  * Arama uzayı: 3 algo × 7-8 param + Ridge alpha
  * Objective: mean val pinball (9 model) → minimize
  * MedianPruner için per-algo kümülatif kayıp bildirilir (step=0/1/2)
  * save_best_params() → best_params.json (algo başlıkları + _meta)
  * top_trials_summary() → en iyi N trial DataFrame
  * plot_study() → parallel coordinate + param importance (HTML)
  * 26/26 birim test geçti (tests/test_optuna_search.py)
  * Gerçek çalışma: 50 trial → en iyi #42 | val pinball: 2.9057 (%69.1 iyileşme)
  * Sonuç: docs/best_params.json
- S8 Robustness testleri ✓ — evaluation/robustness.py
  * RobustnessScenario dataclass + ALL_SCENARIOS (9 senaryo)
  * Rastgele kayıp: %10/%25/%50 → maskeleme + flag güncelleme
  * Burst kayıp: 1h/6h/24h → ardışık blok bozulma
  * Sensör-özgü: G/T_amb/RH → tam sütun sıfırlama
  * Türev özellik yönetimi: G bozulunca k_t=0, T_amb bozulunca T_cell=0
  * build_predict_fn(): flags/noflags iki model için closure
  * evaluate_predictions(): pinball_q01/q05/q09, crps, mae, rmse, coverage
  * run_all_scenarios(): {"flags", "noflags", "dm"} DataFrame'leri
  * diebold_mariano_test(): HLN düzeltmeli, t(n-1), iki yönlü
  * plot_heatmap(): 9 senaryo × 2 model ısı haritası (PNG)
  * 57/57 birim test geçti (tests/test_robustness.py)

Altyapı:
- Repo public ✓ (GitHub raw URL aktif)
- tez_workflow.md markdown'a çevrildi ✓ (AEMO/NREL B planı + git adımları eklendi)
- Projects custom instructions güncellendi ✓ (her iki raw URL + skill/model/token kuralları)

- S9 Baseline modeller ✓ — models/baselines.py
  * KNNQuantile: k komşu quantile (proper k-NN, kneighbors)
  * SVMQuantile: Nystroem + LinearSVR + split-conformal bantlar
  * LSTMQuantile: 2 katmanlı LSTM, pinball loss, erken durdurma, MPS
  * LightTFTQuantile: LSTM + multi-head attention + GRN, pytorch_forecasting YOK
  * make_sequences, evaluate_quantiles, train_all_baselines unified API
  * 43/43 birim test geçti (tests/test_baselines.py)

Açık görevler:
- STAGE-10: Karşılaştırmalı analiz
  * Master tablo: 6 model × (MAE, RMSE, Pinball, CRPS, Coverage, süre)
  * Heatmap: sensör × hata değişimi
  * Olasılıksal bant görselleştirme
  * Diebold-Mariano pairwise testleri (Holm-Bonferroni)
  * figures/ altına PNG + PDF

Sistem:
- claude.ai Projects → düşünme, yazım, karar
- Claude Code (terminal) → çalıştırma, git, dosya
- Makefile komutları: make help ile listele
