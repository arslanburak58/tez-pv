## [27 Mayıs 2026] — STAGE-11 Streamlit Demo İlk Sürüm (Açık Problemli)

**>>> DEVAM KOMUTU: Yeni konuşmayı "STAGE-11 coverage debug" ile başlat <<<**

Aktif adım    : STAGE-11 (app/app.py çalışır durumda, coverage problemi var)
Aktif model   : Sonnet 4.6 / Extended Thinking: kapalı
Son güncelleme: 27 Mayıs 2026
Tıkanıklık    : DKASC coverage 10.1% — diagnostic 0.923 dediği halde app gösteremiyor

Kaynak (Projects için): https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/gunce.md

---

## Okuma listesi (Claude her konuşmada fetch eder)
- https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/stage_log.md
- https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/manifest.json
- https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/methodology_decisions.md

---

Tamamlanan (bu oturumda):
- app/app.py oluşturuldu — iki sekmeli Streamlit demo
- Sekme 1: DKASC Alice Springs test seti + sensör arıza simülasyonu sliderları
- Sekme 2: PVOD Station02 (Mono-Si, 17 MW, Hebei) — eğitime hiç girmemiş veri ile genelleme testi
- station02 seçim gerekçesi: tek Mono-Si istasyon, 38°N enlem (DKASC 23.7°S), %94 capacity factor

Düzeltilen hatalar:
- base_models key formatı: tuple (algo, q) değil, string ("lgbm_q01" vb.) — _col_name() ile çözüldü
- CQR k=2.0 v7 modelde ters çalışıyor — diagnostic gösterdi (0.923 → 0.258), kaldırıldı
- station02 power MB birimi — kW'a çevrildi (×1000)

Açık problemler (yarın çözülecek):

1. **DKASC coverage 10.1% — kritik**
   - Diagnostic koşumunda ilk 5000 satırda coverage 0.923 (CQR'sız)
   - App'te aynı pipeline ile coverage 10.1%
   - Pinball iyileşti (0.7122 → 0.6724) ama coverage kötüleşti
   - Görsel olarak bantlar dar, actual sabah/akşam ramplarında bant DIŞINDA
   - Hipotezler (yarın test edilecek):
     a) App ilk 4 günü gösteriyor — diagnostic 5000 satır (~17 gün) kullandı, alt küme farkı
     b) x_meta sütun sırası diagnostic ile app arasında farklı olabilir
     c) Streamlit cache eski model state'ini tutuyor olabilir
   - Eylem: app içinde aynı diagnostic'i koştur, çıktıyı app'inkiyle karşılaştır

2. **Station02 model tahminleri ~0**
   - Model DKASC skalasında (0-150 kW) eğitildi
   - Station02 17 MW plant — model 113x daha küçük sistem öğrendi
   - Normalize edince model tahminleri 0.009 max, actual 0.7 max
   - Bu beklenen zero-shot transfer limiti (rapor edilebilir bulgu)
   - Ama bant ÇOK dar (q01≈q50≈q09 hepsi 0 civarı)
   - Eylem: Pattern korelasyonunu ölç (model şekli yakalıyor mu) — tahmini ve actual'ı kendi tepelerine göre normalize edip karşılaştır

Sistem:
- claude.ai Projects → düşünme, yazım, karar
- Claude Code (terminal) → çalıştırma, git, dosya
- Makefile komutları: make help ile listele

Checkpoints (main):
- data/processed/meta_models_robust_v7.joblib  ← AKTİF
- data/processed/base_models.joblib            ← string keys (lgbm_q01 vb.)
- data/processed/dataset.joblib                ← X_test, y_test, feature_cols (15)
- app/app.py                                   ← STAGE-11 demo (coverage problemli)
