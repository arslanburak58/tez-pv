## [Mayıs 2026] — STAGE-8 v7 Tamamlandı

**>>> DEVAM KOMUTU: Yeni konuşmayı "STAGE-11 Streamlit demo" ile başlat <<<**

Aktif adım    : STAGE-11 Streamlit demo (app/app.py)
Aktif model   : Sonnet 4.6 / Extended Thinking: kapalı
Son güncelleme: 26 Mayıs 2026
Tıkanıklık    : Yok

Kaynak (Projects için): https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/gunce.md

---

## Okuma listesi (Claude her konuşmada fetch eder)
- https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/stage_log.md
- https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/manifest.json
- https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/methodology_decisions.md

---

Tamamlanan (bu oturumda):
- S0–S9 zaten tamamdı (önceki oturum)
- STAGE-6 REVİZE — Ridge MSE → QuantileLinear (scipy L-BFGS-B + pinball + L2)
- STAGE-10 daylight + CQR k=2.0, stacked CRPS=0.738, coverage=0.842
- meta_robust_v2 → v3 → v4 → v5/v6/v7 deney serisi (kök neden analizi)
- STAGE-8 v7 ile tamamlandı (9/9 DM anlamlı, avg ΔCRPS=+1.44%)
  * meta_models_robust_v7.joblib aktif model
  * v7: v5 aug (burst+random) + QuantileLinearBounded (flag_bound=1.0)
  * Flag katsayıları: q09_G=+0.11, q01_G=-1.0 (bounded)
  * Coverage 0.72–0.88 bandında, 9/9 DM p<0.05

STAGE-8 final özeti (v7):
- Rnd G %10: +0.95%  | Rnd G %20: +1.87%  | Rnd G %30: +2.92%  | Rnd G %50: +4.99%
- Burst G 1h: +2.35% | Burst G 6h: -0.09% | Burst G 24h: -0.79%
- Rnd T_amb %30: +1.18% | Rnd RH %30: -0.44%
- Ortalama: +1.44% (v2: +6.2%), flags iyileştirdi: 3/9

Açık görevler:
- STAGE-11: Streamlit demo (app/app.py)
  * Günlük tahmin görselleştirme (q01/q05/q09 bantlar)
  * Sensör bayrak simülasyonu (interaktif slider)
  * v7 modeli kullan
- STAGE-12: Tez yazımı (Yöntem + Bulgular bölümleri)
- STAGE-13: SCI/SCI-E makale taslağı

Metodoloji notu (savunmaya hazır):
- v7 = "corruption-aware training (v5: burst+random aug) + QuantileLinearBounded (flag_bound=1.0)"
- q09 bounded [-1, +1] → flag katsayısı artık anlamlı semantik: G eksik → q01 düşüyor, q09 hafif artıyor (uncertainty genişlemesi)
- H1 yorumu: Flags CRPS'i küçük miktarda artırıyor (+1.44%) ama 9/9 DM anlamlı → flags bilgi taşıyor. Coverage tutarlı iyileşiyor.

Checkpoints:
- data/processed/meta_models_robust_v7.joblib  ← AKTİF
- data/processed/meta_models_robust_v2.joblib  ← referans (eski)
- figures/robustness_v7_*.png/pdf              ← STAGE-8 görselleri
- figures/robustness_*.png/pdf                 ← v2 görselleri (saklandı)

Sistem:
- claude.ai Projects → düşünme, yazım, karar
- Claude Code (terminal) → çalıştırma, git, dosya
- Makefile komutları: make help ile listele
