# TEZ: Fotovoltaik Sistemlerde Olasılıksal Güç Tahmini
**Öğrenci:** Burak Arslan | **Danışman:** Doç. Dr. Kenan Altun
**Kurum:** Sivas Cumhuriyet Üniversitesi · Fen Bilimleri Enstitüsü · YZ ve Veri Bilimi ABD

---

## ► İLK İŞ — her konuşmada
`docs/gunce.md` dosyasını oku. Aktif aşama, bekleyen görevler orada.

---

## MİMARİ (değişmez referans)

### Model
- **Seviye 0:** XGBoost + LightGBM + CatBoost × q={0.1, 0.5, 0.9} → 9 taban akış
- **Seviye 1:** Ridge × 3 meta-model | girdi: 9 OOF tahmin + 4 missingness flag = 13 özellik
- **Missingness flags (meta-katmanda):** is_G_missing, is_Tamb_missing, is_RH_missing, is_wind_missing

### Fiziksel öznitelikler (pvlib tabanlı)
- cos(θ_z), saat-açısı, air mass, k_t = G/G₀, T_cell = T_amb + G·(NOCT-20)/800
- hour_sin/cos, month_sin/cos (döngüsel kodlama)

### Optimizasyon
- Optuna TPE + MedianPruner | objective: (L_0.1 + L_0.5 + L_0.9) / 3 validation'da

### Robustness testleri (3 eksen)
- Rastgele kayıp: %10 / %25 / %50
- Burst kayıp: 1 / 6 / 24 saat
- Sensör-özgü: yalnızca G / yalnızca T_amb / yalnızca RH

### Veri
- DKASC Alice Springs (2010–2022, saatlik) — birincil
- PVOD v1.0 (2019–2020, 10 istasyon, 271.968 kayıt, 15 dk) — birincil
- AEMO — DKASC için B planı yedek
- NREL NSRDB — meteorolojik yedek (pvlib simülasyonu gerektirir)

### Bölme & doğrulama
- Kronolojik 70/15/15 (shuffle=False, data leakage yasak)
- Walk-Forward Validation: TimeSeriesSplit(gap=24)

### Metrikler
- Deterministik: MAE, RMSE (ŷ_0.5 üzerinden)
- Olasılıksal: Pinball Loss, CRPS (Gneiting & Raftery, 2007 JASA)

### Baseline
- k-NN regressor, SVM (RBF kernel), LSTM (PyTorch MPS, 2 katman), hafif TFT

### Donanım
- Birincil: MacBook Air M4 (MPS backend)
- Yedek: Google Colab T4 (LSTM/Transformer için)

### Hipotez
Missingness flags → meta-öğreniciye eklenince CRPS istatistiksel anlamlı düşer
(flags'sız referans modelle karşılaştırma, Diebold-Mariano testi)

---

## AŞAMA DURUMU

| Stage | Adım | Durum |
|-------|------|-------|
| S0 | Setup | ✓ tamamlandı |
| S1 | Literatür sindirimi | ✓ neredeyse (günce'ye bak) |
| S2 | Veri temini | ⏳ |
| S3 | Fiziksel öznitelik pipeline | ⏳ |
| S4 | Veri bölme + Walk-Forward | ⏳ |
| S5 | Taban öğreniciler (9 model) | ⏳ |
| S6 | Meta-öğrenici + missingness flags | ⏳ |
| S7 | Optuna optimizasyon | ⏳ |
| S8 | Robustness testleri | ⏳ |
| S9 | Baseline modeller | ⏳ |
| S10 | Karşılaştırmalı analiz | ⏳ |
| S11 | Streamlit demo | ⏳ |
| S12 | Tez yazımı | ⏳ |
| S13 | Makale taslağı | ⏳ |

---

## KODLAMA KURALLARI

### Reprodüksiyon (her .py dosyasının başında zorunlu)
```python
import random
import numpy as np
random.seed(42)
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.mps.manual_seed(42)  # Apple Silicon M4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Zorunlu kurallar
- Python 3.11+, tip ipuçları zorunlu (mypy uyumlu)
- **Walk-Forward zorunlu** — random K-Fold zaman serisi için yasak
- **Scaler/imputer SADECE train setinde fit** — val/test'e sadece transform
- **Serileştirme: joblib** — pickle yasak
- **Data leakage**: her stage sonunda `make leakage` çalıştır
- Modüler yapı: features/ models/ evaluation/ robustness/ (tek dosyada 500+ satır yasak)
- Magic number yasak — sabitleri config dict veya dosya başında tanımla

### Yasak davranışlar
- `from X import *`
- Random K-Fold (zaman serisi için)
- Test setinde hyperparameter ayarı
- pickle.dump/load (joblib kullan)
- GPU varsayımı (M4 MPS var ama tree modelleri GPU gerektirmez)

---

## COMMIT KURALI

```
STAGE-N: [ne yapıldı, ölçülebilir]
FIX: [ne düzeltildi]
DOC: [ne güncellendi]

Örnekler:
  STAGE-3: pvlib pipeline birim testli, cos_zenith/kt/T_cell üretiliyor
  STAGE-5: 9 taban model eğitildi, OOF pinball raporlandı
  FIX: KNNImputer train-only fit düzeltildi (leakage)
  DOC: gunce.md STAGE-5 tamamlandı güncellendi
```

---

## DOSYA YAPISI

```
tez-pv/
├── CLAUDE.md              ← bu dosya (Claude Code bağlamı)
├── Makefile               ← otomasyon komutları
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               ← asla commit etme
│   └── processed/
├── features/
│   ├── __init__.py
│   └── physical.py        ← STAGE-3: pvlib pipeline
├── models/
│   ├── __init__.py
│   ├── base_learners.py   ← STAGE-5: 9 taban model
│   ├── meta_learner.py    ← STAGE-6: Ridge + flags
│   └── checkpoints/       ← .joblib dosyaları (gitignore'da)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py         ← MAE, RMSE, Pinball, CRPS
├── robustness/
│   ├── __init__.py
│   └── scenarios.py       ← STAGE-8: 3 eksen × 9 senaryo
├── scripts/
│   ├── commit_stage.sh    ← git commit otomasyon
│   └── check_leakage.py   ← leakage kontrol
├── app/
│   ├── __init__.py
│   └── app.py             ← STAGE-11: Streamlit demo
├── figures/               ← STAGE-10: PNG + PDF çıktılar
├── notebooks/             ← EDA ve keşif
├── tests/                 ← birim testler
└── docs/
    ├── gunce.md           ← EN ÖNEMLİ — her zaman güncel tut
    ├── tez_workflow.md    ← aşama rehberi
    └── literatur_ozeti.md ← referans haritası
```

---

## CLAUDE CODE GÖREV ALANI

Claude Code (terminal) şunları yapar:
- `docs/gunce.md` okuma ve güncelleme
- `git add + commit` (kurallı mesajla)
- Python pipeline çalıştırma ve debug
- Optuna trial yönetimi
- Dosya oluşturma ve düzenleme
- `make` komutlarını çalıştırma

## claude.ai PROJECTS GÖREV ALANI

claude.ai Projects (bu ekran) şunları yapar:
- Literatür sentezi ve yazım
- Yöntem bölümü ve tez bölümleri
- Mimari ve metodoloji kararları
- Danışman stratejisi ve toplantı hazırlığı
- Eleştirel inceleme (skill_reviewer.md)

---

## GÜNCE SENKRONIZASYON

`gunce.md` Google Drive'da yaşıyor. File ID: `1Fy9BiiRBFqKynGKU4Mn7jS1REVytlO3_`

Günceyi güncellemek için:
1. `docs/gunce.md` dosyasını düzenle
2. İçeriği Drive'daki dosyaya da yaz (Google Drive MCP ile)
3. `git commit -m "DOC: gunce.md güncellendi"`

---

Versiyon: Mayıs 2026
