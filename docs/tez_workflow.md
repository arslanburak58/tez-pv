# Tez Workflow Rehberi — Adım Adım

**Tez:** Fotovoltaik Sistemlerde Olasılıksal Güç Tahmini — Sensör Kayıplarına Dayanıklı Yaklaşım  
**Öğrenci:** Burak Arslan · **Danışman:** Doç. Dr. Kenan Altun  
**Kurum:** Sivas Cumhuriyet Üniversitesi · FBE · YZ ve Veri Bilimi ABD

---

## 0. Bu dosya nasıl çalışır

Bu dosya sıralı bir rehber. 14 aşama var. Süre kısıtı yok — kendi hızında ilerle.

**Kurallar:**

1. `docs/gunce.md` aktif aşamayı, bekleyen görevleri ve güncel durumu tutar. Claude her konuşmada önce onu okur.
2. Aşamalar sıralı ama esnek. Bir aşama haftalarca sürebilir, başkası bir gün. Tıkanırsan dur, danışmana sor, sonra devam et.
3. Ekstra sorular serbest. Tez dışı bir şey takılırsa Claude cevaplar.
4. Bir aşamayı bitirdiğinde "Tamamlandı ölçütü"nü kendinle yap. Karşılayamadıysan aşamada kalmaya devam et.
5. Bu dosya yaşıyor. Plan değiştikçe sen güncelle.

---

## 1. Mevcut Durum

Aktif durum `docs/gunce.md` dosyasında tutulur.  
GitHub raw URL (Projects için): `https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/gunce.md`

Günceyi güncellemek için:
1. `docs/gunce.md` dosyasını düzenle
2. `git commit -m "DOC: gunce.md güncellendi"`
3. `git push`

---

## 2. Tezin özü (hızlı referans — değişmez)

**Problem:** PV güç tahmini deterministik nokta tahmininde takılı; fiziksel kısıt yok, sensör arızası altında davranış sistematik ölçülmemiş.

**Yaklaşım:** Fizik kısıtlı + olasılıksal hibrit Stacked Ensemble.

- **Seviye 0:** XGBoost + LightGBM + CatBoost × q={0.1, 0.5, 0.9} → 9 taban akış.
- **Seviye 1:** Ridge × 3 quantile meta-model. Özellik uzayı: 9 taban tahmin + 4 missingness flag.
- **Fiziksel öznitelikler (pvlib):** cos(θ_z), saat-açısı, hava kütlesi, k_t = G/G_0, T_cell (Ross).
- **Optimizasyon:** Optuna TPE + MedianPruner. Objective: ortalama pinball loss.
- **Dayanıklılık:** Rastgele (%10/%25/%50), burst (1/6/24 saat), sensör-özgü (G/T_amb/RH).
- **Veri:** DKASC Alice Springs (2010–2022) + PVOD v1.0 (2019–2020). Kronolojik 70/15/15 + Walk-Forward.
- **Metrikler:** MAE, RMSE, Pinball Loss, CRPS.
- **Baseline:** k-NN, SVM, LSTM, hafif TFT.
- **Donanım:** MacBook Air M4 (MPS), gerekirse Colab T4.

**Dört özgün katkı:** (i) fizik + olasılık + stacking entegrasyonu, (ii) üç eksenli sistematik dayanıklılık çerçevesi, (iii) prediction-odaklı missingness flags (Sperrin et al. 2020), (iv) edge AI uyumluluğu.

**Hipotez:** Missingness flags meta-öğreniciye eklenince CRPS, indicator'sız referansa kıyasla istatistiksel anlamlı düşer.

---

## 3. Claude Projects setup (tek seferlik kurulum)

**Adımlar:**

1. claude.ai → sol menü → Projects → New Project
2. Ad: "YL Tezi — PV Olasılıksal Tahmin"
3. Aşağıdaki metni **Custom instructions** alanına yapıştır:

```
Ben Burak Arslan, Sivas Cumhuriyet Üniversitesi YZ ve Veri Bilimi yüksek lisans
öğrencisiyim. Danışmanım Doç. Dr. Kenan Altun.

Tez konum: Fotovoltaik sistemlerde olasılıksal güç tahmini — sensör kayıplarına
dayanıklı yaklaşım. Fizik kısıtlı + olasılıksal hibrit Stacked Ensemble
(XGBoost+LightGBM+CatBoost × q=0.1,0.5,0.9; Ridge meta-öğrenici; missingness
flags yalnızca meta-katmanda). Veri: DKASC + PVOD v1.0. Metrikler: MAE/RMSE +
Pinball Loss/CRPS. Baseline: k-NN, SVM, LSTM, hafif Transformer. Donanım:
MacBook Air M4 (MPS backend), gerekirse Colab T4.

─── KONUŞMA BAŞINDA OKU ───────────────────────────────────────────────────────

Her yeni konuşmaya başladığında ilk işin şu iki URL'yi okumak:

1. Günce (aktif durum):
   https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/gunce.md
   → Hangi aşamadayım, hangi adımdayım, bekleyen notum var mı.

2. Workflow rehberi (aşama detayları):
   https://raw.githubusercontent.com/arslanburak58/tez-pv/main/docs/tez_workflow.md
   → Aktif aşamanın adımlarını, tamamlandı ölçütlerini buradan al.

Cevabını bu iki dosyanın içeriğine göre konumlandır.

─── SKILL DOSYALARI ───────────────────────────────────────────────────────────

Şu durumlarda ilgili skill dosyasını oku ve oradaki kurallara göre cevapla:
- "eleştir / incele / değerlendir / review et" → skill_reviewer.md
- "yöntem bölümüne yaz / formülü anlat / metodolojik açıkla" → skill_methods_writer.md
- "literatürü sentezle / makaleleri karşılaştır / araştırma boşluğunu ifade et" → skill_lit_synth.md
- Python kodu, debug, pipeline tasarımı, kütüphane sorusu → skill_code_helper.md
Şüphe varsa "şu skill'i devreye alıyorum" diye kısaca belirt, sonra cevaba geç.

─── MODEL VE EXTENDED THINKING ────────────────────────────────────────────────

Default: Sonnet 4.6, Extended Thinking KAPALI.
Aşağıdaki durumlarda cevap vermeden önce uyarı bloğu yaz:

🔺 Opus 4.7 öner:
- Yöntem mimarisi kararı (meta-öğrenici tipi, radiation modeli seçimi vb.)
- Hipotez yorumu veya doğrulama/çürütme kararı
- Kritik özgün katkı paragrafının İLK yazımı (revizyon değil)
- Atiea ve ark. (2025) fark konumlandırması yazımı
- İstatistiksel test seçimi (DM vs Wilcoxon, multiple comparison correction)
- Matematiksel türetim (pinball gradient/hessian vb.)
- Danışman geri bildirimi stratejik yorumu
- skill_reviewer.md aktifken eleştirel inceleme
- Sonuç bölümü ve makale abstract/introduction İLK yazımı

🔻 Extended Thinking aç:
- Çok adımlı debug zinciri (3+ olası neden)
- Veri leakage tespiti (pipeline boyunca sistematik kontrol)
- Optuna sonuçları yorumu (hyperparameter analizi, overfitting riski)
- Robustness'ta beklenmedik bulgu ("neden CRPS düştü, beklenenin tersi")
- İstatistiksel test uygulaması ve yorumu
- 3+ makale karşılaştırmalı sentezi
- Karmaşık trade-off analizi (M4 sınırı, model kapsamı kararları)
- Veri ön işleme strateji tasarımı

🔺🔻 Her ikisi birden (en pahalı, sadece kritik kararlar için):
- Tezin yöntem omurgasını oluşturan tasarım kararları
- Hipotez doğrulanmadığında konumlandırma stratejisi
- Jüri savunması için zayıf nokta analizi
- Hakem yorumlarına strateji belirleme

Uyarı YAPMA (Sonnet + kapalı yeterli):
- Tek paragraf revizyonu, üslup düzeltme
- APA format/atıf düzenlemesi
- Basit Python sözdizimi soruları
- Streamlit UI tasarım soruları
- Şekil/tablo numaralandırma
- gunce.md özet yazımı
- Tez dışı sorular

Uyarı formatı (cevaptan ÖNCE):
> 🔺 Model önerisi: Bu soru için Opus 4.7 öneriyorum. [Tek cümle sebep.]
> Modeli değiştirip tekrar sormak istersen bekleyeyim; devam dersen Sonnet ile gidiyorum.

> 🔻 Extended Thinking önerisi: [Tek cümle sebep.]
> Açıp tekrar sormak istersen bekleyeyim; devam dersen kapalı gidiyorum.

> 🔺🔻 Her ikisini de öneriyorum. [Tek cümle sebep.]

─── TOKEN / KOTA UYARISI ──────────────────────────────────────────────────────

Konuşma uzadıkça context window dolmaya başlar. Şu eşiklerde uyar:

Doluluk ~%70 olduğunda:
> ⚠️ KOTA UYARISI: Bu konuşma uzuyor, context dolmaya başlıyor.
> Önemli bir karar veya yazım varsa yeni konuşmada devam etmeni öneririm.
> Şu an için: günce güncellemesini hazırlayayım mı?

Doluluk ~%90 olduğunda:
> 🔴 SON UYARI: Context neredeyse dolu. Bu mesajdan sonra yeni konuşma aç.
> Günce ve dosya güncellemelerini şimdi hazırlıyorum.
> [Günce bloğunu ve varsa tez_workflow.md güncellemesini hemen ver]

Uyarı sonrası davranış:
- "Günce güncellemesini hazırla" istenirse: kopyala-yapıştır hazır blok ver,
  hangi dosyaya gideceğini belirt (gunce.md veya tez_workflow.md).
- tez_workflow.md güncellemesi gerekiyorsa: değişen bölümü tam olarak yaz,
  "Bölüm X'i şununla değiştir" formatında teslim et.
- Dosya değişikliği gerekmiyorsa "gunce.md'ye şu bloğu ekle" ile bitir.

─── GENEL KURALLAR ────────────────────────────────────────────────────────────

- Mevcut aşamamla ilgili soru sorarsam o aşamanın bağlamında cevapla.
- Mevcut aşamayı bitirdiğimi söylersem "Tamamlandı ölçütü"nü kontrol et, sonra
  bir sonraki aşamaya geçiş için ne yapmam gerektiğini özetle.
- Tez dışı bir şey sorarsam NORMAL cevapla. "Tezine odaklan" deme.
- Aşama atlatma. "Şunu da yapsan iyi olur" baskısı yapma; ben sorduğumda öner.
- Hangi aşamada olduğum belirsizse, önce gunce.md'den okuduğunu söyle.

Cevap tarzı:
- Türkçe. Teknik terimler İngilizce kalsın (XGBoost, quantile, pinball, CRPS).
- Akademik üslup, gevezelik yok. Doğrudan konuya gir.
- Bullet listesi yerine prose tercih et. Tablolar ve formüller serbest.
- Belirsiz noktayı söyle, varsayım yapma. Bilmiyorsan "bilmiyorum" de.
- APA atıf tarzı: Yazar (Yıl).
- M4 donanım kısıtını düşün; ağır işlere "Colab" öner.
- 300 kelime üstüne çıkacaksan başlıklarla parçala.

Yapma:
- Psikanaliz, motivasyon konuşması.
- "Başka nasıl yardımcı olabilirim?", "umarım yardımcı olmuştur" kalıpları.
- Tez dışı sorularda model önerisi veya token uyarısı.
- Aşama atlatma baskısı.
```

**Project'e yüklenecek dosyalar:**

- `tez_oneri.docx` — orijinal proje önerisi.
- `tez_workflow.md` — şu okuduğun dosya.
- `literatur_ozeti.md` — NotebookLM'den üretilip yüklenecek (STAGE-1'de).

**Tek kural:** Tezle ilgili her konuşmayı bu Project içinde aç.

---

## 4. NotebookLM organizasyon (tek seferlik kurulum)

**Temizlik:**

- "Su kararları" iki defter → birini sil veya birleştir (tez dışı).
- "Yapay Zekânın Çalışma Prensipleri" (0 kaynaklı) → sil.
- "Elastic Net" iki defter → birleştir veya birini sil (tezde kullanılmıyor).
- "tez kaynakları" → yeniden adlandır: "TEZ — PV Olasılıksal Tahmin".

**Hedef yapı:**

- Ana defter: "TEZ — PV Olasılıksal Tahmin" (tüm tez referansları).
- Tematik yardımcı: "Yöntem — Stacking ve Quantile" (Wolpert, Ting & Witten, Koenker & Bassett, Salinas, Atiea).
- Tematik yardımcı: "Yöntem — Eksik Veri" (Sperrin, Sisk, Perez-Lebel, Twala, Little & Rubin, Jones, Kuhn & Johnson).
- Tematik yardımcı: "PV Fiziği" (Holmgren pvlib, Duffie & Beckman).

**Köprü stratejisi — kritik:**

Ana defterde aylık veya yeni referans eklendiğinde şu prompt'u çalıştır:

> "Bu defterdeki tüm kaynakları beş başlık altında grupla: (a) Fiziksel öznitelik / pvlib, (b) Stacking ve quantile, (c) Eksik veri prediction-odaklı, (d) PV baseline (LSTM/Transformer), (e) Olasılıksal metrikler. Her kaynak için (1) APA atıf, (2) 2-3 cümle özet, (3) hangi STAGE'de kullanılacak, (4) Atiea ve ark. (2025) karşılaştırmamda hangi rolde. Çıktı tek markdown dosya."

Çıktıyı `literatur_ozeti.md` olarak Project'e yükle. Güncelle, üzerine yaz.

**Hiyerarşi:** Ham kaynağa kadar bakmak gerekirse NotebookLM. Sentez, yorum, yazım yardımı Claude Project.

---

## 5. AŞAMALAR

14 aşama var. Her aşamada: Önceden, Hedef, Adımlar, Claude'a örnek sorular, Tamamlandı ölçütü, Sonraki.

---

### STAGE-0 — Setup

**Önceden:** Claude uygulaması açık, NotebookLM hesabın var.

**Hedef:** Çalışma ortamını kur.

**Adımlar:**

1. `0.1` — Claude Project oluştur (Bölüm 3'teki Custom Instructions'ı yapıştır).
2. `0.2` — `tez_oneri.docx` + bu dosyayı (`tez_workflow.md`) Project'e yükle.
3. `0.3` — NotebookLM'de Bölüm 4'teki temizliği yap (duplicate sil, ana defteri yeniden adlandır).
4. `0.4` — Python ortamı: Python 3.11+ kur. `python -m venv tez-env` ile venv oluştur, aktive et.
5. `0.5` — Çekirdek kütüphaneleri yükle: `pip install numpy pandas scikit-learn xgboost lightgbm catboost pvlib optuna joblib matplotlib seaborn streamlit`
6. `0.6` — PyTorch'u M4 için: `pip install torch torchvision` (MPS otomatik gelir).
7. `0.7` — Git repo kur: `git init`, `.gitignore` (data/raw/ dahil), ilk commit.

**Claude'a örnek sorular:**
- "Custom Instructions'ı şu şekilde uyarladım: [metin]. Eksik veya çelişen bir nokta var mı?"
- "Python ortamı kurmaya çalışıyorum, pyenv vs conda hangisini önerirsin?"

**Tamamlandı ölçütü:**
- Project'te dosyalar yüklü.
- NotebookLM'de ana defter düzgün adlandırılmış.
- `python -c "import xgboost, lightgbm, catboost, pvlib, optuna; print('ok')"` hatasız çalışıyor.
- Git repo temiz, ilk commit var.

**Sonraki:** STAGE-1 (Literatür sindirimi)

---

### STAGE-1 — Literatür sindirimi

**Önceden:** STAGE-0 tamam. Tez önerindeki 24 referansın elektronik kopyaları toplanabilir durumda.

**Hedef:** Tezin literatür temelini içselleştir. Her referansın tezindeki rolünü bil.

**Adımlar:**

1. `1.1` — Tez önerindeki 24 referansın PDF'lerini topla. Erişimi olmayan varsa kütüphane / sci-hub / yazar e-posta yoluyla bul.
2. `1.2` — Hepsini NotebookLM ana defterine yükle.
3. `1.3` — Bölüm 4'teki köprü prompt'unu çalıştır. Çıktıyı `literatur_ozeti.md` olarak kaydet, Project'e yükle.
4. `1.4` — 2024-2026 dönemi için ek 5-10 güncel makale ara: "physics-informed PV forecasting", "probabilistic solar power", "missingness indicators prediction". Ekle.
5. `1.5` — Atiea ve ark. (2025) ile farklarını netleştir. 4 ayrışma noktasını tek paragrafta yaz, `gunce.md`'ye kaydet.
6. `1.6` — Sperrin et al. (2020) prediction vs causal ayrımını bir paragrafta özetle, `gunce.md`'ye kaydet.

**Claude'a örnek sorular:**
- "Şu makalede [X], yazar [Y] iddiasını [Z] kanıtıyla destekliyor. Tezimin hangi katkısıyla bağlanıyor?"
- "Atiea ve ark. ile benim arasındaki en zayıf ayrışma hangisi sence?"

**Tamamlandı ölçütü:**
- NotebookLM ana defterde 30+ kaynak.
- `literatur_ozeti.md` Project'te güncel.
- Atiea ile fark paragrafı + Sperrin prediction-vs-causal paragrafı yazılı.

**Sonraki:** STAGE-2 (Veri temini)

---

### STAGE-2 — Veri temini

**Önceden:** STAGE-1 tamam veya paralel ilerliyor.

**Hedef:** DKASC ve PVOD verisini diskine indir, format ve içeriği anla.

**Adımlar:**

1. `2.1` — DKASC Alice Springs verisi: dkasolarcentre.com.au üzerinden 2010-2022 saatlik veri. Format: CSV. Sütunlar: timestamp, GHI, DHI, DNI, T_amb, wind_speed, PV_output.
2. `2.2` — PVOD v1.0 verisi: Chen & Xu (2022) Scientific Data makalesindeki repository'den 10 istasyon × 271.968 kayıt. 15 dk çözünürlük.
3. `2.3` — Her iki veri için ilk EDA: eksik veri haritası, zaman aralığı kontrolü, gündüz/gece dağılımı, anomali tespiti.
4. `2.4` — Veri sözlüğü (`docs/data_dictionary.md`): her sütun ne, birimi, gözlem sayısı, eksik oranı.
5. `2.5` — Ham CSV'leri `data/raw/` altına koy (gitignore'da, commit etme). `data/processed/` boş başlat.

**B Planı (veri erişim sorunu olursa):**
- DKASC indirilemezse → AEMO (Australian Energy Market Operator) açık veri portalı
- Meteoroloji verisi eksikse → NREL NSRDB API (pvlib simülasyonu ile sentezleme)

**Git adımları:**
```
git add docs/ notebooks/
git commit -m "STAGE-2: [ne yapıldı, ölçülebilir]"
```

**Claude'a örnek sorular:**
- "DKASC'de [şu sütun] var ama tarif belirsiz. pvlib veya literatürde bu nasıl tanımlanıyor?"
- "PVOD 10 istasyondan hangi(ler)i tezimle uyumlu?"

**Tamamlandı ölçütü:**
- İki veri seti yerel diskinde.
- `data_dictionary.md` yazılı.
- Eksik veri oranları, zaman aralıkları belgelenmiş.

**Sonraki:** STAGE-3 (Fiziksel öznitelik pipeline'ı)

---

### STAGE-3 — Fiziksel öznitelik pipeline'ı

**Önceden:** STAGE-2'de ham veri yüklü, sütunlar tanımlı.

**Hedef:** pvlib ile fiziksel öznitelikleri üreten tekrar edilebilir Python pipeline'ı.

**Adımlar:**

1. `3.1` — `features/physical.py` modülü: `get_solar_position()` ile zenith, azimuth, saat-açısı.
2. `3.2` — `cos_zenith` öznitelik (radyana çevirme + cosinüs). Gece değerleri → 0'a clip.
3. `3.3` — Air mass (Kasten-Young): `pvlib.atmosphere.get_relative_airmass()`.
4. `3.4` — Açıklık indeksi `k_t = G / G_0`. G_0 extraterrestrial irradiance pvlib'den.
5. `3.5` — `T_cell = T_amb + G·(NOCT−20)/800`. NOCT varsayılan 46°C.
6. `3.6` — Zaman dönüşümleri: `hour_sin/cos`, `month_sin/cos` (döngüsel kodlama).
7. `3.7` — `StandardScaler` normalizasyon — sadece train setinde fit, val/test transform.
8. `3.8` — Pipeline'ı `sklearn.Pipeline` ile sarmala.
9. `3.9` — Birim testler: bilinen tarih-saat için elle hesaplanmış değerlerle karşılaştırma.

**Git adımları:**
```
git add features/
git commit -m "STAGE-3: pvlib pipeline birim testli, cos_zenith/kt/T_cell üretiliyor"
make leakage
```

**Claude'a örnek sorular:**
- "pvlib'de `get_solarposition()` zenith açısını derece veriyor; radyana çevirip cos alıyorum. Doğru mu?"
- "`k_t` için `G_0`'ı pvlib'den nasıl alıyorum?"

**Tamamlandı ölçütü:**
- `features/physical.py` çalışır, test edilmiş.
- Bilinen referans tarih-saat için manuel hesapla pipeline çıktısı eşleşiyor.
- cos(θ_z), k_t, air mass, T_cell, zaman döngüsel öznitelikleri DataFrame'de mevcut.

**Sonraki:** STAGE-4 (Veri bölme ve doğrulama stratejisi)

---

### STAGE-4 — Veri bölme ve doğrulama

**Önceden:** STAGE-3 tamam, öznitelikler hazır.

**Hedef:** Data leakage'sız train/val/test bölünmesi + Walk-Forward Validation kurgusu.

**Adımlar:**

1. `4.1` — Kronolojik 70/15/15 bölme. `shuffle=False` zorunlu.
2. `4.2` — Train/val/test sınırlarında zaman aşımı kontrolü yap. Başlangıç-bitiş tarihlerini logla.
3. `4.3` — Eksik veri imputasyon: 3 saatten kısa → doğrusal interpolasyon, üstü → `ffill/bfill` (varsayılan). `imputer_strategy` parametresiyle "median" veya "knn" de seçilebilir. **Not:** KNNImputer 950K satırda O(n²) hesaplama maliyeti nedeniyle terk edildi; ffill/bfill zaman serisi için daha uygun ve O(n).
4. `4.4` — Walk-Forward Validation iskeleti: `TimeSeriesSplit(gap=24)`.
5. `4.5` — `make_dataset()` fonksiyonu: ham veri → (X_train, y_train, X_val, y_val, X_test, y_test).

**Git adımları:**
```
git add features/ scripts/
git commit -m "STAGE-4: walk-forward iskelet kuruldu, leakage kontrol edildi"
make leakage
```

**Tamamlandı ölçütü:**
- Veri leakage olmadığı yazılı kanıtla.
- `make_dataset()` aynı seed ile aynı çıktıyı veriyor.
- Walk-Forward iskeleti mock model ile bir tur döndü.

**Sonraki:** STAGE-5 (Taban öğreniciler)

---

### STAGE-5 — Taban öğreniciler

**Önceden:** STAGE-4 tamam, train/val setleri hazır.

**Hedef:** 9 taban model (3 algoritma × 3 quantile) eğitilebilir, OOF tahminleri üretebiliyor.

**Adımlar:**

1. `5.1` — LightGBM quantile: `objective='quantile'`, `alpha=q`. q=0.1, 0.5, 0.9 için ayrı model.
2. `5.2` — CatBoost quantile: `loss_function=f'Quantile:alpha={q}'`. 3 model.
3. `5.3` — XGBoost custom pinball objective: gradient ve hessian elle yazılır. Toy data ile test edilir.
4. `5.4` — Tek tip API: `train_base_learner(algo, q, X_train, y_train) → fitted_model`.
5. `5.5` — Out-of-fold (OOF) tahmin üretici: zaman-uyumlu K-fold ile her taban model için OOF kolonu.
6. `5.6` — Tüm 9 modelin OOF tahminleri tek matrise (X_meta).

**Git adımları:**
```
git add models/
git commit -m "STAGE-5: 9 taban model eğitildi, OOF pinball raporlandı"
```

**Tamamlandı ölçütü:**
- 9 modelin her biri validation'da pinball loss skoru raporluyor.
- X_meta matrisi var (n_train × 9).
- OOF stratejisi zamansal olarak leakage'sız.

**Sonraki:** STAGE-6 (Meta-öğrenici + missingness flags)

---

### STAGE-6 — Meta-öğrenici + missingness flags

**Önceden:** STAGE-5 tamam, X_meta hazır.

**Hedef:** Ridge meta-öğrenici × 3 quantile, missingness flags özellik uzayına eklenmiş.

**Adımlar:**

1. `6.1` — Missingness flags: `is_G_missing`, `is_Tamb_missing`, `is_RH_missing`, `is_wind_missing` binary kolonlar.
2. `6.2` — X_meta zenginleştirme: 9 OOF + 4 flag = 13 kolon.
3. `6.3` — `Ridge` × 3 quantile. Alpha Optuna'da aranacak, şimdilik varsayılan.
4. `6.4` — Meta tahmin: her quantile için ŷ_q üretici fonksiyon.
5. `6.5` — Kalibrasyon kontrolü: %10–%90 bant kapsama oranı (coverage). Hedef ~%80.
6. `6.6` — Pinball loss: tek LightGBM-quantile baseline vs stacked karşılaştır.

**Tamamlandı ölçütü:**
- Meta-öğrenici pipeline çalışıyor.
- Stacked sistem tek LightGBM-quantile baseline'a karşı min %5 pinball iyileşmesi.
- Coverage %75–85 aralığında.

**Sonraki:** STAGE-7 (Optuna)

---

### STAGE-7 — Optuna hiperparametre optimizasyonu

**Önceden:** STAGE-6 tamam, pipeline uçtan uca çalışır.

**Hedef:** Bayesyen aramayla optimize hiperparametreler. Validation pinball loss minimize.

**Adımlar:**

1. `7.1` — Arama uzayı: max_depth (3-12), learning_rate (1e-3–0.3 log), n_estimators (100-2000), reg_alpha, reg_lambda, subsample, colsample.
2. `7.2` — Objective: `(L_0.1 + L_0.5 + L_0.9) / 3` validation üzerinde.
3. `7.3` — `TPESampler` + `MedianPruner`.
4. `7.4` — Trial bütçesi: 50-100 deneme. M4'te uzarsa Colab T4'e taşı.
5. `7.5` — En iyi trial sonucu `best_params.json` olarak kaydet.
6. `7.6` — Görselleştirme: parallel coordinate plot, hyperparameter importance.

**Tamamlandı ölçütü:**
- `best_params.json` yazılı.
- Optimize modelin validation pinball, varsayılan parametrelere kıyasla min %10 düşük.
- En iyi 5 trial'ın hyperparameter dağılımı analiz edilmiş.

**Sonraki:** STAGE-8 (Robustness testleri)

---

### STAGE-8 — Robustness testleri (üç eksen)

**Önceden:** STAGE-7 tamam, optimize model var.

**Hedef:** Üç eksenli sensör kayıp senaryolarında modelin davranışı niceliksel raporlu.

**Adımlar:**

1. `8.1` — `RobustnessScenario` sınıfı: test setine kayıp uygular, missingness flags günceller.
2. `8.2` — Rastgele kayıp: %10, %20, %30, %50 oranında NaN maskeleme.
3. `8.3` — Burst kayıp: rastgele başlangıç + 1/6/24 saat süreli kesintisiz kayıp.
4. `8.4` — Sensör-özgü kayıp: G-only, T_amb-only, RH-only.
5. `8.5` — Her senaryoda Pinball ve CRPS ölç. Coverage değişimini de ekle.
6. `8.6` — Sonuç tablosu: senaryo × metrik matrisi. Heatmap görselleştirme.
7. `8.7` — Flags ile / flagsiz karşılaştırma — Diebold-Mariano testi ile hipotez doğrulama.

**Tamamlandı ölçütü:**
- 9 senaryo (3 eksen × 3 seviye) tamamlanmış sonuçlu.
- Flags ile/flagsiz karşılaştırma istatistiksel testle yapılmış.
- Hipotezin sonucu (doğrulandı/doğrulanmadı) `gunce.md`'de kayıtlı.

**Sonraki:** STAGE-9 (Baseline modeller)

---

### STAGE-9 — Baseline modeller

**Önceden:** STAGE-8 tamam, kendi modelin sağlam.

**Hedef:** k-NN, SVM, LSTM, hafif Transformer aynı veri seti üzerinde eğitilmiş, aynı metriklerle değerlendirilmiş.

**Adımlar:**

1. `9.1` — k-NN regressor (`sklearn.neighbors.KNeighborsRegressor`).
2. `9.2` — SVM regressor (RBF kernel). Büyük veri için Nystroem + LinearSVR yaklaşıklığı.
3. `9.3` — LSTM (PyTorch): 2-3 katmanlı, hidden_size=64-128. MPS backend. Quantile loss çıktısı.
4. `9.4` — Hafif TFT: pytorch-forecasting kütüphanesi.
5. `9.5` — Hesaplama süresi kaydet.
6. `9.6` — Aynı metriklerle değerlendir (MAE, RMSE, Pinball, CRPS).

**Tamamlandı ölçütü:**
- 4 baseline modelin tamamı eğitilmiş, aynı metriklerle değerlendirilmiş.
- Her modelin hesaplama süresi kayıtlı.

**Sonraki:** STAGE-10 (Karşılaştırmalı analiz)

---

### STAGE-10 — Karşılaştırmalı analiz

**Önceden:** STAGE-9 tamam, tüm modellerin sonuçları elde.

**Hedef:** Master tablo, görsel analizler, istatistiksel anlamlılık testleri.

**Adımlar:**

1. `10.1` — Master tablo: rows = 6 model, columns = (MAE, RMSE, Pinball, CRPS, Coverage, eğitim süresi).
2. `10.2` — Heatmap: sensör türü × hata değişimi.
3. `10.3` — Olasılıksal bant görselleştirme: gerçek vs medyan tahmin + %10-%90 bant grafiği.
4. `10.4` — Diebold-Mariano pairwise testleri. Holm-Bonferroni düzeltmesi.
5. `10.5` — Edge AI argümanı: hesaplama süresi vs CRPS scatter plot.
6. `10.6` — Tüm görselleri `figures/` altına PNG + PDF kaydet.

**Tamamlandı ölçütü:**
- Master tablo bitti.
- 5-7 yayın kalitesinde şekil hazır.
- İstatistiksel anlamlılık tablosu var.

**Sonraki:** STAGE-11 (Streamlit demo)

---

### STAGE-11 — Streamlit demo

**Önceden:** STAGE-10 tamam, model + sonuçlar hazır.

**Hedef:** Yerel ortamda çalışan, jüriye sunulabilir interaktif gösterim arayüzü.

**Adımlar:**

1. `11.1` — UI: tarih-saat seçici, tahmin görselleştirme (medyan + bant), senaryo seçici, baseline toggle, metrik paneli.
2. `11.2` — `app/app.py` Streamlit kodu.
3. `11.3` — Model joblib serileştirme: `models/checkpoints/stacked_ensemble.joblib`.
4. `11.4` — README + kullanım kılavuzu.
5. `11.5` — Demo akışı senaryosu (5-10 dakika).

**Tamamlandı ölçütü:**
- `streamlit run app/app.py` lokal'de çalışıyor.
- README ile başkası da çalıştırabiliyor.
- Demo akışı senaryosu yazılı.

**Sonraki:** STAGE-12 (Tez yazımı)

---

### STAGE-12 — Tez yazımı

**Önceden:** STAGE-11 tamam.

**Hedef:** Enstitü formatında tam tez taslağı.

**Adımlar:**

1. `12.1` — Tez şablonunu enstitü web sayfasından indir.
2. `12.2` — Bölüm 1 Giriş: problem tanımı, motivasyon, dört özgün katkı.
3. `12.3` — Bölüm 2 Literatür Taraması: `literatur_ozeti.md`'yi temel al.
4. `12.4` — Bölüm 3 Yöntem: STAGE-3'ten STAGE-7'ye kadar yaptıklarını akademik üslupta anlat.
5. `12.5` — Bölüm 4 Bulgular: master tablo + şekiller + Diebold-Mariano sonuçları.
6. `12.6` — Bölüm 5 Sonuç ve Öneriler: hipotezin doğrulanma durumu, sınırlılıklar, ileri çalışma.
7. `12.7` — Atıflar APA. Mendeley/Zotero ile yönet.
8. `12.8` — İlk taslağı danışmana gönder. Geri bildirim → revizyon.

**Tamamlandı ölçütü:**
- 5 bölüm tam taslak.
- Danışman onayı (en az 1 revizyon turu).
- Atıflar tutarlı APA formatında.
- Plagiarism check geçildi.

**Sonraki:** STAGE-13 (Makale taslağı)

---

### STAGE-13 — Makale taslağı

**Önceden:** STAGE-12'de en az Bölüm 4 bitmiş.

**Hedef:** SCI/SCI-E dergisine gönderilmeye hazır makale taslağı.

**Adımlar:**

1. `13.1` — Hedef dergi: Energies (open access, hızlı dönüş). Alternatif: Renewable Energy, Solar Energy.
2. `13.2` — Submission guidelines oku.
3. `13.3` — Abstract: 200 kelime içinde, dört özgün katkıyı vurgulayan.
4. `13.4` — Introduction: tezin Bölüm 1'inden kısalt.
5. `13.5` — Methodology + Results: tezin Bölüm 3+4'ünden kısalt.
6. `13.6` — Discussion + Conclusion: daha geniş etki vurgusu.
7. `13.7` — Co-author Doç. Dr. Altun ile gözden geçirme.
8. `13.8` — Şekiller 300 dpi + vektörel.
9. `13.9` — Submission.

**Tamamlandı ölçütü:**
- Submission-ready makale taslağı.
- Co-author onayı.
- Submission yapıldı.

**Sonraki:** Tebrikler. Tez bitti, makale yolda.

---

## 6. Prompt şablonları (kopyala-yapıştır)

**Aşama içi soru:**
```
[STAGE-X / adım Y'deyim]
Şu noktada takıldım: [somut tarif]. Beklediğim: [a]. Aldığım: [b].
Çözüm önerisi?
```

**Makale değerlendirme:**
```
Şu makaleyi okudum: [Yazar, Yıl, başlık]. Ana iddiası: [1 cümle].
Tezimle ilişkisi nedir? Hangi STAGE'de işime yarar?
Atiea ve ark. (2025) ile karşılaştırmamda hangi rolde?
```

**Kod hata:**
```
[code block - sadece ilgili 20-50 satır]
Bekledim: [x]. Aldım: [y]. M4 + Python 3.x + [paket versiyonu].
Olası nedenler + düzeltme?
```

**Yöntem paragraf revizyonu:**
```
[paragraph]
Akademik üslupta, gereksiz tekrarsız revize. APA atıflar dokunulmasın.
Tezimin dört özgün katkısını yansıtsın.
```

**Aşama bitirme:**
```
STAGE-X'i bitirdim. Tamamlandı ölçütleri:
- [a]: ✓ / ✗
- [b]: ✓ / ✗
Sonraki aşamaya geçmeli miyim, yoksa eksik bir şey mi var?
```

---

## 7. Token tasarrufu disiplinleri

- **Tek konuşma = tek görev.** Paragraf revize başka, kod hatası başka. Konuşma uzayıp konusunu kaybedince kapat, 5 cümle özetini `gunce.md`'ye yapıştır, yeni konuşma aç.
- **Ham veri yapıştırma yok.** DataFrame'in 1000 satırı değil, `df.describe()` çıktısı veya 5 satırlık örnek.
- **Lokal alıntı.** Tüm tez metni değil, revize edilecek paragraf.
- **Konuşma sonu özet.** "Bu konuşmanın 5 cümle özetini ver" → `gunce.md`'ye yapıştır.

---

## 8. Risk yönetimi

| Risk | İlk işaret | B Planı |
|------|-----------|---------|
| DKASC/PVOD format uyumsuzluğu | Yükleme hatası, kolon eşleşmesi | AEMO açık veri portalı veya NREL NSRDB API |
| Optuna lokal minimum | Pinball 50 deneme sonra düz | Arama uzayı kademeli daralt |
| Burst kayıpta katastrofik düşüş | CRPS %50+ kötüleşme | Clear-sky model fallback |
| M4'te LSTM/Transformer uzun | Eğitim 4 saat+ | Colab T4, LSTM 2 katman + TFT light |
| Plansız sapma | Aşama ölçütleri karşılanmadan ilerleme | `gunce.md`'yi dürüstçe güncelle, geri dön |

---

## 9. Tıkanırsam ne yapayım

- **Belirsizlik:** `gunce.md` oku. Hangi aşamada, hangi adımdayım?
- **Teknik tıkanıklık:** Claude Code'da sor — somut ve lokal. "Hiçbir şey olmuyor" değil, "X scripti Y hatası veriyor, traceback şu."
- **Akademik tıkanıklık:** Danışmanla randevu. Claude kafayı netleştirme aracı, danışman değil.
- **Motivasyon tıkanıklığı:** Bu dosyayı kapat, başka iş yap. Tez maraton, sprint değil.
- **Literatürde cevap yok hissi:** NotebookLM ana defterde sor. Yetersizse 2-3 yeni makale ara, ekle.

---

*Bu dosya senin yıl boyunca açıp baktığın anchor. `gunce.md` güncel kaldıkça Claude seni doğru yerden devam ettirir. Başarılar.*
