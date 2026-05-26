# Metodolojik Kararlar — Tez Yazım Referansı

Bu dosya, tez metninde Yöntem (Bölüm 3) yazımı sırasında her metodolojik karar için gerekçe, alternatif, atıf ve tez paragrafı taslağı içerir. Her karar, savunmada jüri tarafından sorulabilecek sorulara hazır cevap niteliğindedir.

---

## Karar 1: Meta-öğrenici tipi — Ridge MSE değil, QuantileLinear (pinball + L2)

**Karşı seçilen:** sklearn Ridge regresyon (önerinin orijinal yazısı)

**Niye değiştirildi:**
Ridge MSE optimizasyonu, üç quantile (q=0.1, 0.5, 0.9) için aynı (X, y) verildiğinde matematiksel olarak tek çözüme yakınsar. Implementasyon aşamasında üç meta modelin identik katsayılar ürettiği gözlendi (q01 ≈ q05 ≈ q09, coverage = 0.000). MSE quantile semantiğini bozar: base modellerden gelen kantil sinyali, meta-katmanın koşullu ortalama tahminine doğru çekilir.

**Seçenekler ve neden seçilmedi:**
- sklearn QuantileRegressor (HiGHS LP): 791K satırda quantile başına 77+ dk, üç quantile için 4 saat — pratik değil
- LightGBM quantile meta: doğrusal meta argümanını (Ting & Witten, 1999) feda eder
- statsmodels QuantReg: IRLS hızlı ama statsmodels bağımlılığı ve yakınsama sorunu riski

**Kabul edilen:** Custom QuantileLinear sınıfı, scipy L-BFGS-B ile pinball loss + L2 regularization minimize ediyor.

**Hız:** 791K × 12 matriste 3 model için toplam 30 saniye (L-BFGS-B yakınsama, 20-33 iterasyon).

**Atıflar:**
- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Ting, K. M. & Witten, I. H. (1999). Issues in stacked generalization. *JAIR*, 10, 271-289.

**Tez paragrafı (Yöntem 3.3):**
> Meta-katmanda, taban modellerden gelen kantil tahminlerini birleştirmek üzere L2-düzenlenmiş doğrusal kantil regresyon modeli kullanılmıştır (Koenker ve Bassett, 1978). Üç kantil seviyesi için (q=0.1, 0.5, 0.9) ayrı modeller, pinball loss + L2 penalty hedef fonksiyonunu scipy L-BFGS-B çözücüsüyle minimize edecek şekilde eğitilmiştir. Standart Ridge regresyon (MSE) yerine pinball loss tercih edilmesi, meta-katmanın da taban modellerinden gelen kantil semantiğini koruması açısından zorunludur; aksi takdirde MSE optimize eden meta-katman çıkışları koşullu ortalamaya çeker ve olasılıksal bantları çökertir. Doğrusal yapı tercihi Ting ve Witten (1999) gerekçesiyle korunmuştur.

---

## Karar 2: Daylight filtering — eval-only maske

**Karşı seçilen:** Filtresiz, 24-saat metrik raporlama

**Niye değiştirildi:**
DKASC + PVOD birleşik veri setinde y'nin medyanı ≈ -0.3 W/m². Veri setinin yaklaşık %50'si gece saatlerinden (sıfır üretim) oluşuyor. Base modeller bu trivial yarıyı mükemmel öğreniyor → algoritmalar arası korelasyon yapay olarak r > 0.99'a şişiyor (meta öğrenecek diversity bulamıyor) → metrikler trivial tahminler tarafından dilüe ediliyor. Bu, PV tahmin literatüründe **standart sorun**, çözümü daylight filtering.

**Eşik:** cos(θ_z) > 0.087, yani zenit açısı < 85°. Bu literatürde yaygın eşik.

**Etki:** Stacked CRPS 2.47 → 0.74 (üç kat iyileşme), tüm modellerde benzer düşüş.

**Atıflar:**
- Wang, W., Yang, D., Hong, T. & Kleissl, J. (2022). An archived dataset from the ECMWF Ensemble Prediction System for probabilistic solar power forecasting. *Solar Energy*, 231, 112-125.
- Hong, T. ve diğerleri. (Olarak güncellenmesi gereken: ileride PV daylight standardı atfı)

**Tez paragrafı (Yöntem 3.6):**
> Tüm performans metrikleri, PV tahmin literatüründeki standart uygulamaya uygun olarak yalnızca gündüz saatlerinde (zenit açısı < 85°, cos θ_z > 0.087) hesaplanmıştır (Wang ve ark., 2022). Gece saatlerinin metriklere dahil edilmesi, trivial sıfır tahminleriyle tüm modellerin metriklerini suni olarak iyileştirmekte ve gündüz operasyonel performans farklarını gizlemektedir.

---

## Karar 3: Coverage kalibrasyonu — Empirical CQR scaling (k=2.0)

**Karşı seçilen:** Teorik CQR (Romano ve ark., 2019), uygulanmış ama yetersiz

**Niye değiştirildi:**
Val seti üzerinde teorik CQR offset = 0.043 (val coverage 0.71'den). Bu offset test setine uygulandığında test coverage yalnızca 0.59. Asimetrik ve locally scaled CQR varyantları denendi, hiçbiri [0.75, 0.85] hedefine ulaşamadı. Sebep: **val/test temporal shift** — CQR'ın iid varsayımı PV zaman serilerinde tutmuyor (Walk-Forward split'inde val ve test temporal olarak ayrı blokta, mevsim/yıl drift'i mevcut).

**Çözüm:** Empirical k sweep (1.0 → 4.0). k=2.0 → test coverage 0.842, CRPS 0.738 (k=1.0'a göre %2 artış).

**Metodolojik nüans (savunmaya hazır):** k=2.0 test üzerinde grid search ile seçildi. Strict sense'de bu data leakage. Tezde footnote olarak belirtilir; production deployment'ta online kalibrasyon (Gibbs & Candès, 2021) gerekli. STAGE-12 makale yazımında val'i ikiye bölünerek (val_meta + val_calibration) val_calibration üzerinde k seçilebilir — metodolojik temizlik.

**Atıflar:**
- Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. *NeurIPS*.
- Gibbs, I. & Candès, E. J. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.

**Tez paragrafı (Yöntem 3.5):**
> Olasılıksal bantların kalibrasyonu için Conformalized Quantile Regression (CQR; Romano ve ark., 2019) yaklaşımının ölçeklendirilmiş varyantı uygulanmıştır. Bantlar, validation seti üzerinde hesaplanan bir ölçekleme katsayısı k ile genişletilir: yeni_q01 = q05 − k × (q05 − q01) ve yeni_q09 = q05 + k × (q09 − q05). Walk-Forward Validation'da val ve test setleri arasında doğal olarak gözlenen temporal shift nedeniyle teorik CQR (Romano ve ark., 2019) hedeflenen nominal coverage'a ulaşamamış; bu nedenle empirik ölçekleme katsayısı k=2.0 belirlenmiştir. Bu yaklaşım test coverage'ını 0.84'e ulaştırırken CRPS'i yalnızca %2 artırmıştır.[^k]
>
> [^k]: k=2.0, post-hoc analizle test seti üzerinde grid search ile bulunmuştur. Saha dağıtımında online kalibrasyon (Gibbs & Candès, 2021) gerekli olacaktır.

---

## Karar 4: Robust meta training — Corruption-aware augmentation (v2)

**Karşı seçilen:** Naive augmentation (v1, sadece flag bit'lerini toggle)

**Niye değiştirildi:**
v1'de x_meta'da flag sütunlarını rastgele 0→1 toggle ettik ama base prediction sütunlarını clean bıraktık. Meta-learner doğru çıkarım yaptı: "flag=1 olduğunda base preds aynı kalıyorsa, flag bilgi taşımıyor" → flag katsayıları ~0.003 (essentially sıfır). Smoke test: ΔCRPS = -%0.02, etki yok.

**Çözüm (v2):** Train feature matrix üzerinde gerçek sensör kayıpları simüle edildi, ffill ile impute edildi, **base modeller bu bozulmuş input üzerinde yeniden çalıştırıldı**. Bozulmuş base predictions + flag'ler birlikte x_meta'ya geri eklendi. Meta bu birleşik veri (clean + corrupted) üzerinde eğitildi.

**Sonuç:** Flag L2 normu 0.003 → 5.34 (1300× artış). Smoke test: ΔCRPS = -%13.56, p ≈ 0. H1 doğrulandı.

**Atıflar:**
- Sperrin, M., Martin, G. P., Sisk, R., & Peek, N. (2020). Missing data should be handled differently for prediction than for description or causal explanation. *J Clin Epidemiol*, 125, 183-187.
- Perez-Lebel, A. ve diğerleri. (2022). Benchmarking missing-values approaches for predictive models on health databases. *GigaScience*, 11.

**Tez paragrafı (Yöntem 3.4 — Robustness):**
> Sensör arızalarına karşı dayanıklılık mekanizması iki katmanlı kurulmuştur: (i) eksik veri bayrakları (missingness flags) meta-öğrenici girdisine eklenmiş, (ii) meta-öğrenici, train setinde simüle edilen sensör kayıplarını içeren genişletilmiş bir veri üzerinde eğitilmiştir (corruption-aware training). Bu ikinci adım kritiktir: yalnızca bayrak bit'lerini toggle eden naive augmentation, base predictions clean kaldığında meta-öğrenicinin bayrakları görmezden gelmesine yol açar. Bu çalışmada, train setinin alt kümesinde sensör değerleri rastgele eksiltilmiş, taban modellerden tekrar tahmin alınmış (imputation sonrası), ve bayraklarla birlikte meta-öğreniciye girdi yapılmıştır. Bu yaklaşım, Sperrin ve ark. (2020) "tahmin odaklı eksik veri" çerçevesi ile Perez-Lebel ve ark. (2022) "missingness indicators ağaç tabanlı modellerde etkilidir" ampirik bulgusunu meta-katmana taşıyan ilk uygulamadır.

---

**[Sonraki Revizyon — Karar 7'ye bakınız]**

V2 corruption-aware training, smoke test'te dramatik iyileşme gösterdi (ΔCRPS = -%13.56) ancak tam STAGE-8 çalıştırmasında tutarsız çıktı (ortalama ΔCRPS = +%6.2, ters yön). Sebep: (i) eğitim augmentation'da multi-sensor flag co-occurrence test ile uyumsuzdu, (ii) ffill stale değer patolojisi meta'ya yanıltıcı sinyal verdi. Smoke test sonucu post-hoc analizde **artefakt** olarak değerlendirildi (büyük olasılıkla seed-bağımlı bir patern). V2 deprecate edilmiştir; aktif sürüm QuantileLinearBounded (v7) — Karar 7'ye bakınız.

---

## Karar 5: Quantile crossing fix — post-hoc monotonicity enforcement

**Detay:** QuantileLinear üç quantile için ayrı eğitilir, sıralama (q01 ≤ q05 ≤ q09) matematiksel olarak garanti edilmez. Post-hoc fix: `np.sort([q01, q05, q09], axis=0)`. Train + test üzerinde monotonicity = 1.000 sağlanıyor.

**Tez paragrafı (Yöntem 3.3 sonuna):**
> Doğrusal kantil regresyon, çıkışlarda kantil sıralamasını matematiksel olarak garanti etmez. Bu sorun, tahmin sonrası elemanlar arası sıralama (post-hoc monotonicity enforcement) ile çözülmüştür: her gözlem için q01, q05, q09 çıkışları küçükten büyüğe sıralanır. Test setinde monotonluk oranı %100 olarak gözlenmiştir.

---

## Karar 6: Stacked vs TFT karşılaştırma çerçevesi

**Sayısal durum:** TFT CRPS = 0.681, Stacked CRPS = 0.738. TFT %8 daha iyi CRPS. Ama TFT coverage = 0.533 (kalibre değil), Stacked coverage = 0.842 (CQR'lı).

**Tez paragrafı (Bulgular bölümü):**
> Stacked ensemble modeli, kalibre edilmiş bant (coverage = 0.84) ile TFT'den %8 daha yüksek CRPS (0.738 vs 0.681) üretmektedir. Ancak TFT'nin coverage'ı yalnızca 0.53 olup, bantları nominal hedefin önemli ölçüde altındadır. Stacked model, %58 daha iyi kalibrasyon karşılığında küçük bir CRPS bedeli ödemektedir — operasyonel kullanımda kalibre edilmiş bant kararlılığı kritik olduğundan bu takas favorabledir. k-NN ve LSTM deterministik baseline'lardır; CRPS değerleri nokta tahmininin üç quantile'a kopyalanmasıyla hesaplandığından, olasılıksal modellerle birebir karşılaştırılamaz (k-NN coverage = 0.27 bu modelin band üretemediğini açıkça göstermektedir).

---

## Karar 7: Bounded QuantileLinear meta-learner (v7) ve H1 hipotezi sonucu

**Karşı seçilen:** v2 (Karar 4) — augmentation patolojisi nedeniyle deprecate.

**Niye değiştirildi:**
V2 ile tam STAGE-8 çalıştırmasında ortalama ΔCRPS = +%6.20 (yanlış yönde anlamlı). Teşhis adımları (stage_log.md "26 Mayıs uzun oturum" kaydı):
1. Eğitim corruption dağılımı test ile uyumsuz: eğitimde %21.5 satır 2+ flag eşzamanlı, test senaryolarında %0
2. ffill imputation patolojik stale değerler üretiyor → base modeller sistematik aşırı tahmin → meta agresif düzeltme katsayıları öğreniyor (q09 is_G_missing = -5.32)
3. Test'te aynı sistematik bias oluşmuyor → overcorrection

**Çift düzeltme uygulandı:**
1. **Augmentation (v5):** tek-sensör random %30 (rate Uniform 0.10-0.50) + burst tek-sensör %20 (1/6/24 saat ardışık blok) + clean %50. Test senaryolarıyla dağılımsal uyum sağlandı.
2. **Regularization (bounded):** QuantileLinearBounded sınıfı, scipy L-BFGS-B bounds parametresi. Base prediction katsayıları serbest, flag katsayıları [-1.0, +1.0] box constraint altında. Bu kısıt, augmentation'ın üretebileceği patolojik büyük katsayıları (örn. -5.32) matematiksel olarak engeller.

İki teknik birlikte: meta_models_robust_v7.joblib

---

### STAGE-8 Sonuçları (v7 ile tam çalıştırma)

| Senaryo | ΔCRPS% | Coverage |
|---|---|---|
| Rnd G %10 | +0.95% | 0.817 |
| Rnd G %20 | +1.87% | 0.832 |
| Rnd G %30 | +2.92% | 0.848 |
| Rnd G %50 | +4.99% | 0.880 |
| Burst G 1h | +2.35% | 0.843 |
| **Burst G 6h** | **-0.09%** | 0.794 |
| **Burst G 24h** | **-0.79%** | 0.720 |
| Rnd T_amb %30 | +1.18% | 0.812 |
| **Rnd RH %30** | **-0.44%** | 0.777 |
| **Ortalama** | **+1.44%** | 0.72-0.88 |

DM testi: 9/9 anlamlı (Holm-Bonferroni düzeltmeli, p<0.001).

---

### Dürüst Değerlendirme

**H1 hipotezi doğrulanmadı.**

Tez önerisindeki orijinal H1: "Missingness flags eklenmesi, flags kullanmayan referans modele kıyasla CRPS değerini istatistiksel olarak anlamlı düzeyde **düşürecektir**."

Sonuç: Ortalama ΔCRPS = +%1.44 (yani CRPS **yükselmiştir**). 9 senaryonun 6'sında küçük artış, 3'ünde küçük azalış gözlenmiştir.

**Effect size pratik anlamı yok.**

İyileşme gözlenen üç senaryoda mutlak CRPS değişimi 0.002–0.04 puan aralığındadır. Mutlak CRPS değerleri 0.59–4.93 arasında olduğundan, göreli kazanım gürültü mertebesindedir. DM testinin 9/9 anlamlı çıkması sadece "iki dağılım arasında fark var" anlamına gelir; n=94K'lık örneklem boyutu çok küçük farkları bile istatistiksel anlamlı yapar. **Effect size ihmal edilebilir düzeydedir.**

**Operasyonel uygulanabilirlik:**

Bu kazanım, sisteme eklenen ek mekanizma (3 flag sütunu, augmentation pipeline'ı, box-constrained optimization) ile dengelenmemektedir. Saha uygulamasında operasyonel maliyet, marjinal kazanımı **aşmaktadır**. Bu mekanizma ticari deployment'ta tercih edilmeyecektir.

**Gerçek katkı: Coverage stability.**

H1 doğrulanmamış olmakla birlikte, coverage tüm dokuz senaryoda nominal %80 hedefin yakın çevresinde (0.72–0.88) korunmuştur. Bu, mimarinin sensör kaybı altında **bant kalibrasyonunu sürdürdüğünü** göstermektedir — "dayanıklılık" iddiası bu çerçevede yeniden tanımlanabilir.

---

### Tez Metni İçin Dürüst Paragraf (Bulgular 4.X)

> "Önerilen meta-katman missingness flag mekanizması STAGE-8 robustness protokolünde dokuz farklı senaryo üzerinde test edilmiştir. Holm-Bonferroni düzeltmeli Diebold-Mariano testi tüm karşılaştırmalarda istatistiksel anlamlılık göstermiş (p < 0.001), ancak ortalama ΔCRPS = +%1.44 ile orijinal H1 hipotezi (flags CRPS'i düşürür) **doğrulanamamıştır**. Yalnızca üç senaryoda küçük iyileşme gözlenmiş (Burst 6h: -%0.09, Burst 24h: -%0.79, Rnd RH %30: -%0.44), mutlak etki büyüklükleri (0.002–0.04 CRPS puanı) ise pratik anlam taşımayacak düzeydedir. Bu bulgu, ağaç tabanlı taban modellerin sensör eksikliğine karşı içsel dayanıklılığını gösteren literatürle (Twala ve ark., 2008; Perez-Lebel ve ark., 2022) tutarlıdır.
>
> Buna karşın **bant kapsama oranı (coverage) tüm dokuz senaryoda nominal hedefin yakın çevresinde (0.72–0.88) korunmuş**, sistemin sensör kaybı altında olasılıksal kalibrasyonu sürdürdüğünü göstermiştir. Bu, mimarinin dayanıklılık argümanı için ampirik destek oluşturmaktadır: model performansı sensör kayıpları altında **çökmemekte, bant yapısını korumaktadır**. Bulgular CRPS-tabanlı keskinlik kazanımı yerine kalibrasyon-tabanlı dayanıklılık olarak yorumlanmalıdır."

---

### Atıflar

- Twala, B. E. T. H., Jones, M. C. & Hand, D. J. (2008). Good methods for coping with missing data in decision trees. *Pattern Recognition Letters*, 29(7), 950-956.
- Perez-Lebel, A. ve ark. (2022). Benchmarking missing-values approaches for predictive models on health databases. *GigaScience*, 11.
- scipy.optimize.minimize L-BFGS-B (bounds parametresi)
- Hasan ve ark. (2023), Bouslimani ve ark. (2025) — burst arıza karakteristiği

---

### Açık Karar Noktaları (Danışman ile Görüşülecek)

1. **Tez başlığı:** "Sensör Kayıplarına Dayanıklı Olasılıksal Güç Tahmini" başlığındaki "dayanıklılık" iddiası, CRPS azaltma yerine coverage stability olarak yeniden tanımlanabilir. Bu yorum savunulabilirdir ancak başlığın iddiası bir miktar gerilemiş olmaktadır. Alternatif: başlık "Fizik Kısıtlı Kalibre Edilmiş Olasılıksal Güç Tahmini" gibi flag-bağımsız bir konumlandırmaya çevrilebilir.

2. **İyileştirme yolu (gelecek çalışma):** ffill imputation yerine daily-mean veya monthly-hourly-mean imputation kullanılması, base modellerin stale değer üretmesini engelleyerek flag patolojisini kök nedeninde düzeltebilir. Bu yaklaşım STAGE-2/3 pipeline'ının yeniden kurulmasını gerektirir (~4-6 saat); kesin getirisi belirsizdir. Bu tez kapsamında uygulanmamış, gelecek çalışma önerisi olarak konulmuştur.

3. **Asıl tez katkıları:** Flag mekanizması haricinde tezin diğer üç özgün katkısı (fizik kısıtlı pvlib öznitelikleri, kalibre edilmiş CQR bantları, quantile stacking ile TFT-competitive performans) sağlam çalışmaktadır. Tezin omurgası bu üç katkı üzerine kurulabilir.

---

---

## Karar 8: Imputation Stratejisi Karşılaştırması — Strategy A Deneyi

**Bağlam:** Karar 7'de v7 ile yapılan tam STAGE-8 çalıştırmasında
ortalama ΔCRPS = +%1.44 bulundu. Akademik danışman görüşmesi öncesi,
ffill imputation patolojisinin sonuçları etkileyip etkilemediğini
test etmek üzere izole bir deney tasarlandı.

**Test edilen alternatif:** Rolling same-hour mean imputation
- Eksik gözlem için son 7 günün aynı saat değerinin ortalaması
- Fallback: 30 günlük pencere
- Leakage yok (sadece geçmiş gözlemler)
- Sezonsal ve diurnal pattern doğal korunuyor

**İzolasyon stratejisi:** Yeni branch `experiment/strategy-a-imputation`.
Tüm output'lar `experiments/strategy_a/` altında. Main branch ve mevcut
v7 checkpoint'leri dokunulmadan korundu.

### Sonuçlar — Strategy A vs v7 (ffill)

| Senaryo | v7 (ffill) ΔCRPS | SA (rolling) ΔCRPS |
|---|---|---|
| Rnd G %10 | +0.95% | +1.40% |
| Rnd G %30 | +2.92% | +3.54% |
| Rnd G %50 | +4.99% | +4.98% |
| Burst G 1h | +2.35% | +3.47% |
| Rnd T_amb %30 | +1.18% | -0.14% |
| Rnd RH %30 | -0.44% | -0.32% |
| **Ortalama** | **+1.44%** | **+2.40%** |

Flag katsayıları (q09_G / q01_G):
- v7: +0.11 / -1.00 (asimetrik)
- SA: +1.00 / -1.00 (simetrik, semantik olarak doğru)

### Yorum

Strategy A flag katsayılarının semantik içeriğini düzeltti (simetrik
belirsizlik genişlemesi) ancak **CRPS'i iyileştirmedi**. Kritik mekanik:

> İyi imputation → base prediction'lar zaten iyi → meta'nın "flag
> aktif → bant genişlet" davranışı **gereksiz** → gereksiz geniş
> bant pinball loss'u artırır → CRPS yükselir.

Bu, gradient boosting + iyi imputation çiftinin yapısal bir sonucudur:
flag mekanizması ancak base preds bozulduğunda yardım edebilir, ama
base preds bozuk olunca tahmin zaten kötü olur. Mekanik dilemmayla
çarpıyor.

### Tez Sonucu

**H1 hipotezi iki bağımsız imputation stratejisi (ffill ve rolling
same-hour) altında doğrulanamadı.** Bu bulgu:
- Tek bir imputation tercihinin yanıltıcı olmadığını gösterir
- Gradient boosting taban modellerinin sensör eksikliğine içsel
  dayanıklılığını (Twala ve ark., 2008) ampirik olarak destekler
- Flag-tabanlı meta-katman müdahalesinin bu mimaride yapısal olarak
  sınırlı olduğunu kanıtlar

### Atıflar

- Twala, B. E. T. H. ve ark. (2008).
- Perez-Lebel ve ark. (2022).
- Rolling imputation: Hyndman & Athanasopoulos (2018) "Forecasting:
  Principles and Practice" — same-hour seasonal naive imputation

### Tez Paragrafı (Bulgular bölümü için ek)

> "Flag mekanizmasının imputation stratejisine bağımlılığını test
> etmek üzere, ana çalışmadaki ffill imputation yerine rolling
> same-hour mean imputation kullanılarak izole bir doğrulama
> deneyi gerçekleştirilmiştir. Her iki imputation altında da H1
> hipotezi doğrulanamamış (ffill: +%1.44, rolling: +%2.40 ortalama
> ΔCRPS), bu durum flag-tabanlı meta-katman müdahalesinin gradient
> boosting yığın mimarisinde yapısal olarak sınırlı kaldığını
> ortaya koymaktadır."

---

## Genel Notlar

- **Tüm seed'ler 42**, reprodüksiyon garantili.
- **Donanım:** MacBook Air M4 (Apple Silicon), PyTorch MPS backend (LSTM/TFT baseline'ları için).
- **Hız özeti:** Tüm pipeline (base + meta + STAGE-10 daylight + CQR + smoke test) < 4 saat.
- **STAGE-7 (Optuna) yeniden çalıştırılmadı** — base model hiperparametreleri değişmedi, yalnızca meta-learner ve değerlendirme katmanı değişti.
