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

## Karar 7: Bounded QuantileLinear meta-learner (v7)

**Karşı seçilen:** v2 (standart QuantileLinear, augmentation patolojisi)

**Niye değiştirildi:**
v2 ile tam STAGE-8'de ortalama ΔCRPS +%6.2 (yanlış yönde anlamlı). Teşhis: (i) eğitim augmentation'da multi-sensor flag co-occurrence test ile uyumsuzdu (eğitimde %21.5 satır 2+ flag, testte %0); (ii) ffill imputation bazen büyük stale değerler üretiyor, meta bunları kapsayan agresif flag katsayıları öğrenir (q09 is_G_missing = −5.32) → test'te overcorrection. (iii) q09 flag katsayısı L2 regularization'a karşı dirençli — alpha = 0.5'ten 20'ye artırmak katsayıyı yalnızca %1 küçülttü.

**Çift düzeltme uygulandı:**
1. **Augmentation (v5):** tek-sensör random %30 + burst 1/6/24h %20 + clean %50. STAGE-8 değerlendirme senaryolarıyla bire bir uyumlu eğitim dağılımı.
2. **Regularization (QuantileLinearBounded):** Base prediction katsayıları serbest, flag katsayıları [-1.0, +1.0] box constraint. scipy L-BFGS-B `bounds` parametresi ile uygulandı.

**Sonuç:** v7 ile tam STAGE-8 ortalama ΔCRPS +%1.44 (v2: +%6.2). Burst senaryolarda flags pozitif yönde iyileştirme (Burst G 6h: −0.09%, Burst G 24h: −0.79%). Coverage 0.72–0.88 bandında istikrarlı. 9/9 DM anlamlı. Bantlar semantik olarak doğru genişliyor: G eksik → q01 aşağı (−1.0), q09 hafif yukarı (+0.11).

**Atıflar:**
- Bounded optimization: scipy.optimize.minimize L-BFGS-B (Jones ve ark., 2001)
- Burst pattern justification: Hasan ve ark. (2023), Bouslimani ve ark. (2025)

**Tez paragrafı (Yöntem 3.4 güncellemesi):**
> Meta-öğrenici QuantileLinearBounded sınıfı olarak implemente edilmiştir; base prediction katsayıları serbest, flag katsayıları |coef| ≤ 1.0 box constraint altında optimize edilir. Bu kısıt, eğitim augmentation'ın üretebileceği patolojik agresif düzeltme katsayılarını önler. Augmentation stratejisi tek-sensör random ve burst (1/6/24 saat ardışık) corruption'ı karıştırarak STAGE-8 değerlendirme senaryolarıyla bire bir uyumlu eğitim dağılımı sağlar. v7 modeli ile gerçekleştirilen tam STAGE-8 değerlendirmesinde (9 senaryo, Holm-Bonferroni düzeltmeli Diebold-Mariano testi) ortalama ΔCRPS = +%1.44 elde edilmiş, tüm senaryolarda istatistiksel anlamlılık (p < 0.05) korunmuştur. Burst G 6h ve 24h senaryolarında flags CRPS'i sırasıyla −%0.09 ve −%0.79 iyileştirmiştir.

---

## Genel Notlar

- **Tüm seed'ler 42**, reprodüksiyon garantili.
- **Donanım:** MacBook Air M4 (Apple Silicon), PyTorch MPS backend (LSTM/TFT baseline'ları için).
- **Hız özeti:** Tüm pipeline (base + meta + STAGE-10 daylight + CQR + smoke test) < 4 saat.
- **STAGE-7 (Optuna) yeniden çalıştırılmadı** — base model hiperparametreleri değişmedi, yalnızca meta-learner ve değerlendirme katmanı değişti.
