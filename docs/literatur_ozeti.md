(a) Fiziksel Öznitelik Mühendisliği ve pvlib

1. Duffie ve Beckman (2013)
Tam Atıf: Duffie, J. A. ve Beckman, W. A. (2013). Solar engineering of thermal processes (4. baskı). John Wiley & Sons.
Özet: Güneş radyasyonu (difüz ve direkt), saat açıları ve fotovoltaik hücre sıcaklığı gibi temel astronomik ve termal kavramların matematiksel kökenlerini sunan alanın en popüler başvuru kitabıdır. Güneş mühendisliği tasarımları için net formülasyonlar sağlar.
İş Paketi: İP 1 (Fiziksel Öznitelik Mühendisliği) ve İP 3 (Açık-Gökyüzü B-Planı / Fallback).
Atiea (2025) Karşılaştırması: Atiea'nın modeli yalnızca makine öğrenmesinin ham ilişkilerine güvenirken, sizin bu esere dayanarak modele zenit açısı veya T 
cell
​	
  formüllerini eklemeniz "fizik kısıtlı model (physics-informed)" argümanınızın Atiea'ya karşı en büyük üstünlüğüdür.
2. Holmgren ve ark. (2018)
Tam Atıf: Holmgren, W. F., Hansen, C. W. ve Mikofski, M. A. (2018). pvlib python: a python package for modeling solar energy systems. Journal of Open Source Software, 3(29), 884.
Özet: Açık kaynaklı pvlib kütüphanesini tanıtan makaledir. Fiziksel ve astronomik güneş konumlandırma hesaplamaları ile hücre termodinamik modellerini Python programlama dilinde pratik, standart bir arayüze döker.
İş Paketi: İP 1 (Fiziksel Öznitelik Mühendisliği).
Atiea (2025) Karşılaştırması: Atiea ve ark. özellik çıkarımı (feature engineering) için sadece basit istatistiksel korelasyonlar kullanırken, pvlib kullanımı tezinize alan (domain) bilgisi katarak Atiea'nın tahmin çerçevesine ciddi bir mühendislik derinliği ekleyecektir.

--------------------------------------------------------------------------------
(b) Stacked Ensemble ve Quantile Regresyon

3. Akiba ve ark. (2019)
Tam Atıf: Akiba, T., Sano, S., Yanase, T., Ohta, T. ve Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD International Conference, 2623–2631.
Özet: "Define-by-run" prensibi ve TPE (Tree-structured Parzen Estimator) yapısı kullanan son derece hızlı bir hiperparametre optimizasyon çerçevesini tanıtır. Gereksiz iterasyonları asenkron biçimde budayarak (pruning) arama maliyetini düşürür.
İş Paketi: İP 2 (Bayesyen Optimizasyon).
Atiea (2025) Karşılaştırması: Atiea ve ark. da Optuna'yı kullanmıştır; dolayısıyla bu kaynak, tezinizin hiperparametre optimizasyon aşamasında Atiea'nın kanıtlanmış yaklaşımıyla doğrudan paralellik (destek) gösterir.
4. Atiea ve ark. (2025)
Tam Atıf: Atiea, M. A., Abdelghaffar, A. A., Ben Aribia, H., Noureddine, F. ve Shaheen, A. M. (2025). Photovoltaic power generation forecasting with Bayesian optimization and stacked ensemble learning. Results in Engineering, 26, 104950.
Özet: RFR, GBR ve KNN modellerini Seviye-0'da eğitip, LR meta-öğrenici ile sonuçları birleştiren ve bunu Bayesyen optimizasyon (Optuna) ile destekleyen bir makaledir. Çoklu veri setlerinde test edilerek makine öğrenmesi algoritmalarında doğruluğun zirveye taşındığı gösterilmiştir.
İş Paketi: İP 2 (Stacked Ensemble Tasarımı) ve İP 3 (Karşılaştırmalı Analiz).
Atiea (2025) Karşılaştırması: Tezinizin çıkış noktasıdır. Sizin çalışmanız bu makaleyi temel alır (destek) ancak modelin deterministik (tek noktalı) olması ve sensör kayıplarına direncin test edilmemesi (karşı argüman/geliştirme alanı) tezinizi bu makalenin bir üst versiyonu yapar.
5. Chen ve Guestrin (2016)
Tam Atıf: Chen, T. ve Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference, 785–794.
Özet: XGBoost'u sunan orijinal çalışmadır. Dağıtık veri işleyebilen, seyrek verilere (sparsity) duyarlı, matematiksel düzenlileştirme (regularization) barındıran en başarılı gradyan artırma ağaç sistemlerinden biridir.
İş Paketi: İP 2 (Seviye 0 Taban Modellerinin Eğitimi).
Atiea (2025) Karşılaştırması: Atiea ve ark. GBR kullansa da, sizin XGBoost'un "sparsity-aware (seyreklik farkındalığı)" özelliği ile modele girmeniz, Atiea'nın baş edemediği eksik veri yönetimi senaryolarında tezinize büyük avantaj sağlayacaktır.
6. Dorogush ve ark. (2018)
Tam Atıf: Dorogush, A. V., Ershov, V. ve Gulin, A. (2018). CatBoost: gradient boosting with categorical features support. arXiv preprint arXiv:1810.11363.
Özet: Simetrik (oblivious) karar ağaçları kullanan ve hedef sızıntısını (target leakage) engelleyen CatBoost algoritmasının metodolojisidir. Özellikle kategorik verileri doğrudan işleyebilmesiyle ünlüdür.
İş Paketi: İP 2 (Seviye 0 Taban Modellerinin Eğitimi).
Atiea (2025) Karşılaştırması: Atiea CatBoost'u ensemble modeline dahil etmemiştir. Sizin CatBoost'u kullanmanız, Seviye-1 meta-öğreniciye sızıntısız ve mimari olarak çok daha farklı (çeşitliliği artırılmış) tahminler sunarak Atiea'nın yığınlama (stacking) performansını aşmanızı sağlayacaktır.
7. Ke ve ark. (2017)
Tam Atıf: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. ve Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.
Özet: Geleneksel modellere göre eğitim sürelerini drastik şekilde azaltan LightGBM sistemini tanıtır. Özel veri örnekleme teknikleriyle donanım verimliliğini çok yükseltir.
İş Paketi: İP 2 (Seviye 0 Taban Modellerinin Eğitimi).
Atiea (2025) Karşılaştırması: Atiea'nın çalışmasında kullanılan modellerin uzun eğitim sürelerine bir eleştiri olarak, LightGBM'in getireceği büyük hız ve hesaplama verimliliği, özellikle 9 farklı taban modelin eğitileceği quantile regresyon sürecinde tezinizi donanımsal olarak uygulanabilir kılacaktır.
8. Koenker ve Bassett (1978)
Tam Atıf: Koenker, R. ve Bassett, G., Jr. (1978). Regression quantiles. Econometrica, 46(1), 33–50.
Özet: En küçük kareler yöntemine karşı geliştirilen Kantil (Quantile) regresyonun teorik ispatlarını sunar. Veri setlerindeki marjinal veya çarpık dağılımlar için belirli yüzdelik değerleri sağlam şekilde (robust) tahmin etmenin matematiksel temelidir.
İş Paketi: İP 2 (Olasılıksal Quantile Eğitim).
Atiea (2025) Karşılaştırması: Atiea (2025) çalışması yalnızca "nokta (deterministik)" tahmin yapar. Bu makale, tezinizdeki güven aralıkları (q=0.1 ve 0.9) üretiminin arkasındaki temeldir ve Atiea'nın deterministik limitasyonuna karşı koyduğunuz en güçlü yapısal farklılıktır.
9. Ting ve Witten (1999)
Tam Atıf: Ting, K. M. ve Witten, I. H. (1999). Issues in stacked generalization. Journal of Artificial Intelligence Research, 10, 271–289.
Özet: Stacked Generalization tekniğinde meta-öğrenicilerin optimizasyonunu işler. Özellikle çıktıların olasılık formatında olduğu durumlarda çoklu yanıtlı doğrusal modellerin (Ridge gibi) kullanılmasının hata oranını kesin biçimde düşürdüğünü ispatlar.
İş Paketi: İP 2 (Seviye 1 Meta-Öğrenici Tasarımı).
Atiea (2025) Karşılaştırması: Atiea meta-öğrenici olarak Lojistik/Lineer regresyon kullansa da istatistiksel arka planını zayıf bırakmıştır. Sizin meta-öğrenici tercihinizin (Ridge) olasılıksal formatta neden en ideal seçim olduğunu savunan ana "destek" kaynağınızdır.
10. Wolpert (1992)
Tam Atıf: Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241–259.
Özet: Makine öğrenmesindeki zayıf modellerin çıktılarının bir meta-algoritma tarafından çapraz doğrulama mekanizmasıyla birleştirildiği "Yığınlanmış Genelleme (Stacking)" mimarisinin kurucu makalesidir. Çapraz doğrulamanın model ağırlıklandırma felsefesini oluşturur.
İş Paketi: İP 2 (Olasılıksal Stacked Ensemble Tasarımı).
Atiea (2025) Karşılaştırması: Hem sizin tezinizin hem de Atiea (2025)'in ana mimari yaklaşımının literatürdeki tarihsel kökenidir. Atiea'nın günümüzdeki sonuçlarını destekleyen 30 yıllık güçlü bir teorik omurgadır.

--------------------------------------------------------------------------------
(c) Eksik Veri Yönetimi ve Prediction-Odaklı Yaklaşım

11. Bouslimani ve ark. (2025)
Tam Atıf: Bouslimani, M., Benbouzid-Si Tayeb, F., Amirat, Y. ve Benbouzid, M. (2025). Cyber-physical security in smart grids: A comprehensive guide to key research areas, threats, and countermeasures. Applied Sciences, 15(23), 12367.
Özet: Akıllı şebeke operasyonlarındaki Hizmet Reddi (DoS) ve Yanlış Veri Enjeksiyonu gibi siber-fiziksel saldırıları detaylandıran kapsamlı bir derlemedir. Sensör kayıplarının bir gerçeklik olduğunu ortaya koyar.
İş Paketi: İP 3 (Sistematik Dayanıklılık Testleri).
Atiea (2025) Karşılaştırması: Atiea (2025) eksik sensör ve iletişim kopmaları senaryolarını modellememiştir. Teziniz bu kaynak ile, makine öğrenmesi tahmin modellerinin gerçek şebeke koşullarındaki saldırılara/kopmalara karşı neden dayanıklı olması gerektiğini kanıtlayarak Atiea'yı sahanın gerisinde bırakır.
12. Hasan ve ark. (2023)
Tam Atıf: Hasan, M. K., Habib, A. A., Shukur, Z., Ibrahim, F., Islam, S. ve Razzaque, M. A. (2023). Review on cyber-physical and cyber-security system in smart grid: Standards, protocols, constraints, and recommendations. Journal of Network and Computer Applications, 209, 103540.
Özet: Akıllı şebeke veri alışverişlerindeki donanımsal arızalar ile güvenlik zafiyetlerini ve bu ağ yapılarına giren verilerin saflığını nasıl kaybettiğini teknik bir çerçevede anlatır.
İş Paketi: İP 3 (Dayanıklılık Testleri Konsepti).
Atiea (2025) Karşılaştırması: Atiea'nın kusursuz veri seti varsayımıyla çalışan modellerinin şebeke pratiğinde karşılaşacağı veri kısıtlamalarını belirtir. Projenizdeki test tasarımına güçlü bir (applicability) geçerlilik kazandırır.
13. Jones (1996)
Tam Atıf: Jones, M. P. (1996). Indicator and stratification methods for missing explanatory variables in multiple linear regression. Journal of the American Statistical Association, 91(433), 222–230.
Özet: Geleneksel regresyonda veri eksikliğini modele "bayrak değişkenler (missing indicators)" olarak eklemenin varyanslarda asimptotik ve kabul edilemez matematiksel sapmalar (bias) yaratacağını matematiksel formüllerle eleştiren istatistik makalesidir.
İş Paketi: İP 3 (Eksik Veri İtirazları / Literatür Kalkanı).
Atiea (2025) Karşılaştırması: Atiea ve ark. (2025) modellerinde eksik veri işleme stratejisi olarak standart yöntemler kullanır. Jones (1996) sizin bayrak matrisi yönteminize yapılabilecek "geleneksel istatistik" eleştirileridir. Siz bu itirazı diğer kaynaklarla (Sperrin) kıracaksınız.
14. Little ve Rubin (2019)
Tam Atıf: Little, R. J. A. ve Rubin, D. B. (2019). Statistical analysis with missing data (3. baskı). John Wiley & Sons.
Özet: Veri setlerinde "Rastgele Kayıp (MCAR, MAR, MNAR)" türlerinin temellerini kuran ders kitabıdır. Eksik verinin analizinde basit silme işlemlerinin analizi bozacağını ve nedensellik çıkarımlarını saptıracağını gösterir.
İş Paketi: İP 3 (Dayanıklılık ve Sensör Simülasyonları Tasarımı).
Atiea (2025) Karşılaştırması: Atiea'nın çalışmasında eksik veri silinerek veya medyanla doldurularak geçilmiştir. Little ve Rubin'in işaret ettiği nedensellik problemlerini baz alarak, kendi projenizin neden spesifik bir eksik veri stratejisi güttüğünü destekler.
15. Perez-Lebel ve ark. (2022)
Tam Atıf: Perez-Lebel, A., Varoquaux, G., Le Morvan, M., Josse, J. ve Poline, J.-B. (2022). Benchmarking missing-values approaches for predictive models on health databases. GigaScience, 11, giac013.
Özet: İstatistiksel eleştirilerin aksine, tahminsel (predictive) makine öğrenmesi modellerinde "eksiklik bayraklarının" model başarısını kayda değer şekilde artırdığını ağaç tabanlı (XGBoost vb.) modeller üzerinden ampirik verilerle ispatlar.
İş Paketi: İP 3 (Eksik Veri Bayrakları).
Atiea (2025) Karşılaştırması: Atiea'nın eksik veri doldurma zayıflığına karşı geliştirilen eksiklik bayrağı mimarinizi (Atiea bu yöntemi hiç test etmemiştir) savunan ve ağaç modellerinin doğasına atıf yapan güçlü bir deneysel dayanak sağlar.
16. Sisk ve ark. (2023)
Tam Atıf: Sisk, R., Sperrin, M. ve Martin, G. P. (2023). Imputation and missing indicators for handling missing data in the development and deployment of clinical prediction models: A simulation study. Statistical Methods in Medical Research, 32(8), 1461–1477.
Özet: Tıp literatüründeki algoritmaların pratik sahaya taşınmasında, kayıp veriler için missingness indicators kullanımının (model dağıtımında - deployment) modelleri son derece sağlam ve çalışabilir kıldığını ispatlar.
İş Paketi: İP 3 (Eksik Veri Bayrakları Teorisi).
Atiea (2025) Karşılaştırması: Atiea modelinin eksik sensör okuması anında hata verip çökeceğini, kendi modelinizin ise sahada (deployment phase) bu makaledeki prensiplerle sensör kopsa dahi tahmin yapmaya (robustness) devam edebileceğini vurgulamanızı sağlar.
17. Sperrin ve ark. (2020)
Tam Atıf: Sperrin, M., Martin, G. P., Sisk, R. ve Peek, N. (2020). Missing data should be handled differently for prediction than for description or causal explanation. Journal of Clinical Epidemiology, 125, 183–187.
Özet: Jones (1996) tarzı nedensellik (causality) odaklı eleştirilere karşı çıkarak; söz konusu eylem "nedensellik bulmak değil sadece tahmin (prediction) yapmak" olduğunda eksik veri bayraklarının kullanılmasının kesinlikle doğru olduğunu savunan temel metodoloji kalkanıdır.
İş Paketi: İP 3 (Eksik Veri Bayrakları İstatistiksel Savunması).
Atiea (2025) Karşılaştırması: Atiea'nın modeli gibi bir makine öğrenmesi ağının bir nedensellik amacı gütmediğini, bu nedenle tahmin (prediction) odağında kalınarak klasik istatistik eleştirilerinden muaf olunduğunu kanıtlayan nihai tez savunma silahınızdır.
18. Twala ve ark. (2008)
Tam Atıf: Twala, B. E. T. H., Jones, M. C. ve Hand, D. J. (2008). Good methods for coping with missing data in decision trees. Pattern Recognition Letters, 29(7), 950–956.
Özet: Karar ağaçlarında kayıp verilerin nasıl doğal olarak izole edilebileceğini analiz eder. Ağaç algoritmalarının dal (split) yönlerini kullanarak NaN değerlerle verimli biçimde başa çıkabildiğini deneysel olarak gösterir.
İş Paketi: İP 3 (Seviye 0 Taban Modellerinin Eksik Veri Davranışı).
Atiea (2025) Karşılaştırması: Atiea'nın RFR ve GBR'si yüksek doğruluk alsa da ağaçların veri eksikliğindeki doğal direncine atıf yapmamıştır. Bu makale, Seviye-0 ensemble'ınızın eksik verilerde gösterge matrisi dahi olmadan belirli bir toleransa sahip olduğunu över.

--------------------------------------------------------------------------------
(d) PV Tahmin Baseline Metodları (Klasik ML, LSTM, Transformer)

19. Chen ve Xu (2022)
Tam Atıf: Chen, Y. ve Xu, J. (2022). Solar and wind power data from the Chinese
State Grid Renewable Energy Generation Forecasting Competition. Scientific Data, 9(1), 310.
Özet: PVOD v1.0 veri setinin temel referansıdır. NWP ve sahadan gelen ölçümleri
birleştiren bu açık veri deposu, STAGE-2'de 10 istasyon ve 271.968 kayıtlık veri
temini için kullanılacaktır.
İş Paketi: İP 1 (Veri Toplama).
Atiea (2025) Karşılaştırması: Komşu — doğrudan karşılaştırma değil, veri zemini sağlar.
20. Ali ve ark. (2026)
Tam Atıf: Ali, W., Akhtar, F., Ullah, A. ve Kim, W. Y. (2026). A hybrid ensemble learning framework for accurate photovoltaic power prediction. Energies, 19(2), 450–465.
Özet: PVOD v1.0 veri seti üzerinde rastgele orman, XGBoost ve CatBoost kullanan ve zaman özniteliklerini başarıyla değerlendiren bir hibrit makine öğrenmesi uygulamasıdır. Genelleme konusunda ağaç tabanlıların gücünü gösterir.
İş Paketi: İP 1 ve İP 3 (Ağaç Tabanlı Baseline Modelleri).
Atiea (2025) Karşılaştırması: Atiea (2025) çalışmasını destekler. Ali ve ark. da bir ensemble modeli geliştirmiş olup, tezinizdeki ensemble'ın doğruluğunu ve "tree-based" yapıların etkinliğini doğrulayan güçlü ve çok güncel (2026) bir yan referanstır.
21. Kuhn ve Johnson (2013)
Tam Atıf: Kuhn, M. ve Johnson, K. (2013). Applied predictive modeling. Springer.
Özet: Model aşırı öğrenmesini (over-fitting) engelleme yöntemlerini, veri ayırma ve yeniden örnekleme stratejilerini anlatan uygulayıcılar için uçtan uca klasik bir tahmin modellemesi rehberidir.
İş Paketi: İP 1 ve İP 3 (Veri Hazırlama, Değerlendirme Protokolleri).
Atiea (2025) Karşılaştırması: Atiea'nın (2025) kullandığı K-fold çapraz doğrulama gibi temel test yöntemlerinin teorik kural kitabıdır. Tezinizin makine öğrenmesi test/doğrulama protokollerini sağlamlaştırır.
22. Lim ve ark. (2021)
Tam Atıf: Lim, B., Arık, S. Ö., Loeff, N. ve Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4), 1748–1764.
Özet: Zaman serileri için dikkat (attention) mekanizmasını optimize ederek statik, geçmiş ve gelecekte bilinen kısıtları mükemmel şekilde harmanlayan Derin Öğrenme modeli TFT'yi (Temporal Fusion Transformers) tanıtır.
İş Paketi: İP 3 (Karşılaştırmalı Analiz Baseline Modeli).
Atiea (2025) Karşılaştırması: Atiea (2025) kendi modelini sadece basit sığ (shallow) ML algoritmalarıyla karşılaştırmıştır. Siz TFT gibi olağanüstü kompleks ve başarılı bir Deep Learning modelini "baseline" koyarak Atiea'nın makalesinden çok daha sert bir rekabet testi kurmuş olacaksınız.
23. Omitaomu ve Niu (2021)
Tam Atıf: Omitaomu, O. A. ve Niu, H. (2021). Artificial intelligence techniques in smart grid: A survey. Smart Cities, 4(2), 548–568.
Özet: Akıllı şebeke (smart grid) yönetimindeki üretim kestirimi, hata kontrolü ve yük optimizasyonu alanında klasik makine öğrenmesi algoritmalarının nasıl konumlandığını gösteren geniş kapsamlı bir haritadır.
İş Paketi: İP 1 (Problem Tespiti ve Kapsam/Literatür Analizi).
Atiea (2025) Karşılaştırması: Atiea (2025)'in odaklandığı PV üretiminin akıllı şebeke operasyonlarındaki önemini (motivasyonu) geniş bir perspektifte açıklar; Atiea'nın genel çerçevesini destekler niteliktedir.
24. Salinas ve ark. (2020)
Tam Atıf: Salinas, D., Flunkert, V., Gasthaus, J. ve Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3), 1181–1191.
Özet: Amazon tarafından geliştirilen otoregresif Tekrarlayan Sinir Ağları (RNN) algoritması DeepAR'ı kullanarak çok boyutlu zaman serilerinde yüksek başarıyla olasılıksal tahminler (güven aralıkları) üreten çalışmadır.
İş Paketi: İP 3 (Karşılaştırmalı Olasılıksal Baseline Modeli).
Atiea (2025) Karşılaştırması: Atiea'nın çalışmasındaki "nokta tahmin (deterministik)" eksiğine karşı, hem derin öğrenme tabanlı olan hem de olasılıksal tahmin üretebilen bir RNN metodolojisi sunarak projenize Deep Learning alanında bir başka karşılaştırma tabanı sağlar.
25. Vaswani ve ark. (2017)
Tam Atıf: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. ve Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30, 5998–6008.
Özet: Geleneksel tekrarlayan sinir ağlarını kaldıran, sadece "öz-dikkat" (self-attention) mekanizmasına dayalı Transformer ağlarının temellerini atan, yapay zeka dünyasında devrim yaratan makaledir.
İş Paketi: İP 3 (Derin Öğrenme Baseline Altyapısı / TFT Teorisi).
Atiea (2025) Karşılaştırması: Atiea (2025) çalışmasında Attention (Dikkat) mekanizmalı hiçbir derin ağ bulunmaz. Bu kaynak, TFT baseline'ınızı açıklarken Atiea'nın hiç dokunmadığı Deep Learning mekanizmalarının çekirdeğini (self-attention) açıklar.

--------------------------------------------------------------------------------
(e) Olasılıksal Değerlendirme Metrikleri (Pinball, CRPS)

26. Gneiting ve Raftery (2007)
Tam Atıf: Gneiting, T. ve Raftery, A. E. (2007). Strictly proper scoring
rules, prediction, and estimation. Journal of the American Statistical
Association, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
Özet: Puanlama kurallarının teorik çerçevesini kuran temel makaledir.
CRPS'nin strictly proper bir scoring rule olduğunu kanıtlar ve quantile
tahmini için uygun puanlama kurallarını (pinball loss dahil) matematiksel
olarak türetir. Olasılıksal tahmin değerlendirmesinin istatistiksel
tutarlılığını garanti altına alır.
İş Paketi: İP 3 (Performans Değerlendirmesi) ve İP 4 (Raporlama).
Atiea (2025) Karşılaştırması: Atiea ve ark. yalnızca MAE, RMSE, R²
kullanmıştır. Bu makale tezin CRPS ve Pinball Loss kullanımının teorik
gerekçesidir; Atiea'nın değerlendirme vizyonunun ötesine geçtiğini
kanıtlar.
27. Wang ve ark. (2022)
Tam Atıf: Wang, W., Yang, D., Hong, T. ve Kleissl, J. (2022). An archived dataset from the ECMWF Ensemble Prediction System for probabilistic solar power forecasting. Solar Energy, 231, 112–125.
Özet: Küresel Güneş Radyasyonu (GHI) üzerine ECMWF'den elde edilmiş olasılıksal topluluk tahminleri (ensemble forecast) verisini sunar ve bu tahminlerin değerlendirmesinde CRPS (Continuous Ranked Probability Score) metodolojisinin nasıl pratik olarak işlendiğini gösterir.
İş Paketi: İP 3 (Metrik Değerlendirmesi / Quantile Skoru Hesaplama).
Atiea (2025) Karşılaştırması: Atiea (2025) çalışması olasılıksal ensemble modeli (probabilistic output) ve bu çıktıların skorlamasını barındırmaz. Bu makale, ensemble (yığın) çıktılarınızın güneş tahminciliğinde hangi endüstri metrikleriyle doğrulanması gerektiğini öğreterek Atiea'nın metodolojisine metrik bazında net bir üstünlük kurmanızı sağlar.
