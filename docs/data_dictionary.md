# Veri Sözlüğü — STAGE-2 EDA Raporu

**Oluşturulma:** Mayıs 2026  
**Durum:** DKASC ✓ tam | PVOD ✗ ana dosyalar eksik (aşağıya bak)

---

## 1. DKASC Alice Springs (2010–2022)

### 1.1 Genel Bilgi

| Özellik | Değer |
|---------|-------|
| Kaynak | Desert Knowledge Australia Solar Centre (DKASC) |
| Dosyalar | `data/raw/dkasc/dkasc_2010.csv` … `dkasc_2022.csv` (13 dosya) |
| Toplam satır | 1,361,812 |
| Toplam sütun | 196 |
| Zaman aralığı | 2010-01-01 00:00 → 2022-12-31 23:55 |
| Örnekleme frekansı | 5 dakika |
| Zaman damgası biçimi | `YYYY-MM-DD HH:MM:SS` (string, parse gerekir) |
| Tekrarlayan timestamp | 3,456 satır |
| 2023'e sızan satır | 288 (dkasc_2022.csv'nin son kayıtları) |

### 1.2 Hava İstasyonu Sütunları

Tüm sütunlar `101_DKA_WeatherStation_` önekiyle gelir.

| Kısa Ad | Tam Sütun Adı | Eksik % | Min | Max | Notlar |
|---------|--------------|---------|-----|-----|--------|
| `G` | `Global_Horizontal_Radiation` | 0.1 | −985.7 | 2725.6 | W/m², hedef özellik |
| `T_amb` | `Weather_Temperature_Celsius` | 0.1 | −40.0 | 61.9 | °C |
| `RH` | `Weather_Relative_Humidity` | 0.1 | 0.0 | 131.2 | % |
| `wind_speed` | `Wind_Speed` | **47.6** | −923.9 | 54.4 | m/s — **ciddi eksik** |
| `wind_dir` | `Wind_Direction` | 0.3 | −19862 | 30152 | derece — **aykırı değer** |
| `DHI` | `Diffuse_Horizontal_Radiation` | 0.1 | 0.0 | 2135.0 | W/m² |
| `G_tilt` | `Radiation_Global_Tilted` | **34.9** | −0.012 | 1553.4 | W/m² — ciddi eksik |
| `DHI_tilt` | `Radiation_Diffuse_Tilted` | **31.8** | −0.019 | 97322.6 | W/m² — max aykırı |
| `rainfall` | `Weather_Daily_Rainfall` | 0.1 | 0.0 | 67.2 | mm |

### 1.3 Güç Sütunları

| Özellik | Değer |
|---------|-------|
| Aktif güç sütun sayısı | 49 (invertör + faz bazlı) |
| Sütun adı kalıbı | `{ID}_DKA_{Modül}_{Faz}_Active_Power` |
| MasterMeter1 Power aralığı | −1.86 … 241.0 kW |
| MasterMeter1 ortalama | 46.8 kW |
| Negatif güç satırı (anomali) | 868,751 satır (%64'ü gece saatleri) |

### 1.4 Genel Eksik Veri

| Kapsam | Değer |
|--------|-------|
| Toplam hücre | 266,915,152 |
| Eksik hücre | 56,989,710 |
| **Genel eksik oran** | **%21.4** |
| Ana neden | Farklı yıllarda aktif olmayan invertör sütunları |

### 1.5 Anomali Tespiti

| Anomali | Satır Sayısı | Değerlendirme |
|---------|-------------|---------------|
| GHI < 0 | 1 | İhmal edilebilir |
| GHI > 1500 W/m² | 13 | Temizlenecek (fiziksel max ~1361) |
| Sıcaklık > 55°C | 2 | Temizlenecek |
| Sıcaklık < −10°C | 3,047 | Alice Springs'te imkânsız; sensör arızası |
| Nem > %100 | 8,549 | %100'e kırpılacak |
| Rüzgar < 0 | 15 | Temizlenecek |
| Wind_Direction aykırı | Var | −19862 / +30152 → [0, 360] kırp |
| Radiation_Diffuse_Tilted max | 97,322 W/m² | Kesin sensör arızası; temizlenecek |

### 1.6 Yıllık Satır Dağılımı

| Yıl | Satır |
|-----|-------|
| 2010 | 105,120 |
| 2011 | 105,191 |
| 2012 | 105,684 |
| 2013 | 105,365 |
| 2014 | 105,399 |
| 2015 | 105,408 |
| 2016 | 105,696 |
| 2017 | 105,366 |
| 2018 | 105,408 |
| 2019 | 103,193 |
| 2020 | 105,670 |
| 2021 | 100,833 |
| 2022 | 103,191 |
| 2023 (sızan) | 288 |

### 1.7 STAGE-3 İçin Kullanılacak Sütunlar

Tezin fiziksel öznitelik pipeline'ı bu sütunlara dayanır:

```
G       = 101_DKA_WeatherStation_Global_Horizontal_Radiation   [W/m²]
T_amb   = 101_DKA_WeatherStation_Weather_Temperature_Celsius   [°C]
RH      = 101_DKA_WeatherStation_Weather_Relative_Humidity     [%]
wind    = 101_DKA_WeatherStation_Wind_Speed                    [m/s]
target  = 96_DKA_MasterMeter1_Active_Power                     [kW]
```

---

## 2. PVOD v1.0

### 2.1 Genel Bilgi

| Özellik | Değer |
|---------|-------|
| Kaynak | Yao vd. (2021), Solar Energy, doi:10.1016/j.solener.2021.09.050 |
| Dosyalar | `data/raw/pvod/datasets/station00.csv` … `station09.csv` + `metadata.csv` |
| Toplam kayıt | **271,968** ✓ |
| İstasyon sayısı | 10 (Hebei, Çin) |
| Örnekleme frekansı | 15 dakika |
| Zaman aralığı | 2018-06-30 → 2019-06-13 (istasyona göre değişir) |
| Saat dilimi | UTC (yerel UTC+8) |
| Genel eksik veri | **%0.0** (temiz veri seti) |

### 2.2 Sütun Yapısı

| Sütun | Kaynak | Birim | Min | Max | Açıklama |
|-------|--------|-------|-----|-----|----------|
| `date_time` | — | datetime | 2018-08-15 | 2019-06-13 | Zaman damgası |
| `nwp_globalirrad` | NWP | W/m² | 0.0 | 964.1 | Küresel ışınım (model) |
| `nwp_directirrad` | NWP | W/m² | 0.0 | 910.1 | Direkt ışınım (model) |
| `nwp_temperature` | NWP | °C | −19.8 | 41.1 | Hava sıcaklığı (model) |
| `nwp_humidity` | NWP | % | 4.6 | 100.0 | Bağıl nem (model) |
| `nwp_windspeed` | NWP | m/s | 0.05 | 19.7 | Rüzgar hızı (model) |
| `nwp_winddirection` | NWP | ° | 0.0 | 360.0 | Rüzgar yönü (model) |
| `nwp_pressure` | NWP | hPa | 869.8 | 1044.8 | Basınç (model) |
| `lmd_totalirrad` | LMD | W/m² | 0.0 | **1838.0** | Toplam ışınım (ölçüm) — anomali var |
| `lmd_diffuseirrad` | LMD | W/m² | 0.0 | 1122.0 | Difüz ışınım (ölçüm) |
| `lmd_temperature` | LMD | °C | −23.9 | 41.6 | Sıcaklık (ölçüm) |
| `lmd_pressure` | LMD | hPa | 867.8 | 1049.1 | Basınç (ölçüm) |
| `lmd_winddirection` | LMD | ° | 0.0 | 360.0 | Rüzgar yönü (ölçüm) |
| `lmd_windspeed` | LMD | m/s | 0.0 | 16.0 | Rüzgar hızı (ölçüm) |
| `power` | LMD | MW | 0.0 | 35.1 | **Hedef değişken — PV gücü** |

### 2.3 İstasyon Bazlı Özet

| İstasyon | Satır | Kapasite (kWp) | PV Teknoloji | Zaman Aralığı | Tekrar TS | Anomali |
|----------|-------|----------------|-------------|---------------|-----------|---------|
| station00 | 28,896 | 6,600 | Poly-Si | 2018-08-15 → 2019-06-13 | 1 | — |
| station01 | 33,408 | 20,000 | Poly-Si | 2018-06-30 → 2019-06-13 | 0 | — |
| station02 | 30,432 | 17,000 | **Mono-Si** | 2018-07-22 → 2019-06-10 | 0 | — |
| station03 | 14,688 | 20,000 | Poly-Si | 2019-01-11 → 2019-06-13 | 0 | — |
| station04 | 33,408 | 20,000 | Poly-Si | 2018-06-30 → 2019-06-13 | 0 | lmd_irrad>1500: 10 satır |
| station05 | 9,696 | **35,000** | Poly-Si | 2019-03-04 → 2019-06-13 | 0 | — |
| station06 | 31,104 | 15,000 | Poly-Si | 2018-07-13 → 2019-06-13 | 0 | — |
| station07 | 32,928 | 20,000 | Poly-Si | 2018-06-30 → 2019-06-13 | 0 | lmd_irrad>1500: 1 satır |
| station08 | 33,120 | 20,000 | Poly-Si | 2018-06-30 → 2019-06-13 | 0 | — |
| station09 | 24,288 | 20,000 | Poly-Si | 2018-09-25 → 2019-06-13 | 0 | — |
| **Toplam** | **271,968** | — | — | 2018-06-30 → 2019-06-13 | 1 | 11 satır irrad anomali |

### 2.4 Anomali Tespiti

| Anomali | Satır | Değerlendirme |
|---------|-------|---------------|
| `lmd_totalirrad` > 1500 W/m² | 11 (station04: 10, station07: 1) | Fiziksel max ~1361 W/m²; temizlenecek |
| Negatif güç | 0 | Temiz |
| Negatif ışınım | 0 | Temiz |
| Nem > %100 | 0 | Temiz |
| Tekrarlayan timestamp | 1 (station00) | İhmal edilebilir |
| Eksik veri | **%0.0** | Veri seti son derece temiz |

### 2.5 STAGE-3 İçin Kullanılacak Sütunlar

```
G       = lmd_totalirrad      [W/m²]  (ölçüm — ana kaynak)
G_nwp   = nwp_globalirrad    [W/m²]  (model — karşılaştırma)
T_amb   = lmd_temperature    [°C]
RH      = nwp_humidity       [%]     (LMD'de RH yok → NWP kullanılır)
wind    = lmd_windspeed      [m/s]
target  = power              [MW]
```

### 2.6 McClear Yardımcı Dosyalar

CAMS McClear v3.1 açık-gökyüzü ışınım modeli çıktıları. STAGE-3'te clearsky index `k_t = G/G_clearsky` üretimi için kullanılacak.

| Dosya | İstasyon | Satır | Zaman Aralığı |
|-------|---------|-------|---------------|
| `s5_clr_data.csv` | station05 | 500 | 2019-03-04 → 2019-03-09 |
| `s7_clr_data.csv` | station07 | 1,536 | 2019-03-05 → 2019-03-20 |
| `s8_clr_data.csv` | station08 | 96 | 2019-03-05 → 2019-03-06 |

McClear sütunları: `date, TOA, GHI, BHI, DHI, BNI` (W/m², 15 dk, UTC)

---

## 3. STAGE-2 Sonrası Eylem Listesi

| # | Eylem | Öncelik |
|---|-------|---------|
| 1 | DKASC: 2023'e sızan 288 satırı at | Yüksek |
| 2 | DKASC: tekrarlayan 3,456 timestamp'i çöz (ilkini tut) | Yüksek |
| 3 | DKASC: Wind_Speed %47.6 eksik → STAGE-3'te missingness flag olarak ekle | Yüksek |
| 4 | DKASC: Sıcaklık < −10°C satırlarını at (3,047 adet) | Orta |
| 5 | DKASC: Nem > %100 → %100'e kırp (8,549 satır) | Orta |
| 6 | DKASC: Wind_Direction aykırı değerleri kırp [0, 360] | Orta |
| 7 | DKASC: Radiation_Diffuse_Tilted max=97322 → üst sınır kırp | Orta |
| 8 | PVOD: lmd_totalirrad > 1500 W/m² olan 11 satırı at | Düşük |
| 9 | PVOD: station00 tekrarlayan 1 timestamp'i at | Düşük |
