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

### 2.1 Durum — ANA VERİ DOSYALARI EKSİK

> **UYARI:** `data/raw/pvod/datasets/` altında yalnızca McClear yardımcı dosyaları bulunmaktadır.  
> 10 istasyon CSV'si (`Station_0.csv` … `Station_9.csv`) ve `metadata.csv` **eksik**.  
> Kaynak: PVOD README → _"Please download the datasets from the website of the journal Solar Energy."_  
> İndirme adresi: http://www.doi.org/10.11922/sciencedb.01094

Eksik dosya listesi:

```
Station_0.csv  Station_1.csv  Station_2.csv  Station_3.csv  Station_4.csv
Station_5.csv  Station_6.csv  Station_7.csv  Station_8.csv  Station_9.csv
metadata.csv
```

### 2.2 Beklenen Yapı (kaynak: `src/pvodataset.py`)

**Genel Bilgi (hedeflenen):**

| Özellik | Değer |
|---------|-------|
| Toplam kayıt | 271,968 |
| İstasyon sayısı | 10 (Hebei, Çin) |
| Örnekleme frekansı | 15 dakika |
| Zaman aralığı | 2019–2020 |
| Kaynak | Yao vd. (2021), Solar Energy, doi:10.1016/j.solener.2021.09.050 |

**Beklenen sütunlar:**

| Sütun | Kaynak | Birim | Açıklama |
|-------|--------|-------|----------|
| `date_time` | — | datetime | Zaman damgası |
| `nwp_globalirrad` | NWP | W/m² | Küresel ışınım (model) |
| `nwp_dirrectirrad` | NWP | W/m² | Direkt ışınım (model) |
| `nwp_temperature` | NWP | °C | Hava sıcaklığı (model) |
| `nwp_humidity` | NWP | % | Bağıl nem (model) |
| `nwp_windspeed` | NWP | m/s | Rüzgar hızı (model) |
| `nwp_winddirection` | NWP | ° | Rüzgar yönü (model) |
| `nwp_pressure` | NWP | hPa | Basınç (model) |
| `lmd_totalirrad` | LMD | W/m² | Toplam ışınım (ölçüm) |
| `lmd_diffuseirrad` | LMD | W/m² | Difüz ışınım (ölçüm) |
| `lmd_temperature` | LMD | °C | Sıcaklık (ölçüm) |
| `lmd_pressure` | LMD | hPa | Basınç (ölçüm) |
| `lmd_winddirection` | LMD | ° | Rüzgar yönü (ölçüm) |
| `lmd_windspeed` | LMD | m/s | Rüzgar hızı (ölçüm) |
| `power` | LMD | kW | **Hedef değişken — PV gücü** |

### 2.3 Mevcut McClear Yardımcı Dosyalar

CAMS McClear v3.1 açık-gökyüzü ışınım modeli çıktıları. STAGE-3'te `G₀` (ekstraterrestrial + clearsky referans) üretimi için kullanılacak.

| Dosya | İstasyon | Satır | Zaman Aralığı |
|-------|---------|-------|---------------|
| `s5_clr_data.csv` | s5 | 500 | 2019-03-04 → 2019-03-09 |
| `s7_clr_data.csv` | s7 | 1,536 | 2019-03-05 → 2019-03-20 |
| `s7_clr_data_17-19.csv` | s7 | 192 | 2019-03-17 → 2019-03-19 |
| `s7_clr_data_6-7.csv` | s7 | 96 | 2019-03-05 → 2019-03-06 |
| `s8_clr_data.csv` | s8 | 96 | 2019-03-05 → 2019-03-06 |

McClear sütunları: `date, TOA, GHI, BHI, DHI, BNI` (W/m², 15 dk, UTC)

---

## 3. STAGE-2 Sonrası Eylem Listesi

| # | Eylem | Öncelik |
|---|-------|---------|
| 1 | PVOD Station_0-9.csv + metadata.csv indir (sciencedb.01094) | **ACİL** |
| 2 | DKASC: 2023'e sızan 288 satırı at | Yüksek |
| 3 | DKASC: tekrarlayan 3,456 timestamp'i çöz (ilkini tut) | Yüksek |
| 4 | DKASC: Wind_Speed %47.6 eksik → STAGE-3'te missingness flag | Yüksek |
| 5 | DKASC: Wind_Direction aykırı değerleri kırp [0, 360] | Orta |
| 6 | DKASC: Radiation_Diffuse_Tilted max=97322 → üst sınır kırp | Orta |
| 7 | DKASC: Sıcaklık < −10°C satırlarını at (3,047 adet) | Orta |
| 8 | PVOD gelince EDA bölümünü tamamla | PVOD sonrası |
