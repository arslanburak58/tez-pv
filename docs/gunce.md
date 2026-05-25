## [Mayıs 2026] — STAGE-2 Devam Ediyor

Aktif adım: PVOD ana dosyaları eksik, indirilmesi gerekiyor
Sıradaki konuşmada: "PVOD indirildi" veya "STAGE-3'e geç" ile başlat
Aktif model    : Sonnet 4.6 / Extended Thinking: kapalı
Son güncelleme : Mayıs 2026
Tıkanıklık     : PVOD Station_0-9.csv + metadata.csv eksik

Tamamlanan:
- S0 Setup ✓
- S1 Literatür ✓ (harita oturumu + literatur_ozeti.md düzeltmeleri)
- Proje ortamı ✓ (~/Desktop/tez-pv, git, venv, CLAUDE.md, Makefile)
- Claude Code ✓ (çalışıyor, bağlamı doğru okuyor)
- S2 DKASC EDA ✓ (1,361,812 satır, 196 sütun, 5 dk — docs/data_dictionary.md)

Açık görevler:
- PVOD ana dosyaları indir: http://www.doi.org/10.11922/sciencedb.01094
  (Station_0.csv … Station_9.csv + metadata.csv → data/raw/pvod/datasets/)
- PVOD indirildikten sonra EDA tamamla, data_dictionary.md güncelle
- tez_workflow.md'ye AEMO/NREL B planı ve git adımları eklenecek
- STAGE-3: pvlib fiziksel öznitelik pipeline (DKASC üzerinde başlanabilir)

Sistem:
- claude.ai Projects → düşünme, yazım, karar
- Claude Code (terminal) → çalıştırma, git, dosya
- Makefile komutları: make help ile listele