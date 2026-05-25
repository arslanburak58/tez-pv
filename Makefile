.PHONY: help install commit gunce leakage activate test

help:
	@echo ""
	@echo "  Tez PV — Otomasyon Komutları"
	@echo "  ──────────────────────────────────────────"
	@echo "  make install            Paketleri kur"
	@echo "  make commit S=2 M='...' Stage commit at"
	@echo "  make doc M='...'        DOC commit at"
	@echo "  make fix M='...'        FIX commit at"
	@echo "  make gunce              gunce.md'yi aç"
	@echo "  make leakage            Leakage kontrol scripti"
	@echo "  make activate           Venv aktivasyon bilgisi"
	@echo "  make test               Birim testleri çalıştır"
	@echo "  make log                Git log (son 10 commit)"
	@echo ""

install:
	source tez-env/bin/activate && pip install -r requirements.txt

commit:
	@if [ -z "$(S)" ] || [ -z "$(M)" ]; then \
		echo "Kullanım: make commit S=<stage_no> M='<açıklama>'"; exit 1; fi
	git add .
	git commit -m "STAGE-$(S): $(M)"
	@echo "✓ Commit: STAGE-$(S): $(M)"

doc:
	@if [ -z "$(M)" ]; then echo "Kullanım: make doc M='açıklama'"; exit 1; fi
	git add .
	git commit -m "DOC: $(M)"

fix:
	@if [ -z "$(M)" ]; then echo "Kullanım: make fix M='açıklama'"; exit 1; fi
	git add .
	git commit -m "FIX: $(M)"

gunce:
	open docs/gunce.md

leakage:
	source tez-env/bin/activate && python scripts/check_leakage.py

activate:
	@echo "Venv aktif etmek için terminale şunu yaz:"
	@echo "  source $(PWD)/tez-env/bin/activate"

test:
	source tez-env/bin/activate && python -m pytest tests/ -v

log:
	git log --oneline -10
