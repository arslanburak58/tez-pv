"""
Strategy A vs main v7 karşılaştırma tablosu.

Çalıştırma:
  PYTHONPATH=. tez-env/bin/python experiments/strategy_a/scripts/compare_with_main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.meta_learner import FLAG_COLS, QUANTILES, _meta_key, _q_cols

EXPR_DIR    = ROOT / "experiments" / "strategy_a"
DATA_DIR    = EXPR_DIR / "data"
MODEL_DIR   = EXPR_DIR / "models"
MAIN_DATA   = ROOT / "data" / "processed"
MAIN_MODELS = ROOT / "data" / "processed"

# ── 1. Yükle ──────────────────────────────────────────────────────────────────

def _load_or_warn(path: Path):
    if not path.exists():
        print(f"  ! Bulunamadı: {path}")
        return None
    return joblib.load(path)


print("\n" + "=" * 100)
print("STRATEGY A vs MAIN v7 — KARŞILAŞTIRMA RAPORU")
print("=" * 100)

# Main v7 robust meta
mm_v7  = _load_or_warn(MAIN_MODELS / "meta_models_robust_v7.joblib")
# SA robust meta
mm_sa  = _load_or_warn(MODEL_DIR / "meta_models_robust_sa.joblib")
# STAGE-8 results
res_v7 = _load_or_warn(MAIN_DATA / "stage8_results_v7.joblib")
res_sa = _load_or_warn(DATA_DIR / "stage8_results_sa.joblib")

# ── 2. Flag katsayı karşılaştırması ──────────────────────────────────────────

print("\n" + "═" * 90)
print("FLAG KATSAYILARı — v7 (ffill) vs Strategy A (rolling)")
print("═" * 90)
print(f"{'Model':<30}  {'q09_G':>9}  {'q09_T':>9}  {'q09_RH':>9}"
      f"  {'q01_G':>9}  {'q01_T':>9}  {'q01_RH':>9}")
print("─" * 90)

rows = []
for tag, mm in [("v7 (ffill)", mm_v7), ("SA (rolling)", mm_sa)]:
    if mm is None:
        print(f"  {tag}: YÜKLENEMEDİ")
        continue
    c09 = [mm["meta_q09"].coef_[_q_cols(0.9).index(fc)] for fc in FLAG_COLS]
    c01 = [mm["meta_q01"].coef_[_q_cols(0.1).index(fc)] for fc in FLAG_COLS]
    rows.append((tag, c09, c01))
    print(f"{tag:<30}  {c09[0]:>9.4f}  {c09[1]:>9.4f}  {c09[2]:>9.4f}"
          f"  {c01[0]:>9.4f}  {c01[1]:>9.4f}  {c01[2]:>9.4f}")

if len(rows) == 2:
    _, c09_v7, c01_v7 = rows[0]
    _, c09_sa, c01_sa = rows[1]
    diff09 = [sa - v7 for sa, v7 in zip(c09_sa, c09_v7)]
    diff01 = [sa - v7 for sa, v7 in zip(c01_sa, c01_v7)]
    print(f"{'Δ (SA - v7)':<30}  {diff09[0]:>+9.4f}  {diff09[1]:>+9.4f}  {diff09[2]:>+9.4f}"
          f"  {diff01[0]:>+9.4f}  {diff01[1]:>+9.4f}  {diff01[2]:>+9.4f}")
    print()
    print("Yorum:")
    if abs(c09_sa[0]) < abs(c09_v7[0]):
        print(f"  v7 q09_G={c09_v7[0]:.4f} → SA q09_G={c09_sa[0]:.4f}  "
              f"Patoloji azaldı ({'%.1f' % (100*(abs(c09_sa[0])/abs(c09_v7[0])-1))}%)")
    else:
        print(f"  v7 q09_G={c09_v7[0]:.4f} → SA q09_G={c09_sa[0]:.4f}  "
              f"Değişim beklenmedik yönde")

print("═" * 90)

# ── 3. STAGE-8 Δ% tablosu ─────────────────────────────────────────────────────

if res_v7 is not None and res_sa is not None:
    # Senaryo sıralaması aynı olmalı
    print("\n" + "═" * 110)
    print("STAGE-8 ΔCRPS% KARŞILAŞTIRMASI — v7 (ffill) vs Strategy A (rolling)")
    print("+ = flags bozdu  |  - = flags iyileştirdi")
    print("═" * 110)
    print(f"{'Senaryo':<22}  {'ΔCRPS%_v7':>11}  {'ΔCRPS%_SA':>11}  "
          f"{'Fark':>9}  {'v7_Sig':>7}  {'SA_Sig':>7}  "
          f"{'Cov_v7':>8}  {'Cov_SA':>8}")
    print("─" * 110)

    v7_deltas: list[float] = []
    sa_deltas: list[float] = []

    # Match by name
    v7_by_name = {r["name"]: r for r in res_v7}
    sa_by_name = {r["name"]: r for r in res_sa}
    all_names  = [r["name"] for r in res_sa]  # SA order

    for name in all_names:
        rv = v7_by_name.get(name)
        rs = sa_by_name.get(name)
        if rv is None or rs is None:
            print(f"  {name}: eşleştirilemedi")
            continue
        diff = rs["delta_pct"] - rv["delta_pct"]
        v7_sig = "✓" if rv.get("significant") else " "
        sa_sig = "✓" if rs.get("significant") else " "
        print(f"{rs['label']:<22}  "
              f"{rv['delta_pct']:>+11.2f}%  {rs['delta_pct']:>+11.2f}%  "
              f"{diff:>+9.2f}  {v7_sig:>7}  {sa_sig:>7}  "
              f"{rv['cov_with']:>8.4f}  {rs['cov_with']:>8.4f}")
        v7_deltas.append(rv["delta_pct"])
        sa_deltas.append(rs["delta_pct"])

    print("═" * 110)
    v7_mean = float(np.mean(v7_deltas))
    sa_mean = float(np.mean(sa_deltas))
    v7_sig_n = sum(1 for r in res_v7 if r.get("significant"))
    sa_sig_n = sum(1 for r in res_sa if r.get("significant"))
    v7_neg_n = sum(1 for d in v7_deltas if d < 0)
    sa_neg_n = sum(1 for d in sa_deltas if d < 0)

    print(f"\n{'':22}  {'v7 (ffill)':>22}  {'SA (rolling)':>22}")
    print(f"{'Ort. ΔCRPS%':<22}  {v7_mean:>+22.2f}%  {sa_mean:>+22.2f}%")
    print(f"{'Anlamlı (DM p<0.05)':<22}  {v7_sig_n:>22d}  {sa_sig_n:>22d}")
    print(f"{'Flags iyileştirdi':<22}  {v7_neg_n:>22d}  {sa_neg_n:>22d}")

    print()
    print("SONUÇ:")
    if sa_mean < v7_mean:
        print(f"  Strategy A ({sa_mean:+.2f}%) < v7 ({v7_mean:+.2f}%)")
        print("  Rolling imputation flag patolojisini AZALTTI.")
    else:
        print(f"  Strategy A ({sa_mean:+.2f}%) >= v7 ({v7_mean:+.2f}%)")
        print("  Rolling imputation kayda değer iyileştirme sağlamadı.")

    if sa_neg_n > v7_neg_n:
        print(f"  Flags iyileştiren senaryo: v7={v7_neg_n} → SA={sa_neg_n} (artı)")
    elif sa_neg_n < v7_neg_n:
        print(f"  Flags iyileştiren senaryo: v7={v7_neg_n} → SA={sa_neg_n} (eksi, beklenmedik)")
    else:
        print(f"  Flags iyileştiren senaryo sayısı değişmedi: {sa_neg_n}")

else:
    if res_v7 is None:
        print("\n  ! STAGE-8 v7 sonuçları bulunamadı.")
        print("    run_stage8_v7.py'yi önce çalıştırın ve stage8_results_v7.joblib kaydedin.")
    if res_sa is None:
        print("\n  ! Strategy A STAGE-8 sonuçları bulunamadı.")
        print("    Önce run_pipeline_strategy_a.py çalıştırın.")

# ── 4. OOF korelasyon özeti ───────────────────────────────────────────────────

x_meta_sa  = _load_or_warn(DATA_DIR / "x_meta_sa.joblib")
x_meta_v7  = _load_or_warn(ROOT / "data" / "processed" / "x_meta.joblib")

if x_meta_sa is not None and x_meta_v7 is not None:
    Xm_sa = x_meta_sa[0] if isinstance(x_meta_sa, tuple) else x_meta_sa
    Xm_v7 = x_meta_v7[0] if isinstance(x_meta_v7, tuple) else x_meta_v7

    print("\n" + "═" * 80)
    print("OOF X_META ÖZET — v7 vs Strategy A")
    print("═" * 80)
    print(f"{'Stat':<20}  {'v7 (ffill)':>15}  {'SA (rolling)':>15}")
    print("─" * 80)

    for col in Xm_sa.columns[:9]:   # ilk 9 sütun: base model tahminleri
        if col not in Xm_v7.columns:
            continue
        v7_std = float(Xm_v7[col].std())
        sa_std = float(Xm_sa[col].std())
        v7_mn  = float(Xm_v7[col].mean())
        sa_mn  = float(Xm_sa[col].mean())
        print(f"{col:<20}  mean={v7_mn:>7.3f} std={v7_std:>6.3f}  "
              f"mean={sa_mn:>7.3f} std={sa_std:>6.3f}")
    print("═" * 80)
    print("  Düşük std → imputation patolojisi olduğunda base model tahminleri")
    print("  belirli değerlere yığılıyor anlamına gelir.")

print("\n" + "=" * 100)
print("KARŞILAŞTIRMA TAMAMLANDI")
print("=" * 100)
