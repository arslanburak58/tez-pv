"""
STAGE-8 tam koşum — 9 senaryo × meta_models_robust_v2.

Senaryolar:
  Random G: %10 / %20 / %30 / %50
  Burst G:  1h / 6h / 24h  (%30 yoğunluk — birden fazla ardışık pencere)
  Sensör:   T_amb %30 / RH %30

Her senaryo için:
  - with_flags  : meta_robust_v2 + gerçek flag değerleri
  - zero_flags  : meta_robust_v2 + flag sütunları sıfırlanmış (counterfactual)
  - Daylight mask (cos_zenith > 0.087) + CQR k=2.0
  - CRPS / MAE / Coverage / Pinball_q01 / Pinball_q09
  - DM testi → Holm-Bonferroni düzeltmesi
"""

import logging
import math
import random
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

from evaluation.robustness import diebold_mariano_test
from models.base_learners import META_COLS
from models.meta_learner import FLAG_COLS, enrich_x_meta, predict_intervals
from models.baselines import evaluate_quantiles
from evaluation.comparison import enforce_monotonicity, normalize_stacked_preds
from evaluation.cqr import apply_locally_scaled

# ── Sabitler ──────────────────────────────────────────────────────────────────

DAYLIGHT_THRESHOLD: float = 0.087
CQR_K:             float = 2.0
BURST_INTENSITY:   float = 0.30   # toplam burst kapsamı ≈ %30
SENSOR_RATE:       float = 0.30   # sensör-özgü senaryolarda missingness oranı
NOCT:              float = 46.0

# Sensör → flag eşlemesi
SENSOR_TO_FLAG: dict[str, str] = {
    "G":     "is_G_missing",
    "T_amb": "is_Tamb_missing",
    "RH":    "is_RH_missing",
}

# ── 1. Yükle ──────────────────────────────────────────────────────────────────

log.info("Joblib'ler yükleniyor...")
ds  = joblib.load("data/processed/dataset.joblib")
bm  = joblib.load("data/processed/base_models.joblib")
mm  = joblib.load("data/processed/meta_models_robust_v2.joblib")

X_test = ds["X_test"].copy()
y_arr  = np.asarray(ds["y_test"], dtype=np.float64)

dl_mask = X_test["cos_zenith"].to_numpy(dtype=np.float64) > DAYLIGHT_THRESHOLD
log.info("X_test: %s  |  daylight: %d / %d (%.1f%%)",
         X_test.shape, dl_mask.sum(), len(dl_mask), 100 * dl_mask.mean())

# Veri sıklığını indeks'ten otomatik çıkar (saatlik ise 1h, 5dk ise 5min)
if isinstance(X_test.index, pd.DatetimeIndex) and len(X_test) > 1:
    _diff = (X_test.index[1] - X_test.index[0]).total_seconds() / 60.0
    FREQ_MINUTES: int = int(max(1, round(_diff)))
else:
    FREQ_MINUTES = 60   # varsayılan: saatlik
log.info("Veri sıklığı: %d dk/satır", FREQ_MINUTES)


# ── 2. Yardımcı fonksiyonlar ───────────────────────────────────────────────────

def _recompute_tcell(X: pd.DataFrame) -> pd.DataFrame:
    """G veya T_amb değiştiyse T_cell yeniden hesapla."""
    if "T_cell" in X.columns and "T_amb" in X.columns and "G" in X.columns:
        X = X.copy()
        X["T_cell"] = X["T_amb"] + X["G"] * (NOCT - 20.0) / 800.0
    return X


def _apply_corruption(
    X: pd.DataFrame,
    sensor: str,
    row_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Seçilen satırlarda sensor → NaN → ffill/bfill; T_cell yeniden hesapla.
    Döndürür: (X_corrupted, flags_df)
    """
    X_c = X.copy()
    flag_col = SENSOR_TO_FLAG[sensor]

    # Mevcut flag'leri al (clean data'da hepsi 0)
    existing_flags = {fc: np.zeros(len(X_c), dtype=int) for fc in FLAG_COLS}

    # Korupsiyon uygula
    if sensor in X_c.columns:
        X_c.loc[X_c.index[row_mask], sensor] = np.nan
        X_c[sensor] = X_c[sensor].ffill().bfill()

    # T_cell güncelle
    if sensor in ("G", "T_amb"):
        X_c = _recompute_tcell(X_c)

    # Flag güncelle
    existing_flags[flag_col] = row_mask.astype(int)
    flags_df = pd.DataFrame(existing_flags, index=X_c.index)

    return X_c, flags_df


def _apply_multi_sensor_corruption(
    X: pd.DataFrame,
    sensors: list[str],
    masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Birden fazla sensörü aynı anda corrupt et (burst senaryo için)."""
    X_c = X.copy()
    existing_flags = {fc: np.zeros(len(X_c), dtype=int) for fc in FLAG_COLS}

    for sensor, row_mask in masks.items():
        if sensor not in X_c.columns:
            continue
        X_c.loc[X_c.index[row_mask], sensor] = np.nan
        X_c[sensor] = X_c[sensor].ffill().bfill()
        flag_col = SENSOR_TO_FLAG[sensor]
        existing_flags[flag_col] = np.maximum(
            existing_flags[flag_col], row_mask.astype(int)
        )

    # T_cell: G veya T_amb etkilendiyse
    if "G" in masks or "T_amb" in masks:
        X_c = _recompute_tcell(X_c)

    flags_df = pd.DataFrame(existing_flags, index=X_c.index)
    return X_c, flags_df


def _make_burst_mask(n: int, burst_steps: int, intensity: float, rng) -> np.ndarray:
    """
    Ardışık pencereler halinde burst maskesi oluştur.
    Hedef: toplamda ~intensity oranında satır.
    Non-overlapping pencereler rastgele yerleştirilir.
    """
    target_rows = int(n * intensity)
    n_bursts    = max(1, math.ceil(target_rows / burst_steps))

    mask   = np.zeros(n, dtype=bool)
    starts = list(range(0, n - burst_steps + 1, burst_steps))  # adım = pencere
    rng.shuffle(starts)

    covered = 0
    for s in starts:
        if covered >= target_rows:
            break
        end = min(s + burst_steps, n)
        mask[s:end] = True
        covered += end - s

    return mask


def _predict(X_c, flags_df, use_real_flags: bool) -> dict[str, np.ndarray]:
    """Base models → X_meta → meta_robust_v2 → tahmin."""
    base_preds = {col: np.asarray(bm[col].predict(X_c), dtype=np.float64)
                  for col in META_COLS}
    X_meta_df = pd.DataFrame(base_preds, index=X_c.index)

    if use_real_flags:
        X_meta_12 = enrich_x_meta(X_meta_df, flags_df)
    else:
        zero_flags = pd.DataFrame(
            {c: np.zeros(len(X_c), dtype=int) for c in FLAG_COLS},
            index=X_c.index,
        )
        X_meta_12 = enrich_x_meta(X_meta_df, zero_flags)

    raw  = normalize_stacked_preds(predict_intervals(mm, X_meta_12))
    raw  = enforce_monotonicity(raw)
    raw  = enforce_monotonicity(apply_locally_scaled(raw, CQR_K))
    return raw


def _apply_daylight(y, preds, mask):
    n = len(preds["q01"])
    m = mask[:n]
    return y[:n][m], {k: v[m] for k, v in preds.items()}


def _crps_per_obs(y, preds):
    def pb(p, q): r = y - p; return np.where(r >= 0, q * r, (q - 1) * r)
    return (pb(preds["q01"], 0.1) + pb(preds["q05"], 0.5) + pb(preds["q09"], 0.9)) / 3.0


# ── 3. Senaryo tanımları ───────────────────────────────────────────────────────

rng = np.random.default_rng(42)
n   = len(X_test)

SCENARIOS: list[dict] = []

# Random G %10 / %20 / %30 / %50
for rate in (0.10, 0.20, 0.30, 0.50):
    mask = rng.random(n) < rate
    SCENARIOS.append({
        "name":    f"random_G_{int(rate*100):02d}pct",
        "label":   f"Rnd G %{int(rate*100)}",
        "sensor":  "G",
        "mask":    mask,
        "multi":   False,
    })

# Burst G 1h / 6h / 24h  (%30 yoğunluk)
for hours in (1, 6, 24):
    steps_per_hour = max(1, 60 // FREQ_MINUTES)
    burst_steps    = hours * steps_per_hour
    burst_mask     = _make_burst_mask(n, burst_steps, BURST_INTENSITY, rng)
    SCENARIOS.append({
        "name":   f"burst_G_{hours}h",
        "label":  f"Burst G {hours}h",
        "sensor": "G",
        "mask":   burst_mask,
        "multi":  False,
    })

# Sensör-özgü: T_amb %30 / RH %30
for sensor in ("T_amb", "RH"):
    mask = rng.random(n) < SENSOR_RATE
    SCENARIOS.append({
        "name":   f"sensor_{sensor}_30pct",
        "label":  f"Rnd {sensor} %30",
        "sensor": sensor,
        "mask":   mask,
        "multi":  False,
    })

log.info("%d senaryo tanımlandı.", len(SCENARIOS))

# ── 4. Her senaryo için tahmin + metrik ───────────────────────────────────────

log.info("Senaryolar çalıştırılıyor...")

results: list[dict] = []
t_start = time.time()

for i, sc in enumerate(SCENARIOS):
    t0 = time.time()
    log.info("[%d/%d] %s ...", i + 1, len(SCENARIOS), sc["name"])

    X_c, flags_df = _apply_corruption(X_test, sc["sensor"], sc["mask"])

    # with_flags & zero_flags tahminleri
    p_with = _predict(X_c, flags_df, use_real_flags=True)
    p_zero = _predict(X_c, flags_df, use_real_flags=False)

    # Daylight filtresi
    y_dl, pw = _apply_daylight(y_arr, p_with, dl_mask)
    _,    pz = _apply_daylight(y_arr, p_zero, dl_mask)

    # Metrikler
    m_w = evaluate_quantiles(y_dl, pw)
    m_z = evaluate_quantiles(y_dl, pz)

    # DM testi
    crps_w  = _crps_per_obs(y_dl, pw)
    crps_z  = _crps_per_obs(y_dl, pz)
    dm      = diebold_mariano_test(crps_z, crps_w)   # noflags - flags → pozitif = flags iyi

    delta_crps = m_w["crps"] - m_z["crps"]
    delta_pct  = delta_crps / m_z["crps"] * 100 if m_z["crps"] > 0 else float("nan")
    corrupt_pct = float(sc["mask"].mean()) * 100

    results.append({
        "name":         sc["name"],
        "label":        sc["label"],
        "corrupt_pct":  corrupt_pct,
        "crps_with":    m_w["crps"],
        "crps_zero":    m_z["crps"],
        "delta_crps":   delta_crps,
        "delta_pct":    delta_pct,
        "mae_with":     m_w["mae"],
        "mae_zero":     m_z["mae"],
        "cov_with":     float(np.mean((y_dl >= pw["q01"]) & (y_dl <= pw["q09"]))),
        "cov_zero":     float(np.mean((y_dl >= pz["q01"]) & (y_dl <= pz["q09"]))),
        "pb01_with":    m_w["pinball_q01"],
        "pb09_with":    m_w["pinball_q09"],
        "dm_stat":      dm["dm_stat"],
        "dm_p_raw":     dm["p_value"],
        "mean_diff":    dm["mean_diff"],
        "y_dl":         y_dl,       # DM Holm-Bonferroni için sakla
        "crps_w_obs":   crps_w,
        "crps_z_obs":   crps_z,
    })

    log.info("  %.1fs | CRPS_w=%.4f CRPS_z=%.4f Δ=%.2f%% DM_p=%.2e",
             time.time() - t0, m_w["crps"], m_z["crps"], delta_pct, dm["p_value"])

log.info("Toplam süre: %.1f dk", (time.time() - t_start) / 60)

# ── 5. Holm-Bonferroni düzeltmesi ─────────────────────────────────────────────

p_raw = np.array([r["dm_p_raw"] for r in results])
n_tests = len(p_raw)

# Holm-Bonferroni: sıralı p-değerlerine düzeltme uygula
order   = np.argsort(p_raw)
p_adj   = np.ones(n_tests)
for rank, idx in enumerate(order):
    p_adj[idx] = min(1.0, p_raw[idx] * (n_tests - rank))
# Monotonluk garantisi
for i in range(1, n_tests):
    p_adj[order[i]] = max(p_adj[order[i]], p_adj[order[i - 1]])

for i, r in enumerate(results):
    r["dm_p_adj"]      = float(p_adj[i])
    r["significant"]   = bool(p_adj[i] < 0.05)

# ── 6. Master tablo ───────────────────────────────────────────────────────────

print("\n" + "═" * 110)
print("STAGE-8 MASTER ROBUSTNESS TABLOSU  (daylight, CQR k=2.0, meta_robust_v2)")
print("═" * 110)
header = (
    f"{'Senaryo':<22}  {'Corrupt%':>8}  "
    f"{'CRPS_with':>10}  {'CRPS_zero':>10}  {'ΔCRPS%':>8}  "
    f"{'Cov_with':>9}  {'PB01_w':>7}  {'PB09_w':>7}  "
    f"{'DM_p_adj':>10}  {'Anlam':>6}"
)
print(header)
print("─" * 110)

for r in results:
    sig = "✓" if r["significant"] else " "
    print(
        f"{r['label']:<22}  {r['corrupt_pct']:>7.1f}%  "
        f"{r['crps_with']:>10.4f}  {r['crps_zero']:>10.4f}  {r['delta_pct']:>+8.2f}%  "
        f"{r['cov_with']:>9.4f}  {r['pb01_with']:>7.4f}  {r['pb09_with']:>7.4f}  "
        f"{r['dm_p_adj']:>10.2e}  {sig:>6}"
    )

print("═" * 110)

# Özet
sig_count   = sum(1 for r in results if r["significant"])
mean_delta  = float(np.mean([r["delta_pct"] for r in results]))
neg_count   = sum(1 for r in results if r["delta_crps"] < 0)

print(f"\nAnlamlı senaryo: {sig_count} / {len(results)}")
print(f"Ortalama ΔCRPS:  {mean_delta:+.2f}%")
print(f"Flags iyileştirdi (CRPS↓): {neg_count} / {len(results)} senaryo")

# Tez iddiası yargısı
VERDICT_SIG_THRESH   = len(results) // 2   # ≥ %50 senaryo anlamlı
VERDICT_DELTA_THRESH = -5.0                 # ortalama ≥ -%5 iyileşme

if sig_count >= VERDICT_SIG_THRESH and mean_delta <= VERDICT_DELTA_THRESH:
    verdict = "EVET — 'sensor failure robust' iddiası kanıtlanmış ✓"
elif sig_count >= VERDICT_SIG_THRESH:
    verdict = f"KISMÎ — Çoğunluk anlamlı ama ortalama Δ={mean_delta:.1f}% (hedef ≤-5%)"
elif mean_delta <= VERDICT_DELTA_THRESH:
    verdict = f"KISMÎ — İyi iyileşme ama sadece {sig_count}/{len(results)} anlamlı"
else:
    verdict = f"HAYIR — Yeterli anlamlılık yok ({sig_count}/{len(results)}, Δ={mean_delta:.1f}%)"

print(f"\nTez iddiası: {verdict}")
print("═" * 110)

# ── 7. Görsel 1: CRPS Bar Plot ────────────────────────────────────────────────

Path("figures").mkdir(parents=True, exist_ok=True)

labels       = [r["label"] for r in results]
crps_with    = [r["crps_with"]  for r in results]
crps_zero    = [r["crps_zero"]  for r in results]
x            = np.arange(len(labels))
width        = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width / 2, crps_with, width, label="with flags",  color="#2196F3", alpha=0.85)
b2 = ax.bar(x + width / 2, crps_zero, width, label="zero flags",  color="#FF7043", alpha=0.85)

# Anlamlı senaryolara * işareti
for i, r in enumerate(results):
    if r["significant"]:
        ymax = max(crps_with[i], crps_zero[i])
        ax.text(i, ymax + 0.008, "*", ha="center", va="bottom", fontsize=12, color="black")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("CRPS (↓ daha iyi)")
ax.set_title("STAGE-8: CRPS — with flags vs zero flags  (* Holm-Bonferroni p<0.05)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig("figures/robustness_crps_bar.png", dpi=150)
fig.savefig("figures/robustness_crps_bar.pdf")
plt.close(fig)
log.info("Kaydedildi: figures/robustness_crps_bar.png/pdf")

# ── 8. Görsel 2: Heatmap (senaryo × metrik) ───────────────────────────────────

heat_data = pd.DataFrame({
    "CRPS_w":   [r["crps_with"]  for r in results],
    "CRPS_z":   [r["crps_zero"]  for r in results],
    "ΔCRPS%":   [r["delta_pct"]  for r in results],
    "Cov_w":    [r["cov_with"]   for r in results],
    "MAE_w":    [r["mae_with"]   for r in results],
    "PB01_w":   [r["pb01_with"]  for r in results],
    "PB09_w":   [r["pb09_with"]  for r in results],
}, index=labels)

fig, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(heat_data.values.astype(float), aspect="auto", cmap="RdYlGn_r")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

ax.set_xticks(range(len(heat_data.columns)))
ax.set_xticklabels(heat_data.columns, fontsize=9)
ax.set_yticks(range(len(heat_data)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_title("STAGE-8: Robustness Heatmap  (senaryo × metrik)")

for i in range(len(heat_data)):
    for j, val in enumerate(heat_data.values[i]):
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                color="white" if abs(val) > 0.8 * np.abs(heat_data.values).max() else "black")

plt.tight_layout()
fig.savefig("figures/robustness_heatmap.png", dpi=150)
fig.savefig("figures/robustness_heatmap.pdf")
plt.close(fig)
log.info("Kaydedildi: figures/robustness_heatmap.png/pdf")

# ── 9. Görsel 3: Coverage degradation ────────────────────────────────────────

# Temiz (bozulmamış) coverage referansı
flags_clean  = pd.DataFrame({c: np.zeros(len(X_test), dtype=int) for c in FLAG_COLS},
                              index=X_test.index)
base_preds_c = {col: np.asarray(bm[col].predict(X_test), dtype=np.float64) for col in META_COLS}
X_meta_c     = pd.DataFrame(base_preds_c, index=X_test.index)
X_meta_12_c  = enrich_x_meta(X_meta_c, flags_clean)
raw_c        = normalize_stacked_preds(predict_intervals(mm, X_meta_12_c))
raw_c        = enforce_monotonicity(raw_c)
raw_c        = enforce_monotonicity(apply_locally_scaled(raw_c, CQR_K))
y_clean_dl   = y_arr[:len(raw_c["q01"])][dl_mask[:len(raw_c["q01"])]]
p_clean_dl   = {k: v[dl_mask[:len(raw_c["q01"])]] for k, v in raw_c.items()}
cov_clean    = float(np.mean((y_clean_dl >= p_clean_dl["q01"]) & (y_clean_dl <= p_clean_dl["q09"])))

cov_with_list = [r["cov_with"] for r in results]
cov_zero_list = [r["cov_zero"] for r in results]

fig, ax = plt.subplots(figsize=(12, 4))
ax.axhline(cov_clean, color="green", linestyle="--", linewidth=1.5, label=f"Temiz (bozulmamış) = {cov_clean:.3f}")
ax.plot(labels, cov_with_list, "o-", color="#2196F3", label="with flags",  markersize=6)
ax.plot(labels, cov_zero_list, "s--", color="#FF7043", label="zero flags", markersize=6)
ax.axhspan(0.75, 0.85, alpha=0.12, color="green", label="Hedef bant [0.75, 0.85]")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Coverage (q01–q09 bant kapsama oranı)")
ax.set_ylim(max(0, min(cov_with_list + cov_zero_list) - 0.05), 1.02)
ax.set_title("STAGE-8: Coverage Degradation — 9 senaryo")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig("figures/coverage_degradation.png", dpi=150)
fig.savefig("figures/coverage_degradation.pdf")
plt.close(fig)
log.info("Kaydedildi: figures/coverage_degradation.png/pdf")

print(f"\nGörseller kaydedildi: figures/robustness_crps_bar, robustness_heatmap, coverage_degradation")
