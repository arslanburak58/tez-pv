"""
STAGE-8: Robustness testleri — 3 eksen × 3 seviye = 9 senaryo.

Eksenler:
  random: %10 / %25 / %50 rastgele NaN maskeleme (tüm sensörler)
  burst:  1h / 6h / 24h sürekli kesinti (rastgele başlangıç, tüm sensörler)
  sensor: G / T_amb / RH sensörü tamamen eksik

API:
    RobustnessScenario              — senaryo tanımı (dataclass)
    ALL_SCENARIOS                   — 9 senaryo listesi
    apply_scenario(X, flags, s)     → (X_corrupted, flags_updated)
    build_predict_fn(...)           → PredictFn
    evaluate_predictions(y, preds)  → dict[str, float]
    run_all_scenarios(...)          → dict[str, pd.DataFrame]
    diebold_mariano_test(l1, l2)    → dict
    plot_heatmap(df, metric, path)  → Path

Kurallar:
    - Bozulma test setine uygulanır, train/val dokunulmaz
    - Bozulan sensör değerleri 0 ile doldurulur; flags 1'e güncellenir
    - k_t, T_cell → kaynak sensör bozulunca 0'a sıfırlanır
    - Serileştirme: joblib (pickle yasak)
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from models.base_learners import ALGOS, META_COLS, QUANTILES, _col_name, pinball_loss
from models.meta_learner import FLAG_COLS, enrich_x_meta, predict_intervals

random.seed(42)
np.random.seed(42)

log = logging.getLogger(__name__)

# ── Sabitler ───────────────────────────────────────────────────────────────────

RANDOM_LOSS_RATES: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50)
BURST_DURATIONS_H: tuple[int, ...]   = (1, 6, 24)
SENSOR_LOSS_TARGETS: tuple[str, ...] = ("G", "T_amb", "RH")
FREQ_MINUTES_DEFAULT: int            = 5

# Hangi sensör bozulunca hangi türev özellik sıfırlanır
DERIVED_ZERO_MAP: dict[str, list[str]] = {
    "G":     ["k_t"],
    "T_amb": ["T_cell"],
}

SENSOR_FLAG_MAP: dict[str, str] = {
    "G":          "is_G_missing",
    "T_amb":      "is_Tamb_missing",
    "RH":         "is_RH_missing",
    "wind_speed": "is_wind_missing",
}

# Tahmin anahtarı → quantile değeri
PRED_Q_MAP: dict[str, float] = {
    "meta_q01": 0.1,
    "meta_q05": 0.5,
    "meta_q09": 0.9,
}

PredictFn = Callable[[pd.DataFrame, pd.DataFrame], dict[str, np.ndarray]]


# ── RobustnessScenario ─────────────────────────────────────────────────────────

@dataclass
class RobustnessScenario:
    name:   str
    axis:   str          # "random" | "burst" | "sensor"
    level:  str          # "10pct"/"20pct"/"30pct"/"50pct" | "1h"/"6h"/"24h" | "G"/"Tamb"/"RH"
    params: dict[str, Any] = field(default_factory=dict)


ALL_SCENARIOS: list[RobustnessScenario] = [
    # Rastgele kayıp
    RobustnessScenario("random_10pct", "random", "10pct", {"rate": 0.10}),
    RobustnessScenario("random_20pct", "random", "20pct", {"rate": 0.20}),
    RobustnessScenario("random_30pct", "random", "30pct", {"rate": 0.30}),
    RobustnessScenario("random_50pct", "random", "50pct", {"rate": 0.50}),
    # Burst kayıp
    RobustnessScenario("burst_1h",  "burst", "1h",  {"hours": 1}),
    RobustnessScenario("burst_6h",  "burst", "6h",  {"hours": 6}),
    RobustnessScenario("burst_24h", "burst", "24h", {"hours": 24}),
    # Sensör-özgü kayıp
    RobustnessScenario("sensor_G",    "sensor", "G",    {"target": "G"}),
    RobustnessScenario("sensor_Tamb", "sensor", "Tamb", {"target": "T_amb"}),
    RobustnessScenario("sensor_RH",   "sensor", "RH",   {"target": "RH"}),
]


# ── apply_scenario ─────────────────────────────────────────────────────────────

def _corrupt_columns(
    X: pd.DataFrame,
    flags: pd.DataFrame,
    cols: list[str],
    row_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Belirtilen satır maskesinde cols'u 0'a set et, flags'i 1'e güncelle.
    Türev özellikler (k_t, T_cell) da sıfırlanır.
    """
    X = X.copy()
    flags = flags.copy()

    for col in cols:
        if col not in X.columns:
            continue
        X.loc[X.index[row_mask], col] = 0.0
        flag_col = SENSOR_FLAG_MAP.get(col)
        if flag_col and flag_col in flags.columns:
            flags.loc[flags.index[row_mask], flag_col] = 1

        # Türev özellik sıfırla
        for derived in DERIVED_ZERO_MAP.get(col, []):
            if derived in X.columns:
                X.loc[X.index[row_mask], derived] = 0.0

    return X, flags


def apply_scenario(
    X_test: pd.DataFrame,
    flags_test: pd.DataFrame,
    scenario: RobustnessScenario,
    freq_minutes: int = FREQ_MINUTES_DEFAULT,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test setine senaryo uygula; bozulmuş (X, flags) döndür.

    Bozulma yalnızca sensör sütunlarına ve türev özelliklerine uygulanır.
    Train/val'e dokunulmaz.
    """
    rng = np.random.default_rng(seed)
    n = len(X_test)
    sensor_cols = [c for c in SENSOR_FLAG_MAP if c in X_test.columns]

    if scenario.axis == "random":
        rate: float = scenario.params["rate"]
        mask = rng.random(n) < rate
        return _corrupt_columns(X_test, flags_test, sensor_cols, mask)

    elif scenario.axis == "burst":
        hours: int = scenario.params["hours"]
        steps = int(hours * 60 / freq_minutes)
        max_start = max(0, n - steps)
        start = int(rng.integers(0, max_start + 1))
        mask = np.zeros(n, dtype=bool)
        mask[start: start + steps] = True
        return _corrupt_columns(X_test, flags_test, sensor_cols, mask)

    elif scenario.axis == "sensor":
        target: str = scenario.params["target"]
        mask = np.ones(n, dtype=bool)  # tüm satırlar
        return _corrupt_columns(X_test, flags_test, [target], mask)

    else:
        raise ValueError(f"Bilinmeyen eksen: {scenario.axis}")


# ── build_predict_fn ───────────────────────────────────────────────────────────

def build_predict_fn(
    base_models: dict[str, Any],
    meta_models: dict[str, Any],
    use_flags: bool = True,
) -> PredictFn:
    """
    Eğitilmiş modeller üzerinden tahmin fonksiyonu oluştur.

    Args:
        base_models:  {col_name: fitted_model} — 9 taban model
        meta_models:  {"meta_q01": Ridge, ...} — 3 meta-model
        use_flags:    True → 13 özellik (9 OOF + 4 flag), False → 9 özellik

    Returns:
        PredictFn: (X_corrupted, flags_updated) → {"meta_q01": arr, ...}
    """
    def predict(X_corrupted: pd.DataFrame, flags_updated: pd.DataFrame) -> dict[str, np.ndarray]:
        base_preds: dict[str, np.ndarray] = {}
        for algo in ALGOS:
            for q in QUANTILES:
                col = _col_name(algo, q)
                base_preds[col] = np.asarray(base_models[col].predict(X_corrupted))

        X_meta = pd.DataFrame(base_preds, columns=META_COLS, index=X_corrupted.index)

        if use_flags:
            X_in = enrich_x_meta(X_meta, flags_updated)
        else:
            X_in = X_meta

        return predict_intervals(meta_models, X_in)

    return predict


# ── evaluate_predictions ───────────────────────────────────────────────────────

def _per_obs_pinball(
    y: np.ndarray, y_pred: np.ndarray, q: float
) -> np.ndarray:
    """Gözlem başına pinball kaybı (vektörleştirilmiş)."""
    r = y - y_pred
    return np.where(r >= 0, q * r, (q - 1) * r)


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    preds: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Tahmin dict'inden metrik hesapla.

    preds anahtarları: "meta_q01", "meta_q05", "meta_q09"

    Returns:
        dict — pinball_q01/q05/q09, crps, mae, rmse, coverage
    """
    y = np.asarray(y_true)
    q01 = preds["meta_q01"]
    q05 = preds["meta_q05"]
    q09 = preds["meta_q09"]

    pb01 = float(np.mean(_per_obs_pinball(y, q01, 0.1)))
    pb05 = float(np.mean(_per_obs_pinball(y, q05, 0.5)))
    pb09 = float(np.mean(_per_obs_pinball(y, q09, 0.9)))

    return {
        "pinball_q01": pb01,
        "pinball_q05": pb05,
        "pinball_q09": pb09,
        "crps":        (pb01 + pb05 + pb09) / 3,
        "mae":         float(np.mean(np.abs(y - q05))),
        "rmse":        float(np.sqrt(np.mean((y - q05) ** 2))),
        "coverage":    float(np.mean((y >= q01) & (y <= q09))),
    }


# ── run_all_scenarios ──────────────────────────────────────────────────────────

def run_all_scenarios(
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    flags_test: pd.DataFrame,
    predict_fn_flags: PredictFn,
    predict_fn_noflags: PredictFn,
    freq_minutes: int = FREQ_MINUTES_DEFAULT,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Tüm 9 senaryoyu çalıştır; flags ve noflags modellerini karşılaştır.

    Returns:
        dict:
            "flags":    pd.DataFrame (10 × metrik) — baseline + 9 senaryo
            "noflags":  pd.DataFrame (10 × metrik)
            "dm":       pd.DataFrame (9 × DM test sonuçları)
    """
    y = np.asarray(y_test)
    scenarios_to_run = [None] + ALL_SCENARIOS  # None = baseline (bozulmamış)

    rows_flags: list[dict] = []
    rows_noflags: list[dict] = []
    rows_dm: list[dict] = []

    for s in scenarios_to_run:
        if s is None:
            name = "baseline"
            X_c, fl_c = X_test, flags_test
        else:
            name = s.name
            X_c, fl_c = apply_scenario(X_test, flags_test, s, freq_minutes, seed)

        preds_f  = predict_fn_flags(X_c, fl_c)
        preds_nf = predict_fn_noflags(X_c, fl_c)

        m_f  = evaluate_predictions(y, preds_f)
        m_nf = evaluate_predictions(y, preds_nf)

        rows_flags.append({"scenario": name, **m_f})
        rows_noflags.append({"scenario": name, **m_nf})

        if s is not None:
            # Per-gözlem CRPS → DM testi
            pb01_f  = _per_obs_pinball(y, preds_f["meta_q01"],  0.1)
            pb05_f  = _per_obs_pinball(y, preds_f["meta_q05"],  0.5)
            pb09_f  = _per_obs_pinball(y, preds_f["meta_q09"],  0.9)
            crps_f  = (pb01_f + pb05_f + pb09_f) / 3

            pb01_nf = _per_obs_pinball(y, preds_nf["meta_q01"], 0.1)
            pb05_nf = _per_obs_pinball(y, preds_nf["meta_q05"], 0.5)
            pb09_nf = _per_obs_pinball(y, preds_nf["meta_q09"], 0.9)
            crps_nf = (pb01_nf + pb05_nf + pb09_nf) / 3

            dm = diebold_mariano_test(crps_nf, crps_f)  # noflags - flags: pozitif = flags iyi
            rows_dm.append({"scenario": name, **dm})

        log.info(
            "%s | flags crps=%.4f | noflags crps=%.4f",
            name,
            m_f["crps"],
            m_nf["crps"],
        )

    df_flags   = pd.DataFrame(rows_flags).set_index("scenario")
    df_noflags = pd.DataFrame(rows_noflags).set_index("scenario")
    df_dm      = pd.DataFrame(rows_dm).set_index("scenario")

    return {"flags": df_flags, "noflags": df_noflags, "dm": df_dm}


# ── diebold_mariano_test ───────────────────────────────────────────────────────

def diebold_mariano_test(
    losses_1: np.ndarray,
    losses_2: np.ndarray,
) -> dict[str, float | bool]:
    """
    Harvey-Leybourne-Newbold düzeltmeli Diebold-Mariano testi (h=1, iki yönlü).

    H0: E[losses_1 - losses_2] = 0  (eşit doğruluk)
    H1: ≠ 0

    Pozitif DM istatistiği → losses_1 daha büyük → model 2 daha iyi.

    Args:
        losses_1: referans modelin gözlem başına kayıpları
        losses_2: alternatif modelin gözlem başına kayıpları

    Returns:
        dm_stat, p_value, mean_diff, significant (α=0.05)
    """
    d = np.asarray(losses_1, dtype=float) - np.asarray(losses_2, dtype=float)
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West otokovaryans tahmincisi (lag=0, h=1 adım ötesi)
    gamma_0 = np.mean((d - d_bar) ** 2)
    var_dm = gamma_0 / n

    if var_dm <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_diff": float(d_bar), "significant": False}

    dm_stat = d_bar / np.sqrt(var_dm)

    # HLN küçük örnek düzeltmesi (h=1): t(n-1) dağılımı
    hln_factor = np.sqrt((n + 1 - 2 + 1.0 / n) / n)
    dm_hln = dm_stat * hln_factor
    p_value = float(2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1)))

    return {
        "dm_stat":    float(dm_hln),
        "p_value":    p_value,
        "mean_diff":  float(d_bar),
        "significant": p_value < 0.05,
    }


# ── plot_heatmap ───────────────────────────────────────────────────────────────

def plot_heatmap(
    results: dict[str, pd.DataFrame],
    metric: str = "crps",
    save_dir: str = "figures",
) -> Path:
    """
    Senaryo × model (flags/noflags) ısı haritası kaydet.

    Returns:
        Path — kaydedilen PNG
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df_flags   = results["flags"].drop(index="baseline", errors="ignore")
    df_noflags = results["noflags"].drop(index="baseline", errors="ignore")

    data = pd.DataFrame({
        "flags":   df_flags[metric],
        "noflags": df_noflags[metric],
    })

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label=metric.upper())

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Flags", "No Flags"])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_title(f"Robustness — {metric.upper()} (9 senaryo)")

    for i, row in enumerate(data.values):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if val > data.values.max() * 0.7 else "black")

    plt.tight_layout()
    out = Path(save_dir) / f"robustness_heatmap_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Heatmap kaydedildi: %s", out)
    return out
