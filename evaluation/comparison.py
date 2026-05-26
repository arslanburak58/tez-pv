"""
STAGE-10: Karşılaştırmalı analiz — master tablo, görsel analizler, istatistiksel testler.

6 model: stacked_flags, stacked_noflags, knn, svm, lstm, tft
Metrikler: MAE, RMSE, Pinball (q01/q05/q09), CRPS, Coverage, eğitim süresi

API:
    ModelResult                                    — tek model sonuç yapısı (TypedDict)
    normalize_stacked_preds(preds)                 → dict  (meta_q01→q01 çevirme)
    build_master_table(results)                    → pd.DataFrame (6 × 9)
    plot_master_table(df, save_dir)                → list[Path]
    plot_probability_bands(y, preds, name, dir, n) → list[Path]
    apply_holm_bonferroni(p_values)                → list[float]
    dm_pairwise(results)                           → pd.DataFrame
    plot_dm_heatmap(dm_df, save_dir)               → list[Path]
    plot_edge_ai_scatter(df, save_dir)             → list[Path]
    run_comparison(results, save_dir)              → dict[str, Any]

Kurallar:
    - preds dict anahtarları: "q01", "q05", "q09"  (stacked için normalize_stacked_preds)
    - DM pairwise: CRPS bazlı, HLN düzeltmeli, Holm-Bonferroni çoklu karşılaştırma düzeltmesi
    - Tüm görseller PNG + PDF olarak kaydedilir
    - Serileştirme: joblib (pickle yasak)
"""

import logging
import random
from itertools import combinations
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.robustness import diebold_mariano_test

random.seed(42)
np.random.seed(42)

log = logging.getLogger(__name__)

# ── Sabitler ───────────────────────────────────────────────────────────────────

MODEL_ORDER: list[str] = [
    "stacked_flags",
    "stacked_noflags",
    "knn",
    "svm",
    "lstm",
    "tft",
]

MODEL_LABELS: dict[str, str] = {
    "stacked_flags":   "Stacked (Flags)",
    "stacked_noflags": "Stacked (No Flags)",
    "knn":             "k-NN",
    "svm":             "SVM",
    "lstm":            "LSTM",
    "tft":             "Light TFT",
}

ALPHA: float = 0.05
DPI:   int   = 150
BAND_N_POINTS: int = 200


# ── ModelResult TypedDict ──────────────────────────────────────────────────────

class ModelResult(TypedDict):
    """
    Tek model için standart sonuç yapısı.

    preds        → {"q01": np.ndarray, "q05": np.ndarray, "q09": np.ndarray}
                   Stacked model için önce normalize_stacked_preds() çağır.
    metrics      → {"mae", "rmse", "pinball_q01", "pinball_q05", "pinball_q09",
                    "crps", "coverage"}
    y_true       → test seti gerçek değerleri (preds ile aynı uzunlukta)
    train_time_s → eğitim süresi saniye cinsinden
    """

    metrics:      dict[str, float]
    preds:        dict[str, np.ndarray]
    y_true:       np.ndarray
    train_time_s: float


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def normalize_stacked_preds(
    preds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Stacked model çıktısını (meta_q01/q05/q09) → (q01/q05/q09) formatına çevir.
    Zaten doğru formattaysa olduğu gibi döndür.
    """
    if "q01" in preds:
        return preds
    return {
        "q01": preds["meta_q01"],
        "q05": preds["meta_q05"],
        "q09": preds["meta_q09"],
    }


def _per_obs_crps(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
) -> np.ndarray:
    """Gözlem başına CRPS (üç quantile pinball ortalaması)."""
    y = np.asarray(y_true, dtype=np.float64)

    def _pb(pred: np.ndarray, q: float) -> np.ndarray:
        r = y - pred
        return np.where(r >= 0, q * r, (q - 1) * r)

    return (_pb(preds["q01"], 0.1) + _pb(preds["q05"], 0.5) + _pb(preds["q09"], 0.9)) / 3.0


def _save_fig(fig: plt.Figure, save_dir: Path, stem: str) -> list[Path]:
    """PNG ve PDF olarak kaydet; Path listesi döndür."""
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        p = save_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
        log.info("Kaydedildi: %s", p)
    plt.close(fig)
    return paths


# ── build_master_table ─────────────────────────────────────────────────────────

def build_master_table(
    results: dict[str, ModelResult],
) -> pd.DataFrame:
    """
    6 model × 9 sütun master tablosu.

    Sütunlar: MAE, RMSE, Pinball_q01, Pinball_q05, Pinball_q09, CRPS, Coverage, Train_s
    Satır sırası MODEL_ORDER'a göre; eksik modeller atlanır.
    """
    rows: list[dict] = []
    for name in MODEL_ORDER:
        if name not in results:
            continue
        r = results[name]
        m = r["metrics"]
        rows.append({
            "Model":       MODEL_LABELS.get(name, name),
            "MAE":         m["mae"],
            "RMSE":        m["rmse"],
            "Pinball_q01": m["pinball_q01"],
            "Pinball_q05": m["pinball_q05"],
            "Pinball_q09": m["pinball_q09"],
            "CRPS":        m["crps"],
            "Coverage":    m["coverage"],
            "Train_s":     r["train_time_s"],
        })
    return pd.DataFrame(rows).set_index("Model")


# ── plot_master_table ──────────────────────────────────────────────────────────

def plot_master_table(
    df: pd.DataFrame,
    save_dir: str | Path = "figures",
) -> list[Path]:
    """
    Master tabloyu matplotlib tablo figürü olarak PNG + PDF kaydet.
    Her sütunda en iyi değer (Coverage → en büyük, diğerleri → en küçük) yeşil vurgulanır.
    """
    save_dir = Path(save_dir)
    n_rows, n_cols = df.shape

    cell_text = [
        [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        for row in df.values.tolist()
    ]

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.5), max(3, n_rows * 0.8 + 1.5)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=list(df.index),
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for j, col in enumerate(df.columns):
        col_vals = df[col].values.astype(float)
        best_idx = int(np.argmax(col_vals)) if col == "Coverage" else int(np.argmin(col_vals))
        cell = tbl[best_idx + 1, j]
        cell.set_facecolor("#c8e6c9")
        cell.set_text_props(fontweight="bold")

    ax.set_title(
        "Master Karşılaştırma Tablosu — 6 Model × 8 Metrik",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()
    return _save_fig(fig, save_dir, "master_table")


# ── plot_probability_bands ─────────────────────────────────────────────────────

def plot_probability_bands(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    model_name: str = "stacked_flags",
    save_dir: str | Path = "figures",
    n_points: int = BAND_N_POINTS,
) -> list[Path]:
    """
    Son n_points gözlem için gerçek değer vs medyan + %10-%90 bant grafiği.
    Her iki serinin de aynı uzunlukta bitiyor olduğu varsayılır (y[-n:]).
    """
    save_dir = Path(save_dir)
    n   = min(n_points, len(y_true))
    y   = np.asarray(y_true[-n:],       dtype=np.float64)
    q01 = np.asarray(preds["q01"][-n:], dtype=np.float64)
    q05 = np.asarray(preds["q05"][-n:], dtype=np.float64)
    q09 = np.asarray(preds["q09"][-n:], dtype=np.float64)
    xs  = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(xs, q01, q09, alpha=0.25, color="steelblue", label="10%–90% Bant")
    ax.plot(xs, q05, color="steelblue",  linewidth=1.2,          label="Medyan (q=0.5)")
    ax.plot(xs, y,   color="black",      linewidth=0.8,
            linestyle="--", alpha=0.85,                           label="Gerçek")
    ax.set_xlabel("Zaman adımı (test seti sonu)")
    ax.set_ylabel("PV Güç Çıktısı")
    ax.set_title(
        f"Olasılıksal Tahmin Bantları — {MODEL_LABELS.get(model_name, model_name)}"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    return _save_fig(fig, save_dir, f"probability_bands_{model_name}")


# ── apply_holm_bonferroni ──────────────────────────────────────────────────────

def apply_holm_bonferroni(
    p_values: list[float],
) -> list[float]:
    """
    Holm (1979) adım aşağı Bonferroni düzeltmesi.

    Giriş listeyle aynı sırada düzeltilmiş p-değerleri döndürür.
    Monotonluk kısıtı uygulanır (sonraki sıra öncekinden küçük olamaz).
    """
    m = len(p_values)
    if m == 0:
        return []
    order   = np.argsort(p_values)
    sorted_p = np.array(p_values, dtype=float)[order]
    adjusted = np.minimum(1.0, sorted_p * np.arange(m, 0, -1, dtype=float))
    adjusted = np.maximum.accumulate(adjusted)  # monotonluk kısıtı
    result: list[float] = [0.0] * m
    for rank, orig_idx in enumerate(order):
        result[int(orig_idx)] = float(adjusted[rank])
    return result


# ── dm_pairwise ────────────────────────────────────────────────────────────────

def dm_pairwise(
    results: dict[str, ModelResult],
) -> pd.DataFrame:
    """
    Tüm model çiftleri için Diebold-Mariano testi + Holm-Bonferroni düzeltmesi.

    Uzunlukları farklı modeller (LSTM/TFT seq trim vs stacked tam) son n gözlemle
    hizalanır (her iki serinin de aynı son noktada bittiği varsayılır).

    Returns:
        pd.DataFrame — sütunlar: model_i, model_j, dm_stat, mean_diff, p_raw, p_adj, significant
    """
    model_names = [n for n in MODEL_ORDER if n in results]
    rows: list[dict] = []
    raw_p: list[float] = []

    for m_i, m_j in combinations(model_names, 2):
        r_i, r_j = results[m_i], results[m_j]
        n = min(len(r_i["y_true"]), len(r_j["y_true"]))
        loss_i = _per_obs_crps(
            r_i["y_true"][-n:],
            {k: v[-n:] for k, v in r_i["preds"].items()},
        )
        loss_j = _per_obs_crps(
            r_j["y_true"][-n:],
            {k: v[-n:] for k, v in r_j["preds"].items()},
        )
        dm = diebold_mariano_test(loss_i, loss_j)
        rows.append({
            "model_i":   MODEL_LABELS.get(m_i, m_i),
            "model_j":   MODEL_LABELS.get(m_j, m_j),
            "dm_stat":   dm["dm_stat"],
            "mean_diff": dm["mean_diff"],
            "p_raw":     dm["p_value"],
        })
        raw_p.append(dm["p_value"])

    if not rows:
        return pd.DataFrame(
            columns=["model_i", "model_j", "dm_stat", "mean_diff", "p_raw", "p_adj", "significant"]
        )

    p_adj = apply_holm_bonferroni(raw_p)
    for i, row in enumerate(rows):
        row["p_adj"]       = p_adj[i]
        row["significant"] = p_adj[i] < ALPHA

    return pd.DataFrame(rows)


# ── plot_dm_heatmap ────────────────────────────────────────────────────────────

def plot_dm_heatmap(
    dm_df: pd.DataFrame,
    save_dir: str | Path = "figures",
) -> list[Path]:
    """
    DM pairwise p_adj değerlerini kare ısı haritası olarak PNG + PDF kaydet.
    Anlamlı hücreler (*) işaretiyle gösterilir (p_adj < 0.05).
    """
    save_dir = Path(save_dir)

    all_labels_in_df: set[str] = set(dm_df["model_i"].tolist() + dm_df["model_j"].tolist())
    labels = [
        MODEL_LABELS.get(n, n)
        for n in MODEL_ORDER
        if MODEL_LABELS.get(n, n) in all_labels_in_df
    ]
    n = len(labels)
    mat = np.full((n, n), np.nan)
    label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}

    for _, row in dm_df.iterrows():
        i = label_to_idx.get(row["model_i"])
        j = label_to_idx.get(row["model_j"])
        if i is not None and j is not None:
            mat[i, j] = row["p_adj"]
            mat[j, i] = row["p_adj"]

    fig, ax = plt.subplots(figsize=(8, 6))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap="RdYlGn_r", aspect="auto")
    fig.colorbar(im, ax=ax, label="p_adj (Holm-Bonferroni)")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("DM Pairwise Testi — Düzeltilmiş p-Değerleri (α=0.05)")

    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(mat[i, j]):
                continue
            v    = mat[i, j]
            star = "*" if v < ALPHA else ""
            ax.text(
                j, i, f"{v:.3f}{star}",
                ha="center", va="center", fontsize=7,
                color="white" if v < 0.05 else "black",
            )

    plt.tight_layout()
    return _save_fig(fig, save_dir, "dm_pairwise_heatmap")


# ── plot_edge_ai_scatter ───────────────────────────────────────────────────────

def plot_edge_ai_scatter(
    df: pd.DataFrame,
    save_dir: str | Path = "figures",
) -> list[Path]:
    """
    Edge AI argümanı: eğitim süresi (saniye, log ölçek) vs CRPS scatter plot.

    Args:
        df: build_master_table() çıktısı — sütun "Train_s" ve "CRPS" içermeli.
    """
    save_dir = Path(save_dir)
    x     = df["Train_s"].astype(float).values
    y     = df["CRPS"].astype(float).values
    names = df.index.tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=80, zorder=3, color="steelblue")
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Eğitim Süresi (saniye, log ölçek)")
    ax.set_ylabel("CRPS (düşük = iyi)")
    ax.set_xscale("log")
    ax.set_title("Edge AI: Hesaplama Süresi vs CRPS")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    return _save_fig(fig, save_dir, "edge_ai_scatter")


# ── run_comparison ─────────────────────────────────────────────────────────────

def run_comparison(
    results: dict[str, ModelResult],
    save_dir: str | Path = "figures",
) -> dict[str, Any]:
    """
    STAGE-10 tam karşılaştırma pipeline'ı.

    Adımlar:
        1. Master tablo → PNG + PDF
        2. stacked_flags ve stacked_noflags için olasılıksal bant görseli
        3. DM pairwise testi + Holm-Bonferroni → PNG + PDF
        4. Edge AI scatter → PNG + PDF

    Returns:
        {
            "master_table": pd.DataFrame,
            "dm_results":   pd.DataFrame,
            "saved_paths":  list[Path],
        }
    """
    save_dir     = Path(save_dir)
    saved_paths: list[Path] = []

    master_df = build_master_table(results)
    log.info("Master tablo:\n%s", master_df.to_string())
    saved_paths.extend(plot_master_table(master_df, save_dir))

    for model_name in ["stacked_flags", "stacked_noflags"]:
        if model_name not in results:
            continue
        r = results[model_name]
        saved_paths.extend(
            plot_probability_bands(r["y_true"], r["preds"], model_name, save_dir)
        )

    dm_df = dm_pairwise(results)
    log.info("DM pairwise:\n%s", dm_df.to_string())
    if not dm_df.empty:
        saved_paths.extend(plot_dm_heatmap(dm_df, save_dir))

    saved_paths.extend(plot_edge_ai_scatter(master_df, save_dir))

    return {
        "master_table": master_df,
        "dm_results":   dm_df,
        "saved_paths":  saved_paths,
    }
