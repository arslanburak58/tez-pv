"""
STAGE-11 — PV Olasılıksal Tahmin Demo
v7 model (meta_models_robust_v7.joblib) | CQR k=2.0

Sekmeler:
  1. DKASC Alice Springs — test seti + interaktif sensör arızası simülasyonu
  2. PVOD Station02       — hiç görülmemiş veri, genelleme testi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Python 3.13 + pandas 2.2.x uyumluluk yaması: StringDtype.__init__ imzası değişti.
# joblib ile kaydedilmiş eski DataFrame'leri yüklemek için gerekli.
import pandas.core.arrays.string_ as _sd
_sd_orig = _sd.StringDtype.__init__
def _sd_patched(self, *args, **kwargs):
    kwargs.pop("na_value", None)
    _sd_orig(self, *args[:1], **kwargs)
_sd.StringDtype.__init__ = _sd_patched

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

from features.physical import build_physical_features
from models.meta_learner import enrich_x_meta, predict_intervals
from models.base_learners import _col_name

# ── Sabitler ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

FEATURE_COLS: list[str] = [
    "T_amb", "RH", "G",
    "cos_zenith", "hour_angle", "air_mass", "k_t", "T_cell",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_G_missing", "is_Tamb_missing", "is_RH_missing",
]
FLAG_COLS: list[str] = ["is_G_missing", "is_Tamb_missing", "is_RH_missing"]
ALGOS:     list[str] = ["lgbm", "catboost", "xgboost"]
QUANTILES: list[float] = [0.1, 0.5, 0.9]
CQR_K:     float = 2.0
CQR_K_V7:  float = 1.7  # v7 için kalibre edildi (actual>0 filtreli, ~%81 coverage)

DKASC_LOC: dict = {
    "latitude": -23.762,
    "longitude": 133.875,
    "altitude": 546,
    "tz": "Australia/Darwin",
}
S02_LOC: dict = {
    "latitude": 38.05728,
    "longitude": 114.19887,
    "altitude": 50,
    "tz": "Asia/Shanghai",
}
S02_CAPACITY: float = 17_000.0  # kW (power sütunu kW'a çevrildi)

SENSOR_TO_FLAG: dict[str, str] = {
    "G": "is_G_missing",
    "T_amb": "is_Tamb_missing",
    "RH": "is_RH_missing",
}

# ── Model ve veri yükleme ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Modeller yükleniyor…")
def load_models() -> tuple[dict, dict]:
    base = joblib.load(ROOT / "data/processed/base_models.joblib")
    meta = joblib.load(ROOT / "data/processed/meta_models_robust_v7.joblib")
    return base, meta


@st.cache_data(show_spinner="Test seti yükleniyor…")
def load_dkasc_test() -> tuple[pd.DataFrame, pd.Series]:
    d = joblib.load(ROOT / "data/processed/dataset.joblib")
    return d["X_test"], d["y_test"]


@st.cache_data(show_spinner="Station02 verisi yükleniyor…")
def load_station02() -> pd.DataFrame:
    df = pd.read_csv(
        ROOT / "data/raw/pvod/datasets/station02.csv",
        parse_dates=["date_time"],
        index_col="date_time",
    )
    df.index = df.index.tz_localize("Asia/Shanghai")
    df = df.rename(columns={
        "lmd_totalirrad": "G",
        "lmd_temperature": "T_amb",
        "nwp_humidity": "RH",
    })
    df["G"] = df["G"].clip(lower=0.0)
    df["power"] = df["power"] * 1000.0  # PVOD power MW → kW
    return df


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def make_flags(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "is_G_missing":    df["G"].isna().astype(int),
            "is_Tamb_missing": df["T_amb"].isna().astype(int),
            "is_RH_missing":   df["RH"].isna().astype(int),
        },
        index=df.index,
    )


def ffill_impute(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().bfill()


def daylight_mask(X: pd.DataFrame) -> pd.Series:
    """cos_zenith > 0.087 → zenit < 85°"""
    return X["cos_zenith"] > 0.087


def apply_cqr(preds: pd.DataFrame, k: float = CQR_K) -> pd.DataFrame:
    preds = preds.copy()
    mid = preds["q_0.5"]
    preds["q_0.1"] = mid - k * (mid - preds["q_0.1"])
    preds["q_0.9"] = mid + k * (preds["q_0.9"] - mid)
    return preds.clip(lower=0.0)


def simulate_failure(
    X: pd.DataFrame,
    rate_G: float,
    rate_Tamb: float,
    rate_RH: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test setinde sensör arızası simüle et.
    Bozulan sensör → 0, türev öznitelikler güncellenir, flag → 1.
    Dönüş: (X_bozuk, flags)
    """
    rng = np.random.default_rng(seed)
    X_out  = X.copy()
    flags  = X[FLAG_COLS].copy().astype(int)

    rates = {"G": rate_G, "T_amb": rate_Tamb, "RH": rate_RH}
    for sensor, rate in rates.items():
        if rate <= 0:
            continue
        mask = rng.random(len(X)) < rate
        X_out.loc[mask, sensor] = 0.0
        flags.loc[mask, SENSOR_TO_FLAG[sensor]] = 1

        if sensor == "G":
            # k_t ve T_cell G'ye bağlı — sıfıra çek
            X_out.loc[mask, "k_t"]    = 0.0
            X_out.loc[mask, "T_cell"] = X_out.loc[mask, "T_amb"]

    X_out[FLAG_COLS] = flags[FLAG_COLS]
    return X_out, flags


def _apply_cqr_v7(preds: pd.DataFrame) -> pd.DataFrame:
    preds = preds.copy()
    mid = preds["q_0.5"]
    preds["q_0.1"] = (mid - CQR_K_V7 * (mid - preds["q_0.1"])).clip(upper=mid)
    preds["q_0.9"] = (mid + CQR_K_V7 * (preds["q_0.9"] - mid)).clip(lower=mid)
    return preds


def run_inference(
    X: pd.DataFrame,
    flags: pd.DataFrame,
    base_models: dict,
    meta_models: dict,
) -> pd.DataFrame:
    """
    X: FEATURE_COLS sütunlarını içeren DataFrame.
    Dönüş: q_0.1 / q_0.5 / q_0.9 sütunlu DataFrame (CQR uygulanmış).
    """
    # 1. Taban tahminler
    meta_cols: dict[str, np.ndarray] = {}
    for algo in ALGOS:
        for q in QUANTILES:
            col   = _col_name(algo, q)
            model = base_models[col]
            meta_cols[col] = model.predict(X[FEATURE_COLS])
    x_meta = pd.DataFrame(meta_cols, index=X.index)

    # 2. Flags ekle
    x_meta_enriched = enrich_x_meta(x_meta, flags)

    # 3. Meta tahmin
    raw = predict_intervals(meta_models, x_meta_enriched)
    preds = pd.DataFrame({
        "q_0.1": raw["meta_q01"],
        "q_0.5": raw["meta_q05"],
        "q_0.9": raw["meta_q09"],
    }, index=X.index)
    return _apply_cqr_v7(preds)


def compute_metrics(actual: pd.Series, preds: pd.DataFrame) -> dict[str, float]:
    y   = actual.to_numpy()
    q01 = preds["q_0.1"].to_numpy()
    q05 = preds["q_0.5"].to_numpy()
    q09 = preds["q_0.9"].to_numpy()

    mae     = float(np.mean(np.abs(y - q05)))
    rmse    = float(np.sqrt(np.mean((y - q05) ** 2)))
    pb01    = float(np.mean(np.where(y >= q01, 0.1 * (y - q01), 0.9 * (q01 - y))))
    pb05    = float(np.mean(np.where(y >= q05, 0.5 * (y - q05), 0.5 * (q05 - y))))
    pb09    = float(np.mean(np.where(y >= q09, 0.9 * (y - q09), 0.1 * (q09 - y))))
    pinball = (pb01 + pb05 + pb09) / 3.0

    # Coverage sadece pozitif üretim saatlerinde (invertör bekleme satırlarını dışla)
    pos_mask = actual.values > 0
    if pos_mask.sum() > 0:
        coverage = float(np.mean(
            (actual.values[pos_mask] >= q01[pos_mask]) &
            (actual.values[pos_mask] <= q09[pos_mask])
        ))
    else:
        coverage = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Pinball": pinball,
        "Coverage": coverage,
    }


def plot_forecast(
    actual: pd.Series,
    preds: pd.DataFrame,
    title: str,
    ylabel: str = "Güç (kW)",
    n_days: int = 3,
) -> plt.Figure:
    idx  = actual.index
    days = sorted({d.date() for d in idx})[:n_days]
    mask = np.array([d.date() in days for d in idx])

    x   = idx[mask]
    y   = actual.to_numpy()[mask]
    q01 = preds["q_0.1"].to_numpy()[mask]
    q05 = preds["q_0.5"].to_numpy()[mask]
    q09 = preds["q_0.9"].to_numpy()[mask]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(x, q01, q09, alpha=0.22, color="#1E88E5", label="q10–q90 bant")
    ax.plot(x, q05, color="#1E88E5", lw=1.6, label="q50 (medyan)")
    ax.plot(x, y,   color="#E53935", lw=1.3, alpha=0.85, label="Gerçek")
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ── Ana uygulama ──────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="PV Olasılıksal Tahmin",
        page_icon="☀️",
        layout="wide",
    )
    st.title("☀️ Fotovoltaik Sistemlerde Olasılıksal Güç Tahmini")
    st.caption(
        "**v7 model** — QuantileLinearBounded meta-öğrenici · "
        "Corruption-aware eğitim · CQR k=2.0 · XGBoost + LightGBM + CatBoost"
    )

    base_models, meta_models = load_models()

    tab1, tab2 = st.tabs([
        "🏠 DKASC Alice Springs (Test Seti)",
        "🌏 Genelleme: PVOD Station02 (Görülmemiş Veri)",
    ])

    # ── Tab 1: DKASC ──────────────────────────────────────────────────────────
    with tab1:
        st.subheader("DKASC Alice Springs — Test Seti (2021–2023)")
        st.caption(
            "Model bu lokasyonun 2010–2019 verisinde eğitildi. "
            "Grafikte 2021–2023 test döneminden seçilen günler gösterilmektedir."
        )

        X_test, y_test = load_dkasc_test()

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            n_days = st.slider("Gösterilen gün sayısı", 1, 14, 4, key="dkasc_days")
        with col2:
            r_G    = st.slider("G eksikliği (%)", 0, 80, 0, key="dkasc_G") / 100
        with col3:
            r_Tamb = st.slider("T_amb eksikliği (%)", 0, 80, 0, key="dkasc_T") / 100
        with col4:
            r_RH   = st.slider("RH eksikliği (%)", 0, 80, 0, key="dkasc_RH") / 100

        failure_active = any([r_G > 0, r_Tamb > 0, r_RH > 0])

        with st.spinner("Tahmin hesaplanıyor…"):
            if failure_active:
                X_sim, flags_sim = simulate_failure(X_test, r_G, r_Tamb, r_RH)
            else:
                X_sim   = X_test
                flags_sim = X_test[FLAG_COLS].astype(int)

            dl      = daylight_mask(X_sim)
            X_dl    = X_sim[dl]
            y_dl    = y_test[dl]
            fl_dl   = flags_sim[dl]

            preds_dkasc = run_inference(X_dl, fl_dl, base_models, meta_models)

        fig1 = plot_forecast(
            y_dl, preds_dkasc,
            title="DKASC Alice Springs — Tahmin Bantları",
            n_days=n_days,
        )
        st.pyplot(fig1)

        m = compute_metrics(y_dl, preds_dkasc)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",      f"{m['MAE']:.2f} kW")
        c2.metric("RMSE",     f"{m['RMSE']:.2f} kW")
        c3.metric("Pinball",  f"{m['Pinball']:.4f}")
        c4.metric("Coverage (üretim saatleri)", f"{m['Coverage']:.1%}",
                  delta=f"{m['Coverage']-0.80:+.1%} (hedef %80)")

        if failure_active:
            parts = []
            if r_G    > 0: parts.append(f"G: %{r_G*100:.0f}")
            if r_Tamb > 0: parts.append(f"T_amb: %{r_Tamb*100:.0f}")
            if r_RH   > 0: parts.append(f"RH: %{r_RH*100:.0f}")
            st.warning(f"🔧 Sensör arızası aktif → {', '.join(parts)}")

    # ── Tab 2: PVOD Station02 ─────────────────────────────────────────────────
    with tab2:
        st.subheader("PVOD Station02 — Görülmemiş Veri (Genelleme Testi)")

        col_info, col_spec = st.columns([2, 1])
        with col_info:
            st.markdown(
                "**⚠️ Model bu veriyi hiç görmedi.**  \n"
                "Eğitim: Avustralya · Alice Springs · 23.7°S  \n"
                "Tahmin: Çin · Hebei · 38.1°N · Mono-Si · 17 MW"
            )
        with col_spec:
            st.markdown(
                "| Özellik | Değer |\n"
                "|---------|-------|\n"
                "| Teknoloji | Mono-Si |\n"
                "| Kapasite | 17 MW |\n"
                "| Lokasyon | Hebei, Çin |\n"
                "| Dönem | 2018–2019 |\n"
                "| RH kaynağı | NWP (proxy) |"
            )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            n_days_s02 = st.slider("Gösterilen gün sayısı", 1, 14, 4, key="s02_days")
        with col_b:
            normalize = st.checkbox("Kapasiteye normalize et (0–1)", value=True)

        df_s02 = load_station02()

        with st.spinner("Fiziksel öznitelikler + tahmin hesaplanıyor…"):
            flags_s02 = make_flags(df_s02)
            df_imp    = ffill_impute(df_s02[["G", "T_amb", "RH"]])
            phys      = build_physical_features(df_imp, S02_LOC)

            X_s02 = pd.concat(
                [phys, flags_s02],
                axis=1,
            )[FEATURE_COLS]

            dl_s02   = daylight_mask(X_s02)
            X_dl_s02 = X_s02[dl_s02]
            y_s02    = df_s02["power"][dl_s02]
            fl_s02   = flags_s02[dl_s02]

            preds_s02 = run_inference(X_dl_s02, fl_s02, base_models, meta_models)

        # Normalizasyon
        if normalize:
            cap        = S02_CAPACITY
            y_plot     = y_s02 / cap
            preds_plot = preds_s02 / cap
            ylabel     = "Normalize Güç (0–1)"
        else:
            y_plot     = y_s02
            preds_plot = preds_s02
            ylabel     = "Güç (kW)"

        fig2 = plot_forecast(
            y_plot, preds_plot,
            title="PVOD Station02 (Mono-Si, Hebei) — Görülmemiş Veri Tahmini",
            ylabel=ylabel,
            n_days=n_days_s02,
        )
        st.pyplot(fig2)

        m2 = compute_metrics(y_plot, preds_plot)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",      f"{m2['MAE']:.4f}" + (" (norm)" if normalize else " kW"))
        c2.metric("RMSE",     f"{m2['RMSE']:.4f}" + (" (norm)" if normalize else " kW"))
        c3.metric("Pinball",  f"{m2['Pinball']:.4f}")
        c4.metric("Coverage (üretim saatleri)", f"{m2['Coverage']:.1%}",
                  delta=f"{m2['Coverage']-0.80:+.1%} (hedef %80)")

        with st.expander("📌 Yorum / Metodoloji notu"):
            st.markdown(
                "- Model DKASC Alice Springs skalasında (kW) eğitildi; "
                "Station02 tahminleri ölçek uyumsuzluğu içerebilir → normalize görünüm önerilir.\n"
                "- RH değeri Station02'de ölçülmemiş; NWP proxy olarak kullanılmaktadır.\n"
                "- Coverage hedefin altında kalıyorsa bu beklenen bir bulgudur: "
                "CQR k=2.0 DKASC test setine göre kalibre edilmiştir, "
                "farklı iklim/teknoloji altında yeniden kalibrasyon gerekir.\n"
                "- Şekil paterni (gündüz rampa, öğle tepe, akşam düşüş) "
                "doğru yakalanıyorsa model genelleşebilir demektir."
            )


if __name__ == "__main__":
    main()
