"""
Tam eğitim pipeline — DKASC verisiyle uçtan uca.

Adımlar:
  1. DKASC CSV'leri yükle (yalnızca gerekli sütunlar)
  2. make_dataset() → train/val/test böl, impute et, öznitelik üret
  3. Processed veriyi data/processed/ altına kaydet (joblib)
  4. best_params.json parametrelerle 9 taban modeli eğit (OOF + final)
  5. enrich_x_meta() + train_all_meta_learners() → 3 Ridge meta-model
  6. train_all_baselines() → k-NN, SVM, LSTM, TFT
  7. Her adımın süresini logla

Çalıştırma:
    python scripts/run_training.py
    python scripts/run_training.py --force   # mevcut dosyaları yenile
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Proje kökünü path'e ekle (scripts/ altından çalıştırılabilmesi için)
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from features.physical import DKASC_COL_MAP, DKASC_LOCATION
from models.base_learners import (
    ALGOS, META_COLS, QUANTILES, _col_name,
    build_x_meta, pinball_loss, train_all_base_learners,
)
from models.baselines import train_all_baselines
from models.meta_learner import (
    FLAG_COLS, enrich_x_meta, predict_intervals,
    train_all_meta_learners,
)
from scripts.make_dataset import make_dataset

# ── Dizinler ───────────────────────────────────────────────────────────────────

RAW_DIR:          Path = Path("data/raw/dkasc")
PROCESSED_DIR:    Path = Path("data/processed")
CHECKPOINT_DIR:   Path = Path("models/checkpoints")
LOG_DIR:          Path = Path("logs")
BEST_PARAMS_PATH: Path = CHECKPOINT_DIR / "best_params.json"
FREQ_MINUTES:     int  = 5

# Ham CSV'den okunacak sütunlar (timestamp + 5 sensör/hedef)
RAW_SENSOR_COLS: list[str] = list(DKASC_COL_MAP.keys())

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "run_training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ── Yardımcılar ────────────────────────────────────────────────────────────────

def _elapsed(label: str, t0: float) -> None:
    log.info("  ✓ %s tamamlandı | %.1f s", label, time.time() - t0)


def _should_skip(path: Path, force: bool) -> bool:
    """Dosya varsa ve force=False ise True döndür (adım atlanacak)."""
    if path.exists() and not force:
        log.info("  ↩ Atlanıyor (mevcut): %s  — yenilemek için --force", path.name)
        return True
    return False


def _load_best_params() -> dict[str, dict]:
    """best_params.json'dan algo bazlı parametre dicts döndür."""
    if not BEST_PARAMS_PATH.exists():
        log.warning("  best_params.json bulunamadı — DEFAULT_PARAMS kullanılacak")
        return {}
    with open(BEST_PARAMS_PATH) as f:
        bp = json.load(f)
    return {k: v for k, v in bp.items() if not k.startswith("_") and k != "ridge_alpha"}


def _load_ridge_alpha() -> float:
    if not BEST_PARAMS_PATH.exists():
        return 1.0
    with open(BEST_PARAMS_PATH) as f:
        bp = json.load(f)
    return float(bp.get("ridge_alpha", 1.0))


# ── ADIM 1: DKASC CSV yükleme ─────────────────────────────────────────────────

def step1_load_dkasc(force: bool = False) -> pd.DataFrame:
    out_path = PROCESSED_DIR / "dkasc_raw.joblib"
    if _should_skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    csv_files = sorted(RAW_DIR.glob("dkasc_*.csv"))
    log.info("=== ADIM 1: DKASC CSV'leri yükleniyor (%d dosya) ===", len(csv_files))

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        t_csv = time.time()

        # Header'dan hangi sütunlar mevcut diye kontrol et
        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        cols_to_load = ["timestamp"] + [c for c in RAW_SENSOR_COLS if c in header_cols]

        df_year = (
            pd.read_csv(csv_path, usecols=cols_to_load, parse_dates=["timestamp"])
            .set_index("timestamp")
            .rename(columns={k: v for k, v in DKASC_COL_MAP.items() if k in cols_to_load})
        )
        frames.append(df_year)
        log.info("  %s: %d satır | %.1f s", csv_path.name, len(df_year), time.time() - t_csv)

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    log.info("  Toplam: %d satır | sütunlar: %s", len(df), list(df.columns))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(df, out_path)
    _elapsed("DKASC yükleme", t0)
    return df


# ── ADIM 2: make_dataset ───────────────────────────────────────────────────────

def step2_make_dataset(df: pd.DataFrame, force: bool = False) -> dict:
    out_path = PROCESSED_DIR / "dataset.joblib"
    if _should_skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    log.info("=== ADIM 2: make_dataset — 70/15/15 bölme + imputation ===")

    dataset = make_dataset(
        df=df,
        location=DKASC_LOCATION,
        target_col="target",
        freq_minutes=FREQ_MINUTES,
        imputer_path=str(PROCESSED_DIR / "imputer.joblib"),
    )

    si = dataset["split_info"]
    log.info(
        "  Train: %d | Val: %d | Test: %d | Öznitelik: %d",
        si["n_train"], si["n_val"], si["n_test"], len(dataset["feature_cols"]),
    )

    joblib.dump(dataset, out_path)
    _elapsed("make_dataset", t0)
    return dataset


# ── ADIM 3: Taban modeller + OOF + final ──────────────────────────────────────

def step3_base_learners(
    dataset: dict, force: bool = False
) -> tuple[dict, pd.DataFrame, dict]:
    base_path   = PROCESSED_DIR / "base_models.joblib"
    x_meta_path = PROCESSED_DIR / "x_meta.joblib"

    if base_path.exists() and x_meta_path.exists() and not force:
        log.info("  ↩ Atlanıyor (mevcut): base_models.joblib + x_meta.joblib")
        return joblib.load(base_path), *joblib.load(x_meta_path)

    t0 = time.time()
    log.info("=== ADIM 3: 9 taban model — OOF (5 fold) + final eğitim ===")

    X_train    = dataset["X_train"]
    y_train    = dataset["y_train"]
    best_params = _load_best_params()

    # 3a. OOF → X_meta
    log.info("  OOF tahminleri: 9 model × 5 fold = 45 fit...")
    t_oof = time.time()
    X_meta, oof_scores = build_x_meta(X_train, y_train, params_override=best_params)
    log.info("  OOF bitti | %.1f s | X_meta: %s", time.time() - t_oof, X_meta.shape)
    for col, sc in oof_scores.items():
        log.info("    %s: %.4f", col, sc)

    # 3b. Final modeller (tam train seti)
    log.info("  Final taban modeller (tam train seti)...")
    t_final = time.time()
    base_models = train_all_base_learners(
        X_train, y_train,
        checkpoint_dir=str(CHECKPOINT_DIR),
        params_override=best_params,
    )
    log.info("  Final modeller bitti | %.1f s", time.time() - t_final)

    joblib.dump(base_models, base_path)
    joblib.dump((X_meta, oof_scores), x_meta_path)
    _elapsed("Taban modeller", t0)
    return base_models, X_meta, oof_scores


# ── ADIM 4: Meta-öğreniciler ───────────────────────────────────────────────────

def step4_meta_learners(
    dataset: dict, X_meta: pd.DataFrame, force: bool = False
) -> dict:
    out_path = PROCESSED_DIR / "meta_models.joblib"
    if _should_skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    log.info("=== ADIM 4: Ridge meta-öğreniciler (q=0.1/0.5/0.9) ===")

    y_train     = dataset["y_train"]
    flags_train = dataset["X_train"][FLAG_COLS]
    ridge_alpha = _load_ridge_alpha()

    y_meta    = y_train.reindex(X_meta.index)
    X_meta_13 = enrich_x_meta(X_meta, flags_train)
    log.info("  X_meta_13: %s | ridge_alpha=%.6f", X_meta_13.shape, ridge_alpha)

    meta_models = train_all_meta_learners(
        X_meta_13, y_meta,
        alpha=ridge_alpha,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    # Val seti kısa değerlendirme
    log.info("  Val seti değerlendirmesi...")
    base_models = joblib.load(PROCESSED_DIR / "base_models.joblib")
    X_val       = dataset["X_val"]
    y_val       = dataset["y_val"]
    flags_val   = X_val[FLAG_COLS]

    val_base: dict[str, np.ndarray] = {
        _col_name(a, q): np.asarray(base_models[_col_name(a, q)].predict(X_val))
        for a in ALGOS for q in QUANTILES
    }
    X_val_meta_13 = enrich_x_meta(
        pd.DataFrame(val_base, columns=META_COLS, index=X_val.index),
        flags_val,
    )
    val_preds = predict_intervals(meta_models, X_val_meta_13)

    for key, q in [("meta_q01", 0.1), ("meta_q05", 0.5), ("meta_q09", 0.9)]:
        sc = pinball_loss(np.asarray(y_val), val_preds[key], q)
        log.info("  Val pinball %s: %.4f", key, sc)

    joblib.dump(meta_models, out_path)
    _elapsed("Meta-öğreniciler", t0)
    return meta_models


# ── ADIM 5: Baseline modeller ─────────────────────────────────────────────────

def step5_baselines(dataset: dict, force: bool = False) -> dict:
    out_path = PROCESSED_DIR / "baseline_results.joblib"
    if _should_skip(out_path, force):
        return joblib.load(out_path)

    t0 = time.time()
    log.info("=== ADIM 5: Baseline modeller (k-NN, SVM, LSTM, TFT) ===")

    X_tr = dataset["X_train"].values.astype(np.float32)
    y_tr = dataset["y_train"].values.astype(np.float32)
    X_te = dataset["X_test"].values.astype(np.float32)
    y_te = dataset["y_test"].values.astype(np.float32)

    results = train_all_baselines(
        X_tr, y_tr, X_te, y_te,
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    for name, res in results.items():
        m = res["metrics"]
        log.info(
            "  %s | crps=%.4f mae=%.4f coverage=%.3f | süre=%.1fs",
            name, m["crps"], m["mae"], m["coverage"], res["train_time"],
        )

    joblib.dump(results, out_path)
    _elapsed("Baseline modeller", t0)
    return results


# ── Ana akış ───────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    log.info("=" * 60)
    log.info("STAGE-10 EĞİTİM PIPELINE BAŞLADI")
    log.info("force=%s", force)
    log.info("=" * 60)

    # Adım 1-2: Veri
    df      = step1_load_dkasc(force)
    dataset = step2_make_dataset(df, force)
    del df  # bellek boşalt

    # Adım 3: Taban modeller
    _, X_meta, _ = step3_base_learners(dataset, force)

    # Adım 4: Meta-öğreniciler
    step4_meta_learners(dataset, X_meta, force)
    del X_meta

    # Adım 5: Baseline'lar
    step5_baselines(dataset, force)

    elapsed = time.time() - t_total
    log.info("=" * 60)
    log.info("PIPELINE TAMAMLANDI | Toplam: %.1f dk", elapsed / 60)
    log.info("Checkpoint'ler: %s", CHECKPOINT_DIR)
    log.info("Processed:      %s", PROCESSED_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DKASC eğitim pipeline")
    parser.add_argument("--force", action="store_true", help="Mevcut dosyaları yenile")
    args = parser.parse_args()
    main(force=args.force)
