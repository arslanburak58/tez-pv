"""
STAGE-7: Gerçek DKASC verisiyle Optuna araması.

Kullanım:
    python scripts/run_optuna.py [--trials 50] [--years 2015 2016 2017]

Çıktılar:
    models/checkpoints/best_params.json
    figures/optuna_parallel_coordinate.html
    figures/optuna_param_importances.html
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from features.physical import DKASC_COL_MAP, DKASC_LOCATION
from models.meta_learner import FLAG_COLS
from optimization.optuna_search import (
    load_best_params,
    plot_study,
    run_study,
    save_best_params,
    top_trials_summary,
)
from scripts.make_dataset import make_dataset, rename_columns

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Sabitler ───────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw/dkasc")
CHECKPOINT_DIR = Path("models/checkpoints")
FIGURES_DIR = Path("figures")
BEST_PARAMS_PATH = CHECKPOINT_DIR / "best_params.json"

DEFAULT_YEARS = [2015, 2016]  # wind_speed yalnızca 2015-2016'da tam mevcut
DEFAULT_TRIALS = 50
LOG_INTERVAL = 10  # her kaç trial'da bir ara skor logla


# ── Callback: her 10 trial'da bir ara log ─────────────────────────────────────

class _ProgressCallback:
    def __init__(self, interval: int, start_time: float) -> None:
        self.interval = interval
        self.start_time = start_time

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n = trial.number + 1
        if n % self.interval == 0 or n == 1:
            elapsed = time.time() - self.start_time
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned   = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            log.info(
                "Trial %3d | best=%.4f | tamamlanan=%d | pruned=%d | süre=%.0fs",
                n,
                study.best_value,
                len(completed),
                len(pruned),
                elapsed,
            )


# ── Veri yükleme ───────────────────────────────────────────────────────────────

def load_dkasc(years: list[int]) -> pd.DataFrame:
    """Belirtilen yılların CSV'lerini birleştir, rename_columns uygula."""
    frames = []
    for yr in sorted(years):
        path = RAW_DIR / f"dkasc_{yr}.csv"
        if not path.exists():
            log.warning("Dosya bulunamadı, atlanıyor: %s", path)
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df = rename_columns(df, dataset="dkasc")
        # Sadece ilgili sütunları tut
        keep = [c for c in ["G", "T_amb", "RH", "wind_speed", "target"] if c in df.columns]
        frames.append(df[keep])
        log.info("Yüklendi: %s (%d satır)", path.name, len(df))

    if not frames:
        raise FileNotFoundError(f"Hiçbir DKASC dosyası bulunamadı: {RAW_DIR}")

    combined = pd.concat(frames).sort_index()
    # Timezone ekle (pvlib için UTC gerekli; DKASC timestamp'leri yerel → Darwin UTC+9:30)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("Australia/Darwin").tz_convert("UTC")

    # Target temizliği: NaN düşür, negatif gece değerlerini 0'a klip
    before = len(combined)
    combined = combined.dropna(subset=["target"])
    combined["target"] = combined["target"].clip(lower=0)
    log.info("Birleştirildi: %d satır (%d NaN target düşürüldü) | %s → %s",
             len(combined), before - len(combined),
             combined.index[0].date(), combined.index[-1].date())
    return combined


# ── Ana fonksiyon ──────────────────────────────────────────────────────────────

def main(years: list[int], n_trials: int) -> None:
    t0 = time.time()

    # 1. Veri yükleme
    log.info("=== DKASC verisi yükleniyor: %s ===", years)
    df = load_dkasc(years)

    # 2. make_dataset
    log.info("=== make_dataset çalıştırılıyor ===")
    ds = make_dataset(df, DKASC_LOCATION, target_col="target", freq_minutes=5)
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_val   = ds["X_val"]
    y_val   = ds["y_val"]

    log.info("X_train: %s | X_val: %s | features: %d",
             X_train.shape, X_val.shape, len(ds["feature_cols"]))

    # 3. flags_val — make_dataset X_train/val'a dahil etti
    available_flags = [c for c in FLAG_COLS if c in X_val.columns]
    if len(available_flags) < len(FLAG_COLS):
        log.warning("Bazı flag sütunları eksik: %s", set(FLAG_COLS) - set(available_flags))
    flags_val = X_val[available_flags].reindex(columns=FLAG_COLS, fill_value=0)

    # 4. Optuna araması
    log.info("=== Optuna başlıyor: %d trial ===", n_trials)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    callback = _ProgressCallback(interval=LOG_INTERVAL, start_time=t0)

    study = run_study(
        X_train, y_train, X_val, y_val, flags_val,
        n_trials=n_trials,
        study_name="dkasc_optuna",
        storage=f"sqlite:///{CHECKPOINT_DIR}/optuna.db",
        show_progress_bar=False,
        catch=(Exception,),
    )
    study.trials_dataframe()  # cache yenile

    # Her 10 trial callback Optuna'ya verildi; ek log
    for trial in study.trials:
        callback(study, trial)

    # 5. Sonuçlar
    elapsed = time.time() - t0
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    log.info("=== Arama tamamlandı ===")
    log.info("Toplam süre    : %.1f dakika", elapsed / 60)
    log.info("Tamamlanan     : %d", len(completed))
    log.info("Pruned         : %d", len(pruned))
    log.info("En iyi trial   : #%d", study.best_trial.number)
    log.info("En iyi pinball : %.4f", study.best_value)

    # 6. En iyi 5 trial özeti
    top = top_trials_summary(study, n=5)
    print("\n── En iyi 5 trial ──────────────────────────────")
    print(top[["trial_number", "val_pinball"]].to_string(index=False))

    # 7. Parametreleri kaydet
    save_best_params(study, str(BEST_PARAMS_PATH))
    log.info("best_params.json: %s", BEST_PARAMS_PATH)

    # 8. Grafikler
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved = plot_study(study, save_dir=str(FIGURES_DIR))
    for p in saved:
        log.info("Grafik: %s", p)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DKASC Optuna araması")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--years",  type=int, nargs="+", default=DEFAULT_YEARS)
    args = parser.parse_args()
    main(years=args.years, n_trials=args.trials)
