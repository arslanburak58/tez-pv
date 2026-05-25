"""
STAGE-7: Optuna TPE araması — LightGBM / CatBoost / XGBoost hiperparametreleri + Ridge alpha.

Objective : mean val pinball (9 taban model) / 9 — minimize.
Sampler   : TPESampler(seed=42)
Pruner    : MedianPruner(n_startup_trials=5, n_warmup_steps=1)
             Her algo'dan sonra kümülatif kayıp bildirilir (step=0/1/2).

Ridge alpha objective'e dahil edilmez (Ridge nanosaniyede eğitilir; OOF
olmadan val üzerinde eğitmek anlamsız). Best trial'dan alınan alpha, son
pipeline'da kullanılır.

API:
    suggest_base_params(trial, algo)       → dict
    objective(trial, ...)                  → float (minimize)
    run_study(..., n_trials)               → optuna.Study
    save_best_params(study, path)          → None
    load_best_params(path)                 → dict
    plot_study(study, save_dir)            → list[Path]
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from models.base_learners import ALGOS, QUANTILES, _col_name, pinball_loss, train_base_learner

random.seed(42)
np.random.seed(42)

log = logging.getLogger(__name__)

# ── Arama uzayı sınırları ──────────────────────────────────────────────────────

SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "lgbm": {
        "n_estimators":     (100,  2000, "int"),
        "learning_rate":    (1e-3, 0.3,  "log_float"),
        "max_depth":        (3,    12,   "int"),
        "num_leaves":       (15,   255,  "int"),
        "subsample":        (0.5,  1.0,  "float"),
        "colsample_bytree": (0.5,  1.0,  "float"),
        "reg_alpha":        (1e-4, 10.0, "log_float"),
        "reg_lambda":       (1e-4, 10.0, "log_float"),
    },
    "catboost": {
        "iterations":    (100,  2000, "int"),
        "learning_rate": (1e-3, 0.3,  "log_float"),
        "depth":         (3,    10,   "int"),
        "subsample":     (0.5,  1.0,  "float"),
        "reg_lambda":    (1e-4, 10.0, "log_float"),
    },
    "xgboost": {
        "n_estimators":     (100,  2000, "int"),
        "learning_rate":    (1e-3, 0.3,  "log_float"),
        "max_depth":        (3,    12,   "int"),
        "subsample":        (0.5,  1.0,  "float"),
        "colsample_bytree": (0.5,  1.0,  "float"),
        "reg_alpha":        (1e-4, 10.0, "log_float"),
        "reg_lambda":       (1e-4, 10.0, "log_float"),
    },
}

RIDGE_ALPHA_BOUNDS: tuple[float, float] = (1e-4, 100.0)

N_TRIALS_DEFAULT: int = 50
N_STARTUP_TRIALS: int = 5   # MedianPruner: bu kadar trial bitmeden prune yok
N_WARMUP_STEPS: int = 1     # MedianPruner: ilk adımda prune yok


# ── Parametre önerme ───────────────────────────────────────────────────────────

def suggest_base_params(trial: optuna.Trial, algo: str) -> dict[str, Any]:
    """
    Bir algo için SEARCH_SPACE'den Optuna trial parametreleri öner.

    Parametre adları prefix'li: "{algo}_{param}" → çakışma olmaz.
    Dönen dict DEFAULT_PARAMS formatındadır (prefix'siz), train_base_learner'a geçirilir.
    """
    space = SEARCH_SPACE[algo]
    params: dict[str, Any] = {}
    for param, spec in space.items():
        lo, hi, kind = spec
        name = f"{algo}_{param}"
        if kind == "int":
            params[param] = trial.suggest_int(name, int(lo), int(hi))
        elif kind == "float":
            params[param] = trial.suggest_float(name, lo, hi)
        elif kind == "log_float":
            params[param] = trial.suggest_float(name, lo, hi, log=True)
    return params


# ── Objective fonksiyonu ───────────────────────────────────────────────────────

def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    flags_val: pd.DataFrame,
) -> float:
    """
    Bir Optuna trial için val pinball kaybını hesapla.

    Her algo'dan sonra kümülatif kayıp raporlanır (MedianPruner için).
    Ridge alpha da önerilir; objective'i etkilemez ama best_params.json'a girer.

    Returns:
        float — 9 modelin ortalaması (minimize edilir)
    """
    # Ridge alpha önerilir, trial params'a kaydedilir
    trial.suggest_float("ridge_alpha", *RIDGE_ALPHA_BOUNDS, log=True)

    y_val_arr = np.asarray(y_val)
    cumulative_loss = 0.0
    n_evaluated = 0

    for step, algo in enumerate(ALGOS):
        algo_params = suggest_base_params(trial, algo)
        algo_loss = 0.0

        for q in QUANTILES:
            model = train_base_learner(algo, q, X_train, y_train, params=algo_params)
            pred = model.predict(X_val)
            loss = pinball_loss(y_val_arr, np.asarray(pred), q)
            algo_loss += loss
            cumulative_loss += loss
            n_evaluated += 1

        mean_so_far = cumulative_loss / n_evaluated
        trial.report(mean_so_far, step=step)
        log.debug("Trial %d | %s | cumulative pinball=%.4f", trial.number, algo, mean_so_far)

        if trial.should_prune():
            raise optuna.TrialPruned()

    final = cumulative_loss / n_evaluated
    log.info("Trial %d tamamlandı | val pinball=%.4f", trial.number, final)
    return final


# ── Çalışma yönetimi ───────────────────────────────────────────────────────────

def run_study(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    flags_val: pd.DataFrame,
    n_trials: int = N_TRIALS_DEFAULT,
    study_name: str = "pv_optuna",
    storage: str | None = None,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """
    TPESampler + MedianPruner ile Optuna çalışması başlat.

    Args:
        storage: "sqlite:///optuna.db" gibi kalıcı depolama yolu.
                 None → in-memory (restart'ta kaybolur).

    Returns:
        optuna.Study — best_params ve tüm trial geçmişiyle
    """
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=N_WARMUP_STEPS,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, flags_val),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )

    log.info(
        "Çalışma tamamlandı | en iyi trial: %d | val pinball: %.4f",
        study.best_trial.number,
        study.best_value,
    )
    return study


# ── Parametre kayıt / yükleme ─────────────────────────────────────────────────

def save_best_params(study: optuna.Study, path: str = "best_params.json") -> None:
    """
    En iyi trial'ın parametrelerini algo başlığı altında grupla ve JSON'a yaz.

    Yapı:
        {
          "lgbm":    {param: value, ...},
          "catboost": {...},
          "xgboost":  {...},
          "ridge_alpha": float,
          "_meta": {trial_number, val_pinball}
        }
    """
    best = study.best_trial.params

    grouped: dict[str, Any] = {algo: {} for algo in ALGOS}
    for key, val in best.items():
        if key == "ridge_alpha":
            grouped["ridge_alpha"] = val
            continue
        for algo in ALGOS:
            if key.startswith(f"{algo}_"):
                param = key[len(algo) + 1:]
                grouped[algo][param] = val
                break

    grouped["_meta"] = {
        "trial_number": study.best_trial.number,
        "val_pinball":  study.best_value,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False)
    log.info("En iyi parametreler kaydedildi: %s", path)


def load_best_params(path: str = "best_params.json") -> dict[str, Any]:
    """best_params.json'ı oku ve döndür."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Görselleştirme ─────────────────────────────────────────────────────────────

def plot_study(
    study: optuna.Study,
    save_dir: str = "figures",
) -> list[Path]:
    """
    Parallel coordinate + hyperparameter importance grafikleri kaydet.

    HTML olarak kaydedilir (tarayıcıda açılabilir; kaleido gerekmez).

    Returns:
        list[Path] — kaydedilen dosya yolları
    """
    try:
        from optuna.visualization import plot_parallel_coordinate, plot_param_importances
    except ImportError as exc:
        log.warning("optuna.visualization yüklenemedi: %s", exc)
        return []

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plots = [
        ("optuna_parallel_coordinate", plot_parallel_coordinate),
        ("optuna_param_importances",   plot_param_importances),
    ]
    for name, plot_fn in plots:
        try:
            fig = plot_fn(study)
            path = out_dir / f"{name}.html"
            fig.write_html(str(path))
            saved.append(path)
            log.info("Grafik kaydedildi: %s", path)
        except Exception as exc:
            log.warning("%s oluşturulamadı: %s", name, exc)

    return saved


# ── Top-N trial özeti ──────────────────────────────────────────────────────────

def top_trials_summary(study: optuna.Study, n: int = 5) -> pd.DataFrame:
    """
    En iyi N trial'ın val pinball ve hiperparametre değerlerini DataFrame olarak döndür.

    Returns:
        pd.DataFrame — sütunlar: trial_number, val_pinball, {param_name, ...}
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value)[:n]

    rows = []
    for t in top:
        row: dict[str, Any] = {"trial_number": t.number, "val_pinball": t.value}
        row.update(t.params)
        rows.append(row)

    return pd.DataFrame(rows)
