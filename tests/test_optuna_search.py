"""Unit tests for optimization/optuna_search.py — STAGE-7"""

import json
import random

import numpy as np
import optuna
import pandas as pd
import pytest

random.seed(42)
np.random.seed(42)

optuna.logging.set_verbosity(optuna.logging.WARNING)

from models.base_learners import ALGOS, META_COLS, QUANTILES
from models.meta_learner import FLAG_COLS
from optimization.optuna_search import (
    SEARCH_SPACE,
    load_best_params,
    objective,
    plot_study,
    run_study,
    save_best_params,
    suggest_base_params,
    top_trials_summary,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _synthetic(
    n_train: int = 200,
    n_val: int = 60,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = n_train + n_val
    X = pd.DataFrame(rng.uniform(0, 1, (n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(rng.uniform(0, 100, n), name="target")
    flags = pd.DataFrame(
        {col: rng.integers(0, 2, n) for col in FLAG_COLS},
    )
    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]
    flags_val = flags.iloc[n_train:]
    return X_train, y_train, X_val, y_val, flags_val


def _make_study(n_trials: int = 3) -> optuna.Study:
    X_train, y_train, X_val, y_val, flags_val = _synthetic()
    return run_study(
        X_train, y_train, X_val, y_val, flags_val,
        n_trials=n_trials,
        show_progress_bar=False,
    )


# ── SEARCH_SPACE ───────────────────────────────────────────────────────────────

def test_search_space_contains_all_algos() -> None:
    assert set(SEARCH_SPACE.keys()) == set(ALGOS)


def test_search_space_required_params() -> None:
    for algo in ALGOS:
        assert "learning_rate" in SEARCH_SPACE[algo]
        assert "subsample" in SEARCH_SPACE[algo]


# ── suggest_base_params ────────────────────────────────────────────────────────

@pytest.mark.parametrize("algo", ALGOS)
def test_suggest_base_params_keys(algo: str) -> None:
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_base_params(trial, algo)
    expected = set(SEARCH_SPACE[algo].keys())
    assert set(params.keys()) == expected


@pytest.mark.parametrize("algo", ALGOS)
def test_suggest_base_params_bounds(algo: str) -> None:
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_base_params(trial, algo)
    for param, spec in SEARCH_SPACE[algo].items():
        lo, hi, _ = spec
        assert lo <= params[param] <= hi, f"{algo}.{param}={params[param]} out of [{lo},{hi}]"


def test_suggest_base_params_no_prefix_in_keys() -> None:
    """Dönen dict'te 'lgbm_' prefix'i olmamalı — train_base_learner'a geçirilir."""
    study = optuna.create_study()
    trial = study.ask()
    params = suggest_base_params(trial, "lgbm")
    for key in params:
        assert not key.startswith("lgbm_"), f"Prefix sızdı: {key}"


# ── objective ──────────────────────────────────────────────────────────────────

def test_objective_returns_float() -> None:
    X_train, y_train, X_val, y_val, flags_val = _synthetic()
    study = optuna.create_study()
    trial = study.ask()
    val = objective(trial, X_train, y_train, X_val, y_val, flags_val)
    assert isinstance(val, float)


def test_objective_nonnegative() -> None:
    X_train, y_train, X_val, y_val, flags_val = _synthetic()
    study = optuna.create_study()
    trial = study.ask()
    val = objective(trial, X_train, y_train, X_val, y_val, flags_val)
    assert val >= 0.0


def test_objective_suggests_ridge_alpha() -> None:
    """ridge_alpha trial params'a eklenmiş olmalı."""
    X_train, y_train, X_val, y_val, flags_val = _synthetic()
    study = optuna.create_study()
    trial = study.ask()
    objective(trial, X_train, y_train, X_val, y_val, flags_val)
    assert "ridge_alpha" in trial.params


# ── run_study ──────────────────────────────────────────────────────────────────

def test_run_study_returns_study() -> None:
    study = _make_study(n_trials=2)
    assert isinstance(study, optuna.Study)


def test_run_study_n_trials() -> None:
    study = _make_study(n_trials=3)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) >= 1  # MedianPruner prune edebilir; en az 1 tamamlanmış olmalı


def test_run_study_best_value_finite() -> None:
    study = _make_study(n_trials=2)
    assert np.isfinite(study.best_value)


def test_run_study_best_value_nonnegative() -> None:
    study = _make_study(n_trials=2)
    assert study.best_value >= 0.0


def test_run_study_direction_minimize() -> None:
    study = _make_study(n_trials=2)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


# ── save / load best_params ────────────────────────────────────────────────────

def test_save_best_params_creates_file(tmp_path) -> None:
    study = _make_study(n_trials=2)
    path = str(tmp_path / "best_params.json")
    save_best_params(study, path)
    assert (tmp_path / "best_params.json").exists()


def test_save_best_params_valid_json(tmp_path) -> None:
    study = _make_study(n_trials=2)
    path = str(tmp_path / "best_params.json")
    save_best_params(study, path)
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_save_best_params_contains_algos(tmp_path) -> None:
    study = _make_study(n_trials=2)
    path = str(tmp_path / "best_params.json")
    save_best_params(study, path)
    data = load_best_params(path)
    for algo in ALGOS:
        assert algo in data, f"{algo} best_params.json'da yok"


def test_save_best_params_contains_ridge_alpha(tmp_path) -> None:
    study = _make_study(n_trials=2)
    path = str(tmp_path / "best_params.json")
    save_best_params(study, path)
    data = load_best_params(path)
    assert "ridge_alpha" in data


def test_save_best_params_meta_block(tmp_path) -> None:
    study = _make_study(n_trials=2)
    path = str(tmp_path / "best_params.json")
    save_best_params(study, path)
    data = load_best_params(path)
    assert "_meta" in data
    assert "trial_number" in data["_meta"]
    assert "val_pinball" in data["_meta"]


def test_load_best_params_roundtrip(tmp_path) -> None:
    study = _make_study(n_trials=2)
    path = str(tmp_path / "best_params.json")
    save_best_params(study, path)
    data = load_best_params(path)
    assert data["_meta"]["val_pinball"] == pytest.approx(study.best_value, rel=1e-6)


# ── top_trials_summary ─────────────────────────────────────────────────────────

def test_top_trials_summary_shape() -> None:
    study = _make_study(n_trials=3)
    df = top_trials_summary(study, n=2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2


def test_top_trials_summary_sorted() -> None:
    study = _make_study(n_trials=3)
    df = top_trials_summary(study, n=5)
    vals = df["val_pinball"].tolist()
    assert vals == sorted(vals), "En iyi trial'lar sıralı değil"


def test_top_trials_summary_has_required_cols() -> None:
    study = _make_study(n_trials=2)
    df = top_trials_summary(study, n=1)
    assert "trial_number" in df.columns
    assert "val_pinball" in df.columns
