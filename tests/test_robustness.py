"""STAGE-8: evaluation/robustness.py birim testleri."""

import random

import numpy as np
import pandas as pd
import pytest

random.seed(42)
np.random.seed(42)

from evaluation.robustness import (
    ALL_SCENARIOS,
    BURST_DURATIONS_H,
    DERIVED_ZERO_MAP,
    FREQ_MINUTES_DEFAULT,
    RANDOM_LOSS_RATES,
    SENSOR_FLAG_MAP,
    SENSOR_LOSS_TARGETS,
    RobustnessScenario,
    _corrupt_columns,
    _per_obs_pinball,
    apply_scenario,
    build_predict_fn,
    diebold_mariano_test,
    evaluate_predictions,
    plot_heatmap,
    run_all_scenarios,
)
from models.base_learners import META_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

N = 100  # test satırı


def _make_X(n: int = N) -> pd.DataFrame:
    """Tüm sensör ve türev sütunları olan test DataFrame'i."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "G":          rng.uniform(0, 1000, n),
            "T_amb":      rng.uniform(10, 40, n),
            "RH":         rng.uniform(20, 80, n),
            "wind_speed": rng.uniform(0, 10, n),
            "k_t":        rng.uniform(0, 1, n),
            "T_cell":     rng.uniform(20, 60, n),
            "cos_zenith": rng.uniform(0, 1, n),
        }
    )


def _make_flags(n: int = N) -> pd.DataFrame:
    """Sıfır başlangıçlı bayrak DataFrame'i."""
    return pd.DataFrame(
        {
            "is_G_missing":    np.zeros(n, dtype=int),
            "is_Tamb_missing": np.zeros(n, dtype=int),
            "is_RH_missing":   np.zeros(n, dtype=int),
            "is_wind_missing": np.zeros(n, dtype=int),
        }
    )


class _ConstPredictor:
    """Sabit tahmin döndüren sahte model."""
    def __init__(self, value: float = 1.0):
        self.value = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


def _make_base_models(value: float = 1.0) -> dict:
    return {col: _ConstPredictor(value) for col in META_COLS}


def _make_meta_models(value: float = 1.0) -> dict:
    return {
        "meta_q01": _ConstPredictor(value),
        "meta_q05": _ConstPredictor(value),
        "meta_q09": _ConstPredictor(value),
    }


# ── RobustnessScenario ────────────────────────────────────────────────────────

class TestRobustnessScenario:
    def test_fields(self):
        s = RobustnessScenario("test", "random", "10pct", {"rate": 0.1})
        assert s.name == "test"
        assert s.axis == "random"
        assert s.level == "10pct"
        assert s.params["rate"] == 0.1

    def test_default_params(self):
        s = RobustnessScenario("test", "burst", "1h")
        assert s.params == {}


# ── ALL_SCENARIOS ─────────────────────────────────────────────────────────────

class TestAllScenarios:
    def test_count(self):
        assert len(ALL_SCENARIOS) == 10

    def test_axes(self):
        axes = {s.axis for s in ALL_SCENARIOS}
        assert axes == {"random", "burst", "sensor"}

    def test_random_rates(self):
        random_scenarios = [s for s in ALL_SCENARIOS if s.axis == "random"]
        rates = sorted(s.params["rate"] for s in random_scenarios)
        assert rates == sorted(RANDOM_LOSS_RATES)

    def test_burst_hours(self):
        burst_scenarios = [s for s in ALL_SCENARIOS if s.axis == "burst"]
        hours = sorted(s.params["hours"] for s in burst_scenarios)
        assert hours == sorted(BURST_DURATIONS_H)

    def test_sensor_targets(self):
        sensor_scenarios = [s for s in ALL_SCENARIOS if s.axis == "sensor"]
        targets = sorted(s.params["target"] for s in sensor_scenarios)
        assert targets == sorted(SENSOR_LOSS_TARGETS)

    def test_unique_names(self):
        names = [s.name for s in ALL_SCENARIOS]
        assert len(names) == len(set(names))


# ── _corrupt_columns ──────────────────────────────────────────────────────────

class TestCorruptColumns:
    def test_sensor_zeroed(self):
        X = _make_X()
        flags = _make_flags()
        mask = np.zeros(N, dtype=bool)
        mask[:10] = True
        X_c, _ = _corrupt_columns(X, flags, ["G"], mask)
        assert (X_c["G"].iloc[:10] == 0.0).all()
        assert (X_c["G"].iloc[10:] != 0.0).any()

    def test_flag_updated(self):
        X = _make_X()
        flags = _make_flags()
        mask = np.ones(N, dtype=bool)
        _, fl_c = _corrupt_columns(X, flags, ["G"], mask)
        assert (fl_c["is_G_missing"] == 1).all()

    def test_derived_k_t_zeroed_when_G_corrupted(self):
        X = _make_X()
        flags = _make_flags()
        mask = np.ones(N, dtype=bool)
        X_c, _ = _corrupt_columns(X, flags, ["G"], mask)
        assert (X_c["k_t"] == 0.0).all()

    def test_derived_T_cell_zeroed_when_Tamb_corrupted(self):
        X = _make_X()
        flags = _make_flags()
        mask = np.ones(N, dtype=bool)
        X_c, _ = _corrupt_columns(X, flags, ["T_amb"], mask)
        assert (X_c["T_cell"] == 0.0).all()

    def test_original_unchanged(self):
        X = _make_X()
        flags = _make_flags()
        X_orig = X.copy()
        mask = np.ones(N, dtype=bool)
        _corrupt_columns(X, flags, ["G"], mask)
        pd.testing.assert_frame_equal(X, X_orig)

    def test_missing_column_skipped(self):
        X = _make_X().drop(columns=["G"])
        flags = _make_flags()
        mask = np.ones(N, dtype=bool)
        X_c, _ = _corrupt_columns(X, flags, ["G"], mask)
        assert "G" not in X_c.columns

    def test_tamb_flag_col(self):
        X = _make_X()
        flags = _make_flags()
        mask = np.ones(N, dtype=bool)
        _, fl_c = _corrupt_columns(X, flags, ["T_amb"], mask)
        assert (fl_c["is_Tamb_missing"] == 1).all()


# ── apply_scenario ────────────────────────────────────────────────────────────

class TestApplyScenario:
    def test_random_fraction(self):
        X = _make_X(n=1000)
        flags = _make_flags(n=1000)
        s = RobustnessScenario("random_25pct", "random", "25pct", {"rate": 0.25})
        X_c, _ = apply_scenario(X, flags, s, seed=42)
        zeroed = (X_c["G"] == 0.0).mean()
        assert 0.15 < zeroed < 0.40, f"Beklenen ~0.25, alınan {zeroed:.3f}"

    def test_burst_consecutive(self):
        X = _make_X(n=500)
        flags = _make_flags(n=500)
        s = RobustnessScenario("burst_1h", "burst", "1h", {"hours": 1})
        X_c, _ = apply_scenario(X, flags, s, freq_minutes=5, seed=42)
        # 1h × 60dk / 5dk = 12 adım
        zeroed_idx = np.where(X_c["G"] == 0.0)[0]
        if len(zeroed_idx) > 0:
            gaps = np.diff(zeroed_idx)
            assert (gaps == 1).all(), "Burst'ler ardışık olmalı"

    def test_burst_size(self):
        X = _make_X(n=500)
        flags = _make_flags(n=500)
        s = RobustnessScenario("burst_6h", "burst", "6h", {"hours": 6})
        X_c, _ = apply_scenario(X, flags, s, freq_minutes=5, seed=42)
        zeroed = int((X_c["G"] == 0.0).sum())
        assert zeroed == 6 * 60 // 5  # 72 adım

    def test_sensor_full_column(self):
        X = _make_X()
        flags = _make_flags()
        s = RobustnessScenario("sensor_G", "sensor", "G", {"target": "G"})
        X_c, _ = apply_scenario(X, flags, s)
        assert (X_c["G"] == 0.0).all()

    def test_sensor_flag_all_set(self):
        X = _make_X()
        flags = _make_flags()
        s = RobustnessScenario("sensor_G", "sensor", "G", {"target": "G"})
        _, fl_c = apply_scenario(X, flags, s)
        assert (fl_c["is_G_missing"] == 1).all()

    def test_sensor_rh_flag(self):
        X = _make_X()
        flags = _make_flags()
        s = RobustnessScenario("sensor_RH", "sensor", "RH", {"target": "RH"})
        _, fl_c = apply_scenario(X, flags, s)
        assert (fl_c["is_RH_missing"] == 1).all()

    def test_unknown_axis_raises(self):
        X = _make_X()
        flags = _make_flags()
        s = RobustnessScenario("bad", "unknown_axis", "x", {})
        with pytest.raises(ValueError, match="Bilinmeyen eksen"):
            apply_scenario(X, flags, s)

    def test_returns_copies(self):
        X = _make_X()
        flags = _make_flags()
        X_orig = X.copy()
        s = RobustnessScenario("random_50pct", "random", "50pct", {"rate": 0.50})
        apply_scenario(X, flags, s, seed=42)
        pd.testing.assert_frame_equal(X, X_orig)


# ── build_predict_fn ──────────────────────────────────────────────────────────

class TestBuildPredictFn:
    def test_returns_callable(self):
        fn = build_predict_fn(_make_base_models(), _make_meta_models(), use_flags=True)
        assert callable(fn)

    def test_output_keys(self):
        X = _make_X()
        flags = _make_flags()
        fn = build_predict_fn(_make_base_models(), _make_meta_models(), use_flags=True)
        preds = fn(X, flags)
        assert set(preds.keys()) == {"meta_q01", "meta_q05", "meta_q09"}

    def test_output_length(self):
        X = _make_X()
        flags = _make_flags()
        fn = build_predict_fn(_make_base_models(), _make_meta_models(), use_flags=True)
        preds = fn(X, flags)
        for arr in preds.values():
            assert len(arr) == N

    def test_noflags_callable(self):
        X = _make_X()
        flags = _make_flags()
        fn = build_predict_fn(_make_base_models(), _make_meta_models(), use_flags=False)
        preds = fn(X, flags)
        assert set(preds.keys()) == {"meta_q01", "meta_q05", "meta_q09"}


# ── _per_obs_pinball ──────────────────────────────────────────────────────────

class TestPerObsPinball:
    def test_shape(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 1.5, 4.0])
        result = _per_obs_pinball(y, pred, 0.5)
        assert result.shape == (3,)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 10, 50)
        pred = rng.uniform(0, 10, 50)
        for q in [0.1, 0.5, 0.9]:
            assert (_per_obs_pinball(y, pred, q) >= 0).all()

    def test_zero_when_perfect_at_median(self):
        y = np.array([5.0, 5.0])
        pred = np.array([5.0, 5.0])
        result = _per_obs_pinball(y, pred, 0.5)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0])

    def test_formula_below(self):
        # y < pred → (q-1)*(y-pred)
        y = np.array([1.0])
        pred = np.array([2.0])
        q = 0.1
        expected = (q - 1) * (y - pred)  # (0.1-1)*(1-2) = 0.9
        np.testing.assert_almost_equal(_per_obs_pinball(y, pred, q), expected)

    def test_formula_above(self):
        # y >= pred → q*(y-pred)
        y = np.array([3.0])
        pred = np.array([1.0])
        q = 0.9
        expected = q * (y - pred)  # 0.9 * 2 = 1.8
        np.testing.assert_almost_equal(_per_obs_pinball(y, pred, q), expected)


# ── evaluate_predictions ──────────────────────────────────────────────────────

class TestEvaluatePredictions:
    def _make_preds(self, n: int = 50):
        rng = np.random.default_rng(0)
        return {
            "meta_q01": rng.uniform(0, 3, n),
            "meta_q05": rng.uniform(3, 7, n),
            "meta_q09": rng.uniform(7, 10, n),
        }

    def test_all_keys_present(self):
        y = np.random.default_rng(0).uniform(0, 10, 50)
        result = evaluate_predictions(y, self._make_preds())
        expected = {"pinball_q01", "pinball_q05", "pinball_q09", "crps", "mae", "rmse", "coverage"}
        assert set(result.keys()) == expected

    def test_all_floats(self):
        y = np.random.default_rng(0).uniform(0, 10, 50)
        result = evaluate_predictions(y, self._make_preds())
        for k, v in result.items():
            assert isinstance(v, float), f"{k} float değil"

    def test_non_negative_losses(self):
        y = np.random.default_rng(0).uniform(0, 10, 50)
        result = evaluate_predictions(y, self._make_preds())
        for k in ("pinball_q01", "pinball_q05", "pinball_q09", "crps", "mae", "rmse"):
            assert result[k] >= 0, f"{k} negatif"

    def test_coverage_in_range(self):
        y = np.random.default_rng(0).uniform(0, 10, 50)
        result = evaluate_predictions(y, self._make_preds())
        assert 0.0 <= result["coverage"] <= 1.0

    def test_crps_is_mean_of_pinballs(self):
        y = np.random.default_rng(0).uniform(0, 10, 50)
        result = evaluate_predictions(y, self._make_preds())
        expected_crps = (result["pinball_q01"] + result["pinball_q05"] + result["pinball_q09"]) / 3
        assert abs(result["crps"] - expected_crps) < 1e-10

    def test_accepts_series(self):
        y = pd.Series(np.random.default_rng(0).uniform(0, 10, 50))
        result = evaluate_predictions(y, self._make_preds())
        assert "crps" in result

    def test_perfect_median(self):
        y = np.full(50, 5.0)
        preds = {
            "meta_q01": np.full(50, 3.0),
            "meta_q05": np.full(50, 5.0),  # tam isabetle
            "meta_q09": np.full(50, 7.0),
        }
        result = evaluate_predictions(y, preds)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)


# ── diebold_mariano_test ──────────────────────────────────────────────────────

class TestDieboldMarianoTest:
    def test_identical_losses_zero_stat(self):
        losses = np.random.default_rng(0).uniform(0, 1, 100)
        result = diebold_mariano_test(losses, losses)
        assert result["dm_stat"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["significant"] is False

    def test_keys_present(self):
        l1 = np.ones(50)
        l2 = np.zeros(50)
        result = diebold_mariano_test(l1, l2)
        assert set(result.keys()) == {"dm_stat", "p_value", "mean_diff", "significant"}

    def test_positive_stat_when_l1_larger(self):
        # l1 > l2 → model2 daha iyi → pozitif stat
        rng = np.random.default_rng(1)
        l1 = 2.0 + rng.normal(0, 0.01, 200)
        l2 = 1.0 + rng.normal(0, 0.01, 200)
        result = diebold_mariano_test(l1, l2)
        assert result["dm_stat"] > 0

    def test_negative_stat_when_l2_larger(self):
        rng = np.random.default_rng(1)
        l1 = 1.0 + rng.normal(0, 0.01, 200)
        l2 = 2.0 + rng.normal(0, 0.01, 200)
        result = diebold_mariano_test(l1, l2)
        assert result["dm_stat"] < 0

    def test_significant_for_clear_difference(self):
        rng = np.random.default_rng(0)
        l1 = rng.uniform(5, 6, 500)
        l2 = rng.uniform(0, 1, 500)
        result = diebold_mariano_test(l1, l2)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_not_significant_for_noise(self):
        rng = np.random.default_rng(0)
        l1 = rng.uniform(0, 1, 50)
        l2 = rng.uniform(0, 1, 50)
        result = diebold_mariano_test(l1, l2)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_mean_diff_correct(self):
        l1 = np.full(100, 3.0)
        l2 = np.full(100, 1.0)
        result = diebold_mariano_test(l1, l2)
        assert result["mean_diff"] == pytest.approx(2.0)

    def test_zero_variance_guard(self):
        l1 = np.zeros(10)
        l2 = np.zeros(10)
        result = diebold_mariano_test(l1, l2)
        assert result["dm_stat"] == 0.0
        assert result["p_value"] == 1.0


# ── run_all_scenarios ─────────────────────────────────────────────────────────

class TestRunAllScenarios:
    def _setup(self, n: int = N):
        X = _make_X(n)
        flags = _make_flags(n)
        y = np.random.default_rng(0).uniform(0, 10, n)
        fn_f  = build_predict_fn(_make_base_models(2.0), _make_meta_models(5.0), use_flags=True)
        fn_nf = build_predict_fn(_make_base_models(2.0), _make_meta_models(5.0), use_flags=False)
        return X, flags, y, fn_f, fn_nf

    def test_returns_dict_keys(self):
        X, flags, y, fn_f, fn_nf = self._setup()
        results = run_all_scenarios(X, y, flags, fn_f, fn_nf)
        assert set(results.keys()) == {"flags", "noflags", "dm"}

    def test_flags_shape(self):
        X, flags, y, fn_f, fn_nf = self._setup()
        results = run_all_scenarios(X, y, flags, fn_f, fn_nf)
        # baseline + 10 senaryo = 11 satır
        assert len(results["flags"]) == 11
        assert len(results["noflags"]) == 11

    def test_dm_shape(self):
        X, flags, y, fn_f, fn_nf = self._setup()
        results = run_all_scenarios(X, y, flags, fn_f, fn_nf)
        # baseline'da DM yok → 10 satır
        assert len(results["dm"]) == 10

    def test_baseline_row_present(self):
        X, flags, y, fn_f, fn_nf = self._setup()
        results = run_all_scenarios(X, y, flags, fn_f, fn_nf)
        assert "baseline" in results["flags"].index
        assert "baseline" not in results["dm"].index

    def test_metric_columns(self):
        X, flags, y, fn_f, fn_nf = self._setup()
        results = run_all_scenarios(X, y, flags, fn_f, fn_nf)
        expected = {"pinball_q01", "pinball_q05", "pinball_q09", "crps", "mae", "rmse", "coverage"}
        assert expected.issubset(set(results["flags"].columns))

    def test_dm_columns(self):
        X, flags, y, fn_f, fn_nf = self._setup()
        results = run_all_scenarios(X, y, flags, fn_f, fn_nf)
        expected = {"dm_stat", "p_value", "mean_diff", "significant"}
        assert expected.issubset(set(results["dm"].columns))

    def test_scenario_names_in_index(self):
        X, flags, y, fn_f, fn_nf = self._setup()
        results = run_all_scenarios(X, y, flags, fn_f, fn_nf)
        for s in ALL_SCENARIOS:
            assert s.name in results["flags"].index
            assert s.name in results["dm"].index


# ── plot_heatmap ──────────────────────────────────────────────────────────────

class TestPlotHeatmap:
    def _make_results(self) -> dict:
        index = ["baseline"] + [s.name for s in ALL_SCENARIOS]
        metrics = ["pinball_q01", "pinball_q05", "pinball_q09", "crps", "mae", "rmse", "coverage"]
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.uniform(0, 5, (len(index), len(metrics))),
            index=index,
            columns=metrics,
        )
        dm_index = [s.name for s in ALL_SCENARIOS]
        n_s = len(dm_index)
        dm_df = pd.DataFrame(
            {"dm_stat": rng.normal(0, 1, n_s), "p_value": rng.uniform(0, 1, n_s),
             "mean_diff": rng.normal(0, 0.5, n_s), "significant": [False] * n_s},
            index=dm_index,
        )
        return {"flags": df, "noflags": df.copy(), "dm": dm_df}

    def test_returns_path(self, tmp_path):
        results = self._make_results()
        path = plot_heatmap(results, metric="crps", save_dir=str(tmp_path))
        assert path.exists()
        assert path.suffix == ".png"

    def test_file_nonempty(self, tmp_path):
        results = self._make_results()
        path = plot_heatmap(results, metric="crps", save_dir=str(tmp_path))
        assert path.stat().st_size > 0

    def test_metric_in_filename(self, tmp_path):
        results = self._make_results()
        path = plot_heatmap(results, metric="mae", save_dir=str(tmp_path))
        assert "mae" in path.name
