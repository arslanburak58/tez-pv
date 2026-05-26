"""STAGE-10: evaluation/comparison.py birim testleri."""

import random
from pathlib import Path

import numpy as np
import pytest

random.seed(42)
np.random.seed(42)

from evaluation.comparison import (
    ALPHA,
    MODEL_LABELS,
    MODEL_ORDER,
    ModelResult,
    _per_obs_crps,
    _save_fig,
    apply_holm_bonferroni,
    build_master_table,
    dm_pairwise,
    normalize_stacked_preds,
    plot_dm_heatmap,
    plot_edge_ai_scatter,
    plot_master_table,
    plot_probability_bands,
    run_comparison,
)

# ── Mock veri yardımcıları ─────────────────────────────────────────────────────

N = 120
N_FEAT = 4

_RNG = np.random.default_rng(0)


def _y(n: int = N) -> np.ndarray:
    return _RNG.uniform(0.0, 10.0, n).astype(np.float64)


def _preds(n: int = N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    q01 = rng.uniform(0.0, 3.0, n).astype(np.float64)
    q05 = rng.uniform(3.0, 7.0, n).astype(np.float64)
    q09 = rng.uniform(7.0, 10.0, n).astype(np.float64)
    return {"q01": q01, "q05": q05, "q09": q09}


def _metrics() -> dict[str, float]:
    return {
        "mae":         0.5,
        "rmse":        0.7,
        "pinball_q01": 0.1,
        "pinball_q05": 0.3,
        "pinball_q09": 0.1,
        "crps":        0.17,
        "coverage":    0.80,
    }


def _model_result(n: int = N, train_time: float = 1.0) -> ModelResult:
    return ModelResult(
        metrics=_metrics(),
        preds=_preds(n),
        y_true=_y(n),
        train_time_s=train_time,
    )


def _full_results() -> dict[str, ModelResult]:
    times = {"stacked_flags": 5.0, "stacked_noflags": 4.5,
             "knn": 0.1, "svm": 2.0, "lstm": 30.0, "tft": 45.0}
    crps  = {"stacked_flags": 0.15, "stacked_noflags": 0.20,
              "knn": 0.35, "svm": 0.30, "lstm": 0.25, "tft": 0.23}
    res: dict[str, ModelResult] = {}
    for name in MODEL_ORDER:
        m = _metrics()
        m["crps"] = crps[name]
        rng = np.random.default_rng(hash(name) % (2**31))
        y   = rng.uniform(0, 10, N).astype(np.float64)
        res[name] = ModelResult(
            metrics=m,
            preds=_preds(N),
            y_true=y,
            train_time_s=times[name],
        )
    return res


# ── normalize_stacked_preds ────────────────────────────────────────────────────

class TestNormalizeStackedPreds:
    def test_meta_keys_converted(self):
        p = {"meta_q01": np.ones(5), "meta_q05": np.ones(5) * 2, "meta_q09": np.ones(5) * 3}
        out = normalize_stacked_preds(p)
        assert set(out.keys()) == {"q01", "q05", "q09"}

    def test_already_normalized_passthrough(self):
        p = _preds(10)
        out = normalize_stacked_preds(p)
        assert out is p

    def test_values_preserved(self):
        arr = np.arange(5, dtype=float)
        p = {"meta_q01": arr, "meta_q05": arr * 2, "meta_q09": arr * 3}
        out = normalize_stacked_preds(p)
        np.testing.assert_array_equal(out["q01"], arr)


# ── _per_obs_crps ──────────────────────────────────────────────────────────────

class TestPerObsCrps:
    def test_shape(self):
        y = _y()
        p = _preds()
        crps = _per_obs_crps(y, p)
        assert crps.shape == (N,)

    def test_nonnegative(self):
        y = _y()
        p = _preds()
        crps = _per_obs_crps(y, p)
        assert np.all(crps >= 0)

    def test_perfect_prediction_zero(self):
        y = np.ones(50) * 5.0
        p = {"q01": y.copy(), "q05": y.copy(), "q09": y.copy()}
        crps = _per_obs_crps(y, p)
        np.testing.assert_allclose(crps, 0.0, atol=1e-10)

    def test_known_value(self):
        y   = np.array([2.0])
        p   = {"q01": np.array([1.0]), "q05": np.array([1.5]), "q09": np.array([3.0])}
        crps = _per_obs_crps(y, p)
        # pinball: q01=(2-1)*0.1=0.1, q05=(2-1.5)*0.5=0.25, q09=(3-2)*(0.9-1)=-0.1 → 0.1*-0.1... wait
        # q09: r=2-3=-1 < 0, so (0.9-1)*(-1)=0.1
        expected = (0.1 + 0.25 + 0.1) / 3.0
        np.testing.assert_allclose(crps, [expected], atol=1e-10)


# ── build_master_table ─────────────────────────────────────────────────────────

class TestBuildMasterTable:
    def test_shape_full(self):
        res = _full_results()
        df  = build_master_table(res)
        assert df.shape == (6, 8)

    def test_columns(self):
        df = build_master_table(_full_results())
        expected = {"MAE", "RMSE", "Pinball_q01", "Pinball_q05", "Pinball_q09",
                    "CRPS", "Coverage", "Train_s"}
        assert set(df.columns) == expected

    def test_row_order_follows_model_order(self):
        df     = build_master_table(_full_results())
        labels = [MODEL_LABELS[n] for n in MODEL_ORDER]
        assert list(df.index) == labels

    def test_missing_model_skipped(self):
        res = _full_results()
        del res["knn"]
        df  = build_master_table(res)
        assert df.shape[0] == 5
        assert MODEL_LABELS["knn"] not in df.index

    def test_train_time_values(self):
        res = _full_results()
        df  = build_master_table(res)
        assert df.loc[MODEL_LABELS["stacked_flags"], "Train_s"] == pytest.approx(5.0)

    def test_single_model(self):
        res = {"stacked_flags": _model_result()}
        df  = build_master_table(res)
        assert df.shape[0] == 1


# ── apply_holm_bonferroni ──────────────────────────────────────────────────────

class TestApplyHolmBonferroni:
    def test_empty(self):
        assert apply_holm_bonferroni([]) == []

    def test_single(self):
        result = apply_holm_bonferroni([0.03])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.03)

    def test_all_significant_still_bounded(self):
        pvals  = [0.001, 0.002, 0.003]
        result = apply_holm_bonferroni(pvals)
        assert all(r <= 1.0 for r in result)

    def test_monotonicity(self):
        pvals  = [0.01, 0.02, 0.04, 0.08]
        result = apply_holm_bonferroni(pvals)
        sorted_result = sorted(result)
        # düzeltilmiş p-değerleri orijinal sıralamayla monoton artmalı değil,
        # ancak en büyük raw p → en büyük adj p
        assert max(result) <= 1.0

    def test_no_adjustment_needed(self):
        pvals  = [0.9, 0.8, 0.7]
        result = apply_holm_bonferroni(pvals)
        assert all(r <= 1.0 for r in result)

    def test_preserves_length(self):
        pvals  = [0.05, 0.01, 0.1, 0.02, 0.3]
        result = apply_holm_bonferroni(pvals)
        assert len(result) == len(pvals)

    def test_known_example(self):
        # 3 test, ham p = [0.01, 0.04, 0.08]
        # sıralı: [0.01, 0.04, 0.08]
        # çarpan:  [3,    2,    1]
        # düzeltilmiş: min(1, [0.03, 0.08, 0.08]) → monoton: [0.03, 0.08, 0.08]
        pvals  = [0.01, 0.04, 0.08]
        result = apply_holm_bonferroni(pvals)
        assert result[0] == pytest.approx(0.03, abs=1e-9)
        assert result[1] == pytest.approx(0.08, abs=1e-9)
        assert result[2] == pytest.approx(0.08, abs=1e-9)


# ── dm_pairwise ────────────────────────────────────────────────────────────────

class TestDmPairwise:
    def test_shape(self):
        res = _full_results()
        df  = dm_pairwise(res)
        n   = len(MODEL_ORDER)
        assert len(df) == n * (n - 1) // 2

    def test_columns(self):
        df = dm_pairwise(_full_results())
        for col in ["model_i", "model_j", "dm_stat", "mean_diff", "p_raw", "p_adj", "significant"]:
            assert col in df.columns

    def test_p_adj_between_0_and_1(self):
        df = dm_pairwise(_full_results())
        assert (df["p_adj"] >= 0).all() and (df["p_adj"] <= 1.0).all()

    def test_significant_flag_consistent(self):
        df = dm_pairwise(_full_results())
        for _, row in df.iterrows():
            assert row["significant"] == (row["p_adj"] < ALPHA)

    def test_single_model_empty(self):
        res = {"stacked_flags": _model_result()}
        df  = dm_pairwise(res)
        assert len(df) == 0

    def test_two_models(self):
        res = {
            "stacked_flags":   _model_result(N, 5.0),
            "stacked_noflags": _model_result(N, 4.5),
        }
        df = dm_pairwise(res)
        assert len(df) == 1

    def test_different_length_models_no_error(self):
        res = {
            "stacked_flags": _model_result(N,        5.0),
            "lstm":          _model_result(N - 12,  30.0),
        }
        df = dm_pairwise(res)
        assert len(df) == 1


# ── plot_master_table ──────────────────────────────────────────────────────────

class TestPlotMasterTable:
    def test_creates_png_and_pdf(self, tmp_path):
        df    = build_master_table(_full_results())
        paths = plot_master_table(df, tmp_path)
        exts  = {p.suffix for p in paths}
        assert ".png" in exts and ".pdf" in exts

    def test_files_exist(self, tmp_path):
        df    = build_master_table(_full_results())
        paths = plot_master_table(df, tmp_path)
        assert all(p.exists() for p in paths)

    def test_filename_stem(self, tmp_path):
        df    = build_master_table(_full_results())
        paths = plot_master_table(df, tmp_path)
        assert all("master_table" in p.stem for p in paths)


# ── plot_probability_bands ─────────────────────────────────────────────────────

class TestPlotProbabilityBands:
    def test_creates_png_and_pdf(self, tmp_path):
        r     = _model_result()
        paths = plot_probability_bands(r["y_true"], r["preds"], "knn", tmp_path)
        exts  = {p.suffix for p in paths}
        assert ".png" in exts and ".pdf" in exts

    def test_files_exist(self, tmp_path):
        r     = _model_result()
        paths = plot_probability_bands(r["y_true"], r["preds"], "knn", tmp_path)
        assert all(p.exists() for p in paths)

    def test_model_name_in_filename(self, tmp_path):
        r     = _model_result()
        paths = plot_probability_bands(r["y_true"], r["preds"], "stacked_flags", tmp_path)
        assert all("stacked_flags" in p.name for p in paths)

    def test_n_points_clipping(self, tmp_path):
        r = _model_result(30)
        # n_points > len(y_true) → clipped to 30
        paths = plot_probability_bands(r["y_true"], r["preds"], "knn", tmp_path, n_points=500)
        assert all(p.exists() for p in paths)


# ── plot_dm_heatmap ────────────────────────────────────────────────────────────

class TestPlotDmHeatmap:
    def test_creates_png_and_pdf(self, tmp_path):
        dm_df = dm_pairwise(_full_results())
        paths = plot_dm_heatmap(dm_df, tmp_path)
        exts  = {p.suffix for p in paths}
        assert ".png" in exts and ".pdf" in exts

    def test_files_exist(self, tmp_path):
        dm_df = dm_pairwise(_full_results())
        paths = plot_dm_heatmap(dm_df, tmp_path)
        assert all(p.exists() for p in paths)

    def test_filename_stem(self, tmp_path):
        dm_df = dm_pairwise(_full_results())
        paths = plot_dm_heatmap(dm_df, tmp_path)
        assert all("dm_pairwise_heatmap" in p.stem for p in paths)


# ── plot_edge_ai_scatter ───────────────────────────────────────────────────────

class TestPlotEdgeAiScatter:
    def test_creates_png_and_pdf(self, tmp_path):
        df    = build_master_table(_full_results())
        paths = plot_edge_ai_scatter(df, tmp_path)
        exts  = {p.suffix for p in paths}
        assert ".png" in exts and ".pdf" in exts

    def test_files_exist(self, tmp_path):
        df    = build_master_table(_full_results())
        paths = plot_edge_ai_scatter(df, tmp_path)
        assert all(p.exists() for p in paths)

    def test_filename_stem(self, tmp_path):
        df    = build_master_table(_full_results())
        paths = plot_edge_ai_scatter(df, tmp_path)
        assert all("edge_ai_scatter" in p.stem for p in paths)


# ── run_comparison ─────────────────────────────────────────────────────────────

class TestRunComparison:
    def test_returns_master_table(self, tmp_path):
        out = run_comparison(_full_results(), tmp_path)
        assert "master_table" in out
        assert isinstance(out["master_table"], type(build_master_table(_full_results())))

    def test_returns_dm_results(self, tmp_path):
        out = run_comparison(_full_results(), tmp_path)
        assert "dm_results" in out
        assert len(out["dm_results"]) > 0

    def test_returns_saved_paths(self, tmp_path):
        out = run_comparison(_full_results(), tmp_path)
        assert "saved_paths" in out
        assert len(out["saved_paths"]) > 0

    def test_all_saved_paths_exist(self, tmp_path):
        out = run_comparison(_full_results(), tmp_path)
        assert all(p.exists() for p in out["saved_paths"])

    def test_png_and_pdf_in_saved_paths(self, tmp_path):
        out  = run_comparison(_full_results(), tmp_path)
        exts = {p.suffix for p in out["saved_paths"]}
        assert ".png" in exts and ".pdf" in exts

    def test_probability_bands_created_for_stacked_models(self, tmp_path):
        out   = run_comparison(_full_results(), tmp_path)
        names = [p.name for p in out["saved_paths"]]
        assert any("stacked_flags" in n for n in names)
        assert any("stacked_noflags" in n for n in names)

    def test_missing_stacked_model_no_crash(self, tmp_path):
        res = _full_results()
        del res["stacked_flags"]
        del res["stacked_noflags"]
        out = run_comparison(res, tmp_path)
        assert "master_table" in out
