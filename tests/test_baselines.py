"""STAGE-9: models/baselines.py birim testleri."""

import random

import numpy as np
import pytest
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from models.baselines import (
    BASELINE_REGISTRY,
    KNNQuantile,
    LSTMQuantile,
    LightTFTQuantile,
    SVMQuantile,
    evaluate_baseline,
    evaluate_quantiles,
    make_sequences,
    predict_baseline,
    train_baseline,
)

# ── Sabitler ───────────────────────────────────────────────────────────────────

N_FEAT  = 5
N_TRAIN = 100
N_TEST  = 50
SEQ     = 4     # küçük seq_len → hızlı test
EPOCHS  = 2     # hızlı training
HIDDEN  = 8
D       = 8


def _data(n: int = N_TRAIN, f: int = N_FEAT, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (n, f)).astype(np.float32)
    y = rng.uniform(0, 5, n).astype(np.float32)
    return X, y


def _preds(n: int = N_TEST) -> dict:
    rng = np.random.default_rng(0)
    return {
        "q01": rng.uniform(0, 2, n).astype(np.float64),
        "q05": rng.uniform(2, 4, n).astype(np.float64),
        "q09": rng.uniform(4, 6, n).astype(np.float64),
    }


# ── make_sequences ─────────────────────────────────────────────────────────────

class TestMakeSequences:
    def test_x_shape(self):
        X, y = _data(50)
        Xs, ys = make_sequences(X, y, seq_len=4)
        assert Xs.shape == (46, 4, N_FEAT)

    def test_y_len(self):
        X, y = _data(50)
        Xs, ys = make_sequences(X, y, seq_len=4)
        assert len(ys) == 46

    def test_y_alignment(self):
        X, y = _data(20)
        _, ys = make_sequences(X, y, seq_len=5)
        np.testing.assert_array_equal(ys, y[5:])

    def test_stride_correct(self):
        X = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
        y = np.zeros(10)
        Xs, _ = make_sequences(X, y, seq_len=3)
        # Dizi 0 = X[0:3], dizi 1 = X[1:4]
        np.testing.assert_array_equal(Xs[0], X[0:3])
        np.testing.assert_array_equal(Xs[1], X[1:4])


# ── KNNQuantile ────────────────────────────────────────────────────────────────

class TestKNNQuantile:
    def test_fit_returns_self(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=5)
        assert model.fit(X, y) is model

    def test_predict_keys(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=5).fit(X, y)
        preds = model.predict_quantiles(X[:10])
        assert set(preds.keys()) == {"q01", "q05", "q09"}

    def test_predict_shape(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=5).fit(X, y)
        preds = model.predict_quantiles(X[:10])
        for arr in preds.values():
            assert arr.shape == (10,)

    def test_monotonicity_mean(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=5).fit(X, y)
        preds = model.predict_quantiles(X)
        assert preds["q01"].mean() <= preds["q05"].mean()
        assert preds["q05"].mean() <= preds["q09"].mean()

    def test_no_nan(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=5).fit(X, y)
        preds = model.predict_quantiles(X)
        for arr in preds.values():
            assert np.isfinite(arr).all()


# ── SVMQuantile ────────────────────────────────────────────────────────────────

class TestSVMQuantile:
    def _model(self):
        return SVMQuantile(n_components=20, calib_frac=0.2)

    def test_fit_returns_self(self):
        X, y = _data()
        assert self._model().fit(X, y) is not None

    def test_predict_keys(self):
        X, y = _data()
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X[:10])
        assert set(preds.keys()) == {"q01", "q05", "q09"}

    def test_predict_shape(self):
        X, y = _data()
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X[:10])
        for arr in preds.values():
            assert arr.shape == (10,)

    def test_q01_lt_q09(self):
        X, y = _data()
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        assert (preds["q09"] >= preds["q01"]).all()

    def test_no_nan(self):
        X, y = _data()
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        for arr in preds.values():
            assert np.isfinite(arr).all()

    def test_calib_residuals_set(self):
        X, y = _data()
        model = self._model().fit(X, y)
        assert model._dq_low < model._dq_high


# ── LSTMQuantile ───────────────────────────────────────────────────────────────

class TestLSTMQuantile:
    def _model(self):
        return LSTMQuantile(seq_len=SEQ, hidden=HIDDEN, epochs=EPOCHS, batch_size=64, patience=2)

    def test_fit_runs(self):
        X, y = _data(60)
        model = self._model()
        model.fit(X, y)
        assert model._net is not None

    def test_predict_keys(self):
        X, y = _data(60)
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        assert set(preds.keys()) == {"q01", "q05", "q09"}

    def test_predict_shape(self):
        X, y = _data(60)
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        expected = len(X) - SEQ
        for arr in preds.values():
            assert arr.shape == (expected,)

    def test_no_nan(self):
        X, y = _data(60)
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        for arr in preds.values():
            assert np.isfinite(arr).all()


# ── LightTFTQuantile ───────────────────────────────────────────────────────────

class TestLightTFTQuantile:
    def _model(self):
        return LightTFTQuantile(
            seq_len=SEQ, d_model=D, nhead=2,
            epochs=EPOCHS, batch_size=64, patience=2,
        )

    def test_fit_runs(self):
        X, y = _data(60)
        model = self._model()
        model.fit(X, y)
        assert model._net is not None

    def test_predict_keys(self):
        X, y = _data(60)
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        assert set(preds.keys()) == {"q01", "q05", "q09"}

    def test_predict_shape(self):
        X, y = _data(60)
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        expected = len(X) - SEQ
        for arr in preds.values():
            assert arr.shape == (expected,)

    def test_no_nan(self):
        X, y = _data(60)
        model = self._model().fit(X, y)
        preds = model.predict_quantiles(X)
        for arr in preds.values():
            assert np.isfinite(arr).all()

    def test_grn_skip_connection(self):
        from models.baselines import _GRN
        grn = _GRN(8)
        x = torch.randn(4, 8)
        out = grn(x)
        assert out.shape == (4, 8)


# ── evaluate_quantiles ─────────────────────────────────────────────────────────

class TestEvaluateQuantiles:
    def test_all_keys(self):
        y = np.ones(50)
        result = evaluate_quantiles(y, _preds(50))
        expected = {"pinball_q01", "pinball_q05", "pinball_q09", "crps", "mae", "rmse", "coverage"}
        assert set(result.keys()) == expected

    def test_all_floats(self):
        y = np.ones(50)
        result = evaluate_quantiles(y, _preds(50))
        for k, v in result.items():
            assert isinstance(v, float), f"{k} float değil"

    def test_non_negative_losses(self):
        y = np.random.default_rng(0).uniform(0, 5, 50)
        result = evaluate_quantiles(y, _preds(50))
        for k in ("pinball_q01", "pinball_q05", "pinball_q09", "crps", "mae", "rmse"):
            assert result[k] >= 0.0

    def test_coverage_in_range(self):
        y = np.random.default_rng(0).uniform(0, 5, 50)
        result = evaluate_quantiles(y, _preds(50))
        assert 0.0 <= result["coverage"] <= 1.0

    def test_crps_mean_of_pinballs(self):
        y = np.random.default_rng(0).uniform(0, 5, 50)
        result = evaluate_quantiles(y, _preds(50))
        expected = (result["pinball_q01"] + result["pinball_q05"] + result["pinball_q09"]) / 3
        assert abs(result["crps"] - expected) < 1e-10

    def test_perfect_median_zero_mae(self):
        y = np.full(50, 3.0)
        preds = {"q01": np.full(50, 1.0), "q05": np.full(50, 3.0), "q09": np.full(50, 5.0)}
        result = evaluate_quantiles(y, preds)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)


# ── train_baseline / predict_baseline / evaluate_baseline ─────────────────────

class TestTrainBaseline:
    def test_returns_knn(self, tmp_path):
        X, y = _data()
        model = train_baseline("knn", X, y, str(tmp_path), n_neighbors=3)
        assert isinstance(model, KNNQuantile)

    def test_returns_svm(self, tmp_path):
        X, y = _data()
        model = train_baseline("svm", X, y, str(tmp_path), n_components=20)
        assert isinstance(model, SVMQuantile)

    def test_returns_lstm(self, tmp_path):
        X, y = _data(60)
        model = train_baseline(
            "lstm", X, y, str(tmp_path),
            seq_len=SEQ, hidden=HIDDEN, epochs=EPOCHS, batch_size=64,
        )
        assert isinstance(model, LSTMQuantile)

    def test_returns_tft(self, tmp_path):
        X, y = _data(60)
        model = train_baseline(
            "tft", X, y, str(tmp_path),
            seq_len=SEQ, d_model=D, nhead=2, epochs=EPOCHS, batch_size=64,
        )
        assert isinstance(model, LightTFTQuantile)

    def test_file_saved(self, tmp_path):
        X, y = _data()
        train_baseline("knn", X, y, str(tmp_path), n_neighbors=3)
        assert (tmp_path / "baseline_knn.joblib").exists()

    def test_unknown_name_raises(self, tmp_path):
        X, y = _data()
        with pytest.raises(ValueError, match="Bilinmeyen baseline"):
            train_baseline("unknown", X, y, str(tmp_path))


class TestPredictBaseline:
    def test_keys(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=3).fit(X, y)
        preds = predict_baseline("knn", model, X[:10])
        assert set(preds.keys()) == {"q01", "q05", "q09"}

    def test_shape(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=3).fit(X, y)
        preds = predict_baseline("knn", model, X[:10])
        for arr in preds.values():
            assert arr.shape == (10,)


class TestEvaluateBaseline:
    def test_knn_metric_keys(self):
        X, y = _data()
        model = KNNQuantile(n_neighbors=3).fit(X, y)
        result = evaluate_baseline("knn", model, X[:20], y[:20])
        assert "crps" in result and "coverage" in result

    def test_lstm_y_alignment(self):
        X, y = _data(60)
        model = LSTMQuantile(seq_len=SEQ, hidden=HIDDEN, epochs=EPOCHS, batch_size=64).fit(X, y)
        result = evaluate_baseline("lstm", model, X[:30], y[:30])
        # crps hesaplanabilir olmalı (hizalama hatası yoksa)
        assert np.isfinite(result["crps"])

    def test_tft_y_alignment(self):
        X, y = _data(60)
        model = LightTFTQuantile(
            seq_len=SEQ, d_model=D, nhead=2, epochs=EPOCHS, batch_size=64,
        ).fit(X, y)
        result = evaluate_baseline("tft", model, X[:30], y[:30])
        assert np.isfinite(result["crps"])


# ── BASELINE_REGISTRY ─────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_keys_present(self):
        assert set(BASELINE_REGISTRY.keys()) == {"knn", "svm", "lstm", "tft"}

    def test_all_instantiable(self):
        for cls in BASELINE_REGISTRY.values():
            obj = cls()
            assert hasattr(obj, "fit") and hasattr(obj, "predict_quantiles")
