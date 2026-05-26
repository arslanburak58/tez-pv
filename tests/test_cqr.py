"""Unit tests for evaluation/cqr.py"""

import numpy as np
import pytest

from evaluation.cqr import apply_cqr_correction, compute_cqr_offset


def _synthetic_bands(
    n: int = 1000,
    seed: int = 42,
    width: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gerçek değerler ve dar bant üret (kasıtlı düşük coverage)."""
    rng = np.random.default_rng(seed)
    y   = rng.normal(50, 20, n)
    q01 = y - width   # dar bant → düşük coverage
    q09 = y + width
    return y, q01, q09


class TestComputeCqrOffset:
    def test_offset_is_float(self):
        y, q01, q09 = _synthetic_bands()
        off = compute_cqr_offset(y, q01, q09, alpha=0.20)
        assert isinstance(off, float)

    def test_offset_positive_for_bad_band(self):
        """Sabit/kötü bant y'yi takip etmiyorsa offset pozitif olmalı."""
        rng = np.random.default_rng(42)
        n   = 1000
        y   = rng.normal(50, 20, n)
        # Sabit, dar bant — y'nin büyük bölümü dışarıda
        q01 = np.full(n, 48.0)
        q09 = np.full(n, 52.0)
        off = compute_cqr_offset(y, q01, q09, alpha=0.20)
        assert off > 0.0

    def test_coverage_after_correction_near_target(self):
        """
        Offset uygulandıktan sonra calibration coverage ≈ 1-alpha (±0.02 tolerans).
        Conformalization garantisi: (1 - alpha) - 1/(n+1) ≤ coverage ≤ (1 - alpha) + 1/(n+1).
        """
        rng = np.random.default_rng(0)
        n   = 2000
        y   = rng.normal(0, 1, n)
        q01 = rng.normal(-0.5, 0.3, n)
        q09 = rng.normal( 0.5, 0.3, n)

        off = compute_cqr_offset(y, q01, q09, alpha=0.20)
        q01_adj = q01 - off
        q09_adj = q09 + off
        cov = float(np.mean((y >= q01_adj) & (y <= q09_adj)))
        assert abs(cov - 0.80) < 0.02, f"coverage={cov:.4f}, beklenen ~0.80"

    def test_wide_band_offset_nonpositive(self):
        """Bant y'yi tamamen kapsıyorsa offset ≤ 0 (bant daraltılabilir)."""
        rng  = np.random.default_rng(7)
        y    = rng.normal(0, 1, 500)
        q01  = y - 100.0
        q09  = y + 100.0
        off  = compute_cqr_offset(y, q01, q09, alpha=0.20)
        assert off <= 0.0

    def test_k_clamp_no_index_error(self):
        """n=1 kenar durumunda hata vermemeli."""
        off = compute_cqr_offset(
            np.array([5.0]),
            np.array([3.0]),
            np.array([7.0]),
            alpha=0.10,
        )
        assert np.isfinite(off)


class TestApplyCqrCorrection:
    def test_q05_unchanged(self):
        preds = {"q01": np.array([1.0, 2.0]), "q05": np.array([3.0, 4.0]), "q09": np.array([5.0, 6.0])}
        out   = apply_cqr_correction(preds, offset=0.5)
        np.testing.assert_array_almost_equal(out["q05"], preds["q05"])

    def test_q01_shifted_down(self):
        preds = {"q01": np.array([2.0]), "q05": np.array([3.0]), "q09": np.array([4.0])}
        out   = apply_cqr_correction(preds, offset=1.0)
        assert out["q01"][0] == pytest.approx(1.0)

    def test_q09_shifted_up(self):
        preds = {"q01": np.array([2.0]), "q05": np.array([3.0]), "q09": np.array([4.0])}
        out   = apply_cqr_correction(preds, offset=1.0)
        assert out["q09"][0] == pytest.approx(5.0)

    def test_negative_offset_narrows_band(self):
        """Negatif offset → bant daralır."""
        preds = {"q01": np.array([1.0]), "q05": np.array([3.0]), "q09": np.array([5.0])}
        out   = apply_cqr_correction(preds, offset=-0.5)
        assert out["q01"][0] == pytest.approx(1.5)
        assert out["q09"][0] == pytest.approx(4.5)

    def test_output_keys(self):
        preds = {"q01": np.ones(5), "q05": np.ones(5) * 2, "q09": np.ones(5) * 3}
        out   = apply_cqr_correction(preds, offset=0.0)
        assert set(out.keys()) == {"q01", "q05", "q09"}
