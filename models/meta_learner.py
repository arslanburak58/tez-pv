"""
STAGE-6: Ridge × 3 meta-öğrenici — quantile-özgü OOF seçimi + missingness flags.

Her meta-model yalnızca kendi quantile'ına ait 3 OOF tahmini + 3 flag = 6 özellik görür.
Bu sayede Ridge her quantile için farklı girdi → farklı çözüm üretir (q01 ≠ q05 ≠ q09).

Tez hipotezi: is_X_missing flag'leri meta-katmana eklenince CRPS istatistiksel anlamlı
              düşer (Diebold-Mariano testi, STAGE-10).

API:
    enrich_x_meta(X_meta, flags)          → pd.DataFrame (n_oof × 12)
    _q_cols(q)                            → list[str] — 6 quantile-özgü sütun
    train_meta_learner(q, X_meta_12, y)   → fitted Ridge
    train_all_meta_learners(X_meta_12, y) → dict[str, Ridge]
    predict_intervals(models, X_meta_12)  → dict[str, np.ndarray]
    coverage_score(y_true, y_lower, y_upper) → float  (hedef ~0.80)
    compare_baseline(models, X_meta_12, y)   → dict

Kurallar:
    - Ridge(alpha) — MSE kaybı, saniyeler içinde eğitim (LP solver'ın O(n²) yükü yok)
    - Her model kendi q sütunlarını seçer → farklı coef_ → farklı tahminler
    - LightGBM predict → DataFrame (sütun adları korunur)
    - Serileştirme joblib (pickle yasak)
"""

import logging
import random
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.linear_model import Ridge

from models.base_learners import ALGOS, META_COLS, QUANTILES, _col_name, pinball_loss

random.seed(42)
np.random.seed(42)

log = logging.getLogger(__name__)


# ── QuantileLinear ─────────────────────────────────────────────────────────────

class QuantileLinear:
    """
    L2-düzenlenmiş doğrusal kantil regresyon.

    Pinball loss + L2 penalty objective'i scipy L-BFGS-B ile minimize eder.
    sklearn API'sine uyumlu (fit/predict).

    Parameters
    ----------
    quantile : float
        Hedef kantil, (0, 1) aralığında.
    alpha : float, default=1.0
        L2 düzenleme katsayısı.
    max_iter : int, default=500
        L-BFGS-B maksimum iterasyon.
    tol : float, default=1e-6
        Yakınsama toleransı.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> X = rng.normal(0, 1, (n, 3))
    >>> y = X @ [1.0, -0.5, 0.3] + rng.normal(0, 0.5, n)
    >>> m05 = QuantileLinear(quantile=0.5, alpha=0.1).fit(X, y)
    >>> m09 = QuantileLinear(quantile=0.9, alpha=0.1).fit(X, y)
    >>> assert m09.intercept_ > m05.intercept_
    >>> assert not np.allclose(m05.coef_, m09.coef_)
    """

    def __init__(
        self,
        quantile: float,
        alpha: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray
        self.intercept_: float
        self.n_iter_: int
        self.converged_: bool

    def _pinball_l2_objective(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Pinball loss + L2 penalty. params = [intercept, *coef]."""
        intercept = params[0]
        coef = params[1:]
        pred = X @ coef + intercept
        residual = y - pred
        loss = np.where(
            residual >= 0,
            self.quantile * residual,
            (self.quantile - 1) * residual,
        )
        l2 = self.alpha * np.sum(coef ** 2)  # intercept'e L2 uygulanmaz
        return float(np.mean(loss) + l2 / len(y))

    def _pinball_l2_gradient(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Analytical gradient."""
        intercept = params[0]
        coef = params[1:]
        pred = X @ coef + intercept
        residual = y - pred
        grad_pred = np.where(residual >= 0, -self.quantile, -(self.quantile - 1))
        grad_intercept = np.mean(grad_pred)
        grad_coef = (X.T @ grad_pred) / len(y) + 2 * self.alpha * coef / len(y)
        return np.concatenate([[grad_intercept], grad_coef])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileLinear":
        """X: (n, p) ndarray, y: (n,) ndarray."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_features = X.shape[1]
        x0 = np.zeros(n_features + 1)

        result = scipy.optimize.minimize(
            fun=self._pinball_l2_objective,
            x0=x0,
            args=(X, y),
            jac=self._pinball_l2_gradient,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )

        if not result.success:
            warnings.warn(f"QuantileLinear yakınsamadı: {result.message}")

        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:]
        self.n_iter_ = result.nit
        self.converged_ = result.success
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


# ── QuantileLinearBounded ─────────────────────────────────────────────────────


class QuantileLinearBounded(QuantileLinear):
    """
    QuantileLinear with box constraints on flag coefficients.

    Son `n_flag_features` katsayı için L-BFGS-B bounds parametresi ile
    [-flag_bound, +flag_bound] kutu kısıtı uygulanır.
    Diğer katsayılar (base model tahminleri + intercept) serbesttir.

    Parameters
    ----------
    flag_bound : float, default=1.0
        Flag katsayıları için üst sınır: coef ∈ [-flag_bound, +flag_bound].
    n_flag_features : int, default=3
        Kısıtlı flag katsayısı sayısı (son n_flag_features kolon).
    """

    def __init__(
        self,
        quantile: float,
        alpha: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        flag_bound: float = 1.0,
        n_flag_features: int = 3,
    ) -> None:
        super().__init__(quantile, alpha, max_iter, tol)
        self.flag_bound = flag_bound
        self.n_flag_features = n_flag_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileLinearBounded":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_features = X.shape[1]
        n_free = n_features - self.n_flag_features
        x0 = np.zeros(n_features + 1)

        # params = [intercept, free_coef_1..n_free, flag_coef_1..n_flag]
        bounds: list[tuple] = [(None, None)] * (1 + n_free)
        bounds += [(-self.flag_bound, self.flag_bound)] * self.n_flag_features

        result = scipy.optimize.minimize(
            fun=self._pinball_l2_objective,
            x0=x0,
            args=(X, y),
            jac=self._pinball_l2_gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )

        if not result.success:
            warnings.warn(f"QuantileLinearBounded yakınsamadı: {result.message}")

        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:]
        self.n_iter_ = result.nit
        self.converged_ = result.success
        return self


# ── Constants ──────────────────────────────────────────────────────────────────

FLAG_COLS: list[str] = [
    "is_G_missing",
    "is_Tamb_missing",
    "is_RH_missing",
]

META_IN_COLS: list[str] = META_COLS + FLAG_COLS  # 12 özellik (9 OOF + 3 flag)

DEFAULT_ALPHA: float = 1.0


def _meta_key(q: float) -> str:
    return f"meta_q{int(round(q * 10)):02d}"  # meta_q01 | meta_q05 | meta_q09


def _key_to_q(key: str) -> float:
    """meta_q01 → 0.1 | meta_q05 → 0.5 | meta_q09 → 0.9"""
    return int(key[-2:]) / 10.0


def _q_cols(q: float) -> list[str]:
    """Quantile'a ait 3 OOF sütunu + 3 flag = 6 özellik."""
    oof = [_col_name(algo, q) for algo in ALGOS]  # lgbm_q01, catboost_q01, xgboost_q01
    return oof + FLAG_COLS


# ── enrich_x_meta ─────────────────────────────────────────────────────────────

def enrich_x_meta(
    X_meta: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    """
    9 OOF sütununa 3 missingness flag ekle → 12 özellikli X_meta.

    Args:
        X_meta: (n_oof × 9) — build_x_meta() çıktısı; orijinal index korunmuş
        flags:  (n × 3)    — missingness flags; index X_meta ile hizalanabilir

    Returns:
        pd.DataFrame (n_oof × 12) — sütunlar META_IN_COLS sırasında
    """
    missing_oof = set(META_COLS) - set(X_meta.columns)
    if missing_oof:
        raise ValueError(f"X_meta'da eksik OOF sütunları: {missing_oof}")
    missing_flags = set(FLAG_COLS) - set(flags.columns)
    if missing_flags:
        raise ValueError(f"flags'da eksik sütunlar: {missing_flags}")

    flags_aligned = flags.reindex(X_meta.index)[FLAG_COLS].fillna(0).astype(int)
    result = pd.concat([X_meta[META_COLS], flags_aligned], axis=1)
    result.columns = META_IN_COLS
    return result


# ── train_meta_learner ─────────────────────────────────────────────────────────

def train_meta_learner(
    q: float,
    X_meta_12: pd.DataFrame,
    y: pd.Series | np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> QuantileLinear:
    """
    Tek bir quantile için QuantileLinear meta-öğrenici eğit.

    Her model yalnızca kendi quantile'ına ait 3 OOF sütunu + 3 flag = 6 özellik görür.
    QuantileLinear(quantile=q) pinball loss minimize eder → q01 ≠ q05 ≠ q09 garantisi.
    """
    if q not in QUANTILES:
        raise ValueError(f"q={q} geçersiz. Seçenekler: {QUANTILES}")

    cols = _q_cols(q)
    X_q = X_meta_12[cols].to_numpy()
    model = QuantileLinear(quantile=q, alpha=alpha)
    model.fit(X_q, np.asarray(y))
    log.info("Meta-öğrenici eğitildi | q=%.1f | alpha=%.4f | cols=%s", q, alpha, cols)
    return model


# ── train_all_meta_learners ────────────────────────────────────────────────────

def train_all_meta_learners(
    X_meta_12: pd.DataFrame,
    y: pd.Series | np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    checkpoint_dir: str | None = None,
) -> dict[str, QuantileLinear]:
    """
    3 QuantileLinear meta-öğrenicisini eğit (q=0.1 / 0.5 / 0.9).

    Returns:
        dict — anahtar: "meta_q01" | "meta_q05" | "meta_q09"
    """
    models: dict[str, QuantileLinear] = {}
    for q in QUANTILES:
        key = _meta_key(q)
        model = train_meta_learner(q, X_meta_12, y, alpha=alpha)
        models[key] = model
        if checkpoint_dir:
            path = f"{checkpoint_dir}/{key}.joblib"
            joblib.dump(model, path)
            log.info("Kaydedildi: %s", path)
    return models


# ── predict_intervals ──────────────────────────────────────────────────────────

def predict_intervals(
    models: dict[str, QuantileLinear],
    X_meta_12: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    3 quantile tahmini üret; crossing önlemek için post-hoc sort uygular.

    Her model kendi q sütunlarını X_meta_12'den seçer; ardından
    np.sort(axis=0) ile q01 ≤ q05 ≤ q09 monotonluğu garantilenir.

    Returns:
        dict — anahtar: "meta_q01" | "meta_q05" | "meta_q09"
    """
    raw: dict[str, np.ndarray] = {}
    for key, model in models.items():
        q = _key_to_q(key)
        cols = _q_cols(q)
        raw[key] = model.predict(X_meta_12[cols].to_numpy())

    preds_stack = np.stack([raw["meta_q01"], raw["meta_q05"], raw["meta_q09"]], axis=0)
    preds_sorted = np.sort(preds_stack, axis=0)
    return {
        "meta_q01": preds_sorted[0],
        "meta_q05": preds_sorted[1],
        "meta_q09": preds_sorted[2],
    }


# ── coverage_score ─────────────────────────────────────────────────────────────

def coverage_score(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    %10–%90 bant kapsama oranı.

    Nominal kapsama = 0.80 (q=0.9 bandı - q=0.1 bandı).
    İyi kalibre model: 0.75 ≤ coverage ≤ 0.85.

    Returns:
        float — [0, 1]
    """
    y = np.asarray(y_true)
    return float(np.mean((y >= y_lower) & (y <= y_upper)))


# ── compare_baseline ───────────────────────────────────────────────────────────

def compare_baseline(
    meta_models: dict[str, QuantileLinear],
    X_meta_12: pd.DataFrame,
    y_true: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """
    Stacked meta-model vs tek LightGBM-quantile baseline karşılaştır.

    Baseline: X_meta_12 içindeki lgbm_q01 / lgbm_q05 / lgbm_q09 OOF tahminleri.
    Tamamlandı ölçütü: stacked modelin her quantile'da ≥ %5 pinball iyileşmesi.

    Returns:
        dict:
            stacked:     {meta_key: pinball_score}
            baseline:    {meta_key: pinball_score}
            improvement: {meta_key: % iyileşme (pozitif = iyi)}
    """
    y = np.asarray(y_true)

    q_map: dict[str, tuple[float, str]] = {
        _meta_key(0.1): (0.1, "lgbm_q01"),
        _meta_key(0.5): (0.5, "lgbm_q05"),
        _meta_key(0.9): (0.9, "lgbm_q09"),
    }

    stacked_scores: dict[str, float] = {}
    baseline_scores: dict[str, float] = {}

    for meta_key, (q, lgbm_col) in q_map.items():
        cols = _q_cols(q)
        stacked_pred = meta_models[meta_key].predict(X_meta_12[cols].to_numpy())
        stacked_scores[meta_key] = pinball_loss(y, stacked_pred, q)

        baseline_pred = X_meta_12[lgbm_col].to_numpy()
        baseline_scores[meta_key] = pinball_loss(y, baseline_pred, q)

    improvement: dict[str, float] = {}
    for key in stacked_scores:
        base = baseline_scores[key]
        imp = (base - stacked_scores[key]) / base * 100 if base > 0 else 0.0
        improvement[key] = imp
        log.info(
            "%s | stacked=%.4f | baseline=%.4f | iyileşme=%.1f%%",
            key, stacked_scores[key], base, imp,
        )

    return {
        "stacked": stacked_scores,
        "baseline": baseline_scores,
        "improvement": improvement,
    }
