"""Module 4: split-conformal prediction intervals and out-of-distribution detection."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from . import config as C


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of nonconformity scores."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


class SplitConformal:
    """Split conformal regression intervals, calibrated on a held-out set.

    mode="absolute":   score = |y - y_hat|; one interval width for every unit.
    mode="normalized": score = |y - y_hat| / s(x), where s(x) is a difficulty model
                       fitted on out-of-fold *training* residuals, so widths vary per unit.
    The calibration set is never used for model fitting in either mode.
    """

    def __init__(self, model, alpha: float = C.CONFORMAL_ALPHA, mode: str = "absolute"):
        if mode not in {"absolute", "normalized"}:
            raise ValueError(mode)
        self.model, self.alpha, self.mode = model, alpha, mode
        self.difficulty_ = None

    def fit_difficulty(self, X_train, y_train):
        if self.mode == "normalized":
            kf = KFold(5, shuffle=True, random_state=C.RANDOM_STATE)
            oof = cross_val_predict(clone(self.model), X_train, y_train, cv=kf)
            self.difficulty_ = GradientBoostingRegressor(
                n_estimators=200, max_depth=2, learning_rate=0.05, random_state=C.RANDOM_STATE
            ).fit(X_train, np.abs(y_train - oof))
        return self

    def _scale(self, X):
        if self.mode == "absolute":
            return np.ones(len(X))
        return np.maximum(self.difficulty_.predict(X), 1.0)

    def calibrate(self, X_cal, y_cal):
        resid = np.abs(np.asarray(y_cal) - self.model.predict(X_cal))
        self.q_ = conformal_quantile(resid / self._scale(X_cal), self.alpha)
        self.cal_residual_std_ = float(np.std(np.asarray(y_cal) - self.model.predict(X_cal)))
        return self

    def predict_interval(self, X):
        y_hat = self.model.predict(X)
        half = self.q_ * self._scale(X)
        return y_hat, y_hat - half, y_hat + half


def coverage_report(y, lo, hi) -> dict:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width_mhz": float(np.mean(hi - lo)),
        "median_width_mhz": float(np.median(hi - lo)),
    }


class OODDetector:
    """Flags units whose features lie outside the training distribution.

    Two checks (technical document §8.2): per-feature range with a margin, and
    Mahalanobis distance against an empirical training quantile.
    """

    def __init__(self, quantile: float = C.OOD_QUANTILE, range_margin: float = C.OOD_RANGE_MARGIN):
        self.quantile, self.range_margin = quantile, range_margin

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(0)
        self.prec_ = np.linalg.pinv(np.cov(X, rowvar=False))
        span = X.max(0) - X.min(0)
        self.lo_ = X.min(0) - self.range_margin * span
        self.hi_ = X.max(0) + self.range_margin * span
        self.threshold_ = float(np.quantile(self.distance(X), self.quantile))
        return self

    def distance(self, X):
        d = np.asarray(X, dtype=float) - self.mean_
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.prec_, d))

    def check(self, X) -> dict:
        X = np.asarray(X, dtype=float)
        dist = self.distance(X)
        out_of_range = ((X < self.lo_) | (X > self.hi_)).any(1)
        far = dist > self.threshold_
        bad_values = ~np.isfinite(X).all(1)
        return {
            "is_ood": out_of_range | far | bad_values,
            "out_of_range": out_of_range,
            "mahalanobis": dist,
            "mahalanobis_flag": far,
        }
