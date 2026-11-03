"""Module 3: candidate regressors, cross-validated comparison, selection, persistence."""
import hashlib
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from . import config as C
from .features import RingOscAggregates

try:
    from xgboost import XGBRegressor
except ImportError:  # optional per PRD
    XGBRegressor = None


def candidate_models() -> dict[str, Pipeline]:
    rs = C.RANDOM_STATE
    models = {
        "linear_regression": Pipeline([
            ("ro_agg", RingOscAggregates()),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("ro_agg", RingOscAggregates()),
            ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=rs)),
        ]),
        "gradient_boosting": Pipeline([
            ("ro_agg", RingOscAggregates()),
            ("model", GradientBoostingRegressor(
                n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=rs)),
        ]),
    }
    if XGBRegressor is not None:
        models["xgboost"] = Pipeline([
            ("ro_agg", RingOscAggregates()),
            ("model", XGBRegressor(
                n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8,
                colsample_bytree=0.8, random_state=rs, n_jobs=-1)),
        ])
    return models


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae_mhz": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mhz": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_pct": float(100 * mean_absolute_percentage_error(y_true, y_pred)),
    }


def compare_models(X_train, y_train, X_test, y_test, cv_folds: int = 5) -> tuple[pd.DataFrame, dict]:
    """5-fold CV on training data drives selection; test metrics are reported only.

    Returns the comparison table and the fitted models.
    """
    kf = KFold(cv_folds, shuffle=True, random_state=C.RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in candidate_models().items():
        oof = cross_val_predict(model, X_train, y_train, cv=kf)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = model.predict(X_test)
        latency_us = 1e6 * (time.perf_counter() - t0) / len(X_test)
        cv = {f"cv_{k}": v for k, v in regression_metrics(y_train, oof).items()}
        test = {f"test_{k}": v for k, v in regression_metrics(y_test, pred).items()}
        rows.append({"model": name, **cv, **test, "fit_s": fit_s, "latency_us_per_unit": latency_us})
        fitted[name] = model
    table = pd.DataFrame(rows).sort_values("cv_rmse_mhz").reset_index(drop=True)
    return table, fitted


def select_best(table: pd.DataFrame, tolerance_mhz: float = 0.25) -> str:
    """Lowest CV RMSE; among models within `tolerance_mhz` of it, prefer the fastest at inference."""
    best = table["cv_rmse_mhz"].min()
    close = table[table["cv_rmse_mhz"] <= best + tolerance_mhz]
    return str(close.sort_values("latency_us_per_unit").iloc[0]["model"])


def data_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def save_bundle(path, bundle: dict) -> None:
    """Persist model + calibrator + OOD detector with version metadata."""
    bundle = {
        **bundle,
        "meta": {
            **bundle.get("meta", {}),
            "package_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "features": C.FEATURES,
        },
    }
    joblib.dump(bundle, path)
    meta_path = str(path).rsplit(".", 1)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(bundle["meta"], f, indent=2, default=str)


def load_bundle(path) -> dict:
    return joblib.load(path)
