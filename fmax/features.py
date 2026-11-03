"""Feature allowlist and engineered features.

Only columns in config.FEATURES can reach a model. Engineered features are derived
from those columns alone, so they cannot carry target information.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C


def assert_no_leakage(columns) -> None:
    leaked = sorted(set(columns) & set(C.LEAKAGE_COLS))
    if leaked:
        raise ValueError(f"Leakage columns present in model input: {leaked}")
    unknown = sorted(set(columns) - set(C.FEATURES))
    if unknown:
        raise ValueError(f"Columns outside the feature allowlist: {unknown}")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[C.FEATURES].copy()
    assert_no_leakage(X.columns)
    return X


class RingOscAggregates(BaseEstimator, TransformerMixin):
    """Append mean / std / min / max across the 10 ring-oscillator readings.

    Averaging the noisy RO counts gives a lower-noise estimate of the latent
    process speed that tree models otherwise have to assemble from splits.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=C.FEATURES) if not isinstance(X, pd.DataFrame) else X
        ro = X[C.RO_COLS].to_numpy()
        extra = np.column_stack([ro.mean(1), ro.std(1), ro.min(1), ro.max(1)])
        return np.column_stack([X.to_numpy(), extra])

    def get_feature_names_out(self, input_features=None):
        return np.array(C.FEATURES + ["ro_mean", "ro_std", "ro_min", "ro_max"])
