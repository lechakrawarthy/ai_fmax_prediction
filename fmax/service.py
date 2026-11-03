"""Inference service shared by the API and the dashboard: prediction, interval, OOD, search plan."""
import numpy as np
import pandas as pd

from . import config as C
from .features import select_features
from .models import load_bundle
from .search import SimulatedTester, adaptive_search, assign_bin


class FmaxService:
    def __init__(self, bundle_path=C.ARTIFACT_DIR / "model_bundle.joblib"):
        b = load_bundle(bundle_path)
        self.model, self.conformal, self.ood, self.meta = b["model"], b["conformal"], b["ood"], b["meta"]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = select_features(df)
        ood = self.ood.check(X)
        out = pd.DataFrame(index=df.index)
        out["is_ood"] = ood["is_ood"]
        out["mahalanobis"] = ood["mahalanobis"]
        out[["fmax_pred_mhz", "lower_bound_mhz", "upper_bound_mhz"]] = np.nan
        ok = ~ood["is_ood"]
        if ok.any():
            y_hat, lo, hi = self.conformal.predict_interval(X[ok])
            out.loc[ok, "fmax_pred_mhz"] = y_hat
            out.loc[ok, "lower_bound_mhz"] = lo
            out.loc[ok, "upper_bound_mhz"] = hi
        out["action"] = np.where(ok, "windowed_search", "fallback_full_search")
        return out

    def simulate(self, features: dict, fmax_true_mhz: float, seed: int = 0) -> dict:
        """Run the adaptive flow for one unit against a simulated tester with a known true Fmax."""
        p = self.predict(pd.DataFrame([features])).iloc[0]
        t = SimulatedTester(fmax_true_mhz, np.random.default_rng(seed))
        r = adaptive_search(t, p["lower_bound_mhz"], p["upper_bound_mhz"], use_prediction=not p["is_ood"])
        return {
            **{k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in p.items()},
            "measured_mhz": r.measured_mhz,
            "speed_bin": assign_bin(r.measured_mhz),
            "n_tests": r.n_tests,
            "time_ms": r.time_ms,
            "fallback": r.fallback,
            "edge_extension": r.edge,
            "trajectory": [{"freq_mhz": f, "pass": ok} for f, ok in t.log],
        }
