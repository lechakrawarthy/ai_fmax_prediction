"""End-to-end run: validate -> train/compare -> calibrate -> OOD -> benchmark -> report.

Usage:  python -m fmax.pipeline [--data PATH] [--out DIR] [--alpha 0.05]
"""
import argparse
import json
import zlib

import numpy as np
import pandas as pd

from . import config as C
from .data import load_raw, split, true_bin, validate
from .features import select_features
from .models import compare_models, data_fingerprint, regression_metrics, save_bundle, select_best
from .search import (
    SimulatedTester, adaptive_search, assign_bin, bin_rank, binary_search, linear_search, windowed_search,
)
from .uncertainty import OODDetector, SplitConformal, coverage_report

METHOD_SEED = {"linear": 1, "binary": 2, "windowed_supplied": 3, "windowed_ml": 4, "adaptive": 5}


def _rng(unit_id: str, method: str) -> np.random.Generator:
    return np.random.default_rng([C.RANDOM_STATE, zlib.crc32(unit_id.encode()), METHOD_SEED[method]])


def run_search_benchmark(test: pd.DataFrame, y_hat, lo, hi, trusted) -> pd.DataFrame:
    """Simulate every strategy on every test unit. Returns one row per (unit, method)."""
    rows = []
    for i, r in enumerate(test.itertuples(index=False)):
        f = r.fmax_true_mhz
        runs = {
            "linear": lambda t: linear_search(t),
            "binary": lambda t: binary_search(t),
            # Reference: the window shipped with the dataset (evaluation only, never a model input)
            "windowed_supplied": lambda t: windowed_search(t, r.window_low_mhz, r.window_high_mhz),
            "windowed_ml": lambda t: windowed_search(t, lo[i], hi[i]),
            "adaptive": lambda t: adaptive_search(t, lo[i], hi[i], bool(trusted[i])),
        }
        for method, fn in runs.items():
            res = fn(SimulatedTester(f, _rng(r.unit_id, method)))
            rows.append({
                "unit_id": r.unit_id, "method": method, "fmax_true_mhz": f,
                "measured_mhz": res.measured_mhz, "n_tests": res.n_tests, "time_ms": res.time_ms,
                "fallback": res.fallback, "edge": res.edge,
            })
    return pd.DataFrame(rows)


def summarize_benchmark(bench: pd.DataFrame, guard_band: float = C.GUARD_BAND_MHZ) -> pd.DataFrame:
    b = bench.copy()
    b["assigned_bin"] = [assign_bin(m, guard_band) for m in b["measured_mhz"]]
    b["true_bin"] = true_bin(b["fmax_true_mhz"])
    ra = b["assigned_bin"].map(bin_rank)
    rt = b["true_bin"].map(bin_rank)
    b["abs_err"] = (b["measured_mhz"] - b["fmax_true_mhz"]).abs()
    g = b.assign(correct=ra == rt, over=ra > rt, under=ra < rt).groupby("method", sort=False)
    s = pd.DataFrame({
        "mean_tests": g["n_tests"].mean(),
        "p95_tests": g["n_tests"].quantile(0.95),
        "max_tests": g["n_tests"].max(),
        "mean_time_ms": g["time_ms"].mean(),
        "mean_abs_error_mhz": g["abs_err"].mean(),
        "bin_accuracy": g["correct"].mean(),
        "over_bin_escape_rate": g["over"].mean(),
        "over_bin_escapes": g["over"].sum(),
        "under_bin_rate": g["under"].mean(),
        "edge_extension_rate": g["edge"].apply(lambda e: (e != "").mean()),
        "fallback_rate": g["fallback"].mean(),
    })
    s["saving_vs_binary_pct"] = 100 * (1 - s["mean_tests"] / s.loc["binary", "mean_tests"])
    s["saving_vs_linear_pct"] = 100 * (1 - s["mean_tests"] / s.loc["linear", "mean_tests"])
    return s.reset_index()


def guard_band_sweep(bench: pd.DataFrame, method: str = "adaptive", bands=(0, 4, 8, 12, 16)) -> pd.DataFrame:
    sub = bench[bench["method"] == method]
    out = []
    for gb in bands:
        s = summarize_benchmark(pd.concat([sub, bench[bench["method"].isin(["binary", "linear"])]]), gb)
        row = s[s["method"] == method].iloc[0]
        out.append({"guard_band_mhz": gb, "bin_accuracy": row["bin_accuracy"],
                    "over_bin_escape_rate": row["over_bin_escape_rate"], "under_bin_rate": row["under_bin_rate"]})
    return pd.DataFrame(out)


def prediction_only_binning(y_true, y_hat, sigma, ks=(0, 1, 2, 3)) -> pd.DataFrame:
    """Why bins are not assigned from predictions: escape/under-bin trade-off vs guard band k*sigma."""
    rt = pd.Series(true_bin(y_true)).map(bin_rank).to_numpy()
    out = []
    for k in ks:
        ra = pd.Series([assign_bin(v, k * sigma) for v in y_hat]).map(bin_rank).to_numpy()
        out.append({"k_sigma": k, "guard_band_mhz": k * sigma, "bin_accuracy": float(np.mean(ra == rt)),
                    "over_bin_escape_rate": float(np.mean(ra > rt)), "under_bin_rate": float(np.mean(ra < rt))})
    return pd.DataFrame(out)


def run(data_path=C.DATA_PATH, out_dir=C.ARTIFACT_DIR, alpha=C.CONFORMAL_ALPHA, verbose=True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Module 1/2 -------------------------------------------------------------
    raw = load_raw(data_path)
    report = validate(raw)
    log(f"[data] {report.n_rows} rows, splits={report.split_counts}, warnings={report.warnings or 'none'}")
    parts = split(raw)
    X = {k: select_features(v) for k, v in parts.items()}
    y = {k: v[C.TARGET].to_numpy() for k, v in parts.items()}
    validated = raw.copy()
    validated["true_bin_check"] = true_bin(validated[C.TARGET]).astype(str)
    bin_agree = float((validated["true_bin_check"] == validated["true_bin_mhz"].astype(str)).mean())
    validated.to_csv(out_dir / "validated_dataset.csv", index=False)

    # Module 3 ---------------------------------------------------------------
    table, fitted = compare_models(X["train"], y["train"], X["test"], y["test"])
    best_name = select_best(table)
    model = fitted[best_name]
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    log("[models]\n" + table[["model", "cv_rmse_mhz", "cv_r2", "test_rmse_mhz", "test_r2",
                              "latency_us_per_unit"]].round(3).to_string(index=False))
    log(f"[models] selected: {best_name}")

    # Module 4 ---------------------------------------------------------------
    conformal = {}
    for mode in ("absolute", "normalized"):
        cp = SplitConformal(model, alpha=alpha, mode=mode).fit_difficulty(X["train"], y["train"])
        cp.calibrate(X["calibration"], y["calibration"])
        _, lo_m, hi_m = cp.predict_interval(X["test"])
        conformal[mode] = (cp, coverage_report(y["test"], lo_m, hi_m))
        log(f"[conformal:{mode}] q={cp.q_:.2f} test {conformal[mode][1]}")
    cp = conformal["absolute"][0]
    y_hat, lo, hi = cp.predict_interval(X["test"])

    ood = OODDetector().fit(X["train"])
    ood_test = ood.check(X["test"])
    trusted = ~ood_test["is_ood"]
    # OOD detection performance on synthetic shifted units (test features + process excursion)
    rng = np.random.default_rng(C.RANDOM_STATE)
    shifted = X["test"].sample(500, random_state=C.RANDOM_STATE).copy()
    shifted[C.RO_COLS] += rng.normal(0, 1, (len(shifted), 1)) + np.sign(rng.normal(size=(len(shifted), 1))) * 2.5
    ood_detect_rate = float(ood.check(shifted)["is_ood"].mean())
    log(f"[ood] test fallback rate {np.mean(~trusted):.4f}; detection on shifted units {ood_detect_rate:.3f}")

    # Module 5 ---------------------------------------------------------------
    log("[search] simulating strategies on test split ...")
    bench = run_search_benchmark(parts["test"], y_hat, lo, hi, trusted)
    summary = summarize_benchmark(bench)
    gb_sweep = guard_band_sweep(bench)
    pred_only = prediction_only_binning(y["test"], y_hat, cp.cal_residual_std_)
    bench.to_csv(out_dir / "benchmark_runs.csv", index=False)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    gb_sweep.to_csv(out_dir / "guard_band_sweep.csv", index=False)
    pred_only.to_csv(out_dir / "prediction_only_binning.csv", index=False)
    log("[search]\n" + summary[["method", "mean_tests", "p95_tests", "saving_vs_binary_pct", "bin_accuracy",
                                "over_bin_escapes", "fallback_rate"]].round(4).to_string(index=False))

    # Per-device output (PRD §12) -------------------------------------------
    piv = bench.pivot(index="unit_id", columns="method", values="n_tests")
    ad = bench[bench["method"] == "adaptive"].set_index("unit_id")
    test = parts["test"]
    per_unit = pd.DataFrame({
        "unit_id": test["unit_id"],
        "fmax_pred_mhz": y_hat.round(1),
        "lower_bound_mhz": lo.round(1),
        "upper_bound_mhz": hi.round(1),
        "is_ood": ood_test["is_ood"],
        "mahalanobis": ood_test["mahalanobis"].round(3),
        "fmax_true_mhz": y["test"],
        "fmax_measured_mhz": ad.loc[test["unit_id"], "measured_mhz"].to_numpy(),
        "speed_bin": [assign_bin(m) for m in ad.loc[test["unit_id"], "measured_mhz"]],
        "true_bin": true_bin(y["test"]),
        "tests_adaptive": piv.loc[test["unit_id"], "adaptive"].to_numpy(),
        "tests_binary": piv.loc[test["unit_id"], "binary"].to_numpy(),
        "tests_linear": piv.loc[test["unit_id"], "linear"].to_numpy(),
        "edge_extension": ad.loc[test["unit_id"], "edge"].to_numpy(),
    })
    per_unit["prediction_error_mhz"] = (per_unit["fmax_pred_mhz"] - per_unit["fmax_true_mhz"]).round(1)
    per_unit["tests_saved_vs_binary"] = per_unit["tests_binary"] - per_unit["tests_adaptive"]
    per_unit.to_csv(out_dir / "predictions_test.csv", index=False)

    # Persist ----------------------------------------------------------------
    save_bundle(out_dir / "model_bundle.joblib", {
        "model": model, "conformal": cp, "ood": ood,
        "meta": {"model_name": best_name, "alpha": alpha, "conformal_q_mhz": cp.q_,
                 "residual_std_mhz": cp.cal_residual_std_, "data_fingerprint": data_fingerprint(raw)},
    })

    ad_row = summary.set_index("method").loc["adaptive"]
    best_row = table.set_index("model").loc[best_name]
    metrics = {
        "data_validation": report.to_dict() | {"true_bin_recomputed_agreement": bin_agree},
        "selected_model": best_name,
        "alpha": alpha,
        "regression_test": regression_metrics(y["test"], y_hat),
        "baseline_supplied_prediction_test": regression_metrics(y["test"], parts["test"]["fmax_pred_mhz"]),
        "conformal": {m: v[1] | {"q_mhz": v[0].q_} for m, v in conformal.items()},
        "ood": {"test_fallback_rate": float(np.mean(~trusted)), "shifted_unit_detection_rate": ood_detect_rate,
                "mahalanobis_threshold": ood.threshold_},
        "search": summary.set_index("method").to_dict(orient="index"),
        "acceptance": {
            "r2>=0.90": bool(best_row["test_r2"] >= 0.90),
            "coverage~95%": bool(abs(conformal["absolute"][1]["coverage"] - (1 - alpha)) <= 0.02),
            "iteration_reduction>=30%": bool(ad_row["saving_vs_binary_pct"] >= 30),
            "bin_accuracy>=95%": bool(ad_row["bin_accuracy"] >= 0.95),
            "zero_leakage": True,  # enforced by features.assert_no_leakage on every model input
            "fallback_executes": bool(ad_row["fallback_rate"] > 0) or ood_detect_rate > 0.9,
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    from .report import write_report
    write_report(out_dir, metrics, table, summary, gb_sweep, pred_only)
    log(f"[done] artifacts in {out_dir}")
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=C.DATA_PATH)
    p.add_argument("--out", default=C.ARTIFACT_DIR)
    p.add_argument("--alpha", type=float, default=C.CONFORMAL_ALPHA)
    a = p.parse_args()
    from pathlib import Path
    run(Path(a.data), Path(a.out), a.alpha)


if __name__ == "__main__":
    main()
