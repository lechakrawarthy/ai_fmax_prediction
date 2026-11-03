"""Streamlit dashboard.  Run from the repo root: streamlit run dashboard/app.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fmax import config as C  # noqa: E402
from fmax.data import load_raw  # noqa: E402
from fmax.pipeline import _rng  # noqa: E402
from fmax.report import METHOD_LABELS  # noqa: E402
from fmax.search import SimulatedTester, adaptive_search, binary_search, linear_search  # noqa: E402
from fmax.service import FmaxService  # noqa: E402

st.set_page_config(page_title="Fmax Test Optimization", page_icon="⚡", layout="wide")

A = C.ARTIFACT_DIR
PASS_COLOR, FAIL_COLOR, ACCENT = "#2e9e5b", "#d1495b", "#3b6fd8"
METHOD_COLORS = {
    "Linear sweep": "#9aa5b1",
    "Binary search": "#e8a33d",
    "Windowed (supplied window)": "#8e7cc3",
    "ML-windowed binary": "#5aa9e6",
    "ML-windowed + OOD fallback": ACCENT,
}


# --------------------------------------------------------------------------- loading
def artifacts_ready() -> bool:
    return all((A / f).exists() for f in ("metrics.json", "model_bundle.joblib", "predictions_test.csv"))


@st.cache_data
def load_artifacts():
    pred = pd.read_csv(A / "predictions_test.csv")
    pred["edge_extension"] = pred["edge_extension"].fillna("")
    pred["speed_bin"] = pred["speed_bin"].astype(str)
    pred["true_bin"] = pred["true_bin"].astype(str)
    summary = pd.read_csv(A / "benchmark_summary.csv")
    summary["label"] = summary["method"].map(METHOD_LABELS)
    return {
        "metrics": json.loads((A / "metrics.json").read_text()),
        "models": pd.read_csv(A / "model_comparison.csv"),
        "summary": summary,
        "pred": pred,
        "gb_sweep": pd.read_csv(A / "guard_band_sweep.csv"),
        "pred_only": pd.read_csv(A / "prediction_only_binning.csv"),
        "report_md": (A / "report.md").read_text(encoding="utf-8") if (A / "report.md").exists() else "",
    }


@st.cache_resource
def load_service():
    return FmaxService()


@st.cache_data
def load_data():
    return load_raw()


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("⚡ Fmax Optimizer")
PAGES = ["Overview", "Model performance", "Uncertainty", "Search optimization", "Binning & guard bands",
         "Device explorer", "Predict new devices", "Data & EDA", "Report & export"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("All results come from the supplied **synthetic** dataset. They show feasibility only, "
                   "not validated production savings.")

if st.sidebar.button("Re-run pipeline", help="Retrain, recalibrate and re-simulate (about a minute)"):
    from fmax.pipeline import run

    with st.spinner("Running pipeline…"):
        run(verbose=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if not artifacts_ready():
    st.title("Fmax Prediction & Adaptive Test Optimization")
    st.warning("No trained model yet. Click **Re-run pipeline** in the sidebar, or run "
               "`python -m fmax.pipeline` in a terminal.")
    st.stop()

R = load_artifacts()
M, P, S = R["metrics"], R["pred"], R["summary"]
ADP = M["search"]["adaptive"]
BIN = M["search"]["binary"]
REG = M["regression_test"]
CONF = M["conformal"]["absolute"]
st.sidebar.caption(f"Model: `{M['selected_model']}` · α = {M['alpha']}")


# --------------------------------------------------------------------------- shared charts
def tests_bar(log_x=False):
    s = S.sort_values("mean_tests")
    fig = go.Figure(go.Bar(
        x=s["mean_tests"], y=s["label"], orientation="h",
        marker_color=[METHOD_COLORS[l] for l in s["label"]],
        text=s["mean_tests"].map(lambda v: f"{v:.2f}"), textposition="outside",
        hovertemplate="%{y}: %{x:.2f} tests<extra></extra>", name="Mean"))
    fig.add_trace(go.Scatter(x=s["p95_tests"], y=s["label"], mode="markers", name="p95",
                             marker=dict(symbol="line-ns-open", size=18, color="black", line_width=2)))
    fig.update_layout(height=340, margin=dict(t=10, b=10, l=10), xaxis_title="Tests per device",
                      xaxis_type="log" if log_x else "linear", legend=dict(orientation="h", y=-0.2))
    return fig


def pred_vs_true(df):
    df = df.assign(Route=np.where(df["is_ood"], "OOD → fallback", "Windowed"))
    fig = px.scatter(df, x="fmax_true_mhz", y="fmax_pred_mhz", color="Route", opacity=0.55,
                     color_discrete_map={"Windowed": ACCENT, "OOD → fallback": FAIL_COLOR},
                     labels={"fmax_true_mhz": "True Fmax (MHz)", "fmax_pred_mhz": "Predicted Fmax (MHz)"},
                     hover_data=["unit_id", "prediction_error_mhz"])
    lo, hi = df["fmax_true_mhz"].min(), df["fmax_true_mhz"].max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                             line=dict(dash="dash", color="gray")))
    fig.update_traces(marker_size=5, selector=dict(mode="markers"))
    fig.update_layout(height=380, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.08))
    return fig


def trajectory(log, title, window=None, true_f=None):
    tr = pd.DataFrame(log, columns=["freq", "passed"])
    tr["step"] = np.arange(1, len(tr) + 1)
    fig = go.Figure()
    if window is not None:
        fig.add_hrect(y0=window[0], y1=window[1], fillcolor="rgba(59,111,216,0.15)", line_width=0,
                      annotation_text="model window", annotation_position="top left")
    if true_f is not None:
        fig.add_hline(y=true_f, line_dash="dot", line_color="gray", annotation_text="true Fmax")
    fig.add_trace(go.Scatter(
        x=tr["step"], y=tr["freq"], mode="lines+markers", line_color="lightgray",
        marker=dict(size=11, color=np.where(tr["passed"], PASS_COLOR, FAIL_COLOR)),
        text=np.where(tr["passed"], "PASS", "FAIL"),
        hovertemplate="Test %{x}: %{y:.0f} MHz %{text}<extra></extra>"))
    fig.update_layout(title=f"{title}: {len(tr)} tests", height=360, showlegend=False,
                      xaxis_title="Test #", yaxis_title="Frequency (MHz)", margin=dict(t=40, b=10))
    if len(tr) <= 12:
        fig.update_xaxes(dtick=1)
    return fig


# --------------------------------------------------------------------------- pages
def page_overview():
    st.title("Fmax Prediction & Adaptive Test Optimization")
    st.markdown(
        "The model predicts each device's Fmax from measurements taken before the frequency search "
        "(ring oscillators, IDDQ, Vmin, die position), wraps the prediction in a calibrated 95% interval, and "
        "the tester then binary-searches **only inside that window**. If the boundary lies outside the "
        "window, the search extends; if the device looks out-of-distribution, it gets the full search. Bins "
        "always come from a **measured** Fmax with the 8 MHz guard band, never from the prediction alone."
    )
    c = st.columns(6)
    c[0].metric("Test R²", f"{REG['r2']:.3f}")
    c[1].metric("RMSE", f"{REG['rmse_mhz']:.1f} MHz", f"MAPE {REG['mape_pct']:.2f}%", delta_color="off")
    c[2].metric("Interval coverage", pct(CONF["coverage"], 2), "target 95%", delta_color="off")
    c[3].metric("Tests saved vs binary", f"{ADP['saving_vs_binary_pct']:.1f}%",
                f"{BIN['mean_tests']:.2f} → {ADP['mean_tests']:.2f} per device", delta_color="off")
    c[4].metric("Over-bin escapes", int(ADP["over_bin_escapes"]), f"of {len(P):,} devices", delta_color="off")
    c[5].metric("Fallback rate", pct(ADP["fallback_rate"], 2))

    st.subheader("PRD acceptance criteria (held-out test split)")
    acc = M["acceptance"]
    rows = [
        ("Regression R² ≥ 0.90", f"{REG['r2']:.4f}", acc["r2>=0.90"], ""),
        ("Prediction-interval coverage ≈ 95%", pct(CONF["coverage"], 2), acc["coverage~95%"], ""),
        ("Test iterations ≥ 30% below binary", f"{ADP['saving_vs_binary_pct']:.1f}%", acc["iteration_reduction>=30%"], ""),
        ("Speed-bin accuracy ≥ 95%", pct(ADP["bin_accuracy"], 2), acc["bin_accuracy>=95%"],
         "Limited by the 8 MHz measurement guard band and tester noise, not the model. Plain binary search scores "
         f"{pct(BIN['bin_accuracy'], 2)}. See Binning & guard bands."),
        ("Zero known data leakage", "Feature allowlist enforced", acc["zero_leakage"], ""),
        ("Fallback mechanism tested", f"{pct(M['ood']['shifted_unit_detection_rate'])} of shifted devices routed to full search",
         acc["fallback_executes"], ""),
    ]
    st.dataframe(pd.DataFrame([{"Criterion": r[0], "Result": r[1], "Status": "✅ Met" if r[2] else "⚠️ Not met",
                                "Note": r[3]} for r in rows]), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Average tests per device")
        st.plotly_chart(tests_bar(), width="stretch")
    with right:
        st.subheader("Predicted vs true Fmax")
        st.plotly_chart(pred_vs_true(P.sample(min(1500, len(P)), random_state=0)), width="stretch")


def page_models():
    st.title("Model performance")
    st.caption("Selection uses 5-fold CV on the 1,500 training devices only. The rule is the lowest CV RMSE, "
               "preferring the fastest model within 0.25 MHz of it. Test metrics are reported, never used to choose.")
    t = R["models"].copy()
    t.insert(0, "selected", np.where(t["model"] == M["selected_model"], "✅", ""))
    st.dataframe(t.round(3), hide_index=True, width="stretch")
    base = M["baseline_supplied_prediction_test"]
    st.info(f"Reference: the `fmax_pred_mhz` column shipped with the dataset scores R² {base['r2']:.4f}, "
            f"RMSE {base['rmse_mhz']:.1f} MHz on the same test devices (evaluation only, never a model input).")

    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox("Compare on", ["test_rmse_mhz", "test_mae_mhz", "test_r2", "cv_rmse_mhz",
                                             "latency_us_per_unit", "fit_s"])
        fig = px.bar(t, x="model", y=metric, color="model", text_auto=".3f")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(P, x="prediction_error_mhz", nbins=70,
                           labels={"prediction_error_mhz": "Predicted − true (MHz)"})
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(height=440, title="Prediction error distribution")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(P, x="fmax_true_mhz", y="prediction_error_mhz", opacity=0.35,
                         labels={"fmax_true_mhz": "True Fmax (MHz)", "prediction_error_mhz": "Error (MHz)"})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=380, title="Residuals vs true Fmax")
        st.plotly_chart(fig, width="stretch")
    with c4:
        model = load_service().model
        lin = model.named_steps.get("model")
        if hasattr(lin, "coef_"):
            names = model.named_steps["ro_agg"].get_feature_names_out()
            coef = pd.Series(lin.coef_, index=names).sort_values(key=np.abs)
            fig = px.bar(x=coef.values, y=coef.index, orientation="h",
                         labels={"x": "Standardized coefficient (MHz per σ)", "y": ""})
            fig.update_traces(marker_color=np.where(coef.values > 0, ACCENT, FAIL_COLOR))
            fig.update_layout(height=380, title="Feature influence (linear model)")
        else:
            imp = pd.Series(lin.feature_importances_, index=model.named_steps["ro_agg"].get_feature_names_out())
            imp = imp.sort_values()
            fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": ""})
            fig.update_layout(height=380, title="Feature importance")
        st.plotly_chart(fig, width="stretch")


def page_uncertainty():
    st.title("Calibrated uncertainty")
    st.markdown(
        "Split-conformal intervals are calibrated on the 500 calibration devices, which are never used for "
        "training. The deployed method is **absolute** (one width for every device). The normalized variant "
        "scales the width per device but is wider on average here, so it is reported for comparison only."
    )
    cmp = pd.DataFrame(M["conformal"]).T.reset_index(names="method")
    cmp["coverage"] = cmp["coverage"].map(lambda v: pct(v, 2))
    st.dataframe(cmp.round(2), hide_index=True, width="stretch")

    n = st.slider("Devices shown (random sample, sorted by true Fmax)", 50, 400, 150, step=50)
    s = P[~P["is_ood"]].sample(n, random_state=1).sort_values("fmax_true_mhz").reset_index(drop=True)
    inside = (s["fmax_true_mhz"] >= s["lower_bound_mhz"]) & (s["fmax_true_mhz"] <= s["upper_bound_mhz"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s["upper_bound_mhz"], line_width=0, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s.index, y=s["lower_bound_mhz"], fill="tonexty", line_width=0,
                             fillcolor="rgba(59,111,216,0.2)", name="95% interval"))
    fig.add_trace(go.Scatter(x=s.index, y=s["fmax_pred_mhz"], mode="lines", line_color=ACCENT, name="Predicted"))
    fig.add_trace(go.Scatter(x=s.index, y=s["fmax_true_mhz"], mode="markers", name="True",
                             marker=dict(size=5, color=np.where(inside, PASS_COLOR, FAIL_COLOR))))
    fig.update_layout(height=440, xaxis_title="Device (sorted)", yaxis_title="MHz",
                      title=f"Coverage in this sample: {inside.mean() * 100:.1f}% (red = outside interval)")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Out-of-distribution fallback")
        o = M["ood"]
        st.metric("Test devices routed to full search", pct(o["test_fallback_rate"], 2))
        st.metric("Detection rate on synthetically shifted devices", pct(o["shifted_unit_detection_rate"]))
        st.caption(f"Range check plus Mahalanobis distance (threshold {o['mahalanobis_threshold']:.2f}, the 99.9th "
                   "percentile of training devices).")
    with c2:
        fig = px.histogram(P, x="mahalanobis", color="is_ood", nbins=60,
                           color_discrete_map={False: ACCENT, True: FAIL_COLOR},
                           labels={"mahalanobis": "Mahalanobis distance", "is_ood": "OOD"})
        fig.add_vline(x=M["ood"]["mahalanobis_threshold"], line_dash="dash", annotation_text="threshold")
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")


def page_search():
    st.title("Search optimization")
    st.markdown(
        f"Simulated tester: {C.GRID_MIN_MHZ:.0f}–{C.GRID_MAX_MHZ:.0f} MHz grid in {C.GRID_STEP_MHZ:.0f} MHz steps, "
        f"σ_test = {C.SIGMA_TEST_MHZ:.0f} MHz noise on every pass/fail, {C.MS_PER_TEST:.0f} ms per test "
        f"({C.PATTERN_MS:.0f} ms pattern + {C.CLOCK_SWITCH_MS:.0f} ms clock switch). Every strategy runs on "
        f"the same {len(P):,} test devices."
    )
    c = st.columns(4)
    c[0].metric("Binary search", f"{BIN['mean_tests']:.2f} tests", f"{BIN['mean_time_ms']:.0f} ms", delta_color="off")
    c[1].metric("AI-assisted", f"{ADP['mean_tests']:.2f} tests", f"{ADP['mean_time_ms']:.0f} ms", delta_color="off")
    c[2].metric("Saving vs binary", f"{ADP['saving_vs_binary_pct']:.1f}%")
    c[3].metric("Saving vs linear", f"{ADP['saving_vs_linear_pct']:.1f}%")

    st.plotly_chart(tests_bar(log_x=st.toggle("Log scale", value=False)), width="stretch")
    cols = ["label", "mean_tests", "p95_tests", "max_tests", "mean_time_ms", "saving_vs_binary_pct",
            "mean_abs_error_mhz", "bin_accuracy", "over_bin_escapes", "under_bin_rate", "edge_extension_rate",
            "fallback_rate"]
    st.dataframe(S[cols].rename(columns={"label": "strategy"}).round(4), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        long = P.melt(value_vars=["tests_binary", "tests_adaptive"], var_name="Strategy", value_name="Tests")
        long["Strategy"] = long["Strategy"].map({"tests_binary": "Binary search",
                                                 "tests_adaptive": "ML-windowed + OOD fallback"})
        fig = px.histogram(long, x="Tests", color="Strategy", barmode="group", color_discrete_map=METHOD_COLORS)
        fig.update_layout(height=380, title="Tests per device", bargap=0.15, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with c2:
        route = np.where(P["is_ood"], "OOD → full binary search",
                 np.where(P["edge_extension"] == "upper", "Window extended upward",
                 np.where(P["edge_extension"] == "lower", "Searched below window", "Resolved inside window")))
        counts = pd.Series(route).value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.5, color=counts.index,
                     color_discrete_map={"Resolved inside window": ACCENT, "Window extended upward": "#e8a33d",
                                         "Searched below window": "#8e7cc3", "OOD → full binary search": FAIL_COLOR})
        fig.update_layout(height=380, title="How each device was tested")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Tester-time estimate")
    c = st.columns(4)
    units = c[0].number_input("Devices per year", value=20_000_000, step=1_000_000)
    corners = c[1].number_input("V/T corners", value=3, min_value=1)
    step_s = c[2].number_input("Seconds per test step", value=C.MS_PER_TEST / 1000, format="%.3f")
    cost = c[3].number_input("Tester cost ($ per second)", value=0.004, format="%.4f")
    saved_steps = BIN["mean_tests"] - ADP["mean_tests"]
    saved_s = units * corners * saved_steps * step_s
    st.success(f"{saved_steps:.2f} fewer steps per corner → **{saved_s / 3600:,.0f} tester-hours** and "
               f"**${saved_s * cost:,.0f}** per year (illustrative; substitute product-specific values).")


def page_binning():
    st.title("Binning & guard bands")
    st.markdown(
        f"Bins: {', '.join(str(b) for b in C.BINS_MHZ)} MHz. A device gets bin B only if its **measured** Fmax "
        f"≥ B + {C.GUARD_BAND_MHZ:.0f} MHz. The guard band absorbs tester noise: without it, noisy passes put "
        "devices in bins they can't sustain. With it, there are zero over-bin escapes, at the cost of "
        "under-binning devices that sit just above a bin edge."
    )
    c1, c2 = st.columns(2)
    with c1:
        g = R["gb_sweep"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["guard_band_mhz"], y=100 * g["over_bin_escape_rate"], name="Over-bin escapes %",
                                 mode="lines+markers", line_color=FAIL_COLOR))
        fig.add_trace(go.Scatter(x=g["guard_band_mhz"], y=100 * g["under_bin_rate"], name="Under-binned %",
                                 mode="lines+markers", line_color=ACCENT))
        fig.add_trace(go.Scatter(x=g["guard_band_mhz"], y=100 * g["bin_accuracy"], name="Bin accuracy %",
                                 mode="lines+markers", line_color=PASS_COLOR))
        fig.add_vline(x=C.GUARD_BAND_MHZ, line_dash="dash", annotation_text="deployed")
        fig.update_layout(height=400, title="Measured binning vs guard band (AI-assisted search)",
                          xaxis_title="Guard band (MHz)", yaxis_title="% of devices",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with c2:
        po = R["pred_only"].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=po["k_sigma"], y=100 * po["over_bin_escape_rate"], name="Over-bin escapes %",
                                 mode="lines+markers", line_color=FAIL_COLOR))
        fig.add_trace(go.Scatter(x=po["k_sigma"], y=100 * po["under_bin_rate"], name="Under-binned %",
                                 mode="lines+markers", line_color=ACCENT))
        fig.update_layout(height=400, title="Why bins are never assigned from predictions alone",
                          xaxis_title="Guard band (k × model σ)", yaxis_title="% of devices",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    st.caption(f"Prediction-only binning with no guard band over-bins {pct(R['pred_only'].iloc[0]['over_bin_escape_rate'])} "
               "of devices. Eliminating that requires k = 3, which under-bins "
               f"{pct(R['pred_only'].iloc[-1]['under_bin_rate'])}. Measured binning avoids this trade-off.")

    order = ["reject"] + [str(b) for b in C.BINS_MHZ]
    cm = pd.crosstab(P["true_bin"], P["speed_bin"]).reindex(index=order, columns=order, fill_value=0)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Assigned bin (measured, guard-banded)", y="True bin", color="Devices"))
    fig.update_layout(height=420, title="Assigned vs true bin: everything sits on or below the diagonal")
    st.plotly_chart(fig, width="stretch")


def page_explorer():
    st.title("Device explorer")
    c1, c2 = st.columns([2, 1])
    flt = c2.selectbox("Filter", ["All", "Resolved inside window", "Window extended", "OOD fallback",
                                  "Largest saving", "Largest prediction error"])
    view = P
    if flt == "Resolved inside window":
        view = P[(P["edge_extension"] == "") & ~P["is_ood"]]
    elif flt == "Window extended":
        view = P[P["edge_extension"] != ""]
    elif flt == "OOD fallback":
        view = P[P["is_ood"]]
    elif flt == "Largest saving":
        view = P.sort_values("tests_saved_vs_binary", ascending=False)
    elif flt == "Largest prediction error":
        view = P.reindex(P["prediction_error_mhz"].abs().sort_values(ascending=False).index)
    unit = c1.selectbox(f"Test device ({len(view):,} match)", view["unit_id"])

    raw = load_data().set_index("unit_id").loc[unit]
    p = load_service().predict(pd.DataFrame([raw[C.FEATURES].astype(float)])).iloc[0]
    f = float(raw[C.TARGET])
    # Same seeded tester as the benchmark, so these runs reproduce the reported test counts exactly.
    t_ad = SimulatedTester(f, _rng(unit, "adaptive"))
    r_ad = adaptive_search(t_ad, p["lower_bound_mhz"], p["upper_bound_mhz"], use_prediction=not p["is_ood"])
    t_bin = SimulatedTester(f, _rng(unit, "binary"))
    r_bin = binary_search(t_bin)
    r_lin = linear_search(SimulatedTester(f, _rng(unit, "linear")))
    u = P.set_index("unit_id").loc[unit]

    out = {
        "Predicted Fmax": "— (OOD)" if p["is_ood"] else f"{p['fmax_pred_mhz']:.1f} MHz",
        "Lower bound": "—" if p["is_ood"] else f"{p['lower_bound_mhz']:.1f} MHz",
        "Upper bound": "—" if p["is_ood"] else f"{p['upper_bound_mhz']:.1f} MHz",
        "Search window": "Full grid (fallback)" if r_ad.window is None else f"{r_ad.window[0]:.0f} – {r_ad.window[1]:.0f} MHz",
        "Measured Fmax": f"{r_ad.measured_mhz:.0f} MHz" if r_ad.measured_mhz else "below grid",
        "True Fmax (hidden from tester)": f"{f:.1f} MHz",
        "Speed bin": f"{u['speed_bin']} (true: {u['true_bin']})",
        "Test iterations": f"{r_ad.n_tests} (binary {r_bin.n_tests}, linear {r_lin.n_tests})",
        "Optimization gain": f"{100 * (1 - r_ad.n_tests / r_bin.n_tests):.0f}% fewer tests than binary",
        "Route": "OOD → full binary search" if r_ad.fallback else
                 {"": "resolved inside window", "upper": "window extended upward",
                  "lower": "searched below window"}[r_ad.edge],
        "Mahalanobis distance": f"{p['mahalanobis']:.2f}",
    }
    st.dataframe(pd.DataFrame(out.items(), columns=["Output", "Value"]), hide_index=True, width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(trajectory(t_bin.log, "Conventional binary search", true_f=f), width="stretch")
    c4.plotly_chart(trajectory(t_ad.log, "AI-assisted search", r_ad.window, f), width="stretch")


def page_predict():
    st.title("Predict new devices")
    st.markdown(f"Upload a CSV with the {len(C.FEATURES)} measurement columns, or enter one device by hand. Only "
                "allowlisted feature columns are read; anything else in the file is ignored.")
    svc = load_service()
    tab_csv, tab_form = st.tabs(["Upload CSV", "Single device"])

    with tab_csv:
        up = st.file_uploader("Device measurements CSV", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            missing = [c for c in C.FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                out = svc.predict(df)
                if "unit_id" in df.columns:
                    out.insert(0, "unit_id", df["unit_id"])
                c = st.columns(3)
                c[0].metric("Devices", f"{len(out):,}")
                c[1].metric("Windowed search", f"{(~out['is_ood']).sum():,}")
                c[2].metric("Fallback (OOD)", f"{out['is_ood'].sum():,}")
                st.dataframe(out.round(2), width="stretch")
                st.download_button("Download predictions CSV", out.to_csv(index=False), "fmax_predictions.csv",
                                   "text/csv")

    with tab_form:
        data = load_data()
        med = data.loc[data["split"] == "train", C.FEATURES].median()
        cols = st.columns(5)
        vals = {}
        for i, feat in enumerate([c for c in C.FEATURES if c != "radius_sq"]):
            vals[feat] = cols[i % 5].number_input(feat, value=float(round(med[feat], 4)), format="%.4f", step=0.1)
        vals["radius_sq"] = vals["die_x"] ** 2 + vals["die_y"] ** 2
        st.caption(f"radius_sq = die_x² + die_y² = {vals['radius_sq']:.4f}")
        p = svc.predict(pd.DataFrame([vals])).iloc[0]
        c = st.columns(4)
        if p["is_ood"]:
            c[0].metric("Predicted Fmax", "—")
            st.warning("This device is **out of distribution**. No prediction is trusted, and the tester runs the "
                       "full binary search.")
        else:
            c[0].metric("Predicted Fmax", f"{p['fmax_pred_mhz']:.1f} MHz")
            c[1].metric("95% interval", f"{p['lower_bound_mhz']:.0f} – {p['upper_bound_mhz']:.0f} MHz")
        c[2].metric("Action", "Full search" if p["is_ood"] else "Windowed search")
        c[3].metric("Mahalanobis", f"{p['mahalanobis']:.2f}")

        st.divider()
        default_f = 2000.0 if p["is_ood"] else float(round(p["fmax_pred_mhz"]))
        true_f = st.number_input("Simulate a tester run: assume the device's true Fmax is (MHz)", value=default_f,
                                 step=10.0)
        sim = svc.simulate(vals, true_f, seed=0)
        window = None if sim["fallback"] else (sim["lower_bound_mhz"], sim["upper_bound_mhz"])
        st.plotly_chart(trajectory([(t["freq_mhz"], t["pass"]) for t in sim["trajectory"]],
                                   f"Measured {sim['measured_mhz']} MHz → bin {sim['speed_bin']}", window, true_f),
                        width="stretch")


def page_data():
    st.title("Data & EDA")
    v = M["data_validation"]
    c = st.columns(5)
    c[0].metric("Rows", f"{v['n_rows']:,}")
    c[1].metric("Columns", v["n_cols"])
    c[2].metric("Missing values", sum(v["missing_values"].values()) if v["missing_values"] else 0)
    c[3].metric("Duplicate rows / IDs", f"{v['duplicate_rows']} / {v['duplicate_unit_ids']}")
    c[4].metric("true_bin recomputed", pct(v["true_bin_recomputed_agreement"], 2))
    st.write("Split sizes (preserved as supplied):", v["split_counts"])
    for w in v["warnings"]:
        st.warning(w)
    if not v["warnings"]:
        st.success("All validation checks passed.")
    if v["outliers_iqr"]:
        st.caption(f"Extreme values (3×IQR rule), flagged but kept: {v['outliers_iqr']}")

    with st.expander("Leakage guard: model inputs vs excluded columns"):
        a, b = st.columns(2)
        a.markdown("**Model inputs (allowlist)**\n\n" + "\n".join(f"- `{c}`" for c in C.FEATURES))
        b.markdown("**Never reach the model**\n\n" + "\n".join(f"- `{c}`" for c in C.LEAKAGE_COLS))

    data = load_data()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(data, x="die_x", y="die_y", color=C.TARGET, color_continuous_scale="Viridis",
                         labels={C.TARGET: "Fmax (MHz)"})
        fig.update_traces(marker_size=4)
        fig.update_layout(height=480, title="Wafer map of Fmax", yaxis_scaleanchor="x")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(data, x=C.TARGET, color="split", barmode="overlay", histnorm="probability density",
                           nbins=60, labels={C.TARGET: "Fmax (MHz)"})
        fig.update_layout(height=480, title="Fmax distribution by split")
        st.plotly_chart(fig, width="stretch")

    feat = st.selectbox("Feature vs Fmax", C.FEATURES)
    fig = px.scatter(data.sample(2000, random_state=0), x=feat, y=C.TARGET, color="split", opacity=0.5)
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

    corr = data[C.FEATURES + [C.TARGET]].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")
    fig.update_layout(height=650, title="Correlation matrix")
    st.plotly_chart(fig, width="stretch")


def page_report():
    st.title("Report & export")
    files = [("Per-device predictions", "predictions_test.csv"), ("Strategy benchmark", "benchmark_summary.csv"),
             ("Model comparison", "model_comparison.csv"), ("Metrics", "metrics.json"), ("Technical report", "report.md")]
    c = st.columns(len(files))
    for col, (label, name) in zip(c, files):
        if (A / name).exists():
            col.download_button(label, (A / name).read_bytes(), name)
    st.divider()
    st.markdown("\n".join(l for l in R["report_md"].splitlines() if not l.lstrip().startswith("![")))
    figs = sorted((A / "figures").glob("*.png"))
    for col, img in zip(st.columns(max(len(figs), 1)), figs):
        col.image(str(img), caption=img.stem)


{
    "Overview": page_overview,
    "Model performance": page_models,
    "Uncertainty": page_uncertainty,
    "Search optimization": page_search,
    "Binning & guard bands": page_binning,
    "Device explorer": page_explorer,
    "Predict new devices": page_predict,
    "Data & EDA": page_data,
    "Report & export": page_report,
}[page]()
