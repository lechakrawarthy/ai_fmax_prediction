# AI-Driven Adaptive Fmax Prediction & Test-Time Optimization

This project predicts each device's maximum operating frequency (Fmax) from pre-search electrical measurements and wraps the prediction in a calibrated 95% interval. The simulated tester then binary-searches only inside that window. If the window misses, the search extends past it. Out-of-distribution devices fall back to the full search. Speed bins always come from a **measured** Fmax with a guard band, never from the prediction alone.

> The dataset is synthetic. These results show feasibility on simulated data, not validated production savings.

## Quick start

```bash
pip install -r requirements.txt
python -m fmax.pipeline                  # validate → train → calibrate → simulate → artifacts/
streamlit run dashboard/app.py           # dashboard at http://localhost:8501
uvicorn api.main:app --reload            # API at http://localhost:8000/docs
python -m pytest                         # tests
```

## Results (held-out test split, 4,000 devices)

| Metric | Result | PRD target |
|---|---|---|
| Regression R² (linear model, 15 features + RO aggregates) | 0.941 (RMSE 25.4 MHz) | ≥ 0.90 ✅ |
| Prediction-interval coverage (split conformal) | 94.7% | ≈ 95% ✅ |
| Tests per device: linear / binary / AI-assisted | 40.5 / 6.45 / 3.87 | — |
| Reduction vs binary search | 40.0% | ≥ 30% ✅ |
| Over-bin escapes | 0 | — |
| Speed-bin accuracy | 90.5% | ≥ 95% ⚠️ |
| OOD fallback | 0.6% of test devices; 79% of shifted devices detected | tested ✅ |

**Why bin accuracy misses 95%:** the shortfall is under-binning, and it comes from the 8 MHz measurement guard band that absorbs σ = 4 MHz tester noise (as in the technical document). The model isn't the cause: plain binary search scores 90.5% under the same rule. Removing the guard band raises accuracy to 97.3% but lets 1.4% of devices over-bin. The dashboard's *Binning & guard bands* page shows this trade-off.

## Layout

| Path | Contents |
|---|---|
| `fmax/config.py` | Feature allowlist, leakage blocklist, tester grid (1600–2400 MHz, 10 MHz steps), bins, guard band |
| `fmax/data.py` | Loading, schema, missing-value, duplicate and split validation |
| `fmax/features.py` | Leakage guard, ring-oscillator aggregate features |
| `fmax/models.py` | Linear / random forest / gradient boosting / XGBoost; 5-fold CV selection; versioned bundle |
| `fmax/uncertainty.py` | Split-conformal intervals (absolute and normalized), OOD detector |
| `fmax/search.py` | Noisy simulated tester; linear, binary, windowed (Algorithm 1) and adaptive search; binning |
| `fmax/pipeline.py` | End-to-end run, strategy benchmark, guard-band and prediction-only binning studies |
| `fmax/service.py` | Inference service shared by the API and the dashboard |
| `api/main.py` | FastAPI: `GET /health`, `/model`, `/metrics`; `POST /predict`, `/simulate` |
| `dashboard/app.py` | Streamlit dashboard (9 pages) |
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `artifacts/` | Model bundle, metrics, per-device predictions, benchmark, report and figures |

## Dashboard pages

Overview & acceptance criteria · Model performance · Uncertainty & OOD · Search optimization (with tester-time cost estimate) · Binning & guard bands · Device explorer (per-device search trajectories, reproducing the benchmark runs exactly) · Predict new devices (CSV upload or manual entry, with simulated tester run) · Data & EDA · Report & export.

## Data leakage

Only the 15 measurement columns in `fmax/config.py::FEATURES` reach a model. `features.assert_no_leakage` rejects every target-derived or search-outcome column listed in the PRD (§5.4). The supplied `fmax_pred_mhz` and window columns are used only as evaluation references.
