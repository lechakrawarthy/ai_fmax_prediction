# AI-Driven Adaptive Fmax Prediction and Test-Time Optimization: Technical Implementation Report

**Version** 1.0 · November 2026
**Data** Supplied synthetic dataset: 6,000 devices × 25 columns (`synthetic_fmax_dataset.csv`)
**Reference** *AI-Optimized Fmax Search for Test Time Reduction*, technical document v1.0

> All results come from synthetic data. They show that the method works on the simulated population. They are not validated production test-time savings (PRD §15).

---

## 1. Summary

The system predicts each device's maximum operating frequency (Fmax) from measurements taken before the frequency search. It then uses a calibrated 95% prediction interval to narrow that search. The final speed bin always comes from a measured Fmax with the standard guard band, never from the prediction alone.

| Outcome (4,000 held-out test devices) | Result | PRD target |
|---|---|---|
| Regression R² | 0.941 (RMSE 25.4 MHz, MAPE 1.02%) | ≥ 0.90 ✅ |
| Prediction-interval coverage | 94.7% | ≈ 95% ✅ |
| Mean tests per device: binary → AI-assisted | 6.45 → 3.87 (−40.0%) | ≥ 30% reduction ✅ |
| Over-bin escapes | 0 | — |
| Speed-bin accuracy | 90.5% | ≥ 95% ⚠️ (see §7.3) |
| Data leakage | Allowlist enforced in code | Zero known ✅ |
| Fallback mechanism | 0.6% of test devices routed; 79% of shifted devices caught | Tested ✅ |

## 2. Architecture

The implementation follows the six modules of PRD §6. Each module is a separate file in the `fmax` package. The API and the dashboard share one inference service, so both use the same model and decision logic.

| PRD module | Implementation | Responsibility |
|---|---|---|
| 1 Data ingestion | `fmax/data.py` | Load CSV unchanged; schema, dtype, missing-value, duplicate and split checks |
| 2 Preprocessing | `fmax/data.py`, `fmax/features.py` | 3×IQR outlier flagging (flag, never drop); feature allowlist; ring-oscillator aggregates |
| 3 ML prediction engine | `fmax/models.py` | Four candidate regressors, 5-fold CV comparison, selection, versioned persistence |
| 4 Uncertainty estimation | `fmax/uncertainty.py` | Split-conformal intervals; out-of-distribution (OOD) detector |
| 5 Adaptive search | `fmax/search.py` | Noisy simulated tester; linear, binary, windowed and adaptive search; binning |
| 6 Visualization & reporting | `dashboard/app.py`, `api/main.py`, `fmax/report.py` | Streamlit dashboard, FastAPI service, generated report and figures |
| Orchestration | `fmax/pipeline.py` | End-to-end run and benchmark; writes all artifacts |
| Inference | `fmax/service.py` | Prediction + interval + OOD decision for one or many devices |

Data flow: `CSV → validate → split → select_features → model → conformal interval → OOD check → search plan → simulated tester → measured Fmax → bin`.

## 3. Data

### 3.1 Validation results

| Check | Result |
|---|---|
| Shape | 6,000 rows × 25 columns |
| Missing values | 0 |
| Duplicate rows / unit IDs | 0 / 0 |
| Splits | train 1,500 · calibration 500 · test 4,000 (as specified, never reshuffled) |
| Extreme values (3×IQR) | 1 in `ro_03`, 1 in `ro_09`; flagged and kept |
| `true_bin_mhz` recomputed from Fmax | 99.92% agreement; the few differences are rows where 0.1 MHz rounding puts Fmax exactly on a bin edge |

Hard problems (a missing column, non-numeric features, unknown split labels) raise `DataValidationError`. Softer issues become warnings in the report.

### 3.2 Leakage prevention

Only the 15 allowlisted measurements in `config.FEATURES` can reach a model:
`ro_01 … ro_10`, `iddq_norm`, `vmin_norm`, `die_x`, `die_y`, `radius_sq`.

`features.assert_no_leakage` runs on every model input. It raises an error if any PRD §5.4 column appears, or any column outside the allowlist. The supplied `fmax_pred_mhz` and window columns are used only as evaluation references, for example as the "supplied baseline" in §5.

### 3.3 Engineered features

`RingOscAggregates` appends the mean, standard deviation, minimum and maximum of the ten ring-oscillator readings. The mean is a lower-noise estimate of the latent process speed. All four are derived only from allowlisted columns, so they can't carry target information.

## 4. Simulated tester and search algorithms

The tester follows Table 5 of the technical document:

| Parameter | Value |
|---|---|
| Frequency grid | 1,600–2,400 MHz, 10 MHz steps (81 points) |
| Test repeatability | pass if f ≤ Fmax + ε, ε ~ N(0, 4 MHz), drawn independently for every test |
| Time per test | 25 ms pattern application + 5 ms clock switch |
| Bins | 1,800 / 1,900 / 2,000 / 2,100 / 2,200 MHz; device gets bin B if measured Fmax ≥ B + 8 MHz guard band |

Every strategy measures Fmax as the highest grid frequency that passed:

- **Linear sweep:** step up from 1,600 MHz until the first failure.
- **Binary search:** bisect the full grid; ⌈log₂ 82⌉ = 6–7 tests.
- **Windowed search (Algorithm 1):** snap the interval [L, U] to the grid and bisect inside it, treating L as passing and U as failing.
  - If the top of the window passes, the boundary is above it: gallop upward with a doubling stride, then bisect the gap.
  - If everything in the window fails, bisect below it.

  So a wrong window costs extra tests, never a wrong measurement.
- **Adaptive (deployed):** windowed search when the device is in-distribution; full binary search otherwise.

Each (device, strategy) pair uses its own deterministic random stream, seeded from the unit ID. Results are reproducible, and the dashboard's device explorer replays the exact benchmark runs.

## 5. Models

Four regressors were trained on the 1,500 training devices, each preceded by the ring-oscillator aggregates:

| Model | Configuration | CV RMSE (MHz) | Test RMSE | Test R² | Latency (µs/device) |
|---|---|---|---|---|---|
| **Linear regression** (selected) | standardized inputs | 24.37 | **25.38** | **0.941** | **1.3** |
| Random forest | 400 trees, min leaf 3 | 25.29 | 26.77 | 0.934 | 55.5 |
| Gradient boosting | 600 trees, depth 3, lr 0.03, subsample 0.8 | 25.42 | 26.54 | 0.935 | 11.0 |
| XGBoost | 600 trees, depth 3, lr 0.03, subsample/colsample 0.8 | 25.52 | 26.61 | 0.935 | 9.7 |
| Supplied `fmax_pred_mhz` (reference only) | — | — | 26.94 | 0.933 | — |

**Selection rule.** Take the lowest 5-fold CV RMSE on the training split. Among models within 0.25 MHz of it, choose the fastest at inference. The test split is reported but never used for selection.

The linear model won outright. Two things point to a mostly linear relationship in this data: the tree ensembles gain nothing, and their test RMSE is about 1.5 MHz worse. The linear model is also the cheapest to embed in a test program; the technical document's §9.1 lists coefficient tables as an integration option.

## 6. Uncertainty and fallback

### 6.1 Split-conformal intervals

Residuals on the 500 calibration devices, which were never used for fitting, give the finite-sample-corrected quantile q = ⌈(n+1)(1−α)⌉-th smallest |y − ŷ|, with α = 0.05. The interval is [ŷ − q, ŷ + q].

| Variant | Coverage (test) | Mean width | Status |
|---|---|---|---|
| Absolute (constant width) | 94.65% | 96.6 MHz (q = 48.3 MHz) | **Deployed** |
| Normalized (width × learned per-device difficulty) | 95.20% | 104.1 MHz | Reported only |

The normalized variant scales width by a gradient-boosted model of expected |residual|, fitted on out-of-fold training residuals. On this data the errors are close to homoscedastic, so it gives wider windows on average. The constant-width interval is used.

### 6.2 Out-of-distribution detection

A device is routed to the full binary search if any of these hold:

- a feature lies more than 10% of the training range outside the training min/max;
- its Mahalanobis distance from the training data exceeds the 99.9th percentile of training distances (threshold 5.80);
- any feature is missing or non-finite.

| Measure | Result |
|---|---|
| Test devices flagged (fallback rate) | 0.6% |
| Detection on 500 test devices with a synthetic ring-oscillator shift (common offset of ±2.5 plus N(0, 1), in normalized RO units) | 78.8% |

A missed OOD device is still measured correctly: its window may be wrong, but the edge-extension rule finds the boundary at the cost of extra tests.

## 7. Results

### 7.1 Search cost

| Strategy | Mean tests | p95 | Max | Mean time (ms) | Saving vs binary | Edge-extension rate | Fallback rate |
|---|---|---|---|---|---|---|---|
| Linear sweep | 40.47 | 58 | 75 | 1,214 | −527% | — | — |
| Binary search | 6.45 | 7 | 7 | 194 | 0% | — | — |
| Window shipped with dataset | 3.94 | 7 | 9 | 118 | 38.9% | 10.5% | — |
| ML window | 3.84 | 4 | 9 | 115 | 40.5% | 3.9% | — |
| **ML window + OOD fallback (deployed)** | **3.87** | **4** | 9 | **116** | **40.0%** | 4.0% | 0.6% |

The fallback costs 0.03 tests per device on average, the price of sending 0.6% of devices to the full search. These figures match the technical document's own simulation (3.89 vs 6.45 tests, 40%) and the 45–53% reported in the literature for model-guided search [technical document ref. 2].

### 7.2 Measurement accuracy and binning

| Strategy | Mean \|measured − true\| | Bin accuracy | Over-bin escapes | Under-bin rate |
|---|---|---|---|---|
| Linear sweep | 5.84 MHz | 90.3% | 0 | 9.7% |
| Binary search | 5.77 MHz | 90.5% | 0 | 9.5% |
| Deployed flow | 5.73 MHz | 90.5% | 0 | 9.5% |

The deployed flow measures as accurately as binary search, because it measures the same boundary.

### 7.3 Why bin accuracy is 90.5%, not ≥ 95%

The gap is under-binning only, and the model doesn't cause it. The 8 MHz measurement guard band, combined with grid quantization (measured Fmax is biased about half a 10 MHz step low), places devices that sit just above a bin edge one bin down. Plain binary search scores the same 90.5%.

Guard-band sweep for the deployed flow:

| Guard band | Bin accuracy | Over-bin escapes | Under-bin rate |
|---|---|---|---|
| 0 MHz | 97.3% | **1.43%** | 1.3% |
| 4 MHz | 90.5% | 0 | 9.5% |
| 8 MHz (deployed) | 90.5% | 0 | 9.5% |
| 12 MHz | 80.5% | 0 | 19.5% |

You can only reach 95% by dropping the guard band, and that lets 1.4% of devices into bins they can't sustain under tester noise. This is a quality-policy choice for the product owner. The implementation keeps the conservative setting.

### 7.4 Why bins are never assigned from predictions

If bins came from the prediction alone, with a guard band of k × σ (σ = 25.1 MHz):

| k | Bin accuracy | Over-binned | Under-binned |
|---|---|---|---|
| 0 | 79.6% | 9.6% | 10.8% |
| 1 | 69.5% | 2.2% | 28.4% |
| 2 | 49.4% | 0.2% | 50.4% |
| 3 | 28.1% | 0.0% | 71.9% |

Model error (about 25 MHz) is too large relative to the 100 MHz bins for prediction-only binning. This confirms the technical document's §7.4 finding and justifies the measured-binning design required by PRD §8 ("speed-bin decisions must not rely solely on unverified predictions").

## 8. Interfaces

### 8.1 Dashboard (`streamlit run dashboard/app.py`)

| Page | Contents |
|---|---|
| Overview | Headline metrics, acceptance criteria, strategy comparison, predicted vs true |
| Model performance | Comparison table, error distribution, residuals, feature influence |
| Uncertainty | Conformal variants, interval band plot, OOD distance distribution |
| Search optimization | All strategies, tests-per-device histogram, routing breakdown, tester-time cost calculator |
| Binning & guard bands | Guard-band sweep, prediction-only trade-off, assigned vs true bin matrix |
| Device explorer | PRD §12 per-device output, plus binary vs AI-assisted search trajectories (exact benchmark replay) |
| Predict new devices | CSV upload or manual entry, OOD warning, simulated tester run, CSV download |
| Data & EDA | Validation results, leakage lists, wafer map, distributions, correlations |
| Report & export | Downloads for predictions, benchmark, comparison, metrics and report |

### 8.2 REST API (`uvicorn api.main:app`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness and model availability |
| `GET /model` | Model metadata (name, α, q, data fingerprint, version, features) |
| `GET /metrics` | Full evaluation metrics |
| `POST /predict` | List of devices → prediction, interval, Mahalanobis distance, `action` (`windowed_search` or `fallback_full_search`) |
| `POST /simulate` | One device + hidden true Fmax → measured Fmax, bin, test count, trajectory |

## 9. Verification

`python -m pytest` runs 30 tests:

- **Data:** dataset validates and splits are preserved; a missing column raises; supplied bins are reproduced.
- **Leakage:** every PRD §5.4 column is excluded from features; the guard rejects each one individually.
- **Search:** noiseless searches find the grid floor; binary cost is logarithmic; windows that are too low or too high extend correctly; devices that fail everywhere are rejected; fallback runs the full search; the guard band is applied.
- **Uncertainty:** finite-sample conformal quantile; coverage on synthetic data; OOD flags excursions and bad values.
- **API:** prediction and simulation round trip; an OOD device triggers fallback with a correct measurement.

## 10. Reproducibility and versioning

- `python -m fmax.pipeline` regenerates every artifact from the CSV in about a minute.
- All randomness is seeded (`RANDOM_STATE = 42`, plus per-unit tester streams).
- `artifacts/model_bundle.joblib` stores the model, the calibrator and the OOD detector together. Its `.meta.json` sidecar records the package version, creation time, feature list, α, q and a fingerprint of the training data.

## 11. Deviations from the PRD

| PRD item | Decision | Reason |
|---|---|---|
| MAPIE for uncertainty | Custom split conformal (§6.1) | Fewer dependencies; the PRD allows "MAPIE or custom conformal calibration" |
| Speed-bin accuracy ≥ 95% | 90.5% achieved | Set by the guard-band policy, not the model (§7.3) |
| Multiple V/T corners | One corner simulated | The dataset has one Fmax per device |

## 12. Limitations

1. **Synthetic data.** The monitor–Fmax relationships, noise levels and timings are assumptions.
2. **Small sample for escape rates.** Zero over-bin escapes in 4,000 devices bounds the rate only to below about 750 ppm at 95% confidence (rule of three). Qualifying at ppm level needs far more devices.
3. **Simple OOD model.** The OOD check covers single-device feature excursions only. Lot-level drift needs the monitoring described below.

## 13. Recommended next steps

Following the lifecycle in the technical document's §9.2:

1. **Shadow mode** on real lots: predict and log, but keep binning on the reference flow.
2. **Audit sampling:** about 60 fully searched devices per monitoring period to detect a 10 MHz residual bias (σ ≈ 27 MHz, 95% confidence, 90% power).
3. **Drift monitoring:** residual EWMA/CUSUM, per-feature PSI, live interval coverage, fallback and edge-hit rates.
4. **Multi-corner window reuse** (Algorithm 4) for further savings in DVFS binning.
