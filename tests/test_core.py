import numpy as np
import pandas as pd
import pytest

from fmax import config as C
from fmax.data import DataValidationError, load_raw, split, true_bin, validate
from fmax.features import assert_no_leakage, select_features
from fmax.search import (
    GRID, N_GRID, SimulatedTester, adaptive_search, assign_bin, binary_search, linear_search, windowed_search,
)
from fmax.uncertainty import OODDetector, SplitConformal, conformal_quantile


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def noiseless(f):
    return SimulatedTester(f, np.random.default_rng(0), sigma_test=0.0)


def grid_floor(f):
    return GRID[GRID <= f].max()


# --- data / leakage --------------------------------------------------------------
def test_dataset_validates_and_splits_preserved(raw):
    rep = validate(raw)
    assert rep.ok, rep.warnings
    parts = split(raw)
    assert {k: len(v) for k, v in parts.items()} == C.EXPECTED_SPLITS


def test_missing_column_raises(raw):
    with pytest.raises(DataValidationError):
        validate(raw.drop(columns=["ro_03"]))


def test_feature_selection_excludes_every_leakage_column(raw):
    X = select_features(raw)
    assert list(X.columns) == C.FEATURES
    assert not set(X.columns) & set(C.LEAKAGE_COLS)


@pytest.mark.parametrize("col", C.LEAKAGE_COLS)
def test_leakage_guard_rejects(col):
    with pytest.raises(ValueError):
        assert_no_leakage(C.FEATURES + [col])


def test_true_bin_matches_supplied_column(raw):
    off_edge = ~raw[C.TARGET].isin(C.BINS_MHZ)   # labels at exact edges depend on pre-rounding value
    agree = true_bin(raw[C.TARGET]).astype(str) == raw["true_bin_mhz"].astype(str)
    assert agree[off_edge].all()
    assert (~agree).sum() <= 15


# --- search ----------------------------------------------------------------------
@pytest.mark.parametrize("f", [1605.0, 1777.7, 2000.0, 2063.4, 2399.0])
def test_noiseless_searches_find_grid_floor(f):
    for fn in (linear_search, binary_search):
        assert fn(noiseless(f)).measured_mhz == grid_floor(f)
    assert windowed_search(noiseless(f), f - 40, f + 40).measured_mhz == grid_floor(f)


def test_binary_search_cost_is_logarithmic():
    assert binary_search(noiseless(1990.0)).n_tests <= int(np.ceil(np.log2(N_GRID + 1)))


def test_window_too_low_extends_upward():
    r = windowed_search(noiseless(2250.0), 1900, 1950)
    assert r.edge == "upper" and r.measured_mhz == 2250.0


def test_window_too_high_searches_below():
    r = windowed_search(noiseless(1712.0), 2100, 2150)
    assert r.edge == "lower" and r.measured_mhz == 1710.0


def test_device_failing_everywhere_is_rejected():
    r = binary_search(noiseless(1500.0))
    assert r.measured_mhz is None and assign_bin(r.measured_mhz) == C.REJECT


def test_fallback_runs_full_binary_search():
    r = adaptive_search(noiseless(2063.0), 2300, 2350, use_prediction=False)
    assert r.fallback and r.window is None and r.measured_mhz == 2060.0
    assert r.n_tests == binary_search(noiseless(2063.0)).n_tests


def test_assign_bin_uses_guard_band():
    assert assign_bin(2005.0) == 1900          # 2000 + 8 > 2005
    assert assign_bin(2010.0) == 2000
    assert assign_bin(1790.0) == C.REJECT


# --- uncertainty / OOD -----------------------------------------------------------
def test_conformal_quantile_finite_sample():
    s = np.arange(1, 101, dtype=float)
    assert conformal_quantile(s, 0.05) == 96.0


def test_conformal_coverage_on_synthetic_linear_data():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6000, 3))
    y = X @ [3, -2, 1] + rng.normal(0, 1, 6000)
    m = LinearRegression().fit(X[:2000], y[:2000])
    cp = SplitConformal(m, alpha=0.1).calibrate(X[2000:3000], y[2000:3000])
    _, lo, hi = cp.predict_interval(X[3000:])
    cov = np.mean((y[3000:] >= lo) & (y[3000:] <= hi))
    assert 0.87 <= cov <= 0.93


def test_ood_flags_excursions_and_bad_values(raw):
    parts = split(raw)
    Xtr = select_features(parts["train"])
    det = OODDetector().fit(Xtr)
    assert det.check(select_features(parts["test"]))["is_ood"].mean() < 0.02
    bad = Xtr.head(3).copy()
    bad.iloc[0, 0] = 50.0                 # far outside RO range
    bad.iloc[1, :] = bad.iloc[1, :] * 0 + 4.0
    bad.iloc[2, 0] = np.nan
    assert det.check(bad)["is_ood"].all()
