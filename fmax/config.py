"""Project-wide constants: feature allowlist, leakage blocklist, tester and bin settings.

Tester/bin values follow Table 5 of the technical document (synthetic, illustrative).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "synthetic_fmax_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"

RANDOM_STATE = 42

# --- Columns -----------------------------------------------------------------
RO_COLS = [f"ro_{i:02d}" for i in range(1, 11)]
FEATURES = RO_COLS + ["iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"]
TARGET = "fmax_true_mhz"

# Must never reach a model (PRD 5.4).
LEAKAGE_COLS = [
    "fmax_true_mhz",
    "fmax_pred_mhz",
    "window_low_mhz",
    "window_high_mhz",
    "true_bin_mhz",
    "tests_binary",
    "tests_windowed",
    "fmax_measured_windowed_mhz",
    "unit_id",
    "split",
]

EXPECTED_COLUMNS = ["unit_id", "split"] + FEATURES + LEAKAGE_COLS[:-2]
EXPECTED_SPLITS = {"train": 1500, "calibration": 500, "test": 4000}

# --- Tester model ------------------------------------------------------------
GRID_MIN_MHZ = 1600.0
GRID_MAX_MHZ = 2400.0
GRID_STEP_MHZ = 10.0
SIGMA_TEST_MHZ = 4.0          # Gaussian noise on each pass/fail boundary
PATTERN_MS = 25.0             # pattern application per step
CLOCK_SWITCH_MS = 5.0         # clock switch per step
MS_PER_TEST = PATTERN_MS + CLOCK_SWITCH_MS

# --- Binning -----------------------------------------------------------------
BINS_MHZ = [1800, 1900, 2000, 2100, 2200]
GUARD_BAND_MHZ = 8.0
REJECT = "reject"

# --- Uncertainty / fallback --------------------------------------------------
CONFORMAL_ALPHA = 0.05        # 95 % prediction intervals
OOD_QUANTILE = 0.999          # Mahalanobis threshold quantile on training data
OOD_RANGE_MARGIN = 0.10       # allowed overshoot of training min/max, as fraction of range
