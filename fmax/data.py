"""Module 1/2: data ingestion, validation and preprocessing checks."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    missing_values: dict
    duplicate_rows: int
    duplicate_unit_ids: int
    split_counts: dict
    outliers_iqr: dict
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        return {**self.__dict__, "ok": self.ok}


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    """Read the CSV without modifying the source file."""
    return pd.read_csv(path)


def validate(df: pd.DataFrame, strict: bool = True) -> ValidationReport:
    """Schema, dtype, missing-value, duplicate and split checks.

    Hard schema problems raise; softer issues are collected as warnings.
    """
    missing_cols = sorted(set(C.EXPECTED_COLUMNS) - set(df.columns))
    if missing_cols:
        raise DataValidationError(f"Missing columns: {missing_cols}")

    non_numeric = [c for c in C.FEATURES + [C.TARGET] if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric feature/target columns: {non_numeric}")

    warnings = []
    missing = {c: int(n) for c, n in df.isna().sum().items() if n}
    if missing:
        warnings.append(f"Missing values in {list(missing)}")

    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["unit_id"].duplicated().sum())
    if dup_rows or dup_ids:
        warnings.append(f"{dup_rows} duplicate rows, {dup_ids} duplicate unit_ids")

    split_counts = df["split"].value_counts().to_dict()
    unknown = set(split_counts) - set(C.EXPECTED_SPLITS)
    if unknown:
        raise DataValidationError(f"Unknown split labels: {sorted(unknown)}")
    if strict and split_counts != C.EXPECTED_SPLITS:
        warnings.append(f"Split sizes {split_counts} differ from expected {C.EXPECTED_SPLITS}")

    # 3*IQR rule: flag, never drop (PRD: preserve the original dataset)
    outliers = {}
    for c in C.FEATURES + [C.TARGET]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - 3 * iqr) | (df[c] > q3 + 3 * iqr)).sum())
        if n:
            outliers[c] = n

    return ValidationReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        missing_values=missing,
        duplicate_rows=dup_rows,
        duplicate_unit_ids=dup_ids,
        split_counts=split_counts,
        outliers_iqr=outliers,
        warnings=warnings,
    )


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the predefined train / calibration / test partitions (never reshuffled)."""
    return {name: df[df["split"] == name].reset_index(drop=True) for name in C.EXPECTED_SPLITS}


def true_bin(fmax_mhz) -> np.ndarray:
    """Highest bin the device reaches (no guard band).

    Matches `true_bin_mhz` except for the few rows whose 0.1 MHz-rounded Fmax sits exactly
    on a bin edge (the supplied labels were computed before rounding).
    """
    f = np.asarray(fmax_mhz, dtype=float)
    out = np.full(f.shape, C.REJECT, dtype=object)
    for b in C.BINS_MHZ:
        out[f >= b] = b
    return out
