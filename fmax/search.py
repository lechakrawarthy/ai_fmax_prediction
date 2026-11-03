"""Module 5: simulated ATE and frequency-search strategies.

All strategies search the same grid and use the same noisy pass/fail model, so their
test counts are directly comparable. The measured Fmax is the highest grid point that
passed; bins are assigned from that measured value, never from a prediction alone.
"""
from dataclasses import dataclass

import numpy as np

from . import config as C

GRID = np.arange(C.GRID_MIN_MHZ, C.GRID_MAX_MHZ + C.GRID_STEP_MHZ / 2, C.GRID_STEP_MHZ)
N_GRID = len(GRID)


class SimulatedTester:
    """One device on the ATE. pass(f) is true when f is at or below the noisy boundary."""

    def __init__(self, fmax_true_mhz: float, rng: np.random.Generator, sigma_test: float = C.SIGMA_TEST_MHZ):
        self.fmax, self.rng, self.sigma = float(fmax_true_mhz), rng, sigma_test
        self.n_tests = 0
        self.log: list[tuple[float, bool]] = []

    def passes(self, idx: int) -> bool:
        f = GRID[idx]
        ok = bool(f <= self.fmax + self.rng.normal(0.0, self.sigma))
        self.n_tests += 1
        self.log.append((float(f), ok))
        return ok


@dataclass
class SearchResult:
    measured_mhz: float | None   # None: failed at the lowest grid point
    n_tests: int
    method: str
    fallback: bool = False
    edge: str = ""               # "", "upper" or "lower" when the window had to be extended
    window: tuple[float, float] | None = None

    @property
    def time_ms(self) -> float:
        return self.n_tests * C.MS_PER_TEST


def _bisect(t: SimulatedTester, lo: int, hi: int) -> int:
    """Highest passing index given that index lo is known/assumed to pass and hi to fail.

    lo may be -1 and hi may be N_GRID (virtual endpoints that are never tested).
    """
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t.passes(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _as_result(t, idx, method, **kw) -> SearchResult:
    return SearchResult(None if idx < 0 else float(GRID[idx]), t.n_tests, method, **kw)


def linear_search(t: SimulatedTester) -> SearchResult:
    """Step up from the bottom of the grid until the first failure."""
    idx = -1
    while idx + 1 < N_GRID and t.passes(idx + 1):
        idx += 1
    return _as_result(t, idx, "linear")


def binary_search(t: SimulatedTester) -> SearchResult:
    return _as_result(t, _bisect(t, -1, N_GRID), "binary")


def windowed_search(t: SimulatedTester, low_mhz: float, high_mhz: float) -> SearchResult:
    """Algorithm 1: binary search inside [low, high]; extend if the boundary lies outside it."""
    lo = int(np.clip(np.floor((low_mhz - C.GRID_MIN_MHZ) / C.GRID_STEP_MHZ), 0, N_GRID - 1))
    hi = int(np.clip(np.ceil((high_mhz - C.GRID_MIN_MHZ) / C.GRID_STEP_MHZ), lo, N_GRID - 1))
    window = (float(GRID[lo]), float(GRID[hi]))

    best = _bisect(t, lo - 1, hi + 1)
    edge = ""
    if best == hi and hi < N_GRID - 1:
        # Top of window passed: gallop upward with doubling stride, then bisect the gap.
        edge, stride = "upper", 1
        while hi + stride < N_GRID and t.passes(hi + stride):
            hi += stride
            stride *= 2
        best = _bisect(t, hi, min(hi + stride, N_GRID))
    elif best == lo - 1 and lo > 0:
        # Everything in the window failed: the boundary is below it.
        edge = "lower"
        best = _bisect(t, -1, lo)
    return _as_result(t, best, "windowed", edge=edge, window=window)


def adaptive_search(t: SimulatedTester, low_mhz, high_mhz, use_prediction: bool) -> SearchResult:
    """Windowed search when the prediction is trusted, full binary search (fallback) otherwise."""
    if not use_prediction:
        r = binary_search(t)
        r.method, r.fallback = "adaptive", True
        return r
    r = windowed_search(t, low_mhz, high_mhz)
    r.method = "adaptive"
    return r


def assign_bin(measured_mhz, guard_band: float = C.GUARD_BAND_MHZ):
    """Highest bin B with B + guard_band <= measured Fmax; REJECT if none."""
    if measured_mhz is None or not np.isfinite(measured_mhz):
        return C.REJECT
    chosen = C.REJECT
    for b in C.BINS_MHZ:
        if measured_mhz >= b + guard_band:
            chosen = b
    return chosen


def bin_rank(b) -> int:
    return -1 if b == C.REJECT else C.BINS_MHZ.index(int(b))
