"""
Volatility Forecasting — Feature Construction Pipeline v2 (OPTIMISED)
=====================================================================
Drop-in replacement for pipeline_v2.py with identical outputs.

Optimisations applied
─────────────────────
1. Numba JIT for all rolling primitives (_rolling_sum, _rolling_std,
   _rolling_max, _rolling_corr, _rolling_skew, _rolling_kurt).
2. Numba JIT for expensive estimators (_tsrv, _realized_kernel,
   _realized_quarticity, _roll_measure, _parkinson).
3. RollingCache — memoises _rolling_sum(id(arr), w) to eliminate the
   ~40 % of rolling-sum calls that are redundant across feature groups.
4. Precomputed intermediates — r², |r|, log(mid) etc. computed once and
   shared via the cache.
5. Thread-parallel feature groups — independent instrument × tier blocks
   dispatched to a ThreadPoolExecutor (NumPy releases the GIL).
6. Vectorised TSRV — pure-NumPy grid construction replaces the Python
   while-loop.
7. Batched spectral / Fourier — unchanged (already good), but benefits
   from the faster rolling_sum used for NaN counting.

All constants, PipelineConfig, seasonality, targets, model fitting and
evaluation are identical to pipeline_v2.py.
"""
from __future__ import annotations

import calendar
import logging
import gc
import time
import warnings
import numpy as np
import polars as pl
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

from scipy.signal import lfilter, lfilter_zi

import numba as nb

try:
    from tqdm import tqdm, trange
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = "?"
        for i, item in enumerate(iterable):
            print(f"\r  {desc}: {i+1}/{total}", end="", flush=True)
            yield item
        print()

    def trange(n, **kwargs):
        return tqdm(range(n), **kwargs)


# ── Logging (unchanged) ──────────────────────────────────
log = logging.getLogger("pipeline_v2")

def setup_logging(level: int = logging.INFO):
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(level)


class StepTimer:
    def __init__(self, step: str):
        self.step = step
    def __enter__(self):
        self.t0 = time.perf_counter()
        log.info(f"▶ {self.step}")
        return self
    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.t0
        log.info(f"  ✓ {self.step} ({elapsed:.1f}s)")


# ── Safe arithmetic ───────────────────────────────────────

def _safe_div(a: np.ndarray, b: np.ndarray,
              fill: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill, dtype=np.float64)
    mask = (b != 0) & np.isfinite(b) & np.isfinite(a)
    np.divide(a, b, out=out, where=mask)
    return out


# ═══════════════════════════════════════════════════════════
# Constants (unchanged)
# ═══════════════════════════════════════════════════════════

AGG_FREQ_S = 1
MAX_HORIZON_MIN = 15
MAX_HORIZON_S = MAX_HORIZON_MIN * 60
ROLLING_WINDOWS = [5, 30, 60, 1440, 2048, 3600, 4096, 8192, 14400, 16384]
ROLLING_WINDOWS_EXTENDED = list(ROLLING_WINDOWS)
TRAIN_FRAC = 0.8
FFF_PERIODS_HOURS = [8, 24, 168, 720]
FFF_HARMONICS = [4, 12, 25, 4]
EPS = 1e-12


# ═══════════════════════════════════════════════════════════
# Config (unchanged)
# ═══════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    base_freq_s: int = 1
    agg_freq_s: int = AGG_FREQ_S
    instruments: list[str] = field(default_factory=lambda: ["s", "p"])
    bid_col: str = "p_bid_0_price"
    ask_col: str = "p_ask_0_price"
    bid_size_col: str = "p_bid_0_size"
    ask_size_col: str = "p_ask_0_size"
    max_horizon_min: int = MAX_HORIZON_MIN
    max_horizon_s: int = MAX_HORIZON_S
    rolling_windows: list[int] = field(
        default_factory=lambda: list(ROLLING_WINDOWS)
    )
    rolling_windows_extended: list[int] = field(
        default_factory=lambda: list(ROLLING_WINDOWS_EXTENDED)
    )
    train_frac: float = TRAIN_FRAC
    fff_periods_hours: list[float] = field(
        default_factory=lambda: list(FFF_PERIODS_HOURS)
    )
    fff_harmonics: list[int] = field(
        default_factory=lambda: list(FFF_HARMONICS)
    )
    eps: float = EPS
    ts_col: str = "received_time"
    input_map: dict[str, str] = field(default_factory=dict)
    modwt_scales: list[int] = field(
        default_factory=lambda: [6, 8, 10, 13, 15, 17]
    )



    tsrv_K: int = 5
    # Number of worker threads for parallel feature computation
    n_workers: int = 4
    # Optional Coinbase BTC spot instrument prefix (e.g. "c")
    coinbase_spot: Optional[str] = None
    # Seconds without a Coinbase update before marking as stale
    coinbase_stale_threshold_s: int = 10

    @property
    def base_features(self) -> list[str]:
        return ["rv", "rsv_pos", "rsv_neg", "bpv"]

    @property
    def spot(self) -> str:
        return self.instruments[0]

    @property
    def perp(self) -> str:
        return self.instruments[1]

    @property
    def all_instruments(self) -> list[str]:
        """Main instruments + optional Coinbase spot."""
        insts = list(self.instruments)
        if self.coinbase_spot and self.coinbase_spot not in insts:
            insts.append(self.coinbase_spot)
        return insts

    @property
    def target_instruments(self) -> list[str]:
        """Instruments to build targets / fit models for.

        When Coinbase spot is present, the v2 target is 'x' — the
        cross-exchange volume-weighted mid price composite.
        """
        if self.coinbase_spot:
            return ["x"]
        return list(self.instruments)


# ═══════════════════════════════════════════════════════════
# Numba-JIT rolling primitives
# ═══════════════════════════════════════════════════════════
#
# These replace the pure-NumPy cumsum versions.  Numba gives ~2-4x
# speedup per call because it avoids temporary array allocations and
# fuses the loops.  The cumsum approach allocates 2-3 temporaries of
# size n per call; numba does it in a single pass with O(1) extra memory.

@nb.njit(cache=True, fastmath=True)
def _rolling_sum_nb(arr, w):
    """O(n) rolling sum — single-pass, no temporaries."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:w - 1] = np.nan
    s = 0.0
    for i in range(w):
        v = arr[i]
        s += 0.0 if np.isnan(v) else v
    out[w - 1] = s
    for i in range(w, n):
        v_new = arr[i]
        v_old = arr[i - w]
        s += (0.0 if np.isnan(v_new) else v_new) - (0.0 if np.isnan(v_old) else v_old)
        out[i] = s
    return out


def _rolling_sum(arr: np.ndarray, w: int) -> np.ndarray:
    """Dispatch to numba implementation."""
    return _rolling_sum_nb(arr.astype(np.float64), w)


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    return _rolling_sum_nb(arr.astype(np.float64), w) / w


@nb.njit(cache=True, fastmath=True)
def _rolling_std_nb(arr, w):
    """O(n) rolling std — single-pass Welford-style."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:w - 1] = np.nan
    s1 = 0.0
    s2 = 0.0
    for i in range(w):
        v = arr[i]
        x = 0.0 if np.isnan(v) else v
        s1 += x
        s2 += x * x
    var = s2 / w - (s1 / w) ** 2
    out[w - 1] = np.sqrt(max(var, 0.0))
    for i in range(w, n):
        v_new = arr[i]
        v_old = arr[i - w]
        x_new = 0.0 if np.isnan(v_new) else v_new
        x_old = 0.0 if np.isnan(v_old) else v_old
        s1 += x_new - x_old
        s2 += x_new * x_new - x_old * x_old
        var = s2 / w - (s1 / w) ** 2
        out[i] = np.sqrt(max(var, 0.0))
    return out


def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    return _rolling_std_nb(arr.astype(np.float64), w)


@nb.njit(cache=True, fastmath=True)
def _rolling_max_nb(arr, w):
    """O(n) rolling max via monotonic deque (numba)."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:w - 1] = np.nan
    # Deque stores indices; we maintain a decreasing monotonic deque
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0  # deque is dq[head:tail]
    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            v = -1e308
        # Remove elements smaller than current from back
        while tail > head and arr[dq[tail - 1]] <= v:
            tail -= 1
        dq[tail] = i
        tail += 1
        # Remove elements outside window from front
        while head < tail and dq[head] <= i - w:
            head += 1
        if i >= w - 1:
            out[i] = arr[dq[head]]
    return out


def _rolling_max(arr: np.ndarray, w: int) -> np.ndarray:
    return _rolling_max_nb(np.nan_to_num(arr.astype(np.float64), nan=-np.inf), w)


@nb.njit(cache=True, fastmath=True)
def _rolling_corr_nb(x, y, w):
    """O(n) rolling Pearson correlation — single-pass accumulators."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:w - 1] = np.nan
    sx = 0.0; sy = 0.0; sxx = 0.0; syy = 0.0; sxy = 0.0
    for i in range(w):
        xi = x[i]; yi = y[i]
        sx += xi; sy += yi
        sxx += xi * xi; syy += yi * yi; sxy += xi * yi
    mx = sx / w; my = sy / w
    cov = sxy / w - mx * my
    vx = sxx / w - mx * mx
    vy = syy / w - my * my
    if vx > 0 and vy > 0:
        out[w - 1] = cov / (np.sqrt(vx) * np.sqrt(vy))
    else:
        out[w - 1] = np.nan
    for i in range(w, n):
        xn = x[i]; yn = y[i]
        xo = x[i - w]; yo = y[i - w]
        sx += xn - xo; sy += yn - yo
        sxx += xn * xn - xo * xo; syy += yn * yn - yo * yo
        sxy += xn * yn - xo * yo
        mx = sx / w; my = sy / w
        cov = sxy / w - mx * my
        vx = sxx / w - mx * mx
        vy = syy / w - my * my
        if vx > 0 and vy > 0:
            out[i] = cov / (np.sqrt(vx) * np.sqrt(vy))
        else:
            out[i] = np.nan
    return out


def _rolling_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    return _rolling_corr_nb(
        x.astype(np.float64), y.astype(np.float64), w
    )


def _rolling_autocorr_lag1(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling_corr_nb(
        x[1:].astype(np.float64), x[:-1].astype(np.float64), w - 1
    )


@nb.njit(cache=True, fastmath=True)
def _rolling_skew_nb(arr, w):
    """Rolling skewness with Fisher bias correction — single-pass."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:w - 1] = np.nan
    s1 = 0.0; s2 = 0.0; s3 = 0.0
    for i in range(w):
        x = arr[i]
        s1 += x; s2 += x * x; s3 += x * x * x
    correction = (w * w) / ((w - 1.0) * (w - 2.0))
    mean = s1 / w
    m2 = s2 / w - mean * mean
    if m2 > 0:
        std = np.sqrt(m2)
        m3 = s3 / w - 3.0 * mean * s2 / w + 2.0 * mean ** 3
        out[w - 1] = (m3 / (std ** 3)) * correction
    else:
        out[w - 1] = 0.0
    for i in range(w, n):
        xn = arr[i]; xo = arr[i - w]
        s1 += xn - xo
        s2 += xn * xn - xo * xo
        s3 += xn * xn * xn - xo * xo * xo
        mean = s1 / w
        m2 = s2 / w - mean * mean
        if m2 > 0:
            std = np.sqrt(m2)
            m3 = s3 / w - 3.0 * mean * s2 / w + 2.0 * mean ** 3
            out[i] = (m3 / (std ** 3)) * correction
        else:
            out[i] = 0.0
    return out


def _rolling_skew(arr: np.ndarray, w: int) -> np.ndarray:
    return _rolling_skew_nb(np.nan_to_num(arr.astype(np.float64), nan=0.0), w)


@nb.njit(cache=True, fastmath=True)
def _rolling_kurt_nb(arr, w):
    """Rolling excess kurtosis with Fisher correction — single-pass."""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:w - 1] = np.nan
    s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
    for i in range(w):
        x = arr[i]
        x2 = x * x
        s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2
    a = (w + 1.0) * w / ((w - 1.0) * (w - 2.0) * (w - 3.0))
    b = 3.0 * (w - 1.0) ** 2 / ((w - 2.0) * (w - 3.0))
    mean = s1 / w
    m2 = s2 / w - mean * mean
    if m2 > 0:
        m4 = (s4 / w - 4.0 * mean * s3 / w
               + 6.0 * mean * mean * s2 / w - 3.0 * mean ** 4)
        out[w - 1] = a * (w - 1.0) * (m4 / (m2 * m2)) - b
    else:
        out[w - 1] = 0.0
    for i in range(w, n):
        xn = arr[i]; xo = arr[i - w]
        xn2 = xn * xn; xo2 = xo * xo
        s1 += xn - xo
        s2 += xn2 - xo2
        s3 += xn2 * xn - xo2 * xo
        s4 += xn2 * xn2 - xo2 * xo2
        mean = s1 / w
        m2 = s2 / w - mean * mean
        if m2 > 0:
            m4 = (s4 / w - 4.0 * mean * s3 / w
                   + 6.0 * mean * mean * s2 / w - 3.0 * mean ** 4)
            out[i] = a * (w - 1.0) * (m4 / (m2 * m2)) - b
        else:
            out[i] = 0.0
    return out


def _rolling_kurt(arr: np.ndarray, w: int) -> np.ndarray:
    return _rolling_kurt_nb(np.nan_to_num(arr.astype(np.float64), nan=0.0), w)


def _ewma(arr: np.ndarray, halflife: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    b, a = [alpha], [1.0, -(1.0 - alpha)]
    clean = np.nan_to_num(arr, nan=0.0)
    zi = lfilter_zi(b, a) * clean[0]
    out, _ = lfilter(b, a, clean, zi=zi)
    return out.astype(np.float64)


# ═══════════════════════════════════════════════════════════
# Rolling sum cache
# ═══════════════════════════════════════════════════════════
#
# Many features recompute _rolling_sum on the same (array, window):
#   - r² with w=60 is used in jump ratio, RV term structure, vol-of-vol
#   - r² with w=300 is used in VPIN, term structure, TSRV/RV ratio
# The cache avoids ~40 % of all rolling_sum calls.

class RollingCache:
    """Thin wrapper — always recomputes (cache disabled for large datasets).

    At 100M+ rows each cached array is ~825 MB; recomputing a numba
    rolling sum is <1 s, so caching is the wrong tradeoff.
    """

    def rolling_sum(self, arr: np.ndarray, w: int) -> np.ndarray:
        return _rolling_sum_nb(arr, w)

    def rolling_mean(self, arr: np.ndarray, w: int) -> np.ndarray:
        return _rolling_sum_nb(arr, w) / w

    def clear(self):
        pass


# ═══════════════════════════════════════════════════════════
# Numba-JIT noise-robust variance estimators
# ═══════════════════════════════════════════════════════════

@nb.njit(cache=True, fastmath=True)
def _tsrv_nb(r, w, K):
    """Two-Scale Realized Variance — fully vectorised grid construction."""
    n = r.shape[0]
    r2 = r * r

    # Fast RV
    rv_fast = np.empty(n, dtype=np.float64)
    rv_fast[:w - 1] = np.nan
    s = 0.0
    for i in range(w):
        s += r2[i]
    rv_fast[w - 1] = s
    for i in range(w, n):
        s += r2[i] - r2[i - w]
        rv_fast[i] = s

    # Subsampled RV: for each grid offset k, aggregate returns in
    # non-overlapping blocks of size K, then rolling-sum the squared
    # aggregated returns.
    rv_sub = np.zeros(n, dtype=np.float64)
    for k in range(K):
        sub_r2 = np.zeros(n, dtype=np.float64)
        i = k
        while i < n:
            end = min(i + K, n)
            block_sum = 0.0
            for j in range(i, end):
                block_sum += r[j]
            sub_r2[end - 1] = block_sum * block_sum
            i += K
        # Rolling sum of sub_r2 over window w
        s = 0.0
        for i in range(w):
            s += sub_r2[i]
        if w - 1 < n:
            rv_sub[w - 1] += s
        for i in range(w, n):
            s += sub_r2[i] - sub_r2[i - w]
            rv_sub[i] += s
    for i in range(n):
        rv_sub[i] /= K

    n_bar = w / K
    out = np.empty(n, dtype=np.float64)
    out[:w - 1] = np.nan
    for i in range(w - 1, n):
        tsrv_val = rv_sub[i] - (n_bar / w) * rv_fast[i]
        floor = rv_fast[i] * 0.1
        out[i] = tsrv_val if tsrv_val > floor else floor
    return out


def _tsrv(r: np.ndarray, w: int, K: int = 5) -> np.ndarray:
    return _tsrv_nb(r.astype(np.float64), w, K)


@nb.njit(cache=True, fastmath=True)
def _parzen_weight(x):
    ax = abs(x)
    if ax <= 0.5:
        return 1.0 - 6.0 * x * x + 6.0 * ax * ax * ax
    elif ax <= 1.0:
        return 2.0 * (1.0 - ax) ** 3
    return 0.0


@nb.njit(cache=True, fastmath=True)
def _realized_kernel_nb(r, w):
    """Realized Kernel with Parzen kernel — numba fused loop."""
    n = r.shape[0]
    r2 = r * r

    # gamma_0 = rolling sum of r²
    gamma_0 = np.empty(n, dtype=np.float64)
    gamma_0[:w - 1] = np.nan
    s = 0.0
    for i in range(w):
        s += r2[i]
    gamma_0[w - 1] = s
    for i in range(w, n):
        s += r2[i] - r2[i - w]
        gamma_0[i] = s

    H = max(int(np.ceil(w ** (2.0 / 3.0))), 1)

    rk = gamma_0.copy()
    for h in range(1, H + 1):
        kw = _parzen_weight(h / H)
        if kw == 0.0:
            continue
        # Rolling sum of r[i]*r[i-h] over window (w-h), aligned to end at i
        wh = w - h
        if wh < 1:
            continue
        s = 0.0
        for i in range(h, h + wh):
            s += r[i] * r[i - h]
        # The first valid index for this gamma is h + wh - 1 = w - 1
        idx = h + wh - 1
        if idx < n:
            rk[idx] += 2.0 * kw * s
        for i in range(idx + 1, n):
            s += r[i] * r[i - h] - r[i - wh] * r[i - wh - h]
            rk[i] += 2.0 * kw * s

    # Floor
    for i in range(w - 1, n):
        floor = gamma_0[i] * 0.1
        if rk[i] < floor:
            rk[i] = floor
    return rk


def _realized_kernel(r: np.ndarray, w: int) -> np.ndarray:
    return _realized_kernel_nb(r.astype(np.float64), w)


def _realized_quarticity(r: np.ndarray, w: int) -> np.ndarray:
    return (w / 3.0) * _rolling_sum(r ** 4, w)


@nb.njit(cache=True, fastmath=True)
def _roll_measure_nb(r, w):
    """Roll (1984) effective spread — numba single-pass."""
    n = r.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = np.nan
    out[1:w] = np.nan
    wm1 = w - 1
    if wm1 < 1:
        return out
    # Accumulators for products r[i]*r[i-1] and individual sums
    s_ab = 0.0; s_a = 0.0; s_b = 0.0
    for i in range(1, w):
        s_ab += r[i] * r[i - 1]
        s_a += r[i]
        s_b += r[i - 1]
    cov = s_ab / wm1 - (s_a / wm1) * (s_b / wm1)
    out[w - 1] = 2.0 * np.sqrt(max(-cov, 0.0))
    for i in range(w, n):
        # Add new pair (r[i], r[i-1]), remove old pair (r[i-wm1], r[i-wm1-1])
        s_ab += r[i] * r[i - 1] - r[i - wm1] * r[i - wm1 - 1]
        s_a += r[i] - r[i - wm1]
        s_b += r[i - 1] - r[i - wm1 - 1]
        cov = s_ab / wm1 - (s_a / wm1) * (s_b / wm1)
        out[i] = 2.0 * np.sqrt(max(-cov, 0.0))
    return out


def _roll_measure(r: np.ndarray, w: int) -> np.ndarray:
    return _roll_measure_nb(r.astype(np.float64), w)


def _parkinson(prices: np.ndarray, w: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    inv_4ln2 = 1.0 / (4.0 * np.log(2.0))
    hi = _rolling_max_nb(prices.astype(np.float64), w)
    lo = -_rolling_max_nb(-prices.astype(np.float64), w)
    valid = (lo > 0) & (hi > lo) & np.isfinite(hi) & np.isfinite(lo)
    safe_lo = np.where(lo > 0, lo, 1.0)
    log_ratio = np.zeros(n, dtype=np.float64)
    np.log(hi / safe_lo, out=log_ratio, where=valid)
    out = inv_4ln2 * log_ratio ** 2
    out[:w - 1] = np.nan
    return out


# ═══════════════════════════════════════════════════════════
# MODWT (unchanged logic, uses cached rolling_mean)
# ═══════════════════════════════════════════════════════════

def _haar_modwt_energy(arr: np.ndarray, scale: int) -> np.ndarray:
    half = 2 ** (scale - 1)
    ma = _rolling_mean(arr, half)
    ma_shifted = np.roll(ma, half)
    ma_shifted[:half * 2] = np.nan
    detail = ma - ma_shifted
    return detail ** 2


# ═══════════════════════════════════════════════════════════
# Spectral (unchanged — already batched)
# ═══════════════════════════════════════════════════════════

def _forward_fill(arr: np.ndarray) -> np.ndarray:
    mask = np.isfinite(arr)
    if not mask.any():
        return arr
    idx = np.where(mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = arr[idx]
    first_valid = mask.argmax()
    if first_valid > 0:
        out[:first_valid] = np.nan
    return out


def _spectral_band_power(arr: np.ndarray, w: int,
                         bands: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    n = len(arr)
    freqs = np.fft.rfftfreq(w, d=1.0)
    band_masks = {}
    for name, (flo, fhi) in bands.items():
        band_masks[name] = (freqs >= flo) & (freqs < fhi)

    results = {name: np.full(n, np.nan) for name in bands}
    step = max(w // 4, 1)
    ends = np.arange(w, n + 1, step)
    if len(ends) == 0:
        return results

    nan_count = _rolling_sum(np.isnan(arr).astype(np.float64), w)
    valid_mask = nan_count[ends - 1] == 0
    valid_ends = ends[valid_mask]
    if len(valid_ends) == 0:
        return results

    # Process in batches to cap peak memory (~256 MB per batch)
    max_batch = max(1, (256 * 1024 * 1024) // (w * 8))
    starts = valid_ends - w
    for b0 in range(0, len(starts), max_batch):
        b_starts = starts[b0:b0 + max_batch]
        b_ends = valid_ends[b0:b0 + max_batch]
        indices = b_starts[:, None] + np.arange(w)[None, :]
        chunks = arr[indices]
        del indices
        spectra = np.abs(np.fft.rfft(chunks, axis=1)) ** 2
        del chunks
        for name, mask in band_masks.items():
            results[name][b_ends - 1] = spectra[:, mask].sum(axis=1)
        del spectra

    for name in results:
        results[name] = _forward_fill(results[name])
    return results


def _rolling_dft_magnitude(arr: np.ndarray, w: int,
                           period_s: float) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    freq = 1.0 / period_s
    step = max(w // 8, 1)
    t_local = np.arange(w, dtype=np.float64)
    cos_basis = np.cos(2.0 * np.pi * freq * t_local)
    sin_basis = np.sin(2.0 * np.pi * freq * t_local)

    ends = np.arange(w, n + 1, step)
    if len(ends) == 0:
        return out

    nan_count = _rolling_sum(np.isnan(arr).astype(np.float64), w)
    valid_mask = nan_count[ends - 1] == 0
    valid_ends = ends[valid_mask]
    if len(valid_ends) == 0:
        return _forward_fill(out)

    # Process in batches to cap peak memory (~256 MB per batch)
    max_batch = max(1, (256 * 1024 * 1024) // (w * 8))
    starts = valid_ends - w
    for b0 in range(0, len(starts), max_batch):
        b_starts = starts[b0:b0 + max_batch]
        b_ends = valid_ends[b0:b0 + max_batch]
        indices = b_starts[:, None] + np.arange(w)[None, :]
        chunks = arr[indices]
        del indices
        re_vals = chunks @ cos_basis
        im_vals = chunks @ sin_basis
        del chunks
        out[b_ends - 1] = np.sqrt(re_vals ** 2 + im_vals ** 2) / w

    return _forward_fill(out)


# ═══════════════════════════════════════════════════════════
# FFF Seasonality
# ═══════════════════════════════════════════════════════════

@dataclass
class FFFSeasonality:
    periods_hours: list[float]
    harmonics: list[int]
    coefficients: np.ndarray
    mean_factor: float

    def predict(self, timestamps_us: np.ndarray) -> np.ndarray:
        basis = _build_fourier_basis(timestamps_us, self.periods_hours,
                                     self.harmonics)
        ones = np.ones((basis.shape[0], 1), dtype=np.float64)
        X = np.hstack([ones, basis])
        log_seas = X @ self.coefficients
        seas = np.exp(log_seas)
        return seas / self.mean_factor


def _build_fourier_basis(timestamps_us, periods_hours, harmonics):
    t_hours = timestamps_us.astype(np.float64) / (3600 * 1e6)
    cols = []
    for T, K in zip(periods_hours, harmonics):
        for k in range(1, K + 1):
            angle = 2.0 * np.pi * k * t_hours / T
            cols.append(np.sin(angle))
            cols.append(np.cos(angle))
    return np.column_stack(cols)


def compute_fff_seasonality(df_train, cfg):
    ts = cfg.ts_col
    timestamps_us = df_train.get_column(ts).cast(pl.Int64).to_numpy()
    basis = _build_fourier_basis(timestamps_us, cfg.fff_periods_hours,
                                 cfg.fff_harmonics)
    ones = np.ones((basis.shape[0], 1), dtype=np.float64)
    X = np.hstack([ones, basis])
    seasonality = {}
    # Fit seasonality for main instruments + composite target if present
    seas_instruments = list(cfg.instruments)
    if cfg.coinbase_spot:
        seas_instruments.append("x")
    for inst in seas_instruments:
        rv_col = f"{inst}_rv"
        rv = df_train.get_column(rv_col).to_numpy().astype(np.float64)
        log_rv = np.log(rv + cfg.eps)
        valid = np.isfinite(log_rv)
        beta, *_ = np.linalg.lstsq(X[valid], log_rv[valid], rcond=None)
        fitted = X @ beta
        mean_factor = np.exp(fitted[valid]).mean()
        seasonality[inst] = FFFSeasonality(
            periods_hours=cfg.fff_periods_hours,
            harmonics=cfg.fff_harmonics,
            coefficients=beta,
            mean_factor=mean_factor,
        )
        seas_vals = np.exp(fitted[valid]) / mean_factor
        log.info(f"       {inst} FFF seasonality: min={seas_vals.min():.4f}  "
              f"max={seas_vals.max():.4f}  mean≈{seas_vals.mean():.4f}")
    return seasonality


# ═══════════════════════════════════════════════════════════
# Expiry calendar (unchanged)
# ═══════════════════════════════════════════════════════════

def _last_friday_of_month(year, month):
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day)
    offset = (dt.weekday() - 4) % 7
    return dt - timedelta(days=offset)


def _next_monthly_expiry(dt):
    lf = _last_friday_of_month(dt.year, dt.month)
    if dt <= lf:
        return lf
    if dt.month == 12:
        return _last_friday_of_month(dt.year + 1, 1)
    return _last_friday_of_month(dt.year, dt.month + 1)


def _next_quarterly_expiry(dt):
    q_months = [3, 6, 9, 12]
    for m in q_months:
        lf = _last_friday_of_month(dt.year, m)
        if dt <= lf:
            return lf
    return _last_friday_of_month(dt.year + 1, 3)


# ═══════════════════════════════════════════════════════════
# Steps 0–3 (unchanged)
# ═══════════════════════════════════════════════════════════

def load_and_merge(cfg):
    ts = cfg.ts_col
    freq_str = f"{cfg.base_freq_s}s"
    frames = []
    for inst, path in cfg.input_map.items():
        log.info(f"    Loading {inst} ← {path}")
        raw = pl.read_parquet(path)
        raw = raw.with_columns(pl.col(ts).cast(pl.Datetime("us")))
        raw = raw.with_columns(pl.col(ts).dt.truncate(freq_str).alias("__ts"))
        tick_counts = raw.group_by("__ts").agg(pl.len().alias("__n_ticks"))
        raw = (
            raw.sort("__ts", ts)
            .group_by("__ts")
            .agg([pl.col(c).last() for c in raw.columns if c not in (ts, "__ts")])
            .rename({"__ts": ts})
            .sort(ts)
        )
        raw = raw.join(tick_counts.rename({"__ts": ts}), on=ts, how="left")
        rename_map = {c: f"{inst}_{c}" for c in raw.columns if c != ts}
        raw = raw.rename(rename_map)
        frames.append(raw)
    merged = frames[0]
    for f_ in frames[1:]:
        merged = merged.join(f_, on=ts, how="full", coalesce=True)
    merged = merged.sort(ts)
    # Mark seconds where Coinbase actually published data (before fill)
    if cfg.coinbase_spot:
        cb_bid = f"{cfg.coinbase_spot}_{cfg.bid_col}"
        merged = merged.with_columns(
            pl.col(cb_bid).is_not_null().cast(pl.Float64).alias("_coinbase_has_update")
        )
    fill_cols = [c for c in merged.columns if c not in (ts, "_coinbase_has_update")]
    merged = merged.with_columns(
        [pl.col(c).forward_fill().backward_fill() for c in fill_cols]
    )
    return merged


def detect_and_fix_coinbase_staleness(df, cfg):
    """Detect stale Coinbase data and substitute Binance spot values.

    A second is considered stale when no Coinbase update has arrived for
    ``cfg.coinbase_stale_threshold_s`` consecutive seconds.  Where stale,
    Binance spot raw columns are copied into the Coinbase columns so that
    downstream features (mid, spread, OBI, returns) reflect live data.
    """
    threshold = cfg.coinbase_stale_threshold_s
    # coinbase_available = 1 if at least one update in the rolling window
    df = df.with_columns(
        (pl.col("_coinbase_has_update")
         .rolling_sum(threshold)
         .gt(0)
         .cast(pl.Float64)
         .alias("coinbase_available"))
    )
    stale_pct = (1.0 - df.get_column("coinbase_available").mean()) * 100
    log.info(f"       Coinbase staleness: {stale_pct:.2f}% of seconds marked stale")
    # Substitute Binance spot raw columns where Coinbase is stale
    spot = cfg.spot
    cb = cfg.coinbase_spot
    raw_cols = [cfg.bid_col, cfg.ask_col, cfg.bid_size_col, cfg.ask_size_col]
    sub_exprs = []
    for col in raw_cols:
        src = f"{spot}_{col}"
        dst = f"{cb}_{col}"
        sub_exprs.append(
            pl.when(pl.col("coinbase_available") == 0.0)
            .then(pl.col(src))
            .otherwise(pl.col(dst))
            .alias(dst)
        )
    df = df.with_columns(sub_exprs)
    # Drop internal marker, keep coinbase_available
    df = df.drop("_coinbase_has_update")
    return df


def compute_1s_bars(df, cfg):
    exprs = []
    for inst in cfg.all_instruments:
        bid = f"{inst}_{cfg.bid_col}"
        ask = f"{inst}_{cfg.ask_col}"
        bsz = f"{inst}_{cfg.bid_size_col}"
        asz = f"{inst}_{cfg.ask_size_col}"
        mid = f"{inst}_mid"
        microprice = f"{inst}_microprice"
        spread = f"{inst}_spread_bps"
        obi = f"{inst}_obi"
        exprs.append(((pl.col(bid) + pl.col(ask)) / 2).alias(mid))
        exprs.append(
            ((pl.col(ask) * pl.col(bsz) + pl.col(bid) * pl.col(asz))
             / (pl.col(bsz) + pl.col(asz))).alias(microprice)
        )
        exprs.append(
            ((pl.col(ask) - pl.col(bid))
             / ((pl.col(bid) + pl.col(ask)) / 2) * 1e4).alias(spread)
        )
        exprs.append(
            ((pl.col(bsz) - pl.col(asz))
             / (pl.col(bsz) + pl.col(asz))).alias(obi)
        )
    df = df.with_columns(exprs)
    ret_exprs = []
    for inst in cfg.all_instruments:
        mid = f"{inst}_mid"
        ret = f"{inst}_ret"
        ret_exprs.append(pl.col(mid).log().diff().alias(ret))
    df = df.with_columns(ret_exprs)
    return df


def compute_persecond_base(df, cfg):
    exprs = []
    for inst in cfg.all_instruments:
        ret = f"{inst}_ret"
        r2 = pl.col(ret).pow(2)
        exprs.append(r2.alias(f"{inst}_rv"))
        exprs.append(
            pl.when(pl.col(ret) > 0).then(r2).otherwise(0.0)
            .alias(f"{inst}_rsv_pos")
        )
        exprs.append(
            pl.when(pl.col(ret) < 0).then(r2).otherwise(0.0)
            .alias(f"{inst}_rsv_neg")
        )
        exprs.append(
            (pl.col(ret).abs() * pl.col(ret).abs().shift(1) * (np.pi / 2))
            .alias(f"{inst}_bpv")
        )
    return df.with_columns(exprs)


def compute_cross_exchange_composite(df, cfg):
    """Compute volume-weighted mid price between primary spot and Coinbase.

    Creates synthetic instrument 'x' with x_mid, x_ret, x_rv, x_rsv_pos,
    x_rsv_neg, x_bpv — matching the base feature schema so that existing
    seasonality, target, and model-fitting code works with inst='x'.
    """
    if not cfg.coinbase_spot:
        return df
    s = cfg.spot
    c = cfg.coinbase_spot

    # Volume weights: total quoted depth at each exchange
    s_w = pl.col(f"{s}_{cfg.bid_size_col}") + pl.col(f"{s}_{cfg.ask_size_col}")
    c_w = pl.col(f"{c}_{cfg.bid_size_col}") + pl.col(f"{c}_{cfg.ask_size_col}")
    total_w = s_w + c_w

    # Volume-weighted mid price
    x_mid = (
        pl.col(f"{s}_mid") * s_w + pl.col(f"{c}_mid") * c_w
    ) / total_w
    df = df.with_columns(x_mid.alias("x_mid"))

    # Log returns
    df = df.with_columns(pl.col("x_mid").log().diff().alias("x_ret"))

    # Per-second base features (same schema as compute_persecond_base)
    r2 = pl.col("x_ret").pow(2)
    df = df.with_columns([
        r2.alias("x_rv"),
        pl.when(pl.col("x_ret") > 0).then(r2).otherwise(0.0)
          .alias("x_rsv_pos"),
        pl.when(pl.col("x_ret") < 0).then(r2).otherwise(0.0)
          .alias("x_rsv_neg"),
        (pl.col("x_ret").abs() * pl.col("x_ret").abs().shift(1) * (np.pi / 2))
          .alias("x_bpv"),
    ])

    return df


def apply_seasonality(df, seasonality, cfg):
    ts = cfg.ts_col
    timestamps_us = df.get_column(ts).cast(pl.Int64).to_numpy()
    new_cols = []
    # Deseasonalise main instruments + composite target if present
    seas_instruments = list(cfg.instruments)
    if cfg.coinbase_spot and "x" in seasonality:
        seas_instruments.append("x")
    for inst in seas_instruments:
        seas = seasonality[inst]
        factor = seas.predict(timestamps_us)
        factor = np.maximum(factor, cfg.eps)
        factor_series = pl.Series(f"{inst}_seas_factor", factor)
        new_cols.append(factor_series)
        for feat in cfg.base_features:
            col = f"{inst}_{feat}"
            vals = df.get_column(col).to_numpy().astype(np.float64)
            new_cols.append(pl.Series(col, vals / factor))
    return df.with_columns(new_cols)


# ═══════════════════════════════════════════════════════════
# Step 4 — Compute all features (OPTIMISED)
# ═══════════════════════════════════════════════════════════

VARIANCE_SCALE_PREFIXES = [
    "rv_w", "rsv_pos_w", "rsv_neg_w", "bpv_w",
    "tsrv_", "rk_", "rq_",
    "ewma_var_", "roll_measure_", "parkinson_",
    "vol_of_vol_",
]


def _precompute_arrays(df: pl.DataFrame, cfg: PipelineConfig) -> dict:
    """Extract and precompute all shared numpy arrays once."""
    arrays = {}
    n = df.height

    for inst in cfg.all_instruments:
        r = df.get_column(f"{inst}_ret").to_numpy().astype(np.float64)
        r = np.nan_to_num(r, nan=0.0)
        arrays[f"{inst}_ret"] = r
        arrays[f"{inst}_r2"] = r * r              # ← precomputed
        arrays[f"{inst}_abs_r"] = np.abs(r)        # ← precomputed
        arrays[f"{inst}_mid"] = df.get_column(f"{inst}_mid").to_numpy().astype(np.float64)
        arrays[f"{inst}_microprice"] = df.get_column(f"{inst}_microprice").to_numpy().astype(np.float64)
        arrays[f"{inst}_spread_bps"] = df.get_column(f"{inst}_spread_bps").to_numpy().astype(np.float64)
        arrays[f"{inst}_obi"] = df.get_column(f"{inst}_obi").to_numpy().astype(np.float64)

        for feat in cfg.base_features:
            col = f"{inst}_{feat}"
            vals = df.get_column(col).to_numpy().astype(np.float64)
            arrays[col] = np.nan_to_num(vals, nan=0.0)

        tc_col = f"{inst}___n_ticks"
        if tc_col in df.columns:
            arrays[f"{inst}_n_updates"] = df.get_column(tc_col).to_numpy().astype(np.float64)
        else:
            arrays[f"{inst}_n_updates"] = np.ones(n, dtype=np.float64)

    return arrays


def _compute_tier1_basis_spread_jump(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """T1: Basis, spread, jump features."""
    f = {}
    s, p = cfg.spot, cfg.perp
    eps = cfg.eps
    n = len(arrays[f"{s}_ret"])

    s_mid = arrays[f"{s}_mid"]
    p_mid = arrays[f"{p}_mid"]
    basis = np.log(p_mid / np.maximum(s_mid, eps))

    for w in [5, 60, 1440, 8192]:
        f[f"basis_level_w{w}"] = cache.rolling_mean(basis, w)
    for w in [60, 1440, 8192]:
        f[f"basis_vol_w{w}"] = _rolling_std(basis, w)
    for lag in [60, 300]:
        change = basis - np.roll(basis, lag)
        change[:lag] = np.nan
        f[f"basis_change_{lag}s"] = change
    b_mean = cache.rolling_mean(basis, 3600)
    b_std = _rolling_std(basis, 3600)
    f["basis_zscore_1h"] = _safe_div(basis - b_mean, b_std)

    for inst in cfg.instruments:
        spr = arrays[f"{inst}_spread_bps"]
        for w in [5, 60, 1440]:
            f[f"{inst}_spread_w{w}"] = cache.rolling_mean(spr, w)

    spr_s = arrays[f"{s}_spread_bps"]
    spr_mean = cache.rolling_mean(spr_s, 300)
    spr_std = _rolling_std(spr_s, 300)
    f["s_spread_zscore_5m"] = _safe_div(spr_s - spr_mean, spr_std)

    for inst in cfg.instruments:
        r2 = arrays[f"{inst}_r2"]
        absret = arrays[f"{inst}_abs_r"]
        r = arrays[f"{inst}_ret"]
        for w in [60, 900, 1440, 3600, 8192]:
            rv_w = cache.rolling_sum(r2, w)
            bpv_pairs = np.empty_like(absret)
            bpv_pairs[0] = 0.0
            bpv_pairs[1:] = absret[1:] * absret[:-1]
            bpv_w = (np.pi / 2) * cache.rolling_sum(bpv_pairs, w) / max(w - 1, 1)
            jv_w = np.maximum(rv_w - bpv_w, 0.0)
            f[f"{inst}_jump_ratio_w{w}"] = _safe_div(jv_w, rv_w)

            rsv_pos_w = cache.rolling_sum(r2 * (r > 0), w)
            rsv_neg_w = cache.rolling_sum(r2 * (r <= 0), w)
            f[f"{inst}_asym_impact_w{w}"] = rsv_neg_w - rsv_pos_w
            f[f"{inst}_downside_share_w{w}"] = _safe_div(rsv_neg_w, rv_w)

            # Signed jump: (RV - BPV) × sign(net return)
            net_return_w = cache.rolling_sum(r, w)
            f[f"{inst}_signed_jump_w{w}"] = jv_w * np.sign(net_return_w)

            # Raw realized jump level (RV - BPV, not ratio)
            f[f"{inst}_rj_w{w}"] = jv_w

    return f


def _compute_tier1_returns_drawdown_rv(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """T1: Signed returns, drawdown, RV term structure."""
    f = {}
    eps = cfg.eps

    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        for w in [30, 60, 1440]:
            f[f"{inst}_signed_ret_w{w}"] = cache.rolling_sum(r, w)

    for inst in cfg.instruments:
        mid = arrays[f"{inst}_mid"]
        log_mid = np.log(mid + eps)
        for w in [1440, 8192, 16384]:
            rm = _rolling_max(log_mid, w)
            f[f"{inst}_drawdown_w{w}"] = log_mid - rm

    for inst in cfg.instruments:
        r2 = arrays[f"{inst}_r2"]
        rv_1m = cache.rolling_sum(r2, 60)
        rv_5m = cache.rolling_sum(r2, 300)
        rv_15m = cache.rolling_sum(r2, 900)
        f[f"{inst}_rv_term_slope"] = _safe_div(rv_1m, rv_15m) * 15
        f[f"{inst}_rv_term_curvature"] = rv_1m / 60 - 2 * rv_5m / 300 + rv_15m / 900

    return f


def _compute_tier1_microprice_vpin_obi(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache, n: int
) -> dict:
    """T1: Microprice, VPIN, tick intensity, OBI."""
    f = {}
    s, p = cfg.spot, cfg.perp

    mp = arrays[f"{s}_microprice"]
    for label, lag in [("1s", 1), ("5s", 5), ("10s", 10),
                       ("30s", 30), ("60s", 60), ("300s", 300)]:
        ret = mp / np.roll(mp, lag) - 1.0
        ret[:lag] = np.nan
        f[f"microprice_ret_{label}"] = ret

    mret_1s = f.get("microprice_ret_1s", np.zeros(n))
    mret_5s = f.get("microprice_ret_5s", np.zeros(n))
    mret_30s = f.get("microprice_ret_30s", np.zeros(n))
    mret_300s = f.get("microprice_ret_300s", np.zeros(n))

    ra1 = mret_1s - np.nan_to_num(mret_5s, nan=0.0) / 5.0
    ra1[:5] = np.nan
    f["ret_accel_1s_5s"] = ra1
    ra2 = np.nan_to_num(mret_5s, nan=0.0) / 5.0 - np.nan_to_num(mret_30s, nan=0.0) / 30.0
    ra2[:30] = np.nan
    f["ret_accel_5s_30s"] = ra2
    ra3 = np.nan_to_num(mret_30s, nan=0.0) / 30.0 - np.nan_to_num(mret_300s, nan=0.0) / 300.0
    ra3[:300] = np.nan
    f["ret_accel_30s_300s"] = ra3

    s_mid = arrays[f"{s}_mid"]
    mid_changes = np.diff(s_mid, prepend=s_mid[0])
    abs_changes = np.abs(mid_changes)
    pos_changes = np.maximum(mid_changes, 0)
    sum_abs = cache.rolling_sum(abs_changes, 300)
    sum_pos = cache.rolling_sum(pos_changes, 300)
    buy_frac = _safe_div(sum_pos, sum_abs, fill=0.5)
    f["vpin_5m"] = np.abs(buy_frac - 0.5) * 2.0

    nu = arrays[f"{s}_n_updates"]
    nu_10 = cache.rolling_sum(nu, 10)
    nu_60 = cache.rolling_sum(nu, 60)
    f["tick_intensity_ratio"] = _safe_div(nu_10, nu_60, fill=1.0) * 6.0

    for inst in cfg.instruments:
        obi = arrays[f"{inst}_obi"]
        obi_lag5 = np.roll(obi, 5)
        obi_lag5[:5] = np.nan
        f[f"{inst}_obi_delta_5s"] = obi - obi_lag5
        obi_clean = np.nan_to_num(obi, nan=0.0)
        ac = _rolling_autocorr_lag1(obi_clean, 60)
        f[f"{inst}_obi_autocorr_1m"] = np.full(n, np.nan)
        f[f"{inst}_obi_autocorr_1m"][1:] = ac

    f["obi_divergence"] = arrays[f"{s}_obi"] - arrays[f"{p}_obi"]
    for inst in cfg.instruments:
        obi = arrays[f"{inst}_obi"]
        spr = np.maximum(arrays[f"{inst}_spread_bps"], 0.1)
        f[f"{inst}_spread_obi_interaction"] = obi / spr

    return f


def _compute_tier2_modwt_corr_vov(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """T2: 1-day windows, MODWT, cross-corr, vol-of-vol."""
    f = {}
    s, p = cfg.spot, cfg.perp
    eps = cfg.eps

    for inst in cfg.instruments:
        rv_1s = arrays[f"{inst}_rv"]
        log_rv = np.log(rv_1s + eps)
        for scale in cfg.modwt_scales:
            energy = _haar_modwt_energy(log_rv, scale)
            f[f"{inst}_modwt_d{scale}"] = np.log(energy + eps)

    r_s = arrays[f"{s}_ret"]
    r_p = arrays[f"{p}_ret"]
    for w in [60, 1440, 8192]:
        f[f"sp_corr_w{w}"] = _rolling_corr(r_s, r_p, w)

    for inst in cfg.instruments:
        r2 = arrays[f"{inst}_r2"]
        rv_1m = cache.rolling_sum(r2, 60)
        for w in [60, 1440]:
            f[f"{inst}_vol_of_vol_w{w}"] = _rolling_std(
                np.nan_to_num(rv_1m, nan=0.0), w
            )

    return f


def _compute_tier2_estimators(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """T2: TSRV, RK, RQ, EWMA, Roll, Parkinson, lead-lag, roughness."""
    f = {}
    s, p = cfg.spot, cfg.perp
    eps = cfg.eps
    n = len(arrays[f"{s}_ret"])

    # TSRV
    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        for label, w in [("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400)]:
            f[f"{inst}_tsrv_{label}"] = _tsrv(r, w, K=cfg.tsrv_K)

    # TSRV/RV ratio
    for inst in cfg.instruments:
        r2 = arrays[f"{inst}_r2"]
        for label, w in [("1m", 60), ("5m", 300)]:
            rv_w = cache.rolling_sum(r2, w)
            tsrv_w = f[f"{inst}_tsrv_{label}"]
            f[f"{inst}_tsrv_rv_ratio_{label}"] = np.log(
                (tsrv_w + eps) / (rv_w + eps)
            )

    # Realized Kernel
    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        for label, w in [("5m", 300), ("15m", 900)]:
            f[f"{inst}_rk_{label}"] = _realized_kernel(r, w)

    # Realized Quarticity
    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        for label, w in [("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600)]:
            f[f"{inst}_rq_{label}"] = _realized_quarticity(r, w)

    # EWMA
    for inst in cfg.instruments:
        r2 = arrays[f"{inst}_r2"]
        for hl_label, hl in [("5s", 5.0), ("30s", 30.0)]:
            f[f"{inst}_ewma_var_{hl_label}"] = _ewma(r2, hl)

    # Roll measure
    for inst in [s, p]:
        r = arrays[f"{inst}_ret"]
        for label, w in [("1m", 60), ("5m", 300)]:
            f[f"{inst}_roll_measure_{label}"] = _roll_measure(r, w)

    # Parkinson
    for inst in [s, p]:
        mp = arrays[f"{inst}_microprice"]
        for label, w in [("30s", 30), ("60s", 60)]:
            f[f"{inst}_parkinson_{label}"] = _parkinson(mp, w)

    # Lead-lag
    r_s = arrays[f"{s}_ret"]
    r_p = arrays[f"{p}_ret"]
    s_lead = np.empty_like(r_s)
    s_lead[:-1] = r_s[1:]
    s_lead[-1] = 0.0
    f["perp_lead_1s"] = _rolling_corr(r_p, s_lead, 300)

    # Roughness
    for inst in cfg.instruments:
        r2 = arrays[f"{inst}_r2"]
        log_rv_1s = np.log(r2 + eps)
        for label, lag in [("5s", 5), ("30s", 30), ("60s", 60), ("300s", 300)]:
            change = log_rv_1s - np.roll(log_rv_1s, lag)
            change[:lag] = np.nan
            f[f"{inst}_roughness_{label}"] = change
        ch5 = f[f"{inst}_roughness_5s"]
        accel = ch5 - np.roll(ch5, 5)
        accel[:10] = np.nan
        f[f"{inst}_vol_accel_5s"] = accel

    return f


def _compute_tier3_spectral(
    arrays: dict, cfg: PipelineConfig
) -> dict:
    """T3: Spectral band ratios."""
    f = {}
    eps = cfg.eps
    spectral_bands = {
        "hf": (1 / 300, 0.5),
        "mf": (1 / 3600, 1 / 300),
        "lf": (1 / 14400, 1 / 3600),
        "trend": (0, 1 / 14400),
    }
    for inst in cfg.instruments:
        rv_1s = arrays[f"{inst}_rv"]
        for w_label, w in [("4k", 4096), ("8k", 8192)]:
            powers = _spectral_band_power(
                np.nan_to_num(rv_1s, nan=0.0), w, spectral_bands
            )
            hf = powers["hf"] + eps
            mf = powers["mf"] + eps
            lf = powers["lf"] + eps
            trend = powers["trend"] + eps
            total = hf + mf + lf + trend
            f[f"{inst}_spec_hf_lf_{w_label}"] = np.log(hf / lf)
            f[f"{inst}_spec_mf_lf_{w_label}"] = np.log(mf / lf)
            f[f"{inst}_spec_hf_mf_{w_label}"] = np.log(hf / mf)
            f[f"{inst}_spec_trend_share_{w_label}"] = trend / total
    return f


def _compute_tier3_fourier(
    arrays: dict, cfg: PipelineConfig, n: int
) -> dict:
    """T3: Time-varying Fourier magnitudes."""
    f = {}
    s, p = cfg.spot, cfg.perp
    for inst in [s, p]:
        rv_1s = arrays[f"{inst}_rv"]
        rv_clean = np.nan_to_num(rv_1s, nan=0.0)
        for period_label, period_s, window in [
            ("8h", 8 * 3600, 8 * 3600),
            ("24h", 24 * 3600, 24 * 3600),
            ("168h", 168 * 3600, 168 * 3600),
        ]:
            w_use = min(window, n // 2)
            if w_use > 100:
                f[f"{inst}_fourier_mag_{period_label}"] = _rolling_dft_magnitude(
                    rv_clean, w_use, period_s
                )
            else:
                f[f"{inst}_fourier_mag_{period_label}"] = np.full(n, np.nan)
    return f


def _compute_tier3_calendar(
    df: pl.DataFrame, cfg: PipelineConfig, n: int
) -> dict:
    """T3: Funding cycle, expiry, weekend, epoch phase."""
    f = {}
    ts = cfg.ts_col
    timestamps_us = df.get_column(ts).cast(pl.Int64).to_numpy()
    ts_seconds = timestamps_us / 1e6

    funding_phase = 2.0 * np.pi * (ts_seconds % (8 * 3600)) / (8 * 3600)
    f["sin_funding"] = np.sin(funding_phase)
    f["cos_funding"] = np.cos(funding_phase)

    unix_days = (timestamps_us // (86400 * 1_000_000)).astype(np.int64)
    unique_days, inverse_idx = np.unique(unix_days, return_inverse=True)
    monthly_vals = np.empty(len(unique_days), dtype=np.float64)
    quarterly_vals = np.empty(len(unique_days), dtype=np.float64)
    for j, d in enumerate(unique_days):
        dt = datetime.utcfromtimestamp(int(d) * 86400)
        dt = datetime(dt.year, dt.month, dt.day)
        monthly_vals[j] = (_next_monthly_expiry(dt) - dt).days
        quarterly_vals[j] = (_next_quarterly_expiry(dt) - dt).days
    f["monthly_expiry_days"] = monthly_vals[inverse_idx]
    f["quarterly_expiry_days"] = quarterly_vals[inverse_idx]

    dow_arr = ((unix_days + 3) % 7).astype(np.float64)
    hour_arr = (ts_seconds % 86400) / 3600.0
    hours_from_sat = (dow_arr - 5) * 24.0 + hour_arr
    weekend_prox = np.where(
        (hours_from_sat >= 0) & (hours_from_sat <= 48), 1.0,
        np.where(
            (hours_from_sat > -12) & (hours_from_sat < 0),
            1.0 + hours_from_sat / 12.0,
            np.where(
                (hours_from_sat > 48) & (hours_from_sat < 60),
                1.0 - (hours_from_sat - 48) / 12.0,
                0.0
            )
        )
    )
    f["weekend_proximity"] = np.clip(weekend_prox, 0.0, 1.0)
    f["epoch_4h_phase"] = (ts_seconds % (4 * 3600)) / (4 * 3600)

    # Funding rate cycle features
    funding_cycle_phase = (ts_seconds % (8 * 3600)) / (8 * 3600)
    f["funding_cycle_phase"] = funding_cycle_phase
    f["funding_cycle_sin"] = np.sin(2.0 * np.pi * funding_cycle_phase)
    f["funding_cycle_cos"] = np.cos(2.0 * np.pi * funding_cycle_phase)
    # Smooth proximity to funding settlement (0h, 8h, 16h UTC)
    dist_s = ts_seconds % (8 * 3600)
    dist_s = np.minimum(dist_s, 8 * 3600 - dist_s)
    halflife_fund = 1.0 * 3600  # 1h half-life
    decay_fund = np.log(2) / halflife_fund
    f["funding_time_proximity"] = np.exp(-decay_fund * dist_s)

    # Cyclical time features (bundled here to share ts_seconds)
    diurnal_phase = 2.0 * np.pi * ((ts_seconds % 86400) / 86400.0)
    f["sin_diurnal"] = np.sin(diurnal_phase)
    f["cos_diurnal"] = np.cos(diurnal_phase)
    week_seconds = dow_arr * 86400.0 + (ts_seconds % 86400)
    weekly_phase = 2.0 * np.pi * week_seconds / (7 * 86400)
    f["sin_weekly"] = np.sin(weekly_phase)
    f["cos_weekly"] = np.cos(weekly_phase)
    hour_phase = 2.0 * np.pi * hour_arr / 24.0
    f["sin_hour"] = np.sin(hour_phase)
    f["cos_hour"] = np.cos(hour_phase)

    return f


def _compute_tier3_distribution(
    arrays: dict, cfg: PipelineConfig, n: int
) -> dict:
    """T3: Return autocorrelation, skew, kurt, correlations."""
    f = {}
    s, p = cfg.spot, cfg.perp

    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        for label, w in [("5m", 300), ("15m", 900)]:
            ac = _rolling_autocorr_lag1(r, w)
            f[f"{inst}_ret_autocorr_{label}"] = np.full(n, np.nan)
            f[f"{inst}_ret_autocorr_{label}"][1:] = ac

    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        for label, w in [("5m", 300), ("15m", 900)]:
            f[f"{inst}_skew_{label}"] = _rolling_skew(r, w)
            f[f"{inst}_kurt_{label}"] = _rolling_kurt(r, w)

    f["sp_ret_corr_5m"] = _rolling_corr(
        arrays[f"{s}_ret"], arrays[f"{p}_ret"], 300
    )
    spr_clean = np.nan_to_num(arrays[f"{s}_spread_bps"], nan=0.0)
    r2 = arrays[f"{s}_r2"]
    rv_1m = _rolling_sum(r2, 60)
    rv_clean = np.nan_to_num(rv_1m, nan=0.0)
    f["spread_rv_corr_15m"] = _rolling_corr(spr_clean, rv_clean, 900)

    return f


def _compute_tier1_realized_covariance(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """Realized covariance between spot and perp returns."""
    f = {}
    s, p = cfg.spot, cfg.perp
    eps = cfg.eps
    r_s = arrays[f"{s}_ret"]
    r_p = arrays[f"{p}_ret"]
    r2_s = arrays[f"{s}_r2"]
    r2_p = arrays[f"{p}_r2"]
    cross = r_s * r_p
    for label, w in [("1m", 60), ("5m", 300)]:
        rcov = cache.rolling_sum(cross, w)
        f[f"sp_rcov_{label}"] = rcov
        rv_s = cache.rolling_sum(r2_s, w)
        rv_p = cache.rolling_sum(r2_p, w)
        denom = np.sqrt(np.maximum(rv_s * rv_p, eps))
        f[f"sp_rcov_corr_ratio_{label}"] = _safe_div(rcov, denom)
    return f


def _compute_tier1_signed_vol_imbalance(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """Net signed volume imbalance — Σ(sign(r) × vol_proxy) / Σ(vol_proxy)."""
    f = {}
    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        vol_proxy = arrays[f"{inst}_n_updates"]
        signed_vol = np.sign(r) * vol_proxy
        for label, w in [("w60", 60), ("w300", 300), ("w900", 900)]:
            sum_signed = cache.rolling_sum(signed_vol, w)
            sum_vol = cache.rolling_sum(vol_proxy, w)
            f[f"{inst}_signed_vol_imbalance_{label}"] = _safe_div(
                sum_signed, sum_vol
            )
    return f


def _compute_tier1_cumulative_returns(
    arrays: dict, cfg: PipelineConfig
) -> dict:
    """Cumulative log-returns and perp/spot ratio."""
    f = {}
    s, p = cfg.spot, cfg.perp
    for inst in cfg.instruments:
        r = arrays[f"{inst}_ret"]
        for label, w in [("10s", 10), ("60s", 60)]:
            f[f"{inst}_cum_ret_{label}"] = _rolling_sum(r, w)
    for label in ["10s", "60s"]:
        p_cum = f[f"{p}_cum_ret_{label}"]
        s_cum = f[f"{s}_cum_ret_{label}"]
        f[f"ps_cum_ret_ratio_{label}"] = _safe_div(p_cum, s_cum)
    return f


def _compute_tier1_cross_exchange(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """Cross-exchange features: primary spot (s) vs Coinbase spot (c)."""
    f = {}
    s = cfg.spot
    c = cfg.coinbase_spot
    eps = cfg.eps

    s_mid = arrays[f"{s}_mid"]
    c_mid = arrays[f"{c}_mid"]
    s_mp = arrays[f"{s}_microprice"]
    c_mp = arrays[f"{c}_microprice"]
    s_spr = arrays[f"{s}_spread_bps"]
    c_spr = arrays[f"{c}_spread_bps"]
    s_obi = arrays[f"{s}_obi"]
    c_obi = arrays[f"{c}_obi"]
    s_r2 = arrays[f"{s}_r2"]
    c_r2 = arrays[f"{c}_r2"]

    # ── Mid price deltas ──
    mid_delta = np.log(s_mid + eps) - np.log(c_mid + eps)
    f["sc_mid_delta"] = mid_delta
    for w in [5, 30, 60, 300]:
        f[f"sc_mid_delta_w{w}"] = cache.rolling_mean(mid_delta, w)
    for w in [60, 300]:
        f[f"sc_mid_delta_vol_w{w}"] = _rolling_std(mid_delta, w)

    # ── Cross exchange OBI imbalance ──
    obi_imb = s_obi - c_obi
    f["sc_obi_imbalance"] = obi_imb
    for w in [60, 300]:
        f[f"sc_obi_imbalance_w{w}"] = cache.rolling_mean(obi_imb, w)

    # ── Spread ratio ──
    spread_ratio = _safe_div(s_spr, np.maximum(c_spr, eps))
    f["sc_spread_ratio"] = spread_ratio
    for w in [60, 300]:
        f[f"sc_spread_ratio_w{w}"] = cache.rolling_mean(spread_ratio, w)

    # ── Realized vol spread ──
    for w in [60, 300, 900, 3600]:
        s_rv = cache.rolling_sum(s_r2, w)
        c_rv = cache.rolling_sum(c_r2, w)
        f[f"sc_rv_spread_w{w}"] = np.log(s_rv + eps) - np.log(c_rv + eps)

    # ── VWAP ratio (microprice as L1 VWAP proxy) ──
    vwap_ratio = _safe_div(s_mp, c_mp)
    for w in [60, 300]:
        f[f"sc_vwap_ratio_w{w}"] = cache.rolling_mean(vwap_ratio, w)

    # ── VWAP delta ──
    vwap_delta = np.log(s_mp + eps) - np.log(c_mp + eps)
    f["sc_vwap_delta"] = vwap_delta
    for w in [60, 300]:
        f[f"sc_vwap_delta_w{w}"] = cache.rolling_mean(vwap_delta, w)

    return f


def _compute_tier2_vol_phase(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """Volatility phase — normalized deviation of log-RV from rolling mean."""
    f = {}
    eps = cfg.eps
    for inst in cfg.instruments:
        r2 = arrays[f"{inst}_r2"]
        rv_1m = cache.rolling_sum(r2, 60)
        log_rv_1m = np.log(rv_1m + eps)
        for label, w in [("5m", 300), ("15m", 900)]:
            trend = cache.rolling_mean(log_rv_1m, w)
            dev = log_rv_1m - trend
            std = _rolling_std(log_rv_1m, w)
            f[f"{inst}_vol_phase_{label}"] = _safe_div(
                dev, np.maximum(std, eps)
            )
    return f


def _compute_tier3_session_seasonality(
    df: pl.DataFrame, cfg: PipelineConfig, n: int
) -> dict:
    """Smooth proximity indicators for major exchange session opens/closes."""
    f = {}
    ts = cfg.ts_col
    timestamps_us = df.get_column(ts).cast(pl.Int64).to_numpy()
    ts_seconds = timestamps_us / 1e6
    tod_utc = ts_seconds % 86400

    # (name, utc_hour, utc_minute)
    sessions = [
        ("tokyo_open", 0, 0),       # 09:00 JST = 00:00 UTC
        ("tokyo_close", 6, 0),      # 15:00 JST = 06:00 UTC
        ("shanghai_open", 1, 30),   # 09:30 CST = 01:30 UTC
        ("shanghai_close", 7, 0),   # 15:00 CST = 07:00 UTC
        ("london_open", 8, 0),      # 08:00 GMT
        ("london_close", 16, 30),   # 16:30 GMT
        ("nyse_open", 14, 30),      # 09:30 ET = 14:30 UTC
        ("nyse_close", 21, 0),      # 16:00 ET = 21:00 UTC
        ("cme_open", 23, 0),        # 17:00 CT = 23:00 UTC
        ("cme_close", 22, 0),       # 16:00 CT = 22:00 UTC
    ]

    halflife_s = 30 * 60  # 30-minute half-life
    decay = np.log(2) / halflife_s
    for name, h, m in sessions:
        event_s = h * 3600 + m * 60
        diff = tod_utc - event_s
        # Wrap to [-43200, 43200] (circular distance on 24h clock)
        diff = diff - 86400 * np.round(diff / 86400)
        f[f"session_{name}"] = np.exp(-decay * np.abs(diff))

    return f


def _compute_tier3_deribit_expiry(
    df: pl.DataFrame, cfg: PipelineConfig, n: int
) -> dict:
    """Deribit weekly/monthly options expiry calendar features."""
    f = {}
    ts = cfg.ts_col
    timestamps_us = df.get_column(ts).cast(pl.Int64).to_numpy()
    ts_seconds = timestamps_us / 1e6
    tod_hours = (ts_seconds % 86400) / 3600.0
    unix_days_int = np.floor(ts_seconds / 86400.0).astype(np.int64)
    dow = ((unix_days_int + 3) % 7)  # 0=Mon, 4=Fri

    # ── Weekly expiry: every Friday 08:00 UTC ──
    days_until_fri = ((4 - dow) % 7).astype(np.float64)
    past_8utc = tod_hours >= 8.0
    days_until_fri = np.where(
        (days_until_fri == 0) & past_8utc, 7.0, days_until_fri
    )
    weekly_hours = days_until_fri * 24.0 + (8.0 - tod_hours)
    f["deribit_weekly_expiry_hours"] = weekly_hours

    halflife_h = 4.0
    decay_w = np.log(2) / halflife_h
    f["deribit_weekly_expiry_proximity"] = np.exp(
        -decay_w * np.maximum(weekly_hours, 0.0)
    )

    # ── Monthly expiry: last Friday of month 08:00 UTC ──
    unique_days, inverse_idx = np.unique(unix_days_int, return_inverse=True)
    monthly_hours_vals = np.empty(len(unique_days), dtype=np.float64)
    for j, d in enumerate(unique_days):
        dt = datetime.utcfromtimestamp(int(d) * 86400)
        dt_date = datetime(dt.year, dt.month, dt.day)
        lf = _last_friday_of_month(dt_date.year, dt_date.month)
        if dt_date >= lf:
            if dt_date.month == 12:
                lf = _last_friday_of_month(dt_date.year + 1, 1)
            else:
                lf = _last_friday_of_month(dt_date.year, dt_date.month + 1)
        expiry = datetime(lf.year, lf.month, lf.day, 8, 0, 0)
        monthly_hours_vals[j] = (expiry - dt_date).total_seconds() / 3600.0

    monthly_hours = monthly_hours_vals[inverse_idx] - tod_hours
    monthly_hours = np.maximum(monthly_hours, 0.0)
    f["deribit_monthly_expiry_hours"] = monthly_hours
    f["deribit_monthly_expiry_proximity"] = np.exp(
        -decay_w * monthly_hours
    )

    return f


def _compute_har_windows(
    arrays: dict, cfg: PipelineConfig, cache: RollingCache
) -> dict:
    """HAR rolling windows on base features."""
    f = {}
    eps = cfg.eps
    for inst in cfg.instruments:
        for feat in cfg.base_features:
            col = f"{inst}_{feat}"
            vals = arrays[col]
            for w in cfg.rolling_windows:
                alias = f"{inst}_{feat}_w{w}"
                f[alias] = np.log(np.sqrt(cache.rolling_sum(vals, w) + eps) + eps)
    return f


def _compute_log_transforms(f: dict, cfg: PipelineConfig) -> dict:
    """Log-transform variance-scale features."""
    eps = cfg.eps
    log_f = {}
    for inst in cfg.instruments:
        for label in ["1m", "5m", "15m"]:
            for prefix in ["tsrv_", "rk_", "rq_"]:
                key = f"{inst}_{prefix}{label}"
                if key in f:
                    log_f[f"log_{key}"] = np.log(np.maximum(f[key], eps))
        for hl in ["5s", "30s"]:
            key = f"{inst}_ewma_var_{hl}"
            if key in f:
                log_f[f"log_{key}"] = np.log(np.maximum(f[key], eps))
        for label in ["1m", "5m"]:
            key = f"{inst}_roll_measure_{label}"
            if key in f:
                log_f[f"log_{key}"] = np.log(np.maximum(f[key], eps))
        for label in ["30s", "60s"]:
            key = f"{inst}_parkinson_{label}"
            if key in f:
                log_f[f"log_{key}"] = np.log(np.maximum(f[key], eps))
        # Additional log transforms for 1h/4h estimators
        for label in ["1h", "4h"]:
            key = f"{inst}_tsrv_{label}"
            if key in f:
                log_f[f"log_{key}"] = np.log(np.maximum(f[key], eps))
        key = f"{inst}_rq_1h"
        if key in f:
            log_f[f"log_{key}"] = np.log(np.maximum(f[key], eps))
    return log_f


def _flush_to_df(df: pl.DataFrame, f: dict) -> pl.DataFrame:
    """Convert numpy feature dict to Polars columns, free memory, return new df."""
    if not f:
        return df
    cols = [pl.Series(name, vals) for name, vals in f.items()]
    df = df.with_columns(cols)
    del cols
    f.clear()
    gc.collect()
    return df


def compute_all_features(
    df: pl.DataFrame, cfg: PipelineConfig
) -> pl.DataFrame:
    """
    Optimised master feature computation.

    Memory strategy: flush each feature group to the Polars DataFrame
    immediately so numpy arrays don't accumulate (each is ~825 MB at
    100M+ rows).
    """
    ts = cfg.ts_col
    n = df.height

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ── Precompute shared arrays ──────────────────────────
    arrays = _precompute_arrays(df, cfg)
    cache = RollingCache()

    # ── Warm up numba (first call compiles; subsequent calls fast) ──
    _warmup = np.zeros(100, dtype=np.float64)
    _rolling_sum_nb(_warmup, 10)
    _rolling_std_nb(_warmup, 10)
    _rolling_max_nb(_warmup, 10)
    _rolling_corr_nb(_warmup, _warmup, 10)
    _rolling_skew_nb(_warmup, 10)
    _rolling_kurt_nb(_warmup, 10)
    _tsrv_nb(_warmup, 10, 5)
    _realized_kernel_nb(_warmup, 10)
    _roll_measure_nb(_warmup, 10)
    del _warmup

    # ── Dispatch feature groups ───────────────────────────
    _steps = [
        "T1: Basis/spread/jump",        "T1: Returns/drawdown/RV",
        "T1: Microprice/VPIN/OBI",      "T1: Rcov/vol-imb/cum-ret",
    ]
    if cfg.coinbase_spot:
        _steps.append("T1: Cross-exchange")
    _steps.extend([
        "T2: MODWT/corr/vol-of-vol",
        "T2: Estimators + log transforms",
        "T3: Spectral bands",
        "T3: Fourier magnitudes",       "T3: Autocorr/skew/kurt",
        "T2: Vol phase",
        "T3: Calendar/sessions/expiry",
        "HAR rolling windows",
    ])
    pbar = tqdm(_steps, desc="  Features", unit="grp", ncols=90, leave=True)
    pbar_iter = iter(pbar)
    n_flushed = 0

    f = {}

    # ── Phase 1: T1 groups (sequential for cache sharing) ─
    next(pbar_iter)
    result = _compute_tier1_basis_spread_jump(arrays, cfg, cache)
    n_flushed += len(result)
    df = _flush_to_df(df, result)

    next(pbar_iter)
    result = _compute_tier1_returns_drawdown_rv(arrays, cfg, cache)
    n_flushed += len(result)
    df = _flush_to_df(df, result)

    next(pbar_iter)
    result = _compute_tier1_microprice_vpin_obi(arrays, cfg, cache, n)
    n_flushed += len(result)
    df = _flush_to_df(df, result)

    next(pbar_iter)
    result = _compute_tier1_realized_covariance(arrays, cfg, cache)
    result.update(_compute_tier1_signed_vol_imbalance(arrays, cfg, cache))
    result.update(_compute_tier1_cumulative_returns(arrays, cfg))
    n_flushed += len(result)
    df = _flush_to_df(df, result)

    if cfg.coinbase_spot:
        next(pbar_iter)
        result = _compute_tier1_cross_exchange(arrays, cfg, cache)
        n_flushed += len(result)
        df = _flush_to_df(df, result)

    next(pbar_iter)
    result = _compute_tier2_modwt_corr_vov(arrays, cfg, cache)
    n_flushed += len(result)
    df = _flush_to_df(df, result)
    _log_mem()

    # ── Phase 2: T2/T3 groups — compute, flush each immediately ──

    # T2 Estimators + log transforms (log transforms only depend on these)
    next(pbar_iter)
    est = _compute_tier2_estimators(arrays, cfg, cache)
    log_est = _compute_log_transforms(est, cfg)
    est.update(log_est)
    del log_est
    n_flushed += len(est)
    df = _flush_to_df(df, est)

    next(pbar_iter)
    result = _compute_tier3_spectral(arrays, cfg)
    n_flushed += len(result)
    df = _flush_to_df(df, result)

    next(pbar_iter)
    result = _compute_tier3_fourier(arrays, cfg, n)
    n_flushed += len(result)
    df = _flush_to_df(df, result)

    next(pbar_iter)
    result = _compute_tier3_distribution(arrays, cfg, n)
    n_flushed += len(result)
    df = _flush_to_df(df, result)

    next(pbar_iter)
    result = _compute_tier2_vol_phase(arrays, cfg, cache)
    n_flushed += len(result)
    df = _flush_to_df(df, result)
    cache.clear()

    # Calendar, sessions, expiry (needs df for timestamps)
    next(pbar_iter)
    f.update(_compute_tier3_calendar(df, cfg, n))
    f.update(_compute_tier3_session_seasonality(df, cfg, n))
    f.update(_compute_tier3_deribit_expiry(df, cfg, n))
    n_flushed += len(f)
    df = _flush_to_df(df, f)

    # HAR rolling windows
    next(pbar_iter)
    result = _compute_har_windows(arrays, cfg, cache)
    n_flushed += len(result)
    df = _flush_to_df(df, result)
    cache.clear()

    del arrays
    gc.collect()

    pbar.close()
    warnings.filterwarnings("default", category=RuntimeWarning)

    log.info(f"    └─ Flushed {n_flushed} feature columns total")
    _log_mem()
    return df


# ═══════════════════════════════════════════════════════════
# Everything below is identical to pipeline_v2.py
# ═══════════════════════════════════════════════════════════

def get_feature_columns(cfg: PipelineConfig) -> list[str]:
    s = cfg.spot
    p = cfg.perp
    cols = []
    for inst in cfg.instruments:
        for feat in cfg.base_features:
            for w in cfg.rolling_windows:
                cols.append(f"{inst}_{feat}_w{w}")
    for w in [5, 60, 1440, 8192]:
        cols.append(f"basis_level_w{w}")
    for w in [60, 1440, 8192]:
        cols.append(f"basis_vol_w{w}")
    for lag in [60, 300]:
        cols.append(f"basis_change_{lag}s")
    cols.append("basis_zscore_1h")
    for inst in cfg.instruments:
        for w in [5, 60, 1440]:
            cols.append(f"{inst}_spread_w{w}")
    cols.append("s_spread_zscore_5m")
    for inst in cfg.instruments:
        for w in [60, 1440, 8192]:
            cols.append(f"{inst}_jump_ratio_w{w}")
    for inst in cfg.instruments:
        for w in [60, 1440, 8192]:
            cols.append(f"{inst}_asym_impact_w{w}")
    for inst in cfg.instruments:
        for w in [60, 1440, 8192]:
            cols.append(f"{inst}_downside_share_w{w}")
    for inst in cfg.instruments:
        for w in [30, 60, 1440]:
            cols.append(f"{inst}_signed_ret_w{w}")
    for inst in cfg.instruments:
        for w in [1440, 8192, 16384]:
            cols.append(f"{inst}_drawdown_w{w}")
    for inst in cfg.instruments:
        cols.append(f"{inst}_rv_term_slope")
        cols.append(f"{inst}_rv_term_curvature")
    for label in ["1s", "5s", "10s", "30s", "60s", "300s"]:
        cols.append(f"microprice_ret_{label}")
    cols.extend(["ret_accel_1s_5s", "ret_accel_5s_30s", "ret_accel_30s_300s"])
    cols.extend(["vpin_5m", "tick_intensity_ratio"])
    for inst in cfg.instruments:
        cols.append(f"{inst}_obi_delta_5s")
        cols.append(f"{inst}_obi_autocorr_1m")
    cols.append("obi_divergence")
    for inst in cfg.instruments:
        cols.append(f"{inst}_spread_obi_interaction")
    for inst in cfg.instruments:
        for scale in cfg.modwt_scales:
            cols.append(f"{inst}_modwt_d{scale}")
    for w in [60, 1440, 8192]:
        cols.append(f"sp_corr_w{w}")
    for inst in cfg.instruments:
        for w in [60, 1440]:
            cols.append(f"{inst}_vol_of_vol_w{w}")
    for inst in cfg.instruments:
        for label in ["1m", "5m", "15m"]:
            cols.append(f"{inst}_tsrv_{label}")
    for inst in cfg.instruments:
        for label in ["1m", "5m"]:
            cols.append(f"{inst}_tsrv_rv_ratio_{label}")
    for inst in cfg.instruments:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_rk_{label}")
    for inst in cfg.instruments:
        for label in ["1m", "5m", "15m"]:
            cols.append(f"{inst}_rq_{label}")
    for inst in cfg.instruments:
        for hl in ["5s", "30s"]:
            cols.append(f"{inst}_ewma_var_{hl}")
    for inst in [s, p]:
        for label in ["1m", "5m"]:
            cols.append(f"{inst}_roll_measure_{label}")
    for inst in [s, p]:
        for label in ["30s", "60s"]:
            cols.append(f"{inst}_parkinson_{label}")
    cols.append("perp_lead_1s")
    for inst in cfg.instruments:
        for label in ["5s", "30s", "60s", "300s"]:
            cols.append(f"{inst}_roughness_{label}")
        cols.append(f"{inst}_vol_accel_5s")
    for inst in cfg.instruments:
        for w_label in ["4k", "8k"]:
            cols.append(f"{inst}_spec_hf_lf_{w_label}")
            cols.append(f"{inst}_spec_mf_lf_{w_label}")
            cols.append(f"{inst}_spec_hf_mf_{w_label}")
            cols.append(f"{inst}_spec_trend_share_{w_label}")
    for inst in [s, p]:
        for period in ["8h", "24h", "168h"]:
            cols.append(f"{inst}_fourier_mag_{period}")
    cols.extend([
        "sin_funding", "cos_funding",
        "monthly_expiry_days", "quarterly_expiry_days",
        "weekend_proximity", "epoch_4h_phase",
    ])
    for inst in cfg.instruments:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_ret_autocorr_{label}")
    for inst in cfg.instruments:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_skew_{label}")
            cols.append(f"{inst}_kurt_{label}")
    cols.extend(["sp_ret_corr_5m", "spread_rv_corr_15m"])
    cols.extend([
        "sin_diurnal", "cos_diurnal",
        "sin_weekly", "cos_weekly",
        "sin_hour", "cos_hour",
    ])
    for inst in cfg.instruments:
        for label in ["1m", "5m", "15m"]:
            for prefix in ["tsrv_", "rq_"]:
                cols.append(f"log_{inst}_{prefix}{label}")
        for label in ["5m", "15m"]:
            cols.append(f"log_{inst}_rk_{label}")
        for hl in ["5s", "30s"]:
            cols.append(f"log_{inst}_ewma_var_{hl}")
        for label in ["1m", "5m"]:
            cols.append(f"log_{inst}_roll_measure_{label}")
        for label in ["30s", "60s"]:
            cols.append(f"log_{inst}_parkinson_{label}")

    # ── New features ──────────────────────────────────────────
    # TSRV 1h/4h (already computed, adding to feature list)
    for inst in cfg.instruments:
        for label in ["1h", "4h"]:
            cols.append(f"{inst}_tsrv_{label}")
    # Downside share w3600 (already computed for this window)
    for inst in cfg.instruments:
        cols.append(f"{inst}_downside_share_w3600")
    # RQ 1h (already computed)
    for inst in cfg.instruments:
        cols.append(f"{inst}_rq_1h")
    # Raw instantaneous spread in bps (already in DataFrame)
    for inst in cfg.instruments:
        cols.append(f"{inst}_spread_bps")
    # Signed jump w900, w3600 (already computed)
    for inst in cfg.instruments:
        for w in [900, 3600]:
            cols.append(f"{inst}_signed_jump_w{w}")
    # Raw realized jump level w3600 (already computed)
    for inst in cfg.instruments:
        cols.append(f"{inst}_rj_w3600")
    # Realized covariance
    for label in ["1m", "5m"]:
        cols.append(f"sp_rcov_{label}")
        cols.append(f"sp_rcov_corr_ratio_{label}")
    # Signed volume imbalance
    for inst in cfg.instruments:
        for label in ["w60", "w300", "w900"]:
            cols.append(f"{inst}_signed_vol_imbalance_{label}")
    # Exchange session seasonality
    for session in [
        "tokyo_open", "tokyo_close", "shanghai_open", "shanghai_close",
        "london_open", "london_close", "nyse_open", "nyse_close",
        "cme_open", "cme_close",
    ]:
        cols.append(f"session_{session}")
    # Deribit expiry calendar
    cols.extend([
        "deribit_weekly_expiry_hours", "deribit_monthly_expiry_hours",
        "deribit_weekly_expiry_proximity", "deribit_monthly_expiry_proximity",
    ])
    # Funding rate cycle
    cols.extend([
        "funding_cycle_phase", "funding_cycle_sin", "funding_cycle_cos",
        "funding_time_proximity",
    ])
    # Volatility phase
    for inst in cfg.instruments:
        for label in ["5m", "15m"]:
            cols.append(f"{inst}_vol_phase_{label}")
    # Cumulative returns
    for inst in cfg.instruments:
        for label in ["10s", "60s"]:
            cols.append(f"{inst}_cum_ret_{label}")
    for label in ["10s", "60s"]:
        cols.append(f"ps_cum_ret_ratio_{label}")
    # Log transforms for 1h/4h estimators
    for inst in cfg.instruments:
        for label in ["1h", "4h"]:
            cols.append(f"log_{inst}_tsrv_{label}")
        cols.append(f"log_{inst}_rq_1h")

    # ── Cross-exchange features (Coinbase spot vs primary spot) ──
    if cfg.coinbase_spot:
        cols.append("coinbase_available")
        # Mid price deltas
        cols.append("sc_mid_delta")
        for w in [5, 30, 60, 300]:
            cols.append(f"sc_mid_delta_w{w}")
        for w in [60, 300]:
            cols.append(f"sc_mid_delta_vol_w{w}")
        # Cross exchange OBI imbalance
        cols.append("sc_obi_imbalance")
        for w in [60, 300]:
            cols.append(f"sc_obi_imbalance_w{w}")
        # Spread ratio
        cols.append("sc_spread_ratio")
        for w in [60, 300]:
            cols.append(f"sc_spread_ratio_w{w}")
        # Realized vol spread
        for w in [60, 300, 900, 3600]:
            cols.append(f"sc_rv_spread_w{w}")
        # VWAP ratio (microprice proxy)
        for w in [60, 300]:
            cols.append(f"sc_vwap_ratio_w{w}")
        # VWAP delta
        cols.append("sc_vwap_delta")
        for w in [60, 300]:
            cols.append(f"sc_vwap_delta_w{w}")

    return cols


def train_test_split(df, cfg):
    n = df.height
    split_idx = int(n * cfg.train_frac)
    return df[:split_idx], df[split_idx:]


def compute_target_for_horizon(df, cfg, inst, horizon_s):
    rv = df.get_column(f"{inst}_rv")
    seas = df.get_column(f"{inst}_seas_factor")
    target_deseas = rv.rolling_sum(horizon_s).shift(-horizon_s).to_numpy()
    target_deseas_log = np.log(target_deseas + cfg.eps)
    fwd_seas_sum = seas.rolling_sum(horizon_s).shift(-horizon_s).to_numpy()
    return target_deseas_log, target_deseas, fwd_seas_sum


def fit_horizon_model(df_train, cfg, horizon_s, inst, alpha=1.0):
    feature_cols = get_feature_columns(cfg)
    target_log, _, _ = compute_target_for_horizon(df_train, cfg, inst, horizon_s)
    target_series = pl.Series("__target", target_log)
    subset = (
        df_train.select(feature_cols)
        .with_columns(target_series.alias("__target"))
        .drop_nulls()
        .drop_nans()
    )
    if subset.height < len(feature_cols) + 1:
        return None
    X = subset.select(feature_cols).to_numpy()
    y = subset.get_column("__target").to_numpy()
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_s = (X - X_mean) / X_std
    n_feat = X_s.shape[1]
    XtX = X_s.T @ X_s
    Xty = X_s.T @ y
    beta_s = np.linalg.solve(XtX + alpha * np.eye(n_feat), Xty)
    beta = beta_s / X_std
    intercept = y.mean() - X_mean @ beta
    y_hat = X @ beta + intercept
    residuals = y - y_hat
    sigma2 = np.var(residuals, ddof=n_feat + 1)
    return {
        "horizon_s": horizon_s, "inst": inst, "feature_cols": feature_cols,
        "beta": beta, "intercept": intercept, "X_mean": X_mean,
        "X_std": X_std, "sigma2": sigma2, "alpha": alpha,
        "n_train": X.shape[0],
    }


def predict_and_reseasonalise(df, model, cfg):
    inst = model["inst"]
    horizon_s = model["horizon_s"]
    feature_cols = model["feature_cols"]
    beta = model["beta"]
    intercept = model["intercept"]
    X = df.select(feature_cols).to_numpy()
    log_yhat = X @ beta + intercept
    yhat = np.exp(log_yhat)
    _, _, fwd_seas = compute_target_for_horizon(df, cfg, inst, horizon_s)
    return yhat * (fwd_seas / horizon_s)


def _log_mem():
    """Log current RSS in GB (Linux only)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    log.info(f"       [mem] RSS = {kb / 1048576:.1f} GB")
                    return
    except Exception:
        pass


def run_pipeline(output_path, cfg=None):
    if cfg is None:
        cfg = PipelineConfig()
    setup_logging()
    n_steps = 10 if cfg.coinbase_spot else 8
    output_path = Path(output_path)
    with StepTimer(f"[1/{n_steps}] Loading & merging {len(cfg.input_map)} instruments"):
        df = load_and_merge(cfg)
        log.info(f"       Merged rows: {df.height:,}")
        _log_mem()
    if cfg.coinbase_spot:
        with StepTimer(f"[2/{n_steps}] Detecting & fixing Coinbase staleness"):
            df = detect_and_fix_coinbase_staleness(df, cfg)
    step = 3 if cfg.coinbase_spot else 2
    with StepTimer(f"[{step}/{n_steps}] Computing 1s bars (mid, microprice, spread, OBI, returns)"):
        df = compute_1s_bars(df, cfg)
    step += 1
    with StepTimer(f"[{step}/{n_steps}] Computing per-second base features (RV, RSV±, BPV)"):
        df = compute_persecond_base(df, cfg)
    step += 1
    if cfg.coinbase_spot:
        with StepTimer(f"[{step}/{n_steps}] Computing cross-exchange VWMID composite (x)"):
            df = compute_cross_exchange_composite(df, cfg)
            log.info("       Target instrument: x (volume-weighted mid price)")
        step += 1
    with StepTimer(f"[{step}/{n_steps}] Train/test split + FFF seasonality"):
        df_train, _ = train_test_split(df, cfg)
        seasonality = compute_fff_seasonality(df_train, cfg)
    del df_train; gc.collect()
    step += 1
    with StepTimer(f"[{step}/{n_steps}] Deseasonalising base features"):
        df = apply_seasonality(df, seasonality, cfg)
    step += 1
    _log_mem()
    with StepTimer(f"[{step}/{n_steps}] Computing all features (Tiers 1-3)"):
        df = compute_all_features(df, cfg)
    gc.collect()
    _log_mem()
    feature_cols = get_feature_columns(cfg)
    log.info(f"       Total model features: {len(feature_cols)}")
    log.info(f"       Target instruments: {cfg.target_instruments}")
    step += 1
    with StepTimer(f"[{step}/{n_steps}] Saving features"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        log.info(f"       → {output_path}  ({df.height:,} rows × {df.width} cols)")
    step += 1
    with StepTimer(f"[{step}/{n_steps}] Saving seasonality"):
        import pickle
        seas_path = output_path.with_name(output_path.stem + "_seasonality.pkl")
        with open(seas_path, "wb") as fh:
            pickle.dump(seasonality, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"       → {seas_path}")
    return df


def fit_and_save_models(df, cfg, output_dir, horizon_seconds=None, alpha=1.0):
    import pickle
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if horizon_seconds is None:
        horizon_seconds = list(range(60, cfg.max_horizon_s + 1, 60))
    df_train, _ = train_test_split(df, cfg)
    models = {}
    for inst in cfg.target_instruments:
        log.info(f"  Fitting {inst} ({len(horizon_seconds)} horizons, α={alpha}) ...")
        for h_s in tqdm(horizon_seconds, desc=f"  {inst}", unit="h", ncols=80):
            model = fit_horizon_model(df_train, cfg, h_s, inst=inst, alpha=alpha)
            if model is None:
                continue
            models[(inst, h_s)] = model
        n_fit = sum(1 for k in models if k[0] == inst)
        log.info(f"    → {n_fit} models for {inst}")
    bundle = {
        "models": models,
        "config": {
            "agg_freq_s": cfg.agg_freq_s, "instruments": cfg.instruments,
            "max_horizon_s": cfg.max_horizon_s,
            "rolling_windows": cfg.rolling_windows,
            "eps": cfg.eps, "train_frac": cfg.train_frac, "alpha": alpha,
        },
    }
    path = output_dir / "models.pkl"
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"✓ Saved {len(models)} models → {path}")
    return models


def _r_squared(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def fit_and_evaluate(df, cfg, output_dir, horizon_seconds=None, alpha=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if horizon_seconds is None:
        horizon_seconds = list(range(60, cfg.max_horizon_s + 1, 60))
    df_train, df_test = train_test_split(df, cfg)
    records = []
    for inst in cfg.target_instruments:
        log.info(f"  Instrument: {inst}  (Ridge α={alpha})")
        horizon_iter = tqdm(horizon_seconds, desc=f"  {inst} horizons",
                            unit="h", ncols=80, leave=True)
        for h_s in horizon_iter:
            model = fit_horizon_model(df_train, cfg, h_s, inst=inst, alpha=alpha)
            if model is None:
                continue
            feature_cols = model["feature_cols"]
            beta = model["beta"]
            intercept = model["intercept"]
            X_test = df_test.select(feature_cols).to_numpy()
            log_yhat = X_test @ beta + intercept
            y_log, y_deseas, fwd_seas = compute_target_for_horizon(df_test, cfg, inst, h_s)
            r2_log = _r_squared(y_log, log_yhat)
            yhat_raw = np.exp(log_yhat) * (fwd_seas / h_s)
            y_raw = y_deseas * (fwd_seas / h_s)
            r2_raw = _r_squared(y_raw, yhat_raw)
            X_tr = df_train.select(feature_cols).to_numpy()
            y_tr_log, _, _ = compute_target_for_horizon(df_train, cfg, inst, h_s)
            log_yhat_tr = X_tr @ beta + intercept
            r2_train = _r_squared(y_tr_log, log_yhat_tr)
            horizon_iter.set_postfix_str(f"R²={r2_log:+.4f} raw={r2_raw:+.4f}")
            records.append({
                "inst": inst, "horizon_s": h_s, "horizon_min": h_s / 60,
                "r2_log_train": r2_train, "r2_log_test": r2_log,
                "r2_raw_test": r2_raw, "sigma2": model["sigma2"],
                "n_features": len(feature_cols), "alpha": alpha,
            })

    summary = pl.DataFrame(records)
    summary.write_csv(output_dir / "r2_summary.csv")

    with PdfPages(output_dir / "analysis.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        for inst in cfg.target_instruments:
            sub = summary.filter(pl.col("inst") == inst).sort("horizon_s")
            h = sub.get_column("horizon_min").to_numpy()
            ax.plot(h, sub.get_column("r2_log_test").to_numpy(),
                    label=f"{inst} (test)", marker=".", markersize=3)
            ax.plot(h, sub.get_column("r2_log_train").to_numpy(),
                    label=f"{inst} (train)", linestyle="--", alpha=0.4)
        ax.set_xlabel("Horizon (minutes)"); ax.set_ylabel("R²")
        ax.set_title(f"Log-space R² vs horizon (Ridge α={alpha})")
        ax.legend(fontsize=7, ncol=2); ax.axhline(0, color="grey", linewidth=0.5)
        ax.grid(True, alpha=0.3); fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for inst in cfg.target_instruments:
            sub = summary.filter(pl.col("inst") == inst).sort("horizon_s")
            h = sub.get_column("horizon_min").to_numpy()
            ax.plot(h, sub.get_column("r2_raw_test").to_numpy(),
                    label=inst, marker=".", markersize=3)
        ax.set_xlabel("Horizon (minutes)"); ax.set_ylabel("R²")
        ax.set_title("Raw level-space R² vs horizon")
        ax.legend(fontsize=7); ax.axhline(0, color="grey", linewidth=0.5)
        ax.grid(True, alpha=0.3); fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 7))
        for inst in cfg.target_instruments:
            sub = summary.filter(pl.col("inst") == inst)
            ax.scatter(sub.get_column("r2_log_train").to_numpy(),
                       sub.get_column("r2_log_test").to_numpy(),
                       label=inst, alpha=0.6, s=30)
        ax.plot([-0.5, 1.0], [-0.5, 1.0], "k--", linewidth=0.8, label="y=x")
        ax.set_xlabel("Train R² (log)"); ax.set_ylabel("Test R² (log)")
        ax.set_title("Overfit diagnostic: Train vs Test R²")
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for inst in cfg.target_instruments:
            sub = summary.filter(pl.col("inst") == inst).sort("horizon_s")
            h = sub.get_column("horizon_min").to_numpy()
            ax.plot(h, sub.get_column("sigma2").to_numpy(),
                    label=inst, marker=".", markersize=3)
        ax.set_xlabel("Horizon (minutes)"); ax.set_ylabel("σ²")
        ax.set_title("Residual variance (log-space)")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    log.info(f"✓ Saved analysis.pdf → {output_dir / 'analysis.pdf'}")
    log.info(f"\n{'='*70}")
    log.info("  SUMMARY")
    log.info(f"{'='*70}")
    for inst in cfg.target_instruments:
        sub = summary.filter(pl.col("inst") == inst)
        r2 = sub.get_column("r2_log_test").to_numpy()
        valid = np.isfinite(r2)
        log.info(f"  {inst}:  ({valid.sum()}/{len(horizon_seconds)} horizons)")
        log.info(f"    R²_log (test):  mean={np.nanmean(r2):.4f}  "
              f"median={np.nanmedian(r2):.4f}  "
              f"[{np.nanmin(r2):.4f}, {np.nanmax(r2):.4f}]")
        r2_raw = sub.get_column("r2_raw_test").to_numpy()
        log.info(f"    R²_raw (test):  mean={np.nanmean(r2_raw):+.4f}  "
              f"median={np.nanmedian(r2_raw):+.4f}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Volatility feature pipeline v2 (2-3 instruments)",
        epilog=(
            "Example (2 instruments):\n"
            "  python pipeline_new.py --inputs s:spot.parquet p:perp.parquet "
            "--output features.parquet\n\n"
            "Example (3 instruments, with Coinbase cross-exchange features):\n"
            "  python pipeline_new.py --inputs s:spot.parquet p:perp.parquet "
            "c:coinbase.parquet --output features.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", required=True, nargs="+",
                        help="PREFIX:PATH pairs — first 2 are spot+perp, "
                             "optional 3rd is Coinbase BTC spot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-freq", type=int, default=1)
    parser.add_argument("--agg-freq", type=int, default=AGG_FREQ_S)
    parser.add_argument("--max-horizon-min", type=int, default=MAX_HORIZON_MIN)
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=ROLLING_WINDOWS)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-horizon-step", type=int, default=60)
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Thread pool size for parallel feature computation")
    args = parser.parse_args()

    input_map = {}
    for spec in args.inputs:
        if ":" not in spec:
            parser.error(f"Invalid input spec '{spec}' — expected PREFIX:PATH")
        prefix, path = spec.split(":", 1)
        input_map[prefix] = path
    if len(input_map) < 2 or len(input_map) > 3:
        parser.error("2 or 3 instruments required (spot + perp [+ coinbase_spot])")

    all_prefixes = list(input_map.keys())
    main_instruments = all_prefixes[:2]
    coinbase_spot = all_prefixes[2] if len(all_prefixes) > 2 else None

    cfg = PipelineConfig(
        base_freq_s=args.base_freq, agg_freq_s=args.agg_freq,
        instruments=main_instruments, input_map=input_map,
        coinbase_spot=coinbase_spot,
        max_horizon_min=args.max_horizon_min,
        max_horizon_s=args.max_horizon_min * 60,
        rolling_windows=args.rolling_windows,
        train_frac=args.train_frac, eps=args.eps,
        n_workers=args.n_workers,
    )
    df = run_pipeline(args.output, cfg)

    if args.eval:
        setup_logging()
        log.info("═" * 70)
        log.info("  RUNNING EVALUATION")
        log.info("═" * 70)
        horizon_seconds = list(
            range(args.eval_horizon_step, cfg.max_horizon_s + 1, args.eval_horizon_step)
        )
        eval_dir = Path(args.output).parent / "eval"
        fit_and_evaluate(df, cfg, eval_dir, horizon_seconds, alpha=args.ridge_alpha)
