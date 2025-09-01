"""Technical indicators and volume profile utilities.

Fast paths are implemented using Numba JIT-compiled kernels. Public Python
wrappers keep a simple list/None API that plays nice with plotting code.
"""

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _sma_kernel(values: np.ndarray, period: int) -> np.ndarray:
    """Numba kernel for Simple Moving Average over a numpy array.

    NaNs propagate; output is NaN until window is full.
    """
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    if period <= 0 or n == 0:
        out[:] = np.nan
        return out
    running_sum = 0.0
    for i in range(n):
        v = values[i]
        if v != v:
            out[i] = np.nan
            continue
        running_sum += v
        if i >= period:
            pv = values[i - period]
            if pv == pv:
                running_sum -= pv
            else:
                running_sum = 0.0
                for j in range(i - period + 1, i + 1):
                    tv = values[j]
                    if tv == tv:
                        running_sum += tv
        if i >= period - 1:
            out[i] = running_sum / period
        else:
            out[i] = np.nan
    return out

def sma(values, period):
    """Simple Moving Average over a Python list.

    Returns a list of floats with None where value is undefined.
    """
    arr = np.array([np.nan if (v is None) else float(v) for v in values], dtype=np.float64)
    out = _sma_kernel(arr, int(period))
    return [None if (x != x) else float(x) for x in out]


@njit(cache=True, fastmath=True)
def _ema_kernel(values: np.ndarray, period: int) -> np.ndarray:
    """Numba kernel for Exponential Moving Average (classic smoothing)."""
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    if period <= 0 or n == 0:
        out[:] = np.nan
        return out
    multiplier = 2.0 / (period + 1.0)
    running_sum = 0.0
    prev = np.nan
    for i in range(n):
        v = values[i]
        if v != v:
            out[i] = np.nan
            continue
        running_sum += v
        if i < period - 1:
            out[i] = np.nan
            continue
        if i == period - 1:
            prev = running_sum / period
            out[i] = prev
            continue
        current = (v - prev) * multiplier + prev
        prev = current
        out[i] = current
    return out

def ema(values, period):
    """Exponential Moving Average wrapper returning Python list with None gaps."""
    arr = np.array([np.nan if (v is None) else float(v) for v in values], dtype=np.float64)
    out = _ema_kernel(arr, int(period))
    return [None if (x != x) else float(x) for x in out]


def macd(values, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram.

    Returns three Python lists of equal length with None for undefined slots.
    """
    arr = np.array([np.nan if (v is None) else float(v) for v in values], dtype=np.float64)
    macd_arr, signal_arr, hist_arr = _macd_kernel(arr, int(fast), int(slow), int(signal))
    macd_line = [None if (x != x) else float(x) for x in macd_arr]
    signal_line = [None if (x != x) else float(x) for x in signal_arr]
    histogram = [None if (x != x) else float(x) for x in hist_arr]
    return macd_line, signal_line, histogram


@njit(cache=True, fastmath=True)
def _macd_kernel(values: np.ndarray, fast: int, slow: int, signal: int):
    """Numba kernel for MACD calculation using EMA components."""
    n = values.shape[0]
    macd_arr = np.empty(n, dtype=np.float64)
    signal_arr = np.empty(n, dtype=np.float64)
    hist_arr = np.empty(n, dtype=np.float64)
    if n == 0:
        macd_arr[:] = np.nan
        signal_arr[:] = np.nan
        hist_arr[:] = np.nan
        return macd_arr, signal_arr, hist_arr
    ema_fast = _ema_kernel(values, fast)
    ema_slow = _ema_kernel(values, slow)
    for i in range(n):
        macd_arr[i] = ema_fast[i] - ema_slow[i]

    start_idx = slow - 1
    if start_idx < 0:
        start_idx = 0
    if start_idx >= n:
        for i in range(n):
            signal_arr[i] = np.nan
            hist_arr[i] = np.nan
        return macd_arr, signal_arr, hist_arr

    multiplier = 2.0 / (signal + 1.0)

    for i in range(start_idx):
        signal_arr[i] = np.nan
    if start_idx + signal - 1 < n:
        s = 0.0
        for k in range(start_idx, start_idx + signal):
            s += macd_arr[k]
        prev = s / signal
        signal_arr[start_idx + signal - 1] = prev
        for i in range(start_idx + signal, n):
            prev = (macd_arr[i] - prev) * multiplier + prev
            signal_arr[i] = prev
    else:
        for i in range(start_idx, n):
            signal_arr[i] = np.nan

    for i in range(n):
        hist_arr[i] = macd_arr[i] - signal_arr[i]
    return macd_arr, signal_arr, hist_arr


@njit(cache=True, fastmath=True)
def _rsi_kernel(values: np.ndarray, period: int) -> np.ndarray:
    """Numba kernel for RSI (Wilder's smoothing)."""
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0 or period <= 0:
        out[:] = np.nan
        return out
    out[0] = np.nan
    prev = values[0]
    gains = np.empty(n - 1, dtype=np.float64) if n > 1 else np.empty(0, dtype=np.float64)
    losses = np.empty(n - 1, dtype=np.float64) if n > 1 else np.empty(0, dtype=np.float64)
    for i in range(1, n):
        v = values[i]
        if prev == prev and v == v:
            change = v - prev
            gains[i - 1] = change if change > 0.0 else 0.0
            losses[i - 1] = -change if change < 0.0 else 0.0
        else:
            gains[i - 1] = 0.0
            losses[i - 1] = 0.0
        prev = v
    avg_gain = np.nan
    avg_loss = np.nan
    for i in range(1, n):
        if i < period:
            out[i] = np.nan
            continue
        if i == period:
            s_gain = 0.0
            s_loss = 0.0
            for k in range(period):
                s_gain += gains[k]
                s_loss += losses[k]
            avg_gain = s_gain / period
            avg_loss = s_loss / period
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

def rsi(values, period=14):
    """Relative Strength Index (RSI) wrapper returning Python list with None gaps."""
    arr = np.array([np.nan if (v is None) else float(v) for v in values], dtype=np.float64)
    out = _rsi_kernel(arr, int(period))
    return [None if (x != x) else float(x) for x in out]


def volume_profile(prices, volumes, bins=40):
    """Histogram of volume by price bins.

    Returns (hist, edges, max_bin) as numpy arrays/floats; masks non-finite data.
    """
    prices_arr = np.array([np.nan if (p is None) else float(p) for p in prices], dtype=np.float64)
    vols_arr = np.array([0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in volumes], dtype=np.float64)
    mask = np.isfinite(prices_arr) & np.isfinite(vols_arr)
    if not np.any(mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0.0
    p = prices_arr[mask]
    v = vols_arr[mask]
    pmin = float(np.min(p))
    pmax = float(np.max(p))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0.0
    edges = np.linspace(pmin, pmax, int(bins) + 1)
    hist, edges = np.histogram(p, bins=edges, weights=v)
    max_bin = float(np.max(hist)) if np.any(hist) else 0.0
    return hist.astype(np.float64), edges.astype(np.float64), max_bin


def value_area_bounds(hist, edges, coverage=0.7):
    """Compute Value Area Low/High for a histogram using POC expansion.

    Args:
        hist (np.ndarray): Volumes per price bin.
        edges (np.ndarray): Bin edges (len = len(hist) + 1).
        coverage (float): Target cumulative fraction (e.g. 0.7 for 70%).

    Returns:
        tuple[float, float] | None: (VAL, VAH) or None when not computable.
    """
    if hist is None or edges is None or len(hist) == 0 or len(edges) != len(hist) + 1:
        return None
    total = float(np.sum(hist))
    if total <= 0:
        return None
    poc = int(np.argmax(hist))
    low = poc
    high = poc
    covered = float(hist[poc])
    left = poc - 1
    right = poc + 1
    target = float(coverage)
    while covered / total < target:
        left_val = float(hist[left]) if left >= 0 else -1.0
        right_val = float(hist[right]) if right < hist.shape[0] else -1.0
        if left_val < 0 and right_val < 0:
            break
        if right_val > left_val:
            covered += right_val
            high = right
            right += 1
        else:
            covered += left_val
            low = left
            left -= 1
    return float(edges[low]), float(edges[high + 1])
