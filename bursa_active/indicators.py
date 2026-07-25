"""
Pure-pandas indicator library (no TA-Lib, no external TA dependency).

Design rules, enforced by `test_engine_sanity.py`:

* **Vectorized.** Everything is a rolling/ewm/arithmetic op on pandas
  objects — no Python-level bar loops.
* **NaN warmup.** An indicator that needs `n` bars returns NaN until it
  has `n` bars. We never back-fill or seed with garbage.
* **No lookahead.** The value at index `i` is a function of data at
  indices `<= i` only. Truncating the frame after `i` must not change
  the value at `i`.
* **Breakout levels exclude the current bar.** `donchian`, `rolling_max`
  and `rolling_min` default to `exclude_current=True`, i.e. the level at
  bar `i` is the extreme of bars `i-n .. i-1`. That is what makes
  `close[i] > donchian_upper[i]` mean "today's close broke the prior
  N-day high" instead of the tautology "today's close is <= the N-day
  high that includes today" (which is true on every new high bar and
  would silently leak the current bar into its own trigger).

Functions returning more than one line (`macd`, `stochastic`,
`bollinger`, `donchian`) return a `pd.DataFrame` with named columns
rather than a bare Series; everything else returns a `pd.Series`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "sma", "ema", "macd", "rsi", "stochastic", "bollinger",
    "atr", "true_range", "donchian", "rolling_max", "rolling_min",
    "volume_sma",
]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _series(x: pd.Series | pd.DataFrame, name: str = "") -> pd.Series:
    """Coerce to a float Series (a DataFrame column is picked by `name`)."""
    if isinstance(x, pd.DataFrame):
        if name and name in x.columns:
            x = x[name]
        else:                                   # pragma: no cover - misuse
            raise TypeError(f"expected a Series, got DataFrame without {name!r}")
    return pd.Series(x).astype(float)


def _check_window(window: int, label: str = "window") -> int:
    window = int(window)
    if window < 1:
        raise ValueError(f"{label} must be >= 1, got {window}")
    return window


def _wilder_rma(s: pd.Series, period: int) -> pd.Series:
    """
    Wilder's recursive moving average (a.k.a. RMA / smoothed MA).

        rma[t] = rma[t-1] + (x[t] - rma[t-1]) / period

    seeded with the simple average of the first `period` valid
    observations — the textbook Wilder warmup, which differs from a
    plain `ewm(alpha=1/period)` (that one seeds with the first value).

    Implemented by blanking everything before the seed bar, writing the
    SMA seed into the seed bar and letting `ewm(..., adjust=False,
    ignore_na=True)` run the recursion in C. Stays vectorized and keeps
    NaN warmup.
    """
    period = _check_window(period, "period")
    values = s.to_numpy(dtype=float, copy=False)
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size < period:
        return pd.Series(np.nan, index=s.index, name=s.name)

    seed_pos = int(valid[period - 1])
    seeded = s.astype(float).copy()
    seeded.iloc[:seed_pos] = np.nan
    seeded.iloc[seed_pos] = float(values[valid[:period]].mean())
    return seeded.ewm(alpha=1.0 / period, adjust=False, ignore_na=True).mean()


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average. NaN for the first `window - 1` bars."""
    window = _check_window(window)
    return _series(series).rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average (`alpha = 2/(span+1)`, recursive form).

    NaN for the first `span - 1` bars so the warmup is explicit rather
    than a heavily-biased early value.
    """
    span = _check_window(span, "span")
    return _series(series).ewm(span=span, adjust=False, min_periods=span).mean()


def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average of volume — the breakout confirmation gate."""
    return sma(volume, window)


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD. Returns columns ``macd`` (fast EMA - slow EMA), ``signal``
    (EMA of the MACD line) and ``hist`` (macd - signal).
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    close = _series(series)
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line.dropna(), signal).reindex(close.index)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using **Wilder** smoothing (RMA of gains and
    losses, SMA-seeded). First non-NaN value lands on bar `period`
    (bar 0 has no delta). Flat-price windows give RSI = 100 when there
    are no losses and 0 when there are no gains, matching Wilder.
    """
    period = _check_window(period, "period")
    close = _series(series)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # diff() puts NaN at bar 0; keep it so the seed uses `period` real deltas.
    gain[delta.isna()] = np.nan
    loss[delta.isna()] = np.nan

    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> rs is inf/NaN; Wilder's convention is 100 (or 0 if
    # there were no gains either... a dead-flat window scores 50 nowhere,
    # so resolve explicitly).
    flat = (avg_loss == 0) & (avg_gain == 0)
    out = out.where(~(avg_loss == 0), 100.0)
    out = out.where(~flat, 50.0)
    return out.where(avg_gain.notna() & avg_loss.notna())


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Fast stochastic oscillator. Returns columns ``%K`` and ``%D``.

        %K = 100 * (close - lowest_low(k)) / (highest_high(k) - lowest_low(k))
        %D = SMA(%K, d)

    The k-bar window **includes** the current bar — that is the standard
    definition and is not lookahead (it only uses bars <= i). Only
    breakout levels (`donchian`, `rolling_max`) exclude the current bar.
    A zero-range window yields NaN rather than a divide-by-zero.
    """
    k_period = _check_window(k_period, "k_period")
    hi, lo, cl = _series(high), _series(low), _series(close)
    hh = hi.rolling(k_period, min_periods=k_period).max()
    ll = lo.rolling(k_period, min_periods=k_period).min()
    rng = hh - ll
    k = 100.0 * (cl - ll) / rng.where(rng != 0)
    return pd.DataFrame({"%K": k, "%D": sma(k, d_period)})


# ---------------------------------------------------------------------------
# Bands / volatility
# ---------------------------------------------------------------------------

def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0,
              ddof: int = 0) -> pd.DataFrame:
    """
    Bollinger bands. Returns ``mid``, ``upper``, ``lower``, ``bandwidth``.

    `bandwidth = (upper - lower) / mid` — the squeeze metric used by the
    breakout strategy. `ddof=0` (population sigma) is the TA convention;
    pass `ddof=1` for the sample estimator if you prefer.
    """
    window = _check_window(window)
    close = _series(series)
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=ddof)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return pd.DataFrame({
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "bandwidth": (upper - lower) / mid.where(mid != 0),
    })


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True range: max(H-L, |H-C_prev|, |L-C_prev|).

    Bar 0 has no previous close, so it falls back to H-L (data <= 0 only).
    """
    hi, lo, cl = _series(high), _series(low), _series(close)
    prev = cl.shift(1)
    tr = pd.concat([
        hi - lo,
        (hi - prev).abs(),
        (lo - prev).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """
    Average True Range, **Wilder** smoothed (RMA of true range, SMA
    seed). First non-NaN value lands on bar `period - 1`.
    """
    return _wilder_rma(true_range(high, low, close), period)


# ---------------------------------------------------------------------------
# Breakout levels — these exclude the current bar
# ---------------------------------------------------------------------------

def rolling_max(series: pd.Series, window: int, *,
                exclude_current: bool = True) -> pd.Series:
    """
    Rolling maximum over `window` bars.

    With `exclude_current=True` (the default) the value at bar `i` is the
    max of bars `i-window .. i-1` — a `shift(1)` of the raw rolling max.
    Use this for breakout tests: `close[i] > rolling_max(close, 20)[i]`
    reads "close broke the prior 20-day high". Without the shift the
    current bar is inside its own reference window and the comparison can
    never be strictly true on the bar that sets the high, which is a
    subtle form of lookahead.
    """
    window = _check_window(window)
    out = _series(series).rolling(window, min_periods=window).max()
    return out.shift(1) if exclude_current else out


def rolling_min(series: pd.Series, window: int, *,
                exclude_current: bool = True) -> pd.Series:
    """Rolling minimum. See `rolling_max` for the `exclude_current` rule."""
    window = _check_window(window)
    out = _series(series).rolling(window, min_periods=window).min()
    return out.shift(1) if exclude_current else out


def donchian(high: pd.Series, low: pd.Series, window: int = 20, *,
             lower_window: int | None = None,
             exclude_current: bool = True) -> pd.DataFrame:
    """
    Donchian channel. Returns ``upper`` (highest high) and ``lower``
    (lowest low).

    **The current bar is excluded by default** (`exclude_current=True`):
    `upper[i]` is the highest high of bars `i-window .. i-1`, so
    `close[i] > upper[i]` is a genuine "close breaks the prior N-day
    high" entry. Setting `exclude_current=False` includes bar `i` in its
    own channel, which is fine for plotting but wrong for signals.

    `lower_window` lets the exit channel be shorter than the entry
    channel (the classic 55-in / 20-out turtle setup).
    """
    lo_win = window if lower_window is None else lower_window
    return pd.DataFrame({
        "upper": rolling_max(high, window, exclude_current=exclude_current),
        "lower": rolling_min(low, lo_win, exclude_current=exclude_current),
    })
