"""
Rule definitions + parameter grids (DESIGN.md "strategies.py — the grid").

Every strategy is a `StrategySpec`: an entry rule, an exit rule and a stop
config, built exclusively from `bursa_active.indicators`. No ad-hoc rolling
arithmetic lives here -- if a comparison needs a rolling window, it goes
through the indicator library so the whole package shares one NaN-warmup /
no-lookahead contract (see indicators.py's module docstring).

Breakout entries (Donchian, the volume-breakout family) compare `close`
against `rolling_max`/`donchian` with the library default `exclude_current
=True`, i.e. against the highest high of the *prior* N bars -- exactly the
"engine dev" note in indicators.py warns about. Bollinger-band comparisons
use the band as conventionally defined (current bar included in its own
rolling mean/std, which is standard for Bollinger Bands and still causal:
only data <= bar i is used).

Grid size
---------
Reading DESIGN.md's grid section literally (each `{a,b}` is a Cartesian
axis; a plain value is fixed) gives:

    trend/position : sma_cross 8, ema_cross 4, macd 8, donchian 8          = 28
    swing/pullback : rsi_pullback 48, stochastic 16, boll_mr 8, boll_bo 4  = 76
    volume breakout: 12
    ------------------------------------------------------------------------
    total                                                                 = 116

(MACD, stochastic and both Bollinger variants were widened in a DESIGN.md
amendment after an earlier grid -- built to the same literal-enumeration
rule -- landed at 84, short of the doc's own "~150-300" estimate. The RSI
family still supplies the largest single share (48) because it is the only
family crossed on four axes: entry config x trend filter x time stop x
fixed stop.)

Where DESIGN.md's phrasing was genuinely ambiguous we picked the reading
consistent with the rest of the doc:

* SMA/EMA cross and Donchian read "exit on <signal>. ATR trail {none, X}"
  -- the trail is an *additional* protective stop layered on top of the
  always-on signal exit (this mirrors how the engine already lets a stop
  and a signal exit coexist, stop wins on a tie). Donchian's trail is
  therefore crossed onto the entry/exit-window combos (2x2x2=8), not a
  replacement for the low-break exit.
* The volume-breakout family reads "exit ATR trail {2.5,3.0} *or* close<
  SMA20" -- an explicit "or" between three mutually exclusive exit
  mechanisms, so those three are enumerated as alternatives (not crossed
  additively).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import indicators as ind


__all__ = ["StrategySpec", "build_grid"]

SignalFn = Callable[[pd.DataFrame], "tuple[pd.Series, pd.Series]"]


# ---------------------------------------------------------------------------
# Spec container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """One fully-parameterised rule: entry, exit and stop."""

    id: str
    family: str
    timeframe: str          # "swing" | "position"
    description: str        # plain English entry/exit/stop rule
    signal_fn: SignalFn      # df -> (entries, exits), causal, indicator-only
    stop_kind: str | None    # "atr_trail" | "fixed_pct" | None
    stop_param: float = 0.0
    time_stop_days: int | None = None

    def build_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Compute (entries, exits) boolean Series aligned to `df.index`."""
        entries, exits = self.signal_fn(df)
        return entries, exits


# ---------------------------------------------------------------------------
# Small causal helpers (still just indicator-library plumbing, no rolling
# math of their own -- comparisons and cross detection only).
# ---------------------------------------------------------------------------

def _cross_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True the bar `fast` first closes above `slow` (a strict crossover)."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _cross_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True the bar `fast` first closes below `slow` (a strict crossover)."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def _always_false(index: pd.Index) -> pd.Series:
    """Exit signal placeholder for rules that exit purely via the stop."""
    return pd.Series(False, index=index)


# ---------------------------------------------------------------------------
# Family 1: SMA cross (trend / position)
# ---------------------------------------------------------------------------

def _make_sma_cross(fast: int, slow: int) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df["Close"]
        f, s = ind.sma(close, fast), ind.sma(close, slow)
        return _cross_up(f, s), _cross_down(f, s)
    return fn


def _sma_cross_specs() -> list[StrategySpec]:
    specs = []
    for fast in (10, 20):
        for slow in (50, 100):
            for trail in (None, 3.0):
                trail_tag = "notrail" if trail is None else f"trail{trail:g}"
                trail_desc = (
                    "" if trail is None else
                    f", or on a close below a {trail:g}x ATR(14) trailing stop"
                )
                specs.append(StrategySpec(
                    id=f"sma_cross_{fast}_{slow}_{trail_tag}",
                    family="trend_sma_cross",
                    timeframe="position",
                    description=(
                        f"Enter long when the {fast}-day SMA crosses above the "
                        f"{slow}-day SMA; exit on the reverse cross{trail_desc}."
                    ),
                    signal_fn=_make_sma_cross(fast, slow),
                    stop_kind=None if trail is None else "atr_trail",
                    stop_param=0.0 if trail is None else trail,
                ))
    return specs


# ---------------------------------------------------------------------------
# Family 2: EMA cross (trend / position)
# ---------------------------------------------------------------------------

def _make_ema_cross(fast: int, slow: int) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df["Close"]
        f, s = ind.ema(close, fast), ind.ema(close, slow)
        return _cross_up(f, s), _cross_down(f, s)
    return fn


def _ema_cross_specs() -> list[StrategySpec]:
    specs = []
    for fast, slow in ((12, 50), (20, 100)):
        for trail in (None, 3.0):
            trail_tag = "notrail" if trail is None else f"trail{trail:g}"
            trail_desc = (
                "" if trail is None else
                f", or on a close below a {trail:g}x ATR(14) trailing stop"
            )
            specs.append(StrategySpec(
                id=f"ema_cross_{fast}_{slow}_{trail_tag}",
                family="trend_ema_cross",
                timeframe="position",
                description=(
                    f"Enter long when the {fast}-day EMA crosses above the "
                    f"{slow}-day EMA; exit on the reverse cross{trail_desc}."
                ),
                signal_fn=_make_ema_cross(fast, slow),
                stop_kind=None if trail is None else "atr_trail",
                stop_param=0.0 if trail is None else trail,
            ))
    return specs


# ---------------------------------------------------------------------------
# Family 3: MACD trend (trend / position)
# ---------------------------------------------------------------------------

def _make_macd_trend(fast: int, slow: int, signal: int, trend_sma: int) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df["Close"]
        m = ind.macd(close, fast, slow, signal)
        trend = close > ind.sma(close, trend_sma)
        entries = _cross_up(m["macd"], m["signal"]) & trend
        exits = _cross_down(m["macd"], m["signal"])
        return entries, exits
    return fn


def _macd_specs() -> list[StrategySpec]:
    specs = []
    for fast, slow, signal in ((12, 26, 9), (8, 17, 9)):
        for trend_sma in (50, 100):
            for trail in (None, 3.0):
                trail_tag = "notrail" if trail is None else f"trail{trail:g}"
                trail_desc = (
                    "" if trail is None else
                    f", or on a close below a {trail:g}x ATR(14) trailing stop"
                )
                specs.append(StrategySpec(
                    id=f"macd_{fast}_{slow}_{signal}_sma{trend_sma}_{trail_tag}",
                    family="trend_macd",
                    timeframe="position",
                    description=(
                        f"Enter long when the MACD({fast},{slow},{signal}) line "
                        f"crosses above its signal line while close > "
                        f"SMA({trend_sma}) (uptrend filter); exit on the reverse "
                        f"MACD/signal cross{trail_desc}."
                    ),
                    signal_fn=_make_macd_trend(fast, slow, signal, trend_sma),
                    stop_kind=None if trail is None else "atr_trail",
                    stop_param=0.0 if trail is None else trail,
                ))
    return specs


# ---------------------------------------------------------------------------
# Family 4: Donchian breakout (trend / position)
# ---------------------------------------------------------------------------

def _make_donchian(entry_w: int, exit_w: int) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        high, low, close = df["High"], df["Low"], df["Close"]
        entry_ch = ind.donchian(high, low, entry_w)
        exit_ch = ind.donchian(high, low, exit_w)
        entries = close > entry_ch["upper"]
        exits = close < exit_ch["lower"]
        return entries, exits
    return fn


def _donchian_specs() -> list[StrategySpec]:
    specs = []
    for entry_w in (20, 55):
        for exit_w in (10, 20):
            for trail in (None, 3.0):
                trail_tag = "notrail" if trail is None else f"trail{trail:g}"
                trail_desc = (
                    "" if trail is None else
                    f", or on a close below a {trail:g}x ATR(14) trailing stop"
                )
                specs.append(StrategySpec(
                    id=f"donchian_entry{entry_w}_exitlow{exit_w}_{trail_tag}",
                    family="trend_donchian",
                    timeframe="position",
                    description=(
                        f"Enter long on a close above the prior {entry_w}-day "
                        f"high (Donchian breakout); exit on a close below the "
                        f"prior {exit_w}-day low{trail_desc}."
                    ),
                    signal_fn=_make_donchian(entry_w, exit_w),
                    stop_kind=None if trail is None else "atr_trail",
                    stop_param=0.0 if trail is None else trail,
                ))
    return specs


# ---------------------------------------------------------------------------
# Family 5: RSI pullback (swing)
# ---------------------------------------------------------------------------

def _make_rsi_pullback(entry_period: int, entry_thr: float,
                       exit_period: int, exit_thr: float,
                       trend_sma: int) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df["Close"]
        trend = close > ind.sma(close, trend_sma)
        entry_rsi = ind.rsi(close, entry_period)
        exit_rsi = ind.rsi(close, exit_period) if exit_period != entry_period else entry_rsi
        entries = (entry_rsi < entry_thr) & trend
        exits = exit_rsi > exit_thr
        return entries, exits
    return fn


#: (entry_period, entry_threshold, exit_period, exit_threshold) combos, per
#: DESIGN.md: RSI(2) always exits at 75; RSI(14) exits at {60,70}.
_RSI_ENTRY_EXIT_COMBOS: list[tuple[int, float, int, float]] = [
    (2, 10, 2, 75),
    (2, 25, 2, 75),
    (14, 30, 14, 60),
    (14, 30, 14, 70),
    (14, 40, 14, 60),
    (14, 40, 14, 70),
]


def _rsi_pullback_specs() -> list[StrategySpec]:
    specs = []
    for entry_p, entry_thr, exit_p, exit_thr in _RSI_ENTRY_EXIT_COMBOS:
        for trend_sma in (50, 100):
            for time_stop in (10, 15):
                for stop_pct in (5.0, 8.0):
                    spec_id = (
                        f"rsi{entry_p}_pullback_sma{trend_sma}_"
                        f"lt{entry_thr:g}_gt{exit_thr:g}_"
                        f"time{time_stop}_stop{stop_pct:g}"
                    )
                    specs.append(StrategySpec(
                        id=spec_id,
                        family="swing_rsi_pullback",
                        timeframe="swing",
                        description=(
                            f"Enter long when close > SMA({trend_sma}) (uptrend "
                            f"filter) and RSI({entry_p}) < {entry_thr:g}; exit "
                            f"when RSI({exit_p}) > {exit_thr:g}, after "
                            f"{time_stop} trading days, or a {stop_pct:g}% "
                            f"stop-loss from entry -- whichever comes first."
                        ),
                        signal_fn=_make_rsi_pullback(
                            entry_p, entry_thr, exit_p, exit_thr, trend_sma,
                        ),
                        stop_kind="fixed_pct",
                        stop_param=stop_pct,
                        time_stop_days=time_stop,
                    ))
    return specs


# ---------------------------------------------------------------------------
# Family 6: Stochastic (swing)
# ---------------------------------------------------------------------------

def _make_stochastic(k_period: int, k_thresh: float, trend_sma: int) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df["Close"]
        st = ind.stochastic(df["High"], df["Low"], close, k_period, 3)
        k, d = st["%K"], st["%D"]
        trend = close > ind.sma(close, trend_sma)
        entries = _cross_up(k, d) & (k < k_thresh) & trend
        exits = k > 80
        return entries, exits
    return fn


def _stochastic_specs() -> list[StrategySpec]:
    specs = []
    for k_period in (10, 14):
        for k_thresh in (20, 25):
            for trend_sma in (50, 100):
                for stop_pct in (5.0, 8.0):
                    specs.append(StrategySpec(
                        id=(
                            f"stoch_k{k_period}d3_lt{k_thresh:g}_sma{trend_sma}_"
                            f"stop{stop_pct:g}_time15"
                        ),
                        family="swing_stochastic",
                        timeframe="swing",
                        description=(
                            f"Enter long when close > SMA({trend_sma}) (uptrend "
                            f"filter) and %K({k_period},3) crosses above %D "
                            f"while %K < {k_thresh:g}; exit when %K > 80, after "
                            f"15 trading days, or a {stop_pct:g}% stop-loss, "
                            f"whichever comes first."
                        ),
                        signal_fn=_make_stochastic(k_period, k_thresh, trend_sma),
                        stop_kind="fixed_pct",
                        stop_param=stop_pct,
                        time_stop_days=15,
                    ))
    return specs


# ---------------------------------------------------------------------------
# Family 7: Bollinger mean-revert (swing)
# ---------------------------------------------------------------------------

def _make_bollinger_mr(num_std: float, trend_sma: int) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df["Close"]
        bb = ind.bollinger(close, 20, num_std)
        trend = close > ind.sma(close, trend_sma)
        entries = (close < bb["lower"]) & trend
        exits = close > bb["mid"]
        return entries, exits
    return fn


def _bollinger_mr_specs() -> list[StrategySpec]:
    specs = []
    for num_std in (2.0, 2.5):
        for trend_sma in (50, 100):
            for stop_pct in (5.0, 8.0):
                std_tag = f"{num_std:g}".replace(".", "")
                specs.append(StrategySpec(
                    id=f"bollinger_mr_20_{std_tag}_sma{trend_sma}_stop{stop_pct:g}_time15",
                    family="swing_bollinger_mr",
                    timeframe="swing",
                    description=(
                        f"Enter long when close < the lower Bollinger band "
                        f"(20,{num_std:g}) while close > SMA({trend_sma}) (a "
                        f"pullback within an uptrend); exit when close crosses "
                        f"back above the middle band, after 15 trading days, "
                        f"or on a {stop_pct:g}% stop-loss, whichever comes "
                        f"first."
                    ),
                    signal_fn=_make_bollinger_mr(num_std, trend_sma),
                    stop_kind="fixed_pct",
                    stop_param=stop_pct,
                    time_stop_days=15,
                ))
    return specs


# ---------------------------------------------------------------------------
# Family 8: Bollinger breakout (swing)
# ---------------------------------------------------------------------------

#: Trailing window for the bandwidth-squeeze quantile filter.
_BB_SQUEEZE_WINDOW = 120


def _make_bollinger_bo(squeeze_pctile: float, trail: float) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        close = df["Close"]
        bb = ind.bollinger(close, 20, 2.0)
        # Squeeze is measured on *yesterday's* bandwidth against its own
        # trailing 120-day distribution (shift(1) before both the value and
        # its rolling quantile) so today's breakout bar -- whose own
        # bandwidth typically widens on the breakout itself -- can't
        # influence whether "yesterday was squeezed". Still fully causal:
        # only bars <= i-1 feed bar i.
        bw_prev = bb["bandwidth"].shift(1)
        bw_thresh = bw_prev.rolling(
            _BB_SQUEEZE_WINDOW, min_periods=_BB_SQUEEZE_WINDOW,
        ).quantile(squeeze_pctile)
        squeeze = bw_prev <= bw_thresh
        entries = (close > bb["upper"]) & squeeze
        exits = close < bb["mid"]
        return entries, exits
    return fn


def _bollinger_bo_specs() -> list[StrategySpec]:
    specs = []
    for squeeze_pct in (25, 40):
        for trail in (2.5, 3.0):
            specs.append(StrategySpec(
                id=f"bollinger_bo_20_2_sq{squeeze_pct}_trail{trail:g}".replace(".", ""),
                family="swing_bollinger_bo",
                timeframe="swing",
                description=(
                    f"Enter long when close breaks above the upper Bollinger "
                    f"band (20,2), but only if yesterday's bandwidth was in "
                    f"the bottom {squeeze_pct}% of its trailing 120-day range "
                    f"(a volatility squeeze); exit when close falls below the "
                    f"middle band, or on a {trail:g}x ATR(14) trailing stop, "
                    f"whichever comes first."
                ),
                signal_fn=_make_bollinger_bo(squeeze_pct / 100.0, trail),
                stop_kind="atr_trail",
                stop_param=trail,
            ))
    return specs


# ---------------------------------------------------------------------------
# Family 9: Volume-confirmed breakout (swing)
# ---------------------------------------------------------------------------

def _make_volume_breakout(entry_w: int, vol_mult: float,
                          exit_kind: str) -> SignalFn:
    def fn(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        high, close, volume = df["High"], df["Close"], df["Volume"]
        hi = ind.rolling_max(high, entry_w, exclude_current=True)
        vsma = ind.volume_sma(volume, 20)
        entries = (close > hi) & (volume > vol_mult * vsma)
        if exit_kind == "sma20":
            exits = close < ind.sma(close, 20)
        else:
            exits = _always_false(df.index)      # exit purely via the stop
        return entries, exits
    return fn


def _volume_breakout_specs() -> list[StrategySpec]:
    specs = []
    for entry_w in (20, 55):
        for vol_mult in (1.5, 2.0):
            vol_tag = f"{vol_mult:g}".replace(".", "")
            for exit_kind, stop_kind, stop_param, exit_desc in (
                ("trail25", "atr_trail", 2.5, "via a 2.5x ATR(14) trailing stop"),
                ("trail30", "atr_trail", 3.0, "via a 3.0x ATR(14) trailing stop"),
                ("sma20", None, 0.0, "when close falls below SMA(20)"),
            ):
                specs.append(StrategySpec(
                    id=f"volbreak_hi{entry_w}_vol{vol_tag}_{exit_kind}",
                    family="swing_volume_breakout",
                    timeframe="swing",
                    description=(
                        f"Enter long on a close above the prior {entry_w}-day "
                        f"high with volume > {vol_mult:g}x its 20-day average "
                        f"(a volume-confirmed breakout); exit {exit_desc}."
                    ),
                    signal_fn=_make_volume_breakout(entry_w, vol_mult, exit_kind),
                    stop_kind=stop_kind,
                    stop_param=stop_param,
                ))
    return specs


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def build_grid() -> list[StrategySpec]:
    """Enumerate every family/parameter combo in DESIGN.md's grid section."""
    specs: list[StrategySpec] = []
    specs += _sma_cross_specs()
    specs += _ema_cross_specs()
    specs += _macd_specs()
    specs += _donchian_specs()
    specs += _rsi_pullback_specs()
    specs += _stochastic_specs()
    specs += _bollinger_mr_specs()
    specs += _bollinger_bo_specs()
    specs += _volume_breakout_specs()

    ids = [s.id for s in specs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate spec ids in build_grid(): {dupes}")
    return specs
