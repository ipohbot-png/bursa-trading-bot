"""
Per-run performance metrics.

Everything here is computed from the engine's output: the trade list and
the daily equity curve. All returns are **after** Bursa/Maybank fees,
because the engine charges them on both legs before the curve is marked.

Conventions (chosen once, so grid CSVs stay comparable):

* `max_drawdown_pct` is a **positive magnitude** (12.3 means the equity
  curve fell 12.3% peak-to-trough).
* `profit_factor` is capped at `PROFIT_FACTOR_CAP` (999.0) when there are
  no losing trades. `inf` is not JSON/CSV friendly and would sort weirdly
  in the ranking; a cap keeps "all winners" at the top of the sort while
  staying serialisable. With zero trades it is 0.0.
* `fee_drag_pct` is `total_fees / |gross pnl| * 100`, i.e. what share of
  the pre-fee edge the broker took. It is `None` when gross P&L is
  exactly zero (undefined, not zero).
* `sharpe` and `cagr` are **`None`** — not 0.0, not a number — whenever
  the run is degenerate: no trades at all, or an equity curve that fell
  to `RUIN_FRACTION` (5%) of starting cash or below. Near-zero equity
  makes daily percentage returns explode (a +320% "day" out of RM 30 of
  residual cash) and can flip the sign of the Sharpe ratio, so a −80%
  strategy scores +0.45. There is no meaningful risk-adjusted number to
  report on a blown-up account; `None` says so. **Ranking code must
  treat `None` as the worst possible value, never as zero or missing.**
* `sharpe` is built on daily **log** returns, `ln(equity[t]/equity[t-1])`,
  annualised by `sqrt(252)`, risk-free 0. Log returns are used precisely
  so the sign of the Sharpe ratio always agrees with the sign of
  `total_return_pct`: the mean log return is positive exactly when the
  curve ends above where it began. (Arithmetic returns break that —
  volatility drag can hand a losing run a positive mean daily return.)
  A non-positive equity value anywhere makes the log undefined and is
  ruin by any definition, so it yields `None`.
* `insufficient_sample` flags `n_trades < MIN_TRADES` (8) — DESIGN.md
  says those rules are excluded from ranking.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:                                   # pragma: no cover
    from .engine import BacktestResult


__all__ = [
    "compute_metrics", "MIN_TRADES", "PROFIT_FACTOR_CAP", "TRADING_DAYS",
    "RUIN_FRACTION",
]

#: Equity at or below this fraction of starting cash = ruin; `sharpe` and
#: `cagr` become None because the ratios stop meaning anything there.
RUIN_FRACTION = 0.05

#: DESIGN.md: rules with fewer in-sample trades are not ranked.
MIN_TRADES = 8

#: Stand-in for an infinite profit factor (no losing trades).
PROFIT_FACTOR_CAP = 999.0

#: Bursa trades ~252 days a year; used to annualise Sharpe and CAGR.
TRADING_DAYS = 252


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a positive percentage."""
    eq = equity.dropna()
    if len(eq) < 2:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak.where(peak != 0) - 1.0).min()
    if not np.isfinite(dd):
        return 0.0
    return round(abs(float(dd)) * 100.0, 4)


def _sharpe(equity: pd.Series) -> float | None:
    """
    Annualised Sharpe of daily **log** equity returns, risk-free = 0.

        r[t] = ln(equity[t] / equity[t-1])
        sharpe = mean(r) / stdev(r) * sqrt(252)

    Log returns make the sign of the ratio agree with the sign of the
    compounded return: `mean(ln ratios) > 0` exactly when the curve
    finished above where it started. Arithmetic returns do not have that
    property — volatility drag lets a losing curve post a positive mean
    daily return.

    Returns None when the curve is unusable: a non-positive equity value
    anywhere (log undefined — that is ruin by any definition). A flat
    curve has zero variance and scores 0.0.
    """
    eq = equity.dropna()
    if len(eq) < 3:
        return 0.0
    values = eq.to_numpy(dtype=float)
    if np.any(values <= 0):
        return None                      # wiped out: no meaningful ratio
    ratios = values[1:] / values[:-1]
    if ratios.size == 0 or np.any(ratios <= 0) or not np.all(np.isfinite(ratios)):
        return None
    rets = np.log(ratios)
    sd = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return round(float(np.mean(rets)) / sd * float(np.sqrt(TRADING_DAYS)), 4)


def _cagr(start: float, end: float, index: pd.Index) -> float:
    """Compound annual growth rate in %, on calendar elapsed time."""
    if start <= 0 or end <= 0 or len(index) < 2:
        return 0.0
    try:
        years = (index[-1] - index[0]).days / 365.25
    except (TypeError, AttributeError):             # non-datetime index
        years = len(index) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return round(((end / start) ** (1.0 / years) - 1.0) * 100.0, 4)


def _buyhold_return_pct(df: pd.DataFrame) -> float:
    """
    Close-to-close return over the same window, **before** fees.

    Deliberately un-netted: it is the "do nothing clever" yardstick, and
    one round trip of fees on RM 20k is ~0.4%, which would only flatter
    the strategies we are testing.
    """
    if df is None or "Close" not in getattr(df, "columns", []):
        return 0.0
    close = df["Close"].astype(float).dropna()
    if len(close) < 2 or close.iloc[0] <= 0:
        return 0.0
    return round((float(close.iloc[-1]) / float(close.iloc[0]) - 1.0) * 100.0, 4)


def compute_metrics(result: "BacktestResult", df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise one backtest run.

    Parameters
    ----------
    result
        The `BacktestResult` produced by `engine.run_backtest`.
    df
        The OHLCV frame the run was executed on — used for the
        buy-and-hold benchmark over the identical window.

    Returns
    -------
    dict
        See the module docstring for the conventions behind the edge
        cases (0 trades, no losers, zero gross P&L).
    """
    trades = list(result.trades)
    equity = result.equity if result.equity is not None else pd.Series(dtype=float)
    eq = equity.dropna()

    start = float(result.starting_cash) or (float(eq.iloc[0]) if len(eq) else 0.0)
    end = float(eq.iloc[-1]) if len(eq) else start

    rets = np.array([t.return_pct for t in trades], dtype=float)
    nets = np.array([t.pnl_net for t in trades], dtype=float)
    gross = float(np.sum([t.pnl_gross for t in trades])) if trades else 0.0
    fees = float(np.sum([t.entry_fees + t.exit_fees for t in trades])) if trades else 0.0

    wins = nets[nets > 0]
    losses = nets[nets < 0]

    if not trades:
        profit_factor = 0.0
    elif losses.size == 0:
        profit_factor = PROFIT_FACTOR_CAP if wins.size else 0.0
    else:
        profit_factor = round(float(wins.sum()) / abs(float(losses.sum())), 4)
        profit_factor = min(profit_factor, PROFIT_FACTOR_CAP)

    fee_drag_pct: float | None
    if gross == 0.0:
        fee_drag_pct = None
    else:
        fee_drag_pct = round(fees / abs(gross) * 100.0, 4)

    # Ruin / no-trade guard: ratio metrics are not merely small on a
    # wiped-out curve, they are actively misleading (see module docstring).
    ruined = bool(len(eq)) and start > 0 and float(eq.min()) <= RUIN_FRACTION * start
    degenerate = (not trades) or ruined
    sharpe: float | None = None if degenerate else _sharpe(equity)
    cagr: float | None = None if degenerate else _cagr(start, end, eq.index)

    return {
        "total_return_pct":          round((end / start - 1.0) * 100.0, 4) if start else 0.0,
        "cagr":                      cagr,
        "max_drawdown_pct":          _max_drawdown_pct(equity),
        "win_rate":                  round(100.0 * float((nets > 0).sum()) / len(trades), 4) if trades else 0.0,
        "profit_factor":             profit_factor,
        "avg_win_pct":               round(float(rets[nets > 0].mean()), 4) if wins.size else 0.0,
        "avg_loss_pct":              round(float(rets[nets < 0].mean()), 4) if losses.size else 0.0,
        "expectancy_pct_per_trade":  round(float(rets.mean()), 4) if trades else 0.0,
        "n_trades":                  len(trades),
        "avg_holding_days":          round(float(np.mean([t.holding_days for t in trades])), 4) if trades else 0.0,
        "exposure_pct":              round(float(result.exposure_pct), 4),
        "total_fees_rm":             round(fees, 2),
        "fee_drag_pct":              fee_drag_pct,
        "buyhold_return_pct":        _buyhold_return_pct(df),
        "sharpe":                    sharpe,
        # --- run bookkeeping, not headline metrics ---
        "skipped_trades":            int(result.skipped),
        "open_at_end":               bool(result.open_at_end),
        "insufficient_sample":       len(trades) < MIN_TRADES,
    }
