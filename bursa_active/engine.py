"""
Single-stock, long-only, event-driven backtester.

Execution model (DESIGN.md "Execution & fee model"):

* Signals are computed on the close of day `t`; the fill happens at the
  **open of day t+1**. Never a same-bar fill.
* If the t+1 open is missing or zero we fill at the t+1 close; if there
  is no t+1 bar at all the order is carried to the next available bar.
* One position at a time, no pyramiding, no shorting, no leverage.
* Position size: `floor(min(budget_rm, cash) / fill_price / 100) * 100`
  (Bursa board lot). A **fixed RM budget per trade, capped by the cash
  actually on hand** — no margin, no borrowing, and no compounding above
  `budget_rm` when the account grows. Zero affordable lots → the entry
  is skipped and counted in `BacktestResult.skipped`.
* Fills only happen on bars that actually traded: a bar with no usable
  price, or with `Volume` zero/NaN, cannot fill an order — it is carried
  forward to the next bar that has both.
* Fees are charged on both legs through `bursa_momentum.fees.one_leg_fee`.
* Stops are evaluated on the **close** only. Daily bars cannot honestly
  model an intraday stop fill, so a close below the stop exits at the
  next open. Say so in any report built on this engine.

In-position priority on each close: **stop → time stop → exit signal**
(the stop wins when several fire on the same bar).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bursa_momentum.fees import one_leg_fee

from . import indicators


__all__ = ["Trade", "BacktestResult", "run_backtest", "ATR_PERIOD"]

#: ATR lookback used by the "atr_trail" stop (DESIGN.md says ATR14).
ATR_PERIOD = 14

#: Accepted values for `stop_kind`.
STOP_KINDS = ("atr_trail", "fixed_pct", None)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """One completed round trip. All money values are RM."""

    entry_date:   pd.Timestamp
    entry_price:  float          # actual fill price (next bar's open)
    shares:       int
    exit_date:    pd.Timestamp
    exit_price:   float
    entry_fees:   float
    exit_fees:    float
    pnl_gross:    float          # shares * (exit_price - entry_price)
    pnl_net:      float          # pnl_gross - entry_fees - exit_fees
    return_pct:   float          # pnl_net / (shares * entry_price) * 100
    holding_days: int            # trading bars between the two fills
    # "signal" | "stop" | "trail" | "time" | "eod" | "eod_no_price"
    exit_reason:  str
    open_at_end:  bool = False   # True when closed by the end of the data


@dataclass
class BacktestResult:
    """Everything one (rule, ticker, window) run produced."""

    trades:        list[Trade] = field(default_factory=list)
    equity:        pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    exposure_pct:  float = 0.0        # % of bars holding a position
    skipped:       int = 0            # entries dropped: budget < 1 board lot
    open_at_end:   bool = False
    starting_cash: float = 0.0
    metrics:       dict[str, Any] = field(default_factory=dict)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def trades_frame(self) -> pd.DataFrame:
        """Trades as a DataFrame, ready for `to_csv`."""
        if not self.trades:
            return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
        return pd.DataFrame([t.__dict__ for t in self.trades])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_series(sig: pd.Series | None, index: pd.Index, label: str) -> np.ndarray:
    """Align a signal Series to `index` as a plain bool array."""
    if sig is None:
        return np.zeros(len(index), dtype=bool)
    if not isinstance(sig, pd.Series):
        raise TypeError(f"{label} must be a pd.Series of bools")
    aligned = sig.reindex(index)
    return aligned.fillna(False).astype(bool).to_numpy()


def _column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        raise KeyError(f"df is missing required column {name!r}")
    return df[name].astype(float).to_numpy()


def _lots(budget: float, price: float) -> int:
    """
    Board-lot sizing: whole 100-share lots affordable at `price`.

    `budget` is already the cash-capped budget (see `run_backtest`), so a
    negative or tiny budget simply yields 0 lots rather than borrowing.
    """
    if not np.isfinite(price) or price <= 0 or not np.isfinite(budget) or budget <= 0:
        return 0
    return int(math.floor(budget / price / 100.0)) * 100


def _affordable_lots(budget: float, cash: float, price: float) -> int:
    """
    Lots for one entry: `floor(min(budget, cash) / price / 100) * 100`,
    then shaved by whole lots until the **fee also fits** in cash.

    Without the second step an entry sized at exactly `cash` still ends
    up borrowing the brokerage (~RM 20-40), which is how a "no margin"
    sim quietly runs a negative balance. Shaving costs at most a couple
    of iterations — fees are ~0.4% of notional.
    """
    lots = _lots(min(float(budget), float(cash)), price)
    while lots > 0 and lots * price + one_leg_fee(lots * price) > cash:
        lots -= 100
    return lots


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    *,
    stop_kind: str | None = None,
    stop_param: float = 0.0,
    time_stop_days: int | None = None,
    budget_rm: float = 20_000.0,
) -> BacktestResult:
    """
    Walk `df` bar by bar and simulate the rule described by `entries`,
    `exits` and the stop.

    Parameters
    ----------
    df
        Daily OHLCV with at least `Open`, `High`, `Low`, `Close`, sorted
        ascending by a DatetimeIndex.
    entries, exits
        Boolean Series aligned to `df.index`, True on the **signal** day
        `t`. The fill lands on `t+1`. Missing labels are treated as False.
    stop_kind
        - ``"atr_trail"``: ratcheting trailing stop.
          ``trail = max(trail_prev, close - stop_param * ATR14)``; the
          trail never moves down. Breach when ``close < trail``, exit
          reason ``"trail"``.
        - ``"fixed_pct"``: breach when
          ``close < entry_price * (1 - stop_param / 100)``, exit reason
          ``"stop"``.
        - ``None``: no price stop.
    stop_param
        ATR multiple, or stop percentage — see `stop_kind`.
    time_stop_days
        Exit once ``holding_days >= time_stop_days``, where holding days
        are **trading bars** since the entry fill (0 on the fill bar).
        Exit reason ``"time"``. Note the one-bar execution lag:
        ``time_stop_days=N`` breaches on the close of bar ``entry + N``
        and fills at the next open, so the recorded
        ``Trade.holding_days`` is ``N + 1``, not ``N``.
    budget_rm
        Budget per trade **and** the starting equity of the curve. Each
        entry is sized at ``min(budget_rm, cash)``: the account never
        borrows, and profits above `budget_rm` are not re-risked.

    Returns
    -------
    BacktestResult
        Trades, the daily equity curve (`cash + shares * close`, starting
        at `budget_rm`), exposure, skipped-entry count and the metrics
        dict from `metrics.compute_metrics`.

    Notes
    -----
    * Entry and exit signals on the same bar while flat cancel out — the
      engine refuses to open a position it would immediately queue for
      sale (no same-bar round trip).
    * A position still open on the last bar is closed at that bar's
      close with ``exit_reason="eod"`` and ``open_at_end=True``. Exit
      fees are charged, so the curve is not flattered by an un-exited
      winner. If the last bar has no price, the mark-out walks back to
      the most recent finite close; if the frame has no finite close at
      all the position cannot be valued and the trade is recorded with
      ``exit_reason="eod_no_price"`` at zero P&L (entry fees still sunk).
    * ``BacktestResult.skipped`` counts every entry signal that did not
      become a position: too little cash for one board lot, and signals
      on the final bar that have no next bar to fill on.
    """
    if stop_kind not in STOP_KINDS:
        raise ValueError(f"stop_kind must be one of {STOP_KINDS}, got {stop_kind!r}")
    if df is None or len(df) == 0:
        # Empty window (a slice with no bars). Return the *full* metrics
        # dict, not {}, so downstream CSV/ranking code never KeyErrors.
        from .metrics import compute_metrics

        empty = BacktestResult(
            equity=pd.Series(dtype=float), starting_cash=float(budget_rm),
        )
        empty.metrics = compute_metrics(empty, df)
        return empty
    if not df.index.is_monotonic_increasing:
        raise ValueError("df.index must be sorted ascending")

    index = df.index
    n = len(index)

    open_ = _column(df, "Open")
    close = _column(df, "Close")
    # Mark-to-market on the last known close so a NaN bar can't blank equity.
    mark = pd.Series(close, index=index).ffill().to_numpy()

    # A bar with no volume is a bar on which the stock did not trade — we
    # cannot have been filled there. Frames without a Volume column are
    # assumed tradeable throughout.
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float).to_numpy()
        tradeable = np.isfinite(vol) & (vol > 0)
    else:
        tradeable = np.ones(n, dtype=bool)

    entry_sig = _bool_series(entries, index, "entries")
    exit_sig = _bool_series(exits, index, "exits")

    atr_arr: np.ndarray | None = None
    if stop_kind == "atr_trail":
        atr_arr = indicators.atr(
            df["High"], df["Low"], df["Close"], ATR_PERIOD
        ).to_numpy(dtype=float)

    cash = float(budget_rm)
    shares = 0
    equity = np.full(n, np.nan)
    held = np.zeros(n, dtype=bool)

    trades: list[Trade] = []
    skipped = 0

    # Open-position state
    entry_i = -1
    entry_price = 0.0
    entry_fees = 0.0
    trail = np.nan

    # Pending order: (side, reason) to be filled on the current bar.
    pending: tuple[str, str] | None = None

    def fill_price(i: int) -> float | None:
        """
        Price an order can fill at on bar `i`: the open, else the close.

        Returns None — order carried to the next bar — when the bar has
        no usable price **or** did not trade (Volume 0/NaN).
        """
        if not tradeable[i]:
            return None
        px = open_[i]
        if np.isfinite(px) and px > 0:
            return float(px)
        px = close[i]
        if np.isfinite(px) and px > 0:
            return float(px)
        return None

    for i in range(n):
        # --- 1. fill whatever was queued yesterday ------------------------
        if pending is not None:
            px = fill_price(i)
            if px is not None:
                side, reason = pending
                if side == "BUY":
                    # Fixed RM budget per trade, capped by cash on hand:
                    # the sim never borrows and never compounds above budget.
                    lots = _affordable_lots(budget_rm, cash, px)
                    if lots <= 0:
                        skipped += 1          # cash < one board lot
                    else:
                        fee = one_leg_fee(lots * px)
                        cash -= lots * px + fee
                        shares = lots
                        entry_i, entry_price, entry_fees = i, px, fee
                        trail = np.nan
                    pending = None
                else:                          # SELL
                    fee = one_leg_fee(shares * px)
                    cash += shares * px - fee
                    trades.append(_make_trade(
                        index, entry_i, entry_price, shares, entry_fees,
                        i, px, fee, reason, open_at_end=False,
                    ))
                    shares = 0
                    entry_i, entry_price, entry_fees, trail = -1, 0.0, 0.0, np.nan
                    pending = None
            # else: no usable price on this bar — keep the order queued.

        # --- 2. evaluate the close of bar i -------------------------------
        c = close[i]
        if shares > 0:
            reason: str | None = None

            if stop_kind == "atr_trail":
                if atr_arr is not None and np.isfinite(atr_arr[i]) and np.isfinite(c):
                    candidate = c - float(stop_param) * atr_arr[i]
                    trail = candidate if not np.isfinite(trail) else max(trail, candidate)
                if np.isfinite(trail) and np.isfinite(c) and c < trail:
                    reason = "trail"
            elif stop_kind == "fixed_pct":
                level = entry_price * (1.0 - float(stop_param) / 100.0)
                if np.isfinite(c) and c < level:
                    reason = "stop"

            if reason is None and time_stop_days is not None:
                if (i - entry_i) >= int(time_stop_days):
                    reason = "time"

            if reason is None and exit_sig[i]:
                reason = "signal"

            if reason is not None and pending is None and i + 1 < n:
                pending = ("SELL", reason)
            elif reason is not None and i + 1 >= n:
                pass                          # handled by the EOD close below

        elif pending is None and entry_sig[i] and not exit_sig[i]:
            if i + 1 < n:
                # Flat: enter, unless the same bar also says exit
                # (no same-bar round trip).
                pending = ("BUY", "entry")
            else:
                # Signal on the final bar — no next open to fill at.
                skipped += 1

        # --- 3. mark to market --------------------------------------------
        # Flat -> equity is just cash, even if this bar has no price at all.
        # Holding with no price anywhere yet -> hold at the entry fill so a
        # NaN close cannot blank the curve (and poison drawdown/Sharpe).
        if shares == 0:
            equity[i] = cash
        else:
            px_mark = mark[i] if np.isfinite(mark[i]) else entry_price
            equity[i] = cash + shares * px_mark
        held[i] = shares > 0

    # --- 4. close anything still open at the end of the data --------------
    open_at_end = False
    if shares > 0:
        last = n - 1
        # `mark` is a forward fill, so mark[last] is already the most
        # recent finite close; scan defensively anyway. Never hand a NaN
        # to the fee engine — fees.apply_fees does math.ceil() and would
        # raise on nan (and bursa_momentum/ is locked, so guard here).
        px = float(mark[last]) if np.isfinite(mark[last]) else float("nan")
        if not np.isfinite(px):
            finite = np.flatnonzero(np.isfinite(close))
            px = float(close[finite[-1]]) if finite.size else float("nan")

        if np.isfinite(px) and px > 0:
            fee = one_leg_fee(shares * px)
            cash += shares * px - fee
            trades.append(_make_trade(
                index, entry_i, entry_price, shares, entry_fees,
                last, px, fee, "eod", open_at_end=True,
            ))
            equity[last] = cash        # realised: the exit fee is now paid
        else:
            # No finite close anywhere in the frame: the position cannot
            # be valued, so it is not marked out. Record it at zero P&L
            # (the entry fee is still sunk) and flag it for the report
            # rather than inventing an exit price.
            trades.append(_make_trade(
                index, entry_i, entry_price, shares, entry_fees,
                last, entry_price, 0.0, "eod_no_price", open_at_end=True,
            ))
            equity[last] = cash + shares * entry_price
        shares = 0
        open_at_end = True

    equity_s = pd.Series(equity, index=index, name="equity")
    result = BacktestResult(
        trades=trades,
        equity=equity_s,
        exposure_pct=round(100.0 * float(held.sum()) / n, 4) if n else 0.0,
        skipped=skipped,
        open_at_end=open_at_end,
        starting_cash=float(budget_rm),
    )

    # Imported here (not at module scope) so metrics.py can type-hint
    # BacktestResult without an import cycle.
    from .metrics import compute_metrics

    result.metrics = compute_metrics(result, df)
    return result


def _make_trade(index: pd.Index, entry_i: int, entry_price: float, shares: int,
                entry_fees: float, exit_i: int, exit_price: float,
                exit_fees: float, reason: str, *, open_at_end: bool) -> Trade:
    """Assemble a Trade record and its derived P&L fields."""
    pnl_gross = shares * (exit_price - entry_price)
    pnl_net = pnl_gross - entry_fees - exit_fees
    notional = shares * entry_price
    return Trade(
        entry_date=index[entry_i],
        entry_price=round(float(entry_price), 6),
        shares=int(shares),
        exit_date=index[exit_i],
        exit_price=round(float(exit_price), 6),
        entry_fees=round(float(entry_fees), 2),
        exit_fees=round(float(exit_fees), 2),
        pnl_gross=round(float(pnl_gross), 2),
        pnl_net=round(float(pnl_net), 2),
        return_pct=round(100.0 * pnl_net / notional, 4) if notional else 0.0,
        holding_days=int(exit_i - entry_i),
        exit_reason=reason,
        open_at_end=open_at_end,
    )
