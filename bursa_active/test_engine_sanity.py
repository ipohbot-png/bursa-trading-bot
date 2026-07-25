"""
Sanity tests for bursa_active.engine / indicators.

Plain asserts, no pytest required:

    python bursa_active/test_engine_sanity.py

Every number the engine produces is re-derived here by hand (or by
calling `bursa_momentum.fees` directly), so a silent change in fill
timing, lot sizing, stop arithmetic or fee handling fails loudly.
"""
from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable

import numpy as np
import pandas as pd

# Allow `python bursa_active/test_engine_sanity.py` from the repo root or
# from inside the package directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bursa_momentum.fees import one_leg_fee                      # noqa: E402

from bursa_active import indicators as ind                       # noqa: E402
from bursa_active.engine import run_backtest                     # noqa: E402


TOL = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_df(closes: list[float] | np.ndarray, *,
            start: str = "2020-01-01") -> pd.DataFrame:
    """
    Synthetic OHLCV where **Open[i] == Close[i-1]** (Open[0] == Close[0]).

    That identity makes every "fill at next open" assertion hand-checkable:
    a signal on bar t fills at Close[t].
    """
    close = np.asarray(closes, dtype=float)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(close.shape, 1_000_000.0),
        },
        index=pd.bdate_range(start, periods=len(close)),
    )


def flags(index: pd.Index, *positions: int) -> pd.Series:
    """Boolean signal Series that is True only at the given bar numbers."""
    s = pd.Series(False, index=index)
    for p in positions:
        s.iloc[p] = True
    return s


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ---------------------------------------------------------------------------
# 1. Fill timing + fee-exact P&L on a rising series
# ---------------------------------------------------------------------------

def test_entry_fill_next_open_and_pnl() -> None:
    closes = [1.00 + 0.01 * i for i in range(100)]
    df = make_df(closes)
    entries = flags(df.index, 20)
    exits = pd.Series(False, index=df.index)

    res = run_backtest(df, entries, exits, stop_kind=None, budget_rm=20_000.0)

    assert len(res.trades) == 1, f"expected 1 trade, got {len(res.trades)}"
    t = res.trades[0]

    # Signal on bar 20 -> fill on bar 21's open, which equals Close[20] = 1.20.
    assert t.entry_date == df.index[21], t.entry_date
    assert approx(t.entry_price, df["Open"].iloc[21]), t.entry_price
    assert approx(t.entry_price, 1.20), t.entry_price

    # Board lots: floor(20000 / 1.20 / 100) * 100 = 16600
    assert t.shares == 16_600, t.shares

    # Never exited -> closed at the last close, flagged eod.
    assert t.exit_reason == "eod" and t.open_at_end and res.open_at_end
    assert t.exit_date == df.index[-1]
    assert approx(t.exit_price, 1.99), t.exit_price

    exp_entry_fee = one_leg_fee(16_600 * 1.20)
    exp_exit_fee = one_leg_fee(16_600 * 1.99)
    exp_gross = 16_600 * (1.99 - 1.20)
    exp_net = exp_gross - exp_entry_fee - exp_exit_fee

    assert approx(t.entry_fees, exp_entry_fee), (t.entry_fees, exp_entry_fee)
    assert approx(t.exit_fees, exp_exit_fee), (t.exit_fees, exp_exit_fee)
    assert approx(t.pnl_gross, exp_gross, 1e-4), (t.pnl_gross, exp_gross)
    assert approx(t.pnl_net, exp_net, 1e-4), (t.pnl_net, exp_net)
    assert approx(t.return_pct, 100.0 * exp_net / (16_600 * 1.20), 1e-4)

    # Equity curve: starts at budget, ends at realised cash.
    assert approx(res.equity.iloc[0], 20_000.0)
    exp_final = 20_000.0 - 16_600 * 1.20 - exp_entry_fee + 16_600 * 1.99 - exp_exit_fee
    assert approx(res.equity.iloc[-1], exp_final, 1e-4), (res.equity.iloc[-1], exp_final)

    # Metrics agree with the hand-computed trade.
    m = res.metrics
    assert m["n_trades"] == 1 and m["win_rate"] == 100.0
    assert approx(m["total_fees_rm"], round(exp_entry_fee + exp_exit_fee, 2), 1e-4)
    assert approx(m["total_return_pct"], (exp_final / 20_000.0 - 1.0) * 100.0, 1e-4)
    assert m["profit_factor"] == 999.0, "all-winners profit factor is capped at 999"
    assert m["insufficient_sample"] is True and m["skipped_trades"] == 0


# ---------------------------------------------------------------------------
# 2. Exit signal fills at the next open; holding_days counts bars
# ---------------------------------------------------------------------------

def test_exit_signal_next_open_and_holding_days() -> None:
    closes = [5.00 + 0.02 * i for i in range(60)]
    df = make_df(closes)
    res = run_backtest(df, flags(df.index, 10), flags(df.index, 30),
                       stop_kind=None, budget_rm=20_000.0)

    assert len(res.trades) == 1, len(res.trades)
    t = res.trades[0]
    assert t.entry_date == df.index[11] and approx(t.entry_price, df["Open"].iloc[11])
    assert t.exit_date == df.index[31], t.exit_date
    assert approx(t.exit_price, df["Open"].iloc[31]), t.exit_price
    assert t.exit_reason == "signal", t.exit_reason
    assert t.holding_days == 20, t.holding_days          # bar 31 - bar 11
    assert not t.open_at_end and not res.open_at_end
    # Flat afterwards -> exposure is exactly 20 of 60 bars.
    assert approx(res.exposure_pct, 100.0 * 20 / 60, 1e-4), res.exposure_pct


# ---------------------------------------------------------------------------
# 3. Fixed 5% stop fires on the right bar
# ---------------------------------------------------------------------------

def test_fixed_pct_stop() -> None:
    # Flat at 10.00 through bar 6, then a controlled slide.
    closes = [10.00] * 7 + [9.90, 9.60, 9.40, 9.20, 9.00, 8.80]
    df = make_df(closes)
    res = run_backtest(df, flags(df.index, 5), pd.Series(False, index=df.index),
                       stop_kind="fixed_pct", stop_param=5.0, budget_rm=20_000.0)

    assert len(res.trades) == 1, len(res.trades)
    t = res.trades[0]
    assert approx(t.entry_price, 10.00), t.entry_price   # Open[6] == Close[5]

    stop_level = 10.00 * 0.95                            # 9.50
    breach = next(i for i in range(6, len(closes)) if closes[i] < stop_level)
    assert breach == 9, breach                           # 9.60 holds, 9.40 breaks

    assert t.exit_reason == "stop", t.exit_reason
    assert t.exit_date == df.index[breach + 1], t.exit_date
    assert approx(t.exit_price, df["Open"].iloc[breach + 1]), t.exit_price
    assert approx(t.exit_price, 9.40), t.exit_price      # next open == Close[9]
    assert t.pnl_net < 0

    # The bar *before* the breach must not have exited.
    assert closes[breach - 1] >= stop_level


# ---------------------------------------------------------------------------
# 4. ATR trail ratchets up only, and exits on the first close below it
# ---------------------------------------------------------------------------

def test_atr_trail_ratchets_and_exits() -> None:
    rise = [10.00 + 0.25 * i for i in range(40)]
    peak = rise[-1]
    fall = [peak - 0.30 * (i + 1) for i in range(25)]
    df = make_df(rise + fall)
    k = 3.0

    res = run_backtest(df, flags(df.index, 30), pd.Series(False, index=df.index),
                       stop_kind="atr_trail", stop_param=k, budget_rm=20_000.0)

    assert len(res.trades) == 1, len(res.trades)
    t = res.trades[0]
    entry_bar = 31
    assert t.entry_date == df.index[entry_bar]

    # Re-derive the trail exactly as the engine does: update, then compare.
    atr = ind.atr(df["High"], df["Low"], df["Close"], 14).to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    trail = np.nan
    levels: list[float] = []
    breach = None
    for i in range(entry_bar, len(df)):
        cand = close[i] - k * atr[i]
        trail = cand if not np.isfinite(trail) else max(trail, cand)
        levels.append(trail)
        if close[i] < trail:
            breach = i
            break

    assert breach is not None, "crafted series should breach the trail"
    arr = np.array(levels)
    assert np.all(np.diff(arr) >= -1e-12), "trail must never ratchet down"
    assert arr[-1] > arr[0], "trail should have ratcheted up during the rise"

    assert t.exit_reason == "trail", t.exit_reason
    assert t.exit_date == df.index[breach + 1], (t.exit_date, df.index[breach + 1])
    assert approx(t.exit_price, df["Open"].iloc[breach + 1]), t.exit_price
    # Nothing below the trail before the breach bar.
    assert all(close[entry_bar + j] >= levels[j] for j in range(len(levels) - 1))


# ---------------------------------------------------------------------------
# 5. Lot rounding: an unaffordable board lot skips the trade
# ---------------------------------------------------------------------------

def test_lot_rounding_skips_trade() -> None:
    df = make_df([250.0] * 40)
    res = run_backtest(df, flags(df.index, 10), pd.Series(False, index=df.index),
                       stop_kind=None, budget_rm=20_000.0)

    # floor(20000 / 250 / 100) = 0 lots -> skipped, engine survives.
    assert res.trades == [], res.trades
    assert res.skipped == 1, res.skipped
    assert res.exposure_pct == 0.0
    assert bool((res.equity == 20_000.0).all()), res.equity.unique()
    assert res.metrics["n_trades"] == 0
    assert res.metrics["profit_factor"] == 0.0
    assert res.metrics["fee_drag_pct"] is None     # gross pnl == 0 -> undefined
    assert res.metrics["sharpe"] is None           # no trades -> not a number
    assert res.metrics["cagr"] is None
    assert res.metrics["skipped_trades"] == 1

    # Cheap enough and it trades again (sanity on the boundary).
    res2 = run_backtest(make_df([150.0] * 40), flags(df.index, 10),
                        pd.Series(False, index=df.index), budget_rm=20_000.0)
    assert res2.skipped == 0 and len(res2.trades) == 1
    assert res2.trades[0].shares == 100

    # At exactly RM 200 one board lot costs the entire budget, leaving
    # nothing for the brokerage -> unaffordable, not a borrowed lot.
    res3 = run_backtest(make_df([200.0] * 40), flags(df.index, 10),
                        pd.Series(False, index=df.index), budget_rm=20_000.0)
    assert res3.trades == [] and res3.skipped == 1


# ---------------------------------------------------------------------------
# 6. No lookahead: truncating the future cannot change the past
# ---------------------------------------------------------------------------

def test_no_lookahead_indicators() -> None:
    rng = np.random.default_rng(42)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    df = make_df(close.to_numpy())
    cut = 120
    trunc = df.iloc[:cut]

    cases: dict[str, tuple[pd.Series, pd.Series]] = {
        "sma20": (ind.sma(df["Close"], 20), ind.sma(trunc["Close"], 20)),
        "rsi14": (ind.rsi(df["Close"], 14), ind.rsi(trunc["Close"], 14)),
        "rsi2":  (ind.rsi(df["Close"], 2), ind.rsi(trunc["Close"], 2)),
        "atr14": (ind.atr(df["High"], df["Low"], df["Close"], 14),
                  ind.atr(trunc["High"], trunc["Low"], trunc["Close"], 14)),
        "donch_up": (ind.donchian(df["High"], df["Low"], 20)["upper"],
                     ind.donchian(trunc["High"], trunc["Low"], 20)["upper"]),
        "donch_lo": (ind.donchian(df["High"], df["Low"], 20)["lower"],
                     ind.donchian(trunc["High"], trunc["Low"], 20)["lower"]),
    }
    for name, (full, part) in cases.items():
        pd.testing.assert_series_equal(
            full.iloc[:cut], part, check_names=False,
            obj=f"{name} changed when future bars were removed",
        )
        assert part.notna().any(), f"{name} is entirely NaN — test is vacuous"

    # Warmup is NaN, not a fabricated early value.
    assert ind.sma(df["Close"], 20).iloc[:19].isna().all()
    assert ind.rsi(df["Close"], 14).iloc[:14].isna().all()
    assert ind.atr(df["High"], df["Low"], df["Close"], 14).iloc[:13].isna().all()

    # Donchian upper at bar i is the max of bars i-20 .. i-1 (current bar OUT).
    up = ind.donchian(df["High"], df["Low"], 20)["upper"]
    for i in (25, 60, 137):
        assert approx(up.iloc[i], df["High"].iloc[i - 20:i].max()), i
        assert not approx(up.iloc[i], df["High"].iloc[i - 19:i + 1].max()) or \
            df["High"].iloc[i] <= df["High"].iloc[i - 20:i].max()

    # Including the current bar makes a "close > N-day high" breakout
    # impossible to trigger on the bar that sets the high — that is the
    # lookahead trap the default guards against.
    incl = ind.donchian(df["High"], df["Low"], 20, exclude_current=False)["upper"]
    assert bool((df["High"] <= incl + 1e-12).loc[incl.notna()].all())


# ---------------------------------------------------------------------------
# 7. Priority + no same-bar round trip (extra guards on engine semantics)
# ---------------------------------------------------------------------------

def test_priority_and_no_same_bar_roundtrip() -> None:
    closes = [10.00] * 6 + [9.90, 9.40, 9.30, 9.20, 9.10, 9.00]
    df = make_df(closes)

    # Exit signal and stop breach on the same bar -> the stop wins.
    exits = flags(df.index, 7)
    res = run_backtest(df, flags(df.index, 4), exits,
                       stop_kind="fixed_pct", stop_param=5.0)
    assert len(res.trades) == 1 and res.trades[0].exit_reason == "stop"

    # Time stop beats a later exit signal.
    rising = make_df([10.00 + 0.05 * i for i in range(40)])
    res2 = run_backtest(rising, flags(rising.index, 5), flags(rising.index, 30),
                        stop_kind=None, time_stop_days=10)
    t2 = res2.trades[0]
    assert t2.exit_reason == "time" and t2.holding_days == 11, \
        (t2.exit_reason, t2.holding_days)   # breach at entry+10, fill at +11

    # Entry and exit True on the same bar while flat -> no position opened.
    both = flags(rising.index, 5)
    res3 = run_backtest(rising, both, both, stop_kind=None)
    assert res3.trades == [] and res3.skipped == 0
    assert bool((res3.equity == 20_000.0).all())


# ---------------------------------------------------------------------------
# 8. Engine causality: truncating the future cannot rewrite the past
# ---------------------------------------------------------------------------

def test_engine_is_causal() -> None:
    rng = np.random.default_rng(11)
    df = make_df(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, 300))))
    fast, slow = ind.sma(df["Close"], 5), ind.sma(df["Close"], 20)
    entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    full = run_backtest(df, entries, exits, stop_kind="atr_trail", stop_param=2.5)
    cut = 200
    part = run_backtest(df.iloc[:cut], entries.iloc[:cut], exits.iloc[:cut],
                        stop_kind="atr_trail", stop_param=2.5)

    assert full.trades, "fixture should generate trades"
    horizon = df.index[cut - 2]     # trades that closed before the cut
    a = [t for t in full.trades if t.exit_date <= horizon]
    b = [t for t in part.trades if t.exit_date <= horizon]
    assert a and len(a) == len(b), (len(a), len(b))
    for x, y in zip(a, b):
        assert x.__dict__ == y.__dict__, (x, y)

    # Equity up to the cut is identical too.
    pd.testing.assert_series_equal(full.equity.iloc[:cut - 1],
                                   part.equity.iloc[:cut - 1], check_names=False)


# ---------------------------------------------------------------------------
# 9. REGRESSION: the sim must never borrow (cash >= 0 on every entry)
# ---------------------------------------------------------------------------

def cash_after_entry(res, df: pd.DataFrame, trade) -> float:
    """Reconstruct cash right after an entry fill: equity - marked position."""
    bar = df.index.get_loc(trade.entry_date)
    return float(res.equity.iloc[bar]) - trade.shares * float(df["Close"].iloc[bar])


def test_no_borrowing_on_losing_sequence() -> None:
    # Sawtooth grind lower: every round trip loses, so cash falls well
    # below the RM 20k budget and later entries must size down.
    closes = [100.0 * (0.97 ** i) * (1.06 if i % 10 in (0, 1) else 1.0)
              for i in range(110)]
    df = make_df(closes)
    entry_bars = list(range(5, 100, 12))
    entries = flags(df.index, *entry_bars)
    exits = flags(df.index, *[b + 8 for b in entry_bars])

    res = run_backtest(df, entries, exits, stop_kind=None, budget_rm=20_000.0)

    assert len(res.trades) >= 5, len(res.trades)
    assert all(t.pnl_net < 0 for t in res.trades), "fixture should be all losers"

    for t in res.trades:
        cash = cash_after_entry(res, df, t)
        assert cash >= -1e-6, f"borrowed RM {-cash:,.2f} on {t.entry_date.date()}"
        # Never deploy more than the budget, and never more than was held.
        assert t.shares * t.entry_price <= 20_000.0 + 1e-6

    # Sizing actually shrank as the account did — proof the cap binds.
    notionals = [t.shares * t.entry_price for t in res.trades]
    assert notionals[-1] < notionals[0], notionals

    assert float(res.equity.min()) > 0.0, res.equity.min()
    assert bool(res.equity.notna().all())
    # Equity delta still reconciles with the trade ledger.
    booked = sum(t.pnl_net for t in res.trades)
    assert approx(res.equity.iloc[-1] - 20_000.0, booked, 1e-3)


# ---------------------------------------------------------------------------
# 10. REGRESSION: sharpe never disagrees in sign with the actual return
# ---------------------------------------------------------------------------

def test_sharpe_sign_and_ruin_guard() -> None:
    # A steady grind lower, held throughout: unambiguously a losing run.
    df = make_df([100.0 * (0.99 ** i) for i in range(150)])
    res = run_backtest(df, flags(df.index, 5), pd.Series(False, index=df.index),
                       stop_kind=None, budget_rm=20_000.0)
    m = res.metrics
    assert m["total_return_pct"] < 0, m["total_return_pct"]
    assert m["sharpe"] is None or m["sharpe"] < 0, m["sharpe"]
    assert m["cagr"] is None or m["cagr"] < 0, m["cagr"]

    # Near-total wipeout: ratio metrics are meaningless, so they are None.
    # Priced at RM 1.00 so board lots absorb essentially the whole budget
    # (at RM 100 a single lot is 1/2 the budget and "ruin" never arrives).
    crash_df = make_df([1.0] * 10 + [1.0 * (0.92 ** i) for i in range(1, 120)])
    res2 = run_backtest(crash_df, flags(crash_df.index, 5),
                        pd.Series(False, index=crash_df.index),
                        stop_kind=None, budget_rm=20_000.0)
    assert res2.metrics["total_return_pct"] < -80.0, res2.metrics["total_return_pct"]
    assert res2.equity.min() <= 0.05 * 20_000.0, res2.equity.min()
    assert res2.metrics["sharpe"] is None, res2.metrics["sharpe"]
    assert res2.metrics["cagr"] is None, res2.metrics["cagr"]

    # A profitable run must score a positive Sharpe.
    up = make_df([10.0 * (1.004 ** i) for i in range(150)])
    res3 = run_backtest(up, flags(up.index, 5), pd.Series(False, index=up.index),
                        stop_kind=None, budget_rm=20_000.0)
    assert res3.metrics["total_return_pct"] > 0, res3.metrics["total_return_pct"]
    assert res3.metrics["sharpe"] is not None and res3.metrics["sharpe"] > 0, \
        res3.metrics["sharpe"]
    assert res3.metrics["cagr"] > 0, res3.metrics["cagr"]

    # Log returns make the sign of Sharpe track the sign of the compounded
    # return by construction — check it on a noisy round trip too.
    rng = np.random.default_rng(3)
    for seed_scale in (0.004, 0.02):
        noisy = make_df(50.0 * np.exp(np.cumsum(
            rng.normal(-0.0015, seed_scale, 220))))
        r = run_backtest(noisy, flags(noisy.index, 3),
                         pd.Series(False, index=noisy.index), stop_kind=None)
        m2 = r.metrics
        if m2["sharpe"] is not None:
            assert (m2["sharpe"] > 0) == (m2["total_return_pct"] > 0), \
                (m2["sharpe"], m2["total_return_pct"])


def test_sharpe_matches_real_scenario() -> None:
    """
    Reviewer's headline case: 0217.KL sma_cross_10_50_trail3 over its full
    cached history is a losing run (-9.2%), so Sharpe must not be positive.

    Skipped (not failed) when the audit cache is absent, so the suite stays
    runnable on a clean checkout.
    """
    cache = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             + r"\.cache\active\0217.KL.parquet")
    if not os.path.exists(cache):
        print("      (skipped: no cached 0217.KL data)")
        return

    px = pd.read_parquet(cache)
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    fast, slow = ind.sma(px["Close"], 10), ind.sma(px["Close"], 50)
    entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    res = run_backtest(px, entries, exits, stop_kind="atr_trail",
                       stop_param=3.0, budget_rm=20_000.0)

    m = res.metrics
    assert m["n_trades"] > 10, m["n_trades"]
    assert m["total_return_pct"] < 0, m["total_return_pct"]
    assert m["sharpe"] is None or m["sharpe"] < 0, \
        f"losing run scored sharpe {m['sharpe']} on return {m['total_return_pct']}%"
    print(f"      0217.KL sma_cross_10_50_trail3: "
          f"return={m['total_return_pct']}%  sharpe={m['sharpe']}")


# ---------------------------------------------------------------------------
# 11. REGRESSION: a zero-volume bar cannot fill an order
# ---------------------------------------------------------------------------

def test_zero_volume_bar_carries_order_forward() -> None:
    df = make_df([10.0 + 0.10 * i for i in range(40)])
    df.loc[df.index[11], "Volume"] = 0.0        # halted / no trades
    df.loc[df.index[12], "Volume"] = np.nan     # missing volume, same treatment

    res = run_backtest(df, flags(df.index, 10), flags(df.index, 20),
                       stop_kind=None, budget_rm=20_000.0)

    assert len(res.trades) == 1, len(res.trades)
    t = res.trades[0]
    # Signal on bar 10 would normally fill on bar 11 — that bar did not
    # trade, nor did 12, so the fill lands on bar 13.
    assert t.entry_date == df.index[13], t.entry_date
    assert approx(t.entry_price, df["Open"].iloc[13]), t.entry_price

    # Same rule on the way out.
    df2 = make_df([10.0 + 0.10 * i for i in range(40)])
    df2.loc[df2.index[21], "Volume"] = 0.0
    res2 = run_backtest(df2, flags(df2.index, 10), flags(df2.index, 20),
                        stop_kind=None, budget_rm=20_000.0)
    assert res2.trades[0].exit_date == df2.index[22], res2.trades[0].exit_date
    assert res2.trades[0].holding_days == 11, res2.trades[0].holding_days


# ---------------------------------------------------------------------------
# 12. REGRESSION: empty frame still returns the full metrics dict
# ---------------------------------------------------------------------------

def test_empty_df_returns_full_metrics() -> None:
    normal = run_backtest(make_df([10.0 + 0.1 * i for i in range(40)]),
                          flags(make_df([10.0] * 40).index, 5),
                          pd.Series(False, index=make_df([10.0] * 40).index))
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"],
                         index=pd.DatetimeIndex([]))
    res = run_backtest(empty, pd.Series(dtype=bool), pd.Series(dtype=bool))

    assert res.trades == [] and res.equity.empty
    assert set(res.metrics) == set(normal.metrics), (
        set(normal.metrics) ^ set(res.metrics))
    assert len(res.metrics) == 18, len(res.metrics)
    assert res.metrics["n_trades"] == 0
    assert res.metrics["sharpe"] is None and res.metrics["cagr"] is None
    assert res.metrics["total_return_pct"] == 0.0

    # A real-world empty slice (window outside the data) behaves the same.
    df = make_df([10.0] * 40)
    sliced = df.loc[df.index[-1] + pd.Timedelta(days=30):]
    res2 = run_backtest(sliced, pd.Series(dtype=bool), pd.Series(dtype=bool))
    assert set(res2.metrics) == set(normal.metrics)


# ---------------------------------------------------------------------------
# 13. REGRESSION: EOD mark-out with no usable close never reaches the fees
# ---------------------------------------------------------------------------

def test_eod_without_price_does_not_crash_fees() -> None:
    # Opens are tradeable but every close is NaN: the entry fills, the
    # position can never be valued. math.ceil(nan) in fees would raise.
    df = make_df([10.0 + 0.05 * i for i in range(30)])
    df["Close"] = np.nan

    res = run_backtest(df, flags(df.index, 5), pd.Series(False, index=df.index),
                       stop_kind=None, budget_rm=20_000.0)

    assert len(res.trades) == 1, len(res.trades)
    t = res.trades[0]
    assert t.exit_reason == "eod_no_price", t.exit_reason
    assert t.open_at_end and res.open_at_end
    assert t.exit_fees == 0.0 and t.pnl_gross == 0.0
    assert approx(t.pnl_net, -t.entry_fees)
    assert bool(res.equity.notna().all()), "equity must stay finite"
    assert res.metrics["n_trades"] == 1


# ---------------------------------------------------------------------------
# 14. REGRESSION: an entry signal on the final bar is counted as skipped
# ---------------------------------------------------------------------------

def test_final_bar_entry_counted_as_skipped() -> None:
    df = make_df([10.0 + 0.05 * i for i in range(30)])
    res = run_backtest(df, flags(df.index, len(df) - 1),
                       pd.Series(False, index=df.index), stop_kind=None)
    assert res.trades == []
    assert res.skipped == 1, res.skipped
    assert res.metrics["skipped_trades"] == 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS: list[Callable[[], None]] = [
    test_entry_fill_next_open_and_pnl,
    test_exit_signal_next_open_and_holding_days,
    test_fixed_pct_stop,
    test_atr_trail_ratchets_and_exits,
    test_lot_rounding_skips_trade,
    test_no_lookahead_indicators,
    test_priority_and_no_same_bar_roundtrip,
    test_engine_is_causal,
    test_no_borrowing_on_losing_sequence,
    test_sharpe_sign_and_ruin_guard,
    test_sharpe_matches_real_scenario,
    test_zero_volume_bar_carries_order_forward,
    test_empty_df_returns_full_metrics,
    test_eod_without_price_does_not_crash_fees,
    test_final_bar_entry_counted_as_skipped,
]


def main() -> int:
    failures = 0
    for fn in TESTS:
        try:
            fn()
        except Exception:                              # noqa: BLE001
            failures += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS  {fn.__name__}")
    total = len(TESTS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
