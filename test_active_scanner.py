"""
Sanity tests for active_scanner.py.

Plain asserts, no pytest required (matches bursa_active/test_engine_sanity.py):

    python test_active_scanner.py

Everything here is synthetic and offline -- no yfinance calls, no real
state/log files touched. `process_ticker` / `evaluate_spec` are pure
functions (df + state dict in, new state + events out), so the tests
inject frames and state directly rather than monkeypatching I/O.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

from active_scanner import (
    build_alert_messages,
    build_trend_alert_messages,
    load_promoted_specs,
    process_ticker,
)


TICKER = "5330.KL"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_df(closes: list[float], *, start: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic daily OHLCV. Open[i] == Close[i-1]; High/Low bracket it."""
    close = np.asarray(closes, dtype=float)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame(
        {
            "Open": open_, "High": high, "Low": low, "Close": close,
            "Volume": np.full(close.shape, 1_000_000.0),
        },
        index=pd.bdate_range(start, periods=len(close)),
    )


def entry_ready_df() -> pd.DataFrame:
    """
    70 bars of a slow, steady uptrend (keeps close > SMA(50)) followed by a
    sharp 2-bar dip at the very end (drives RSI(2) well under 25) --
    exactly the rsi2 pullback-in-an-uptrend entry setup both promoted rules
    share.
    """
    up = [1.00 * (1.008 ** i) for i in range(70)]
    dip = [up[-1] * 0.90, up[-1] * 0.90 * 0.95]
    return make_df(up + dip)


def flat_state() -> dict:
    return {}


def long_state(entry_date: str, entry_price: float, bars_held: int, spec_id: str) -> dict:
    return {
        "last_bar_date": entry_date,   # irrelevant here; process_ticker overwrites per-call
        "rules": {
            spec_id: {
                "position": "long",
                "entry_date": entry_date,
                "entry_price": entry_price,
                "bars_held": bars_held,
            },
        },
    }


# ---------------------------------------------------------------------------
# Test runner plumbing
# ---------------------------------------------------------------------------

_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {name}")
    else:
        _FAILURES.append(name)
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# 0. promoted_rules.json <-> build_grid() lookup does not drift
# ---------------------------------------------------------------------------

def test_specs_resolve():
    print("\n[0] promoted specs resolve via build_grid()")
    specs = load_promoted_specs()
    check("exactly 2 promoted specs", len(specs) == 2, f"got {len(specs)}")
    ids = {s.id for s in specs}
    check(
        "both are rsi2 pullback variants (time 10 vs 15)",
        ids == {
            "rsi2_pullback_sma50_lt25_gt75_time15_stop5",
            "rsi2_pullback_sma50_lt25_gt75_time10_stop5",
        },
        f"got {ids}",
    )
    return specs


# ---------------------------------------------------------------------------
# 1. Entry fires exactly once; no duplicate on same-day re-run
# ---------------------------------------------------------------------------

def test_entry_once_no_duplicate(specs):
    print("\n[1] entry alert fires exactly once; no duplicate same-day re-run")
    df = entry_ready_df()
    spec = next(s for s in specs if s.time_stop_days == 15)

    result1 = process_ticker(TICKER, df, [spec], flat_state())
    check("first run: exactly one event", len(result1.events) == 1, f"{result1.events}")
    if result1.events:
        check("first run: event is an entry", result1.events[0].event == "entry")
    check(
        "first run: new state is long",
        result1.new_state["rules"][spec.id]["position"] == "long",
    )

    # Re-run with the SAME frame (same last bar) and the state produced by
    # run 1 -- simulates invoking the scanner twice in one day.
    result2 = process_ticker(TICKER, df, [spec], result1.new_state)
    check("second same-day run: no events", len(result2.events) == 0, f"{result2.events}")
    check("second same-day run: flagged as no-new-bar", result2.no_new_bar)
    check(
        "second same-day run: state unchanged",
        result2.new_state["rules"][spec.id]["position"] == "long",
    )

    messages = build_alert_messages(result1.events)
    check("exactly one alert message for the ticker", len(messages) == 1, f"{messages}")
    if messages:
        check("alert message says ENTRY", messages[0].startswith("ENTRY signal:"))


# ---------------------------------------------------------------------------
# 2. Exit fires on each of the three reasons
# ---------------------------------------------------------------------------

def test_exit_rsi_reason(specs):
    print("\n[2a] exit fires on RSI(2) > 75")
    spec = next(s for s in specs if s.time_stop_days == 15)
    # Uptrend, then a sharp 2-bar rally at the end -> RSI(2) spikes > 75.
    up = [1.00 * (1.01 ** i) for i in range(70)]
    rally = [up[-1] * 1.05, up[-1] * 1.05 * 1.05]
    df = make_df(up + rally)

    entry_date = df.index[-3].date().isoformat()   # entered a couple of bars ago
    entry_price = float(df["Close"].iloc[-3]) * 0.9  # well below current close: no fixed-stop
    state = long_state(entry_date, entry_price, bars_held=2, spec_id=spec.id)

    result = process_ticker(TICKER, df, [spec], state)
    check("exactly one event", len(result.events) == 1, f"{result.events}")
    if result.events:
        ev = result.events[0]
        check("event is an exit", ev.event == "exit")
        check("reason is rsi_exit only", ev.reasons == ["rsi_exit"], f"{ev.reasons}")


def test_exit_time_stop_reason(specs):
    print("\n[2b] exit fires on time stop reached (checked for both 10d and 15d rules)")
    # Flattish prices near the end so RSI(2) exit and the 5% fixed stop
    # don't also fire -- isolates the time-stop reason.
    flat = [1.00 * (1.001 ** i) for i in range(90)]
    df = make_df(flat)

    for spec in specs:
        entry_pos = (len(df) - 1) - spec.time_stop_days
        entry_date = df.index[entry_pos].date().isoformat()
        entry_price = float(df["Close"].iloc[entry_pos]) * 0.99  # close, not a stop breach
        state = long_state(entry_date, entry_price, bars_held=0, spec_id=spec.id)

        result = process_ticker(TICKER, df, [spec], state)
        check(
            f"time-stop {spec.time_stop_days}d: exactly one event",
            len(result.events) == 1, f"{result.events}",
        )
        if result.events:
            ev = result.events[0]
            check(f"time-stop {spec.time_stop_days}d: event is an exit", ev.event == "exit")
            check(
                f"time-stop {spec.time_stop_days}d: reason includes time_stop",
                "time_stop" in ev.reasons, f"{ev.reasons}",
            )


def test_exit_fixed_stop_reason(specs):
    print("\n[2c] exit fires on 5% fixed stop-loss breach")
    spec = next(s for s in specs if s.time_stop_days == 15)
    up = [1.00 * (1.01 ** i) for i in range(70)]
    crash = [up[-1] * 0.90, up[-1] * 0.90 * 0.90]   # sharp crash at the very end
    df = make_df(up + crash)

    entry_date = df.index[-3].date().isoformat()
    entry_price = float(df["Close"].iloc[-3])       # entered right before the crash
    state = long_state(entry_date, entry_price, bars_held=2, spec_id=spec.id)

    result = process_ticker(TICKER, df, [spec], state)
    check("exactly one event", len(result.events) == 1, f"{result.events}")
    if result.events:
        ev = result.events[0]
        check("event is an exit", ev.event == "exit")
        check("reason includes fixed_stop", "fixed_stop" in ev.reasons, f"{ev.reasons}")


# ---------------------------------------------------------------------------
# 3. Weekend / holiday no-op
# ---------------------------------------------------------------------------

def test_weekend_no_op(specs):
    print("\n[3] weekend/holiday run with no new bar is a no-op")
    df = entry_ready_df()
    spec = next(s for s in specs if s.time_stop_days == 15)
    last_date_str = df.index[-1].date().isoformat()

    # State already claims we processed this exact last bar (simulating a
    # Saturday run where yfinance returns Friday's bar again).
    state = {"last_bar_date": last_date_str, "rules": {}}
    result = process_ticker(TICKER, df, [spec], state)

    check("no events fired", len(result.events) == 0, f"{result.events}")
    check("flagged as no-new-bar", result.no_new_bar)
    check("note mentions no new bar", result.note is not None and "no new bar" in result.note)
    check("state passed through unchanged", result.new_state == state)


# ---------------------------------------------------------------------------
# Trend-watch fixtures
# ---------------------------------------------------------------------------

def trend_break_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (prefix, full): `prefix` is a steady 78-bar uptrend (close stays above
    its 50-day SMA throughout, so bootstrapping on it records trend
    "above"). `full` appends a sharp 2-bar crash that pulls the close
    below its own 50-day SMA -- an above -> below transition on the last
    bar relative to `prefix`.
    """
    up = [1.00 * (1.01 ** i) for i in range(78)]
    drop = [up[-1] * 0.80, up[-1] * 0.80 * 0.90]
    full = make_df(up + drop)
    return full.iloc[:78], full


def trend_resume_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror of `trend_break_frames`: a downtrend (bootstraps "below"),
    then a sharp 2-bar rally that pushes the close back above its 50-day
    SMA -- a below -> above transition."""
    down = [1.00 * (0.99 ** i) for i in range(78)]
    rally = [down[-1] * 1.25, down[-1] * 1.25 * 1.15]
    full = make_df(down + rally)
    return full.iloc[:78], full


# ---------------------------------------------------------------------------
# 4. Trend watch: bootstrap is silent
# ---------------------------------------------------------------------------

def test_trend_bootstrap_silent(specs):
    print("\n[4] trend watch: first-ever run for a ticker is silent (no alert)")
    prefix, _ = trend_break_frames()
    result = process_ticker(TICKER, prefix, specs, {})
    check("no trend_event on bootstrap", result.trend_event is None)
    check(
        "trend recorded in new_state for next run",
        result.new_state.get("trend") in ("above", "below"),
        f"{result.new_state.get('trend')}",
    )


# ---------------------------------------------------------------------------
# 5. Trend watch: above -> below fires exactly once, no duplicate same-day
# ---------------------------------------------------------------------------

def test_trend_break_fires_once_no_duplicate(specs):
    print("\n[5] trend watch: above -> below fires exactly once, no duplicate same-day")
    prefix, full = trend_break_frames()

    bootstrap = process_ticker(TICKER, prefix, specs, {})
    check("bootstrap trend is 'above'", bootstrap.new_state.get("trend") == "above",
          f"{bootstrap.new_state.get('trend')}")

    result = process_ticker(TICKER, full, specs, bootstrap.new_state)
    check("trend_event fired", result.trend_event is not None)
    if result.trend_event:
        check("event is trend_break", result.trend_event.event == "trend_break")
        check("new trend is 'below'", result.new_state.get("trend") == "below")

    messages = build_trend_alert_messages([result.trend_event] if result.trend_event else [])
    check(
        "exactly one TREND BREAK message",
        len(messages) == 1 and messages[0].startswith("TREND BREAK:"),
        f"{messages}",
    )

    # Re-run with the SAME frame (same last bar) -- must not duplicate.
    result2 = process_ticker(TICKER, full, specs, result.new_state)
    check("no duplicate trend_event on same-day re-run", result2.trend_event is None)
    check("same-day re-run flagged as no-new-bar", result2.no_new_bar)


# ---------------------------------------------------------------------------
# 6. Trend watch: below -> above fires
# ---------------------------------------------------------------------------

def test_trend_resume_fires(specs):
    print("\n[6] trend watch: below -> above fires")
    prefix, full = trend_resume_frames()

    bootstrap = process_ticker(TICKER, prefix, specs, {})
    check("bootstrap trend is 'below'", bootstrap.new_state.get("trend") == "below",
          f"{bootstrap.new_state.get('trend')}")

    result = process_ticker(TICKER, full, specs, bootstrap.new_state)
    check("trend_event fired", result.trend_event is not None)
    if result.trend_event:
        check("event is trend_resume", result.trend_event.event == "trend_resume")
        check("new trend is 'above'", result.new_state.get("trend") == "above")

    messages = build_trend_alert_messages([result.trend_event] if result.trend_event else [])
    check(
        "exactly one TREND RESUME message",
        len(messages) == 1 and messages[0].startswith("TREND RESUME:"),
        f"{messages}",
    )


# ---------------------------------------------------------------------------
# 7. Trend watch: legacy state file without a "trend" key doesn't crash
# ---------------------------------------------------------------------------

def test_legacy_state_without_trend_key(specs):
    print("\n[7] trend watch: legacy state file without a 'trend' key doesn't crash")
    prefix, full = trend_break_frames()
    # Simulate a state file written by the pre-trend-watch version of the
    # scanner: has last_bar_date + rules, but no "trend" key at all.
    legacy_state = {"last_bar_date": prefix.index[-1].date().isoformat(), "rules": {}}
    assert "trend" not in legacy_state

    result = process_ticker(TICKER, full, specs, legacy_state)   # must not raise
    check("no spurious trend alert the first time a legacy state sees a trend",
          result.trend_event is None)
    check("new_state now carries a trend key", "trend" in result.new_state)


# ---------------------------------------------------------------------------
# 8. --dry-run performs zero state/log writes, even with events pending
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing():
    print("\n[8] --dry-run performs zero state/log writes even when events fire")
    import active_scanner as scanner

    prefix, full = trend_break_frames()
    fake_frames = {t: full for t in scanner.TICKERS}
    # Prior state claims "above" as of the prefix's last bar -- `full`'s
    # extra bars flip it to "below", guaranteeing a real trend_event (and
    # therefore a non-trivial write attempt) for this run.
    fake_prior_state = {
        t: {"last_bar_date": prefix.index[-1].date().isoformat(), "rules": {}, "trend": "above"}
        for t in scanner.TICKERS
    }

    calls = {"save_state": 0, "append_log": 0}

    def fake_download_history(tickers, **kwargs):
        return dict(fake_frames)

    def fake_load_state(*a, **k):
        return dict(fake_prior_state)

    def fake_save_state(*a, **k):
        calls["save_state"] += 1

    def fake_append_log(*a, **k):
        calls["append_log"] += 1

    originals = {
        "download_history": scanner.download_history,
        "load_state": scanner.load_state,
        "save_state": scanner.save_state,
        "append_log": scanner.append_log,
    }
    orig_argv = sys.argv
    scanner.download_history = fake_download_history
    scanner.load_state = fake_load_state
    scanner.save_state = fake_save_state
    scanner.append_log = fake_append_log
    sys.argv = ["active_scanner.py", "--dry-run"]
    try:
        rc = scanner.main()
    finally:
        scanner.download_history = originals["download_history"]
        scanner.load_state = originals["load_state"]
        scanner.save_state = originals["save_state"]
        scanner.append_log = originals["append_log"]
        sys.argv = orig_argv

    check("main() returns success", rc == 0, f"rc={rc}")
    check("save_state never called under --dry-run", calls["save_state"] == 0)
    check("append_log never called under --dry-run", calls["append_log"] == 0)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def main() -> int:
    try:
        specs = test_specs_resolve()
        test_entry_once_no_duplicate(specs)
        test_exit_rsi_reason(specs)
        test_exit_time_stop_reason(specs)
        test_exit_fixed_stop_reason(specs)
        test_weekend_no_op(specs)
        test_trend_bootstrap_silent(specs)
        test_trend_break_fires_once_no_duplicate(specs)
        test_trend_resume_fires(specs)
        test_legacy_state_without_trend_key(specs)
        test_dry_run_writes_nothing()
    except Exception:
        traceback.print_exc()
        return 1

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s) failed:")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
