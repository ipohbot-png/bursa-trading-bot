"""
active_scanner.py
==================
Daily live signal scanner for the two promoted `bursa_active` rules
(DESIGN.md, "active_scanner.py (Phase 7)").

SIGNAL-ONLY -- READ THIS BEFORE TOUCHING THIS FILE
---------------------------------------------------
This script NEVER places, proposes to auto-execute, or simulates placing an
order with any broker. It only:
  1. downloads the latest daily bars for the two promoted targets,
  2. re-computes the promoted rules' entry/exit signals EXACTLY as the
     backtester would (same `StrategySpec` objects, resolved from
     `strategies.build_grid()` -- never hand-rederived here),
  3. compares today's signal against locally-tracked state, and
  4. prints a summary and (optionally) sends a Telegram notification.

Nothing here talks to a broker API. If you are tempted to add order
placement, don't -- open a new, clearly-labelled script instead so this
one stays auditable as signal-only.

Run
---
    python active_scanner.py                # live: evaluates, alerts, persists state
    python active_scanner.py --dry-run       # evaluate + print only, no writes/alerts

State & logs (both at the project root)
----------------------------------------
    active_signal_state.json   per (ticker, spec_id): position/entry/bars_held,
                               plus a per-ticker "trend": "above"|"below" 50-day
                               SMA watch (independent of the swing-rule states --
                               see the "Trend watch" section below).
    active_signal_log.jsonl    one JSON object per signal event, append-only

Trend watch
-----------
Independent of the two swing rules, every run also checks whether each
ticker's latest close is above or below its 50-day SMA -- the backtested
uptrend filter both promoted rules use as an entry gate. This is for a
buy-and-hold reader who ignores the swing entries/exits but still wants a
heads-up when the tested uptrend condition breaks (or resumes). It only
alerts on a state TRANSITION (never every day the stock happens to be
below its SMA), and stays silent on the very first run for a ticker so
"no stored trend yet" doesn't read as a spurious alert.

Telegram
--------
Reuses `bursa_momentum_scanner.send_telegram` (same env vars, same
endpoint). Silently no-ops -- with a printed note -- if
`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` are unset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:            # allow running from anywhere
    sys.path.insert(0, str(PROJECT_ROOT))

from bursa_active import indicators as ind          # noqa: E402
from bursa_active import strategies                 # noqa: E402
from bursa_active.strategies import StrategySpec     # noqa: E402
from bursa_momentum.data import download_history     # noqa: E402

try:
    # Reuse the existing scanner's Telegram helper verbatim -- same env
    # vars, same endpoint, same failure handling. It imports cleanly
    # (bursa_momentum_scanner.py has no import-time side effects beyond
    # defining a Path constant), so there is nothing to replicate.
    from bursa_momentum_scanner import send_telegram as _send_telegram_raw  # noqa: E402
except Exception:  # pragma: no cover - defensive only
    # Fallback ONLY if the import above ever breaks (e.g. a future refactor
    # renames/removes send_telegram). Small, deliberate duplicate of the
    # same function so this scanner never hard-depends on the other one.
    import requests as _requests

    def _send_telegram_raw(message: str) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat:
            return
        try:
            _requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat, "text": message},
                timeout=10,
            )
        except Exception as exc:
            print(f"[scanner] telegram send failed: {exc!s}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMOTED_RULES_PATH = PROJECT_ROOT / "active_results" / "promoted_rules.json"
STATE_PATH = PROJECT_ROOT / "active_signal_state.json"
LOG_PATH = PROJECT_ROOT / "active_signal_log.jsonl"

TICKERS: list[str] = ["5330.KL", "0217.KL"]           # TMK Chemical, Powerwell Holdings
TICKER_NAMES: dict[str, str] = {
    "5330.KL": "TMK Chemical",
    "0217.KL": "Powerwell Holdings",
}

#: Bars needed before SMA(50)/RSI(2) diagnostics are meaningful, plus margin.
MIN_WARMUP_BARS = 60

_REASON_TEXT: dict[str, str] = {
    "rsi_exit": "RSI(2) > 75",
    "time_stop": "time stop reached",
    "fixed_stop": "5% stop-loss breach",
    "signal": "entry rule fired",
}


def _ticker_label(ticker: str) -> str:
    name = TICKER_NAMES.get(ticker)
    return f"{ticker} ({name})" if name else ticker


def _rule_label(spec: StrategySpec) -> str:
    days = spec.time_stop_days
    return f"time-stop {days}d rule" if days is not None else spec.id


def _format_reasons(reasons: Iterable[str]) -> str:
    return ", ".join(_REASON_TEXT.get(r, r) for r in reasons)


# ---------------------------------------------------------------------------
# Promoted rule resolution -- spec_id -> StrategySpec, never re-derived here
# ---------------------------------------------------------------------------

def load_promoted_specs(path: Path = PROMOTED_RULES_PATH) -> list[StrategySpec]:
    """Resolve every promoted spec_id to its StrategySpec via build_grid().

    This is the only place rule parameters "come from" -- signal
    construction is therefore identical to whatever the backtester used to
    promote the rule in the first place.
    """
    raw = json.loads(path.read_text())
    grid = {s.id: s for s in strategies.build_grid()}
    specs: list[StrategySpec] = []
    for rule in raw:
        spec_id = rule["spec_id"]
        if spec_id not in grid:
            raise KeyError(
                f"promoted spec_id {spec_id!r} not found in strategies.build_grid() "
                f"-- the grid and {path.name} have drifted apart"
            )
        specs.append(grid[spec_id])
    return specs


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _default_rule_state() -> dict[str, Any]:
    return {"position": "flat", "entry_date": None, "entry_price": None, "bars_held": 0}


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[scanner] failed to parse {path.name}: {exc!s} -- starting fresh.")
        return {}


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def append_log(records: list[dict[str, Any]], path: Path = LOG_PATH) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Diagnostics (display-only). The actual entry/exit decision always comes
# from spec.build_signals(); these are just human-readable context for the
# console summary (both promoted rules use RSI(2) / SMA(50)).
# ---------------------------------------------------------------------------

@dataclass
class Diagnostics:
    close: float
    rsi2: float | None
    sma50: float | None


def compute_diagnostics(df: pd.DataFrame) -> Diagnostics:
    close = df["Close"]
    rsi2 = ind.rsi(close, 2)
    sma50 = ind.sma(close, 50)
    last_rsi2 = float(rsi2.iloc[-1]) if pd.notna(rsi2.iloc[-1]) else None
    last_sma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    return Diagnostics(close=float(close.iloc[-1]), rsi2=last_rsi2, sma50=last_sma50)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class SpecEvent:
    ticker: str
    spec: StrategySpec
    event: str            # "entry" | "exit"
    reasons: list[str]
    close: float
    date: str


@dataclass
class SpecSummaryRow:
    ticker: str
    spec_id: str
    position: str
    rsi2: float | None
    sma50: float | None
    close: float
    fired: str
    trend: str | None = None


@dataclass
class TrendEvent:
    """A buy-and-hold trend-watch transition -- independent of the swing
    rules' entry/exit states (see module docstring, "Trend watch")."""

    ticker: str
    event: str            # "trend_break" | "trend_resume"
    close: float
    sma50: float
    date: str


@dataclass
class TickerResult:
    ticker: str
    new_state: dict[str, Any]
    events: list[SpecEvent]
    summary_rows: list[SpecSummaryRow]
    no_new_bar: bool
    note: str | None = None
    trend_event: TrendEvent | None = None


# ---------------------------------------------------------------------------
# Per-(ticker, spec) evaluation -- pure, no I/O, unit-testable
# ---------------------------------------------------------------------------

def _bars_held(df: pd.DataFrame, entry_date: str | None, fallback: int) -> int:
    """
    Trading bars between `entry_date` and the last bar in `df`, matching
    the backtest engine's `(i - entry_i)` convention (0 on the entry bar
    itself). Falls back to `fallback + 1` if the entry date has scrolled
    out of the downloaded window (shouldn't happen with a 1y download and
    a <=15-day time stop, but this keeps a long-running deployment safe).
    """
    if entry_date is None:
        return 0
    try:
        entry_ts = pd.Timestamp(entry_date)
        pos = df.index.get_loc(entry_ts)
    except KeyError:
        return fallback + 1
    return (len(df) - 1) - int(pos)


def evaluate_trend(
    ticker: str, diag: Diagnostics, stored_trend: str | None, date_str: str,
) -> tuple[str | None, TrendEvent | None]:
    """
    Buy-and-hold trend watch: is the latest close above or below the
    50-day SMA, and did that flip since the last run?

    Returns (new_trend, event_or_None). `stored_trend is None` means this
    is the first run ever for this ticker -- record silently, no alert
    (there is nothing to have "transitioned" from). Pure function, no I/O.
    """
    if diag.sma50 is None:
        # Insufficient warmup for SMA(50) -- carry forward whatever we had
        # (None on a true bootstrap) rather than guessing.
        return stored_trend, None

    new_trend = "above" if diag.close >= diag.sma50 else "below"
    if stored_trend is None or stored_trend == new_trend:
        return new_trend, None

    event_type = "trend_break" if new_trend == "below" else "trend_resume"
    return new_trend, TrendEvent(ticker, event_type, diag.close, diag.sma50, date_str)


def evaluate_spec(
    ticker: str, df: pd.DataFrame, spec: StrategySpec, rule_state: dict[str, Any],
) -> tuple[dict[str, Any], SpecEvent | None, SpecSummaryRow]:
    """
    Evaluate one promoted rule against the latest bar of `df`.

    Returns (new_rule_state, event_or_None, summary_row). Pure function --
    no file or network I/O -- so it is directly unit-testable with a
    synthetic frame and an injected state dict.
    """
    entries, exits = spec.build_signals(df)
    entries = entries.reindex(df.index).fillna(False)
    exits = exits.reindex(df.index).fillna(False)

    last_date = df.index[-1]
    date_str = last_date.date().isoformat() if hasattr(last_date, "date") else str(last_date)[:10]
    close_today = float(df["Close"].iloc[-1])
    diag = compute_diagnostics(df)

    position = rule_state.get("position", "flat")
    event: SpecEvent | None = None
    fired = "no signal"

    if position == "flat":
        if bool(entries.iloc[-1]):
            new_state = {
                "position": "long",
                "entry_date": date_str,
                "entry_price": close_today,
                "bars_held": 0,
            }
            event = SpecEvent(ticker, spec, "entry", ["signal"], close_today, date_str)
            fired = "ENTRY"
        else:
            new_state = dict(rule_state)
    else:  # long
        reasons: list[str] = []

        if spec.stop_kind == "fixed_pct" and rule_state.get("entry_price") is not None:
            stop_level = float(rule_state["entry_price"]) * (1.0 - float(spec.stop_param) / 100.0)
            if pd.notna(close_today) and close_today < stop_level:
                reasons.append("fixed_stop")

        bars_held = int(rule_state.get("bars_held", 0))
        if spec.time_stop_days is not None:
            bars_held = _bars_held(df, rule_state.get("entry_date"), bars_held)
            if bars_held >= int(spec.time_stop_days):
                reasons.append("time_stop")
        else:
            bars_held += 1

        if bool(exits.iloc[-1]):
            reasons.append("rsi_exit")

        if reasons:
            new_state = _default_rule_state()
            event = SpecEvent(ticker, spec, "exit", reasons, close_today, date_str)
            fired = f"EXIT ({_format_reasons(reasons)})"
        else:
            new_state = dict(rule_state)
            new_state["bars_held"] = bars_held

    summary = SpecSummaryRow(
        ticker=ticker, spec_id=spec.id, position=position,
        rsi2=diag.rsi2, sma50=diag.sma50, close=close_today, fired=fired,
    )
    return new_state, event, summary


# ---------------------------------------------------------------------------
# Per-ticker orchestration -- pure, no I/O
# ---------------------------------------------------------------------------

def process_ticker(
    ticker: str, df: pd.DataFrame, specs: list[StrategySpec], ticker_state: dict[str, Any],
) -> TickerResult:
    """Evaluate every promoted spec for one ticker's latest bar."""
    ticker_state = ticker_state or {}
    last_date = df.index[-1]
    date_str = last_date.date().isoformat() if hasattr(last_date, "date") else str(last_date)[:10]

    rules_state = dict(ticker_state.get("rules", {}))
    # .get("trend") on a plain dict is backward compatible with a state file
    # written before the trend watch existed -- a missing key just reads as
    # None (bootstrap), never a crash.
    stored_trend = ticker_state.get("trend")
    stored_last_bar = ticker_state.get("last_bar_date")

    if stored_last_bar == date_str:
        # Idempotency / weekend-holiday guard: the latest bar we just
        # downloaded is the same one we already processed last run -- do
        # not re-evaluate signals (or the trend watch) against it, so a
        # second same-day run (or a weekend/holiday run with no fresh
        # close) can never fire a duplicate alert.
        diag = compute_diagnostics(df)
        # Display-only fallback: if the trend has never been bootstrapped
        # (brand-new ticker, or a legacy state file predating the trend
        # watch) but we already have enough bars for SMA(50), show today's
        # above/below relationship in the console for reference. This never
        # touches persisted state or fires an alert -- the real bootstrap
        # (which *does* get persisted) only happens on a genuine new bar.
        display_trend = stored_trend
        if display_trend is None and diag.sma50 is not None:
            display_trend = "above" if diag.close >= diag.sma50 else "below"
        rows = [
            SpecSummaryRow(
                ticker=ticker, spec_id=spec.id,
                position=rules_state.get(spec.id, _default_rule_state()).get("position", "flat"),
                rsi2=diag.rsi2, sma50=diag.sma50, close=diag.close,
                fired="no new bar", trend=display_trend,
            )
            for spec in specs
        ]
        return TickerResult(
            ticker=ticker, new_state=ticker_state, events=[], summary_rows=rows,
            no_new_bar=True, note=f"no new bar since last run ({date_str})",
        )

    new_rules_state: dict[str, Any] = {}
    events: list[SpecEvent] = []
    rows: list[SpecSummaryRow] = []
    for spec in specs:
        rule_state = rules_state.get(spec.id, _default_rule_state())
        new_rule_state, event, row = evaluate_spec(ticker, df, spec, rule_state)
        new_rules_state[spec.id] = new_rule_state
        if event is not None:
            events.append(event)
        rows.append(row)

    diag = compute_diagnostics(df)
    new_trend, trend_event = evaluate_trend(ticker, diag, stored_trend, date_str)
    for row in rows:
        row.trend = new_trend

    new_ticker_state = {"last_bar_date": date_str, "rules": new_rules_state, "trend": new_trend}
    return TickerResult(
        ticker=ticker, new_state=new_ticker_state, events=events, summary_rows=rows,
        no_new_bar=False, trend_event=trend_event,
    )


# ---------------------------------------------------------------------------
# Alert message building (collapses coincident signals across the two
# near-identical promoted rules into one message per ticker)
# ---------------------------------------------------------------------------

def build_alert_messages(events: list[SpecEvent]) -> list[str]:
    """
    One message per ticker. When both promoted rules fire the same
    (event, reasons) on the same day for the same ticker -- the common
    case, since they share every parameter except the time stop -- they
    are collapsed into a single line noting both rules; genuinely
    divergent outcomes (e.g. only the 10-day-time-stop rule exits while
    the 15-day rule is still holding) get their own lines.
    """
    messages: list[str] = []
    by_ticker: dict[str, list[SpecEvent]] = {}
    for e in events:
        by_ticker.setdefault(e.ticker, []).append(e)

    for ticker, tick_events in by_ticker.items():
        groups: dict[tuple[str, tuple[str, ...]], list[SpecEvent]] = {}
        for e in tick_events:
            key = (e.event, tuple(e.reasons))
            groups.setdefault(key, []).append(e)

        lines: list[str] = []
        for (etype, reasons), es in groups.items():
            rule_note = " & ".join(_rule_label(e.spec) for e in es)
            close = es[0].close
            description = es[0].spec.description
            if etype == "entry":
                lines.append(
                    f"ENTRY signal: {_ticker_label(ticker)} -- {description} "
                    f"(rule: {rule_note}) -- signal close RM {close:.2f}; if acting, "
                    f"reference execution is next session's open. Suggested stop: "
                    f"5% below fill. Not financial advice, informational signal only."
                )
            else:
                lines.append(
                    f"EXIT signal ({_format_reasons(reasons)}): {_ticker_label(ticker)} -- "
                    f"{description} (rule: {rule_note}) -- signal close RM {close:.2f}. "
                    f"Not financial advice, informational signal only."
                )
        messages.append("\n\n".join(lines))
    return messages


def build_trend_alert_messages(trend_events: list[TrendEvent]) -> list[str]:
    """One message per trend-watch transition (break or resume)."""
    messages: list[str] = []
    for te in trend_events:
        if te.event == "trend_break":
            messages.append(
                f"TREND BREAK: {_ticker_label(te.ticker)} closed below its 50-day SMA "
                f"(close RM {te.close:.2f} vs SMA RM {te.sma50:.2f}). The backtested "
                f"uptrend filter is now off -- if holding, this was the tested exit/risk "
                f"line. Not financial advice, informational signal only."
            )
        else:
            messages.append(
                f"TREND RESUME: {_ticker_label(te.ticker)} closed back above its 50-day "
                f"SMA (close RM {te.close:.2f} vs SMA RM {te.sma50:.2f}). The backtested "
                f"uptrend filter is back on. Not financial advice, informational signal "
                f"only."
            )
    return messages


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_alert(message: str, *, dry_run: bool) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("[scanner] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set -- skipping Telegram send.")
        return
    if dry_run:
        print("[scanner] --dry-run: Telegram send skipped.")
        return
    _send_telegram_raw(message)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(rows: list[SpecSummaryRow]) -> None:
    header = (
        f"{'ticker':<10}{'spec_id':<50}{'state':<7}{'RSI2':>8}{'SMA50':>10}"
        f"{'close':>10}  {'trend':<7}fired"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        rsi_s = f"{r.rsi2:.1f}" if r.rsi2 is not None else "n/a"
        sma_s = f"{r.sma50:.3f}" if r.sma50 is not None else "n/a"
        trend_s = r.trend if r.trend is not None else "n/a"
        print(
            f"{r.ticker:<10}{r.spec_id:<50}{r.position:<7}{rsi_s:>8}{sma_s:>10}"
            f"{r.close:>10.3f}  {trend_s:<7}{r.fired}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Daily signal-only scanner for the two promoted bursa_active rules "
            "on 5330.KL (TMK) and 0217.KL (Powerwell). Never places, proposes, "
            "or simulates placing an order."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate and print only; skip Telegram sends and state/log writes.",
    )
    args = parser.parse_args()

    print("[scanner] active_scanner -- SIGNAL-ONLY, no orders are ever placed.")

    try:
        specs = load_promoted_specs()
    except Exception as exc:
        print(f"[scanner] failed to load promoted rules: {exc!s}")
        return 1
    print(f"[scanner] loaded {len(specs)} promoted rule(s): {[s.id for s in specs]}")

    frames = download_history(TICKERS, period="1y")
    available = {
        t: df.sort_index() for t, df in frames.items()
        if t in TICKERS and df is not None and not df.empty
    }

    if not available:
        print("[scanner] yfinance returned no usable data for either target ticker. "
              "Aborting -- state untouched.")
        return 1

    state = load_state()
    all_rows: list[SpecSummaryRow] = []
    all_events: list[SpecEvent] = []
    all_trend_events: list[TrendEvent] = []
    new_state: dict[str, Any] = dict(state)

    for ticker in TICKERS:
        df = available.get(ticker)
        if df is None:
            print(f"[scanner] {ticker}: no data returned by yfinance this run -- skipped.")
            continue
        if len(df) < MIN_WARMUP_BARS:
            print(f"[scanner] {ticker}: only {len(df)} bar(s) available (< {MIN_WARMUP_BARS}) "
                  f"-- insufficient for SMA(50)/RSI warmup, skipping evaluation.")
            continue

        result = process_ticker(ticker, df, specs, state.get(ticker, {}))
        if result.note:
            print(f"[scanner] {ticker}: {result.note}")
        all_rows.extend(result.summary_rows)
        all_events.extend(result.events)
        if result.trend_event is not None:
            all_trend_events.append(result.trend_event)
        new_state[ticker] = result.new_state

    print()
    print("=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    print_summary(all_rows)
    print()

    if all_events or all_trend_events:
        log_records = [
            {
                "date": e.date,
                "ticker": e.ticker,
                "spec_id": e.spec.id,
                "event": e.event,
                "close": round(e.close, 4),
                "reason": e.reasons[0] if len(e.reasons) == 1 else e.reasons,
            }
            for e in all_events
        ] + [
            {
                "date": te.date,
                "ticker": te.ticker,
                "spec_id": None,
                "event": te.event,
                "close": round(te.close, 4),
                "reason": f"close {te.close:.4f} vs sma50 {te.sma50:.4f}",
            }
            for te in all_trend_events
        ]
        if args.dry_run:
            print(f"[scanner] --dry-run: {len(log_records)} signal event(s) computed but NOT "
                  f"written to {LOG_PATH.name} / {STATE_PATH.name}.")
        else:
            append_log(log_records)
            save_state(new_state)
            print(f"[scanner] {len(log_records)} signal event(s) logged -> {LOG_PATH.name}; "
                  f"state saved -> {STATE_PATH.name}")

        for message in build_alert_messages(all_events) + build_trend_alert_messages(all_trend_events):
            print("-" * 100)
            print(message)
            send_alert(message, dry_run=args.dry_run)
    else:
        print("[scanner] no entry/exit signal events today.")
        if not args.dry_run:
            save_state(new_state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
