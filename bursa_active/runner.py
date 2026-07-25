"""
Grid orchestration: IS ranking, OOS/TMK/controls evaluation, promotion,
and all `active_results/` CSV/JSON outputs (DESIGN.md "runner.py").

Windowing methodology (read this before touching the window constants)
------------------------------------------------------------------------
Signals are computed **once** on each ticker's full cached history via
`spec.build_signals(df_full)` -- never on a pre-sliced frame. A window is
applied afterwards by slicing the OHLCV frame to `[start:end]` and
reindexing the already-computed entries/exits Series onto that slice's
index. Any entry/exit signal dated outside the slice's index is simply not
present in the reindexed Series, so it cannot open or close a trade in that
window -- which is exactly "restrict trades to entries whose signal date
falls inside the window". This keeps every indicator's warmup honest
(computed against the real history before the window, not a truncated
stub) while still producing window-scoped trades and metrics.

One known limitation this does not (and structurally cannot, without
editing engine.py) fully solve: `engine.run_backtest` recomputes its *own*
ATR14 internally from whatever `df` it is handed, purely for the
`atr_trail` stop. For a spec using that stop, a trade opened in the first
~14 bars of a sliced window may therefore go unprotected by the trailing
stop until the engine's internal ATR warms up on that slice -- even though
the entry/exit *signals* themselves had full warmup from the real history.
This is a consequence of engine.py's API (by design, it is a pure
`(df, entries, exits, stop_kind, stop_param) -> BacktestResult` function
with no "extra warmup rows" concept) rather than a bug in it; it is called
out again in summary.json and the final report rather than patched here.

Promotion rule (DESIGN.md "Anti-overfitting protocol" item 3)
------------------------------------------------------------------------
"profitable after fees in-sample AND in both OOS sets AND beats
buy-and-hold risk-adjusted (or materially cuts drawdown) on at least one
target" is operationalised as, per spec:

* profitable(window)   := n_trades > 0 and total_return_pct > 0
* beats_buyhold(window) := total_return_pct > buyhold_return_pct
* dd_edge(window)      := buyhold_max_drawdown_pct > 0 and
                          max_drawdown_pct <= DD_MATERIALITY_RATIO * buyhold_max_drawdown_pct
* target_edge          := (beats_buyhold or dd_edge) on powerwell_oos OR on tmk_full
* promoted             := profitable(is) and profitable(oos) and profitable(tmk) and target_edge

`DD_MATERIALITY_RATIO` (0.75, i.e. strategy drawdown must be at most 75% of
buy-and-hold's -- a 25%+ relative cut) is a documented choice: DESIGN.md
says "materially lower" without a number. Every individual criterion is
kept as its own column in grid_oos.csv / summary.json so nothing is folded
into the boolean silently.

Controls (DESIGN.md item 4) never gate `promoted` -- they can only add the
informational `flagged_target_only_suspect` column (a promoted rule that
is profitable on essentially none of the 6 controls).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import data_audit
from .engine import BacktestResult, run_backtest
from .strategies import StrategySpec, build_grid


__all__ = ["main", "TOP_N_DEFAULT"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "active_results"
AUDIT_PATH = RESULTS_DIR / "data_audit.json"

BUDGET_RM = 20_000.0
TOP_N_DEFAULT = 15

TMK = "5330.KL"          # validation only -- too short to tune on
POWERWELL = "0217.KL"    # tuning target, IS/OOS split

POWERWELL_IS_START = "2020-01-22"
POWERWELL_IS_END = "2023-12-31"
POWERWELL_OOS_START = "2024-01-01"

#: "Materially lower" drawdown, as a fraction of buy-and-hold's drawdown.
#: DESIGN.md doesn't give a number -- this is our documented choice.
DD_MATERIALITY_RATIO = 0.75

#: A promoted rule profitable on this few (or fewer) of the 6 controls is
#: flagged suspect ("only works on the two targets" -- DESIGN.md item 4).
SUSPECT_CONTROL_PROFIT_FRAC = 1.0 / 6.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_controls() -> list[str]:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"{AUDIT_PATH} not found -- run data_audit.run_audit() / "
            f"active_backtest.py first so the controls list exists."
        )
    audit = json.loads(AUDIT_PATH.read_text())
    controls = audit["controls"]
    if not controls:
        raise ValueError("data_audit.json has an empty controls list.")
    return controls


def _load_all_data(controls: list[str]) -> dict[str, pd.DataFrame]:
    tickers = [TMK, POWERWELL] + controls
    data: dict[str, pd.DataFrame] = {}
    for t in tickers:
        data[t] = data_audit.load_cached(t)
    return data


# ---------------------------------------------------------------------------
# Windowed backtest helper
# ---------------------------------------------------------------------------

def _reindex_bool(sig: pd.Series, index: pd.Index) -> pd.Series:
    return sig.reindex(index).fillna(False).astype(bool)


def run_spec_on_window(
    spec: StrategySpec,
    df_full: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """
    Compute signals on `df_full` (full history), then slice to `[start:end]`
    and run the engine on the slice. See the module docstring's "Windowing
    methodology" for why signals are built before slicing, not after.
    """
    entries_full, exits_full = spec.build_signals(df_full)
    df_window = df_full.loc[start:end]
    entries = _reindex_bool(entries_full, df_window.index)
    exits = _reindex_bool(exits_full, df_window.index)
    result = run_backtest(
        df_window, entries, exits,
        stop_kind=spec.stop_kind, stop_param=spec.stop_param,
        time_stop_days=spec.time_stop_days, budget_rm=BUDGET_RM,
    )
    return result, df_window


def _buyhold_max_dd_pct(df_window: pd.DataFrame) -> float:
    """Peak-to-trough decline of a simple buy-and-hold of `Close` (positive %)."""
    if df_window is None or "Close" not in getattr(df_window, "columns", []):
        return 0.0
    close = df_window["Close"].astype(float).dropna()
    if len(close) < 2 or close.iloc[0] <= 0:
        return 0.0
    peak = close.cummax()
    dd = (close / peak.where(peak != 0) - 1.0).min()
    if not pd.notna(dd):
        return 0.0
    return round(abs(float(dd)) * 100.0, 4)


def _spec_row(
    spec: StrategySpec, result: BacktestResult, ticker: str, window: str,
    df_window: pd.DataFrame,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "spec_id": spec.id,
        "family": spec.family,
        "timeframe": spec.timeframe,
        "ticker": ticker,
        "window": window,
        "stop_kind": spec.stop_kind,
        "stop_param": spec.stop_param,
        "time_stop_days": spec.time_stop_days,
    }
    row.update(result.metrics)
    row["buyhold_max_drawdown_pct"] = _buyhold_max_dd_pct(df_window)
    return row


# ---------------------------------------------------------------------------
# Trade/equity output files
# ---------------------------------------------------------------------------

def _write_trades_and_equity(
    spec: StrategySpec, ticker: str,
    window_results: list[tuple[str, BacktestResult]],
) -> None:
    trade_frames = []
    for window, result in window_results:
        tf = result.trades_frame()
        if not tf.empty:
            tf = tf.copy()
            tf.insert(0, "window", window)
            trade_frames.append(tf)
    trades_df = (
        pd.concat(trade_frames, ignore_index=True) if trade_frames
        else pd.DataFrame(columns=["window", "entry_date", "entry_price", "shares",
                                   "exit_date", "exit_price", "entry_fees",
                                   "exit_fees", "pnl_gross", "pnl_net",
                                   "return_pct", "holding_days", "exit_reason",
                                   "open_at_end"])
    )
    trades_df.to_csv(RESULTS_DIR / f"trades_{spec.id}_{ticker}.csv", index=False)

    equity_frames = []
    for window, result in window_results:
        eq = result.equity.rename("equity").to_frame()
        eq["window"] = window
        equity_frames.append(eq)
    equity_df = (
        pd.concat(equity_frames) if equity_frames
        else pd.DataFrame(columns=["equity", "window"])
    )
    equity_df.index.name = "date"
    equity_df.to_csv(RESULTS_DIR / f"equity_{spec.id}_{ticker}.csv")


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

def _profitable(row: pd.Series) -> bool:
    return bool(row["n_trades"] > 0 and row["total_return_pct"] > 0)


def _beats_buyhold(row: pd.Series) -> bool:
    return bool(row["total_return_pct"] > row["buyhold_return_pct"])


def _dd_edge(row: pd.Series) -> bool:
    bh_dd = row["buyhold_max_drawdown_pct"]
    if bh_dd <= 0:
        return False
    return bool(row["max_drawdown_pct"] <= DD_MATERIALITY_RATIO * bh_dd)


def compute_promotions(grid_oos_df: pd.DataFrame, top_ids: list[str]) -> pd.DataFrame:
    """Per-spec promotion verdict + every individual criterion, spelled out."""
    records = []
    for spec_id in top_ids:
        sub = grid_oos_df[grid_oos_df["spec_id"] == spec_id]
        is_row = sub[sub["window"] == "powerwell_is"].iloc[0]
        oos_row = sub[sub["window"] == "powerwell_oos"].iloc[0]
        tmk_row = sub[sub["window"] == "tmk_full"].iloc[0]
        control_rows = sub[sub["window"] == "control_full"]

        profitable_is = _profitable(is_row)
        profitable_oos = _profitable(oos_row)
        profitable_tmk = _profitable(tmk_row)

        beats_bh_oos, dd_edge_oos = _beats_buyhold(oos_row), _dd_edge(oos_row)
        beats_bh_tmk, dd_edge_tmk = _beats_buyhold(tmk_row), _dd_edge(tmk_row)
        oos_edge = beats_bh_oos or dd_edge_oos
        tmk_edge = beats_bh_tmk or dd_edge_tmk
        target_edge = oos_edge or tmk_edge

        promoted = profitable_is and profitable_oos and profitable_tmk and target_edge

        control_n = len(control_rows)
        control_profitable_count = int((control_rows["total_return_pct"] > 0).sum()) if control_n else 0
        control_profit_frac = (control_profitable_count / control_n) if control_n else 0.0
        flagged_target_only = bool(
            promoted and control_n > 0 and control_profit_frac <= SUSPECT_CONTROL_PROFIT_FRAC
        )

        records.append({
            "spec_id": spec_id,
            "profitable_is": profitable_is,
            "profitable_oos": profitable_oos,
            "profitable_tmk": profitable_tmk,
            "beats_buyhold_oos": beats_bh_oos,
            "beats_buyhold_tmk": beats_bh_tmk,
            "dd_edge_oos": dd_edge_oos,
            "dd_edge_tmk": dd_edge_tmk,
            "target_edge": target_edge,
            "promoted": promoted,
            "control_n": control_n,
            "control_profitable_count": control_profitable_count,
            "control_profitable_frac": round(control_profit_frac, 4),
            "flagged_target_only_suspect": flagged_target_only,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(top_n: int = TOP_N_DEFAULT) -> dict[str, Any]:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    controls = _load_controls()
    print(f"[runner] controls loaded from data_audit.json: {controls}")

    data = _load_all_data(controls)
    df_powerwell = data[POWERWELL]
    df_tmk = data[TMK]
    print(f"[runner] Powerwell (0217.KL): {len(df_powerwell)} bars "
          f"{df_powerwell.index[0].date()} -> {df_powerwell.index[-1].date()}")
    print(f"[runner] TMK (5330.KL):      {len(df_tmk)} bars "
          f"{df_tmk.index[0].date()} -> {df_tmk.index[-1].date()}")

    grid = build_grid()
    grid_family_counts = dict(Counter(s.family for s in grid))
    print(f"[runner] grid built: {len(grid)} specs across "
          f"{len(grid_family_counts)} families")

    # --- Pass 1: rank the full grid on Powerwell in-sample -----------------
    is_rows = []
    for i, spec in enumerate(grid, 1):
        result, df_win = run_spec_on_window(
            spec, df_powerwell, POWERWELL_IS_START, POWERWELL_IS_END,
        )
        is_rows.append(_spec_row(spec, result, POWERWELL, "powerwell_is", df_win))
        if i % 10 == 0 or i == len(grid):
            print(f"[runner] {i}/{len(grid)} specs (IS ranking)...")

    grid_is_df = pd.DataFrame(is_rows)
    grid_is_df.to_csv(RESULTS_DIR / "grid_is.csv", index=False)
    print(f"[runner] wrote {RESULTS_DIR / 'grid_is.csv'} ({len(grid_is_df)} rows)")

    eligible = grid_is_df[~grid_is_df["insufficient_sample"]].copy()
    eligible = eligible.sort_values(
        by=["expectancy_pct_per_trade", "profit_factor", "max_drawdown_pct"],
        ascending=[False, False, True],
    )
    excluded_n = len(grid_is_df) - len(eligible)
    top = eligible.head(top_n)
    top_ids = top["spec_id"].tolist()
    print(f"[runner] {excluded_n}/{len(grid_is_df)} specs excluded "
          f"(insufficient_sample, n_trades < 8)")
    print(f"[runner] top {len(top_ids)} IS rules selected (of {len(eligible)} eligible)")

    spec_by_id = {s.id: s for s in grid}

    # --- Pass 2: top-N on Powerwell OOS + TMK full + 6 controls -------------
    oos_rows: list[dict[str, Any]] = []
    for n, spec_id in enumerate(top_ids, 1):
        spec = spec_by_id[spec_id]
        per_ticker_results: dict[str, list[tuple[str, BacktestResult]]] = {}

        def _run(df_full: pd.DataFrame, start: str | None, end: str | None,
                 ticker: str, window: str) -> None:
            result, df_win = run_spec_on_window(spec, df_full, start, end)
            oos_rows.append(_spec_row(spec, result, ticker, window, df_win))
            per_ticker_results.setdefault(ticker, []).append((window, result))

        _run(df_powerwell, POWERWELL_IS_START, POWERWELL_IS_END, POWERWELL, "powerwell_is")
        _run(df_powerwell, POWERWELL_OOS_START, None, POWERWELL, "powerwell_oos")
        _run(df_tmk, None, None, TMK, "tmk_full")
        for c in controls:
            _run(data[c], None, None, c, "control_full")

        for ticker, window_results in per_ticker_results.items():
            _write_trades_and_equity(spec, ticker, window_results)

        print(f"[runner] top-{top_n} eval {n}/{len(top_ids)}: {spec_id} done")

    grid_oos_df = pd.DataFrame(oos_rows)
    grid_oos_df.to_csv(RESULTS_DIR / "grid_oos.csv", index=False)
    print(f"[runner] wrote {RESULTS_DIR / 'grid_oos.csv'} ({len(grid_oos_df)} rows)")

    controls_df = grid_oos_df[grid_oos_df["window"] == "control_full"]
    controls_df.to_csv(RESULTS_DIR / "controls.csv", index=False)
    print(f"[runner] wrote {RESULTS_DIR / 'controls.csv'} ({len(controls_df)} rows)")

    # --- Promotion -----------------------------------------------------------
    promotions_df = compute_promotions(grid_oos_df, top_ids)
    promoted_ids = promotions_df.loc[promotions_df["promoted"], "spec_id"].tolist()
    print(f"[runner] promoted: {promoted_ids if promoted_ids else '(none)'}")

    promoted_rules = []
    for spec_id in promoted_ids:
        spec = spec_by_id[spec_id]
        promoted_rules.append({
            "spec_id": spec.id,
            "family": spec.family,
            "timeframe": spec.timeframe,
            "description": spec.description,
            "stop_kind": spec.stop_kind,
            "stop_param": spec.stop_param,
            "time_stop_days": spec.time_stop_days,
        })
    (RESULTS_DIR / "promoted_rules.json").write_text(
        json.dumps(promoted_rules, indent=2, default=str),
    )
    print(f"[runner] wrote {RESULTS_DIR / 'promoted_rules.json'} "
          f"({len(promoted_rules)} promoted)")

    # --- Summary --------------------------------------------------------------
    top_table_cols = ["spec_id", "family", "n_trades", "expectancy_pct_per_trade",
                      "profit_factor", "max_drawdown_pct", "win_rate",
                      "total_return_pct", "sharpe"]
    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - t_start, 2),
        "grid_size": len(grid),
        "grid_family_counts": grid_family_counts,
        "budget_rm": BUDGET_RM,
        "windows": {
            "powerwell_is": [POWERWELL_IS_START, POWERWELL_IS_END],
            "powerwell_oos": [POWERWELL_OOS_START, str(df_powerwell.index[-1].date())],
            "tmk_full": [str(df_tmk.index[0].date()), str(df_tmk.index[-1].date())],
            "controls": controls,
        },
        "excluded_insufficient_sample": int(excluded_n),
        "top_n_requested": top_n,
        "top_15": top[top_table_cols].to_dict("records"),
        "promotions": promotions_df.to_dict("records"),
        "promoted_rule_ids": promoted_ids,
        "dd_materiality_ratio": DD_MATERIALITY_RATIO,
        "suspect_control_profit_frac": SUSPECT_CONTROL_PROFIT_FRAC,
        "notes": [
            "Signals are computed once on each ticker's full cached history; "
            "window slicing only restricts which entries can open trades "
            "(see runner.run_spec_on_window docstring).",
            "engine.run_backtest recomputes ATR14 internally from whatever df "
            "it receives, so an atr_trail stop may be unprotected for the "
            "first ~14 bars of a sliced window -- a consequence of engine.py's "
            "API, not of the signal construction here.",
            f"'Materially lower drawdown' means strategy max_drawdown_pct <= "
            f"{DD_MATERIALITY_RATIO} x buy-and-hold max_drawdown_pct over the "
            f"same window -- DESIGN.md gives no exact number; this is our "
            f"documented threshold.",
            "Control performance never gates `promoted`; it only sets "
            "`flagged_target_only_suspect` per DESIGN.md item 4.",
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[runner] wrote {RESULTS_DIR / 'summary.json'}")
    print(f"[runner] done in {summary['elapsed_sec']}s")

    return summary


if __name__ == "__main__":
    main()
