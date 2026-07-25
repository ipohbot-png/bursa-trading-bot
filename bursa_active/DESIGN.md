# Bursa Active — Single-Stock Technical Backtester (Design Spec)

Goal: find technical entry/exit rules that work on BOTH TMK Chemical
(5330.KL) and Powerwell Holdings (0217.KL), netted through the Maybank
2026 fee schedule, and ship a daily Telegram alert scanner for the
winning rules. Signal-only: nothing in this package ever places orders.

## Tickers

| Name | Ticker | Listed | Role |
|---|---|---|---|
| TMK Chemical | 5330.KL | Dec 2024 | target — VALIDATION ONLY (too short to tune on) |
| Powerwell Holdings | 0217.KL | Dec 2019 | target — tuning + OOS split |
| Controls (6 comparable Main/ACE names, chosen in data audit) | — | — | overfitting canary |

## Anti-overfitting protocol (non-negotiable)

1. Parameters are tuned ONLY on Powerwell in-sample: 2020-01-01 → 2023-12-31.
2. Out-of-sample: Powerwell 2024-01-01 → present, and ALL of TMK's history.
3. A rule "wins" only if it is profitable after fees in-sample AND in both
   OOS sets AND beats buy-and-hold risk-adjusted (or materially cuts drawdown)
   on at least one target.
4. Control stocks: winning rules are re-run on 6 comparable KLSE names.
   Report the spread. A rule that only works on the two targets is flagged
   as suspect in the report — never silently promoted.
5. No per-stock parameter tuning. One rule set, shared parameters.

## Execution & fee model

- Daily OHLCV from yfinance via `bursa_momentum.data.download_history`
  (auto_adjust=True).
- Signal computed on day t close → fill at day t+1 OPEN. No same-bar fills.
- If t+1 open is missing/zero, fill at t+1 close; if no bar, next available open.
- Long-only. One position per stock at a time. No pyramiding, no shorting.
- Position size: RM 20,000 budget per trade; shares = floor(20000 / fill_price
  / 100) * 100 (board lot). If lots == 0, skip trade (log it).
- Fees on BOTH sides via `bursa_momentum.fees` (brokerage 0.10% min RM8,
  SST 8% on brokerage, clearing 0.03%, stamp duty RM1.50/RM1000 rounded up).
- Stops: evaluated on CLOSE (close below stop → exit next open). No intraday
  stop fills — daily data cannot honestly model them. State this in the report.

## Package layout

```
bursa_active/
├── __init__.py
├── DESIGN.md          — this file
├── indicators.py      — pure-pandas indicator library (no TA-Lib)
├── engine.py          — single-stock long-only event-driven backtester
├── metrics.py         — per-run performance metrics
├── strategies.py      — rule definitions + parameter grids
├── runner.py          — grid orchestration: IS/OOS/controls, ranking, CSV out
└── data_audit.py      — fetch + validate + cache OHLCV for all tickers
active_backtest.py     — CLI entry point (project root)
active_scanner.py      — daily live signal scanner + Telegram (project root)
active_results/        — all outputs (grid CSVs, trades, equity curves, report)
```

## indicators.py

Pure functions, `pd.Series/DataFrame` in → `pd.Series` out, vectorized:
`sma, ema, macd (line/signal/hist), rsi (Wilder), stochastic (%K/%D),
bollinger (mid/upper/lower/bandwidth), atr (Wilder), donchian (upper/lower),
rolling_max, volume_sma`. Each must handle NaN warmup correctly (no lookahead:
value at index i uses data ≤ i only).

## engine.py

```python
@dataclass
class Trade:  # entry_date, entry_price, shares, exit_date, exit_price,
              # entry_fees, exit_fees, pnl_gross, pnl_net, return_pct,
              # holding_days, exit_reason ("signal"|"stop"|"trail"|"time"|"eod")

def run_backtest(df: pd.DataFrame, entries: pd.Series, exits: pd.Series,
                 *, stop_kind: str | None, stop_param: float,
                 budget_rm: float = 20_000.0) -> BacktestResult
```

- `entries`/`exits` are boolean Series aligned to df.index (True on signal day t).
- Engine walks bars: not in position + entry[t] → buy at open[t+1].
  In position: stop check on close[t] (ATR trail / fixed pct), exit signal,
  or time stop → sell at open[t+1]. Stop beats signal if both fire.
- Open position at end of data closed at last close, exit_reason="eod",
  flagged `open_at_end=True`.
- `BacktestResult`: trades list, daily equity Series (cash + mark-to-market),
  exposure, and the metrics dict from metrics.py.

## metrics.py

total_return_pct, cagr, max_drawdown (on equity curve), win_rate,
profit_factor, avg_win_pct, avg_loss_pct, expectancy_pct_per_trade,
n_trades, avg_holding_days, exposure_pct, total_fees_rm,
fee_drag_pct (fees / gross pnl), buyhold_return_pct over same window,
sharpe (daily, annualized √252, on equity returns).
Rules with n_trades < 8 in-sample are marked "insufficient sample" and
excluded from ranking.

## strategies.py — the grid

Every strategy = (entry rule, exit rule, stop). Families:

**Trend / position (weeks–months):**
- SMA cross: fast {10,20} × slow {50,100}; exit on reverse cross. ATR trail {none, 3.0}.
- EMA cross: {12/50, 20/100}; same exits.
- MACD: (fast,slow,signal) ∈ {(12,26,9), (8,17,9)}; entry line>signal cross while
  close>SMA{50,100}; exit reverse cross. ATR trail {none, 3.0}.
- Donchian: entry {20,55}-day high breakout; exit {10,20}-day low break. ATR trail 3.0 variant.

**Swing / pullback (3–15 days):**
- RSI pullback: trend filter close>SMA{50,100}; entry RSI(2)<{10,25} or RSI(14)<{30,40};
  exit RSI(2)>75 / RSI(14)>{60,70} or time stop {10,15} days. Fixed stop {5%,8%} variants.
- Stochastic: %K({10,14},3)<{20,25} crossing up over %D, trend filter close>SMA{50,100};
  exit %K>80; time stop 15d; fixed stop {5%,8%}.
- Bollinger mean-revert: close<lower(20,{2.0,2.5}) while close>SMA{50,100};
  exit at mid band; time stop 15d; stop {5%,8%}.
- Bollinger breakout: close>upper(20,2) after bandwidth in bottom {25%,40%} of
  trailing 120d; exit close<mid; ATR trail {2.5,3.0}.

**Volume breakout (swing):**
- {20,55}-day high AND volume>{1.5,2.0}× volume_sma(20); exit ATR trail {2.5,3.0}
  or close<SMA20.

Grid ≈ 150–300 combos total. Keep it enumerable in a list[StrategySpec]; each
spec has a stable string id like `sma_cross_10_50_trail3`.

## runner.py

1. Load cached data (from data_audit) for targets + controls.
2. Run full grid on Powerwell IS → rank by (expectancy after fees, profit
   factor, max DD), drop insufficient-sample rules.
3. Top ~15 IS rules → run on Powerwell OOS + TMK full + 6 controls.
4. Emit `active_results/grid_is.csv`, `grid_oos.csv`, `controls.csv`,
   per-winner `trades_<id>_<ticker>.csv` + equity curves, `summary.json`.
5. Deterministic: no randomness anywhere.

## active_scanner.py (Phase 7)

- Daily run (Task Scheduler / manual). Downloads latest bars for the two
  targets, evaluates ONLY the promoted winning rules (read from
  `active_results/promoted_rules.json`), compares today vs yesterday signal
  state, sends Telegram message on new entry/exit/stop trigger, appends to
  `active_paper_log.json`. Reuses telegram send util from bursa_momentum
  scanner. Never trades.

## Non-goals

- No intraday data, no shorting, no leverage, no order placement.
- No per-stock parameter optimization.
- Do not modify anything in `bursa_momentum/` (import from it only).
