"""
Bursa Discovery Scanner — Full-Market Mean Reversion (v2)
=========================================================
Scans the Bursa Malaysia (KLSE) listed-issuer universe for oversold
mean-reversion setups, applies the Maybank 2026 fee schedule, and returns a
Top-5 opportunities table ranked by net expected profit.

Ticker source priority
----------------------
1. `bursa_listings.csv` next to this script (parsed from Bursa's official
   "List of Companies" PDF — 1,058 issuers, authoritative).
2. Wikipedia scrape (network fallback).
3. Hand-curated 20 large-caps (last-resort offline fallback).

Strategy
--------
- Mean         : 20-period SMA (the "fair value" target for reversion)
- Trigger      : Close < Lower Bollinger Band (20, 2SD)  AND  RSI(14) < 30
- Trend filter : Close > 200-day SMA  (we only fade dips inside an uptrend)
- Stop         : 1 ATR(14) below entry
- Target       : 20-SMA (mean) — defines the Reward leg of the RRR

Maybank 2026 Fee Schedule (per leg; round-trip computes both)
-------------------------------------------------------------
- Brokerage : 0.10% of trade value, min RM 8.00
- SST       : 8% of brokerage
- Clearing  : 0.03% of trade value
- Stamp Duty: RM 1.50 per RM 1,000 of trade value (rounded UP)

Sanity Check
------------
A candidate is dropped if net expected profit < 3x total round-trip fees.

Requirements
------------
    pip install yfinance pandas numpy requests lxml beautifulsoup4

Run
---
    python bursa_scanner.py
"""

from __future__ import annotations

import csv
import math
import time
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CAPITAL_PER_TRADE_RM = 10_000.0     # Notional position size used to estimate profit
BATCH_SIZE = 50                     # Tickers per yfinance batch download
MAX_WORKERS = 8                     # Threads for batch processing
HISTORY_PERIOD = "1y"               # Enough for 200-SMA + buffer
TOP_N = 5                           # Final ranking depth
FEE_COVERAGE_MULTIPLE = 3.0         # Profit must exceed this x round-trip fees

# Mean-reversion parameters
SMA_FAST = 20
SMA_TREND = 200
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30
ATR_PERIOD = 14

# Path to the authoritative CSV (shipped beside the script)
LISTINGS_CSV = Path(__file__).parent / "bursa_listings.csv"

# ---------------------------------------------------------------------------
# TELEGRAM NOTIFICATIONS
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = "8200748506:AAEksgSmlgYn-BPCI6-hok3mbzkFo43go2Q"
TELEGRAM_CHAT_ID   = "8671734227"


ALERT_STATE_FILE = Path(__file__).parent / ".last_alert.json"


def send_telegram(message: str) -> None:
    """Send a message to the configured Telegram chat. Silently skips on error."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception:
        pass  # never let a notification failure crash the scanner


def save_alert_state(message: str) -> None:
    """Persist the last alert so the retry script can check for a reply."""
    import json, datetime
    state = {
        "sent_at": datetime.datetime.now().isoformat(),
        "message": message,
    }
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# PHASE 1 — TICKER DISCOVERY
# ---------------------------------------------------------------------------

# Hand-curated last-resort fallback: 20 large/mid-cap Main Market names.
FALLBACK_TICKERS = [
    "1155.KL",  # Maybank
    "1295.KL",  # Public Bank
    "1023.KL",  # CIMB Group
    "1066.KL",  # RHB Bank
    "5347.KL",  # Tenaga Nasional
    "6033.KL",  # Petronas Gas
    "5681.KL",  # Petronas Dagangan
    "6012.KL",  # Maxis
    "4863.KL",  # TM (Telekom Malaysia)
    "6947.KL",  # CelcomDigi
    "5398.KL",  # Gamuda
    "5285.KL",  # SD Guthrie (ex-Sime Darby Plantation)
    "4197.KL",  # Sime Darby
    "3182.KL",  # Genting
    "4715.KL",  # Genting Malaysia
    "5168.KL",  # Hartalega
    "7113.KL",  # Top Glove
    "0138.KL",  # MyEG Services
    "5225.KL",  # IHH Healthcare
    "1961.KL",  # IOI Corp
]


def _is_equity_code(code: str) -> bool:
    """
    True if a Bursa code is a tradable common stock.

    Filters out:
    - ETFs                  (suffix EA, EB) e.g. 0820EA = FBM KLCI ETF
    - Bond funds / sukuk    (suffix GA, GB) e.g. 0400GB = DanaInfra Sukuk
    - Stapled securities    (suffix SS)     e.g. 5235SS = KLCC Stapled
    yfinance handles bare 4-5 digit equity codes; everything else either
    isn't fungible by retail or has unreliable history feeds on Yahoo.
    """
    return code.isdigit() and 4 <= len(code) <= 5


def _load_listings_csv() -> list[str] | None:
    """Read the shipped CSV and return yfinance-formatted tickers, or None."""
    if not LISTINGS_CSV.exists():
        return None
    try:
        tickers: list[str] = []
        with LISTINGS_CSV.open() as f:
            for row in csv.DictReader(f):
                code = row["stock_code"].strip()
                if _is_equity_code(code):
                    tickers.append(f"{code}.KL")
        if len(tickers) < 100:
            return None
        print(f"[discover] Loaded {len(tickers)} equity tickers from {LISTINGS_CSV.name}.")
        return tickers
    except Exception as exc:
        print(f"[discover] CSV read failed: {exc!s}")
        return None


def _scrape_wikipedia() -> list[str] | None:
    """Scrape Wikipedia as a network fallback. Returns None on failure."""
    url = "https://en.wikipedia.org/wiki/Companies_listed_on_Bursa_Malaysia"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        codes: list[str] = []
        for tbl in tables:
            for col in tbl.columns:
                s = tbl[col].astype(str).str.strip()
                hits = s[s.str.match(r"^\d{4,5}$")]
                if len(hits) >= 20:
                    codes.extend(hits.tolist())
        codes = sorted(set(codes))
        if len(codes) < 50:
            return None
        print(f"[discover] Scraped {len(codes)} tickers from Wikipedia.")
        return [f"{c}.KL" for c in codes]
    except Exception as exc:
        print(f"[discover] Wikipedia scrape failed: {exc!s}")
        return None


def discover_bursa_tickers() -> list[str]:
    """
    Resolve the ticker universe with a three-tier fallback strategy:
    CSV (authoritative) -> Wikipedia (network) -> 20-ticker hard-coded list.
    """
    for source in (_load_listings_csv, _scrape_wikipedia):
        tickers = source()
        if tickers:
            return tickers
    print(f"[discover] All live sources failed. Using {len(FALLBACK_TICKERS)}-ticker fallback.")
    return FALLBACK_TICKERS.copy()


# ---------------------------------------------------------------------------
# PHASE 2 — FEE ENGINE (Maybank 2026)
# ---------------------------------------------------------------------------

def apply_fees(trade_value: float) -> dict:
    """
    Compute one-leg transaction costs per Maybank's 2026 retail rate card.

    Components
    ----------
    brokerage : 0.10% of trade value, minimum RM 8.00
    sst       : 8% of brokerage
    clearing  : 0.03% of trade value (real cap RM 1,000; never hit at RM 10k)
    stamp     : RM 1.50 per RM 1,000 (rounded UP to next thousand)

    Returns a breakdown dict including 'total'.
    """
    trade_value = float(trade_value)

    brokerage = max(trade_value * 0.001, 8.00)
    sst = brokerage * 0.08
    clearing = trade_value * 0.0003
    # Stamp duty rounds UP to the next RM 1,000 of contract value
    stamp = math.ceil(trade_value / 1000.0) * 1.50

    total = brokerage + sst + clearing + stamp
    return {
        "brokerage": round(brokerage, 2),
        "sst": round(sst, 2),
        "clearing": round(clearing, 2),
        "stamp": round(stamp, 2),
        "total": round(total, 2),
    }


def round_trip_fees(entry_value: float, exit_value: float) -> float:
    """Total fees for both the buy and the sell legs."""
    return apply_fees(entry_value)["total"] + apply_fees(exit_value)["total"]


# ---------------------------------------------------------------------------
# PHASE 2 — INDICATORS
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range (Wilder)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# PHASE 2 — STRATEGY EVALUATION
# ---------------------------------------------------------------------------

@dataclass
class Opportunity:
    ticker: str
    price: float
    mean_sma20: float
    distance_pct: float        # how far below the mean (negative %)
    stop_price: float
    rsi: float
    expected_profit_rm: float  # net of round-trip Maybank fees
    rrr: float                 # reward / risk, net of fees
    total_fees_rm: float


def evaluate_ticker(ticker: str, df: pd.DataFrame) -> Opportunity | None:
    """
    Apply the mean-reversion rules to one ticker's OHLCV history.
    Returns an Opportunity if all gates pass and the sanity check holds, else None.
    """
    if df is None or df.empty or len(df) < SMA_TREND + 5:
        return None

    df = df.dropna().copy()
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return None

    close = df["Close"]
    sma20 = close.rolling(SMA_FAST).mean()
    std20 = close.rolling(SMA_FAST).std()
    lower_bb = sma20 - BB_STD * std20
    sma200 = close.rolling(SMA_TREND).mean()
    rsi_series = rsi(close)
    atr_series = atr(df)

    last = -1
    price = float(close.iloc[last])
    mean = float(sma20.iloc[last])
    lb = float(lower_bb.iloc[last])
    trend = float(sma200.iloc[last])
    rsi_now = float(rsi_series.iloc[last])
    atr_now = float(atr_series.iloc[last])

    if any(math.isnan(x) for x in [price, mean, lb, trend, rsi_now, atr_now]):
        return None

    # ----- Entry gates -----
    if not (price < lb and rsi_now < RSI_OVERSOLD):
        return None
    if not (price > trend):
        return None
    if atr_now <= 0 or price <= 0:
        return None

    # ----- Trade sizing & projected P&L -----
    shares = math.floor(CAPITAL_PER_TRADE_RM / price / 100) * 100  # Bursa lot = 100
    if shares < 100:
        return None  # too expensive for our notional

    entry_value = shares * price
    target_value = shares * mean         # exit at the mean
    stop_price = price - atr_now         # 1 ATR stop

    gross_profit = target_value - entry_value
    fees = round_trip_fees(entry_value, target_value)
    net_profit = gross_profit - fees

    if net_profit <= 0:
        return None

    # ----- Sanity check: profit must dwarf fees -----
    if net_profit < FEE_COVERAGE_MULTIPLE * fees:
        return None

    risk_per_share = price - stop_price
    if risk_per_share <= 0:
        return None
    risk_total = (
        shares * risk_per_share
        + apply_fees(shares * stop_price)["total"]
        + apply_fees(entry_value)["total"]
    )
    rrr = net_profit / risk_total if risk_total > 0 else 0.0

    return Opportunity(
        ticker=ticker,
        price=round(price, 4),
        mean_sma20=round(mean, 4),
        distance_pct=round((price / mean - 1) * 100, 2),
        stop_price=round(stop_price, 4),
        rsi=round(rsi_now, 2),
        expected_profit_rm=round(net_profit, 2),
        rrr=round(rrr, 2),
        total_fees_rm=round(fees, 2),
    )


# ---------------------------------------------------------------------------
# PHASE 3 — BATCH EXECUTION
# ---------------------------------------------------------------------------

def chunk(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def download_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download one batch of tickers from yfinance. Returns ticker -> OHLCV df."""
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            period=HISTORY_PERIOD,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as exc:
        print(f"[batch] Download failed for {len(tickers)} tickers: {exc!s}")
        return {}

    out: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.get_level_values(0):
                out[t] = data[t].dropna(how="all")
    else:
        # Single-ticker case: yfinance returns a flat frame
        if len(tickers) == 1:
            out[tickers[0]] = data.dropna(how="all")
    return out


def scan_universe(tickers: list[str]) -> list[Opportunity]:
    """Iterate over batches in parallel, evaluate each ticker, collect winners."""
    opportunities: list[Opportunity] = []
    batches = list(chunk(tickers, BATCH_SIZE))
    print(f"[scan] Scanning {len(tickers)} tickers across {len(batches)} batches of {BATCH_SIZE}...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_batch, b): b for b in batches}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                frames = fut.result()
            except Exception as exc:
                print(f"[scan] Batch {i} errored: {exc!s}")
                continue

            hits_this_batch = 0
            for t, df in frames.items():
                opp = evaluate_ticker(t, df)
                if opp is not None:
                    opportunities.append(opp)
                    hits_this_batch += 1

            print(f"[scan]   Batch {i}/{len(batches)} done — {len(frames)} frames, {hits_this_batch} hits.")
            time.sleep(0.2)  # courtesy delay

    return opportunities


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def print_top_table(opps: list[Opportunity], n: int = TOP_N) -> None:
    if not opps:
        print(
            "\nNo qualifying mean-reversion setups today.\n"
            "Either the market is calm, the trend filter is excluding dips, "
            "or fees are eating the edge."
        )
        return

    ranked = sorted(opps, key=lambda o: o.expected_profit_rm, reverse=True)[:n]

    rows = []
    for r, o in enumerate(ranked, 1):
        rows.append({
            "#": r,
            "Ticker": o.ticker,
            "Price (RM)": f"{o.price:.3f}",
            "Dist. from Mean": f"{o.distance_pct:+.2f}%",
            "Net Profit (RM)": f"{o.expected_profit_rm:,.2f}",
            "Fees (RM)": f"{o.total_fees_rm:,.2f}",
            "RRR": f"{o.rrr:.2f}",
            "RSI": f"{o.rsi:.1f}",
        })

    out = pd.DataFrame(rows)
    print("\n" + "=" * 82)
    print(f"  TOP {len(ranked)} BURSA MEAN-REVERSION OPPORTUNITIES (net of Maybank fees)")
    print(f"  Notional per trade: RM {CAPITAL_PER_TRADE_RM:,.0f}  |  Min profit/fee ratio: {FEE_COVERAGE_MULTIPLE}x")
    print("=" * 82)
    print(out.to_string(index=False))
    print("=" * 82)
    print(f"  Total qualifying setups in universe: {len(opps)}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    tickers = discover_bursa_tickers()
    opps = scan_universe(tickers)
    print_top_table(opps, TOP_N)

    if not opps:
        return

    # Build a compact Telegram summary for the top setups
    ranked = sorted(opps, key=lambda o: o.expected_profit_rm, reverse=True)[:TOP_N]
    lines = [f"🔍 Bursa Scanner — {len(opps)} setup(s) found today\n"]
    for i, o in enumerate(ranked, 1):
        lines.append(
            f"{i}. {o.ticker}  RM{o.price:.3f}  "
            f"RSI {o.rsi:.0f}  "
            f"Dist {o.distance_pct:+.1f}%  "
            f"Net profit RM{o.expected_profit_rm:,.0f}  "
            f"RRR {o.rrr:.2f}"
        )
    lines.append(f"\nCapital per trade: RM{CAPITAL_PER_TRADE_RM:,.0f}")
    lines.append("\nReply 'ok' to acknowledge — otherwise I'll remind you in 2 hours.")
    alert_msg = "\n".join(lines)
    send_telegram(alert_msg)
    save_alert_state(alert_msg)


if __name__ == "__main__":
    main()
