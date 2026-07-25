"""
Data audit — fetch, cache and validate OHLCV history for bursa_active.

Downloads full daily history for the two single-stock backtest targets
(TMK Chemical 5330.KL, Powerwell Holdings 0217.KL) plus a handful of
comparable KLSE control candidates, caches each ticker's cleaned frame to
disk, and runs a battery of sanity checks — coverage, zero-volume days,
one-day price shocks, calendar gaps, liquidity — whose results are printed
as a table and saved to `active_results/data_audit.json`.

The 6 control candidates that come back with the healthiest data become the
CONTROLS canary set used by the rest of `bursa_active` (see DESIGN.md).

Run directly:
    python -m bursa_active.data_audit
    python bursa_active/data_audit.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running both as a package module (python -m bursa_active.data_audit)
# and as a plain script (python bursa_active/data_audit.py).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bursa_momentum.data import download_history  # noqa: E402

try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _HAS_PARQUET = True
    except ImportError:
        _HAS_PARQUET = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / ".cache" / "active"
OUTPUT_DIR   = PROJECT_ROOT / "active_results"
REPORT_PATH  = OUTPUT_DIR / "data_audit.json"

START_DATE = "2019-01-01"

TARGETS: list[str] = ["5330.KL", "0217.KL"]  # TMK Chemical, Powerwell Holdings

# Comparable mid/small-cap Main/ACE industrials & manufacturers, verified
# against bursa_listings.csv. The audit downloads all of these and keeps the
# 6 with the healthiest data as the CONTROLS canary set (DESIGN.md item 4).
CONTROL_CANDIDATES: list[str] = [
    "0233.KL",  # PEKAT Group
    "0215.KL",  # Solarvest Holdings (SLVEST)
    "0151.KL",  # Kelington Group (KGB)
    "7231.KL",  # Wellcall Holdings
    "7133.KL",  # United U-Li Corp (ULICORP)
    "8117.KL",  # PGF Capital
    "7245.KL",  # Citaglobal (CITAGLB)
    "7773.KL",  # EP Manufacturing (EPMB)
    # Extra fallbacks, tried only if the 8 above don't yield 6 clean names.
    "7034.KL",  # Thong Guan Industries (plastic packaging manufacturer)
    "7247.KL",  # SCGM Bhd (plastic packaging manufacturer)
    "7089.KL",  # Lii Hen Industries (furniture manufacturer)
]

ALL_TICKERS: list[str] = TARGETS + CONTROL_CANDIDATES

#: Tickers whose latest download failed and that fell back to a stale
#: on-disk cache during the most recent `fetch_all` call. Consumed by
#: `run_audit` to flag the affected audit records.
STALE_CACHE_TICKERS: set[str] = set()

BUDGET_RM               = 20_000.0   # DESIGN.md position size
LIQUIDITY_MAX_PARTICIPATION_PCT = 10.0  # position must be <10% of median daily value
SPLIT_FLAG_THRESHOLD_PCT = 40.0      # single-day |close-to-close| move flagged as suspect
GAP_DAYS_THRESHOLD       = 7         # calendar gap between consecutive bars
MIN_CONTROL_BARS         = 200       # ~1 trading year — floor for a usable control


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------

def _cache_file_for(ticker: str) -> Path | None:
    """Return the existing cache file for `ticker` (parquet preferred), or None."""
    for ext in ("parquet", "csv"):
        p = CACHE_DIR / f"{ticker}.{ext}"
        if p.exists():
            return p
    return None


def _is_fresh(path: Path, max_age_hours: int = 24) -> bool:
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and trim a raw yfinance frame to OHLCV columns."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep]
    return df.dropna(how="all")


def save_cache(ticker: str, df: pd.DataFrame) -> Path:
    """Write a cleaned OHLCV frame to disk (parquet if available, else CSV)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _HAS_PARQUET:
        path = CACHE_DIR / f"{ticker}.parquet"
        df.to_parquet(path)
    else:
        path = CACHE_DIR / f"{ticker}.csv"
        df.to_csv(path)
    return path


def load_cached(ticker: str) -> pd.DataFrame:
    """Load a ticker's cached OHLCV frame from disk (parquet if present, else CSV)."""
    parquet_path = CACHE_DIR / f"{ticker}.parquet"
    csv_path = CACHE_DIR / f"{ticker}.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"No cached data for {ticker} in {CACHE_DIR}")
    df.index = pd.to_datetime(df.index)
    return df


def fetch_all(
    tickers: list[str] | None = None,
    *,
    force: bool = False,
    start: str = START_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Download + cache full OHLCV history for `tickers` (default ALL_TICKERS).

    Skips any ticker whose cache file is fresher than 24h unless force=True.
    yfinance is occasionally flaky, so a ticker that comes back empty (or
    errors) on the first attempt is retried once before being given up on.

    If the download still fails but a **stale cache** exists on disk, the
    cached frame is used rather than dropping the ticker from the study
    (a flaky Yahoo morning must not silently shrink the control set). The
    ticker is recorded in `STALE_CACHE_TICKERS` so `run_audit` can mark it
    `stale_cache_used` in the audit record.
    """
    tickers = tickers or ALL_TICKERS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STALE_CACHE_TICKERS.clear()

    results: dict[str, pd.DataFrame] = {}
    to_download: list[str] = []

    for t in tickers:
        cached = _cache_file_for(t)
        if cached is not None and not force and _is_fresh(cached):
            print(f"[audit] {t}: cache fresh ({cached.name}) — skipping download.")
            results[t] = load_cached(t)
        else:
            to_download.append(t)

    if to_download:
        frames = download_history(to_download, start=start)
        missing = [t for t in to_download if t not in frames or frames[t].empty]
        if missing:
            print(f"[audit] {len(missing)} ticker(s) came back empty — retrying once: {missing}")
            retry = download_history(missing, start=start)
            frames.update(retry)

        for t in to_download:
            df = frames.get(t)
            if df is None or df.empty:
                # Download failed — fall back to whatever is cached, even
                # if it is older than the freshness window.
                try:
                    cached_df = load_cached(t)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    print(f"[audit]   {t}: NO DATA after retry and no cache "
                          f"({exc.__class__.__name__}) — skipped.")
                    continue
                if cached_df is None or cached_df.empty:
                    print(f"[audit]   {t}: NO DATA after retry, cache empty — skipped.")
                    continue
                cached_df = _clean(cached_df)
                STALE_CACHE_TICKERS.add(t)
                last = cached_df.index[-1].strftime("%Y-%m-%d") if len(cached_df) else "?"
                print(f"[audit]   {t}: NO DATA after retry — USING STALE CACHE "
                      f"({len(cached_df)} bars, last {last}).")
                results[t] = cached_df
                continue
            df = _clean(df)
            path = save_cache(t, df)
            print(f"[audit]   {t}: {len(df)} bars cached -> {path.relative_to(PROJECT_ROOT)}")
            results[t] = df

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class TickerAudit:
    ticker: str
    ok: bool
    role: str = ""
    n_bars: int = 0
    first_date: str | None = None
    last_date: str | None = None
    zero_volume_days: int = 0
    zero_volume_pct: float = 0.0
    max_move_pct: float = 0.0
    max_move_date: str | None = None
    split_flag: bool = False
    n_gaps_gt_7d: int = 0
    median_daily_value_rm_90d: float = 0.0
    position_pct_of_median_value: float | None = None
    liquidity_ok: bool = False
    stale_cache_used: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_ticker(ticker: str, df: pd.DataFrame | None, *, role: str = "",
                 stale_cache: bool = False) -> TickerAudit:
    """
    Run the data-quality checks for a single ticker's cached OHLCV frame.

    `stale_cache=True` marks a ticker whose live download failed and whose
    data therefore came from an out-of-date cache file — the frame is
    usable but its last bar may be days or weeks old.
    """
    if df is None or df.empty:
        return TickerAudit(ticker=ticker, ok=False, role=role,
                           stale_cache_used=stale_cache,
                           notes="no data available")

    n = len(df)
    first_date = df.index[0]
    last_date = df.index[-1]

    if "Volume" in df.columns:
        zero_volume_days = int((df["Volume"].fillna(0) == 0).sum())
    else:
        zero_volume_days = 0
    zero_volume_pct = 100.0 * zero_volume_days / n if n else 0.0

    notes: list[str] = []
    if stale_cache:
        notes.append("stale_cache_used: download failed, served from an "
                     "out-of-date cache file — last bar may be stale")

    close = df["Close"] if "Close" in df.columns else pd.Series(dtype=float)
    max_move_pct = 0.0
    max_move_date: str | None = None
    split_flag = False
    # pandas 3.x pct_change() no longer pads across NaN, so a single
    # missing close turns its neighbours into NaN returns and can hide the
    # very spike this check exists to find. Drop the gaps first and take
    # the move across the surviving consecutive closes.
    clean_close = close.dropna()
    if len(clean_close) > 1:
        daily_ret = clean_close.pct_change().abs().dropna()
        peak = float(daily_ret.max()) if not daily_ret.empty else 0.0
        if daily_ret.empty or not pd.notna(peak) or peak <= 0.0:
            # Dead-flat or unusable series: there is no "biggest move" to
            # point at, so leave the date empty instead of reporting the
            # arbitrary first index that idxmax() returns on all-zeros.
            max_move_pct = 0.0
            max_move_date = None
        else:
            max_move_pct = peak * 100.0
            idx = daily_ret.idxmax()
            max_move_date = idx.strftime("%Y-%m-%d") if pd.notna(idx) else None
        split_flag = max_move_pct > SPLIT_FLAG_THRESHOLD_PCT
        if split_flag:
            notes.append(
                f"max single-day move {max_move_pct:.1f}% on {max_move_date} "
                f"exceeds {SPLIT_FLAG_THRESHOLD_PCT:.0f}% — possible split/data error"
            )

    gaps = df.index.to_series().diff().dt.days.dropna()
    n_gaps = int((gaps > GAP_DAYS_THRESHOLD).sum())

    median_value = 0.0
    position_pct: float | None = None
    liquidity_ok = False
    if "Close" in df.columns and "Volume" in df.columns:
        tail = df.tail(90)
        daily_value = (tail["Close"] * tail["Volume"]).dropna()
        if not daily_value.empty:
            median_value = float(daily_value.median())
            if median_value > 0:
                position_pct = 100.0 * BUDGET_RM / median_value
                liquidity_ok = position_pct < LIQUIDITY_MAX_PARTICIPATION_PCT
            else:
                notes.append("median 90-bar traded value is zero — effectively untradeable")

    if n < MIN_CONTROL_BARS:
        notes.append(f"only {n} bars of history (< {MIN_CONTROL_BARS}) — short/young listing")

    return TickerAudit(
        ticker=ticker,
        ok=True,
        role=role,
        n_bars=n,
        first_date=first_date.strftime("%Y-%m-%d"),
        last_date=last_date.strftime("%Y-%m-%d"),
        zero_volume_days=zero_volume_days,
        zero_volume_pct=round(zero_volume_pct, 2),
        max_move_pct=round(max_move_pct, 2),
        max_move_date=max_move_date,
        split_flag=split_flag,
        n_gaps_gt_7d=n_gaps,
        median_daily_value_rm_90d=round(median_value, 2),
        position_pct_of_median_value=(round(position_pct, 3) if position_pct is not None else None),
        liquidity_ok=liquidity_ok,
        stale_cache_used=stale_cache,
        notes="; ".join(notes),
    )


def select_controls(
    candidate_audits: dict[str, TickerAudit],
    *,
    n: int = 6,
    min_bars: int = MIN_CONTROL_BARS,
) -> list[str]:
    """
    Pick the first `n` control candidates (in CONTROL_CANDIDATES order) with
    usable data: fetched OK, enough history, and not flagged as a probable
    split/data error. Falls back to whatever candidates are available if
    fewer than `n` pass the bar.
    """
    good = [
        t for t in CONTROL_CANDIDATES
        if t in candidate_audits
        and candidate_audits[t].ok
        and candidate_audits[t].n_bars >= min_bars
        and not candidate_audits[t].split_flag
    ]
    if len(good) < n:
        print(f"[audit] WARNING: only {len(good)} control candidates passed quality "
              f"checks (< {n} requested). Using all of them.")
        return good
    return good[:n]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(audits: dict[str, TickerAudit]) -> None:
    header = (
        f"{'Ticker':<10} {'Role':<10} {'First':<11} {'Last':<11} {'Bars':>6} "
        f"{'ZeroVol%':>9} {'MaxMove%':>9} {'Gaps>7d':>8} {'MedValRM':>12} {'Liquid':>7}"
    )
    print("[audit] " + header)
    print("[audit] " + "-" * len(header))
    for t, a in audits.items():
        if not a.ok:
            print(f"[audit] {t:<10} {a.role:<10} {'--':<11} {'--':<11} {'--':>6} "
                  f"{'--':>9} {'--':>9} {'--':>8} {'--':>12} {'NO DATA':>7}")
            continue
        liquid = "YES" if a.liquidity_ok else "no"
        print(
            f"[audit] {t:<10} {a.role:<10} {a.first_date:<11} {a.last_date:<11} "
            f"{a.n_bars:>6} {a.zero_volume_pct:>8.2f}% {a.max_move_pct:>8.2f}% "
            f"{a.n_gaps_gt_7d:>8} {a.median_daily_value_rm_90d:>12,.0f} {liquid:>7}"
        )
        if a.notes:
            print(f"[audit]     note: {a.notes}")


def run_audit(*, force: bool = False) -> dict[str, Any]:
    """Fetch, cache, validate every ticker and return the full audit report."""
    print(f"[audit] Fetching OHLCV from {START_DATE} for {len(ALL_TICKERS)} tickers "
          f"(targets + control candidates)...")
    frames = fetch_all(ALL_TICKERS, force=force)

    stale = set(STALE_CACHE_TICKERS)
    if stale:
        print(f"[audit] WARNING: served from stale cache (download failed): {sorted(stale)}")

    audits: dict[str, TickerAudit] = {}
    for t in TARGETS:
        audits[t] = audit_ticker(t, frames.get(t), role="target",
                                 stale_cache=t in stale)
    for t in CONTROL_CANDIDATES:
        audits[t] = audit_ticker(t, frames.get(t), role="candidate",
                                 stale_cache=t in stale)

    controls = select_controls(audits)
    for t in controls:
        audits[t].role = "control"
    for t in CONTROL_CANDIDATES:
        if t not in controls:
            audits[t].role = "candidate (rejected)"

    _print_table(audits)

    print(f"[audit] Selected controls ({len(controls)}): {controls}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": START_DATE,
        "targets": TARGETS,
        "control_candidates": CONTROL_CANDIDATES,
        "controls": controls,
        "budget_rm": BUDGET_RM,
        "liquidity_max_participation_pct": LIQUIDITY_MAX_PARTICIPATION_PCT,
        "split_flag_threshold_pct": SPLIT_FLAG_THRESHOLD_PCT,
        "gap_days_threshold": GAP_DAYS_THRESHOLD,
        "stale_cache_used": sorted(stale),
        "tickers": {t: a.to_dict() for t, a in audits.items()},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[audit] Report saved -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return report


if __name__ == "__main__":
    run_audit()
