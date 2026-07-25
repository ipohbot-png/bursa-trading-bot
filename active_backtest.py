"""
CLI entry point for the bursa_active technical backtester (DESIGN.md).

    python active_backtest.py [--force-refresh] [--top N]

Fetches/refreshes cached OHLCV via `bursa_active.data_audit.fetch_all`,
then runs the full grid + IS/OOS/controls evaluation via
`bursa_active.runner.main`. All outputs land in `active_results/`.
"""
from __future__ import annotations

import argparse

from bursa_active import data_audit, runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Re-download OHLCV even if the on-disk cache is fresh.",
    )
    parser.add_argument(
        "--top", type=int, default=runner.TOP_N_DEFAULT,
        help=f"Number of top in-sample rules to evaluate OOS "
             f"(default {runner.TOP_N_DEFAULT}).",
    )
    args = parser.parse_args()

    data_audit.fetch_all(force=args.force_refresh)
    runner.main(top_n=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
