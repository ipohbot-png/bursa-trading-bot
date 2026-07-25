"""
bursa_active — single-stock technical rule research for Bursa Malaysia.

Signal-only. Nothing in this package ever places an order; it evaluates
entry/exit rules on daily OHLCV, nets them through the Maybank fee
schedule (`bursa_momentum.fees`) and reports honest performance.

See DESIGN.md for the authoritative spec.
"""
from __future__ import annotations

__all__ = ["indicators", "engine", "metrics"]
