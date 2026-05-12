"""
Bursa Listings Updater
======================
Auto-refreshes `bursa_listings.csv` by downloading Bursa Malaysia's official
"List of Companies" PDF and re-parsing it.

The PDF URL on Bursa's CDN is NOT stable across revisions — they re-mint a new
asset hash each time the file is republished (typically monthly). This script
handles that with a two-stage lookup:

  1. Try the last-known URL (in the cache).
  2. If that 404s or returns a stale file, scrape the parent listings page
     at https://www.bursamalaysia.com/regulation/listing/listing_resources
     to discover the current URL.

The CSV is only rewritten if the new PDF actually differs from the cached one
(SHA-256 compared), so this is safe to run on a cron/scheduler. Exit code:
  0 = up-to-date or refreshed successfully
  1 = network/parse failure (CSV unchanged)
  2 = invariants broken (e.g. < 500 issuers parsed)

Usage
-----
    python update_listings.py                  # run from anywhere
    python update_listings.py --force          # force re-download
    python update_listings.py --status         # show cache status, no fetch
    python update_listings.py --schedule       # print a sample crontab line

Requirements
------------
    pip install requests beautifulsoup4 pypdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "bursa_listings.csv"
CACHE_PATH = SCRIPT_DIR / ".bursa_cache.json"
PDF_BACKUP_DIR = SCRIPT_DIR / ".bursa_pdf_history"

# The current, observed URL (May 2026). The updater will refresh this
# automatically when Bursa rotates the asset hash.
SEED_PDF_URL = (
    "https://www.bursamalaysia.com/sites/5d809dcf39fba22790cad230/"
    "assets/66a71153e6414a8b25f23ecc/List_of_Companies.pdf"
)

# Bursa's listing-resources page where the PDF link is published.
LISTING_RESOURCES_URL = (
    "https://www.bursamalaysia.com/regulation/listing/listing_resources"
)

# Bursa's CDN rejects requests without a real browser fingerprint (403).
# These headers match what Firefox sends.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf;q=0.95,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.bursamalaysia.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

MIN_EXPECTED_ISSUERS = 500   # Sanity floor — Bursa has ~1,000 listed
HTTP_TIMEOUT = 30


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------

@dataclass
class Cache:
    """Persistent state across runs."""
    last_url: str = SEED_PDF_URL
    last_sha256: str = ""
    last_fetched_utc: str = ""
    last_issuer_count: int = 0

    @classmethod
    def load(cls) -> "Cache":
        if not CACHE_PATH.exists():
            return cls()
        try:
            data = json.loads(CACHE_PATH.read_text())
            return cls(**data)
        except Exception:
            return cls()

    def save(self) -> None:
        CACHE_PATH.write_text(json.dumps(self.__dict__, indent=2))


# ---------------------------------------------------------------------------
# URL DISCOVERY
# ---------------------------------------------------------------------------

def _discover_current_url() -> str | None:
    """
    Scrape Bursa's listing-resources page for the current List_of_Companies
    PDF URL. Returns None on failure.

    Bursa's CMS embeds asset links with patterns like:
       /sites/.../assets/<hash>/List_of_Companies[_suffix].pdf
    We grab the most recent one (lexicographic on the asset hash is a fair
    proxy since hashes are time-ordered ObjectIds).
    """
    try:
        resp = requests.get(
            LISTING_RESOURCES_URL,
            headers=BROWSER_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[discover] listing page fetch failed: {exc!s}", file=sys.stderr)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    pattern = re.compile(
        r"https://www\.bursamalaysia\.com/sites/[^/]+/assets/[^/]+/"
        r"List_of_Companies[^\"' >]*\.pdf",
        re.IGNORECASE,
    )

    found: set[str] = set()
    # Search both anchor hrefs and the raw HTML (some links are in JS payloads)
    for a in soup.find_all("a", href=True):
        m = pattern.search(a["href"])
        if m:
            found.add(m.group(0))
    for m in pattern.finditer(resp.text):
        found.add(m.group(0))

    if not found:
        print("[discover] no List_of_Companies PDF link on listing page", file=sys.stderr)
        return None

    # Pick the URL whose embedded asset-hash sorts highest (MongoDB ObjectIds
    # encode timestamp in the first 4 bytes, so newer ≈ lexicographically larger)
    def asset_hash(url: str) -> str:
        m = re.search(r"/assets/([^/]+)/", url)
        return m.group(1) if m else ""

    best = max(found, key=asset_hash)
    print(f"[discover] resolved current URL: {best}")
    return best


# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------

def _download_pdf(url: str) -> bytes | None:
    """Fetch the PDF as bytes. Returns None on failure."""
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=HTTP_TIMEOUT)
        if resp.status_code == 404:
            print("[download] 404 — URL has rotated")
            return None
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[download] failed: {exc!s}", file=sys.stderr)
        return None

    if not resp.content.startswith(b"%PDF"):
        print("[download] response is not a PDF (got HTML or error page)", file=sys.stderr)
        return None

    print(f"[download] got {len(resp.content):,} bytes from {url}")
    return resp.content


# ---------------------------------------------------------------------------
# PARSE
# ---------------------------------------------------------------------------

# Each company row in the PDF looks like:
#   NO  COMPANY NAME ...  STOCK_CODE  TEAM
ROW_RE = re.compile(
    r"^\s*(\d{1,4})\s+(.+?)\s+([0-9A-Z]{4,7})\s+(\d)\s*$"
)


def _parse_pdf(pdf_bytes: bytes) -> list[tuple[int, str, str, int]]:
    """
    Extract (no, name, stock_code, team) rows from the PDF.

    The first page is a contact-info header; data starts on page 2. We use
    pypdf so this works without poppler/pdftotext being installed.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    text_chunks = []
    for page in reader.pages:
        try:
            text_chunks.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(text_chunks)

    rows: list[tuple[int, str, str, int]] = []
    seen_codes: set[str] = set()

    for line in text.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        no, name, code, team = m.groups()
        try:
            no_i = int(no)
        except ValueError:
            continue
        if no_i > 1100:  # skip junk rows / headers
            continue
        if code in seen_codes:
            continue
        seen_codes.add(code)
        rows.append((no_i, name.strip(), code, int(team)))

    rows.sort(key=lambda r: r[0])
    return rows


def _write_csv(rows: list[tuple[int, str, str, int]]) -> None:
    """Replace bursa_listings.csv atomically."""
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "name", "stock_code", "team"])
        w.writerows(rows)
    tmp.replace(CSV_PATH)


def _backup_pdf(pdf_bytes: bytes, sha: str) -> None:
    """Archive the PDF so you have a paper trail of every version seen."""
    PDF_BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = PDF_BACKUP_DIR / f"List_of_Companies_{stamp}_{sha[:8]}.pdf"
    if not path.exists():
        path.write_bytes(pdf_bytes)


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def update(force: bool = False) -> int:
    """Main refresh routine. Returns process exit code."""
    cache = Cache.load()
    print(f"[updater] cached URL:    {cache.last_url}")
    print(f"[updater] cached sha256: {cache.last_sha256[:16] or '(none)'}")
    print(f"[updater] last refresh:  {cache.last_fetched_utc or '(never)'}")

    # Strategy: try cached URL first; if it fails, discover the new one.
    pdf_bytes = _download_pdf(cache.last_url)
    used_url = cache.last_url

    if pdf_bytes is None:
        print("[updater] cached URL failed; discovering current URL...")
        new_url = _discover_current_url()
        if not new_url:
            print("[updater] FATAL: cannot locate current PDF URL")
            return 1
        if new_url != cache.last_url:
            pdf_bytes = _download_pdf(new_url)
            used_url = new_url
        if pdf_bytes is None:
            print("[updater] FATAL: discovered URL also failed to download")
            return 1

    sha = hashlib.sha256(pdf_bytes).hexdigest()

    if not force and sha == cache.last_sha256 and CSV_PATH.exists():
        print(f"[updater] PDF unchanged (sha256={sha[:16]}...). Nothing to do.")
        # Still bump the fetched_utc so we know the check ran
        cache.last_fetched_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cache.last_url = used_url
        cache.save()
        return 0

    print(f"[updater] PDF changed (or --force). Parsing...")
    rows = _parse_pdf(pdf_bytes)
    print(f"[updater] parsed {len(rows)} rows")

    if len(rows) < MIN_EXPECTED_ISSUERS:
        print(
            f"[updater] FATAL: only {len(rows)} issuers parsed "
            f"(expected >= {MIN_EXPECTED_ISSUERS}). "
            f"PDF format may have changed. CSV NOT replaced.",
            file=sys.stderr,
        )
        return 2

    # All checks passed — commit.
    _backup_pdf(pdf_bytes, sha)
    _write_csv(rows)

    cache.last_url = used_url
    cache.last_sha256 = sha
    cache.last_fetched_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cache.last_issuer_count = len(rows)
    cache.save()

    print(f"[updater] OK — wrote {CSV_PATH.name} ({len(rows)} issuers)")
    return 0


def status() -> int:
    cache = Cache.load()
    print(f"CSV file:        {CSV_PATH}{' (exists)' if CSV_PATH.exists() else ' (MISSING)'}")
    print(f"Cached URL:      {cache.last_url}")
    print(f"Cached SHA-256:  {cache.last_sha256 or '(none)'}")
    print(f"Last fetch:      {cache.last_fetched_utc or '(never)'}")
    print(f"Issuer count:    {cache.last_issuer_count or '(unknown)'}")
    return 0


def print_schedule() -> int:
    """Print a sample crontab entry."""
    script = Path(__file__).resolve()
    py = sys.executable
    print("# Refresh Bursa listings every Monday at 06:30 local time:")
    print(f"30 6 * * 1  {py} {script} >> {script.parent}/update.log 2>&1")
    print()
    print("# Or daily (Bursa typically republishes within ~1 month of changes):")
    print(f"30 6 * * *  {py} {script} >> {script.parent}/update.log 2>&1")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Refresh bursa_listings.csv from Bursa Malaysia.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--force", action="store_true", help="Re-download even if hash matches.")
    g.add_argument("--status", action="store_true", help="Show cache info and exit.")
    g.add_argument("--schedule", action="store_true", help="Print sample cron entry and exit.")
    args = p.parse_args()

    if args.status:
        return status()
    if args.schedule:
        return print_schedule()
    return update(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
