"""
Bursa Scanner — Telegram Retry Notifier
========================================
Runs 2 hours after bursa_scanner.py. If the user hasn't replied to the
alert in Telegram, resends the original alert with a reminder prefix.

Scheduled via Windows Task Scheduler (see setup in README / task BursaScannerRetry).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

TELEGRAM_BOT_TOKEN = "8200748506:AAEksgSmlgYn-BPCI6-hok3mbzkFo43go2Q"
TELEGRAM_CHAT_ID   = "8671734227"
ALERT_STATE_FILE   = Path(__file__).parent / ".last_alert.json"


def get_last_user_reply_time() -> datetime | None:
    """Return the timestamp of the most recent message the user sent to the bot."""
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            params={"limit": 20},
            timeout=10,
        )
        updates = resp.json().get("result", [])
        for update in reversed(updates):
            msg = update.get("message", {})
            sender = msg.get("from", {})
            # Only count messages from the real user (not the bot itself)
            if not sender.get("is_bot") and str(msg.get("chat", {}).get("id")) == TELEGRAM_CHAT_ID:
                return datetime.fromtimestamp(msg["date"], tz=timezone.utc)
    except Exception as exc:
        print(f"[retry] getUpdates failed: {exc}")
    return None


def send_telegram(message: str) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=10,
        )
    except Exception as exc:
        print(f"[retry] sendMessage failed: {exc}")


def main() -> int:
    if not ALERT_STATE_FILE.exists():
        print("[retry] No alert state file found — nothing to retry.")
        return 0

    state = json.loads(ALERT_STATE_FILE.read_text())
    sent_at = datetime.fromisoformat(state["sent_at"]).astimezone(timezone.utc)
    original_message = state["message"]

    print(f"[retry] Original alert sent at {sent_at.isoformat()}")

    last_reply = get_last_user_reply_time()
    if last_reply and last_reply > sent_at:
        print(f"[retry] User replied at {last_reply.isoformat()} — no resend needed.")
        ALERT_STATE_FILE.unlink(missing_ok=True)
        return 0

    print("[retry] No reply detected — resending alert.")
    reminder = "⚠️ Reminder (no reply received):\n\n" + original_message
    send_telegram(reminder)
    # Remove state so we don't retry more than once
    ALERT_STATE_FILE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
