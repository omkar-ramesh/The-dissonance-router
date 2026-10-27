"""Public demo guard: a per-IP rate limit and a daily spend cut-off.

Only active when PUBLIC_DEMO=1. In-memory, so it resets when the server restarts; that is
enough for a single-instance hackathon demo, not for a production service.
"""
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock

from fastapi import HTTPException, Request

from src import config

MIN_PRIVATE_ID = 12     # session ids shorter than this are guessable, so refused in public mode


class Guard:
    def __init__(self, per_minute: int, daily_budget_usd: float):
        self.per_minute, self.daily_budget = per_minute, daily_budget_usd
        self._hits: dict[str, deque] = defaultdict(deque)
        self._day, self._spent = "", 0.0
        self._lock = Lock()

    def admit(self, ip: str) -> None:
        """Raise 429 if this IP is over its rate or today's budget is used up."""
        now = time.monotonic()
        with self._lock:
            self._roll_day()
            if self._spent >= self.daily_budget:
                raise HTTPException(429, "Today's demo budget is used up. It resets at 00:00 UTC.")
            hits = self._hits[ip]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                raise HTTPException(429, f"Too many messages: the demo allows {self.per_minute} per minute.")
            hits.append(now)

    def spend(self, usd: float) -> None:
        with self._lock:
            self._roll_day()
            self._spent += usd or 0.0

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._day:
            self._day, self._spent = today, 0.0


GUARD = Guard(config.RATE_LIMIT_PER_MIN, config.DAILY_BUDGET_USD)


def client_ip(request: Request) -> str:
    # Behind a hosting proxy the first X-Forwarded-For entry is the visitor. It can be
    # spoofed, which is acceptable for a demo rate limit.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def admit(request: Request) -> None:
    """FastAPI dependency for endpoints that call a model."""
    if config.PUBLIC_DEMO:
        GUARD.admit(client_ip(request))


def require_private_id(user_id: str) -> None:
    if config.PUBLIC_DEMO and len(user_id) < MIN_PRIVATE_ID:
        raise HTTPException(422, f"In the public demo, session ids must be at least {MIN_PRIVATE_ID} characters.")


def cost_of(result: dict) -> float:
    """Estimated spend of one answer, including the signal reader's calls."""
    calls = [result.get("stats") or {}, *(result.get("reader_stats") or [])]
    return sum(c.get("est_cost_usd") or 0.0 for c in calls)
