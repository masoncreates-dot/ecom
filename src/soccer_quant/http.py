"""HTTP with an on-disk TTL cache, retries and a per-day request budget.

The cache is what keeps API-Football inside its daily quota: squads and
player stats are refetched every couple of days, injuries every few hours,
odds every half hour, lineups every few minutes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

FOREVER = None  # ttl for immutable data (finished seasons, archived weather)


class BudgetExceeded(RuntimeError):
    pass


class RequestBudget:
    """Counts requests per UTC day for one provider, persisted to disk."""

    def __init__(self, path: Path, daily_limit: int):
        self.path = path
        self.daily_limit = daily_limit

    def _state(self) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state.get("day") == today:
                return state
        return {"day": today, "used": 0, "remote_remaining": None}

    @property
    def remaining(self) -> int:
        state = self._state()
        local = self.daily_limit - state["used"]
        remote = state.get("remote_remaining")
        return local if remote is None else min(local, remote)

    def consume(self, remote_remaining: int | None = None) -> None:
        state = self._state()
        state["used"] += 1
        if remote_remaining is not None:
            state["remote_remaining"] = remote_remaining
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state))

    def check(self) -> None:
        if self.remaining <= 0:
            raise BudgetExceeded(f"daily request budget of {self.daily_limit} used up ({self.path.name})")


class HttpClient:
    def __init__(self, cache_dir: Path, *, session: requests.Session | None = None,
                 timeout: float = 30.0, retries: int = 3, backoff: float = 2.0):
        self.cache_dir = cache_dir
        self.session = session or requests.Session()
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def _cache_path(self, url: str, params: dict | None) -> Path:
        key = json.dumps([url, sorted((params or {}).items())], default=str)
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"

    def cached(self, url: str, params: dict | None, ttl: float | None) -> Any | None:
        path = self._cache_path(url, params)
        if not path.exists():
            return None
        entry = json.loads(path.read_text())
        if ttl is not None and time.time() - entry["fetched_at"] > ttl:
            return None
        return entry["body"]

    def get(self, url: str, *, params: dict | None = None, headers: dict | None = None,
            ttl: float | None = 3600, as_json: bool = True, budget: RequestBudget | None = None,
            validate: Callable[[Any], None] | None = None) -> Any:
        """Cached GET. ``validate`` may raise to keep an error payload out of the cache."""
        hit = self.cached(url, params, ttl)
        if hit is not None:
            return hit
        if budget is not None:
            budget.check()
        body = self._fetch(url, params, headers, as_json, budget)
        if validate is not None:
            validate(body)
        path = self._cache_path(url, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": time.time(), "url": url, "params": params, "body": body}))
        return body

    def _fetch(self, url, params, headers, as_json, budget):
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if budget is not None:
                    remaining = resp.headers.get("x-ratelimit-requests-remaining")
                    budget.consume(int(remaining) if remaining and remaining.isdigit() else None)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} from {url}", response=resp)
                resp.raise_for_status()
                return resp.json() if as_json else _decode(resp.content)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last_error = exc
                if attempt < self.retries:
                    wait = self.backoff * 2**attempt
                    log.info("retrying %s in %.0fs (%s)", url, wait, exc)
                    time.sleep(wait)
        raise last_error  # type: ignore[misc]


def _decode(content: bytes) -> str:
    """CSV feeds often omit the charset; honour a UTF-8 BOM and fall back to Latin-1."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")
