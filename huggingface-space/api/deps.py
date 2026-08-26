from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict

from fastapi import Header

from core.config import AppConfig
from core.errors import RateLimitError, UnauthorizedError


@dataclass
class RateLimiter:
    limit_per_minute: int
    _hits: Dict[str, Deque[float]]
    _lock: threading.Lock

    def __init__(self, limit_per_minute: int):
        self.limit_per_minute = limit_per_minute
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> None:
        now = time.time()
        cutoff = now - 60.0
        with self._lock:
            q = self._hits[identity]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.limit_per_minute:
                raise RateLimitError()
            q.append(now)


def require_api_key(config: AppConfig, x_api_key: str | None = Header(default=None)) -> None:
    if not config.api_key_enabled:
        return
    if not config.api_key:
        raise UnauthorizedError("API key security is enabled but API_KEY is empty")
    if x_api_key != config.api_key:
        raise UnauthorizedError()


def enforce_rate_limit(config: AppConfig, limiter: RateLimiter, identity: str) -> None:
    if not config.rate_limit_enabled:
        return
    limiter.check(identity)

