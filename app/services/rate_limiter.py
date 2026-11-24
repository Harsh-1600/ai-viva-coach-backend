"""In-memory sliding-window rate limiter for high-cost AI generation and evaluation endpoints."""

import time
import threading
from typing import Dict, List
from fastapi import Request, HTTPException, status


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter tracking request timestamps per client IP / identifier.
    Thread-safe with re-entrant lock.
    """
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = [now]
                return True

            # Prune timestamps older than window
            cutoff = now - self.window_seconds
            valid_timestamps = [t for t in self.clients[client_id] if t > cutoff]
            self.clients[client_id] = valid_timestamps

            if len(valid_timestamps) < self.max_requests:
                self.clients[client_id].append(now)
                return True

            return False


# Global rate limiter instances:
# 1. AI rate limiter: 300 requests per 60 seconds per client
ai_rate_limiter = SlidingWindowRateLimiter(max_requests=300, window_seconds=60)
# 2. Auth rate limiter: 100 login/register attempts per 60 seconds per IP
auth_rate_limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)


def check_ai_rate_limit(request: Request):
    """Dependency to enforce rate limits on expensive AI endpoints."""
    client_ip = request.client.host if request.client else "unknown_client"
    # Also check forwarded IP if behind proxy
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    if not ai_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for AI operations. Please wait before submitting additional prompts.",
            headers={"Retry-After": "60"}
        )
