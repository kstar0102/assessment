"""
Token aware sliding window rate limiter backed by an on disk SQLite file.

Model
  Every admitted request writes a row (tenant, ts, tokens). A tenant's usage
  is SUM(tokens) over rows with ts inside the last WINDOW seconds. That's a
  true sliding log, not fixed buckets, so there's no burst at window edges.

Flow per request
  reserve()  estimate tokens up front (prompt + max_tokens) and atomically
             check + insert inside BEGIN IMMEDIATE. That write lock makes the
             check and the insert one step, so N concurrent requests can't all
             see the same headroom and overshoot. Works across processes too.
  commit()   once the provider answers, swap the estimate for real usage.
  release()  if no provider served the request, give the tokens back.

Eviction
  Expired rows are deleted on every reserve (indexed on ts, so it's cheap),
  which keeps the table at roughly one window of traffic.

SQLite calls are blocking, so they run in a worker thread via
asyncio.to_thread and never stall the event loop.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant  TEXT    NOT NULL,
    ts      REAL    NOT NULL,
    tokens  INTEGER NOT NULL CHECK (tokens >= 0)
);
CREATE INDEX IF NOT EXISTS ix_usage_tenant_ts ON token_usage (tenant, ts);
CREATE INDEX IF NOT EXISTS ix_usage_ts ON token_usage (ts);
"""


class RateLimited(Exception):
    def __init__(self, retry_after: float, used: int, limit: int):
        super().__init__("rate limited")
        self.retry_after = retry_after
        self.used = used
        self.limit = limit


class RequestTooLarge(Exception):
    pass


@dataclass(frozen=True)
class Reservation:
    id: int
    tenant: str
    tokens: int
    remaining: int


class SlidingWindowLimiter:
    def __init__(self, db_path: str, limit_tokens: int = 50_000, window_seconds: float = 60.0, clock=time.time):
        self.db_path = db_path
        self.limit = limit_tokens
        self.window = window_seconds
        self.clock = clock
        self._local = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = self._conn()
        conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
        return conn

    # ---- sync core (runs in a worker thread) ------------------------------

    def _reserve_sync(self, tenant: str, tokens: int, now: float) -> Reservation:
        if tokens > self.limit:
            raise RequestTooLarge()
        conn = self._conn()
        cutoff = now - self.window
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM token_usage WHERE ts <= ?", (cutoff,))
            used = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?",
                (tenant, cutoff),
            ).fetchone()[0]

            if used + tokens > self.limit:
                retry_after = self._retry_after(conn, tenant, cutoff, used, tokens, now)
                conn.execute("COMMIT")  # keep the eviction
                raise RateLimited(retry_after, used, self.limit)

            cur = conn.execute("INSERT INTO token_usage (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens))
            conn.execute("COMMIT")
            return Reservation(cur.lastrowid, tenant, tokens, self.limit - used - tokens)
        except RateLimited:
            raise
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def _retry_after(self, conn, tenant, cutoff, used, tokens, now) -> float:
        """Seconds until enough old rows slide out of the window to fit `tokens`."""
        need = used + tokens - self.limit
        freed = 0
        for ts, t in conn.execute(
            "SELECT ts, tokens FROM token_usage WHERE tenant = ? AND ts > ? ORDER BY ts ASC", (tenant, cutoff)
        ):
            freed += t
            if freed >= need:
                return max(0.0, ts + self.window - now)
        return self.window

    def _commit_sync(self, res_id: int, actual_tokens: int) -> None:
        self._conn().execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, int(actual_tokens)), res_id))

    def _release_sync(self, res_id: int) -> None:
        self._conn().execute("DELETE FROM token_usage WHERE id = ?", (res_id,))

    def _usage_sync(self, tenant: str, now: float) -> int:
        return self._conn().execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?", (tenant, now - self.window)
        ).fetchone()[0]

    # ---- async API -------------------------------------------------------

    async def reserve(self, tenant: str, tokens: int, now: float | None = None) -> Reservation:
        return await asyncio.to_thread(self._reserve_sync, tenant, tokens, self.clock() if now is None else now)

    async def commit(self, reservation: Reservation, actual_tokens: int) -> None:
        await asyncio.to_thread(self._commit_sync, reservation.id, actual_tokens)

    async def release(self, reservation: Reservation) -> None:
        await asyncio.to_thread(self._release_sync, reservation.id)

    async def usage(self, tenant: str, now: float | None = None) -> int:
        return await asyncio.to_thread(self._usage_sync, tenant, self.clock() if now is None else now)


def estimate_tokens(body: dict, default_max_tokens: int = 1024) -> int:
    """Cheap upper bound: ~4 chars per token for the prompt plus requested completion."""
    chars = 0
    for m in body.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):  # content parts
            chars += sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
        chars += 16  # role + framing overhead
    prompt = (chars + 3) // 4
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens
    return prompt + int(max_tokens)
