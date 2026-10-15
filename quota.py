"""Atomic pilot quotas for one shared SQLite file on the same host.

Reserve before contacting a provider. Accepted attempts keep their cost even
when the provider fails; slots are released in finally or after their TTL.
Counters store HMAC pseudonyms of UID and guest peer IP with a persistent salt
in the same DB. They are pseudonymous, not anonymous. No raw UID/IP is stored.
Only request.client is used: configure trusted proxy forwarding in the server,
never accept an arbitrary X-Forwarded-For header here.
"""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import ipaddress
import math
import os
from pathlib import Path
import secrets
import time
from uuid import uuid4

import aiosqlite
from fastapi import HTTPException, Request

from security import Principal


PROVIDER_TIMEOUT_SECONDS = 90


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="El control de uso no está disponible. Intenta más tarde.")


def _exceeded(kind: str, retry_after: float) -> HTTPException:
    messages = {
        "day": "Llegaste al límite diario de práctica. Vuelve a intentarlo cuando se renueve tu cuota.",
        "burst": "Hiciste varias solicitudes seguidas. Espera un momento y vuelve a intentar.",
        "slots": "Ya tienes una práctica en curso. Espera a que termine.",
    }
    return HTTPException(
        status_code=429, detail=messages[kind],
        headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
    )


class QuotaStore:
    def __init__(
        self, db_path: str | Path, *, guest_daily: int = 30, user_daily: int = 120,
        guest_burst: int = 10, user_burst: int = 30,
        guest_concurrent: int = 1, user_concurrent: int = 2,
        window_seconds: int = 60, slot_ttl_seconds: int = 180,
        clock=time.time,
    ):
        settings = (guest_daily, user_daily, guest_burst, user_burst,
                    guest_concurrent, user_concurrent, window_seconds, slot_ttl_seconds)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in settings):
            raise ValueError("Quota limits must be positive integers")
        if str(db_path) == ":memory:":
            raise ValueError("QuotaStore requires a shared SQLite file")
        self.db_path = Path(db_path)
        self.guest_daily, self.user_daily = guest_daily, user_daily
        self.guest_burst, self.user_burst = guest_burst, user_burst
        self.guest_concurrent, self.user_concurrent = guest_concurrent, user_concurrent
        self.window_seconds, self.slot_ttl_seconds = window_seconds, slot_ttl_seconds
        self.clock = clock

    async def _schema(self, db):
        await db.execute("CREATE TABLE IF NOT EXISTS quota_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        await db.execute("""CREATE TABLE IF NOT EXISTS quota_counters (
            subject TEXT NOT NULL, period TEXT NOT NULL, bucket INTEGER NOT NULL,
            used INTEGER NOT NULL CHECK (used >= 0), PRIMARY KEY (subject, period, bucket)
        )""")
        await db.execute("""CREATE TABLE IF NOT EXISTS quota_slots (
            reservation TEXT NOT NULL, subject TEXT NOT NULL, expires_at REAL NOT NULL,
            PRIMARY KEY (reservation, subject)
        )""")
        await db.execute("CREATE INDEX IF NOT EXISTS quota_slot_subject ON quota_slots(subject, expires_at)")
        await db.execute(
            "INSERT OR IGNORE INTO quota_metadata(key,value) VALUES ('salt',?)",
            (secrets.token_hex(32),),
        )
        async with db.execute("SELECT value FROM quota_metadata WHERE key='salt'") as cursor:
            row = await cursor.fetchone()
        return row[0]

    def _subjects(self, salt: str, principal: Principal, request: Request) -> list[str]:
        def pseudonym(kind, value):
            digest = hmac.new(salt.encode(), f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()
            return f"{kind}:{digest}"
        subjects = [pseudonym("uid", principal.uid)]
        if principal.is_guest:
            peer = request.client.host if request.client else "unknown-client"
            try:
                peer = str(ipaddress.ip_address(peer))
            except ValueError:
                peer = peer.lower()
            subjects.append(pseudonym("ip", peer))
        return subjects

    async def _reserve(self, principal: Principal, request: Request, cost: int) -> str:
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
            raise ValueError("Quota cost must be a positive integer")
        now = self.clock()
        if not math.isfinite(now):
            raise _unavailable()
        day = math.floor(now / 86400)
        window = math.floor(now / self.window_seconds)
        daily_limit = self.guest_daily if principal.is_guest else self.user_daily
        burst_limit = self.guest_burst if principal.is_guest else self.user_burst
        concurrent_limit = self.guest_concurrent if principal.is_guest else self.user_concurrent
        reservation = uuid4().hex
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path, timeout=5, isolation_level=None) as db:
                await db.execute("BEGIN IMMEDIATE")
                try:
                    salt = await self._schema(db)
                    subjects = self._subjects(salt, principal, request)
                    await db.execute("DELETE FROM quota_slots WHERE expires_at <= ?", (now,))
                    await db.execute("""DELETE FROM quota_counters
                        WHERE (period='day' AND bucket < ?) OR (period='burst' AND bucket < ?)""",
                        (day - 1, window - 1))
                    for subject in subjects:
                        for period, bucket, limit, bucket_seconds in (
                            ("day", day, daily_limit, 86400),
                            ("burst", window, burst_limit, self.window_seconds),
                        ):
                            async with db.execute(
                                "SELECT used FROM quota_counters WHERE subject=? AND period=? AND bucket=?",
                                (subject, period, bucket),
                            ) as cursor:
                                row = await cursor.fetchone()
                            if (row[0] if row else 0) + cost > limit:
                                raise _exceeded(period, (bucket + 1) * bucket_seconds - now)
                        async with db.execute(
                            "SELECT COUNT(*),MIN(expires_at) FROM quota_slots WHERE subject=?", (subject,)
                        ) as cursor:
                            count, first_expiry = await cursor.fetchone()
                        if count >= concurrent_limit:
                            raise _exceeded("slots", first_expiry - now)

                    # Check every UID/IP constraint before writing any charge.
                    for subject in subjects:
                        for period, bucket in (("day", day), ("burst", window)):
                            await db.execute("""INSERT INTO quota_counters(subject,period,bucket,used)
                                VALUES (?,?,?,?) ON CONFLICT(subject,period,bucket)
                                DO UPDATE SET used=used+excluded.used""", (subject, period, bucket, cost))
                        await db.execute(
                            "INSERT INTO quota_slots(reservation,subject,expires_at) VALUES (?,?,?)",
                            (reservation, subject, now + self.slot_ttl_seconds),
                        )
                    await db.commit()
                    return reservation
                except BaseException:
                    await db.rollback()
                    raise
        except (OSError, aiosqlite.Error):
            raise _unavailable() from None

    async def _release(self, reservation: str):
        try:
            async with aiosqlite.connect(self.db_path, timeout=5, isolation_level=None) as db:
                await db.execute("DELETE FROM quota_slots WHERE reservation=?", (reservation,))
        except (OSError, aiosqlite.Error):
            raise _unavailable() from None

    @asynccontextmanager
    async def consume(self, principal: Principal, request: Request, cost: int = 1):
        reservation = await self._reserve(principal, request, cost)
        try:
            yield
        finally:
            cleanup = asyncio.create_task(self._release(reservation))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Finish slot cleanup even if the request is cancelled again.
                await cleanup
                raise


_default_store = None
_default_configuration = None


def get_quota_store() -> QuotaStore:
    """Read environment config lazily; no DB is opened by importing this file."""
    global _default_store, _default_configuration
    names = {
        "guest_daily": ("QUOTA_GUEST_DAILY", 30), "user_daily": ("QUOTA_USER_DAILY", 120),
        "guest_burst": ("QUOTA_GUEST_BURST", 10), "user_burst": ("QUOTA_USER_BURST", 30),
        "guest_concurrent": ("QUOTA_GUEST_CONCURRENT", 1), "user_concurrent": ("QUOTA_USER_CONCURRENT", 2),
        "window_seconds": ("QUOTA_WINDOW_SECONDS", 60), "slot_ttl_seconds": ("QUOTA_SLOT_TTL_SECONDS", 180),
    }
    try:
        settings = {field: int(os.getenv(name, str(default))) for field, (name, default) in names.items()}
        timeout = int(os.getenv("PROVIDER_TIMEOUT_SECONDS", str(PROVIDER_TIMEOUT_SECONDS)))
        if not 1 <= timeout <= 120:
            raise ValueError("Provider timeout must be between 1 and 120 seconds")
        if settings["slot_ttl_seconds"] < timeout + 30:
            raise ValueError("Quota slot TTL must exceed the provider timeout")
        path = os.getenv("QUOTA_DB_PATH", "data/access_quota.db")
        if not path.strip():
            raise ValueError("Missing quota database path")
        configuration = (path, *settings.values())
        if _default_store is None or configuration != _default_configuration:
            _default_store = QuotaStore(path, **settings)
            _default_configuration = configuration
        return _default_store
    except (TypeError, ValueError):
        raise _unavailable() from None


@asynccontextmanager
async def consume(principal: Principal, request: Request, cost: int = 1):
    async with get_quota_store().consume(principal, request, cost):
        yield
