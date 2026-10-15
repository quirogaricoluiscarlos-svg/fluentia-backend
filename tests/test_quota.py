"""Quota transactions use only temporary DBs, fake clocks and test identities."""

import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import aiosqlite
from fastapi import HTTPException
from starlette.requests import Request

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1]), *sys.path]):
    import quota
    from security import Principal


def request(ip="203.0.113.4", forwarded=None):
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "method": "POST", "path": "/", "headers": headers, "client": (ip, 1234)})


class Clock:
    def __init__(self):
        self.now = 1728000000.0  # UTC day and 60-second window boundary.

    def __call__(self):
        return self.now


class QuotaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "quota.sqlite"
        self.clock = Clock()
        self.guest = Principal("anonymous-test-1", True)
        self.user = Principal("registered-test-1", False)
        self.req = request()
        self.store = self.make_store()

    def make_store(self, **overrides):
        settings = dict(guest_daily=3, user_daily=6, guest_burst=3, user_burst=6,
                        guest_concurrent=1, user_concurrent=2, slot_ttl_seconds=5, clock=self.clock)
        settings.update(overrides)
        return quota.QuotaStore(self.path, **settings)

    async def spend(self, store=None, principal=None, req=None, cost=1):
        async with (store or self.store).consume(principal or self.guest, req or self.req, cost):
            pass

    async def rejected(self, store=None, principal=None, req=None, cost=1):
        with self.assertRaises(HTTPException) as error:
            await self.spend(store, principal, req, cost)
        self.assertEqual(error.exception.status_code, 429)
        self.assertGreaterEqual(int(error.exception.headers["Retry-After"]), 1)
        return error.exception

    async def rows(self, sql, parameters=()):
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(sql, parameters) as cursor:
                return await cursor.fetchall()

    async def test_pilot_defaults_are_distinct_for_guest_and_registered_user(self):
        defaults = quota.QuotaStore(self.path)
        self.assertEqual((defaults.guest_daily, defaults.user_daily), (30, 120))
        self.assertEqual((defaults.guest_burst, defaults.user_burst), (10, 30))
        self.assertEqual((defaults.guest_concurrent, defaults.user_concurrent), (1, 2))
        self.assertFalse(self.path.exists())

    async def test_daily_cost_is_weighted_and_resets_at_utc_day(self):
        store = self.make_store(guest_burst=10)
        await self.spend(store, cost=2)
        await self.spend(store)
        error = await self.rejected(store)
        self.assertEqual(error.headers["Retry-After"], "86400")
        self.clock.now += 86400
        await self.spend(store)

    async def test_registered_user_has_its_own_larger_daily_budget(self):
        store = self.make_store(guest_burst=100, user_burst=100)
        for _ in range(6):
            await self.spend(store, self.user)
        await self.rejected(store, self.user)

    async def test_burst_counter_blocks_before_daily_quota_and_resets(self):
        store = self.make_store(guest_daily=30, guest_burst=2)
        await self.spend(store, cost=2)
        error = await self.rejected(store)
        self.assertEqual(error.headers["Retry-After"], "60")
        self.clock.now += 60
        await self.spend(store, cost=2)

    async def test_guest_uid_rotation_cannot_bypass_shared_ip_daily_quota(self):
        store = self.make_store(guest_daily=2, guest_burst=100)
        await self.spend(store, Principal("guest-one", True))
        await self.spend(store, Principal("guest-two", True))
        await self.rejected(store, Principal("guest-three", True))
        await self.spend(store, Principal("guest-three", True), request("203.0.113.5"))

    async def test_guest_ip_rotation_cannot_bypass_uid_quota(self):
        store = self.make_store(guest_daily=1, guest_burst=100)
        await self.spend(store)
        await self.rejected(store, req=request("203.0.113.5"))

    async def test_guest_burst_is_shared_by_ip_and_untrusted_forwarded_header_is_ignored(self):
        store = self.make_store(guest_daily=30, guest_burst=2)
        await self.spend(store, Principal("guest-one", True), request(forwarded="198.51.100.1"))
        await self.spend(store, Principal("guest-two", True), request(forwarded="198.51.100.2"))
        await self.rejected(store, Principal("guest-three", True), request(forwarded="198.51.100.3"))

    async def test_guest_simultaneous_slot_is_shared_by_ip(self):
        store = self.make_store(guest_daily=30, guest_burst=30)
        async with store.consume(self.guest, self.req):
            error = await self.rejected(store, Principal("guest-two", True))
            self.assertEqual(error.headers["Retry-After"], "5")
            await self.spend(store, Principal("guest-two", True), request("203.0.113.5"))
        await self.spend(store, Principal("guest-two", True))

    async def test_registered_user_can_have_two_slots_and_other_users_do_not_share_ip_quota(self):
        store = self.make_store(user_daily=30, user_burst=30)
        async with store.consume(self.user, self.req):
            async with store.consume(self.user, self.req):
                await self.rejected(store, self.user)
                await self.spend(store, Principal("registered-test-2", False))
        self.assertEqual(await self.rows("SELECT COUNT(*) FROM quota_slots"), [(0,)])

    async def test_separate_store_instances_cannot_overshoot_shared_budget_under_concurrency(self):
        first = self.make_store(guest_daily=5, guest_burst=100, guest_concurrent=30)
        second = self.make_store(guest_daily=5, guest_burst=100, guest_concurrent=30)

        async def attempt(index):
            try:
                async with (first if index % 2 else second).consume(self.guest, self.req):
                    await asyncio.sleep(0)
                return True
            except HTTPException as error:
                self.assertEqual(error.status_code, 429)
                return False

        outcomes = await asyncio.gather(*(attempt(index) for index in range(25)))
        self.assertEqual(sum(outcomes), 5)
        self.assertEqual(await self.rows("SELECT used FROM quota_counters WHERE period='day' ORDER BY subject"), [(5,), (5,)])
        self.assertEqual(await self.rows("SELECT COUNT(*) FROM quota_slots"), [(0,)])

    async def test_provider_exception_releases_slot_but_does_not_refund_cost(self):
        store = self.make_store(guest_daily=1, guest_burst=100)
        with self.assertRaises(RuntimeError):
            async with store.consume(self.guest, self.req):
                raise RuntimeError("fake provider failure")
        self.assertEqual(await self.rows("SELECT COUNT(*) FROM quota_slots"), [(0,)])
        await self.rejected(store)

    async def test_cancellation_releases_slot_and_retry_still_costs_a_unit(self):
        entered = asyncio.Event()
        never = asyncio.Event()

        async def worker():
            async with self.store.consume(self.guest, self.req):
                entered.set()
                await never.wait()

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(await self.rows("SELECT COUNT(*) FROM quota_slots"), [(0,)])
        await self.spend()
        self.assertEqual(await self.rows("SELECT used FROM quota_counters WHERE period='day' ORDER BY subject"), [(2,), (2,)])

    async def test_expired_slot_from_dead_process_is_cleaned_without_refund(self):
        await self.store._reserve(self.guest, self.req, 1)  # Process died before finally.
        await self.rejected()
        self.clock.now += 6
        await self.spend()
        self.assertEqual(await self.rows("SELECT COUNT(*) FROM quota_slots"), [(0,)])
        self.assertEqual(await self.rows("SELECT used FROM quota_counters WHERE period='day' ORDER BY subject"), [(2,), (2,)])

    async def test_counter_pseudonyms_and_salt_survive_store_restart(self):
        await self.spend()
        salt_before = await self.rows("SELECT value FROM quota_metadata WHERE key='salt'")
        subjects_before = await self.rows("SELECT DISTINCT subject FROM quota_counters ORDER BY subject")
        await self.spend(self.make_store())
        self.assertEqual(salt_before, await self.rows("SELECT value FROM quota_metadata WHERE key='salt'"))
        self.assertEqual(subjects_before, await self.rows("SELECT DISTINCT subject FROM quota_counters ORDER BY subject"))
        raw = self.path.read_bytes()
        self.assertNotIn(self.guest.uid.encode(), raw)
        self.assertNotIn(b"203.0.113.4", raw)

    async def test_stale_counters_are_cleaned(self):
        await self.spend()
        self.clock.now += 3 * 86400
        await self.spend()
        self.assertEqual(await self.rows("SELECT COUNT(*) FROM quota_counters"), [(4,)])

    async def test_invalid_cost_or_limits_never_open_a_database(self):
        for cost in [0, -1, True, 1.5, "1"]:
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                await self.spend(cost=cost)
        for settings in [{"guest_daily": 0}, {"user_burst": True}, {"slot_ttl_seconds": -1}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.make_store(**settings)
        with self.assertRaises(ValueError):
            quota.QuotaStore(":memory:")
        self.assertFalse(self.path.exists())

    async def test_storage_failure_returns_503_before_provider(self):
        blocked = Path(self.directory.name) / "blocked"
        blocked.write_text("not a directory")
        store = quota.QuotaStore(blocked / "quota.sqlite")
        provider_called = False
        with self.assertRaises(HTTPException) as error:
            async with store.consume(self.guest, self.req):
                provider_called = True
        self.assertEqual(error.exception.status_code, 503)
        self.assertFalse(provider_called)

    async def test_lazy_env_configuration_enforces_timeout_buffer_without_opening_db(self):
        with patch.dict(os.environ, {"QUOTA_DB_PATH": str(self.path)}, clear=True):
            store = quota.get_quota_store()
            self.assertEqual(store.slot_ttl_seconds, 180)
            self.assertEqual(store.db_path, self.path)
        for config in [
            {"QUOTA_SLOT_TTL_SECONDS": "119"}, {"QUOTA_GUEST_DAILY": "0"},
            {"QUOTA_USER_BURST": "invalid"}, {"QUOTA_DB_PATH": " "},
        ]:
            with self.subTest(config=config), patch.dict(os.environ, config, clear=True):
                with self.assertRaises(HTTPException) as error:
                    quota.get_quota_store()
                self.assertEqual(error.exception.status_code, 503)
        self.assertFalse(self.path.exists())

    async def test_module_consume_uses_injected_store_without_real_db(self):
        with patch.object(quota, "get_quota_store", return_value=self.store):
            async with quota.consume(self.guest, self.req):
                pass
        self.assertEqual(await self.rows("SELECT COUNT(*) FROM quota_slots"), [(0,)])

    async def test_env_timeout_120_with_ttl_120_fails_closed(self):
        config = {"QUOTA_DB_PATH": str(self.path), "PROVIDER_TIMEOUT_SECONDS": "120", "QUOTA_SLOT_TTL_SECONDS": "120"}
        with patch.dict(os.environ, config, clear=True):
            with self.assertRaises(HTTPException) as error:
                quota.get_quota_store()
        self.assertEqual(error.exception.status_code, 503)
        self.assertFalse(self.path.exists())

    async def test_env_timeout_120_with_ttl_180_is_allowed(self):
        config = {"QUOTA_DB_PATH": str(self.path), "PROVIDER_TIMEOUT_SECONDS": "120", "QUOTA_SLOT_TTL_SECONDS": "180"}
        with patch.dict(os.environ, config, clear=True):
            self.assertEqual(quota.get_quota_store().slot_ttl_seconds, 180)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
