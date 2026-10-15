"""Session ownership/lifecycle tests with no provider, dotenv or DB access."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


BACKEND = Path(__file__).resolve().parents[1]
SETTINGS = {
    "SCENARIO_SESSION_TTL_SECONDS": "1800",
    "SCENARIO_MAX_TURNS": "20",
    "SCENARIO_MAX_SESSIONS_PER_OWNER": "3",
    "SCENARIO_MAX_SESSIONS_TOTAL": "1000",
    "SCENARIO_HISTORY_TURNS": "10",
}


def completion(data=None):
    if data is None:
        data = {
            "ai_response": "Hello! How are you today?",
            "correction": None,
            "explanation": "Tu oración es correcta.",
            "objectives_met": ["Saludar"],
        }
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(data))
    )])


def load_scenarios(settings=None):
    services = ModuleType("services")
    services.transcribe_audio = AsyncMock(return_value="Hello!")
    create = AsyncMock(return_value=completion())
    services.deepseek_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    quality_log = ModuleType("quality_log")
    quality_log.log_ai_call = AsyncMock()
    name = "fluentia_scenario_sessions_test"
    spec = importlib.util.spec_from_file_location(name, BACKEND / "scenarios.py")
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, {
            "services": services, "quality_log": quality_log, name: module,
        }),
        patch.dict(os.environ, {**SETTINGS, **(settings or {})}),
        patch.object(sys, "path", [str(BACKEND), *sys.path]),
    ):
        spec.loader.exec_module(module)
    return module, services.transcribe_audio, create


class ScenarioSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scenarios, self.transcribe, self.create = load_scenarios()

    def start(self, owner="alice"):
        result = self.scenarios.start_scenario("smalltalk", owner_id=owner)
        self.assertIsInstance(result, tuple)
        return result[0]

    async def turn(self, session_id, owner="alice"):
        return await self.scenarios.process_turn(
            session_id, b"fake wav", "turn.wav", owner_id=owner,
        )

    def assert_pristine(self, session_id):
        session = self.scenarios._sessions[session_id]
        self.assertEqual(session.turn_count, 0)
        self.assertEqual(session.total_score, 0)
        self.assertEqual(session.corrections_count, 0)
        self.assertEqual(session.objectives_met, [])
        self.assertEqual(len(session.messages), 1)
        self.assertFalse(session.turn_in_progress)

    def test_start_keeps_android_payload_and_records_owner(self):
        session_id, data = self.scenarios.start_scenario("cafe", owner_id="alice")
        session = self.scenarios._sessions[session_id]
        self.assertEqual(session.owner_id, "alice")
        self.assertEqual(data["session_id"], session_id)
        self.assertEqual(data["scenario"]["id"], "cafe")
        self.assertEqual(data["turn_count"], 0)
        self.assertEqual(data["objectives_met"], [])
        self.assertEqual(data["ai_message"], session.messages[0]["content"])
        self.transcribe.assert_not_awaited()
        self.create.assert_not_awaited()

    def test_owner_is_required_and_invalid_scenario_creates_no_session(self):
        with self.assertRaises(TypeError):
            self.scenarios.start_scenario("smalltalk")
        for owner in (None, "", "   "):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                self.scenarios.start_scenario("smalltalk", owner_id=owner)
        self.assertIsNone(self.scenarios.start_scenario("missing", owner_id="alice"))
        self.assertEqual(self.scenarios._sessions, {})

    async def test_foreign_or_unknown_session_never_reaches_providers(self):
        session_id = self.start()
        for requested_id in (session_id, "missing"):
            self.assertIsNone(await self.turn(requested_id, owner="bob"))
            self.assertIsNone(self.scenarios.end_scenario(requested_id, owner_id="bob"))
        self.assertIn(session_id, self.scenarios._sessions)
        self.assert_pristine(session_id)
        self.transcribe.assert_not_awaited()
        self.create.assert_not_awaited()

    async def test_legacy_unowned_session_cannot_be_accessed(self):
        session = self.scenarios.ConversationSession("legacy", "smalltalk")
        self.scenarios._sessions[session.session_id] = session
        for owner in ("alice", None, ""):
            with self.subTest(owner=owner):
                self.assertIsNone(await self.turn("legacy", owner))
                self.assertIsNone(self.scenarios.end_scenario("legacy", owner_id=owner))
        self.transcribe.assert_not_awaited()
        self.create.assert_not_awaited()

    async def test_idle_session_expires_at_ttl_before_provider_calls(self):
        with patch.object(self.scenarios, "_session_now", return_value=100):
            session_id = self.start()
        with patch.object(self.scenarios, "_session_now", return_value=1900):
            self.assertIsNone(await self.turn(session_id))
            self.assertIsNone(self.scenarios.end_scenario(session_id, owner_id="alice"))
        self.assertNotIn(session_id, self.scenarios._sessions)
        self.transcribe.assert_not_awaited()
        self.create.assert_not_awaited()

    async def test_successful_turn_renews_idle_ttl(self):
        with patch.object(self.scenarios, "_session_now", return_value=100):
            session_id = self.start()
        with patch.object(self.scenarios, "_session_now", return_value=1800):
            self.assertEqual((await self.turn(session_id))["turn_count"], 1)
        self.assertEqual(self.scenarios._sessions[session_id].last_activity_at, 1800)
        with patch.object(self.scenarios, "_session_now", return_value=2000):
            self.assertIsInstance(
                self.scenarios.end_scenario(session_id, owner_id="alice"), dict,
            )

    def test_owner_and_total_session_limits_return_429(self):
        self.scenarios.SCENARIO_MAX_SESSIONS_PER_OWNER = 2
        self.scenarios.SCENARIO_MAX_SESSIONS_TOTAL = 3
        self.start()
        self.start()
        self.assertEqual(
            self.scenarios.start_scenario("smalltalk", owner_id="alice")["status_code"], 429,
        )
        self.start(owner="bob")
        self.assertEqual(
            self.scenarios.start_scenario("smalltalk", owner_id="carol")["status_code"], 429,
        )
        self.assertEqual(len(self.scenarios._sessions), 3)

    def test_start_capacity_check_is_atomic_between_threads(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda _: self.scenarios.start_scenario("smalltalk", owner_id="alice"),
                range(20),
            ))
        self.assertEqual(sum(isinstance(result, tuple) for result in results), 3)
        self.assertEqual(sum(isinstance(result, dict) for result in results), 17)
        self.assertEqual(len(self.scenarios._sessions), 3)

    def test_expired_sessions_are_purged_before_capacity_checks(self):
        self.scenarios.SCENARIO_MAX_SESSIONS_PER_OWNER = 1
        with patch.object(self.scenarios, "_session_now", return_value=100):
            old_session_id = self.start()
        with patch.object(self.scenarios, "_session_now", return_value=1900):
            new_session_id = self.start()
        self.assertNotEqual(new_session_id, old_session_id)
        self.assertNotIn(old_session_id, self.scenarios._sessions)

    async def test_concurrent_turn_and_end_are_rejected_without_provider_call(self):
        session_id = self.start()
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_transcription(*_):
            entered.set()
            await release.wait()
            return "Hello!"

        self.transcribe.side_effect = slow_transcription
        first = asyncio.create_task(self.turn(session_id))
        await entered.wait()
        self.assertEqual((await self.turn(session_id))["status_code"], 409)
        self.assertEqual(
            self.scenarios.end_scenario(session_id, owner_id="alice")["status_code"], 409,
        )
        self.assertIsNone(await self.turn(session_id, owner="bob"))
        self.transcribe.assert_awaited_once()
        self.create.assert_not_awaited()
        release.set()
        self.assertEqual((await first)["turn_count"], 1)
        self.assertEqual(
            self.scenarios.end_scenario(session_id, owner_id="alice")["total_turns"], 1,
        )

    async def test_active_turn_survives_ttl_and_still_counts_against_capacity(self):
        self.scenarios.SCENARIO_MAX_SESSIONS_PER_OWNER = 1
        session_id = self.start()
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_provider(**_):
            entered.set()
            await release.wait()
            return completion()

        self.create.side_effect = slow_provider
        task = asyncio.create_task(self.turn(session_id))
        await entered.wait()
        session = self.scenarios._sessions[session_id]
        with patch.object(
            self.scenarios, "_session_now",
            return_value=session.last_activity_at + self.scenarios.SCENARIO_SESSION_TTL_SECONDS + 1,
        ):
            result = self.scenarios.start_scenario("smalltalk", owner_id="alice")
            self.assertEqual(result["status_code"], 429)
            self.assertIn(session_id, self.scenarios._sessions)
            self.assertEqual((await self.turn(session_id))["status_code"], 409)
            release.set()
            self.assertEqual((await task)["turn_count"], 1)
            self.assertFalse(session.turn_in_progress)
            self.assertIsInstance(
                self.scenarios.end_scenario(session_id, owner_id="alice"), dict,
            )

    async def test_cancellation_during_transcription_or_provider_releases_guard(self):
        for dependency in (self.transcribe, self.create):
            with self.subTest(dependency=dependency):
                session_id = self.start()
                entered = asyncio.Event()

                async def blocked(*_, **__):
                    entered.set()
                    await asyncio.Future()

                dependency.side_effect = blocked
                task = asyncio.create_task(self.turn(session_id))
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assert_pristine(session_id)
                dependency.side_effect = None
                self.assertEqual((await self.turn(session_id))["turn_count"], 1)
                self.scenarios.end_scenario(session_id, owner_id="alice")

    async def test_transcription_failure_leaves_no_committed_turn_and_can_retry(self):
        session_id = self.start()
        self.transcribe.side_effect = RuntimeError("provider detail must stay private")
        result = await self.turn(session_id)
        self.assertEqual(result["status_code"], 503)
        self.assertNotIn("provider detail", result["error"])
        self.create.assert_not_awaited()
        self.assert_pristine(session_id)
        self.transcribe.side_effect = None
        self.assertEqual((await self.turn(session_id))["turn_count"], 1)

    async def test_silence_or_invalid_transcription_does_not_count_as_turn(self):
        session_id = self.start()
        for transcription in ("   ", None, 12):
            with self.subTest(transcription=transcription):
                self.transcribe.return_value = transcription
                result = await self.turn(session_id)
                self.assertIn("error", result)
                self.assertEqual(result["turn_count"], 0)
                self.assert_pristine(session_id)
        self.create.assert_not_awaited()

    async def test_provider_failure_retries_and_rolls_back_history(self):
        session_id = self.start()
        self.create.side_effect = RuntimeError("provider unavailable")
        self.assertEqual((await self.turn(session_id))["status_code"], 503)
        self.assertEqual(self.create.await_count, 2)
        self.assert_pristine(session_id)
        self.create.side_effect = None
        result = await self.turn(session_id)
        self.assertEqual(result["turn_count"], 1)
        self.assertEqual(len(self.scenarios._sessions[session_id].messages), 3)

    async def test_invalid_ai_response_does_not_advance_or_award_score(self):
        session_id = self.start()
        self.create.return_value = completion({"ai_response": "", "objectives_met": []})
        self.assertEqual((await self.turn(session_id))["status_code"], 503)
        self.assertEqual(self.create.await_count, 2)
        self.assert_pristine(session_id)
        summary = self.scenarios.end_scenario(session_id, owner_id="alice")
        self.assertEqual(summary["total_turns"], 0)
        self.assertEqual(summary["xp_earned"], 0)

    async def test_unexpected_exception_releases_guard_without_committing(self):
        session_id = self.start()
        with patch.object(
            self.scenarios, "_build_system_prompt", side_effect=RuntimeError("internal"),
        ):
            with self.assertRaises(RuntimeError):
                await self.turn(session_id)
        self.assert_pristine(session_id)
        self.assertEqual((await self.turn(session_id))["turn_count"], 1)

    async def test_turn_limit_is_checked_before_transcription(self):
        self.scenarios.SCENARIO_MAX_TURNS = 2
        session_id = self.start()
        self.assertEqual((await self.turn(session_id))["turn_count"], 1)
        self.assertEqual((await self.turn(session_id))["turn_count"], 2)
        self.assertEqual((await self.turn(session_id))["status_code"], 409)
        self.assertEqual(self.transcribe.await_count, 2)
        self.assertEqual(self.create.await_count, 2)
        summary = self.scenarios.end_scenario(session_id, owner_id="alice")
        self.assertEqual(summary["total_turns"], 2)

    async def test_history_preserves_greeting_and_latest_complete_turn_pairs(self):
        self.scenarios.SCENARIO_HISTORY_TURNS = 2
        session_id = self.start()
        greeting = self.scenarios._sessions[session_id].messages[0].copy()
        for i in range(8):
            self.transcribe.return_value = f"Learner turn {i}"
            await self.turn(session_id)
        session = self.scenarios._sessions[session_id]
        self.assertEqual(session.turn_count, 8)
        self.assertEqual(len(session.messages), 5)
        self.assertEqual(session.messages[0], greeting)
        self.assertEqual(session.messages[1]["content"], "Learner turn 6")
        self.assertEqual(session.messages[3]["content"], "Learner turn 7")
        self.assertEqual([m["role"] for m in session.messages], [
            "assistant", "user", "assistant", "user", "assistant",
        ])
        for provider_call in self.create.await_args_list:
            self.assertLessEqual(len(provider_call.kwargs["messages"]), 7)

    async def test_successful_objectives_response_cannot_mutate_internal_state(self):
        session_id = self.start()
        result = await self.turn(session_id)
        result["objectives_met"].append("unexpected external mutation")
        self.assertEqual(self.scenarios._sessions[session_id].objectives_met, ["Saludar"])

    async def test_end_is_single_use_and_removes_session(self):
        session_id = self.start()
        await self.turn(session_id)
        summary = self.scenarios.end_scenario(session_id, owner_id="alice")
        self.assertEqual(summary["total_turns"], 1)
        self.assertEqual(summary["average_score"], 100)
        self.assertNotIn(session_id, self.scenarios._sessions)
        self.assertIsNone(self.scenarios.end_scenario(session_id, owner_id="alice"))
        self.assertIsNone(await self.turn(session_id))
        self.transcribe.assert_awaited_once()

    def test_configuration_requires_positive_integers_and_bounds_history(self):
        for name in SETTINGS:
            for value in ("0", "-1", "oops"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    load_scenarios({name: value})
        scenarios, _, _ = load_scenarios({
            "SCENARIO_MAX_TURNS": "4", "SCENARIO_HISTORY_TURNS": "8",
        })
        self.assertEqual(scenarios.SCENARIO_HISTORY_TURNS, 4)


if __name__ == "__main__":
    unittest.main()
