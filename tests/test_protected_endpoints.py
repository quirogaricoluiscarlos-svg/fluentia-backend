"""HTTP security regression: only fake providers, verified-token mocks and temp DBs."""

from pathlib import Path
import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import test_practice_contract as practice_fixture
import test_quality_runner as quality_fixture
import httpx
from fastapi import HTTPException
from security import Principal, _TokenRejected, _AuthUnavailable
from quota import QuotaStore
import provider_access


class ProtectedEndpointTests(unittest.TestCase):
    def setUp(self):
        practice_fixture.PracticeHttpTests.setUp(self)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "quota.db"
        self.store = QuotaStore(self.database, guest_daily=6, guest_burst=100,
                                user_daily=120, user_burst=30)
        self.budget_patch = patch.object(provider_access, "consume", self.store.consume)
        self.budget_patch.start()
        self.addCleanup(self.budget_patch.stop)
        self.main.provider_access = provider_access.provider_access

    def assess(self, data=None):
        return practice_fixture.PracticeHttpTests.assess(self, data)

    def assert_no_ai(self):
        for service in (self.services.transcribe_audio, self.services.correct_grammar,
                        self.services.assess_pronunciation, self.services.generate_tts):
            service.assert_not_awaited()

    def test_missing_uid_placeholder_and_invalid_tokens_cannot_consume_any_provider(self):
        self.main.app.dependency_overrides.clear()
        for authorization in [None, "uid-only", "Bearer placeholder", "Bearer uid-only"]:
            headers = {} if authorization is None else {"Authorization": authorization}
            with self.subTest(authorization=authorization):
                for path in ["/assess", "/scenario/turn?session_id=known", "/placement/evaluate?item_id=1"]:
                    response = self.client.post(path, headers=headers,
                        data={"activity_id": "f1"}, files={"audio": ("audio.wav", self.audio, "audio/wav")})
                    self.assertEqual(response.status_code, 401)
                self.assertEqual(self.client.get("/tts?text=Hello", headers=headers).status_code, 401)
        self.assert_no_ai()
        self.assertFalse(self.database.exists())

    def test_rejected_signed_token_does_not_consume_provider_or_quota(self):
        self.main.app.dependency_overrides.clear()
        with patch.dict("os.environ", {"FIREBASE_PROJECT_ID": "test-project"}), \
             patch("security._verify_id_token", side_effect=_TokenRejected):
            response = self.client.get("/tts?text=Hello", headers={"Authorization": "Bearer test.token.signature"})
        self.assertEqual(response.status_code, 401)
        self.assert_no_ai()
        self.assertFalse(self.database.exists())

    def test_auth_configuration_outage_is_closed_without_quota_or_ai(self):
        self.main.app.dependency_overrides.clear()
        with patch.dict("os.environ", {"FIREBASE_PROJECT_ID": "test-project"}), \
             patch("security._verify_id_token", side_effect=_AuthUnavailable):
            response = self.client.get("/tts?text=Hello", headers={"Authorization": "Bearer test.token.signature"})
        self.assertEqual(response.status_code, 503)
        self.assert_no_ai()
        self.assertFalse(self.database.exists())

    def test_verified_firebase_guest_identity_reaches_bounded_provider(self):
        self.main.app.dependency_overrides.clear()
        self.services.generate_tts.return_value = b"fake-mp3"
        claims = {"uid": "firebase-guest", "sub": "firebase-guest", "firebase": {"sign_in_provider": "anonymous"}}
        with patch.dict("os.environ", {"FIREBASE_PROJECT_ID": "test-project", "AUTH_ALLOW_GUEST": "true"}), \
             patch("security._verify_id_token", return_value=claims):
            response = self.client.get("/tts?text=Hello", headers={"Authorization": "Bearer test.token.signature"})
        self.assertEqual(response.status_code, 200)
        self.services.generate_tts.assert_awaited_once_with("Hello")
        self.assertTrue(self.database.exists())

    def test_daily_quota_denies_before_third_assessment_and_includes_retry_after(self):
        self.assertEqual(self.assess().status_code, 200)
        self.assertEqual(self.assess().status_code, 200)
        response = self.assess()
        self.assertEqual(response.status_code, 429)
        self.assertGreater(int(response.headers["Retry-After"]), 0)
        self.assertEqual(self.services.transcribe_audio.await_count, 2)

    def test_new_anonymous_uid_does_not_bypass_shared_ip_budget(self):
        self.assertEqual(self.assess().status_code, 200)
        self.assertEqual(self.assess().status_code, 200)
        self.main.app.dependency_overrides[self.main.authenticate] = lambda: Principal("new-guest", True)
        self.assertEqual(self.assess().status_code, 429)
        self.assertEqual(self.services.transcribe_audio.await_count, 2)

    def test_invalid_wav_is_rejected_before_quota_or_transcription(self):
        response = self.client.post("/assess", data={"activity_id": "f1"},
            files={"audio": ("audio.wav", b"not WAV", "audio/wav")})
        self.assertIn(response.status_code, [415, 422])
        self.assert_no_ai()
        self.assertFalse(self.database.exists())

    def test_oversized_multipart_is_rejected_before_parsing_or_provider(self):
        response = self.client.post("/assess", data={"activity_id": "f1"},
            files={"audio": ("audio.wav", b"x" * 1200000, "audio/wav")})
        self.assertEqual(response.status_code, 413)
        self.assert_no_ai()
        self.assertFalse(self.database.exists())

    def test_provider_names_are_canonical_even_when_filename_is_misleading(self):
        response = self.client.post("/assess", data={"activity_id": "f1"},
            files={"audio": ("other.mp3", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.services.transcribe_audio.assert_awaited_once_with(self.audio, "audio.wav")

    def test_scenario_routes_pass_verified_owner_and_recoverable_error_status(self):
        self.main.start_scenario = Mock(return_value=("session", {"session_id": "session"}))
        response = self.client.post("/scenario/start?scenario_id=cafe")
        self.assertEqual(response.status_code, 200)
        self.main.start_scenario.assert_called_once_with("cafe", owner_id="test-guest")
        self.main.process_turn = AsyncMock(return_value={"error": "Espera al turno activo.", "status_code": 409})
        response = self.client.post("/scenario/turn?session_id=session", files={"audio": ("audio.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 409)
        self.main.process_turn.assert_awaited_once_with("session", self.audio, "audio.wav", owner_id="test-guest")
        self.main.end_scenario = Mock(return_value={"error": "Espera al turno activo.", "status_code": 409})
        self.assertEqual(self.client.post("/scenario/end?session_id=session").status_code, 409)
        self.main.end_scenario.assert_called_once_with("session", owner_id="test-guest")

    def test_foreign_or_expired_session_has_generic_not_found(self):
        self.main.process_turn = AsyncMock(return_value=None)
        response = self.client.post("/scenario/turn?session_id=foreign", files={"audio": ("audio.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 404)
        self.assert_no_ai()

    def test_placement_validates_audio_then_uses_authenticated_budget(self):
        self.main.get_placement_item = Mock(return_value={"id": 1, "phrase": "Hello"})
        self.main.evaluate_placement_item = AsyncMock(return_value={"score": 50})
        response = self.client.post("/placement/evaluate?item_id=1", files={"audio": ("audio.wav", self.audio, "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.main.evaluate_placement_item.assert_awaited_once_with(self.audio, "audio.wav", {"id": 1, "phrase": "Hello"})

    def test_provider_timeout_releases_slot_before_next_request(self):
        async def blocked(*_args):
            await asyncio.Event().wait()
        self.services.transcribe_audio.side_effect = blocked
        with patch.dict("os.environ", {"PROVIDER_TIMEOUT_SECONDS": "1"}):
            self.assertEqual(self.assess().status_code, 504)
        self.services.generate_tts.return_value = b"fake-mp3"
        self.assertEqual(self.client.get("/tts?text=Hello").status_code, 200)

    def test_invalid_provider_timeout_fails_closed_before_quota_or_ai(self):
        with patch.dict("os.environ", {"PROVIDER_TIMEOUT_SECONDS": "0"}):
            self.assertEqual(self.assess().status_code, 503)
        self.assert_no_ai()
        self.assertFalse(self.database.exists())

    def test_non_admin_cannot_run_paid_golden_set_even_with_eval_key(self):
        with patch.dict("os.environ", {"EVAL_KEY": "test-only-key-" + "x" * 32}):
            response = self.client.get("/eval/run", headers={"X-Eval-Key": "test-only-key-" + "x" * 32})
        self.assertEqual(response.status_code, 403)
        self.main.run_golden_set.assert_not_awaited()
        self.assert_no_ai()

    def test_admin_invalid_or_unicode_key_is_forbidden_without_server_error(self):
        self.main.app.dependency_overrides[self.main.authenticate] = lambda: Principal("admin", False, True)
        with patch.dict("os.environ", {"EVAL_KEY": "test-only-key-" + "x" * 32}):
            response = self.client.get("/eval/run", headers={"X-Eval-Key": "different-key"})
            self.assertEqual(response.status_code, 403)
            with self.assertRaises(HTTPException) as caught:
                # HTTPX rejects non-ASCII header values before transmission.
                self.main._check_eval_key("clave con ñ", Principal("admin", False, True))
            self.assertEqual(caught.exception.status_code, 403)
        self.main.run_golden_set.assert_not_awaited()

    def test_admin_weak_configured_eval_key_is_rejected_without_quota(self):
        self.main.app.dependency_overrides[self.main.authenticate] = lambda: Principal("admin", False, True)
        key = "x" * 31
        with patch.dict("os.environ", {"EVAL_KEY": key}):
            for endpoint in ("/eval/run", "/eval/metrics"):
                self.assertEqual(self.client.get(endpoint, headers={"X-Eval-Key": key}).status_code, 403)
        self.main.run_golden_set.assert_not_awaited()
        self.main.get_metrics.assert_not_awaited()
        self.assertFalse(self.database.exists())

    def test_admin_golden_set_is_bounded_and_receives_verified_identity_context(self):
        self.main.app.dependency_overrides[self.main.authenticate] = lambda: Principal("admin", False, True)
        self.main.run_golden_set.return_value = {"total": 0, "passed": 0}
        authorization = "Bearer test.token.signature"
        key = "test-only-key-" + "x" * 32
        with patch.dict("os.environ", {"EVAL_KEY": key}):
            response = self.client.get("/eval/run", headers={"X-Eval-Key": key, "Authorization": authorization})
        self.assertEqual(response.status_code, 200)
        self.main.run_golden_set.assert_awaited_once()
        call = self.main.run_golden_set.await_args
        self.assertEqual(call.args, ("http://quality-internal",))
        self.assertEqual(call.kwargs["authorization"], authorization)
        self.assertEqual(call.kwargs["owner_id"], "admin")
        self.assertIsInstance(call.kwargs["transport"], httpx.ASGITransport)
        self.assertIs(call.kwargs["transport"].app, self.main.app)

    def test_admin_internal_checks_use_same_app_auth_and_quota_without_a_listening_port(self):
        runner = quality_fixture.runner
        self.main.run_golden_set = runner.run_golden_set
        self.main.app.dependency_overrides.clear()
        self.services.transcribe_audio.return_value = ""
        key = "test-only-key-" + "x" * 32
        claims = {"uid": "admin", "sub": "admin", "admin": True,
                  "firebase": {"sign_in_provider": "password"}}
        with (
            patch.dict("os.environ", {"EVAL_KEY": key, "FIREBASE_PROJECT_ID": "test-project"}),
            patch("security._verify_id_token", return_value=claims) as verify,
            patch.object(runner, "_test_grammar", AsyncMock(return_value=[])),
            patch.object(runner, "_test_pronunciation", AsyncMock(return_value=[])),
            patch.object(runner, "_test_placement", AsyncMock(return_value=[])),
            patch.object(runner, "_test_scenarios", AsyncMock(return_value=[])),
        ):
            response = self.client.get("/eval/run", headers={
                "X-Eval-Key": key, "Authorization": "Bearer test.token.signature",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 2)
        self.assertEqual(response.json()["passed"], 2)
        self.assertEqual(verify.call_count, 3)  # The admin request and both /assess calls.
        self.assertEqual(self.services.transcribe_audio.await_count, 2)

    def test_admin_timeout_keeps_cost_and_releases_slot(self):
        self.main.app.dependency_overrides[self.main.authenticate] = lambda: Principal("admin", False, True)
        self.store.user_concurrent = 1
        self.store.clock = lambda: 1728000000.0

        async def blocked(*_args, **_kwargs):
            await asyncio.Event().wait()

        self.main.run_golden_set.side_effect = blocked
        key = "test-only-key-" + "x" * 32
        with patch.dict("os.environ", {"EVAL_KEY": key, "PROVIDER_TIMEOUT_SECONDS": "1"}):
            response = self.client.get("/eval/run", headers={"X-Eval-Key": key})
        self.assertEqual(response.status_code, 504)
        # A second paid evaluation costs another 20 units and exceeds burst 30,
        # while a small request succeeds because the timeout released its slot.
        self.main.run_golden_set.side_effect = None
        with patch.dict("os.environ", {"EVAL_KEY": key}):
            self.assertEqual(self.client.get("/eval/run", headers={"X-Eval-Key": key}).status_code, 429)
        self.services.generate_tts.return_value = b"fake-mp3"
        self.assertEqual(self.client.get("/tts?text=Hello").status_code, 200)
