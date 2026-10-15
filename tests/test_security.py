"""Firebase verification tests with an SDK stub; never use ADC or network."""

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from starlette.requests import Request

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1]), *sys.path]):
    import security


TOKEN = "header.payload.signature"


def request(headers=None):
    return Request({
        "type": "http", "method": "GET", "path": "/", "headers": headers or [],
        "client": ("203.0.113.1", 1234), "scheme": "https", "server": ("test", 443),
    })


class SecurityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.admin = ModuleType("firebase_admin")
        self.auth = ModuleType("firebase_admin.auth")
        for name in (
            "InvalidIdTokenError", "ExpiredIdTokenError", "RevokedIdTokenError",
            "UserDisabledError", "UserNotFoundError", "TenantIdMismatchError", "CertificateFetchError",
        ):
            setattr(self.auth, name, type(name, (Exception,), {}))
        self.claims = {"uid": "user-1", "sub": "user-1", "firebase": {"sign_in_provider": "password"}}
        self.auth.verify_id_token = Mock(return_value=self.claims)
        self.app = SimpleNamespace(project_id="test-project")
        self.admin.auth = self.auth
        self.admin.get_app = Mock(return_value=self.app)
        self.admin.initialize_app = Mock(return_value=self.app)
        self.modules = patch.dict(sys.modules, {"firebase_admin": self.admin, "firebase_admin.auth": self.auth})
        self.environment = patch.dict(os.environ, {"FIREBASE_PROJECT_ID": "test-project"}, clear=True)
        self.modules.start()
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.modules.stop)

    async def rejected(self, authorization, status=401, req=None):
        with self.assertRaises(HTTPException) as error:
            await security.authenticate(req or request(), authorization)
        self.assertEqual(error.exception.status_code, status)
        self.assertNotIn(TOKEN, str(error.exception.detail))
        return error.exception

    async def test_registered_user_uses_real_sdk_verification_and_revocation_check(self):
        req = request()
        principal = await security.authenticate(req, f"Bearer {TOKEN}")
        self.assertEqual(principal, security.Principal("user-1", False))
        self.assertEqual(req.state.principal, principal)
        self.auth.verify_id_token.assert_called_once_with(TOKEN, app=self.app, check_revoked=True)

    async def test_missing_uid_literal_placeholder_and_malformed_headers_are_rejected_before_sdk(self):
        for header in [
            "", "user-1", "Bearer user-1", "Bearer placeholder", "Bearer ",
            f"bearer {TOKEN}", f"Bearer  {TOKEN}", f" Bearer {TOKEN}", f"Bearer {TOKEN} ",
            f"Bearer {TOKEN}\n", "Bearer one.two", "Bearer one.two.three.four", None,
            "Bearer " + "a" * security.MAX_AUTHORIZATION_LENGTH + ".b.c",
        ]:
            with self.subTest(header=str(header)[:30]):
                await self.rejected(header)
        self.auth.verify_id_token.assert_not_called()

    async def test_duplicate_authorization_headers_are_rejected(self):
        req = request([(b"authorization", f"Bearer {TOKEN}".encode()), (b"authorization", b"Bearer other")])
        await self.rejected(f"Bearer {TOKEN}", req=req)
        self.auth.verify_id_token.assert_not_called()

    async def test_anonymous_firebase_user_is_guest_by_default(self):
        self.claims["firebase"]["sign_in_provider"] = "anonymous"
        principal = await security.authenticate(request(), f"Bearer {TOKEN}")
        self.assertTrue(principal.is_guest)
        self.assertFalse(principal.is_admin)

    async def test_guest_policy_is_explicit_and_registered_users_still_work(self):
        os.environ["AUTH_ALLOW_GUEST"] = "false"
        self.claims["firebase"]["sign_in_provider"] = "anonymous"
        await self.rejected(f"Bearer {TOKEN}", status=403)
        self.claims["firebase"]["sign_in_provider"] = "password"
        self.assertFalse((await security.authenticate(request(), f"Bearer {TOKEN}")).is_guest)

    async def test_admin_requires_bool_true_verified_claim_and_never_guest(self):
        for claim, expected in [(True, True), (False, False), ("true", False), (1, False), (None, False)]:
            with self.subTest(claim=claim):
                self.claims["admin"] = claim
                self.assertEqual((await security.authenticate(request(), f"Bearer {TOKEN}")).is_admin, expected)
        self.claims["admin"] = True
        self.claims["firebase"]["sign_in_provider"] = "anonymous"
        self.assertFalse((await security.authenticate(request(), f"Bearer {TOKEN}")).is_admin)

    async def test_invalid_expired_revoked_disabled_or_deleted_user_is_401(self):
        for name in ("InvalidIdTokenError", "ExpiredIdTokenError", "RevokedIdTokenError", "UserDisabledError", "UserNotFoundError"):
            with self.subTest(name=name):
                self.auth.verify_id_token.side_effect = getattr(self.auth, name)("sensitive provider reason")
                error = await self.rejected(f"Bearer {TOKEN}")
                self.assertNotIn("sensitive", error.detail)
                self.assertEqual(error.headers["WWW-Authenticate"], "Bearer")

    async def test_invalid_verified_claim_shape_fails_closed(self):
        for claims in [
            None, {}, {"uid": ""}, {"uid": 123}, {"uid": "user-1", "firebase": {}},
            {"uid": "user-1", "sub": "another-user", "firebase": {"sign_in_provider": "password"}},
            {"uid": "x" * 129, "firebase": {"sign_in_provider": "password"}},
        ]:
            with self.subTest(claims=claims):
                self.auth.verify_id_token.return_value = claims
                await self.rejected(f"Bearer {TOKEN}")

    async def test_missing_project_bad_guest_config_or_auth_emulator_is_503(self):
        configurations = [
            {}, {"FIREBASE_PROJECT_ID": " "},
            {"FIREBASE_PROJECT_ID": "test-project", "AUTH_ALLOW_GUEST": "maybe"},
            {"FIREBASE_PROJECT_ID": "test-project", "FIREBASE_AUTH_EMULATOR_HOST": "localhost:9099"},
        ]
        for config in configurations:
            with self.subTest(config=config), patch.dict(os.environ, config, clear=True):
                await self.rejected(f"Bearer {TOKEN}", status=503)
        self.auth.verify_id_token.assert_not_called()

    async def test_initialization_uses_adc_named_app_and_explicit_project(self):
        self.admin.get_app.side_effect = ValueError("no app yet")
        await security.authenticate(request(), f"Bearer {TOKEN}")
        self.admin.initialize_app.assert_called_once_with(options={"projectId": "test-project"}, name="fluentia-auth")

    async def test_missing_sdk_credentials_certificates_or_project_mismatch_is_503(self):
        with patch.dict(sys.modules, {"firebase_admin": None}):
            await self.rejected(f"Bearer {TOKEN}", status=503)
        self.admin.get_app.side_effect = ValueError("no app")
        self.admin.initialize_app.side_effect = RuntimeError("ADC private path")
        error = await self.rejected(f"Bearer {TOKEN}", status=503)
        self.assertNotIn("ADC", error.detail)
        self.admin.get_app.side_effect = None
        self.admin.initialize_app.side_effect = None
        self.auth.verify_id_token.side_effect = self.auth.CertificateFetchError("certificate URL")
        await self.rejected(f"Bearer {TOKEN}", status=503)
        self.auth.verify_id_token.side_effect = RuntimeError("credential/network failure")
        await self.rejected(f"Bearer {TOKEN}", status=503)
        self.auth.verify_id_token.side_effect = None
        self.app.project_id = "other-project"
        await self.rejected(f"Bearer {TOKEN}", status=503)

    async def test_sdk_work_is_dispatched_to_threadpool(self):
        with patch.object(security, "run_in_threadpool", new=AsyncMock(return_value=self.claims)) as pool:
            await security.authenticate(request(), f"Bearer {TOKEN}")
        pool.assert_awaited_once_with(security._verify_id_token, TOKEN, "test-project")


if __name__ == "__main__":
    unittest.main()
