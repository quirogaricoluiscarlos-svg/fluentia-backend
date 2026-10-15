"""Firebase ID-token authentication for paid backend operations.

The SDK is imported lazily and uses Application Default Credentials supplied
outside the repository. No token, credential, or provider exception is logged.
https://firebase.google.com/docs/auth/admin/verify-id-tokens
https://firebase.google.com/docs/auth/admin/manage-sessions
"""

from dataclasses import dataclass
import os
import re
import threading

from fastapi import Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool


MAX_AUTHORIZATION_LENGTH = 8192
_BEARER = re.compile(r"Bearer ([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)")
_APP_NAME = "fluentia-auth"
_APP_LOCK = threading.Lock()


@dataclass(frozen=True)
class Principal:
    uid: str
    is_guest: bool
    is_admin: bool = False


class _TokenRejected(Exception):
    pass


class _AuthUnavailable(Exception):
    pass


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Necesitas una sesión válida. Vuelve a iniciar sesión.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="La autenticación no está disponible. Intenta más tarde.")


def _configuration() -> tuple[str, bool]:
    project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
    if not project_id or len(project_id) > 128 or any(char.isspace() for char in project_id):
        raise _unavailable()
    # The Admin SDK trusts unsigned emulator tokens when this variable is set.
    # This backend has no development-auth bypass on its public endpoints.
    if os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "").strip():
        raise _unavailable()
    allow_guest = os.getenv("AUTH_ALLOW_GUEST", "true").strip().lower()
    if allow_guest not in {"true", "false", "1", "0", "yes", "no"}:
        raise _unavailable()
    return project_id, allow_guest in {"true", "1", "yes"}


def _verify_id_token(token: str, project_id: str) -> dict:
    """Synchronous SDK work; authenticate always calls this in a threadpool."""
    try:
        import firebase_admin
        from firebase_admin import auth
    except ImportError:
        raise _AuthUnavailable() from None

    try:
        with _APP_LOCK:
            try:
                app = firebase_admin.get_app(_APP_NAME)
            except ValueError:
                # initialize_app uses ADC (e.g. GOOGLE_APPLICATION_CREDENTIALS
                # pointing to a mounted service account), never a repo secret.
                app = firebase_admin.initialize_app(
                    options={"projectId": project_id}, name=_APP_NAME
                )
            if app.project_id != project_id:
                raise _AuthUnavailable()
    except Exception:
        raise _AuthUnavailable() from None

    invalid_types = (ValueError,) + tuple(
        error_type for name in (
            "InvalidIdTokenError", "ExpiredIdTokenError", "RevokedIdTokenError",
            "UserDisabledError", "UserNotFoundError", "TenantIdMismatchError",
        )
        if isinstance(error_type := getattr(auth, name, None), type)
        and issubclass(error_type, Exception)
    )
    try:
        return auth.verify_id_token(token, app=app, check_revoked=True)
    except invalid_types:
        raise _TokenRejected() from None
    except Exception:
        # Certificates, credentials, permissions and network failures all
        # fail closed. They are unavailable auth, not a valid guest session.
        raise _AuthUnavailable() from None


async def authenticate(
    request: Request, authorization: str = Header(default="")
) -> Principal:
    if (
        not isinstance(authorization, str)
        or len(authorization) > MAX_AUTHORIZATION_LENGTH
        or len(request.headers.getlist("authorization")) > 1
    ):
        raise _unauthorized()
    match = _BEARER.fullmatch(authorization)
    if not match:
        raise _unauthorized()
    project_id, allow_guest = _configuration()
    try:
        claims = await run_in_threadpool(_verify_id_token, match.group(1), project_id)
    except _TokenRejected:
        raise _unauthorized() from None
    except _AuthUnavailable:
        raise _unavailable() from None
    except Exception:
        raise _unavailable() from None

    if not isinstance(claims, dict):
        raise _unauthorized()
    uid = claims.get("uid")
    firebase = claims.get("firebase")
    provider = firebase.get("sign_in_provider") if isinstance(firebase, dict) else None
    if (
        not isinstance(uid, str) or not uid.strip() or len(uid) > 128
        or ("sub" in claims and claims["sub"] != uid)
        or not isinstance(provider, str) or not provider.strip()
    ):
        raise _unauthorized()
    is_guest = provider == "anonymous"
    if is_guest and not allow_guest:
        raise HTTPException(status_code=403, detail="El acceso de invitado no está habilitado.")
    principal = Principal(uid=uid, is_guest=is_guest, is_admin=not is_guest and claims.get("admin") is True)
    request.state.principal = principal
    return principal
