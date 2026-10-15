"""Bound multipart/JSON request bodies before FastAPI parses or spools them."""

import asyncio
import os
from starlette.responses import JSONResponse


class RequestSizeLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        try:
            max_audio = int(os.getenv("MAX_AUDIO_BYTES", "1048576"))
            maximum = int(os.getenv("MAX_REQUEST_BYTES", str(max_audio + 65536)))
            if not 0 < max_audio <= 5 * 1024 * 1024 or not max_audio + 4096 <= maximum <= 6 * 1024 * 1024:
                raise ValueError()
        except ValueError:
            return await JSONResponse({"detail": "El servicio de audio no está disponible."}, 503)(scope, receive, send)

        declared = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
        try:
            if len(declared) > 1 or (declared and int(declared[0]) < 0):
                raise ValueError()
            if declared and int(declared[0]) > maximum:
                return await JSONResponse({"detail": "La solicitud es demasiado grande."}, 413)(scope, receive, send)
        except ValueError:
            return await JSONResponse({"detail": "La solicitud no es válida."}, 400)(scope, receive, send)

        body = bytearray()
        exceeded = False
        try:
            async with asyncio.timeout(30):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > maximum:
                        exceeded = True
                        break
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "La subida tardó demasiado. Vuelve a intentarlo."}, 408)(scope, receive, send)

        if exceeded:
            return await JSONResponse({"detail": "La solicitud es demasiado grande."}, 413)(scope, receive, send)

        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
