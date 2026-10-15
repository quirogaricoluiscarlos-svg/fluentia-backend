"""ASGI body limits do not trust Content-Length or send duplicate errors."""

import asyncio
from contextlib import asynccontextmanager
import unittest
from unittest.mock import AsyncMock, patch
from starlette.responses import JSONResponse
from request_limits import RequestSizeLimitMiddleware


class RequestLimitTests(unittest.IsolatedAsyncioTestCase):
    async def run_request(self, chunks, headers=None, method="POST", sender=None):
        self.called = False
        self.payload = None
        sent = []

        async def app(scope, receive, send):
            self.called = True
            self.payload = (await receive()).get("body")
            await JSONResponse({"ok": True})(scope, receive, send)

        async def send(message):
            if sender:
                await sender(message)
            sent.append(message)

        receive = AsyncMock(side_effect=chunks)
        scope = {"type": "http", "method": method, "headers": headers or []}
        with patch.dict("os.environ", {"MAX_AUDIO_BYTES": "512", "MAX_REQUEST_BYTES": "6000"}):
            await RequestSizeLimitMiddleware(app)(scope, receive, send)
        return sent, receive

    async def test_valid_stream_is_replayed_without_changing_bytes(self):
        sent, _ = await self.run_request([
            {"type": "http.request", "body": b"first", "more_body": True},
            {"type": "http.request", "body": b"second", "more_body": False},
        ])
        self.assertTrue(self.called)
        self.assertEqual(self.payload, b"firstsecond")
        self.assertEqual(sent[0]["status"], 200)

    async def test_oversized_content_length_is_rejected_without_reading_body(self):
        sent, receive = await self.run_request([], headers=[(b"content-length", b"6001")])
        self.assertEqual(sent[0]["status"], 413)
        receive.assert_not_awaited()
        self.assertFalse(self.called)

    async def test_chunked_body_or_forged_small_content_length_cannot_exceed_limit(self):
        for headers in [[], [(b"content-length", b"1")]]:
            with self.subTest(headers=headers):
                sent, _ = await self.run_request([
                    {"type": "http.request", "body": b"x" * 4000, "more_body": True},
                    {"type": "http.request", "body": b"x" * 2001, "more_body": False},
                ], headers=headers)
                self.assertEqual(sent[0]["status"], 413)
                self.assertFalse(self.called)

    async def test_error_response_is_sent_after_receive_timeout_scope_closes(self):
        active = False
        @asynccontextmanager
        async def deadline(_seconds):
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False
        async def slow_send(_message):
            self.assertFalse(active, "Sending an error inside receive timeout can produce a second response.start")
            await asyncio.sleep(0)
        with patch("request_limits.asyncio.timeout", deadline):
            sent, _ = await self.run_request([
                {"type": "http.request", "body": b"x" * 6001, "more_body": False}
            ], sender=slow_send)
        self.assertEqual(sum(item["type"] == "http.response.start" for item in sent), 1)
        self.assertEqual(sent[0]["status"], 413)

    async def test_invalid_or_duplicate_length_is_rejected(self):
        for headers in [[(b"content-length", b"-1")], [(b"content-length", b"invalid")],
                        [(b"content-length", b"1"), (b"content-length", b"1")]]:
            with self.subTest(headers=headers):
                sent, _ = await self.run_request([], headers=headers)
                self.assertEqual(sent[0]["status"], 400)

    async def test_disconnect_does_not_enter_endpoint(self):
        sent, _ = await self.run_request([{"type": "http.disconnect"}])
        self.assertFalse(self.called)
        self.assertEqual(sent, [])

    async def test_invalid_configuration_fails_closed(self):
        app = AsyncMock()
        send = AsyncMock()
        with patch.dict("os.environ", {"MAX_REQUEST_BYTES": "invalid"}):
            await RequestSizeLimitMiddleware(app)({"type": "http", "method": "POST"}, AsyncMock(), send)
        app.assert_not_awaited()
        self.assertEqual(send.await_args_list[0].args[0]["status"], 503)
