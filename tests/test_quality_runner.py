"""Regression tests for the golden runner's verdicts, with no live evaluations."""

import asyncio
import importlib.util
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import test_scenario_sessions as scenario_fixture


spec = importlib.util.spec_from_file_location(
    "quality_runner_under_test", Path(__file__).resolve().parents[1] / "test_quality.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class QualityRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def verdicts(self, status, data=None):
        client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(
            status_code=status, json=lambda: data
        )))
        return await runner._test_transcription(client, "http://local-only"), client

    async def test_http_errors_never_pass_non_speech_tests(self):
        for status in [400, 422, 500, 502]:
            with self.subTest(status=status):
                results, _ = await self.verdicts(status)
                self.assertTrue(all(not result["passed"] for result in results))

    async def test_explicit_non_speech_contract_passes_and_supplies_activity(self):
        results, client = await self.verdicts(200, {
            "evaluation_status": "no_speech", "grammar_status": "unavailable",
            "transcription": "", "pronunciation": None,
        })
        self.assertTrue(all(result["passed"] for result in results))
        for call in client.post.await_args_list:
            self.assertEqual(call.kwargs["data"], {"activity_id": "f1"})

    async def test_empty_legacy_response_never_passes(self):
        results, _ = await self.verdicts(200, {"transcription": ""})
        self.assertTrue(all(not result["passed"] for result in results))

    async def test_hallucinated_text_or_evaluation_never_passes(self):
        for data in [{"evaluation_status": "complete", "transcription": "Thank you."},
                     {"evaluation_status": "no_speech", "grammar_status": "unavailable",
                      "transcription": "", "pronunciation": {"overall_score": 100}}]:
            with self.subTest(data=data):
                results, _ = await self.verdicts(200, data)
                self.assertTrue(all(not result["passed"] for result in results))

    async def test_grammar_outage_cannot_pass_correct_sentence_case(self):
        services = ModuleType("services")
        services.correct_grammar = AsyncMock(side_effect=RuntimeError("mock outage"))
        with patch.dict(sys.modules, {"services": services}):
            results = await runner._test_grammar(None, "http://local-only")
        self.assertTrue(all(not result["passed"] for result in results))
        self.assertTrue(all(call.kwargs == {"strict": True} for call in services.correct_grammar.await_args_list))

    async def test_transcription_checks_forward_authorization(self):
        client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(
            status_code=401,
        )))
        await runner._test_transcription(client, "http://local-only", "Bearer test.token.signature")
        for call in client.post.await_args_list:
            self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer test.token.signature"})

    async def test_scenario_success_or_error_always_removes_only_its_owned_session(self):
        for outcome in ("success", "provider_error", "invalid_response"):
            with self.subTest(outcome=outcome):
                scenarios, _, create = scenario_fixture.load_scenarios()
                unrelated_id, _ = scenarios.start_scenario("smalltalk", owner_id="unrelated")
                if outcome == "provider_error":
                    create.side_effect = RuntimeError("mock outage")
                elif outcome == "invalid_response":
                    create.return_value = scenario_fixture.completion({"ai_response": ""})
                with (
                    patch.dict(sys.modules, {"scenarios": scenarios}),
                    patch.object(sys, "path", [str(scenario_fixture.BACKEND), *sys.path]),
                    patch.object(runner, "SCENARIO_CASES", runner.SCENARIO_CASES[:1]),
                    patch.object(scenarios, "end_scenario", wraps=scenarios.end_scenario) as end,
                ):
                    results = await runner._test_scenarios(None, "http://local-only", "admin-owner")
                self.assertEqual(results[0]["passed"], outcome == "success", results[0]["reason"])
                end.assert_called_once()
                self.assertEqual(end.call_args.kwargs, {"owner_id": "admin-owner"})
                self.assertEqual(set(scenarios._sessions), {unrelated_id})

    async def test_scenario_cancellation_cleans_session_and_propagates_cancellation(self):
        scenarios, _, create = scenario_fixture.load_scenarios()
        entered = asyncio.Event()

        async def blocked(*_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

        create.side_effect = blocked
        with (
            patch.dict(sys.modules, {"scenarios": scenarios}),
            patch.object(runner, "SCENARIO_CASES", runner.SCENARIO_CASES[:1]),
            patch.object(scenarios, "end_scenario", wraps=scenarios.end_scenario) as end,
        ):
            task = asyncio.create_task(runner._test_scenarios(None, "http://local-only", "admin-owner"))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        end.assert_called_once()
        self.assertEqual(end.call_args.kwargs, {"owner_id": "admin-owner"})
        self.assertEqual(scenarios._sessions, {})


class RemoteGoldenRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_request_targets_selected_backend_without_local_providers(self):
        requests = []
        result = {"total": 19, "passed": 19, "failed": 0}

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=result)

        with patch.object(runner, "run_golden_set", AsyncMock()) as local_runner:
            actual = await runner.fetch_remote_golden_set(
                "https://remote.invalid/api/", id_token="test.token.signature",
                eval_key="k" * 32, transport=httpx.MockTransport(respond),
            )
        self.assertEqual(actual, result)
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "https://remote.invalid/api/eval/run")
        self.assertEqual(requests[0].method, "GET")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test.token.signature")
        self.assertEqual(requests[0].headers["X-Eval-Key"], "k" * 32)
        local_runner.assert_not_awaited()

    async def test_http_errors_and_redirects_fail_without_local_runner_or_retry(self):
        for status in (301, 401, 403, 429, 503):
            requests = []

            def reject(request):
                requests.append(request)
                return httpx.Response(status, headers={"Location": "https://other.invalid"})

            with (
                self.subTest(status=status),
                patch.object(runner, "run_golden_set", AsyncMock()) as local_runner,
                self.assertRaises(httpx.HTTPStatusError),
            ):
                await runner.fetch_remote_golden_set(
                    "https://remote.invalid", id_token="test.token.signature",
                    eval_key="k" * 32, transport=httpx.MockTransport(reject),
                )
            self.assertEqual(len(requests), 1)
            local_runner.assert_not_awaited()

    async def test_missing_or_invalid_results_cannot_report_cli_success(self):
        for data in ({}, [], {"total": 0, "passed": 0, "failed": 0},
                     {"total": 19, "passed": 19, "failed": False},
                     {"total": 19, "passed": 18, "failed": 0}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                await runner.fetch_remote_golden_set(
                    "https://remote.invalid", id_token="test.token.signature", eval_key="k" * 32,
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data)),
                )


class GoldenCliTests(unittest.TestCase):
    def test_missing_operator_environment_never_requests_remote_or_local_evaluation(self):
        for env in ({}, {"EVAL_ID_TOKEN": "test.token.signature"}, {"EVAL_KEY": "k" * 32},
                    {"EVAL_ID_TOKEN": "test.token.signature", "EVAL_KEY": "k" * 31}):
            with (
                self.subTest(configured=list(env)),
                patch.dict(os.environ, env, clear=True),
                patch.object(runner, "fetch_remote_golden_set", AsyncMock()) as remote,
                patch.object(runner, "run_golden_set", AsyncMock()) as local,
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(runner.main(["https://remote.invalid"]), 2)
            remote.assert_not_called()
            local.assert_not_called()

    def test_cli_uses_operator_environment_and_returns_backend_verdict(self):
        for passed, failed, exit_code in ((19, 0, 0), (18, 1, 1)):
            with (
                self.subTest(failed=failed),
                patch.dict(os.environ, {"EVAL_ID_TOKEN": "test.token.signature", "EVAL_KEY": "k" * 32}, clear=True),
                patch.object(runner, "fetch_remote_golden_set", AsyncMock(
                    return_value={"total": 19, "passed": passed, "failed": failed},
                )) as remote,
                patch.object(runner, "run_golden_set", AsyncMock()) as local,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.main(["https://remote.invalid"]), exit_code)
            remote.assert_awaited_once_with("https://remote.invalid", id_token="test.token.signature", eval_key="k" * 32)
            local.assert_not_called()

    def test_cli_http_error_returns_failure_without_echoing_credentials(self):
        error = httpx.HTTPStatusError(
            "test.token.signature " + "k" * 32,
            request=httpx.Request("GET", "https://remote.invalid/eval/run"),
            response=httpx.Response(403),
        )
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"EVAL_ID_TOKEN": "test.token.signature", "EVAL_KEY": "k" * 32}, clear=True),
            patch.object(runner, "fetch_remote_golden_set", AsyncMock(side_effect=error)) as remote,
            patch.object(runner, "run_golden_set", AsyncMock()) as local,
            redirect_stderr(output),
        ):
            self.assertEqual(runner.main(["https://remote.invalid"]), 1)
        self.assertIn("HTTP 403", output.getvalue())
        self.assertNotIn("test.token.signature", output.getvalue())
        self.assertNotIn("k" * 32, output.getvalue())
        remote.assert_awaited_once()
        local.assert_not_called()


if __name__ == "__main__":
    unittest.main()
