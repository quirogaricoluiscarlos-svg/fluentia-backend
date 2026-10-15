"""Grammar failures never become successful assessments in strict mode.

All provider SDKs and logging are replaced before services is loaded. These
tests do not read .env, instantiate real clients, write a DB, or use network.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


def load_services():
    """Load an isolated services module with only fake provider dependencies."""
    groq = ModuleType("groq")
    groq.AsyncGroq = Mock()
    openai = ModuleType("openai")
    openai.AsyncOpenAI = Mock()
    httpx = ModuleType("httpx")
    httpx.AsyncClient = Mock()
    quality_log = ModuleType("quality_log")
    quality_log.log_ai_call = AsyncMock()
    dependencies = {
        "groq": groq,
        "openai": openai,
        "httpx": httpx,
        "quality_log": quality_log,
    }
    spec = importlib.util.spec_from_file_location(
        "fluentia_services_unit_test", BACKEND_DIR / "services.py"
    )
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, dependencies),
        patch.dict(os.environ, {
            "GROQ_API_KEY": "unit-test-only",
            "DEEPSEEK_API_KEY": "unit-test-only",
        }),
        patch.object(sys, "path", [str(BACKEND_DIR), *sys.path]),
    ):
        spec.loader.exec_module(module)
    return module


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class GrammarTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.services = load_services()
        self.create = AsyncMock()
        self.services.deepseek_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        )

    async def test_explicit_null_correction_with_feedback_is_valid(self):
        self.create.return_value = completion(json.dumps({
            "correction": None,
            "explanation": "Tu oración es correcta.",
        }))
        self.assertEqual(
            await self.services.correct_grammar("I am happy.", strict=True),
            (None, "Tu oración es correcta."),
        )
        self.create.assert_awaited_once()

    async def test_nonempty_correction_is_valid(self):
        self.create.return_value = completion(json.dumps({
            "correction": "I am happy.",
            "explanation": "Usa el verbo am con I.",
        }))
        self.assertEqual(
            await self.services.correct_grammar("I happy.", strict=True),
            ("I am happy.", "Usa el verbo am con I."),
        )

    async def test_invalid_response_retries_then_can_succeed(self):
        self.create.side_effect = [
            completion('{"correction": null, "explanation": ""}'),
            completion('{"correction": null, "explanation": "Tu oración es correcta."}'),
        ]
        self.assertEqual(
            await self.services.correct_grammar("I am happy.", strict=True),
            (None, "Tu oración es correcta."),
        )
        self.assertEqual(self.create.await_count, 2)

    async def test_missing_empty_or_invalid_fields_fail_strict_evaluation(self):
        cases = [
            {},
            {"explanation": "Tu oración es correcta."},
            {"correction": None},
            {"correction": None, "explanation": None},
            {"correction": None, "explanation": ""},
            {"correction": None, "explanation": "   "},
            {"correction": None, "explanation": 100},
            {"correction": None, "explanation": []},
            {"correction": "", "explanation": "Tu oración es correcta."},
            {"correction": "  ", "explanation": "Tu oración es correcta."},
            {"correction": True, "explanation": "Tu oración es correcta."},
            {"correction": 100, "explanation": "Tu oración es correcta."},
            {"correction": {}, "explanation": "Tu oración es correcta."},
            {"correction": [], "explanation": "Tu oración es correcta."},
            [],
            None,
        ]
        for data in cases:
            with self.subTest(data=data):
                self.create.return_value = completion(json.dumps(data))
                self.create.side_effect = None
                with self.assertRaises(self.services.GrammarEvaluationError):
                    await self.services.correct_grammar("I am happy.", strict=True)

    async def test_spanish_and_length_guardrails_still_apply(self):
        for explanation in ["Your sentence is correct. Good job.", "la " * 200]:
            with self.subTest(explanation=explanation[:30]):
                self.create.return_value = completion(json.dumps({
                    "correction": None, "explanation": explanation,
                }))
                with self.assertRaises(self.services.GrammarEvaluationError):
                    await self.services.correct_grammar("I am happy.", strict=True)

    async def test_invalid_json_or_empty_provider_content_fails_strict(self):
        for content in ["not json", "", None, "```json\n{}\n```"]:
            with self.subTest(content=content):
                self.create.return_value = completion(content)
                with self.assertRaises(self.services.GrammarEvaluationError):
                    await self.services.correct_grammar("I am happy.", strict=True)

    async def test_provider_failure_raises_sanitized_error_after_retry(self):
        self.create.side_effect = RuntimeError("provider unavailable")
        with self.assertRaises(self.services.GrammarEvaluationError) as error:
            await self.services.correct_grammar("I am happy.", strict=True)
        self.assertEqual(self.create.await_count, 2)
        self.assertNotIn("provider unavailable", str(error.exception))

    async def test_default_provider_failure_preserves_legacy_fallback(self):
        self.create.side_effect = RuntimeError("provider unavailable")
        self.assertEqual(await self.services.correct_grammar("I am happy."), (None, None))
        self.assertEqual(self.create.await_count, 2)

    async def test_default_invalid_response_preserves_retry_feedback(self):
        self.create.return_value = completion("{}")
        correction, explanation = await self.services.correct_grammar("I am happy.")
        self.assertIsNone(correction)
        self.assertEqual(explanation, "No pudimos evaluar tu respuesta. Intentá de nuevo.")

    async def test_strict_is_keyword_only(self):
        with self.assertRaises(TypeError):
            await self.services.correct_grammar("I am happy.", True)
        self.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
