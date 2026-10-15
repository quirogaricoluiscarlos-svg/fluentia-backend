"""Local HTTP contract regression tests. Every provider and DB call is mocked."""

import importlib.util
import json
import re
import sys
import io
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from practice_contract import ACTIVITIES, normalize_phrase, resolve_activity
from security import Principal


def valid_wav():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return stream.getvalue()


def module_with(name, functions):
    module = ModuleType(name)
    for function in functions:
        setattr(module, function, AsyncMock())
    return module


class ReferenceTests(unittest.TestCase):
    def test_catalog_matches_all_android_activities(self):
        source = BACKEND.parent / "app/src/main/java/com/fluentia/ai/presentation/home/HomeViewModel.kt"
        if not source.exists():
            self.skipTest("Standalone backend checkout; Android curriculum is unavailable")
        entries = re.findall(
            r'PracticeActivity\("([^"\n]+)",\s*"(?:[^"\\]|\\.)*",\s*"((?:[^"\\]|\\.)*)"',
            source.read_text(encoding="utf-8"),
        )
        expected = {key: json.loads('"' + phrase + '"') for key, phrase in entries}
        expected["f1"] = None
        self.assertEqual(len(entries), 81)
        self.assertEqual(ACTIVITIES, expected)

    def test_case_punctuation_apostrophes_contractions_and_numbers(self):
        self.assertEqual(normalize_phrase("I’m 3 years old!"), normalize_phrase("I am three years old."))
        self.assertEqual(normalize_phrase("I can't go."), normalize_phrase("I cannot go."))
        self.assertEqual(normalize_phrase("It doesn't fit."), normalize_phrase("It does not fit."))
        self.assertNotEqual(normalize_phrase("I can go."), normalize_phrase("I cannot go."))
        self.assertNotEqual(normalize_phrase("How are you?"), normalize_phrase("How are you doing today?"))

    def test_canonical_reference_and_free_instruction(self):
        self.assertEqual(resolve_activity("g2", "HOW ARE YOU DOING TODAY!"), ("g2", ACTIVITIES["g2"]))
        self.assertEqual(resolve_activity("f1", None), ("f1", None))

    def test_spoken_gate_and_doctor_aliases_match_the_same_reference(self):
        self.assertEqual(normalize_phrase("gate B12"), normalize_phrase("gate B twelve"))
        self.assertEqual(normalize_phrase("Doctor Smith"), normalize_phrase("Dr. Smith"))

    def test_unknown_missing_or_mismatched_activity_is_rejected(self):
        for activity_id, phrase in [(None, None), ("unknown", "Hello"), ("g2", None),
                                    ("g2", ACTIVITIES["g1"]), ("f1", "Say anything")]:
            with self.subTest(activity_id=activity_id, phrase=phrase), self.assertRaises(ValueError):
                resolve_activity(activity_id, phrase)


class PracticeHttpTests(unittest.TestCase):
    def setUp(self):
        self.services = module_with("services", (
            "transcribe_audio", "correct_grammar", "assess_pronunciation",
            "get_pronunciation_guides", "generate_tts",
        ))
        self.services.transcribe_audio.return_value = ACTIVITIES["g2"]
        self.services.correct_grammar.return_value = (None, "Tu oración es correcta.")
        self.pronunciation = {
            "accuracy_score": 100, "fluency_score": 100, "completeness_score": 100,
            "prosody_score": None, "overall_score": 100,
            "words": [{"word": "How", "accuracy_score": 100, "error_type": None}],
        }
        self.services.assess_pronunciation.return_value = self.pronunciation
        dotenv = ModuleType("dotenv")
        dotenv.load_dotenv = Mock()  # Never read the real .env during tests.
        pedagogy = module_with("pedagogy", (
            "check_level_progression", "determine_level", "evaluate_placement_item",
            "get_placement_item", "get_placement_items", "get_session_goals",
        ))
        pedagogy.CURRICULUM = {}
        dependencies = {
            "services": self.services, "dotenv": dotenv, "pedagogy": pedagogy,
            "scenarios": module_with("scenarios", ("end_scenario", "list_scenarios", "process_turn", "start_scenario")),
            "quality_log": module_with("quality_log", ("cleanup_old_records", "get_metrics")),
            "test_quality": module_with("test_quality", ("run_golden_set",)),
        }
        spec = importlib.util.spec_from_file_location("practice_api_under_test", BACKEND / "main.py")
        main = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(main)
        self.main = main
        self.dependencies = dependencies
        self.audio = valid_wav()
        main.app.dependency_overrides[main.authenticate] = lambda: Principal("test-guest", True)
        @asynccontextmanager
        async def no_provider_budget(*args, **kwargs):
            yield
        # This suite tests evaluation, not quotas; the security integration suite
        # separately uses a real quota store backed by a temporary database.
        main.provider_access = no_provider_budget
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def assess(self, data=None):
        if data is None:
            data = {"activity_id": "g2", "target_phrase": ACTIVITIES["g2"]}
        return self.client.post("/assess", data=data, files={"audio": ("audio.wav", self.audio, "audio/wav")})

    def test_complete_assessment_uses_reference_and_strict_grammar(self):
        self.services.transcribe_audio.return_value = "HOW ARE YOU DOING TODAY!"
        response = self.assess()
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["evaluation_status"], "complete")
        self.assertEqual(data["grammar_status"], "evaluated")
        self.assertEqual(data["activity_id"], "g2")
        self.assertEqual(data["target_phrase"], ACTIVITIES["g2"])
        self.assertIsNone(data["pronunciation"]["prosody_score"])
        self.assertEqual(data["pronunciation"]["overall_score"], 100)
        self.services.assess_pronunciation.assert_awaited_once_with(self.audio, ACTIVITIES["g2"])
        self.services.correct_grammar.assert_awaited_once_with("HOW ARE YOU DOING TODAY!", strict=True)

    def test_missing_activity_is_rejected_before_any_provider_call(self):
        self.assertEqual(self.assess({}).status_code, 422)
        self.services.transcribe_audio.assert_not_awaited()

    def test_unknown_activity_or_wrong_reference_is_rejected_before_transcription(self):
        for data in [{"activity_id": "missing", "target_phrase": "Hello"},
                     {"activity_id": "g2", "target_phrase": ACTIVITIES["g1"]},
                     {"activity_id": "g2"}, {"activity_id": "f1", "target_phrase": "instruction"}]:
            with self.subTest(data=data):
                self.assertEqual(self.assess(data).status_code, 422)
        self.services.transcribe_audio.assert_not_awaited()

    def test_silence_is_explicitly_not_evaluable(self):
        self.services.transcribe_audio.return_value = "  "
        response = self.assess()
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["evaluation_status"], "no_speech")
        self.assertEqual(data["grammar_status"], "unavailable")
        self.assertIsNone(data["pronunciation"])
        self.assertTrue(data["feedback"])
        self.services.correct_grammar.assert_not_awaited()
        self.services.assess_pronunciation.assert_not_awaited()

    def test_other_phrase_cannot_complete_the_requested_activity(self):
        self.services.transcribe_audio.return_value = ACTIVITIES["g1"]
        data = self.assess().json()
        self.assertEqual(data["evaluation_status"], "off_target")
        self.assertIsNone(data["pronunciation"])
        self.services.correct_grammar.assert_not_awaited()
        self.services.assess_pronunciation.assert_not_awaited()

    def test_partial_phrase_cannot_complete_the_requested_activity(self):
        self.services.transcribe_audio.return_value = "How are you?"
        self.assertEqual(self.assess().json()["evaluation_status"], "off_target")

    def test_spoken_catalog_aliases_do_not_reject_correct_practice(self):
        for activity_id, spoken in [("t1", ACTIVITIES["t1"].replace("B12", "B twelve")),
                                    ("d6", ACTIVITIES["d6"].replace("Doctor", "Dr."))]:
            with self.subTest(activity_id=activity_id):
                self.services.transcribe_audio.return_value = spoken
                response = self.assess({"activity_id": activity_id, "target_phrase": ACTIVITIES[activity_id]})
                self.assertEqual(response.json()["evaluation_status"], "complete")

    def test_free_practice_evaluates_speech_instead_of_the_instruction(self):
        self.services.transcribe_audio.return_value = "I enjoy learning English."
        data = self.assess({"activity_id": "f1"}).json()
        self.assertEqual(data["evaluation_status"], "complete")
        self.assertIsNone(data["target_phrase"])
        self.services.assess_pronunciation.assert_awaited_once_with(self.audio, "I enjoy learning English.")

    def test_pronunciation_outage_keeps_grammar_feedback_but_is_partial(self):
        self.services.assess_pronunciation.return_value = None
        data = self.assess().json()
        self.assertEqual(data["evaluation_status"], "partial")
        self.assertEqual(data["grammar_status"], "evaluated")
        self.assertEqual(data["explanation"], "Tu oración es correcta.")
        self.assertIsNone(data["pronunciation"])

    def test_grammar_outage_keeps_pronunciation_but_never_perfect_grammar(self):
        self.services.correct_grammar.side_effect = RuntimeError("mock evaluator outage")
        data = self.assess().json()
        self.assertEqual(data["evaluation_status"], "partial")
        self.assertEqual(data["grammar_status"], "unavailable")
        self.assertIsNone(data["explanation"])
        self.assertIsNotNone(data["pronunciation"])

    def test_both_evaluators_unavailable_is_explicit(self):
        self.services.correct_grammar.side_effect = RuntimeError("mock outage")
        self.services.assess_pronunciation.side_effect = RuntimeError("mock outage")
        data = self.assess().json()
        self.assertEqual(data["evaluation_status"], "unavailable")
        self.assertIsNone(data["pronunciation"])

    def test_absent_or_empty_grammar_feedback_never_passes(self):
        for result in [(None, None), (None, ""), ("", "la oración"), (None, 100)]:
            with self.subTest(result=result):
                self.services.correct_grammar.return_value = result
                data = self.assess().json()
                self.assertEqual(data["evaluation_status"], "partial")
                self.assertEqual(data["grammar_status"], "unavailable")

    def test_invalid_pronunciation_schema_does_not_create_zero_scores(self):
        for result in [{}, {**self.pronunciation, "fluency_score": None},
                       {**self.pronunciation, "overall_score": float("nan")},
                       {**self.pronunciation, "accuracy_score": True},
                       {**self.pronunciation, "fluency_score": "100"},
                       {**self.pronunciation, "words": []}]:
            with self.subTest(result=result):
                self.services.assess_pronunciation.return_value = result
                data = self.assess().json()
                self.assertEqual(data["evaluation_status"], "partial")
                self.assertIsNone(data["pronunciation"])

    def test_optional_guide_failure_does_not_invalidate_evaluation(self):
        self.services.assess_pronunciation.return_value = {
            **self.pronunciation, "words": [{"word": "How", "accuracy_score": 50}],
        }
        self.services.get_pronunciation_guides.side_effect = RuntimeError("mock coaching outage")
        self.assertEqual(self.assess().json()["evaluation_status"], "complete")

    def test_transcription_outage_returns_http_error(self):
        self.services.transcribe_audio.side_effect = RuntimeError("mock transcription outage")
        self.assertEqual(self.assess().status_code, 502)
        self.services.correct_grammar.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
