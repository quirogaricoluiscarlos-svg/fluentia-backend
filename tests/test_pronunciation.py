"""Contract fixtures and mocked Azure requests; no provider calls.

Flat fixture follows the REST response documented by Microsoft:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text-short
Nested fields follow the documented SDK representation; the REST transport
still requires RecognitionStatus='Success'.
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment
"""

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
with patch.object(sys, "path", [str(BACKEND_DIR), *sys.path]):
    from pronunciation import parse_pronunciation_response
with patch.object(sys, "path", [str(Path(__file__).resolve().parent), *sys.path]):
    from test_grammar import load_services


REST_FIXTURE = {
    "RecognitionStatus": "Success",
    "DisplayText": "Good morning.",
    "NBest": [{
        "Display": "Good morning.",
        "AccuracyScore": 100.0,
        "FluencyScore": 100.0,
        "CompletenessScore": 100.0,
        "ProsodyScore": 87.8,
        "PronScore": 95.1,
        "Words": [
            {"Word": "good", "AccuracyScore": 100.0, "ErrorType": "None"},
            {"Word": "morning", "AccuracyScore": 100.0, "ErrorType": "None"},
        ],
    }],
}


class PronunciationParserTests(unittest.TestCase):
    def test_documented_rest_fields_are_read_without_zero_defaults(self):
        result = parse_pronunciation_response(REST_FIXTURE)
        self.assertEqual(result["accuracy_score"], 100.0)
        self.assertEqual(result["fluency_score"], 100.0)
        self.assertEqual(result["completeness_score"], 100.0)
        self.assertEqual(result["prosody_score"], 87.8)
        self.assertAlmostEqual(result["overall_score"], 97.0, places=1)
        self.assertEqual(result["words"][0], {
            "word": "good", "accuracy_score": 100.0, "error_type": None,
        })

    def test_missing_prosody_is_null_and_excluded_from_mean(self):
        data = deepcopy(REST_FIXTURE)
        del data["NBest"][0]["ProsodyScore"]
        result = parse_pronunciation_response(data)
        self.assertIsNone(result["prosody_score"])
        self.assertEqual(result["overall_score"], 100.0)

    def test_explicit_null_prosody_is_unavailable(self):
        data = deepcopy(REST_FIXTURE)
        data["NBest"][0]["ProsodyScore"] = None
        result = parse_pronunciation_response(data)
        self.assertIsNone(result["prosody_score"])
        self.assertEqual(result["overall_score"], 100.0)

    def test_real_zero_prosody_is_included_in_mean(self):
        data = deepcopy(REST_FIXTURE)
        data["NBest"][0]["ProsodyScore"] = 0.0
        result = parse_pronunciation_response(data)
        self.assertEqual(result["prosody_score"], 0.0)
        self.assertEqual(result["overall_score"], 75.0)

    def test_documented_nested_metrics_and_word_errors_are_supported(self):
        data = {
            "RecognitionStatus": "Success",
            "NBest": [{
                "PronunciationAssessment": {
                    "AccuracyScore": 80,
                    "FluencyScore": 75,
                    "CompletenessScore": 90,
                },
                "Words": [{
                    "Word": "hello",
                    "PronunciationAssessment": {
                        "AccuracyScore": 65,
                        "ErrorType": "Mispronunciation",
                    },
                }],
            }],
        }
        result = parse_pronunciation_response(data)
        self.assertEqual(result["overall_score"], 81.7)
        self.assertIsNone(result["prosody_score"])
        self.assertEqual(result["words"][0]["error_type"], "Mispronunciation")

    def test_unsuccessful_or_missing_recognition_is_not_evaluable(self):
        for status in [None, "NoMatch", "InitialSilenceTimeout", "BabbleTimeout", "Error", 0, True]:
            with self.subTest(status=status):
                data = deepcopy(REST_FIXTURE)
                data["RecognitionStatus"] = status
                self.assertIsNone(parse_pronunciation_response(data))

    def test_incomplete_required_metrics_are_not_evaluable(self):
        for field in ["AccuracyScore", "FluencyScore", "CompletenessScore"]:
            with self.subTest(field=field):
                data = deepcopy(REST_FIXTURE)
                del data["NBest"][0][field]
                self.assertIsNone(parse_pronunciation_response(data))

    def test_non_numeric_non_finite_and_out_of_range_metrics_are_rejected(self):
        invalid = [-1, 101, float("nan"), float("inf"), float("-inf"), True, False, "100", [], {}]
        for field in ["AccuracyScore", "FluencyScore", "CompletenessScore", "ProsodyScore"]:
            for value in invalid:
                with self.subTest(field=field, value=value):
                    data = deepcopy(REST_FIXTURE)
                    data["NBest"][0][field] = value
                    self.assertIsNone(parse_pronunciation_response(data))

    def test_malformed_response_or_nbest_is_not_evaluable(self):
        for data in [None, [], {}, {"RecognitionStatus": "Success"}]:
            with self.subTest(data=data):
                self.assertIsNone(parse_pronunciation_response(data))
        for n_best in [None, [], {}, [None], ["invalid"]]:
            with self.subTest(n_best=n_best):
                self.assertIsNone(parse_pronunciation_response({
                    "RecognitionStatus": "Success", "NBest": n_best,
                }))

    def test_empty_or_malformed_words_are_not_evaluable(self):
        for words in [None, [], {}, [None], ["hello"]]:
            with self.subTest(words=words):
                data = deepcopy(REST_FIXTURE)
                data["NBest"][0]["Words"] = words
                self.assertIsNone(parse_pronunciation_response(data))
        for word in [
            {"Word": "good"},
            {"Word": "", "AccuracyScore": 100},
            {"Word": "   ", "AccuracyScore": 100},
            {"Word": 100, "AccuracyScore": 100},
            {"AccuracyScore": 100},
            {"Word": "good", "AccuracyScore": None},
            {"Word": "good", "AccuracyScore": float("nan")},
            {"Word": "good", "AccuracyScore": 101},
            {"Word": "good", "AccuracyScore": True},
            {"Word": "good", "AccuracyScore": 100, "ErrorType": True},
            {"Word": "good", "AccuracyScore": 100, "ErrorType": "unknown"},
        ]:
            with self.subTest(word=word):
                data = deepcopy(REST_FIXTURE)
                data["NBest"][0]["Words"] = [word]
                self.assertIsNone(parse_pronunciation_response(data))

    def test_zero_and_miscue_scores_are_preserved(self):
        data = deepcopy(REST_FIXTURE)
        best = data["NBest"][0]
        best.update({"AccuracyScore": 0, "FluencyScore": 0, "CompletenessScore": 0})
        del best["ProsodyScore"]
        best["Words"] = [{"Word": "good", "AccuracyScore": 0, "ErrorType": "Omission"}]
        result = parse_pronunciation_response(data)
        self.assertEqual(result["overall_score"], 0.0)
        self.assertEqual(result["words"][0]["accuracy_score"], 0.0)
        self.assertEqual(result["words"][0]["error_type"], "Omission")

    def test_incomplete_flat_fields_do_not_fall_back_to_conflicting_nested_fields(self):
        data = deepcopy(REST_FIXTURE)
        best = data["NBest"][0]
        best["PronunciationAssessment"] = {
            "AccuracyScore": 100, "FluencyScore": 100, "CompletenessScore": 100,
        }
        del best["AccuracyScore"]
        self.assertIsNone(parse_pronunciation_response(data))


class AzureRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.services = load_services()
        self.response = SimpleNamespace(status_code=200, json=Mock(return_value=REST_FIXTURE))
        self.post = AsyncMock(return_value=self.response)
        self.http = SimpleNamespace(post=self.post)
        self.context = Mock()
        self.context.__aenter__ = AsyncMock(return_value=self.http)
        self.context.__aexit__ = AsyncMock(return_value=False)
        self.services.httpx.AsyncClient = Mock(return_value=self.context)

    async def test_reference_and_miscue_are_sent_to_azure(self):
        with patch.dict(os.environ, {"AZURE_SPEECH_KEY": "unit-test-only", "AZURE_SPEECH_REGION": "eastus"}):
            result = await self.services.assess_pronunciation(b"fake-wav", "Good morning.")
        self.assertIsNotNone(result)
        args = self.post.await_args
        self.assertIn("language=en-US&format=detailed", args.args[0])
        self.assertEqual(args.kwargs["content"], b"fake-wav")
        config = json.loads(base64.b64decode(args.kwargs["headers"]["Pronunciation-Assessment"]))
        self.assertEqual(config["ReferenceText"], "Good morning.")
        self.assertTrue(config["EnableMiscue"])
        self.assertEqual(config["Dimension"], "Comprehensive")
        self.assertEqual(config["GradingSystem"], "HundredMark")

    async def test_missing_key_or_empty_reference_skips_provider(self):
        with patch.dict(os.environ, {"AZURE_SPEECH_KEY": ""}):
            self.assertIsNone(await self.services.assess_pronunciation(b"fake-wav", "Good morning."))
        with patch.dict(os.environ, {"AZURE_SPEECH_KEY": "unit-test-only"}):
            self.assertIsNone(await self.services.assess_pronunciation(b"fake-wav", "   "))
        self.post.assert_not_awaited()

    async def test_http_failure_returns_unavailable(self):
        self.response.status_code = 503
        self.response.text = "Service unavailable"
        with patch.dict(os.environ, {"AZURE_SPEECH_KEY": "unit-test-only"}):
            self.assertIsNone(await self.services.assess_pronunciation(b"fake-wav", "Good morning."))

    async def test_invalid_json_or_network_failure_returns_unavailable(self):
        with patch.dict(os.environ, {"AZURE_SPEECH_KEY": "unit-test-only"}):
            self.response.json.side_effect = ValueError("invalid JSON")
            self.assertIsNone(await self.services.assess_pronunciation(b"fake-wav", "Good morning."))
            self.post.side_effect = RuntimeError("network unavailable")
            self.assertIsNone(await self.services.assess_pronunciation(b"fake-wav", "Good morning."))


if __name__ == "__main__":
    unittest.main()
