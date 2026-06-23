"""Fluentia AI — Golden set evaluation (F4).

Runs ~20 test cases against the live backend endpoints.
Can be invoked as a script or imported by the /eval/run endpoint.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"


def _make_wav_header(data_size: int, sample_rate: int = 16000) -> bytes:
    import struct
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header


def _generate_silence_wav(duration_seconds: float = 1.0) -> bytes:
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    data = b"\x00\x00" * num_samples
    return _make_wav_header(len(data)) + data


def _generate_tone_wav(frequency: float = 440.0, duration_seconds: float = 2.0) -> bytes:
    import math
    import struct
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    samples = []
    for i in range(num_samples):
        value = int(16000 * math.sin(2 * math.pi * frequency * i / sample_rate))
        samples.append(struct.pack("<h", value))
    data = b"".join(samples)
    return _make_wav_header(len(data)) + data


# --- Test case definitions ---


GRAMMAR_CASES = [
    {
        "name": "grammar_correct_sentence",
        "input": "I like to eat pizza and drink coffee.",
        "expect_correction_null": True,
    },
    {
        "name": "grammar_age_error",
        "input": "I have 25 years.",
        "expect_correction_null": False,
    },
    {
        "name": "grammar_agree_error",
        "input": "I am agree with you.",
        "expect_correction_null": False,
    },
    {
        "name": "grammar_past_error",
        "input": "Yesterday I go to the store.",
        "expect_correction_null": False,
    },
    {
        "name": "grammar_complex_correct",
        "input": "If I had more time, I would travel around the world.",
        "expect_correction_null": True,
    },
]

PRONUNCIATION_CASES = [
    {
        "name": "pronunciation_thought",
        "words": ["thought"],
    },
    {
        "name": "pronunciation_world",
        "words": ["world"],
    },
    {
        "name": "pronunciation_schedule",
        "words": ["schedule", "comfortable"],
    },
]

PLACEMENT_CASES = [
    {"name": "placement_a1", "text": "Hello, my name is Carlos.", "level": "A1", "expected_score_min": 60},
    {"name": "placement_a2", "text": "Yesterday I went to the supermarket.", "level": "A2", "expected_score_min": 50},
    {"name": "placement_b1", "text": "If I had more free time, I would travel.", "level": "B1", "expected_score_min": 40},
    {"name": "placement_b2", "text": "Despite the challenges, the company managed to increase revenue.", "level": "B2", "expected_score_min": 30},
]

SCENARIO_CASES = [
    {"name": "scenario_smalltalk_greeting", "scenario_id": "smalltalk", "user_text": "Hi! I'm doing great, thank you. The weather is nice today."},
    {"name": "scenario_cafe_order", "scenario_id": "cafe", "user_text": "I would like a large coffee with milk, please."},
    {"name": "scenario_shopping_ask", "scenario_id": "shopping", "user_text": "I'm looking for a blue t-shirt, size medium."},
    {"name": "scenario_directions_ask", "scenario_id": "directions", "user_text": "Excuse me, can you tell me how to get to the train station?"},
    {"name": "scenario_plans_propose", "scenario_id": "plans", "user_text": "How about we go to the movies on Saturday afternoon?"},
]

TRANSCRIPTION_CASES = [
    {"name": "transcription_silence", "audio_type": "silence"},
    {"name": "transcription_tone", "audio_type": "tone"},
]


# --- Test runners ---


async def _test_grammar(client: httpx.AsyncClient, base_url: str) -> list[dict]:
    results = []
    for case in GRAMMAR_CASES:
        try:
            from services import correct_grammar
            correction, explanation = await correct_grammar(case["input"])
            passed = True
            reason = ""

            if case["expect_correction_null"]:
                if correction is not None:
                    passed = False
                    reason = f"Expected no correction but got: {correction}"
            else:
                if correction is None:
                    passed = False
                    reason = "Expected a correction but got null"

            if explanation and passed:
                from guardrails import _detect_language
                lang = _detect_language(explanation)
                if lang == "en":
                    passed = False
                    reason = f"Explanation is in English: {explanation[:100]}"

            results.append({"case": case["name"], "passed": passed, "reason": reason})
        except Exception as e:
            results.append({"case": case["name"], "passed": False, "reason": str(e)})
    return results


async def _test_pronunciation(client: httpx.AsyncClient, base_url: str) -> list[dict]:
    results = []
    for case in PRONUNCIATION_CASES:
        try:
            from services import get_pronunciation_guides
            guides = await get_pronunciation_guides(case["words"])
            passed = True
            reason = ""

            if not guides:
                passed = False
                reason = "No guides returned"
            else:
                for g in guides:
                    if "word" not in g or "pronunciation" not in g:
                        passed = False
                        reason = f"Guide missing fields: {g}"
                        break
                    from guardrails import _IPA_CHARS
                    if any(c in _IPA_CHARS for c in g["pronunciation"]):
                        passed = False
                        reason = f"IPA detected in pronunciation: {g['pronunciation']}"
                        break

            results.append({"case": case["name"], "passed": passed, "reason": reason})
        except Exception as e:
            results.append({"case": case["name"], "passed": False, "reason": str(e)})
    return results


async def _test_placement(client: httpx.AsyncClient, base_url: str) -> list[dict]:
    results = []
    for case in PLACEMENT_CASES:
        try:
            from pedagogy import PLACEMENT_ITEMS
            from services import deepseek_client
            import json as _json

            item = next(i for i in PLACEMENT_ITEMS if i["level"] == case["level"])

            prompt = f'Eres un evaluador de nivel CEFR de inglés para hispanohablantes.\nEl estudiante intentó decir esta frase de nivel {case["level"]}: "{item["phrase"]}"\nLo que realmente dijo fue: "{case["text"]}"\n\nEvalúa de 0 a 100.\n\nResponde SOLO en JSON:\n{{"score": 75, "feedback": "explicación breve en español"}}'

            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            data = _json.loads(text)
            score = data.get("score", 0)

            passed = True
            reason = ""
            if score < case["expected_score_min"]:
                passed = False
                reason = f"Score {score} below minimum {case['expected_score_min']}"

            results.append({"case": case["name"], "passed": passed, "reason": reason})
        except Exception as e:
            results.append({"case": case["name"], "passed": False, "reason": str(e)})
    return results


async def _test_scenarios(client: httpx.AsyncClient, base_url: str) -> list[dict]:
    results = []
    for case in SCENARIO_CASES:
        try:
            from scenarios import start_scenario, _sessions, SCENARIOS as SCENARIO_DEFS, deepseek_client, _build_system_prompt
            import json as _json

            result = start_scenario(case["scenario_id"])
            if not result:
                results.append({"case": case["name"], "passed": False, "reason": "Failed to start scenario"})
                continue

            session_id, _ = result
            session = _sessions[session_id]
            scenario = SCENARIO_DEFS[case["scenario_id"]]

            session.messages.append({"role": "user", "content": case["user_text"]})
            session.turn_count += 1

            system_prompt = _build_system_prompt(scenario, session.objectives_met)

            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *session.messages,
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            data = _json.loads(text)

            from guardrails import validate_scenario_turn
            valid, reason = validate_scenario_turn(data)

            del _sessions[session_id]

            results.append({"case": case["name"], "passed": valid, "reason": reason if not valid else ""})
        except Exception as e:
            results.append({"case": case["name"], "passed": False, "reason": str(e)})
    return results


async def _test_transcription(client: httpx.AsyncClient, base_url: str) -> list[dict]:
    results = []
    for case in TRANSCRIPTION_CASES:
        try:
            if case["audio_type"] == "silence":
                audio = _generate_silence_wav(1.0)
            else:
                audio = _generate_tone_wav(440.0, 2.0)

            response = await client.post(
                f"{base_url}/assess",
                files={"audio": ("test.wav", audio, "audio/wav")},
                timeout=30.0,
            )

            passed = True
            reason = ""

            if case["audio_type"] == "silence":
                if response.status_code == 200:
                    data = response.json()
                    if data.get("transcription", "").strip():
                        passed = False
                        reason = f"Should detect empty audio but got: {data['transcription'][:50]}"
            else:
                if response.status_code not in (200, 502):
                    passed = False
                    reason = f"Unexpected status {response.status_code}"

            results.append({"case": case["name"], "passed": passed, "reason": reason})
        except Exception as e:
            results.append({"case": case["name"], "passed": False, "reason": str(e)})
    return results


# --- Main runner ---


async def run_golden_set(base_url: str = DEFAULT_BASE_URL) -> dict:
    start = time.time()
    all_results = []

    async with httpx.AsyncClient() as client:
        grammar_results = await _test_grammar(client, base_url)
        all_results.extend(grammar_results)

        pron_results = await _test_pronunciation(client, base_url)
        all_results.extend(pron_results)

        placement_results = await _test_placement(client, base_url)
        all_results.extend(placement_results)

        scenario_results = await _test_scenarios(client, base_url)
        all_results.extend(scenario_results)

        transcription_results = await _test_transcription(client, base_url)
        all_results.extend(transcription_results)

    duration = time.time() - start
    passed = sum(1 for r in all_results if r["passed"])
    failed = sum(1 for r in all_results if not r["passed"])
    failures = [r for r in all_results if not r["passed"]]

    return {
        "total": len(all_results),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(all_results) * 100, 1) if all_results else 0,
        "duration_seconds": round(duration, 1),
        "failures": failures,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    print(f"Running golden set against {base}...")
    result = asyncio.run(run_golden_set(base))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["failed"] == 0 else 1)
