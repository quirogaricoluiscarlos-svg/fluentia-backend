"""Fluentia AI — Guardrails for AI responses (F4)."""

import re

_SPANISH_MARKERS = {
    "el", "la", "los", "las", "es", "en", "de", "que", "un", "una",
    "por", "para", "con", "como", "pero", "esta", "este", "muy",
    "bien", "tu", "su", "del", "al", "se", "no", "más", "ya",
    "tiene", "puede", "hace", "está", "son", "hay", "fue", "ser",
    "oración", "correcto", "perfecto", "error", "verbo",
}

_ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "have", "has", "had",
    "that", "this", "with", "for", "but", "not", "you", "all",
    "can", "her", "one", "our", "out", "day", "get", "make",
    "like", "just", "over", "such", "take", "than", "them",
    "good", "job", "great", "well", "nice", "your", "would",
    "could", "should", "very", "also", "some", "been", "about",
    "sentence", "correct", "wrong", "right", "word", "verb",
}

_IPA_CHARS = set("əðθʃʒŋɪʊæɑɔɛʌːˈˌ")


def _detect_language(text: str) -> str:
    words = set(re.findall(r"[a-záéíóúüñ]+", text.lower()))
    es_count = len(words & _SPANISH_MARKERS)
    en_count = len(words & _ENGLISH_MARKERS)
    if es_count > en_count:
        return "es"
    if en_count > es_count:
        return "en"
    return "unknown"


def validate_correction(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Response is not a dict"
    if "explanation" not in data:
        return False, "Missing 'explanation' field"
    if "correction" not in data:
        return False, "Missing 'correction' field"
    explanation = data.get("explanation") or ""
    if len(explanation) > 500:
        return False, f"Explanation too long: {len(explanation)} chars"
    if explanation and _detect_language(explanation) == "en":
        return False, "Explanation appears to be in English, expected Spanish"
    return True, "ok"


def validate_pronunciation_guides(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Response is not a dict"
    guides = data.get("guides")
    if not isinstance(guides, list):
        return False, "Missing or invalid 'guides' array"
    for i, g in enumerate(guides):
        if not isinstance(g, dict):
            return False, f"Guide {i} is not a dict"
        if "word" not in g or "pronunciation" not in g:
            return False, f"Guide {i} missing 'word' or 'pronunciation'"
        pron = g["pronunciation"]
        if any(c in _IPA_CHARS for c in pron):
            return False, f"Guide for '{g['word']}' contains IPA characters"
    return True, "ok"


def validate_placement_eval(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Response is not a dict"
    score = data.get("score")
    if not isinstance(score, (int, float)) or score < 0 or score > 100:
        return False, f"Invalid score: {score}"
    feedback = data.get("feedback", "")
    if len(feedback) > 300:
        return False, f"Feedback too long: {len(feedback)} chars"
    if feedback and _detect_language(feedback) == "en":
        return False, "Feedback appears to be in English, expected Spanish"
    return True, "ok"


def validate_scenario_turn(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Response is not a dict"
    ai_response = data.get("ai_response")
    if not ai_response or not isinstance(ai_response, str):
        return False, "Missing or empty 'ai_response'"
    if len(ai_response) > 200:
        return False, f"ai_response too long: {len(ai_response)} chars"
    objectives = data.get("objectives_met")
    if not isinstance(objectives, list):
        return False, "'objectives_met' is not a list"
    explanation = data.get("explanation", "")
    if explanation and _detect_language(explanation) == "en":
        return False, "Explanation appears to be in English, expected Spanish"
    return True, "ok"


def validate_transcription(text: str) -> tuple[bool, str]:
    if not text or len(text.strip()) < 2:
        return False, "Transcription is empty or too short"
    words = text.strip().split()
    if len(words) >= 50:
        unique = set(w.lower() for w in words)
        if len(unique) <= 2:
            return False, "Whisper hallucination detected: repeated words"
    return True, "ok"
