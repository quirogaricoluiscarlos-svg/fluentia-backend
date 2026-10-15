"""Validate Azure short-audio pronunciation results without inventing scores.

REST contract: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text-short
The documented SDK result also places scores under PronunciationAssessment:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment
"""

import math


_SCORE_FIELDS = {"AccuracyScore", "FluencyScore", "CompletenessScore", "ProsodyScore"}
_WORD_ERRORS = {"None", "Omission", "Insertion", "Mispronunciation"}


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= 100 or not math.isfinite(value):
        return None
    return float(value)


def _assessment_fields(result: dict) -> dict | None:
    # REST scores are flat. Support nested scores without filling missing flat
    # fields from another representation of the same result.
    if _SCORE_FIELDS.intersection(result) or "ErrorType" in result:
        return result
    nested = result.get("PronunciationAssessment")
    return nested if isinstance(nested, dict) else None


def parse_pronunciation_response(data: object) -> dict | None:
    """Return a complete assessment, or None when it cannot be evaluated.

    Accuracy, fluency, completeness and word accuracy are required. Prosody is
    optional, and stays None when Azure did not assess it. overall_score is the
    app's arithmetic mean, distinct from Azure's weighted PronScore.
    """
    if not isinstance(data, dict) or data.get("RecognitionStatus") != "Success":
        return None
    n_best = data.get("NBest")
    if not isinstance(n_best, list) or not n_best or not isinstance(n_best[0], dict):
        return None
    best = n_best[0]
    assessment = _assessment_fields(best)
    if assessment is None:
        return None

    accuracy = _score(assessment.get("AccuracyScore"))
    fluency = _score(assessment.get("FluencyScore"))
    completeness = _score(assessment.get("CompletenessScore"))
    if accuracy is None or fluency is None or completeness is None:
        return None
    raw_prosody = assessment.get("ProsodyScore")
    prosody = _score(raw_prosody) if raw_prosody is not None else None
    if raw_prosody is not None and prosody is None:
        return None

    raw_words = best.get("Words")
    if not isinstance(raw_words, list) or not raw_words:
        return None
    words = []
    for word in raw_words:
        if not isinstance(word, dict):
            return None
        text = word.get("Word")
        word_assessment = _assessment_fields(word)
        if not isinstance(text, str) or not text.strip() or word_assessment is None:
            return None
        word_accuracy = _score(word_assessment.get("AccuracyScore"))
        error = word_assessment.get("ErrorType")
        if word_accuracy is None:
            return None
        if error is not None and (not isinstance(error, str) or error not in _WORD_ERRORS):
            return None
        words.append({
            "word": text,
            "accuracy_score": word_accuracy,
            "error_type": None if error == "None" else error,
        })

    scores = [accuracy, fluency, completeness]
    if prosody is not None:
        scores.append(prosody)
    return {
        "accuracy_score": accuracy,
        "fluency_score": fluency,
        "completeness_score": completeness,
        "prosody_score": prosody,
        "overall_score": round(sum(scores) / len(scores), 1),
        "words": words,
    }
