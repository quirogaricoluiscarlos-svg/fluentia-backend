"""Practice identity and response contract. No network/provider dependencies."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


ACTIVITIES = json.loads(Path(__file__).with_name("practice_catalog.json").read_text(encoding="utf-8"))


def normalize_phrase(text: str) -> str:
    """Ignore punctuation/case and common spoken contractions, not missing words."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"\bdr\.?\s", "doctor ", text)
    # STT can write spoken flight gates as B12 or B twelve.
    text = re.sub(r"\b([a-z])(\d+)\b", r"\1 \2", text)
    expansions = {
        "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
        "it's": "it is", "that's": "that is", "there's": "there is", "what's": "what is",
        "he's": "he is", "she's": "she is", "let's": "let us", "i'd": "i would",
        "i've": "i have", "we've": "we have", "i'll": "i will", "you'll": "you will",
        "can't": "cannot", "can not": "cannot", "won't": "will not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "haven't": "have not", "hasn't": "has not", "wouldn't": "would not",
        "couldn't": "could not", "shouldn't": "should not",
    }
    for contraction, expanded in expansions.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    numbers = dict(zip(map(str, range(21)), (
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    ).split()))
    return " ".join(numbers.get(word, word) for word in re.findall(r"[a-z0-9]+", text))


def resolve_activity(activity_id: str | None, target_phrase: str | None) -> tuple[str | None, str | None]:
    """Resolve a canonical reference before spending any provider calls."""
    if not activity_id:
        raise ValueError("Selecciona una actividad antes de grabar.")
    if activity_id not in ACTIVITIES:
        raise ValueError("La actividad no existe. Vuelve a abrir el ejercicio.")
    reference = ACTIVITIES[activity_id]
    if reference is None:  # f1 is an instruction, not a phrase to repeat.
        if target_phrase:
            raise ValueError("La práctica libre no requiere una frase objetivo.")
        return activity_id, None
    if not target_phrase or normalize_phrase(target_phrase) != normalize_phrase(reference):
        raise ValueError("La frase no corresponde a la actividad. Vuelve a abrir el ejercicio.")
    return activity_id, reference


class PronunciationGuide(BaseModel):
    word: str
    pronunciation: str
    tip: str | None = None


class WordScore(BaseModel):
    word: str = Field(min_length=1)
    accuracy_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    error_type: str | None = None
    pronunciation_guide: PronunciationGuide | None = None


class Pronunciation(BaseModel):
    accuracy_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    fluency_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    completeness_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    prosody_score: float | None = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    overall_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    words: list[WordScore] = Field(min_length=1)


class PracticeResponse(BaseModel):
    transcription: str
    correction: str | None = None
    explanation: str | None = None
    pronunciation: Pronunciation | None = None
    model_audio_url: str | None = None
    prompt_version: str = "practice-v2"
    activity_id: str | None = None
    target_phrase: str | None = None
    evaluation_status: Literal["complete", "partial", "unavailable", "no_speech", "off_target"]
    grammar_status: Literal["evaluated", "unavailable"] = "unavailable"
    feedback: str | None = None
