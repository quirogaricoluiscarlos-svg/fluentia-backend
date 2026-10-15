"""Fluentia AI — Service integrations (F1)."""

import base64
import difflib
import json
import logging
import math
import os
import re

import time
from xml.sax.saxutils import escape

import edge_tts
import httpx
from groq import AsyncGroq
from openai import AsyncOpenAI

from guardrails import (
    validate_correction,
    validate_pronunciation_guides,
    validate_transcription,
)
from quality_log import log_ai_call
from pronunciation import parse_pronunciation_response

logger = logging.getLogger(__name__)

groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

deepseek_client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

CORRECTION_PROMPT = """\
Eres un tutor de inglés para hispanohablantes de Latinoamérica.
El usuario te enviará una oración en inglés que acaba de decir en voz alta.

Responde SOLO en JSON con este formato exacto:
{"correction": "oración corregida o null", "explanation": "explicación en español"}

Reglas:
- Si tiene errores gramaticales, de vocabulario o estructura: corrígela y explica en español simple.
- Si es correcta: {"correction": null, "explanation": "¡Perfecto! Tu oración es gramaticalmente correcta."}
- Solo JSON, sin texto adicional."""

PRONUNCIATION_GUIDE_PROMPT = """\
Eres un experto en fonética inglesa para hispanohablantes de Latinoamérica.
Te daré una lista de palabras en inglés que el usuario pronunció mal.

Para cada palabra, devuelve cómo se pronuncia escrito de forma que un hispanohablante lo pueda leer y pronunciar correctamente.
Usa sonidos del español, NO IPA. Ejemplo: "thought" → "zot", "schedule" → "skéyul", "the" → "da", "world" → "wérld".

Responde SOLO en JSON con este formato:
{"guides": [{"word": "thought", "pronunciation": "zot", "tip": "La 'th' suena como una 'z' suave con la lengua entre los dientes"}]}

Solo JSON, sin texto adicional."""


async def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    start = time.time()
    error_msg = None
    try:
        transcription = await groq_client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3",
            language="en",
        )
        text = transcription.text
        valid, reason = validate_transcription(text)
        latency = int((time.time() - start) * 1000)
        await log_ai_call("transcribe_audio", filename, text, valid, 0, latency)
        if not valid:
            logger.warning("Transcription guardrail failed: %s", reason)
            return ""
        return text
    except Exception as e:
        error_msg = str(e)
        latency = int((time.time() - start) * 1000)
        await log_ai_call("transcribe_audio", filename, "", False, 0, latency, error_msg)
        raise


class GrammarEvaluationError(RuntimeError):
    """The grammar provider did not return a valid, complete evaluation."""


def _validate_grammar_result(data: object) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Response is not a dict"
    explanation = data.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        return False, "Missing or empty explanation"
    if "correction" not in data:
        return False, "Missing correction"
    correction = data["correction"]
    if correction is not None and (not isinstance(correction, str) or not correction.strip()):
        return False, "Correction must be nonempty text or null"
    return validate_correction(data)


async def correct_grammar(
    transcription: str, *, strict: bool = False
) -> tuple[str | None, str | None]:
    last_error = None
    for attempt in range(2):
        start = time.time()
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": CORRECTION_PROMPT},
                    {"role": "user", "content": transcription},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text)
            valid, reason = _validate_grammar_result(data)
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "correct_grammar", transcription, text, valid, attempt, latency
            )
            if valid:
                return data.get("correction"), data.get("explanation")
            logger.warning("Correction guardrail failed (attempt %d): %s", attempt + 1, reason)
        except Exception as e:
            last_error = e
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "correct_grammar", transcription, "", False, attempt, latency, str(e)
            )
            logger.error("DeepSeek correction failed: %s", e)
            if attempt == 1 and not strict:
                return None, None
    if strict:
        raise GrammarEvaluationError(
            "No se pudo completar la evaluación gramatical. Intentá de nuevo."
        ) from last_error
    return None, "No pudimos evaluar tu respuesta. Intentá de nuevo."


# ─── Proveedores de voz (gratis por defecto) ───
# PRONUNCIATION_PROVIDER: "whisper" (gratis, por defecto) | "azure"
# TTS_PROVIDER:           "edge"    (gratis, por defecto) | "azure"
# Si el proveedor elegido falla, se intenta el otro (Azure solo si hay clave).

PRONUNCIATION_PROVIDER = os.getenv("PRONUNCIATION_PROVIDER", "whisper").lower()
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge").lower()
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "-10%")

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("\u2019", "'"))


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, v)))


def _to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    return {}


def score_from_whisper(data: dict, reference_text: str) -> dict | None:
    """Estimación de pronunciación con Whisper (sin Azure).

    Es una APROXIMACIÓN, no evaluación fonética:
    - accuracy: confianza del reconocedor (exp(avg_logprob) por segmento),
      con penalización por vacilación y por palabras distintas a la referencia.
    - fluency: palabras por minuto y pausas largas (> 0.8 s).
    - completeness: % de palabras de la referencia reconocidas.
    - prosody: None (no se mide; el contrato lo permite).
    """
    words = data.get("words") or []
    segments = data.get("segments") or []
    if not words:
        return None

    seg_conf = [
        (
            float(s.get("start", 0.0)),
            float(s.get("end", 0.0)),
            _clamp(100.0 * math.exp(float(s.get("avg_logprob", -0.5)))),
        )
        for s in segments
    ] or [(0.0, 1e9, 70.0)]

    def conf_at(t: float) -> float:
        for start, end, c in seg_conf:
            if start - 0.05 <= t <= end + 0.05:
                return c
        return seg_conf[-1][2]

    rec = []  # (token, start, end, conf)
    for w in words:
        toks = _tokens(str(w.get("word", "")))
        if not toks:
            continue
        start, end = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        for t in toks:
            rec.append((t, start, end, conf_at((start + end) / 2)))
    if not rec:
        return None

    rec_scores = []
    for t, start, end, c in rec:
        expected = 0.15 + 0.07 * len(t)
        penalty = 15.0 if (end - start) > 2.2 * expected else 0.0
        rec_scores.append(_clamp(c - penalty))

    ref = _tokens(reference_text) or [r[0] for r in rec]
    rec_tokens = [r[0] for r in rec]
    matcher = difflib.SequenceMatcher(a=ref, b=rec_tokens, autojunk=False)

    out_words = []
    matched = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                matched += 1
                out_words.append({"word": ref[i1 + k],
                                  "accuracy_score": round(rec_scores[j1 + k], 1),
                                  "error_type": None})
        elif op == "replace":
            used: set[int] = set()
            for k in range(i2 - i1):
                target = ref[i1 + k]
                best_j, best_r = None, 0.0
                for j in range(j1, j2):
                    if j in used:
                        continue
                    r = difflib.SequenceMatcher(a=target, b=rec_tokens[j]).ratio()
                    if r > best_r:
                        best_j, best_r = j, r
                if best_j is not None and best_r >= 0.5:
                    used.add(best_j)
                    out_words.append({"word": target,
                                      "accuracy_score": round(_clamp(min(rec_scores[best_j], 30.0 + 40.0 * best_r)), 1),
                                      "error_type": "Mispronunciation"})
                else:
                    out_words.append({"word": target, "accuracy_score": 0.0,
                                      "error_type": "Omission"})
        elif op == "delete":
            for k in range(i2 - i1):
                out_words.append({"word": ref[i1 + k], "accuracy_score": 0.0,
                                  "error_type": "Omission"})
        # "insert": palabras extra, no se listan.

    if not out_words:
        return None
    scored = [w["accuracy_score"] for w in out_words if w["error_type"] != "Omission"]
    accuracy = sum(scored) / len(scored) if scored else 0.0
    completeness = 100.0 * matched / len(ref) if ref else 0.0

    span = max(rec[-1][2] - rec[0][1], 0.5)
    wpm = len(rec) / (span / 60.0)
    long_pauses = sum(1 for a, b in zip(rec, rec[1:]) if b[1] - a[2] > 0.8)
    fluency = _clamp(100.0 - 10.0 * long_pauses - max(0.0, 90.0 - wpm) * 0.6)

    accuracy, completeness = _clamp(accuracy), _clamp(completeness)
    return {
        "accuracy_score": round(accuracy, 1),
        "fluency_score": round(fluency, 1),
        "completeness_score": round(completeness, 1),
        "prosody_score": None,
        "overall_score": round((accuracy + fluency + completeness) / 3.0, 1),
        "words": out_words,
    }


async def _assess_whisper(audio_bytes: bytes, reference_text: str) -> dict | None:
    if not isinstance(reference_text, str) or not reference_text.strip():
        return None
    try:
        resp = await groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3",
            language="en",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0.0,
        )
        return score_from_whisper(_to_dict(resp), reference_text)
    except Exception as e:
        logger.error("Whisper pronunciation failed: %s", e)
        return None


async def assess_pronunciation(
    audio_bytes: bytes, reference_text: str
) -> dict | None:
    order = (
        [_assess_azure, _assess_whisper]
        if PRONUNCIATION_PROVIDER == "azure"
        else [_assess_whisper, _assess_azure]
    )
    for fn in order:
        result = await fn(audio_bytes, reference_text)
        if result:
            return result
    return None


async def _assess_azure(
    audio_bytes: bytes, reference_text: str
) -> dict | None:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    if not key or not isinstance(reference_text, str) or not reference_text.strip():
        return None

    url = (
        f"https://{region}.stt.speech.microsoft.com/"
        "speech/recognition/conversation/cognitiveservices/v1"
        "?language=en-US&format=detailed"
    )

    pron_config = {
        "ReferenceText": reference_text,
        "GradingSystem": "HundredMark",
        "Granularity": "Word",
        "Dimension": "Comprehensive",
        "EnableMiscue": True,
    }
    encoded = base64.b64encode(json.dumps(pron_config).encode()).decode()

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        "Pronunciation-Assessment": encoded,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(url, content=audio_bytes, headers=headers)

        if resp.status_code != 200:
            logger.warning("Azure pronunciation HTTP %s: %s", resp.status_code, resp.text)
            return None

        return parse_pronunciation_response(resp.json())
    except Exception as e:
        logger.error("Azure pronunciation failed: %s", e)
        return None


async def get_pronunciation_guides(words: list[str]) -> list[dict]:
    if not words:
        return []
    words_str = json.dumps(words)
    for attempt in range(2):
        start = time.time()
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": PRONUNCIATION_GUIDE_PROMPT},
                    {"role": "user", "content": words_str},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text)
            valid, reason = validate_pronunciation_guides(data)
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "get_pronunciation_guides", words_str, text, valid, attempt, latency
            )
            if valid:
                return data.get("guides", [])
            logger.warning("Pronunciation guide guardrail failed (attempt %d): %s", attempt + 1, reason)
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "get_pronunciation_guides", words_str, "", False, attempt, latency, str(e)
            )
            logger.error("Pronunciation guide failed: %s", e)
            if attempt == 1:
                return []
    return []


async def _edge_tts(text: str) -> bytes | None:
    """TTS gratis (voces neurales de Microsoft Edge, sin API key). Devuelve MP3."""
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                buf.extend(chunk["data"])
        return bytes(buf) or None
    except Exception as e:
        logger.error("Edge TTS failed: %s", e)
        return None


async def generate_tts(text: str) -> bytes | None:
    order = (
        [_azure_tts, _edge_tts] if TTS_PROVIDER == "azure" else [_edge_tts, _azure_tts]
    )
    for fn in order:
        audio = await fn(text)
        if audio:
            return audio
    return None


async def _azure_tts(text: str) -> bytes | None:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    if not key:
        return None

    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        '<voice name="en-US-JennyNeural">'
        f'<prosody rate="-10%">{escape(text)}</prosody>'
        '</voice></speak>'
    )

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(url, content=ssml, headers=headers)
        if resp.status_code == 200:
            return resp.content
        logger.warning("Azure TTS HTTP %s: %s", resp.status_code, resp.text)
        return None
    except Exception as e:
        logger.error("Azure TTS failed: %s", e)
        return None
