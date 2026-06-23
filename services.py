"""Fluentia AI — Service integrations (F1)."""

import base64
import json
import logging
import os

import time

import httpx
from groq import AsyncGroq
from openai import AsyncOpenAI

from guardrails import (
    validate_correction,
    validate_pronunciation_guides,
    validate_transcription,
)
from quality_log import log_ai_call

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


async def correct_grammar(transcription: str) -> tuple[str | None, str | None]:
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
            valid, reason = validate_correction(data)
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "correct_grammar", transcription, text, valid, attempt, latency
            )
            if valid:
                return data.get("correction"), data.get("explanation")
            logger.warning("Correction guardrail failed (attempt %d): %s", attempt + 1, reason)
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "correct_grammar", transcription, "", False, attempt, latency, str(e)
            )
            logger.error("DeepSeek correction failed: %s", e)
            if attempt == 1:
                return None, None
    return None, "No pudimos evaluar tu respuesta. Intentá de nuevo."


async def assess_pronunciation(
    audio_bytes: bytes, reference_text: str
) -> dict | None:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    if not key:
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
    }
    encoded = base64.b64encode(json.dumps(pron_config).encode()).decode()

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "audio/wav",
        "Pronunciation-Assessment": encoded,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(url, content=audio_bytes, headers=headers)

        if resp.status_code != 200:
            logger.warning("Azure pronunciation HTTP %s: %s", resp.status_code, resp.text)
            return None

        data = resp.json()
        n_best = data.get("NBest", [])
        if not n_best:
            return None

        best = n_best[0]
        pron = best.get("PronunciationAssessment", {})
        words = []
        for w in best.get("Words", []):
            wa = w.get("PronunciationAssessment", {})
            error = wa.get("ErrorType")
            words.append({
                "word": w.get("Word", ""),
                "accuracy_score": wa.get("AccuracyScore", 0.0),
                "error_type": error if error != "None" else None,
            })

        return {
            "accuracy_score": pron.get("AccuracyScore", 0.0),
            "fluency_score": pron.get("FluencyScore", 0.0),
            "completeness_score": pron.get("CompletenessScore", 0.0),
            "prosody_score": pron.get("ProsodyScore", 0.0),
            "words": words,
        }
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


async def generate_tts(text: str) -> bytes | None:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    if not key:
        return None

    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        '<voice name="en-US-JennyNeural">'
        f'<prosody rate="-10%">{text}</prosody>'
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
