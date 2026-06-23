"""Fluentia AI — Service integrations (F1)."""

import base64
import json
import logging
import os

import httpx
from groq import AsyncGroq
from openai import AsyncOpenAI

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
    transcription = await groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
        language="en",
    )
    return transcription.text


async def correct_grammar(transcription: str) -> tuple[str | None, str | None]:
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
        return data.get("correction"), data.get("explanation")
    except Exception as e:
        logger.error("DeepSeek correction failed: %s", e)
        return None, None


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
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": PRONUNCIATION_GUIDE_PROMPT},
                {"role": "user", "content": json.dumps(words)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        return data.get("guides", [])
    except Exception as e:
        logger.error("Pronunciation guide failed: %s", e)
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
