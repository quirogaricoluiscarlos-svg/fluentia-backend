"""Fluentia AI — Backend FastAPI (F2: real APIs + scenarios + pronunciation)."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from pedagogy import (
    check_level_progression,
    determine_level,
    evaluate_placement_item,
    get_placement_item,
    get_placement_items,
    get_session_goals,
    CURRICULUM,
)
from scenarios import end_scenario, list_scenarios, process_turn, start_scenario
from services import (
    assess_pronunciation,
    correct_grammar,
    generate_tts,
    get_pronunciation_guides,
    transcribe_audio,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fluentia AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PronunciationGuide(BaseModel):
    word: str
    pronunciation: str
    tip: str | None = None


class WordScore(BaseModel):
    word: str
    accuracy_score: float
    error_type: str | None = None
    pronunciation_guide: PronunciationGuide | None = None


class Pronunciation(BaseModel):
    accuracy_score: float
    fluency_score: float
    completeness_score: float
    prosody_score: float
    overall_score: float
    words: list[WordScore]


class PracticeResponse(BaseModel):
    transcription: str
    correction: str | None = None
    explanation: str | None = None
    pronunciation: Pronunciation | None = None
    model_audio_url: str | None = None
    prompt_version: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/assess", response_model=PracticeResponse)
async def assess(
    audio: UploadFile = File(...),
    authorization: str = Header(""),
):
    audio_bytes = await audio.read()
    size_kb = len(audio_bytes) / 1024
    logger.info("Audio received: %s (%.1f KB)", audio.filename, size_kb)

    try:
        transcription = await transcribe_audio(
            audio_bytes, audio.filename or "audio.wav"
        )
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=502, detail="Error al transcribir el audio.")

    if not transcription.strip():
        return PracticeResponse(
            transcription="",
            explanation="No se detectó habla en el audio. Intentá de nuevo.",
            prompt_version="f2-v1",
        )

    (correction, explanation), pron_data = await asyncio.gather(
        correct_grammar(transcription),
        assess_pronunciation(audio_bytes, transcription),
    )

    pronunciation = None
    if pron_data:
        weak_words = [
            w["word"]
            for w in pron_data["words"]
            if w["accuracy_score"] < 80
        ]

        guides = {}
        if weak_words:
            guide_list = await get_pronunciation_guides(weak_words)
            guides = {g["word"].lower(): g for g in guide_list}

        word_scores = []
        for w in pron_data["words"]:
            guide = guides.get(w["word"].lower())
            pg = None
            if guide:
                pg = PronunciationGuide(
                    word=guide["word"],
                    pronunciation=guide.get("pronunciation", ""),
                    tip=guide.get("tip"),
                )
            word_scores.append(WordScore(
                word=w["word"],
                accuracy_score=w["accuracy_score"],
                error_type=w["error_type"],
                pronunciation_guide=pg,
            ))

        overall = (
            pron_data["accuracy_score"]
            + pron_data["fluency_score"]
            + pron_data["completeness_score"]
            + pron_data["prosody_score"]
        ) / 4.0

        pronunciation = Pronunciation(
            accuracy_score=pron_data["accuracy_score"],
            fluency_score=pron_data["fluency_score"],
            completeness_score=pron_data["completeness_score"],
            prosody_score=pron_data["prosody_score"],
            overall_score=round(overall, 1),
            words=word_scores,
        )

    return PracticeResponse(
        transcription=transcription,
        correction=correction,
        explanation=explanation,
        pronunciation=pronunciation,
        prompt_version="f2-v1",
    )


@app.get("/tts")
async def tts(text: str = Query(..., min_length=1, max_length=500)):
    audio_bytes = await generate_tts(text)
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="Error al generar audio.")
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ─── F2: Scenario endpoints ───


@app.get("/scenarios")
async def get_scenarios():
    return list_scenarios()


@app.post("/scenario/start")
async def scenario_start(scenario_id: str):
    result = start_scenario(scenario_id)
    if not result:
        raise HTTPException(status_code=404, detail="Escenario no encontrado.")
    _, data = result
    return data


@app.post("/scenario/turn")
async def scenario_turn(
    session_id: str,
    audio: UploadFile = File(...),
):
    audio_bytes = await audio.read()
    logger.info("Scenario turn: session=%s, file=%s", session_id, audio.filename)

    result = await process_turn(
        session_id, audio_bytes, audio.filename or "audio.wav"
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/scenario/end")
async def scenario_end(session_id: str):
    result = end_scenario(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    return result


# ─── F3: Pedagogy endpoints ───


@app.get("/placement/items")
async def placement_items():
    return get_placement_items()


@app.post("/placement/evaluate")
async def placement_evaluate(
    item_id: int = Query(...),
    audio: UploadFile = File(...),
):
    item = get_placement_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item no encontrado.")

    audio_bytes = await audio.read()
    result = await evaluate_placement_item(audio_bytes, audio.filename or "audio.wav", item)
    return result


@app.post("/placement/result")
async def placement_result(results: list[dict]):
    level_data = determine_level(results)
    return level_data


@app.get("/curriculum/{level}")
async def curriculum(level: str):
    goals = get_session_goals(level.upper())
    return goals


@app.post("/progression/check")
async def progression_check(level: str = Query(...), scores: list[float] = Query(...)):
    result = check_level_progression(level.upper(), scores)
    return result
