"""Fluentia AI — Backend FastAPI (F2: real APIs + scenarios + pronunciation)."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import os
import hmac
import httpx

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import ValidationError
from security import Principal, authenticate
from audio_limits import read_validated_audio
from request_limits import RequestSizeLimitMiddleware
from provider_access import provider_access
from practice_contract import (
    PracticeResponse, Pronunciation, PronunciationGuide, normalize_phrase, resolve_activity,
)

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
from quality_log import cleanup_old_records, get_metrics
from test_quality import run_golden_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fluentia AI API", version="2.0.0")
app.add_middleware(RequestSizeLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_cleanup():
    await cleanup_old_records(30)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/assess", response_model=PracticeResponse)
async def assess(
    request: Request,
    audio: UploadFile = File(...),
    principal: Principal = Depends(authenticate),
    activity_id: str = Form(..., min_length=1, max_length=64),
    target_phrase: str | None = Form(default=None, max_length=500),
):
    try:
        activity_id, reference = resolve_activity(activity_id, target_phrase)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    audio_bytes = await read_validated_audio(audio)
    async with provider_access(principal, request, cost=3):
        return await _assess_valid_audio(audio_bytes, activity_id, reference)


async def _assess_valid_audio(audio_bytes: bytes, activity_id: str, reference: str | None):

    try:
        transcription = await transcribe_audio(
            audio_bytes, "audio.wav"
        )
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise HTTPException(status_code=502, detail="Error al transcribir el audio.")

    if not transcription.strip():
        return PracticeResponse(
            transcription="",
            activity_id=activity_id,
            target_phrase=reference,
            evaluation_status="no_speech",
            feedback="No detectamos habla. Acércate al micrófono y vuelve a grabar.",
        )

    if reference and normalize_phrase(transcription) != normalize_phrase(reference):
        return PracticeResponse(
            transcription=transcription,
            activity_id=activity_id,
            target_phrase=reference,
            evaluation_status="off_target",
            feedback="La grabación no coincide con la frase del ejercicio. Escúchala y repítela completa.",
        )

    grammar_result, pron_data = await asyncio.gather(
        correct_grammar(transcription, strict=True),
        assess_pronunciation(audio_bytes, reference or transcription),
        return_exceptions=True,
    )

    grammar_valid = (
        isinstance(grammar_result, tuple) and len(grammar_result) == 2
        and (grammar_result[0] is None or (
            isinstance(grammar_result[0], str) and bool(grammar_result[0].strip())
        ))
        and isinstance(grammar_result[1], str) and bool(grammar_result[1].strip())
    )
    correction, explanation = grammar_result if grammar_valid else (None, None)
    pronunciation = None
    if isinstance(pron_data, dict):
        try:
            pronunciation = Pronunciation.model_validate(pron_data, strict=True)
        except ValidationError:
            logger.warning("Pronunciation response did not meet the practice contract")

    # Optional coaching must not turn an otherwise valid assessment into a failure.
    if pronunciation:
        weak_words = list(dict.fromkeys(w.word for w in pronunciation.words if w.accuracy_score < 80))
        if weak_words:
            try:
                guides = {g["word"].lower(): g for g in await get_pronunciation_guides(weak_words)}
                for word in pronunciation.words:
                    if word.word.lower() in guides:
                        word.pronunciation_guide = PronunciationGuide.model_validate(guides[word.word.lower()])
            except Exception:
                logger.warning("Pronunciation coaching unavailable")

    if grammar_valid and pronunciation:
        status, feedback = "complete", None
    elif grammar_valid or pronunciation:
        status = "partial"
        feedback = "La evaluación quedó incompleta. Puedes ver el feedback disponible y volver a intentarlo; este intento no suma XP."
    else:
        status = "unavailable"
        feedback = "No pudimos evaluar el audio. Vuelve a intentarlo; este fallo no cambia tu progreso."

    return PracticeResponse(
        transcription=transcription,
        correction=correction,
        explanation=explanation,
        pronunciation=pronunciation,
        activity_id=activity_id,
        target_phrase=reference,
        evaluation_status=status,
        grammar_status="evaluated" if grammar_valid else "unavailable",
        feedback=feedback,
    )


@app.get("/tts")
async def tts(request: Request, text: str = Query(..., min_length=1, max_length=500),
              principal: Principal = Depends(authenticate)):
    async with provider_access(principal, request):
        audio_bytes = await generate_tts(text)
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="Error al generar audio.")
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ─── F2: Scenario endpoints ───


@app.get("/scenarios")
async def get_scenarios():
    return list_scenarios()


@app.post("/scenario/start")
async def scenario_start(request: Request, scenario_id: str = Query(..., min_length=1, max_length=64),
                         principal: Principal = Depends(authenticate)):
    async with provider_access(principal, request):
        result = start_scenario(scenario_id, owner_id=principal.uid)
    if not result:
        raise HTTPException(status_code=404, detail="Escenario no encontrado.")
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(result.get("status_code", 400), result["error"])
    _, data = result
    return data


@app.post("/scenario/turn")
async def scenario_turn(
    request: Request,
    session_id: str = Query(..., min_length=1, max_length=64),
    audio: UploadFile = File(...),
    principal: Principal = Depends(authenticate),
):
    audio_bytes = await read_validated_audio(audio)
    async with provider_access(principal, request, cost=2):
        result = await process_turn(session_id, audio_bytes, "audio.wav", owner_id=principal.uid)
    if result is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if "error" in result:
        raise HTTPException(status_code=result.get("status_code", 400), detail=result["error"])
    return result


@app.post("/scenario/end")
async def scenario_end(session_id: str = Query(..., min_length=1, max_length=64),
                       principal: Principal = Depends(authenticate)):
    result = end_scenario(session_id, owner_id=principal.uid)
    if not result:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    if "error" in result:
        raise HTTPException(result.get("status_code", 400), result["error"])
    return result


# ─── F3: Pedagogy endpoints ───


@app.get("/placement/items")
async def placement_items():
    return get_placement_items()


@app.post("/placement/evaluate")
async def placement_evaluate(
    request: Request,
    item_id: int = Query(...),
    audio: UploadFile = File(...),
    principal: Principal = Depends(authenticate),
):
    item = get_placement_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item no encontrado.")

    audio_bytes = await read_validated_audio(audio)
    async with provider_access(principal, request, cost=2):
        result = await evaluate_placement_item(audio_bytes, "audio.wav", item)
    return result


@app.post("/placement/result")
async def placement_result(results: list[dict], principal: Principal = Depends(authenticate)):
    level_data = determine_level(results)
    return level_data


@app.get("/curriculum/{level}")
async def curriculum(level: str):
    goals = get_session_goals(level.upper())
    return goals


@app.post("/progression/check")
async def progression_check(level: str = Query(...), scores: list[float] = Query(...),
                            principal: Principal = Depends(authenticate)):
    result = check_level_progression(level.upper(), scores)
    return result


# ─── F4: Eval endpoints ───


def _check_eval_key(key: str, principal: Principal):
    expected = os.getenv("EVAL_KEY", "")
    if not principal.is_admin or len(expected) < 32 or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Invalid eval key.")


@app.get("/eval/run")
async def eval_run(request: Request, x_eval_key: str = Header(""),
                   principal: Principal = Depends(authenticate)):
    _check_eval_key(x_eval_key, principal)
    async with provider_access(principal, request, cost=20):
        result = await run_golden_set("http://quality-internal",
            authorization=request.headers.get("authorization", ""), owner_id=principal.uid,
            transport=httpx.ASGITransport(app=request.app))
    return result


@app.get("/eval/metrics")
async def eval_metrics(
    hours: int = Query(default=24, ge=1, le=720),
    x_eval_key: str = Header(""),
    principal: Principal = Depends(authenticate),
):
    _check_eval_key(x_eval_key, principal)
    metrics = await get_metrics(hours)
    return metrics
