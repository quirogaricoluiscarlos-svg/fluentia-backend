"""Fluentia AI — Scenario definitions and conversation engine (F2)."""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from time import monotonic as _session_now

from services import deepseek_client, transcribe_audio

from guardrails import validate_scenario_turn
from quality_log import log_ai_call

logger = logging.getLogger(__name__)


def _positive_setting(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


SCENARIO_SESSION_TTL_SECONDS = _positive_setting("SCENARIO_SESSION_TTL_SECONDS", 1800)
SCENARIO_MAX_TURNS = _positive_setting("SCENARIO_MAX_TURNS", 20)
SCENARIO_MAX_SESSIONS_PER_OWNER = _positive_setting("SCENARIO_MAX_SESSIONS_PER_OWNER", 3)
SCENARIO_MAX_SESSIONS_TOTAL = _positive_setting("SCENARIO_MAX_SESSIONS_TOTAL", 1000)
SCENARIO_HISTORY_TURNS = min(
    _positive_setting("SCENARIO_HISTORY_TURNS", 10), SCENARIO_MAX_TURNS
)

SCENARIOS = {
    "smalltalk": {
        "title": "Charla con el vecino",
        "description": "Conversación casual con tu vecino sobre el día, el clima y planes.",
        "icon": "🏘️",
        "difficulty": "A1",
        "ai_character": "Sam, a friendly neighbor",
        "ai_greeting": "Hey there! Beautiful day today, isn't it? How's everything going?",
        "objective": "Mantener una conversación casual de al menos 5 turnos.",
        "objectives_checklist": ["Saludar", "Hablar del clima", "Contar algo de tu día", "Hacer una pregunta", "Despedirte"],
    },
    "cafe": {
        "title": "En la cafetería",
        "description": "Pedí tu café y algo para comer en una cafetería.",
        "icon": "☕",
        "difficulty": "A2",
        "ai_character": "Alex, a barista at a cozy coffee shop",
        "ai_greeting": "Hi! Welcome to Sunrise Coffee. What can I get for you today?",
        "objective": "Pedir una bebida, algo para comer y pagar.",
        "objectives_checklist": ["Pedir una bebida", "Pedir algo para comer", "Preguntar el precio", "Agradecer y pagar", "Despedirte"],
    },
    "shopping": {
        "title": "De compras",
        "description": "Buscá y comprá algo en una tienda de ropa.",
        "icon": "👕",
        "difficulty": "A2",
        "ai_character": "Jordan, a helpful store assistant",
        "ai_greeting": "Hello! Welcome to our store. Are you looking for anything in particular today?",
        "objective": "Encontrar un producto, preguntar talle/precio y decidir.",
        "objectives_checklist": ["Decir qué buscás", "Preguntar por talle o color", "Preguntar el precio", "Decidir si comprás", "Agradecer"],
    },
    "directions": {
        "title": "Pidiendo indicaciones",
        "description": "Estás perdido en una ciudad y necesitás llegar a un lugar.",
        "icon": "🗺️",
        "difficulty": "A2",
        "ai_character": "Morgan, a local resident who knows the city well",
        "ai_greeting": "Hi there! You look a bit lost. Can I help you find something?",
        "objective": "Preguntar cómo llegar, entender las indicaciones y agradecer.",
        "objectives_checklist": ["Decir a dónde querés ir", "Preguntar cómo llegar", "Confirmar que entendiste", "Preguntar distancia o tiempo", "Agradecer"],
    },
    "plans": {
        "title": "Planes con un amigo",
        "description": "Organizá una salida con un amigo para el fin de semana.",
        "icon": "🎬",
        "difficulty": "B1",
        "ai_character": "Riley, your good friend who wants to hang out",
        "ai_greeting": "Hey! So, do you have any plans this weekend? I was thinking we could do something fun!",
        "objective": "Proponer un plan, acordar hora y lugar.",
        "objectives_checklist": ["Proponer una actividad", "Acordar un día y hora", "Elegir un lugar", "Confirmar el plan", "Despedirte con entusiasmo"],
    },
}


def _build_system_prompt(scenario: dict, objectives_met: list[str]) -> str:
    met_str = ", ".join(objectives_met) if objectives_met else "ninguno aún"
    return f"""\
You are {scenario['ai_character']} in a roleplay conversation to help a Spanish-speaking student practice English.

RULES:
1. Stay in character as {scenario['ai_character']}. Respond naturally in English, 1-3 sentences max.
2. Keep your language at {scenario['difficulty']} CEFR level — simple and clear.
3. After your in-character response, evaluate the student's English.
4. Respond ONLY in this JSON format:

{{"ai_response": "your in-character reply in English",
  "correction": "corrected version of what the student said, or null if correct",
  "explanation": "brief explanation of errors in Spanish, or '¡Perfecto!' if no errors",
  "objectives_met": ["list of objectives achieved so far from the checklist"]}}

SCENARIO OBJECTIVE: {scenario['objective']}
OBJECTIVES CHECKLIST: {json.dumps(scenario['objectives_checklist'])}
OBJECTIVES ALREADY MET: {met_str}

Be encouraging. If the student struggles, gently guide them. Never break character in ai_response."""


@dataclass
class ConversationSession:
    session_id: str
    scenario_id: str
    owner_id: str | None = None
    created_at: float = field(default_factory=_session_now)
    last_activity_at: float = field(default_factory=_session_now)
    turn_in_progress: bool = False
    messages: list[dict] = field(default_factory=list)
    objectives_met: list[str] = field(default_factory=list)
    turn_count: int = 0
    corrections_count: int = 0
    total_score: float = 0.0


# Process-local storage: a restart loses sessions; multiple workers cannot share
# them. A shared/persistent store remains necessary before scaling deployment.
_sessions: dict[str, ConversationSession] = {}
_sessions_lock = threading.RLock()


def _purge_expired(now: float) -> None:
    """Called with _sessions_lock held. Never remove an in-flight turn."""
    expired = [
        session_id for session_id, session in _sessions.items()
        if not session.turn_in_progress
        and now - session.last_activity_at >= SCENARIO_SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        del _sessions[session_id]


def _get_owned_session(session_id: str, owner_id: str) -> ConversationSession | None:
    """Unknown, expired and foreign sessions have the same public result."""
    _purge_expired(_session_now())
    session = _sessions.get(session_id)
    if not isinstance(owner_id, str) or not owner_id.strip():
        return None
    if session is None or session.owner_id != owner_id:
        return None
    return session


def start_scenario(
    scenario_id: str, *, owner_id: str
) -> tuple[str, dict] | dict | None:
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("An authenticated owner is required")
    scenario = SCENARIOS.get(scenario_id)
    if not scenario:
        return None

    with _sessions_lock:
        now = _session_now()
        _purge_expired(now)
        owner_sessions = sum(session.owner_id == owner_id for session in _sessions.values())
        if owner_sessions >= SCENARIO_MAX_SESSIONS_PER_OWNER:
            return {
                "error": "Ya tenés varias conversaciones abiertas. Finalizá una antes de iniciar otra.",
                "status_code": 429,
            }
        if len(_sessions) >= SCENARIO_MAX_SESSIONS_TOTAL:
            return {
                "error": "Hay muchas conversaciones activas. Intentá de nuevo más tarde.",
                "status_code": 429,
            }
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            session_id=session_id,
            scenario_id=scenario_id,
            owner_id=owner_id,
            created_at=now,
            last_activity_at=now,
            messages=[{"role": "assistant", "content": scenario["ai_greeting"]}],
        )
        _sessions[session_id] = session

    return session_id, {
        "session_id": session_id,
        "scenario": {
            "id": scenario_id,
            "title": scenario["title"],
            "description": scenario["description"],
            "icon": scenario["icon"],
            "difficulty": scenario["difficulty"],
            "objective": scenario["objective"],
            "objectives_checklist": scenario["objectives_checklist"],
        },
        "ai_message": scenario["ai_greeting"],
        "turn_count": 0,
        "objectives_met": [],
    }


async def process_turn(
    session_id: str, audio_bytes: bytes, filename: str, *, owner_id: str
) -> dict | None:
    with _sessions_lock:
        session = _get_owned_session(session_id, owner_id)
        if session is None:
            return None
        if session.turn_in_progress:
            return {
                "error": "Estamos procesando tu turno anterior. Esperá antes de enviar otro.",
                "status_code": 409,
            }
        if session.turn_count >= SCENARIO_MAX_TURNS:
            return {
                "error": "Llegaste al límite de turnos. Finalizá esta conversación para iniciar otra.",
                "status_code": 409,
            }
        session.turn_in_progress = True

    try:
        return await _process_owned_turn(session, audio_bytes, filename)
    finally:
        # Cancellation and every error release the guard. The completed request
        # starts a fresh idle TTL even when the learner needs to retry.
        with _sessions_lock:
            session.last_activity_at = _session_now()
            session.turn_in_progress = False


async def _process_owned_turn(
    session: ConversationSession, audio_bytes: bytes, filename: str
) -> dict:
    scenario = SCENARIOS[session.scenario_id]

    try:
        transcription = await transcribe_audio(audio_bytes, filename)
    except Exception:
        logger.warning("Scenario transcription unavailable")
        return {
            "error": "No pudimos transcribir el audio. Intentá de nuevo.",
            "status_code": 503,
        }
    if not isinstance(transcription, str) or not transcription.strip():
        return {
            "error": "No se detectó habla en el audio. Intentá de nuevo.",
            "turn_count": session.turn_count,
        }

    # Build a provisional history: failed/invalid/cancelled requests must not
    # count as learner turns or leave unmatched user messages behind.
    messages = [*session.messages, {"role": "user", "content": transcription}]

    system_prompt = _build_system_prompt(scenario, session.objectives_met)

    ai_data = None
    for attempt in range(2):
        start = time.time()
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text)
            valid, reason = validate_scenario_turn(data)
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "process_turn", transcription, text, valid, attempt, latency
            )
            if valid:
                ai_data = data
                break
            logger.warning("Scenario turn guardrail failed (attempt %d): %s", attempt + 1, reason)
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            await log_ai_call(
                "process_turn", transcription, "", False, attempt, latency, str(e)
            )
            logger.error("Scenario turn failed: %s", e)
            if attempt == 1:
                return {
                    "error": "Error al procesar el turno. Intentá de nuevo.",
                    "status_code": 503,
                }

    if ai_data is None:
        return {
            "error": "No pudimos generar una respuesta válida. Intentá de nuevo.",
            "status_code": 503,
        }

    data = ai_data

    ai_response = data.get("ai_response", "")
    correction = data.get("correction")
    explanation = data.get("explanation")
    new_objectives = data.get("objectives_met", [])

    messages.append({"role": "assistant", "content": ai_response})
    session.messages = [messages[0], *messages[1:][-2 * SCENARIO_HISTORY_TURNS:]]
    session.turn_count += 1

    if correction:
        session.corrections_count += 1

    for obj in new_objectives:
        if obj not in session.objectives_met:
            session.objectives_met.append(obj)

    score = 100 if not correction else 60
    session.total_score += score

    all_met = len(session.objectives_met) >= len(scenario["objectives_checklist"])

    return {
        "session_id": session.session_id,
        "turn_count": session.turn_count,
        "transcription": transcription,
        "ai_response": ai_response,
        "correction": correction,
        "explanation": explanation,
        "objectives_met": list(session.objectives_met),
        "objectives_total": scenario["objectives_checklist"],
        "all_objectives_met": all_met,
    }


def end_scenario(session_id: str, *, owner_id: str) -> dict | None:
    with _sessions_lock:
        session = _get_owned_session(session_id, owner_id)
        if session is None:
            return None
        if session.turn_in_progress:
            return {
                "error": "Estamos procesando tu último turno. Esperá antes de finalizar.",
                "status_code": 409,
            }
        del _sessions[session_id]

    scenario = SCENARIOS[session.scenario_id]
    avg_score = (session.total_score / session.turn_count) if session.turn_count > 0 else 0

    xp = 0
    if session.turn_count >= 3:
        xp = 20
    if session.turn_count >= 5:
        xp = 40
    if len(session.objectives_met) >= len(scenario["objectives_checklist"]):
        xp += 30
    if avg_score >= 80:
        xp += 20

    return {
        "scenario_id": session.scenario_id,
        "scenario_title": scenario["title"],
        "total_turns": session.turn_count,
        "average_score": round(avg_score, 1),
        "objectives_met": session.objectives_met,
        "objectives_total": scenario["objectives_checklist"],
        "corrections_count": session.corrections_count,
        "xp_earned": xp,
    }


def list_scenarios() -> list[dict]:
    return [
        {
            "id": sid,
            "title": s["title"],
            "description": s["description"],
            "icon": s["icon"],
            "difficulty": s["difficulty"],
            "objective": s["objective"],
        }
        for sid, s in SCENARIOS.items()
    ]
