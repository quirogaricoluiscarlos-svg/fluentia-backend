"""Fluentia AI — Scenario definitions and conversation engine (F2)."""

import json
import logging
import uuid
from dataclasses import dataclass, field

from services import deepseek_client, transcribe_audio

logger = logging.getLogger(__name__)

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
    messages: list[dict] = field(default_factory=list)
    objectives_met: list[str] = field(default_factory=list)
    turn_count: int = 0
    corrections_count: int = 0
    total_score: float = 0.0


_sessions: dict[str, ConversationSession] = {}


def start_scenario(scenario_id: str) -> tuple[str, dict] | None:
    scenario = SCENARIOS.get(scenario_id)
    if not scenario:
        return None

    session_id = str(uuid.uuid4())
    session = ConversationSession(
        session_id=session_id,
        scenario_id=scenario_id,
    )
    session.messages.append({
        "role": "assistant",
        "content": scenario["ai_greeting"],
    })

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
    session_id: str, audio_bytes: bytes, filename: str
) -> dict | None:
    session = _sessions.get(session_id)
    if not session:
        return None

    scenario = SCENARIOS[session.scenario_id]

    transcription = await transcribe_audio(audio_bytes, filename)
    if not transcription.strip():
        return {
            "error": "No se detectó habla en el audio. Intentá de nuevo.",
            "turn_count": session.turn_count,
        }

    session.messages.append({"role": "user", "content": transcription})
    session.turn_count += 1

    system_prompt = _build_system_prompt(scenario, session.objectives_met)

    try:
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
        data = json.loads(text)
    except Exception as e:
        logger.error("Scenario turn failed: %s", e)
        return {"error": "Error al procesar el turno. Intentá de nuevo."}

    ai_response = data.get("ai_response", "")
    correction = data.get("correction")
    explanation = data.get("explanation")
    new_objectives = data.get("objectives_met", [])

    session.messages.append({"role": "assistant", "content": ai_response})

    if correction:
        session.corrections_count += 1

    for obj in new_objectives:
        if obj not in session.objectives_met:
            session.objectives_met.append(obj)

    score = 100 if not correction else 60
    session.total_score += score

    all_met = len(session.objectives_met) >= len(scenario["objectives_checklist"])

    return {
        "session_id": session_id,
        "turn_count": session.turn_count,
        "transcription": transcription,
        "ai_response": ai_response,
        "correction": correction,
        "explanation": explanation,
        "objectives_met": session.objectives_met,
        "objectives_total": scenario["objectives_checklist"],
        "all_objectives_met": all_met,
    }


def end_scenario(session_id: str) -> dict | None:
    session = _sessions.pop(session_id, None)
    if not session:
        return None

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
