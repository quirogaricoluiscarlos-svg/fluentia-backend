"""Fluentia AI — CEFR Pedagogy engine (F3)."""

import json
import logging

from services import deepseek_client, transcribe_audio

logger = logging.getLogger(__name__)

CEFR_LEVELS = ["A1", "A2", "B1", "B2"]

PLACEMENT_ITEMS = [
    {"id": 1, "level": "A1", "phrase": "Hello, my name is Carlos.", "skill": "greeting"},
    {"id": 2, "level": "A1", "phrase": "I like to eat pizza and drink coffee.", "skill": "preferences"},
    {"id": 3, "level": "A2", "phrase": "Yesterday I went to the supermarket and bought some fruit.", "skill": "past_tense"},
    {"id": 4, "level": "A2", "phrase": "Could you tell me where the nearest bus stop is?", "skill": "asking_directions"},
    {"id": 5, "level": "B1", "phrase": "If I had more free time, I would travel around the world.", "skill": "conditionals"},
    {"id": 6, "level": "B1", "phrase": "I've been studying English for three years and I'm getting better.", "skill": "present_perfect"},
    {"id": 7, "level": "B2", "phrase": "Despite the economic challenges, the company managed to increase its revenue.", "skill": "complex_structures"},
    {"id": 8, "level": "B2", "phrase": "Had I known about the meeting earlier, I would have prepared a presentation.", "skill": "advanced_conditionals"},
]

CURRICULUM = {
    "A1": {
        "title": "Principiante",
        "description": "Frases básicas, presentaciones y necesidades inmediatas",
        "can_do": [
            "Presentarte y saludar",
            "Pedir cosas simples (comida, bebida)",
            "Decir tu nombre, edad y de dónde sos",
            "Entender instrucciones muy simples",
            "Usar presente simple y verbos básicos",
        ],
        "target_structures": ["present simple", "to be", "basic questions", "there is/are"],
        "vocabulary_topics": ["greetings", "numbers", "food", "family", "colors"],
    },
    "A2": {
        "title": "Elemental",
        "description": "Situaciones cotidianas, pasado simple y planes futuros",
        "can_do": [
            "Hacer compras y pedir en restaurantes",
            "Describir tu rutina diaria",
            "Hablar sobre el pasado (ayer, la semana pasada)",
            "Hacer planes simples para el futuro",
            "Pedir y dar indicaciones básicas",
        ],
        "target_structures": ["past simple", "going to", "can/could", "comparatives"],
        "vocabulary_topics": ["shopping", "travel", "daily routine", "weather", "health"],
    },
    "B1": {
        "title": "Intermedio",
        "description": "Opiniones, experiencias y situaciones hipotéticas",
        "can_do": [
            "Dar tu opinión y explicar por qué",
            "Hablar de experiencias pasadas con present perfect",
            "Usar condicionales (if I had...)",
            "Entender y participar en conversaciones de trabajo",
            "Contar historias y anécdotas",
        ],
        "target_structures": ["present perfect", "conditionals", "passive voice", "reported speech"],
        "vocabulary_topics": ["work", "education", "opinions", "media", "environment"],
    },
    "B2": {
        "title": "Intermedio alto",
        "description": "Temas abstractos, argumentación y expresiones idiomáticas",
        "can_do": [
            "Argumentar y defender una posición",
            "Entender textos complejos y noticias",
            "Usar estructuras avanzadas naturalmente",
            "Hablar de temas abstractos (economía, sociedad)",
            "Usar expresiones idiomáticas comunes",
        ],
        "target_structures": ["mixed conditionals", "inversions", "cleft sentences", "advanced modals"],
        "vocabulary_topics": ["politics", "science", "idioms", "business", "culture"],
    },
}

EVALUATE_PROMPT = """\
Eres un evaluador de nivel CEFR de inglés para hispanohablantes.
El estudiante intentó decir esta frase de nivel {level}: "{expected}"
Lo que realmente dijo fue: "{actual}"

Evalúa de 0 a 100 qué tan bien lo dijo considerando:
- Gramática correcta (40%)
- Vocabulario apropiado (30%)
- Cercanía a la frase esperada (30%)

Responde SOLO en JSON:
{{"score": 75, "feedback": "explicación breve en español de cómo mejorar"}}"""


async def evaluate_placement_item(
    audio_bytes: bytes, filename: str, item: dict
) -> dict:
    transcription = await transcribe_audio(audio_bytes, filename)

    if not transcription.strip():
        return {
            "item_id": item["id"],
            "level": item["level"],
            "expected": item["phrase"],
            "actual": "",
            "score": 0,
            "feedback": "No se detectó habla. Intentá de nuevo.",
        }

    try:
        prompt = EVALUATE_PROMPT.format(
            level=item["level"],
            expected=item["phrase"],
            actual=transcription.strip(),
        )
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        score = min(100, max(0, int(data.get("score", 0))))
        feedback = data.get("feedback", "")
    except Exception as e:
        logger.error("Placement evaluation failed: %s", e)
        score = 50
        feedback = "No se pudo evaluar completamente."

    return {
        "item_id": item["id"],
        "level": item["level"],
        "expected": item["phrase"],
        "actual": transcription.strip(),
        "score": score,
        "feedback": feedback,
    }


def determine_level(item_results: list[dict]) -> dict:
    level_scores = {"A1": [], "A2": [], "B1": [], "B2": []}
    for result in item_results:
        level = result["level"]
        if level in level_scores:
            level_scores[level].append(result["score"])

    level_averages = {}
    for level in CEFR_LEVELS:
        scores = level_scores[level]
        level_averages[level] = sum(scores) / len(scores) if scores else 0

    assigned_level = "A1"
    for level in CEFR_LEVELS:
        if level_averages[level] >= 60:
            assigned_level = level
        else:
            break

    overall_avg = sum(r["score"] for r in item_results) / len(item_results) if item_results else 0

    curriculum = CURRICULUM[assigned_level]

    return {
        "assigned_level": assigned_level,
        "level_title": curriculum["title"],
        "level_description": curriculum["description"],
        "overall_score": round(overall_avg, 1),
        "level_scores": {k: round(v, 1) for k, v in level_averages.items()},
        "can_do": curriculum["can_do"],
        "next_level": CEFR_LEVELS[CEFR_LEVELS.index(assigned_level) + 1]
        if assigned_level != "B2"
        else None,
    }


def get_session_goals(level: str) -> dict:
    curriculum = CURRICULUM.get(level, CURRICULUM["A1"])
    return {
        "level": level,
        "title": curriculum["title"],
        "target_structures": curriculum["target_structures"],
        "vocabulary_topics": curriculum["vocabulary_topics"],
        "can_do": curriculum["can_do"],
    }


def check_level_progression(level: str, recent_scores: list[float]) -> dict:
    if len(recent_scores) < 5:
        return {"action": "stay", "current_level": level}

    avg = sum(recent_scores[-5:]) / 5

    current_idx = CEFR_LEVELS.index(level)

    if avg >= 85 and current_idx < len(CEFR_LEVELS) - 1:
        new_level = CEFR_LEVELS[current_idx + 1]
        return {
            "action": "level_up",
            "current_level": level,
            "new_level": new_level,
            "message": f"¡Felicitaciones! Subiste a nivel {new_level}.",
        }
    elif avg < 40 and current_idx > 0:
        new_level = CEFR_LEVELS[current_idx - 1]
        return {
            "action": "level_down",
            "current_level": level,
            "new_level": new_level,
            "message": f"Vamos a reforzar el nivel {new_level} para afianzar las bases.",
        }
    else:
        return {"action": "stay", "current_level": level}


def get_placement_items() -> list[dict]:
    return [
        {"id": item["id"], "level": item["level"], "phrase": item["phrase"]}
        for item in PLACEMENT_ITEMS
    ]


def get_placement_item(item_id: int) -> dict | None:
    for item in PLACEMENT_ITEMS:
        if item["id"] == item_id:
            return item
    return None
