import os
import re
import json
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

import firebase_admin
from firebase_admin import credentials, auth

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------- Firebase Admin (clave desde variable de entorno, NUNCA en el repo) ----------
# En Render crea la env var FIREBASE_CREDENTIALS con el JSON COMPLETO de la service account.
firebase_json = os.getenv("FIREBASE_CREDENTIALS")
if not firebase_json:
    raise RuntimeError("Falta la variable de entorno FIREBASE_CREDENTIALS")
cred = credentials.Certificate(json.loads(firebase_json))
firebase_admin.initialize_app(cred)


# ---------- Rate limiter (por usuario si hay token; si no, por IP) ----------
def rate_key(request: Request) -> str:
    return getattr(request.state, "uid", None) or get_remote_address(request)


limiter = Limiter(key_func=rate_key)

app = FastAPI(title="Fluentia AI API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Una app Android nativa NO envia Origin, asi que CORS no es tu defensa real
# (lo es el token de Firebase). Aun asi, restringe lo que de verdad uses en web.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------- DeepSeek ----------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
# Modelo actualizado: el alias 'deepseek-chat' se retira el 24/07/2026.
DEEPSEEK_MODEL = "deepseek-v4-flash"


# ---------- Auth ----------
async def verify_token(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token requerido")
    id_token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalido o expirado")
    request.state.uid = decoded["uid"]
    return decoded


# ---------- Modelos ----------
class CorrectionRequest(BaseModel):
    text: str
    module: str = "general"
    level: str = "BEGINNER"


class CorrectionResponse(BaseModel):
    corrected: str
    explanation: str
    score: int
    improved_version: str


MODULE_PROMPTS = {
    "personal_introduction": "El usuario se esta presentando.",
    "daily_routine": "El usuario habla de su rutina diaria.",
    "work_conversation": "El usuario conversa sobre temas de trabajo.",
    "travel_basics": "El usuario practica ingles para viajar.",
    "job_interview": "El usuario practica una entrevista de trabajo.",
}


# ---------- Parseo robusto del JSON del modelo ----------
def parse_ai_json(raw: str) -> dict:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


@app.post("/practice", response_model=CorrectionResponse)
@limiter.limit("20/minute")
async def practice(request: Request, body: CorrectionRequest, user=Depends(verify_token)):
    contexto = MODULE_PROMPTS.get(body.module, "Conversacion general en ingles.")
    prompt = f"""
Eres Emma, tutora de ingles. Contexto del modulo: {contexto}
Nivel del estudiante: {body.level}.
El usuario dijo (transcripcion de su voz): "{body.text}"

Tareas:
1. Corrige errores de gramatica, vocabulario y eleccion de palabras (NO pronunciacion, solo tienes texto).
2. Explicacion BREVE en espanol del error principal.
3. Puntaje 0-100 de que tan natural/correcta fue la frase.
4. Version mejorada y mas natural de lo que quiso decir.

Devuelve SOLO un objeto JSON con las claves:
"corrected", "explanation", "score", "improved_version".
"""
    try:
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system",
                 "content": "Tutor de ingles profesional. Responde unicamente en JSON valido."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        result = parse_ai_json(completion.choices[0].message.content)
        return CorrectionResponse(
            corrected=str(result.get("corrected", body.text)),
            explanation=str(result.get("explanation", "")),
            score=int(result.get("score", 0)),
            improved_version=str(result.get("improved_version", "")),
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR /practice] {e}")
        raise HTTPException(status_code=502, detail="No se pudo procesar la correccion")


@app.get("/health")
async def health():
    return {"status": "OK"}
