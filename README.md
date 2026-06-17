# Backend Fluentia AI (FastAPI)

## Variables de entorno (en Render -> Environment)
- `DEEPSEEK_API_KEY` : tu API key de DeepSeek (platform.deepseek.com).
- `FIREBASE_CREDENTIALS` : el JSON COMPLETO de la cuenta de servicio de Firebase
  (Consola Firebase -> Configuracion del proyecto -> Cuentas de servicio ->
   Generar nueva clave privada). Pega TODO el contenido del archivo como valor.

## Probar localmente
    pip install -r requirements.txt
    # exporta las variables de entorno antes de correr
    uvicorn main:app --reload --port 8000

## Desplegar en Render (gratis)
1. Sube esta carpeta a un repositorio de GitHub.
2. Render.com -> New + -> Web Service -> conecta el repo.
3. Environment: Python.
   - Build Command:  pip install -r requirements.txt
   - Start Command:  uvicorn main:app --host 0.0.0.0 --port $PORT
4. Agrega las dos variables de entorno de arriba.
5. Copia la URL (ej. https://fluentia-api.onrender.com) y ponla en
   app/build.gradle.kts -> BASE_URL (con "/" al final).

## Anti cold-start (plan gratis)
Configura un ping cada 10-14 min a /health con cron-job.org o UptimeRobot
para que el servicio no se duerma.
