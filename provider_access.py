"""Reserve bounded consumption and stop provider chains before quota slots expire."""

import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import HTTPException
from quota import consume


@asynccontextmanager
async def provider_access(principal, request, *, cost=1):
    try:
        timeout = int(os.getenv("PROVIDER_TIMEOUT_SECONDS", "90"))
        if not 1 <= timeout <= 120:
            raise ValueError()
    except ValueError:
        raise HTTPException(503, "El servicio no está disponible por el momento.")
    async with consume(principal, request, cost=cost):
        try:
            async with asyncio.timeout(timeout):
                yield
        except TimeoutError:
            raise HTTPException(504, "La evaluación tardó demasiado. Vuelve a intentarlo.") from None
