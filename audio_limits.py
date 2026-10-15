"""Bounded upload reading and validation for the app's PCM WAV recordings.

The RIFF container is checked directly: Python's wave reader can accept a
truncated data chunk or a RIFF size that does not match the actual upload.
No provider is called by this module.
"""

import math
import os
import struct

from fastapi import HTTPException, UploadFile


DEFAULT_MAX_AUDIO_BYTES = 1_048_576
HARD_MAX_AUDIO_BYTES = 5 * 1_048_576
DEFAULT_MAX_AUDIO_SECONDS = 30.0
READ_CHUNK_BYTES = 65_536
WAV_MEDIA_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave"})


def _configured_limits() -> tuple[int, float]:
    try:
        max_bytes = int(os.getenv("MAX_AUDIO_BYTES", str(DEFAULT_MAX_AUDIO_BYTES)))
        max_seconds = float(os.getenv("MAX_AUDIO_SECONDS", str(DEFAULT_MAX_AUDIO_SECONDS)))
        if not 0 < max_bytes <= HARD_MAX_AUDIO_BYTES:
            raise ValueError("Invalid byte limit")
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= 30:
            raise ValueError("Invalid duration limit")
    except (TypeError, ValueError, OverflowError) as error:
        raise HTTPException(
            status_code=503,
            detail="La validación de audio no está disponible. Inténtalo más tarde.",
        ) from error
    return max_bytes, max_seconds


def _validate_media_type(audio: UploadFile) -> None:
    media_type = (audio.content_type or "").split(";", 1)[0].strip().lower()
    if media_type in WAV_MEDIA_TYPES:
        return
    if media_type == "application/octet-stream" and (audio.filename or "").lower().endswith(".wav"):
        return
    raise HTTPException(status_code=415, detail="Envía una grabación en formato WAV.")


def _invalid_wav(detail: str = "La grabación WAV está incompleta o dañada. Vuelve a grabarla.") -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def _validate_wav(data: bytes | bytearray, max_seconds: float) -> None:
    if len(data) < 12:
        raise _invalid_wav()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(status_code=415, detail="El contenido del archivo no es WAV PCM.")
    if struct.unpack_from("<I", data, 4)[0] != len(data) - 8:
        raise _invalid_wav()

    offset = 12
    format_found = False
    data_found = False
    sample_rate = block_align = frame_count = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise _invalid_wav()
        chunk_id = data[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        payload_offset = offset + 8
        payload_end = payload_offset + chunk_size
        # Odd chunks have a padding byte that is outside the declared size.
        padded_end = payload_end + (chunk_size & 1)
        if padded_end > len(data):
            raise _invalid_wav()

        if chunk_id == b"fmt ":
            if format_found or data_found or chunk_size < 16:
                raise _invalid_wav()
            compression, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
                "<HHIIHH", data, payload_offset,
            )
            if compression != 1 or channels != 1 or sample_rate != 16_000 or bits != 16:
                raise HTTPException(
                    status_code=415,
                    detail="La grabación debe ser WAV PCM de 16 bits, mono y 16 kHz.",
                )
            if block_align != 2 or byte_rate != sample_rate * block_align:
                raise _invalid_wav()
            if chunk_size != 16:
                if chunk_size < 18:
                    raise _invalid_wav()
                extension_size = struct.unpack_from("<H", data, payload_offset + 16)[0]
                if extension_size != chunk_size - 18:
                    raise _invalid_wav()
            format_found = True
        elif chunk_id == b"data":
            if not format_found or data_found or chunk_size == 0 or chunk_size % block_align:
                raise _invalid_wav()
            frame_count = chunk_size // block_align
            data_found = True
        offset = padded_end

    if not format_found or not data_found:
        raise _invalid_wav()
    if not 0 < frame_count / sample_rate <= max_seconds:
        raise _invalid_wav(
            f"La grabación debe durar más de 0 y como máximo {max_seconds:g} segundos.",
        )


async def read_validated_audio(audio: UploadFile) -> bytes:
    """Read at most the configured limit + 1 byte, validate, and close upload.

    Rejections use 413 for size, 415 for an unsupported media/audio format,
    422 for damaged or out-of-duration audio, and 503 for invalid server limits.
    """
    try:
        max_bytes, max_seconds = _configured_limits()
        _validate_media_type(audio)
        data = bytearray()
        while True:
            read_size = min(READ_CHUNK_BYTES, max_bytes + 1 - len(data))
            try:
                chunk = await audio.read(read_size)
            except OSError as error:
                raise _invalid_wav("No pudimos leer la grabación. Vuelve a grabarla.") from error
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="La grabación es demasiado grande. Graba un audio más corto.",
                )
        _validate_wav(data, max_seconds)
        return bytes(data)
    finally:
        await audio.close()
