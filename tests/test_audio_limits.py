"""Audio upload validation tests; no API, secret, network or database access."""

import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from audio_limits import (
    DEFAULT_MAX_AUDIO_BYTES, HARD_MAX_AUDIO_BYTES, READ_CHUNK_BYTES, read_validated_audio,
)


def pcm_wav(frames=160, channels=1, sample_rate=16_000, sample_width=2):
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\0" * frames * channels * sample_width)
    return output.getvalue()


def chunk(chunk_id, payload, declared_size=None, padded=True):
    size = len(payload) if declared_size is None else declared_size
    return chunk_id + struct.pack("<I", size) + payload + (b"\0" if size & 1 and padded else b"")


def riff(*chunks):
    content = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(content)) + content


PCM_FORMAT = struct.pack("<HHIIHH", 1, 1, 16_000, 32_000, 2, 16)
FMT = chunk(b"fmt ", PCM_FORMAT)
DATA = chunk(b"data", b"\0\0" * 160)


class FakeUpload:
    def __init__(self, data, filename="audio.wav", content_type="audio/wav", fragment_size=None):
        self.stream = io.BytesIO(data)
        self.filename = filename
        self.content_type = content_type
        self.fragment_size = fragment_size
        self.read_sizes = []
        self.bytes_read = 0
        self.close_calls = 0

    async def read(self, size=-1):
        self.read_sizes.append(size)
        if size <= 0:
            raise AssertionError("Audio reader must always request a positive bounded size")
        result = self.stream.read(min(size, self.fragment_size) if self.fragment_size else size)
        self.bytes_read += len(result)
        return result

    async def close(self):
        self.close_calls += 1
        self.stream.close()


class AudioLimitsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Every test sets limits explicitly rather than reading a local .env.
        self.environment = patch.dict(os.environ, {
            "MAX_AUDIO_BYTES": str(DEFAULT_MAX_AUDIO_BYTES), "MAX_AUDIO_SECONDS": "30",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    async def reject(self, data, status, **upload_kwargs):
        upload = FakeUpload(data, **upload_kwargs)
        with self.assertRaises(HTTPException) as raised:
            await read_validated_audio(upload)
        self.assertEqual(raised.exception.status_code, status)
        self.assertEqual(upload.close_calls, 1)
        return upload

    async def test_real_upload_file_is_validated_and_closed(self):
        data = pcm_wav()
        with tempfile.TemporaryFile() as file:
            file.write(data)
            file.seek(0)
            upload = UploadFile(file=file, filename="audio.wav", headers=Headers({"content-type": "audio/wav"}))
            self.assertEqual(await read_validated_audio(upload), data)
            self.assertTrue(file.closed)

    async def test_valid_pcm_with_silence_is_accepted(self):
        data = pcm_wav()
        upload = FakeUpload(data)
        self.assertEqual(await read_validated_audio(upload), data)
        self.assertEqual(upload.close_calls, 1)

    async def test_all_supported_media_types_are_accepted(self):
        data = pcm_wav()
        for mime in ["audio/wav", "audio/x-wav", "audio/wave", "Audio/WAV; charset=binary", "application/octet-stream"]:
            with self.subTest(mime=mime):
                upload = FakeUpload(data, filename="AUDIO.WAV", content_type=mime)
                self.assertEqual(await read_validated_audio(upload), data)
                self.assertEqual(upload.close_calls, 1)

    async def test_wav_media_type_does_not_need_filename_extension(self):
        for filename in [None, "recording", "recording.mp3"]:
            with self.subTest(filename=filename):
                self.assertEqual(await read_validated_audio(FakeUpload(pcm_wav(), filename=filename)), pcm_wav())

    async def test_unsupported_mime_is_rejected_before_reading(self):
        for mime in ["audio/mpeg", "text/plain", "image/wav", "", None]:
            with self.subTest(mime=mime):
                upload = await self.reject(pcm_wav(), 415, content_type=mime)
                self.assertEqual(upload.read_sizes, [])

    async def test_octet_stream_requires_wav_extension(self):
        for filename in [None, "", "audio.mp3", "audio.wav.exe"]:
            with self.subTest(filename=filename):
                upload = await self.reject(pcm_wav(), 415, content_type="application/octet-stream", filename=filename)
                self.assertEqual(upload.read_sizes, [])

    async def test_wav_filename_and_mime_cannot_disguise_mp3(self):
        for mime in ["audio/wav", "application/octet-stream"]:
            with self.subTest(mime=mime):
                await self.reject(b"ID3" + b"fake MP3 data" * 5, 415, content_type=mime)

    async def test_incomplete_or_empty_header_is_rejected(self):
        for data in [b"", b"RIFF", pcm_wav()[:11]]:
            with self.subTest(length=len(data)):
                await self.reject(data, 422)

    async def test_non_wave_container_is_rejected(self):
        for data in [b"RIFX" + pcm_wav()[4:], pcm_wav()[:8] + b"AVI " + pcm_wav()[12:]]:
            with self.subTest(magic=data[:12]):
                await self.reject(data, 415)

    async def test_riff_declared_size_must_match_upload(self):
        data = pcm_wav()
        for size in [0, len(data) - 9, len(data) - 7, 0xFFFFFFFF]:
            with self.subTest(size=size):
                await self.reject(data[:4] + struct.pack("<I", size) + data[8:], 422)

    async def test_declared_chunk_size_bomb_is_rejected(self):
        for chunk_id in [b"fmt ", b"data", b"JUNK"]:
            with self.subTest(chunk_id=chunk_id):
                await self.reject(riff(FMT, chunk(chunk_id, b"\0\0", declared_size=0xFFFFFFFE)), 422)

    async def test_truncated_data_is_rejected_even_with_adjusted_riff_size(self):
        data = pcm_wav()[:-2]
        data = data[:4] + struct.pack("<I", len(data) - 8) + data[8:]
        await self.reject(data, 422)

    async def test_partial_chunk_header_is_rejected(self):
        for trailing in [b"x", b"data", b"data\0\0\0"]:
            with self.subTest(trailing=trailing):
                await self.reject(riff(FMT, DATA, trailing), 422)

    async def test_missing_format_or_data_is_rejected(self):
        for data in [riff(), riff(FMT), riff(DATA), riff(chunk(b"JUNK", b""))]:
            with self.subTest(data=data[:20]):
                await self.reject(data, 422)

    async def test_duplicate_format_or_data_is_rejected(self):
        for data in [riff(FMT, FMT, DATA), riff(FMT, DATA, DATA), riff(DATA, FMT)]:
            with self.subTest(size=len(data)):
                await self.reject(data, 422)

    async def test_incomplete_format_chunk_is_rejected(self):
        for size in [0, 14, 15, 17]:
            with self.subTest(size=size):
                await self.reject(riff(chunk(b"fmt ", (PCM_FORMAT + b"\0")[:size]), DATA), 422)

    async def test_valid_pcm_format_extension_is_accepted(self):
        data = riff(chunk(b"fmt ", PCM_FORMAT + b"\0\0"), DATA)
        self.assertEqual(await read_validated_audio(FakeUpload(data)), data)

    async def test_format_extension_size_must_match_payload(self):
        for extra in [struct.pack("<H", 1), struct.pack("<H", 0) + b"xx"]:
            with self.subTest(extra=extra):
                await self.reject(riff(chunk(b"fmt ", PCM_FORMAT + extra), DATA), 422)

    async def test_pcm_alignment_and_byte_rate_must_match_actual_format(self):
        for byte_rate, alignment in [(31_999, 2), (0, 2), (32_000, 0), (32_000, 1), (32_000, 4)]:
            with self.subTest(byte_rate=byte_rate, alignment=alignment):
                format_data = struct.pack("<HHIIHH", 1, 1, 16_000, byte_rate, alignment, 16)
                await self.reject(riff(chunk(b"fmt ", format_data), DATA), 422)

    async def test_compressed_or_float_audio_is_unsupported(self):
        for compression in [2, 3, 6, 7, 0xFFFE]:
            with self.subTest(compression=compression):
                format_data = struct.pack("<HHIIHH", compression, 1, 16_000, 32_000, 2, 16)
                await self.reject(riff(chunk(b"fmt ", format_data), DATA), 415)

    async def test_stereo_and_other_sample_rates_or_bit_depths_are_unsupported(self):
        for kwargs in [{"channels": 2}, {"sample_rate": 8_000}, {"sample_rate": 48_000},
                       {"sample_width": 1}, {"sample_width": 3}, {"sample_width": 4}]:
            with self.subTest(format=kwargs):
                await self.reject(pcm_wav(**kwargs), 415)

    async def test_empty_and_unaligned_data_is_rejected(self):
        for payload in [b"", b"x", b"xxx"]:
            with self.subTest(size=len(payload)):
                await self.reject(riff(FMT, chunk(b"data", payload)), 422)

    async def test_metadata_chunks_and_padding_are_accepted(self):
        data = riff(chunk(b"JUNK", b"x"), FMT, chunk(b"LIST", b"info"), DATA, chunk(b"JUNK", b""))
        self.assertEqual(await read_validated_audio(FakeUpload(data)), data)

    async def test_missing_odd_chunk_padding_is_rejected(self):
        await self.reject(riff(FMT, DATA, chunk(b"JUNK", b"x", padded=False)), 422)

    async def test_thirty_second_boundary_is_accepted(self):
        data = pcm_wav(frames=30 * 16_000)
        upload = FakeUpload(data)
        self.assertEqual(await read_validated_audio(upload), data)
        self.assertTrue(all(0 < size <= READ_CHUNK_BYTES for size in upload.read_sizes))

    async def test_one_frame_over_thirty_seconds_is_rejected(self):
        await self.reject(pcm_wav(frames=30 * 16_000 + 1), 422)

    async def test_configured_shorter_duration_is_enforced(self):
        with patch.dict(os.environ, {"MAX_AUDIO_SECONDS": "0.01"}):
            data = pcm_wav(frames=160)
            self.assertEqual(await read_validated_audio(FakeUpload(data)), data)
            await self.reject(pcm_wav(frames=161), 422)

    async def test_one_frame_is_valid_and_duration_is_derived_from_payload(self):
        data = pcm_wav(frames=1)
        self.assertEqual(await read_validated_audio(FakeUpload(data)), data)

    async def test_byte_limit_boundary_is_inclusive(self):
        data = pcm_wav()
        with patch.dict(os.environ, {"MAX_AUDIO_BYTES": str(len(data))}):
            self.assertEqual(await read_validated_audio(FakeUpload(data)), data)
        with patch.dict(os.environ, {"MAX_AUDIO_BYTES": str(len(data) - 1)}):
            upload = await self.reject(data, 413)
            self.assertEqual(upload.bytes_read, len(data))

    async def test_excessive_upload_reads_only_limit_plus_one(self):
        for limit in [1, 100, READ_CHUNK_BYTES, DEFAULT_MAX_AUDIO_BYTES]:
            with self.subTest(limit=limit), patch.dict(os.environ, {"MAX_AUDIO_BYTES": str(limit)}):
                upload = await self.reject(b"x" * (limit + 50_000), 413)
                self.assertEqual(upload.bytes_read, limit + 1)
                self.assertTrue(all(0 < size <= READ_CHUNK_BYTES for size in upload.read_sizes))

    async def test_fragmented_reads_are_collected_until_eof(self):
        data = pcm_wav()
        upload = FakeUpload(data, fragment_size=7)
        self.assertEqual(await read_validated_audio(upload), data)
        self.assertEqual(upload.bytes_read, len(data))
        self.assertGreater(len(upload.read_sizes), 1)

    async def test_invalid_byte_limit_fails_closed_without_reading(self):
        for value in ["", "invalid", "1.5", "0", "-1", str(HARD_MAX_AUDIO_BYTES + 1)]:
            with self.subTest(value=value), patch.dict(os.environ, {"MAX_AUDIO_BYTES": value}):
                upload = await self.reject(pcm_wav(), 503)
                self.assertEqual(upload.read_sizes, [])

    async def test_invalid_duration_limit_fails_closed_without_reading(self):
        for value in ["", "invalid", "0", "-1", "nan", "inf", "-inf", "30.001"]:
            with self.subTest(value=value), patch.dict(os.environ, {"MAX_AUDIO_SECONDS": value}):
                upload = await self.reject(pcm_wav(), 503)
                self.assertEqual(upload.read_sizes, [])

    async def test_hard_byte_limit_itself_is_valid_configuration(self):
        with patch.dict(os.environ, {"MAX_AUDIO_BYTES": str(HARD_MAX_AUDIO_BYTES)}):
            self.assertEqual(await read_validated_audio(FakeUpload(pcm_wav())), pcm_wav())

    async def test_upload_is_closed_when_read_fails(self):
        class BrokenUpload(FakeUpload):
            async def read(self, size):
                raise OSError("Simulated upload failure")

        upload = BrokenUpload(pcm_wav())
        with self.assertRaises(HTTPException) as raised:
            await read_validated_audio(upload)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(upload.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
