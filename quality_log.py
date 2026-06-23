"""Fluentia AI — Quality logging to SQLite (F4)."""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("QUALITY_DB_PATH", "quality_log.db")
_db_initialized = False


async def _init_db():
    global _db_initialized
    if _db_initialized:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS ai_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                function_name TEXT NOT NULL,
                input_text TEXT,
                output_raw TEXT,
                output_valid INTEGER NOT NULL DEFAULT 1,
                retry_count INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                error TEXT
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_calls_timestamp ON ai_calls(timestamp)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_calls_function ON ai_calls(function_name)"
        )
        await db.commit()
    _db_initialized = True


async def log_ai_call(
    function_name: str,
    input_text: str,
    output_raw: str,
    output_valid: bool,
    retry_count: int,
    latency_ms: int,
    error: str | None = None,
):
    await _init_db()
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """INSERT INTO ai_calls
                   (timestamp, function_name, input_text, output_raw,
                    output_valid, retry_count, latency_ms, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    function_name,
                    input_text[:2000],
                    output_raw[:5000],
                    1 if output_valid else 0,
                    retry_count,
                    latency_ms,
                    error,
                ),
            )
            await db.commit()
    except Exception as e:
        logger.error("Failed to log AI call: %s", e)


async def get_metrics(hours: int = 24) -> dict:
    await _init_db()
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN output_valid = 1 THEN 1 ELSE 0 END) as valid, "
            "SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as retried, "
            "SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors, "
            "AVG(latency_ms) as avg_latency "
            "FROM ai_calls WHERE timestamp >= ?",
            (since,),
        )
        row = await cursor.fetchone()
        total = row[0] or 0
        valid = row[1] or 0
        retried = row[2] or 0
        errors = row[3] or 0
        avg_latency = row[4] or 0

        cursor = await db.execute(
            "SELECT function_name, COUNT(*) as calls, "
            "SUM(CASE WHEN output_valid = 1 THEN 1 ELSE 0 END) as fn_valid, "
            "AVG(latency_ms) as fn_latency "
            "FROM ai_calls WHERE timestamp >= ? "
            "GROUP BY function_name",
            (since,),
        )
        by_function = {}
        async for frow in cursor:
            fn_calls = frow[1] or 0
            fn_valid = frow[2] or 0
            by_function[frow[0]] = {
                "calls": fn_calls,
                "valid_rate": round(fn_valid / fn_calls * 100, 1) if fn_calls else 0,
                "avg_latency_ms": round(frow[3] or 0),
            }

    return {
        "period_hours": hours,
        "total_calls": total,
        "valid_rate": round(valid / total * 100, 1) if total else 0,
        "retry_rate": round(retried / total * 100, 1) if total else 0,
        "error_rate": round(errors / total * 100, 1) if total else 0,
        "avg_latency_ms": round(avg_latency),
        "by_function": by_function,
    }


async def cleanup_old_records(days: int = 30):
    await _init_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            result = await db.execute(
                "DELETE FROM ai_calls WHERE timestamp < ?", (cutoff,)
            )
            await db.commit()
            logger.info("Cleaned up %d old quality log records", result.rowcount)
    except Exception as e:
        logger.error("Failed to cleanup old records: %s", e)
