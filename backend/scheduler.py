"""
Background scheduler – generates and inserts synthetic delivery data
into Supabase every 15 minutes, and auto-retrains the ML model periodically.
"""

import asyncio
import logging
from datetime import date

from generator import generate_daily_rows
from db import upsert_rows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

INTERVAL_SECONDS = 15 * 60  # 15 minutes
RETRAIN_EVERY_N_TICKS = 4   # retrain every 4 ticks (~1 hour)

_tick_count = 0


async def tick() -> None:
    """Run one generation + upsert cycle, and periodically retrain the model.

    - First run of the day → INSERTs new rows for all workers.
    - Subsequent runs same day → UPDATEs deliveries, earnings, hours, etc.
    - Every RETRAIN_EVERY_N_TICKS cycles → retrain the ML model.
    """
    global _tick_count
    _tick_count += 1

    today = date.today()
    rows = generate_daily_rows(target_date=today)
    logger.info("Generated %d rows for %s", len(rows), today.isoformat())

    try:
        upsert_rows(rows)
        logger.info(
            "Upserted %d rows into Supabase (insert or update).",
            len(rows),
        )
    except Exception as exc:
        logger.error("Supabase upsert failed: %s", exc)

    # Auto-retrain the ML model periodically
    if _tick_count % RETRAIN_EVERY_N_TICKS == 0:
        await run_retrain()


async def run_retrain() -> None:
    """Run the model retraining pipeline in a thread to avoid blocking."""
    logger.info("[scheduler] Starting auto-retrain...")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _retrain_sync)
        logger.info("[scheduler] Auto-retrain completed successfully.")
    except Exception as exc:
        logger.error("[scheduler] Auto-retrain failed: %s", exc)


def _retrain_sync() -> None:
    """Synchronous wrapper for the retrain pipeline."""
    from ml.retrain import retrain
    retrain()


async def run_scheduler() -> None:
    """Loop forever, calling tick() every INTERVAL_SECONDS."""
    logger.info(
        "Scheduler started — generating data every %d seconds (%d min). "
        "Model retrains every %d ticks.",
        INTERVAL_SECONDS,
        INTERVAL_SECONDS // 60,
        RETRAIN_EVERY_N_TICKS,
    )
    while True:
        await tick()
        await asyncio.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(run_scheduler())
