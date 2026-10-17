"""Long-running incremental updater (APScheduler). Run with `forecaster data schedule`."""

from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from forecaster.config import Settings
from forecaster.data.pipeline import scheduled_job
from forecaster.logging_setup import get_logger

log = get_logger(__name__)


def build_scheduler(s: Settings) -> BlockingScheduler:
    sched = BlockingScheduler(timezone="UTC")
    # Shortly after each 4h candle closes (and therefore after the daily close at 00:00 UTC).
    trigger = CronTrigger(hour=s.update_cron_hours, minute=s.update_cron_minute, timezone="UTC")
    sched.add_job(
        scheduled_job, trigger, args=[s], id="incremental_update",
        max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
    return sched


def run_scheduler(s: Settings, run_now: bool = True) -> None:
    if run_now:
        scheduled_job(s)
    sched = build_scheduler(s)
    log.info("scheduler started", extra={"hours": s.update_cron_hours, "minute": s.update_cron_minute})
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopped")
