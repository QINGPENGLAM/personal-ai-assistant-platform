from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from .jobs import JobService

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, jobs: JobService, enqueue):
        self.jobs = jobs
        self.enqueue = enqueue
        self._task = None

    async def tick(self):
        if self.jobs.repository.database.storage.status()["active"]:
            return
        now = self.jobs.clock()
        database = self.jobs.repository.database
        with database.read() as connection:
            agents = connection.execute(
                """SELECT DISTINCT agent_id FROM jobs WHERE enabled = 1 AND next_run_at <=
                   ? ORDER BY agent_id""",
                (now,),
            ).fetchall()
        for identity in agents:
            agent_id = identity["agent_id"]
            changed = False
            async with self.jobs.coordination.agent_locks([agent_id]):
                with database.transaction() as connection:
                    if connection.execute(
                        "SELECT protection_latched FROM platform_storage_state WHERE id=1"
                    ).fetchone()[0]:
                        continue
                    agent = connection.execute(
                        "SELECT * FROM agents WHERE id = ?", (agent_id,)
                    ).fetchone()
                    rows = connection.execute(
                        """SELECT * FROM jobs WHERE agent_id = ? AND enabled=1 AND next_run_at<=?
                           ORDER BY next_run_at,id""",
                        (agent_id, now),
                    ).fetchall()
                    for job in rows:
                        queued = connection.execute(
                            "SELECT COUNT(*) FROM runs WHERE agent_id=? AND status='queued'",
                            (agent_id,),
                        ).fetchone()[0]
                        active = connection.execute(
                            """SELECT 1 FROM runs WHERE job_id=? AND status IN
                               ('queued','starting','running','stopping')""",
                            (job["id"],),
                        ).fetchone()
                        if queued >= 20 or active:
                            continue
                        interval = job["interval_seconds"]
                        missed = (now - job["next_run_at"]) // (interval * 1000) if interval else 0
                        scheduled = (
                            job["next_run_at"] + missed * interval * 1000
                            if interval
                            else job["next_run_at"]
                        )
                        run_id = str(uuid4())
                        connection.execute(
                            """INSERT INTO runs(id,agent_id,source,job_id,prompt,scheduled_for,
                               coalesced_count,status,model,created_at)
                               VALUES (?,?,'job',?,?,?,?, 'queued',?,?)""",
                            (
                                run_id,
                                agent_id,
                                job["id"],
                                job["prompt"],
                                scheduled,
                                missed,
                                agent["model"],
                                now,
                            ),
                        )
                        connection.execute(
                            "UPDATE jobs SET enabled=?, next_run_at=? WHERE id=?",
                            (
                                int(interval is not None),
                                scheduled + interval * 1000 if interval else None,
                                job["id"],
                            ),
                        )
                        payload = json.dumps(
                            {
                                "status": "queued",
                                "scheduled_for": scheduled,
                                "coalesced_count": missed,
                            }
                        )
                        connection.execute(
                            """INSERT INTO events(run_id,sequence,kind,payload_json,created_at)
                               VALUES (?,1,'state',?,?)""",
                            (run_id, payload, now),
                        )
                        changed = True
            if changed:
                self.enqueue(agent_id)

    def start(self):
        self._task = asyncio.create_task(self._loop(), name="job-scheduler")

    async def _loop(self):
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("Scheduled tasks could not be queued; the next tick will retry.")
            await asyncio.sleep(1)

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
