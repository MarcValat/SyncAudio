"""A minimal in-memory background job registry, for progress-tracked long operations.

Each job runs its target function in its own thread; the function receives a
``log`` callback (the same one the CLI already passes to ``plan_corrections``
etc. to print "[analyse] ..." lines) which appends to a lock-protected
message list instead. A WebSocket handler polls that list and streams new
messages to the client as they arrive. There's no message broker or
persistence: this is a single-process local sidecar, not a distributed
system, and jobs are lost on restart -- acceptable for a tool a person is
actively watching run.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

JobStatus = Literal["running", "done", "error"]


@dataclass
class Job:
    id: str
    status: JobStatus = "running"
    messages: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, message: str) -> None:
        with self._lock:
            self.messages.append(message)

    def snapshot(self, since: int = 0) -> tuple[list[str], int, JobStatus, Any, str | None]:
        """New messages since index ``since``, the new cursor, and the job's current state."""
        with self._lock:
            return list(self.messages[since:]), len(self.messages), self.status, self.result, self.error

    def finish(self, result: Any) -> None:
        with self._lock:
            self.result = result
            self.status = "done"

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.status = "error"


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def start_job(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Job:
    """Run ``fn(*args, log=job.log, **kwargs)`` in a background thread; return its ``Job`` immediately."""
    job = Job(id=uuid.uuid4().hex)
    with _jobs_lock:
        _jobs[job.id] = job

    def runner() -> None:
        try:
            result = fn(*args, log=job.log, **kwargs)
            job.finish(result)
        except Exception as exc:  # noqa: BLE001 - a background thread has no other way to surface a failure
            job.fail(str(exc))

    threading.Thread(target=runner, daemon=True).start()
    return job


def get_job(job_id: str) -> Job | None:
    with _jobs_lock:
        return _jobs.get(job_id)
