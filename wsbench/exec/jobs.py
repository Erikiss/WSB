"""Single-worker FIFO queue and bounded, process-local result storage."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Lock, Thread
from uuid import uuid4

from .executor import run_script
from .verdict import agreement, failed, prepare


class JobError(Exception):
    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass
class Job:
    script: str
    predictions: dict | None
    verdicts: list
    id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "queued"
    stop: Event = field(default_factory=Event)
    states: dict | None = field(default_factory=dict)


class Jobs:
    def __init__(self):
        self._jobs = OrderedDict()
        self._queue = Queue()
        self._lock = Lock()
        self._worker = None

    def submit(self, script, predictions=None):
        _, verdicts = prepare(script, predictions)
        job = Job(script, predictions, verdicts)
        with self._lock:
            self._jobs[job.id] = job
            if len(self._jobs) > 5:
                self._jobs[list(self._jobs)[-6]].states = None
            if len(self._jobs) > 50:
                self._jobs.popitem(last=False)
            self._queue.put(job)
            if self._worker is None:
                self._worker = Thread(target=self._work, name="wsbench-worker", daemon=True)
                self._worker.start()
        return job.id

    def _get(self, run_id):
        job = self._jobs.get(run_id)
        if job is None:
            raise JobError(404, "E_NOT_FOUND", "unknown run")
        return job

    def snapshot(self, run_id):
        with self._lock:
            job = self._get(run_id)
            verdicts = deepcopy(job.verdicts)
            current = next((v["line"] for v in verdicts if v["status"] == "RUNNING"), None)
            return dict(run_id=job.id, status=job.status, verdicts=verdicts, current_line=current)

    def cancel(self, run_id):
        with self._lock:
            job = self._get(run_id)
            if job.status in ("queued", "running"):
                job.stop.set()
                if job.status == "queued":
                    self._end_unfinished(job, "cancelled")
                    job.status = "cancelled"
        return {"ok": True}

    @staticmethod
    def _end_unfinished(job, message):
        for verdict in job.verdicts:
            if verdict["status"] in ("RUNNING", "PENDING"):
                failed(verdict, "E_RUNTIME", message)
                agreement(verdict, job.predictions)

    def _work(self):
        while True:
            job = self._queue.get()
            try:
                with self._lock:
                    if job.status == "cancelled":
                        continue
                    job.status = "running"
                self._execute(job)
            except Exception as exc:
                with self._lock:
                    self._end_unfinished(job, str(exc))
            finally:
                with self._lock:
                    job.status = "cancelled" if job.stop.is_set() else "done"
                self._queue.task_done()

    def _execute(self, job):
        pending = {}

        def on_state(label, line, weights, norm_kind):
            pending[line] = (weights, norm_kind)

        def on_update(snapshot):
            with self._lock:
                if job.states is not None:
                    for row in snapshot:
                        line = row["line"]
                        if line in pending:
                            weights, norm_kind = pending[line]
                            changed = {t["name"]: weights[t["name"]] for t in row["metrics"]["changed_tensors"]}
                            job.states[line] = row["model"], changed, norm_kind
                pending.clear()
                job.verdicts = snapshot

        run_script(job.script, on_update=on_update, should_stop=job.stop.is_set,
                   on_state=on_state, predictions=job.predictions)

    def tensors(self, run_id, line, name):
        with self._lock:
            states = self._get(run_id).states
            if states is None:
                raise JobError(410, "E_EXPIRED", "state expired; re-run the script")
            if line not in states:
                raise JobError(404, "E_NOT_FOUND", "no tensor state for this line")
            model, changed, _ = states[line]
        from wsbench.models.registry import load_model
        before = load_model(model).W0.get(name)
        after = changed.get(name, before)
        if before is None and after is None:
            raise JobError(404, "E_NOT_FOUND", "unknown tensor name")
        return before, after


jobs = Jobs()
