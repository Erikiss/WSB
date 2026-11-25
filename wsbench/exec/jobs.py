"""Single-worker FIFO queue with bounded result storage, in memory and as JSON files under WSB_HOME/history."""
import json
import os
import re
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Lock, Thread
from uuid import uuid4

from wsbench.paths import home

from .executor import run_script
from .verdict import agreement, failed, prepare

KEEP = 50
SAVE_EVERY_S = 2.0
RUN_ID = re.compile(r"[0-9a-f]{32}")
INTERRUPTED = "interrupted: the server restarted before this statement finished"


def _history():
    return home() / "history"


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
    source: str = "editor"
    created: float = field(default_factory=time.time)
    saved: float = 0.0


class Jobs:
    def __init__(self):
        self._jobs = OrderedDict()
        self._queue = Queue()
        self._lock = Lock()
        self._worker = None

    def submit(self, script, predictions=None, source="editor"):
        _, verdicts = prepare(script, predictions)
        job = Job(script, predictions, verdicts, source=source)
        with self._lock:
            self._jobs[job.id] = job
            if len(self._jobs) > 5:
                self._jobs[list(self._jobs)[-6]].states = None
            if len(self._jobs) > KEEP:
                evicted, _ = self._jobs.popitem(last=False)
                self._forget(evicted)
            self._queue.put(job)
            if self._worker is None:
                self._worker = Thread(target=self._work, name="wsbench-worker", daemon=True)
                self._worker.start()
        self._save(job)
        self._prune()
        return job.id

    def _get(self, run_id):
        job = self._jobs.get(run_id)
        if job is None:
            raise JobError(404, "E_NOT_FOUND", "unknown run")
        return job

    @staticmethod
    def _record(job):
        verdicts = deepcopy(job.verdicts)
        current = next((v["line"] for v in verdicts if v["status"] == "RUNNING"), None)
        return dict(run_id=job.id, status=job.status, verdicts=verdicts, current_line=current,
                    script=job.script, source=job.source, created=job.created)

    def snapshot(self, run_id):
        with self._lock:
            job = self._jobs.get(run_id)
            if job is not None:
                return self._record(job)
        stored = self._load(run_id)
        if stored is None:
            raise JobError(404, "E_NOT_FOUND", "unknown run")
        return stored

    def recent(self, limit=20):
        """Newest first: runs of this process and runs stored by earlier processes."""
        with self._lock:
            live = {job.id: self._record(job) for job in self._jobs.values()}
        rows = dict(live)
        try:
            names = [p.stem for p in _history().glob("*.json")]
        except OSError:
            names = []
        for name in names:
            if name not in rows:
                stored = self._load(name)
                if stored is not None:
                    rows[name] = stored
        ordered = sorted(rows.values(), key=lambda r: r["created"], reverse=True)[:limit]
        return [dict(run_id=r["run_id"], status=r["status"], source=r["source"], created=r["created"],
                     script=r["script"]) for r in ordered]

    # ---- history files -------------------------------------------------------------------------------------------

    def _save(self, job):
        with self._lock:
            record = self._record(job)
            record["predictions"] = job.predictions
            job.saved = time.monotonic()
        try:
            folder = _history()
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f".{job.id}.tmp"
            tmp.write_text(json.dumps(record), "utf-8")
            os.replace(tmp, folder / f"{job.id}.json")
        except (OSError, TypeError, ValueError):
            pass                      # history is a convenience; a full disk must not break a run

    @staticmethod
    def _load(run_id):
        if not RUN_ID.fullmatch(run_id or ""):
            return None
        try:
            record = json.loads((_history() / f"{run_id}.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        predictions = record.pop("predictions", None)
        if record.get("status") in ("queued", "running"):      # written by a process that no longer exists
            for verdict in record["verdicts"]:
                if verdict["status"] in ("RUNNING", "PENDING"):
                    failed(verdict, "E_RUNTIME", INTERRUPTED)
                    agreement(verdict, predictions)
            record.update(status="cancelled", current_line=None)
        return record

    @staticmethod
    def _forget(run_id):
        try:
            (_history() / f"{run_id}.json").unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _prune():
        try:
            files = sorted(_history().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[KEEP:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass

    def cancel(self, run_id):
        with self._lock:
            job = self._jobs.get(run_id)
            if job is not None and job.status in ("queued", "running"):
                job.stop.set()
                if job.status == "queued":
                    self._end_unfinished(job, "cancelled")
                    job.status = "cancelled"
        if job is None:
            if self._load(run_id) is None:
                raise JobError(404, "E_NOT_FOUND", "unknown run")
            return {"ok": True}       # a stored run of an earlier process: nothing left to cancel
        self._save(job)
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
                self._save(job)
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
                due = time.monotonic() - job.saved >= SAVE_EVERY_S
            if due:
                self._save(job)

        run_script(job.script, on_update=on_update, should_stop=job.stop.is_set,
                   on_state=on_state, predictions=job.predictions)

    def tensors(self, run_id, line, name):
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None and self._load(run_id) is None:
                raise JobError(404, "E_NOT_FOUND", "unknown run")
            states = None if job is None else job.states
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
