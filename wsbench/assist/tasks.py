"""Run assist requests in the background, so the page never waits on one long HTTP request.

Each request is stored as JSON in WSB_HOME/assist_history/assist-<id>.json: the request, its status and, when it has
finished, the complete assist result with the model replies.
"""
import json
import os
import re
import time
from threading import Lock, Thread
from uuid import uuid4

from wsbench.paths import home

from .pipeline import AssistRequestError, _parameters, assist

KEEP = 20
TASK_ID = re.compile(r"[0-9a-f]{32}")
INTERRUPTED = "interrupted: the server restarted before the models answered"


class AssistTasks:
    def __init__(self):
        self._tasks = {}
        self._lock = Lock()

    @staticmethod
    def _folder():
        return home() / "assist_history"

    def start(self, request, submit):
        _parameters(request)                       # reject a bad request at once, with HTTP 400
        task_id = uuid4().hex
        record = dict(assist_id=task_id, status="running", created=time.time(), result=None, error=None,
                      request={k: request.get(k) for k in ("text", "model", "roles", "execute")})
        with self._lock:
            self._tasks[task_id] = record
            for old in sorted(self._tasks, key=lambda k: self._tasks[k]["created"])[:-KEEP]:
                del self._tasks[old]
        self._save(record)
        Thread(target=self._run, args=(task_id, request, submit), name="wsbench-assist", daemon=True).start()
        return task_id

    def _run(self, task_id, request, submit):
        try:
            update = dict(status="done", result=assist(request, submit))
        except AssistRequestError as exc:
            update = dict(status="error", error={"code": exc.code, "message": str(exc)})
        except Exception as exc:                   # the page must always get an answer
            update = dict(status="error", error={"code": "E_RUNTIME", "message": f"{type(exc).__name__}: {exc}"})
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            record.update(update)
        self._save(record)
        self._prune()

    def get(self, task_id):
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                return dict(record)
        return self._load(task_id)

    def latest(self):
        with self._lock:
            records = [dict(r) for r in self._tasks.values()]
        seen = {r["assist_id"] for r in records}
        try:
            names = [p.stem[len("assist-"):] for p in self._folder().glob("assist-*.json")]
        except OSError:
            names = []
        for name in names:
            if name not in seen:
                stored = self._load(name)
                if stored is not None:
                    records.append(stored)
        return max(records, key=lambda r: r["created"], default=None)

    def _save(self, record):
        try:
            folder = self._folder()
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f".assist-{record['assist_id']}.tmp"
            tmp.write_text(json.dumps(record), "utf-8")
            os.replace(tmp, folder / f"assist-{record['assist_id']}.json")
        except (OSError, TypeError, ValueError):
            pass

    def _load(self, task_id):
        if not TASK_ID.fullmatch(task_id or ""):
            return None
        try:
            record = json.loads((self._folder() / f"assist-{task_id}.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if record.get("status") == "running":      # written by a process that no longer exists
            record.update(status="error", error={"code": "E_RUNTIME", "message": INTERRUPTED})
        return record

    def _prune(self):
        try:
            files = sorted(self._folder().glob("assist-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[KEEP:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass


tasks = AssistTasks()
