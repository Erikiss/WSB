"""Background explore loops with stored state (WSB_HOME/explore_history/explore-<id>.json)."""
import json
import os
import re
import time
from copy import deepcopy
from threading import Event, Lock, Thread
from uuid import uuid4

from wsbench.paths import home

from .explore import explore, parameters

KEEP = 20
TASK_ID = re.compile(r"[0-9a-f]{32}")
INTERRUPTED = "interrupted: the server restarted"


class ExploreTasks:
    def __init__(self):
        self._tasks = {}
        self._control = {}
        self._lock = Lock()

    @staticmethod
    def _folder():
        return home() / "explore_history"

    def start(self, request, submit, snapshot, cancel_run):
        parameters(request)                        # reject a bad request at once, with HTTP 400
        task_id = uuid4().hex
        record = dict(explore_id=task_id, status="running", created=time.time(), revision=0, state=None, error=None,
                      request={k: request.get(k) for k in ("text", "model", "rounds", "budget_usd", "roles", "planner")})
        control = dict(stop=Event(), answered=Event(), answer=None)
        with self._lock:
            self._tasks[task_id] = record
            self._control[task_id] = control
            for old in sorted(self._tasks, key=lambda k: self._tasks[k]["created"])[:-KEEP]:
                del self._tasks[old]
                self._control.pop(old, None)
        self._save(task_id)
        Thread(target=self._run, args=(task_id, request, submit, snapshot, cancel_run, control),
               name="wsbench-explore", daemon=True).start()
        return task_id

    def _run(self, task_id, request, submit, snapshot, cancel_run, control):
        def on_update(state):
            with self._lock:
                record = self._tasks.get(task_id)
                if record is None:
                    return
                record["state"] = deepcopy(state)
                record["revision"] += 1
                record["status"] = "waiting" if "waiting" in state else "running"
            self._save(task_id)

        def ask(question):
            control["answered"].clear()
            while not control["answered"].wait(0.2):
                if control["stop"].is_set():
                    return None
            return control["answer"]

        try:
            state = explore(request, submit, snapshot, cancel_run, on_update=on_update,
                            should_stop=control["stop"].is_set, ask=ask)
            update = dict(status="cancelled" if state["stopped"] == "cancelled" else "done", state=state)
        except Exception as exc:                   # the page must always get an answer
            update = dict(status="error", error={"code": "E_RUNTIME", "message": f"{type(exc).__name__}: {exc}"})
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return
            record.update(update)
            record["revision"] += 1
        self._save(task_id)
        self._prune()

    def get(self, task_id):
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                return deepcopy(record)
        return self._load(task_id)

    def answer(self, task_id, text):
        with self._lock:
            record, control = self._tasks.get(task_id), self._control.get(task_id)
            if record is None or control is None:
                return None
            if record["status"] != "waiting":
                return False
            control["answer"] = text
            control["answered"].set()
        return True

    def cancel(self, task_id):
        with self._lock:
            control = self._control.get(task_id)
        if control is None:
            return self._load(task_id) is not None
        control["stop"].set()
        return True

    def latest(self):
        with self._lock:
            records = [deepcopy(r) for r in self._tasks.values()]
        seen = {r["explore_id"] for r in records}
        try:
            names = [p.stem[len("explore-"):] for p in self._folder().glob("explore-*.json")]
        except OSError:
            names = []
        for name in names:
            if name not in seen:
                stored = self._load(name)
                if stored is not None:
                    records.append(stored)
        return max(records, key=lambda r: r["created"], default=None)

    def _save(self, task_id):
        with self._lock:
            record = deepcopy(self._tasks.get(task_id))
        if record is None:
            return
        try:
            folder = self._folder()
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f".explore-{task_id}.tmp"
            tmp.write_text(json.dumps(record), "utf-8")
            os.replace(tmp, folder / f"explore-{task_id}.json")
        except (OSError, TypeError, ValueError):
            pass

    def _load(self, task_id):
        if not TASK_ID.fullmatch(task_id or ""):
            return None
        try:
            record = json.loads((self._folder() / f"explore-{task_id}.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if record.get("status") in ("running", "waiting"):      # written by a process that no longer exists
            record.update(status="error", error={"code": "E_RUNTIME", "message": INTERRUPTED})
        return record

    def _prune(self):
        try:
            files = sorted(self._folder().glob("explore-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[KEEP:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass


tasks = ExploreTasks()
