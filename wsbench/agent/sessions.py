"""Conversations with the Claude Managed Agent. Anthropic hosts the agent loop; WSBench executes the tool calls.

One conversation = one Managed Agents session. Each turn runs in a background thread: it opens the event stream,
sends the researcher's message, answers the agent's custom tool calls from the local executor and records a
transcript, which is stored in WSB_HOME/agent_history/ and served to the page.
"""
import json
import os
import re
import time
from copy import deepcopy
from threading import Event, Lock, Thread
from uuid import uuid4

from wsbench.paths import home

from . import spec as agent_spec
from . import tools as agent_tools

KEEP = 20
MAX_BUDGET_USD = 20.0
CONV_ID = re.compile(r"[0-9a-f]{32}")
INTERRUPTED = "interrupted: the server restarted during this turn. Start a new conversation."
CONSOLE_URL = "https://platform.claude.com/workspaces/default/sessions/{}"


class AgentError(Exception):
    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code, self.code = status_code, code


def available():
    return os.getenv("WSB_LLM") == "mock" or bool(os.getenv("ANTHROPIC_API_KEY"))


def _text(blocks):
    return "".join(getattr(b, "text", "") for b in (blocks or []) if getattr(b, "type", None) == "text")


def _cents(list_cost):
    try:
        return int(getattr(list_cost, "amount", None) or list_cost["amount"])
    except (TypeError, KeyError, ValueError):
        return None


class AgentSessions:
    def __init__(self, client_factory=None):
        self._records, self._control = {}, {}
        self._lock = Lock()
        self._setup_lock = Lock()
        self._client_factory = client_factory
        self._client = None

    # ---- Anthropic resources ------------------------------------------------------------------------------------

    def client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            elif os.getenv("WSB_LLM") == "mock":
                from .mock import MockPlatform
                self._client = MockPlatform()
            else:
                import anthropic
                self._client = anthropic.Anthropic(max_retries=2)
        return self._client

    @staticmethod
    def _ids_path():
        return home() / "agent.json"

    def ensure_agent(self):
        """Create the environment and the agent once and reuse their IDs; update the agent when its spec changed."""
        with self._setup_lock:
            path = self._ids_path()
            try:
                ids = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                ids = {}
            client, spec, digest = self.client(), agent_spec.spec(), agent_spec.spec_hash()
            if not ids.get("environment_id"):
                environment = client.beta.environments.create(
                    name="wsbench", config={"type": "cloud", "networking": {"type": "limited"}})
                ids["environment_id"] = environment.id
            if not ids.get("agent_id"):
                agent = client.beta.agents.create(**spec)
                ids.update(agent_id=agent.id, version=agent.version, spec_hash=digest)
            elif ids.get("spec_hash") != digest:
                agent = client.beta.agents.update(ids["agent_id"], **spec)
                ids.update(version=agent.version, spec_hash=digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(ids), "utf-8")
            return ids

    # ---- public API ---------------------------------------------------------------------------------------------

    def start(self, text, budget_usd, jobs):
        if not available():
            raise AgentError(400, "E_NO_KEY", "ANTHROPIC_API_KEY is not set on the server")
        if not isinstance(text, str) or not text.strip():
            raise AgentError(400, "E_BAD_REQUEST", "text must be a non-empty string")
        if isinstance(budget_usd, bool) or not isinstance(budget_usd, (int, float)) or not 0.5 <= budget_usd <= MAX_BUDGET_USD:
            raise AgentError(400, "E_BAD_REQUEST", f"budget_usd must be a number from 0.5 to {MAX_BUDGET_USD}")
        conv_id = uuid4().hex
        record = dict(agent_conv_id=conv_id, session_id=None, console_url=None, status="running", created=time.time(),
                      revision=0, budget_usd=float(budget_usd), cost_usd=None, error=None, items=[], seen=[], answered=[])
        with self._lock:
            self._records[conv_id] = record
            self._control[conv_id] = dict(stop=Event())
            for old in sorted(self._records, key=lambda k: self._records[k]["created"])[:-KEEP]:
                del self._records[old]
                self._control.pop(old, None)
        self._begin_turn(conv_id, text.strip(), jobs)
        return conv_id

    def message(self, conv_id, text, jobs):
        if not isinstance(text, str) or not text.strip():
            raise AgentError(400, "E_BAD_REQUEST", "text must be a non-empty string")
        with self._lock:
            record = self._records.get(conv_id)
            if record is None:
                raise AgentError(404, "E_NOT_FOUND", "unknown conversation (conversations of an earlier server process cannot be continued)")
            if record["status"] != "idle":
                raise AgentError(400, "E_BAD_REQUEST", f"the conversation is {record['status']}, not waiting for a message")
            record["status"] = "running"
            self._control[conv_id]["stop"].clear()
        self._begin_turn(conv_id, text.strip(), jobs)
        return {"ok": True}

    def interrupt(self, conv_id):
        with self._lock:
            record, control = self._records.get(conv_id), self._control.get(conv_id)
        if record is None:
            if self._load(conv_id) is None:
                raise AgentError(404, "E_NOT_FOUND", "unknown conversation")
            return {"ok": True}
        control["stop"].set()
        if record["session_id"] and record["status"] == "running":
            try:
                self.client().beta.sessions.events.send(session_id=record["session_id"], events=[{"type": "user.interrupt"}])
            except Exception:
                pass
        return {"ok": True}

    def get(self, conv_id):
        with self._lock:
            record = self._records.get(conv_id)
            if record is not None:
                return self._public(record)
        stored = self._load(conv_id)
        return None if stored is None else self._public(stored)

    def latest(self):
        with self._lock:
            records = [deepcopy(r) for r in self._records.values()]
        seen = {r["agent_conv_id"] for r in records}
        try:
            names = [p.stem[len("agent-"):] for p in (home() / "agent_history").glob("agent-*.json")]
        except OSError:
            names = []
        for name in names:
            if name not in seen:
                stored = self._load(name)
                if stored is not None:
                    records.append(stored)
        best = max(records, key=lambda r: r["created"], default=None)
        return None if best is None else self._public(best)

    @staticmethod
    def _public(record):
        return {k: deepcopy(v) for k, v in record.items() if k not in ("seen", "answered")}

    # ---- one turn -----------------------------------------------------------------------------------------------

    def _begin_turn(self, conv_id, text, jobs):
        self._append(conv_id, {"kind": "user", "text": text})
        Thread(target=self._turn, args=(conv_id, text, jobs), name="wsbench-agent", daemon=True).start()

    def _append(self, conv_id, item=None, **fields):
        with self._lock:
            record = self._records.get(conv_id)
            if record is None:
                return
            if item is not None:
                record["items"].append(item)
            record.update(fields)
            record["revision"] += 1
        self._save(conv_id)

    def _turn(self, conv_id, text, jobs):
        try:
            self._run_turn(conv_id, text, jobs)
        except Exception as exc:                   # the page must always get an answer
            self._append(conv_id, {"kind": "error", "text": f"{type(exc).__name__}: {exc}"[:600]}, status="error",
                         error=f"{type(exc).__name__}: {exc}"[:600])

    def _run_turn(self, conv_id, text, jobs):
        client = self.client()
        with self._lock:
            record = self._records[conv_id]
            session_id, budget = record["session_id"], record["budget_usd"]
            stop = self._control[conv_id]["stop"]
        if session_id is None:
            ids = self.ensure_agent()
            session = client.beta.sessions.create(
                agent={"type": "agent", "id": ids["agent_id"], "version": ids["version"]},
                environment_id=ids["environment_id"], title="WSBench: " + text[:60],
                budget={"type": "limit", "max_list_cost": {"amount": str(int(round(budget * 100))), "currency": "USD"}},
            )
            session_id = session.id
            self._append(conv_id, session_id=session_id, console_url=CONSOLE_URL.format(session_id))

        message = {"type": "user.message", "content": [{"type": "text", "text": text}]}
        sent, outcome, drops = False, None, 0
        while outcome is None:
            try:
                # Stream first, then send: the stream only delivers events that occur after it opens.
                with client.beta.sessions.events.stream(session_id=session_id) as stream:
                    if not sent:
                        client.beta.sessions.events.send(session_id=session_id, events=[message])
                        sent = True
                    else:                           # reconnect: the stream has no replay, so read the gap from history
                        last_status = None
                        for event in client.beta.sessions.events.list(session_id=session_id).data:
                            unseen = getattr(event, "id", None) not in self._records[conv_id]["seen"]
                            if unseen and getattr(event, "type", "").startswith("session.status_"):
                                last_status = event
                            self._handle(conv_id, session_id, event, jobs, stop, replay=True)
                        if last_status is not None:     # the turn may have ended while we were disconnected
                            outcome = self._status(conv_id, last_status)
                    if outcome is None:
                        for event in stream:
                            outcome = self._handle(conv_id, session_id, event, jobs, stop)
                            if outcome is not None:
                                break
                        else:
                            raise ConnectionError("the event stream ended before the turn did")
            except AgentError:
                raise
            except Exception as exc:
                drops += 1
                if not sent or drops > 5:
                    raise
                self._append(conv_id, {"kind": "note", "text": f"connection lost, reconnecting ({type(exc).__name__})"})
                time.sleep(min(2 * drops, 10))

        cost = None
        try:
            cents = _cents(getattr(client.beta.sessions.retrieve(session_id=session_id).usage, "list_cost", None))
            cost = None if cents is None else cents / 100
        except Exception:
            pass
        fields = {"status": outcome}
        if cost is not None:
            fields["cost_usd"] = cost
        self._append(conv_id, **fields)

    def _handle(self, conv_id, session_id, event, jobs, stop, replay=False):
        """Record one event. Returns the conversation's new status when the turn has ended, else None."""
        event_id, kind = getattr(event, "id", None), getattr(event, "type", "")
        with self._lock:
            record = self._records[conv_id]
            if event_id:
                if event_id in record["seen"]:
                    return None
                record["seen"].append(event_id)
        if kind == "agent.message":
            text = _text(getattr(event, "content", None))
            if text.strip():
                self._append(conv_id, {"kind": "agent", "text": text})
        elif kind == "agent.custom_tool_use":
            with self._lock:
                done = event_id in record["answered"]
                if not done:
                    record["answered"].append(event_id)
            if done:
                return None
            arguments = dict(getattr(event, "input", None) or {})
            item = {"kind": "tool", "name": event.name, "input": arguments, "result": None, "run_id": None}
            self._append(conv_id, item)

            def on_run(run_id):
                item["run_id"] = run_id
                self._append(conv_id)

            result = agent_tools.execute(event.name, arguments, jobs, stop.is_set, on_run)
            item["result"] = result
            self._append(conv_id)
            self.client().beta.sessions.events.send(session_id=session_id, events=[{
                "type": "user.custom_tool_result", "custom_tool_use_id": event_id,
                "content": [{"type": "text", "text": result}]}])
        elif kind == "session.error":
            detail = getattr(event, "error", None)
            self._append(conv_id, {"kind": "error", "text": str(getattr(detail, "message", None) or detail or "session error")[:600]})
        elif kind.startswith("session.status_") and not replay:
            return self._status(conv_id, event)
        return None

    def _status(self, conv_id, event):
        """Map a session.status_* event to the conversation's status at the end of a turn, or None to keep going."""
        kind = getattr(event, "type", "")
        if kind == "session.status_terminated":
            return "terminated"
        if kind != "session.status_idle":
            return None
        reason = getattr(getattr(event, "stop_reason", None), "type", None)
        if reason == "requires_action":
            return None                             # waiting for our tool result, which _handle sends
        if reason == "budget_reached":
            self._append(conv_id, {"kind": "note", "text": "The budget of this conversation is spent."})
            return "budget"
        if reason == "retries_exhausted":
            self._append(conv_id, {"kind": "error", "text": "The session stopped after repeated errors."})
            return "error"
        return "idle"

    # ---- storage ------------------------------------------------------------------------------------------------

    def _save(self, conv_id):
        with self._lock:
            record = deepcopy(self._records.get(conv_id))
        if record is None:
            return
        try:
            folder = home() / "agent_history"
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f".agent-{conv_id}.tmp"
            tmp.write_text(json.dumps(record), "utf-8")
            os.replace(tmp, folder / f"agent-{conv_id}.json")
            files = sorted(folder.glob("agent-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[KEEP:]:
                old.unlink(missing_ok=True)
        except (OSError, TypeError, ValueError):
            pass

    @staticmethod
    def _load(conv_id):
        if not CONV_ID.fullmatch(conv_id or ""):
            return None
        try:
            record = json.loads((home() / "agent_history" / f"agent-{conv_id}.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if record.get("status") == "running":
            record.update(status="error", error=INTERRUPTED)
        elif record.get("status") == "idle":
            record["status"] = "closed"             # the session still exists, but this process cannot continue it
        return record


sessions = AgentSessions()
