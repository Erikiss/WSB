"""Conversations with the Claude Managed Agent. Anthropic hosts the agent loop; WSBench executes the tool calls.

One conversation = one Managed Agents session. Each turn runs in a background thread: it opens the event stream,
sends the researcher's message, answers the agent's custom tool calls from the local executor and records a
transcript, which is stored in WSB_HOME/agent_history/ and served to the page.

A conversation survives the server: the session lives on the Claude Platform, so after a restart or a failed turn
the next message first re-attaches (reads the session's history, settles tool calls nobody answered) and then
continues. Only a terminated session or a spent budget ends a conversation.
"""
import hashlib
import json
import os
import re
import time
import unicodedata
from copy import deepcopy
from threading import Event, Lock, Thread
from uuid import uuid4

from wsbench.paths import home

from . import spec as agent_spec
from . import tools as agent_tools

KEEP = 20
MAX_BUDGET_USD = 20.0
CONV_ID = re.compile(r"[0-9a-f]{32}")
INTERRUPTED = "The server restarted during this turn. You can continue the conversation with a new message."
HIDDEN = ("seen", "delivered", "resume")        # bookkeeping that the page does not need
SEND_ATTEMPTS = 6
MAX_DROPS = 5
_sleep = time.sleep


def console_url(session_id):
    """Link to the session in the Claude Console. The workspace must be the one the API key belongs to."""
    workspace = os.getenv("WSB_ANTHROPIC_WORKSPACE", "default").strip() or "default"
    return f"https://platform.claude.com/workspaces/{workspace}/sessions/{session_id}"


class AgentError(Exception):
    def __init__(self, status_code, code, message):
        super().__init__(message)
        self.status_code, self.code = status_code, code


def available():
    return os.getenv("WSB_LLM") == "mock" or bool(os.getenv("ANTHROPIC_API_KEY"))


def session_title(text, limit=60):
    """A one-line title for the Claude Console. The platform rejects control and format characters in titles
    (line breaks, tabs, soft hyphens, zero-width characters), which pasted text often contains."""
    clean = "".join("" if unicodedata.category(ch) == "Cf" else " " if unicodedata.category(ch)[0] in "CZ" else ch
                    for ch in text)     # format characters vanish; controls and separators become a space
    return "WSBench: " + " ".join(clean.split())[:limit].rstrip()


def _text(blocks):
    parts = []
    for block in blocks or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(getattr(block, "text", ""))
        elif kind == "redacted":
            parts.append("[withheld by the platform]")
    return "".join(parts)


def _cents(list_cost):
    try:
        return int(getattr(list_cost, "amount", None) or list_cost["amount"])
    except (TypeError, KeyError, ValueError):
        return None


def _anthropic():
    try:
        import anthropic
        return anthropic
    except ImportError:
        return None


def _transient(exc):
    """Connection problems and overload: worth another try. Everything else (4xx) is a real refusal."""
    anthropic = _anthropic()
    if anthropic is None:
        return False
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(exc, anthropic.APIStatusError) and isinstance(status, int) and (status in (408, 409, 429) or status >= 500)


def _is_bad_request(exc):
    anthropic = _anthropic()
    return anthropic is not None and isinstance(exc, anthropic.BadRequestError)


def _gone(exc):
    """The stored agent, environment or session no longer exists (deleted, archived, or another workspace)."""
    anthropic = _anthropic()
    if anthropic is None:
        return False
    if isinstance(exc, anthropic.NotFoundError):
        return True
    return isinstance(exc, anthropic.BadRequestError) and "archived" in str(exc).lower()


class AgentSessions:
    def __init__(self, client_factory=None):
        self._records, self._control, self._results = {}, {}, {}
        self._lock = Lock()
        self._setup_lock = Lock()
        self._client_factory = client_factory
        self._client = None
        self._mock_ids = {}

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

    @staticmethod
    def _fingerprint():
        key = os.getenv("ANTHROPIC_API_KEY", "")
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def _read_ids(self):
        if os.getenv("WSB_LLM") == "mock":            # a demo must never write IDs that live mode would trust
            return dict(self._mock_ids)
        try:
            ids = json.loads(self._ids_path().read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        return ids if ids.get("key_fp") == self._fingerprint() else {}      # another key = another workspace

    def _write_ids(self, ids):
        if os.getenv("WSB_LLM") == "mock":
            self._mock_ids = dict(ids)
            return
        try:
            path = self._ids_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**ids, "key_fp": self._fingerprint()}), "utf-8")
        except OSError:
            pass

    @staticmethod
    def _find(listing, name):
        """An existing, unarchived resource of this name (after a wiped WSB_HOME), so nothing is created twice."""
        try:
            for item in listing(limit=100):
                if getattr(item, "name", None) == name and not getattr(item, "archived_at", None):
                    return item
        except Exception:
            pass
        return None

    def ensure_agent(self, force=False):
        """Create the environment and the agent once and reuse their IDs; update the agent when its spec changed.
        IDs are written after every step; force=True forgets the stored IDs (the resources are gone)."""
        with self._setup_lock:
            ids = {} if force else self._read_ids()
            client, spec, digest = self.client(), agent_spec.spec(), agent_spec.spec_hash()
            if not ids.get("environment_id"):
                environment = self._find(lambda **kw: client.beta.environments.list(**kw), "wsbench") or \
                    client.beta.environments.create(name="wsbench", config={"type": "cloud", "networking": {"type": "limited"}})
                ids["environment_id"] = environment.id
                self._write_ids(ids)
            if ids.get("agent_id") and ids.get("spec_hash") != digest:
                try:
                    agent = client.beta.agents.update(ids["agent_id"], **spec)
                    ids.update(version=agent.version, spec_hash=digest)
                except Exception as exc:
                    if not _gone(exc):
                        raise
                    ids.pop("agent_id", None)             # the agent was deleted or archived: make a new one
            if not ids.get("agent_id"):
                found = self._find(lambda **kw: client.beta.agents.list(**kw), spec["name"])
                if found is not None:
                    agent = client.beta.agents.update(found.id, **spec)
                else:
                    agent = client.beta.agents.create(**spec)
                ids.update(agent_id=agent.id, version=agent.version, spec_hash=digest)
            self._write_ids(ids)
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
                      revision=0, budget_usd=float(budget_usd), cost_usd=None, error=None, items=[], seen=[], delivered=[])
        self._register(conv_id, record)
        self._begin_turn(conv_id, text.strip(), jobs)
        return conv_id

    def _register(self, conv_id, record):
        with self._lock:
            self._records[conv_id] = record
            self._control[conv_id] = dict(stop=Event())
            self._results[conv_id] = {}
            for old in sorted(self._records, key=lambda k: self._records[k]["created"])[:-KEEP]:
                del self._records[old]
                self._control.pop(old, None)
                self._results.pop(old, None)

    def message(self, conv_id, text, jobs):
        if not isinstance(text, str) or not text.strip():
            raise AgentError(400, "E_BAD_REQUEST", "text must be a non-empty string")
        with self._lock:
            known = conv_id in self._records
        if not known:                               # a conversation of an earlier server process: re-attach to its session
            stored = self._load(conv_id)
            if stored is None or not stored.get("session_id"):
                raise AgentError(404, "E_NOT_FOUND", "unknown conversation")
            self._register(conv_id, stored)
        with self._lock:
            record = self._records[conv_id]
            if record["status"] == "error" and record.get("session_id"):
                record["resume"] = True             # the turn failed: re-sync with the session before sending
            elif record["status"] != "idle":
                raise AgentError(400, "E_BAD_REQUEST", f"the conversation is {record['status']}, not waiting for a message")
            elif not known:
                record["resume"] = True
            record.update(status="running", error=None)
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
        return {k: deepcopy(v) for k, v in record.items() if k not in HIDDEN}

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
            message = f"{type(exc).__name__}: {exc}"[:600]
            self._append(conv_id, {"kind": "error", "text": message}, status="error", error=message)

    def _send(self, session_id, events):
        """Send events; connection problems and overload are retried with a growing pause."""
        for attempt in range(SEND_ATTEMPTS):
            try:
                return self.client().beta.sessions.events.send(session_id=session_id, events=events)
            except Exception as exc:
                if not _transient(exc) or attempt == SEND_ATTEMPTS - 1:
                    raise
                _sleep(min(2 ** (attempt + 1), 30))

    def _create_session(self, text, budget):
        def create(ids):
            return self.client().beta.sessions.create(
                agent={"type": "agent", "id": ids["agent_id"], "version": ids["version"]},
                environment_id=ids["environment_id"], title=session_title(text),
                budget={"type": "limit", "max_list_cost": {"amount": str(int(round(budget * 100))), "currency": "USD"}},
            )
        try:
            return create(self.ensure_agent())
        except Exception as exc:
            if not _gone(exc):
                raise
            return create(self.ensure_agent(force=True))       # the stored agent or environment is gone

    def _history(self, session_id):
        """The whole event history, oldest first. The list is paged, and iterating it follows the pages."""
        return list(self.client().beta.sessions.events.list(session_id=session_id, limit=1000))

    def _mark_delivered(self, conv_id, event_id):
        with self._lock:
            record = self._records.get(conv_id)
            if record is not None and event_id not in record["delivered"]:
                record["delivered"].append(event_id)
            self._results.get(conv_id, {}).pop(event_id, None)

    def _resync(self, conv_id, session_id):
        """Re-attach to a session after a failed turn or a server restart: show what the agent said meanwhile and
        settle tool calls that nobody answered, because the platform refuses a new message while a call is pending."""
        history = self._history(session_id)
        answered = {getattr(e, "custom_tool_use_id", None) for e in history if getattr(e, "type", "") == "user.custom_tool_result"}
        with self._lock:
            known = set(self._records[conv_id]["seen"])
            delivered = set(self._records[conv_id]["delivered"])
        for event in history:
            event_id, kind = getattr(event, "id", None), getattr(event, "type", "")
            if kind == "agent.custom_tool_use" and event_id not in answered and event_id not in delivered:
                with self._lock:
                    kept = self._results.get(conv_id, {}).get(event_id)        # executed here, but the send had failed
                self._send(session_id, [{"type": "user.custom_tool_result", "custom_tool_use_id": event_id, "content": [
                    {"type": "text", "text": kept or "WSBench lost the connection during this call, so it did not run. "
                                                    "Run it again if it is still needed."}]}])
                self._append(conv_id, {"kind": "note", "text": f"A tool call ({getattr(event, 'name', '?')}) was left unanswered; "
                                                               + ("its result was sent now." if kept else "it was settled without running.")})
                self._mark_delivered(conv_id, event_id)
            elif kind == "agent.message" and event_id and event_id not in known:
                text = _text(getattr(event, "content", None))
                if text.strip():
                    self._append(conv_id, {"kind": "agent", "text": text})
        with self._lock:
            record = self._records[conv_id]
            record["seen"].extend(e.id for e in history if getattr(e, "id", None) and e.id not in known)
            record["delivered"].extend(i for i in answered if i and i not in record["delivered"])

    def _run_turn(self, conv_id, text, jobs):
        client = self.client()
        with self._lock:
            record = self._records[conv_id]
            session_id, budget = record["session_id"], record["budget_usd"]
            resume = record.pop("resume", False)
            stop = self._control[conv_id]["stop"]
        if session_id is None:
            session_id = self._create_session(text, budget).id
            self._append(conv_id, session_id=session_id, console_url=console_url(session_id))
        elif resume:
            self._resync(conv_id, session_id)
        if stop.is_set():
            self._append(conv_id, {"kind": "note", "text": "Stopped before the message was sent."}, status="idle")
            return

        message = {"type": "user.message", "content": [{"type": "text", "text": text}]}
        sent, outcome, drops = False, None, 0
        while outcome is None:
            try:
                # Stream first, then send: the stream only delivers events that occur after it opens.
                with client.beta.sessions.events.stream(session_id=session_id) as stream:
                    if not sent:
                        self._send(session_id, [message])
                        sent = True
                    else:                           # reconnect: the stream has no replay, so read the gap from history
                        history = self._history(session_id)
                        for event in history:
                            if getattr(event, "type", "") == "user.custom_tool_result":
                                self._mark_delivered(conv_id, getattr(event, "custom_tool_use_id", None))
                        last_status = None
                        for event in history:
                            with self._lock:
                                fresh = getattr(event, "id", None) not in self._records[conv_id]["seen"]
                            if fresh and getattr(event, "type", "").startswith("session.status_"):
                                last_status = event
                            self._handle(conv_id, session_id, event, jobs, stop, replay=True)
                        if last_status is not None:     # the turn may have ended while we were disconnected
                            outcome = self._status(conv_id, last_status)
                    if outcome is None:
                        for event in stream:
                            drops = 0                   # the limit counts consecutive failures
                            outcome = self._handle(conv_id, session_id, event, jobs, stop)
                            if outcome is not None:
                                break
                        else:
                            raise ConnectionError("the event stream ended before the turn did")
            except AgentError:
                raise
            except Exception as exc:
                drops += 1
                if not sent or drops > MAX_DROPS:
                    raise
                self._append(conv_id, {"kind": "note", "text": f"connection lost, reconnecting ({type(exc).__name__})"})
                _sleep(min(2 * drops, 10))

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

    def _deliver(self, conv_id, session_id, event_id, result, stop):
        """Send a tool result. A result that the platform no longer wants (the researcher pressed Stop and the
        interrupt already closed the call) is dropped; any other failure is raised, and the call is retried after the
        reconnect because it stays undelivered."""
        event = {"type": "user.custom_tool_result", "custom_tool_use_id": event_id,
                 "content": [{"type": "text", "text": result}]}
        try:
            self._send(session_id, [event])
        except Exception as exc:
            if not (stop.is_set() and _is_bad_request(exc)):
                raise
        self._mark_delivered(conv_id, event_id)
        if stop.is_set():                           # the result may have restarted the agent: stop it again
            try:
                self.client().beta.sessions.events.send(session_id=session_id, events=[{"type": "user.interrupt"}])
            except Exception:
                pass

    def _handle(self, conv_id, session_id, event, jobs, stop, replay=False):
        """Record one event. Returns the conversation's new status when the turn has ended, else None."""
        event_id, kind = getattr(event, "id", None), getattr(event, "type", "")
        with self._lock:
            record = self._records[conv_id]
            if event_id and kind != "agent.custom_tool_use":
                if event_id in record["seen"]:
                    return None
                record["seen"].append(event_id)
        if kind == "agent.message":
            text = _text(getattr(event, "content", None))
            if text.strip():
                self._append(conv_id, {"kind": "agent", "text": text})
        elif kind == "agent.custom_tool_use":
            with self._lock:
                if event_id in record["delivered"]:
                    return None
                result = self._results[conv_id].get(event_id)
            if result is None:                      # not executed yet; a result that failed to send is sent again as it is
                arguments = dict(getattr(event, "input", None) or {})
                item = {"kind": "tool", "name": event.name, "input": arguments, "result": None, "run_id": None}
                self._append(conv_id, item)

                def on_run(run_id):
                    item["run_id"] = run_id
                    self._append(conv_id)

                result = agent_tools.execute(event.name, arguments, jobs, stop.is_set, on_run)
                item["result"] = result
                with self._lock:
                    self._results[conv_id][event_id] = result
                self._append(conv_id)
            self._deliver(conv_id, session_id, event_id, result, stop)
        elif kind == "user.custom_tool_result":
            self._mark_delivered(conv_id, getattr(event, "custom_tool_use_id", None))
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
        if reason == "retries_exhausted":           # the turn is dead, but the session accepts a new message
            self._append(conv_id, {"kind": "error", "text": "The platform gave up on this turn after repeated errors "
                                   "(overload, a rate limit or a tool limit). Send your message again."})
            return "idle"
        if reason == "refusal":                     # a safety classifier declined the answer
            details = getattr(event, "stop_details", None)
            category, explanation = getattr(details, "category", None), getattr(details, "explanation", None)
            self._append(conv_id, {"kind": "error", "text": "The model declined to answer this turn"
                                   + (f" ({category})" if category else "") + (f": {explanation}" if explanation else "")
                                   + ". Rephrase and send again."})
            return "idle"
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
        record.setdefault("seen", [])
        record.setdefault("delivered", [])
        if record.get("status") == "running":       # written by a process that no longer exists
            record.update(status="error", error=INTERRUPTED)
        return record


sessions = AgentSessions()
