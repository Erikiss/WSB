"""Local execution of the agent's custom tools. Each returns a short JSON string."""
import json
import time

from wsbench.assist.explore import digest
from wsbench.lang.hints import hints
from wsbench.lang.parser import parse
from wsbench.lang.validate import validate

from .text import clean_deep

MAX_RESULT_CHARS = 20_000
RUN_TIMEOUT_S = 20 * 60


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _size(value):
    return len(_dump(value))


def _json(value):
    """The tool result as compact JSON text. A result over the limit is shortened but stays valid JSON: the longest
    field loses its tail and the result says what was cut ("truncated", "returned_chars" or "omitted")."""
    value = clean_deep(value)
    text = _dump(value)
    if len(text) <= MAX_RESULT_CHARS:
        return text
    value = dict(value) if isinstance(value, dict) else {"result": value}
    marks = ("truncated", "truncated_field", "returned_chars", "omitted", "omitted_from")
    for _ in range(12):
        excess = len(text) - MAX_RESULT_CHARS
        if excess <= 0:
            return text
        key = max((k for k in value if k not in marks), key=lambda k: _size(value[k]), default=None)
        if key is None:
            break
        field = value[key]
        if isinstance(field, str):
            keep = max(0, len(field) - excess - 160)         # room for the markers added below
            value[key] = field[:keep]
            value.update(truncated=True, truncated_field=key, returned_chars=keep)
        elif isinstance(field, list) and field:
            keep = max(0, min(len(field) - 1, int(len(field) * (1 - (excess + 160) / max(_size(field), 1)))))
            value[key] = field[:keep]
            value.update(truncated=True, omitted=value.get("omitted", 0) + len(field) - keep, omitted_from=key)
        else:
            value[key] = "[too large to show]"
            value["truncated"] = True
        text = _dump(value)
    return _dump({"error": "the result is too large to return", "chars": len(text)})


def wsl_check(script):
    statements, errors = parse(script)
    errors = sorted(errors + validate(statements), key=lambda e: (e["line"], e["col"]))
    return _json({"statements": len(statements), "errors": errors, "hints": hints(statements)})


def wsl_run(script, jobs, should_stop=lambda: False, on_run=lambda run_id: None):
    run_id = jobs.submit(script, source="agent")
    on_run(run_id)
    deadline = time.monotonic() + RUN_TIMEOUT_S
    while True:
        run = jobs.snapshot(run_id)
        if run["status"] in ("done", "cancelled"):
            break
        if should_stop() or time.monotonic() > deadline:
            jobs.cancel(run_id)
        time.sleep(0.3)
    return _json({"run_id": run_id, "status": run["status"], "verdicts": digest(run["verdicts"])})


def arc_tasks(single_primitive_only=False, contains=""):
    from wsbench.arc.data import list_tasks
    rows = [r for r in list_tasks() if (not single_primitive_only or r.get("single_primitive"))
            and (not contains or contains in r["task"] or contains in (r.get("original_id") or ""))]
    return _json({"count": len(rows), "tasks": rows[:60]})


def arc_task(task):
    from wsbench.arc.data import task_detail
    detail = task_detail(task)
    return _json({k: detail.get(k) for k in ("task", "original_id", "train", "test", "n_arcgen", "solution_raw", "bytes_raw")})


def execute(name, arguments, jobs, should_stop=lambda: False, on_run=lambda run_id: None):
    """Run one tool call. Errors are returned as text so the agent can react to them."""
    try:
        if name == "wsl_check":
            return wsl_check(str(arguments.get("script", "")))
        if name == "wsl_run":
            return wsl_run(str(arguments.get("script", "")), jobs, should_stop, on_run)
        if name == "arc_tasks":
            return arc_tasks(bool(arguments.get("single_primitive_only", False)), str(arguments.get("contains", "")))
        if name == "arxiv_paper":
            from . import arxiv
            part = arguments.get("part", 0)
            return _json(arxiv.read(str(arguments.get("paper", "")), part if type(part) is int else 0))
        if name == "lesswrong_post":
            from . import lesswrong
            part = arguments.get("part", 0)
            return _json(lesswrong.read(str(arguments.get("post", "")), part if type(part) is int else 0))
        if name == "github_repo":
            from . import github
            part = arguments.get("part", 0)
            return _json(github.read(str(arguments.get("repo", "")), str(arguments.get("path", "")),
                                     part if type(part) is int else 0, str(arguments.get("search", ""))))
        if name == "arc_task":
            return arc_task(str(arguments.get("task", "")))
        return _json({"error": f"unknown tool {name}"})
    except Exception as exc:
        return _json({"error": f"{type(exc).__name__}: {exc}"})
