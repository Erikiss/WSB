"""Translate into WSL, check proposed statements, and submit a measured script."""

import json
import re

from wsbench.lang.card import LANGUAGE_CARD
from wsbench.lang.parser import parse
from wsbench.lang.printer import print_statement
from wsbench.lang.validate import validate
from wsbench.models.registry import SHAPES

from . import openrouter
from .prompts import ALT_SYSTEM, CRITIC_SYSTEM, TRANSLATE_SYSTEM

_TEMPLATES = {"translate": TRANSLATE_SYSTEM, "critic": CRITIC_SYSTEM, "alternatives": ALT_SYSTEM}
_MODEL_LINE = re.compile(r"^\s*model(?:\s|$)")
_PREDICTIONS = {"exact", "near", "edit", "pass", "fail", "unknown"}


class AssistRequestError(ValueError):
    code = "E_BAD_REQUEST"


def _block(text, language):
    fence = "`" * 3
    match = re.search(rf"^[ \t]*{fence}{language}[ \t]*\r?\n(.*?)^[ \t]*{fence}[ \t]*\r?$",
                      text, flags=re.MULTILINE | re.DOTALL)
    if match is not None:
        return match.group(1)


def _check(model, block):
    # Preserve the original block's line numbers while enforcing the selected model.
    lines = ["" if _MODEL_LINE.match(line) else line for line in block.splitlines()]
    statements, parse_errors = parse("\n".join([f"model {model}", *lines]))
    errors = sorted([*parse_errors, *validate(statements)], key=lambda e: (e["line"], e["col"]))
    rejected = {error["line"] for error in errors}
    return [stmt for stmt in statements if stmt.line not in rejected], errors


def _syntax(message):
    return dict(line=None, col=None, code="E_SYNTAX", message=message)


def _critic_entries(text, count):
    block = _block(text, "json")
    if block is None:
        raise ValueError("missing json block")
    entries = json.loads(block)
    if not isinstance(entries, list):
        raise ValueError("expected a JSON list")
    accepted = {}
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("n")) is not int:
            raise ValueError("expected a numbered critic entry")
        if not 1 <= entry["n"] <= count:
            continue
        if not isinstance(entry.get("predict"), str):
            raise ValueError("expected predict in each critic entry")
        if entry["predict"] not in _PREDICTIONS:
            continue
        if not all(isinstance(entry.get(key), str) for key in ("reason", "fix")):
            raise ValueError("expected reason and fix in each critic entry")
        accepted.setdefault(entry["n"], entry)
    return sorted(accepted.items())


def _parameters(request):
    text, model = request.get("text"), request.get("model")
    if not isinstance(text, str) or not isinstance(model, str):
        raise AssistRequestError("text and model must be strings")
    if SHAPES.get(model, {}).get("family") not in {"neox", "txl", "grid"}:
        raise AssistRequestError("model must be a NeoX or Transformer-XL model or grid:task001 through grid:task400")
    roles = request.get("roles", {})
    if not isinstance(roles, dict) or any(
        k not in openrouter.DEFAULT_ROLES or not isinstance(v, str) or not v.strip() for k, v in roles.items()
    ):
        raise AssistRequestError("roles must map assist roles to model IDs")
    execute = request.get("execute", True)
    if not isinstance(execute, bool):
        raise AssistRequestError("execute must be a boolean")
    return text, model, {**openrouter.DEFAULT_ROLES, **roles}, execute


def assist(request: dict, submit) -> dict:
    text, model, roles, execute = _parameters(request)
    result = {
        "script": f"model {model}", "run_id": None, "cost_usd": 0.0,
        "roles": dict.fromkeys(roles), "predictions": {}, "parse_errors": [],
    }

    def call(role, messages):
        system = _TEMPLATES[role].replace("{LANGUAGE_CARD}", LANGUAGE_CARD).replace("{MODEL}", model)
        reply = openrouter.chat(role, roles[role], system, messages)
        result["roles"][role] = reply
        result["cost_usd"] += reply["cost_usd"]
        return reply

    def record(role, errors):
        result["parse_errors"].extend({"role": role, **error} for error in errors)

    source = f"Source text:\n<<<\n{text}\n>>>"
    messages = [{"role": "user", "content": source}]
    for attempt in range(2):
        reply = call("translate", messages)
        block = _block(reply["text"], "wsl") if reply["ok"] else None
        if block is None:
            record("translate", [_syntax(reply["error"] or "missing wsl block")])
            return result
        statements, errors = _check(model, block)
        if errors and attempt == 0:
            diagnostics = "\n".join("line {line} col {col} {code} {message}".format(**e) for e in errors)
            messages = [
                *messages, {"role": "assistant", "content": reply["text"]},
                {"role": "user", "content": f"Your WSL did not parse. Errors:\n{diagnostics}\n"
                                            "Return the corrected wsl block only."},
            ]
            continue
        record("translate", errors)
        break
    if not statements:
        record("translate", [_syntax("no testable statement: the translator found no claim that the available operations can check")])
        return result

    proposed = [print_statement(stmt, with_label=False) for stmt in statements[:8]]
    numbered = "\n".join(f"{n}. {statement}" for n, statement in enumerate(proposed, 1))
    lines = [f"t{n}: {statement}" for n, statement in enumerate(proposed, 1)]
    critic = call("critic", [{"role": "user", "content": f"Model: {model}\nStatements:\n{numbered}"}])
    if critic["ok"]:
        try:
            entries = _critic_entries(critic["text"], len(proposed))
        except (ValueError, RecursionError) as exc:
            record("critic", [_syntax(str(exc))])
            entries = []
        for n, entry in entries:
            result["predictions"][f"t{n}"] = entry["predict"]
            if entry["fix"].strip():
                fixes, errors = _check(model, entry["fix"])
                if errors or len(fixes) != 1:
                    record("critic", errors or [_syntax("fix must give exactly one statement")])
                else:
                    lines.append(f"fix{n}: " + print_statement(fixes[0], with_label=False))

    alternatives = call("alternatives", [{
        "role": "user", "content": f"{source}\nModel: {model}\nStatements already proposed:\n{numbered}",
    }])
    if alternatives["ok"]:
        block = _block(alternatives["text"], "wsl")
        if block is None:
            record("alternatives", [_syntax("missing wsl block")])
        else:
            extra, errors = _check(model, block)
            record("alternatives", errors)
            for n, stmt in enumerate(extra[:3], 1):
                lines.append(f"alt{n}: " + print_statement(stmt, with_label=False))
    result["script"] = "\n".join([result["script"], *lines])
    if execute:
        result["run_id"] = submit(result["script"], result["predictions"])
    return result
