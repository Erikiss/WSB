"""Verdict construction shared by synchronous scripts and queued jobs."""
from collections import defaultdict
import re

from wsbench.lang.hints import hints
from wsbench.lang.parser import DIRECTIVE, parse, valid_model
from wsbench.lang.printer import print_statement
from wsbench.lang.validate import validate


# Quoted values can contain a literal hash. Only a hash outside a string starts a comment.
_QUOTED_OR_COMMENT = re.compile(r'("(?:[^"\\]|\\.)*")|#.*$')


def failed(verdict, code, message):
    verdict.update(status="ERROR", measured=None, error={"code": code, "message": message})


def agreement(verdict, predictions):
    if predictions is None:
        return
    predicted = predictions.get(verdict["label"])
    outcome = verdict["measured"]
    if verdict["claim"] == "solve":
        outcome = "pass" if verdict["status"] == "CONFIRMED" else "fail"
    result = "n/a"
    if predicted not in (None, "unknown") and outcome is not None and verdict["status"] != "ERROR":
        result = "agree" if predicted == outcome else "disagree"
    verdict.update(predicted=predicted, agree=result)


def prepare(script, predictions=None):
    statements, syntax = parse(script)
    by_line = {st.line: st for st in statements}
    errors = {e["line"]: e for e in syntax + validate(statements)}
    line_hints = defaultdict(list)
    for entry in hints(statements):
        line_hints[entry["line"]].append(entry)
    verdicts = []
    model = None
    for line, raw in enumerate(script.splitlines(), 1):
        source = raw.partition("#")[0].strip()
        directive = DIRECTIVE.fullmatch(raw)
        if directive and directive[1] == "model":
            name = directive[2].partition("#")[0].strip()
            model = name if valid_model(name) else None
        st = by_line.get(line)
        if st is None and line not in errors:
            continue
        if st:
            label = st.display_label
            source = print_statement(st, with_label=False)
        else:
            if directive is None:
                source = _QUOTED_OR_COMMENT.sub(lambda m: m[1] or "", raw).strip()
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*:(?!=)", source)
            label = match[1] if match else f"line{line}"
        verdict = dict(line=line, label=label, source=source, model=st.model if st else model,
                       claim=st.claim if st else None, tol=st.tol if st else None,
                       status="PENDING", measured=None, metrics={}, hints=line_hints[line],
                       warnings=[], error=None, elapsed_ms=0, predicted=None, agree=None)
        if line in errors:
            error = errors[line]
            failed(verdict, error["code"], error["message"])
            agreement(verdict, predictions)
        verdicts.append(verdict)
    return by_line, verdicts
