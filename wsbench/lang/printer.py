"""Canonical WSL source rendering."""
from .ast import Assignment, Call, Range, String, Target


def render(value):
    if isinstance(value, Assignment):
        return f"{render(value.target)} := {render(value.expr)}"
    if isinstance(value, Target):
        part = f".H[{value.head}]" if value.part == "head" else f".{value.part}" if value.part else ""
        return f"L[{value.spelling}]{part}"
    if isinstance(value, Call):
        if value.name == "align":
            return f"align({render(value.args[0])} -> {render(value.args[1])})"
        args = [(a.key + "=" if a.key else "") + render(a.value) for a in value.arguments]
        return f"{value.name}({', '.join(args)})"
    if isinstance(value, list):
        return "[" + ", ".join(map(render, value)) + "]"
    if isinstance(value, Range):
        return "[" + ":".join(map(render, value)) + "]"
    if isinstance(value, String):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def print_statement(stmt, with_label=True):
    label = f"{stmt.label}: " if with_label and stmt.label is not None else ""
    claim = f"approx({render(stmt.tol)})" if stmt.claim == "approx" else stmt.claim
    return label + claim + " " + "; ".join(map(render, stmt.ops))
