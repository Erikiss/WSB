"""Syntax tree of WSL. Equality ignores source locations and model context."""
from dataclasses import dataclass, field
import re
from typing import Any


def location(default=1):
    return field(default=default, compare=False, repr=False)


def reserved(name):
    return bool(re.fullmatch(r"[LH](?:[0-9]+)?|line[0-9]+", name))


class String(str):
    """A quoted string, distinguished from a bare WSL identifier."""


class Range(tuple):
    def __new__(cls, a, b, s, col=1):
        result = super().__new__(cls, (a, b, s))
        result.col = col
        return result


class ListValue(list):
    def __init__(self, items, cols, col):
        super().__init__(items)
        self.cols = cols
        self.col = col


@dataclass
class Target:
    spelling: str
    part: str | None = None
    head: int | None = None
    index: int | None = location(None)
    col: int = location()
    part_col: int = location()


@dataclass
class Argument:
    key: str | None
    value: Any
    col: int = location()
    key_col: int = location()


@dataclass
class Call:
    name: str
    arguments: list[Argument]
    col: int = location()

    @property
    def args(self):
        return [arg.value for arg in self.arguments if arg.key is None]

    @property
    def kwargs(self):
        return {arg.key: arg.value for arg in self.arguments if arg.key is not None}


@dataclass
class Assignment:
    target: Target
    expr: Target | Call
    col: int = location()


@dataclass
class Statement:
    line: int = field(compare=False)
    label: str | None
    claim: str
    tol: int | float | None
    ops: list[Assignment | Call]
    model: str | None = location(None)
    seed: int = location(0)
    label_col: int = location()
    claim_col: int = location()

    @property
    def family(self):
        if self.model is None:
            return None
        if self.model.startswith("grid:"):
            return "grid"
        return "modadd" if self.model == "modadd" else "neox"

    @property
    def display_label(self):
        return self.label or f"line{self.line}"
