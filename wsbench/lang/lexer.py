"""Line-local tokenisation with 1-based source positions."""
from dataclasses import dataclass
import re


class Problem(ValueError):
    def __init__(self, code, col, message):
        super().__init__(message)
        self.code = code
        self.col = col

    def diagnostic(self, line):
        return {"line": line, "col": self.col, "code": self.code, "message": str(self)}


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    col: int


TOKEN = re.compile(
    r'(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)'
    r'|(?P<FLOAT>(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)'
    r'|(?P<INT>[0-9]+)'
    r'|(?P<STRING>"(?:[^"\\]|\\["\\])*")'
    r'|(?P<OP>:=|->|[()\[\],;:=.+-])'
    r'|(?P<SPACE>\s+)'
)


def lex(line):
    tokens = []
    offset = 0
    while offset < len(line) and line[offset] != "#":
        match = TOKEN.match(line, offset)
        if not match:
            raise Problem("E_SYNTAX", offset + 1, "expected a WSL token or a double-quoted string")
        kind = match.lastgroup
        value = match.group()
        if kind != "SPACE":
            tokens.append(Token(value if kind == "OP" else kind, value, offset + 1))
        offset = match.end()
    tokens.append(Token("EOF", "", len(line[:offset].rstrip()) + 1))
    return tokens
