"""Recursive-descent WSL parser. Each source line is a recovery boundary."""
import math
import re

from wsbench.models.registry import SHAPES
from .ast import Argument, Assignment, Call, ListValue, Range, Statement, String, Target
from .lexer import Problem, lex


DIRECTIVE = re.compile(r"^\s*(model|seed)(?:\s+|$)(.*)$")
LNUMBER = re.compile(r"L([0-9]+)$")
HNUMBER = re.compile(r"H([0-9]+)$")
CLAIMS = {"gauge", "reparam", "edit", "approx", "train", "path", "solve"}


def valid_model(name):
    if name in ("neox-tiny", "pythia-14m", "modadd"):
        return True
    return bool(re.fullmatch(r"grid:task[0-9]{3}", name)) and 1 <= int(name[-3:]) <= 400


def expand_range(a, b, s):
    a, b, s = float(a), float(b), float(s)
    if not all(map(math.isfinite, (a, b, s))) or s == 0 or (b - a) * s < 0:
        raise ValueError("range step must reach the endpoint")
    quotient = abs((b - a) / s)
    nearest = round(quotient)
    endpoint = a + nearest * s
    count = nearest if abs(endpoint - b) <= 1e-9 * max(1, abs(s)) else math.floor(quotient)
    return [a + n * s for n in range(count + 1)]


class LineParser:
    def __init__(self, line, model):
        self.tokens = lex(line)
        self.pos = 0
        self.shape = SHAPES.get(model, {})

    @property
    def token(self):
        return self.tokens[self.pos]

    def peek(self, kind):
        return self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].kind == kind

    def take(self, kind):
        token = self.token
        if token.kind != kind:
            raise Problem("E_SYNTAX", token.col, f"expected {kind}, found {token.text or 'end of line'}")
        self.pos += 1
        return token

    def eat(self, kind):
        if self.token.kind == kind:
            self.take(kind)
            return True
        return False

    def number(self):
        sign = -1 if self.eat("-") else 1
        token = self.token
        if token.kind == "INT":
            return sign * int(self.take("INT").text)
        if token.kind == "FLOAT":
            return sign * float(self.take("FLOAT").text)
        raise Problem("E_SYNTAX", token.col, "expected an integer or floating-point number")

    def layer_start(self):
        return self.token.kind == "IDENT" and (
            LNUMBER.fullmatch(self.token.text) or (self.token.text == "L" and self.peek("["))
        )

    def layer(self, with_part=False):
        token = self.take("IDENT")
        literal = LNUMBER.fullmatch(token.text)
        if literal:
            index = int(literal[1])
            spelling = str(index)
        else:
            self.take("[")
            if self.token.text == "mid":
                self.take("IDENT")
                spelling = "mid"
                offset = 0
                if self.token.kind in ("+", "-"):
                    sign = self.take(self.token.kind).text
                    delta = int(self.take("INT").text)
                    spelling += f"{sign}{delta}"
                    offset = delta if sign == "+" else -delta
                n = self.shape.get("n_layers")
                index = n // 2 + offset if n is not None else None
            else:
                index = int(self.take("INT").text)
                spelling = str(index)
            self.take("]")
        target = Target(spelling, index=index, col=token.col)
        if with_part and self.eat("."):
            part = self.take("IDENT")
            target.part_col = part.col
            if part.text in ("mlp", "heads"):
                target.part = part.text
            elif part.text == "H" or HNUMBER.fullmatch(part.text):
                target.part = "head"
                if part.text == "H":
                    self.take("[")
                    target.head = int(self.take("INT").text)
                    self.take("]")
                else:
                    target.head = int(part.text[1:])
            else:
                raise Problem("E_SYNTAX", part.col, "expected mlp, heads or a head index")
        return target

    def expression(self):
        return self.layer() if self.layer_start() else self.call()

    def bracketed(self):
        col = self.take("[").col
        if self.eat("]"):
            return ListValue([], [], col)
        first_col = self.token.col
        value = self.value()
        if self.eat(":"):
            if type(value) not in (int, float):
                raise Problem("E_SYNTAX", first_col, "expected a number at the start of the range")
            end = self.number()
            self.take(":")
            step = self.number()
            self.take("]")
            return Range(value, end, step, col)
        items, cols = [value], [first_col]
        while self.eat(","):
            cols.append(self.token.col)
            items.append(self.value())
        self.take("]")
        return ListValue(items, cols, col)

    def value(self):
        token = self.token
        if token.kind in ("INT", "FLOAT", "-"):
            return self.number()
        if token.kind == "STRING":
            self.take("STRING")
            # These are the only escapes accepted by lex.
            return String(re.sub(r'\\(["\\])', r'\1', token.text[1:-1]))
        if token.kind == "[":
            return self.bracketed()
        if token.text in ("true", "false"):
            self.take("IDENT")
            return token.text == "true"
        if self.layer_start():
            return self.layer(with_part=True)
        if token.kind == "IDENT" and token.text != "resid" and self.peek("("):
            return self.call()
        return self.take("IDENT").text

    def call(self):
        name = self.take("IDENT")
        self.take("(")
        args = []
        if name.text == "align":
            col = self.token.col
            args.append(Argument(None, self.expression(), col))
            self.take("->")
            col = self.token.col
            args.append(Argument(None, self.expression(), col))
        elif self.token.kind != ")":
            while True:
                key_col = self.token.col
                key = None
                if self.token.kind == "IDENT" and self.peek("="):
                    key = self.take("IDENT").text
                    self.take("=")
                col = self.token.col
                args.append(Argument(key, self.value(), col, key_col))
                if not self.eat(","):
                    break
        self.take(")")
        return Call(name.text, args, name.col)

    def op(self):
        if self.layer_start():
            target = self.layer()
            self.take(":=")
            return Assignment(target, self.expression(), target.col)
        return self.call()

    def statement(self, line, model, seed):
        label, label_col = None, self.token.col
        if self.token.kind == "IDENT" and self.peek(":"):
            label = self.take("IDENT").text
            self.take(":")
        claim = self.take("IDENT")
        if claim.text not in CLAIMS:
            raise Problem("E_SYNTAX", claim.col, "expected a claim kind")
        tol = None
        if claim.text == "approx":
            tol = 0.05
            if self.eat("("):
                tol = self.number()
                self.take(")")
        ops = [self.op()]
        while self.eat(";"):
            ops.append(self.op())
        self.take("EOF")
        return Statement(line, label, claim.text, tol, ops, model, seed, label_col, claim.col)


def parse(script):
    statements, errors = [], []
    model, seed = None, 0
    for line, source in enumerate(script.splitlines(), 1):
        prefix = source.partition("#")[0].strip()
        if not prefix:
            continue
        try:
            directive = DIRECTIVE.fullmatch(source)
            if directive:
                name, value = directive[1], directive[2].partition("#")[0].strip()
                if name == "model":
                    model = value if valid_model(value) else None
                    if model is None:
                        raise Problem("E_MODEL", 1, f"unknown model: {value}")
                elif re.fullmatch(r"[0-9]+", value):
                    seed = int(value)
                else:
                    raise Problem("E_SYNTAX", 1, "expected a nonnegative integer seed")
                continue
            statements.append(LineParser(source, model).statement(line, model, seed))
        except Problem as exc:
            errors.append(exc.diagnostic(line))
        except (ValueError, RecursionError) as exc:
            errors.append(Problem("E_SYNTAX", 1, f"could not parse line: {exc}").diagnostic(line))
    return statements, errors
