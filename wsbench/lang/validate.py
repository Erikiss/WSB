"""Static WSL checks, in the diagnostic priority order of the language."""
import math
from wsbench.models.registry import SHAPES, TXL_UNDEFINED, param_shapes, txl_tied
from .ast import Assignment, Call, Range, String, Target, reserved
from .lexer import Problem


NEOX = {"fold_ln", "center_writers", "ln_to_rms", "rotate", "scale", "permute",
        "vo", "qk", "rope_pairs", "delete", "swap", "gridop"}
# Transformer-XL has the layer-local operations; the others have no meaning there (UNDEFINED gives the reason).
TXL = NEOX - {"fold_ln", "center_writers", "ln_to_rms", "rope_pairs"}
LAYER_VALUES = {"perm", "uniform", "lerp", "neg", "neg_writers", "align", "block"}
TOP = {
    **{claim: NEOX for claim in ("edit", "gauge", "reparam", "approx")},
    "train": {"modadd"}, "path": {"lerp", "add", "scale", "cosine"},
    "solve": {"prim", "chain", "compile", "golf"},
}
WEIGHTS = {"neox", "txl"}           # families whose statements edit the weight tensors of a language model
CLAIMS = {"neox": {"gauge", "reparam", "edit", "approx"}, "txl": {"gauge", "reparam", "edit", "approx"},
          "grid": {"solve"}, "modadd": {"train", "path"}}
NAMES = {
    "neox": NEOX | LAYER_VALUES,
    "txl": TXL | LAYER_VALUES,
    "modadd": TOP["train"] | TOP["path"] | {"ckpt", "diff", "grokfast", "at"},
    "grid": TOP["solve"],
}
# Operations, or forms of them, that a family does not define: E_MODEL with the reason.
UNDEFINED = {"txl": TXL_UNDEFINED}
PRIMITIVES = {"identity", "rot90", "rot180", "rot270", "hmirror", "vmirror", "dmirror", "cmirror"}
SQUARE_ONLY = {"rot90", "rot270", "dmirror", "cmirror"}

# Every positional and keyword slot in a signature has a WSL value type.
# A trailing ! on a keyword indicates it is required.
SIGNATURES = {
    **{name: ((), {}) for name in ("fold_ln", "center_writers", "ln_to_rms", "golf")},
    "resid_rotate": (("resid", "resid_basis"), {"layers": "layers", "embed": "bool", "unembed": "bool"}),
    "mlp_rotate": (("mlp", "orth"), {}),
    "resid_scale": (("resid", "number"), {}),
    "mlp_scale": (("mlp", "uniform"), {}),
    "permute": (("permutable", "permutation"), {}),
    "vo": (("head", "head_basis"), {}),
    "qk": (("head", "head_basis"), {"dims": "dims"}),
    "rope_pairs": (("head",), {"phis!": "phis", "scale": "number"}),
    "swap": (("layer", "layer"), {}), "delete": (("layer",), {}),
    "gridop": (("primitive", "block"), {}), "block": (("string",) + ("integer",) * 4, {}),
    "perm": (("permutation_entries",), {}), "uniform": (("number", "number"), {}),
    "layer_lerp": (("expression", "expression", "number"), {}),
    "neg": (("expression",), {}), "neg_writers": (("expression",), {}),
    "align": (("expression", "expression"), {}),
    "modadd": ((), {"run!": "run", "p": "integer", "frac": "number", "steps": "integer",
                    "seed": "integer", "lr": "number", "wd": "number", "filter": "filter",
                    "save": "saved_steps", "intervene": "interventions"}),
    "at": (("integer", "intervention_scale"), {}), "intervention_scale": (("number",), {}),
    "grokfast": (("number", "number"), {}), "ckpt": (("run", "integer"), {}),
    "diff": (("ckpt", "ckpt"), {}), "path_lerp": (("ckpt", "ckpt"), {"t!": "range"}),
    "add": (("ckpt", "diff"), {"t!": "range"}), "path_scale": (("ckpt",), {"c!": "range"}),
    "cosine": (("vector", "vector"), {}),
    "prim": (("primitive",), {}), "compile": (("primitive",), {"size": "size", "from": "from"}),
}

# Call names that may occupy a value slot. Other slots expect literal data.
CALLS = {
    "expression": {"lerp", "neg", "neg_writers", "align"}, "permutation": {"perm"},
    "uniform": {"uniform"}, "block": {"block"}, "filter": {"grokfast"}, "at": {"at"},
    "ckpt": {"ckpt"}, "diff": {"diff"}, "vector": {"ckpt", "diff"},
    "intervention_scale": {"scale"}, "prim": {"prim"},
}
ENUMS = {
    "resid": {"resid"}, "resid_basis": {"orth", "orth1"}, "orth": {"orth"},
    "head_basis": {"gl", "orth"}, "dims": {"all", "pass"}, "primitive": PRIMITIVES,
    "size": {"fixed", "gated"}, "from": {"all", "train"},
}


def bad(col, message):
    raise Problem("E_BAD_ARG", col, message)


def member(value, choices):
    return type(value) is str and value in choices


def resid_form(call):
    return bool(call.args) and call.args[0] == "resid"


def signature(call, family, slot):
    name = call.name
    if name in ("rotate", "scale") and family in WEIGHTS:
        name = ("resid_" if resid_form(call) else "mlp_") + name
    elif name == "scale":
        name = "intervention_scale" if slot == "intervention_scale" else "path_scale"
    elif name == "lerp":
        name = "layer_lerp" if family in WEIGHTS else "path_lerp"
    if name == "chain":
        return ("prim",) * max(1, len(call.args)), {}
    return SIGNATURES[name]


def typed_args(call, positional, keywords):
    """Yield all arguments, even malformed ones, so names take precedence."""
    pos = 0
    for arg in call.arguments:
        if arg.key is None:
            slot = positional[pos] if pos < len(positional) else None
            pos += 1
        else:
            slot = keywords.get(arg.key, keywords.get(arg.key + "!"))
        yield arg, slot


def check_name(name, allowed, family, col):
    if name in allowed:
        return
    if name in NAMES[family]:
        bad(col, f"{name} is not allowed here")
    if name in UNDEFINED.get(family, {}):
        raise Problem("E_MODEL", col, UNDEFINED[family][name])
    if any(name in names for names in NAMES.values()):
        raise Problem("E_MODEL", col, f"{name} belongs to another model family")
    raise Problem("E_UNKNOWN_OP", col, f"unknown call {name}")


def check_form(call, family):
    """E_MODEL for a form of an allowed call that the family does not define: on Transformer-XL rotate(resid, ...),
    scale(resid, ...) and qk(..., dims=pass)."""
    undefined = UNDEFINED.get(family, {})
    if call.name in ("rotate", "scale") and resid_form(call) and call.name + "_resid" in undefined:
        raise Problem("E_MODEL", call.col, undefined[call.name + "_resid"])
    if call.name == "qk" and "qk_pass" in undefined:
        for arg in call.arguments:
            if arg.key == "dims" and member(arg.value, {"pass"}):
                raise Problem("E_MODEL", arg.key_col, undefined["qk_pass"])


def check_names(value, slot, family, allowed=None):
    if isinstance(value, Assignment):
        if family not in WEIGHTS:
            raise Problem("E_MODEL", value.col, "assignments require a neox or txl model")
        check_names(value.expr, "expression", family)
    elif isinstance(value, Call):
        check_name(value.name, CALLS.get(slot, ()) if allowed is None else allowed, family, value.col)
        check_form(value, family)
        for arg, expected in typed_args(value, *signature(value, family, slot)):
            check_names(arg.value, expected, family)
    elif isinstance(value, list):
        for item in value:
            check_names(item, "at" if slot == "interventions" else None, family)


def layout(call, positional, keywords):
    seen = set()
    pos = 0
    for arg in call.arguments:
        if arg.key is None:
            if seen or pos >= len(positional):
                bad(arg.col, f"unexpected positional argument to {call.name}")
            pos += 1
        elif arg.key in seen:
            bad(arg.key_col, f"repeated keyword {arg.key}")
        elif arg.key not in keywords and arg.key + "!" not in keywords:
            bad(arg.key_col, f"unknown keyword {arg.key}")
        else:
            seen.add(arg.key)
    if pos < len(positional):
        bad(call.col, f"{call.name} requires {len(positional)} positional arguments")
    for key in keywords:
        if key.endswith("!") and key[:-1] not in seen:
            bad(call.col, f"{call.name} requires {key[:-1]}=")


def number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def literal_ok(value, slot):
    if slot in ENUMS:
        return member(value, ENUMS[slot])
    if slot in ("permutable", "layer", "expression", "mlp", "head"):
        parts = {"permutable": ("mlp", "heads"), "layer": (None,), "expression": (None,),
                 "mlp": ("mlp",), "head": ("head",)}
        return isinstance(value, Target) and value.part in parts[slot]
    if slot == "number":
        return number(value)
    if slot == "integer":
        return type(value) is int
    if slot == "string":
        return isinstance(value, String)
    if slot == "bool":
        return type(value) is bool
    if slot == "run":
        return type(value) is str and not reserved(value)
    if slot == "range":
        if isinstance(value, Range) and all(map(number, value)):
            a, b, s = value
            return s != 0 and (b - a) * s >= 0
        return False
    if slot == "permutation":
        return member(value, {"randperm"})
    if slot == "filter":
        return member(value, {"none"})
    return False


def check_value(value, slot, col, shape, parent):
    if isinstance(value, Call):
        check_call(value, slot, shape)
        return
    if slot == "layers" and member(value, {"all"}):
        return
    if slot in ("layers", "permutation_entries", "saved_steps", "phis", "interventions"):
        if not isinstance(value, list):
            bad(col, f"expected a list for {slot}")
        if slot == "phis" and len(value) != shape["rd"] // 2:
            bad(col, "expected one angle per rotary pair")
        for i, item in enumerate(value):
            item_col = value.cols[i]
            if slot == "interventions":
                if not isinstance(item, Call):
                    bad(item_col, "expected at(step, scale(c))")
                check_call(item, "at", shape)
            elif slot == "phis":
                if not number(item):
                    bad(item_col, "expected a numerical angle")
            elif type(item) is not int:
                bad(item_col, "expected an integer")
            elif slot == "saved_steps" and not 0 <= item <= parent.kwargs.get("steps", 3000):
                bad(item_col, "saved step must be in 0..steps")
        return
    if not literal_ok(value, slot):
        bad(col, f"expected {slot} in {parent.name}")


def check_call(call, slot, shape):
    pos, kw = signature(call, shape["family"], slot)
    layout(call, pos, kw)
    # Save bounds depend on steps even when steps appears later in the source.
    if call.name == "modadd" and type(call.kwargs.get("steps", 3000)) is not int:
        arg = next(a for a in call.arguments if a.key == "steps")
        bad(arg.col, "expected integer steps")
    for arg, expected in typed_args(call, pos, kw):
        check_value(arg.value, expected, arg.col, shape, call)
    args = call.args
    if call.name == "permute" and isinstance(args[1], Call):
        n = shape["d_mlp" if args[0].part == "mlp" else "n_heads"]
        if sorted(args[1].args[0]) != list(range(n)):
            bad(args[1].col, f"expected a permutation of 0..{n - 1}")
    elif call.name == "uniform" and not 0 < args[0] < args[1]:
        bad(call.col, "uniform requires 0 < lo < hi")
    elif call.name == "block" and (args[1] >= args[2] or args[3] >= args[4]):
        bad(call.col, "block bounds must be strictly increasing")


def walk(value):
    """Yield syntax nodes in source order, including nested calls."""
    yield value
    if isinstance(value, Call):
        for arg in value.arguments:
            yield from walk(arg.value)
    elif isinstance(value, Assignment):
        yield from walk(value.target)
        yield from walk(value.expr)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def layer_bound(i, col, shape, model):
    n = shape["n_layers"]
    if not 0 <= i < n:
        raise Problem("E_TARGET", col, f"layer {i} out of range ({model} has L0..L{n - 1})")


def check_target(node, shape, model):
    if isinstance(node, Target):
        layer_bound(node.index, node.col, shape, model)
        if node.head is not None and not 0 <= node.head < shape["n_heads"]:
            raise Problem("E_TARGET", node.part_col, f"head {node.head} out of range")
    if isinstance(node, Call) and node.name == "rotate":
        layers = node.kwargs.get("layers")
        if isinstance(layers, list):
            for i, col in zip(layers, layers.cols):
                layer_bound(i, col, shape, model)
    if isinstance(node, Call) and node.name == "block":
        name, r0, r1, c0, c1 = node.args
        dims = param_shapes(shape).get(name)
        stored = txl_tied(shape).get(name) if shape["family"] == "txl" else None
        if stored is not None:
            raise Problem("E_TARGET", node.arguments[0].col, f"{name} is tied to {stored}: use that name "
                          "(one tensor, used by the embedding and the softmax)")
        if dims is None or len(dims) != 2:
            raise Problem("E_TARGET", node.arguments[0].col, f"unknown or non-matrix parameter {name}")
        if not (0 <= r0 < r1 <= dims[0] and 0 <= c0 < c1 <= dims[1]):
            raise Problem("E_TARGET", node.col, f"block outside {dims[0]} by {dims[1]} tensor")


def check(stmt, seen):
    if stmt.label:
        if reserved(stmt.label):
            bad(stmt.label_col, f"reserved label {stmt.label}")
        if stmt.label in seen:
            raise Problem("E_LABEL", stmt.label_col, f"duplicate label {stmt.label}")
        seen.add(stmt.label)
    if stmt.family is None:
        raise Problem("E_MODEL", stmt.claim_col, "no model in force")
    if stmt.claim not in CLAIMS[stmt.family]:
        raise Problem("E_MODEL", stmt.claim_col, f"{stmt.claim} is invalid for {stmt.model}")
    if stmt.claim in ("train", "path", "solve") and len(stmt.ops) != 1:
        bad(stmt.claim_col, f"{stmt.claim} requires one op")
    for op in stmt.ops:
        check_names(op, None, stmt.family, TOP[stmt.claim] & NAMES[stmt.family])
    shape = SHAPES[stmt.model]
    for op in stmt.ops:
        if isinstance(op, Assignment):
            check_value(op.expr, "expression", op.expr.col, shape, op)
        else:
            check_call(op, None, shape)
    if stmt.family in WEIGHTS:
        for op in stmt.ops:
            for node in walk(op):
                check_target(node, shape, stmt.model)
        for op in stmt.ops:
            if isinstance(op, Call) and op.name == "gridop" and op.args[0] in SQUARE_ONLY:
                _, r0, r1, c0, c1 = op.args[1].args
                if r1 - r0 != c1 - c0:
                    raise Problem("E_SHAPE", op.col, f"{op.args[0]} requires a square block")


def validate(statements):
    errors, seen = [], set()
    for stmt in statements:
        try:
            check(stmt, seen)
        except Problem as exc:
            errors.append(exc.diagnostic(stmt.line))
    return errors
