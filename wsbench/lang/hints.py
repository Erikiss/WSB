"""Static scientific hints, evaluated only for valid NeoX and Transformer-XL statements."""
from .ast import Call, Target
from .validate import validate, walk


MESSAGES = {
    "H_GELU_SCALE": "GELU is not positively homogeneous, so per-neuron rescaling is not a symmetry. Use permute(L[{i}].mlp, randperm), or write 'edit' to measure it.",
    "H_MLP_ROTATE": "Rotating MLP neurons is not a symmetry through an elementwise activation. Only permutations are.",
    "H_RELU_ROTATE": "Rotating MLP neurons is not a symmetry through ReLU. Only permutations and positive per-neuron rescalings are.",
    "H_RESID_NOFOLD": "LayerNorm gains pin a basis. Put fold_ln() before the rotation.",
    "H_RESID_ONES": "LayerNorm centres the stream. Use orth1 (R·1 = 1), or center_writers(); ln_to_rms() first.",
    "H_RESID_PARTIAL": "A residual rotation is a symmetry only if it is applied to every reader and writer, the embedding and the unembedding.",
    "H_ROPE_QK": "RoPE breaks GL symmetry on the rotary dims. Use dims=pass, or rope_pairs(...) for the rotary dims.",
    "H_RMS_ORDER": "ln_to_rms() is exact only after fold_ln(); center_writers().",
    "H_LERP_ALIGN": "Mixing two layers without align() depends on the permutation gauge.",
    "H_EDIT_GAUGE": "Every op here is a known symmetry: expect W_NOOP.",
}


def mlp_target(op):
    return bool(op.args) and isinstance(op.args[0], Target) and op.args[0].part == "mlp"


def known_symmetry(op, family="neox"):
    if not isinstance(op, Call):
        return False
    if family == "txl":    # ReLU: positive per-neuron rescaling is exact; Q/K acts on all dims (no RoPE)
        return op.name in {"permute", "vo", "qk"} or (op.name == "scale" and mlp_target(op))
    if op.name == "qk":
        return op.kwargs.get("dims", "pass") == "pass"
    return op.name in {"permute", "vo", "rope_pairs", "fold_ln", "center_writers"}


def statement_hints(stmt):
    found = {}
    neox = stmt.family == "neox"

    def mark(code, node, **values):
        found.setdefault(code, {"code": code, "line": stmt.line, "col": node.col,
                                "message": MESSAGES[code].format(**values)})

    earlier = set()
    for op in stmt.ops:
        if isinstance(op, Call):
            args, kw = op.args, op.kwargs
            if stmt.claim == "gauge":
                if op.name in ("scale", "rotate") and mlp_target(op):
                    if op.name == "rotate":
                        mark("H_MLP_ROTATE" if neox else "H_RELU_ROTATE", op)
                    elif neox:
                        mark("H_GELU_SCALE", op, i=args[0].index)
                if op.name == "rotate" and args[0] == "resid":
                    if "fold_ln" not in earlier:
                        mark("H_RESID_NOFOLD", op)
                    if args[1] == "orth" and "ln_to_rms" not in earlier:
                        mark("H_RESID_ONES", op)
                    if kw.get("layers", "all") != "all" or not kw.get("embed", True) or not kw.get("unembed", True):
                        mark("H_RESID_PARTIAL", op)
                if op.name == "qk" and kw.get("dims") == "all" and neox:
                    mark("H_ROPE_QK", op)
            if op.name == "ln_to_rms" and "center_writers" not in earlier:
                mark("H_RMS_ORDER", op)
            earlier.add(op.name)
        for node in walk(op):
            if isinstance(node, Call) and node.name == "lerp":
                a, b = node.args[:2]
                if isinstance(a, Target) and isinstance(b, Target) and a.index != b.index:
                    mark("H_LERP_ALIGN", op)
    if stmt.claim == "edit" and all(known_symmetry(op, stmt.family) for op in stmt.ops):
        mark("H_EDIT_GAUGE", stmt.ops[0])
    return [found[code] for code in MESSAGES if code in found]


def hints(statements):
    invalid = {error["line"] for error in validate(statements)}
    return [hint for stmt in statements if stmt.line not in invalid and stmt.family in ("neox", "txl")
            for hint in statement_hints(stmt)]
