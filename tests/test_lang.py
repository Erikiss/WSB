from pathlib import Path
import subprocess
import sys

import pytest

from wsbench.lang.parser import parse, expand_range
from wsbench.lang.printer import print_statement
from wsbench.lang.validate import validate


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "wsbench/examples").glob("*.wsl")) + [ROOT / "tests/data/tiny_battery.wsl",
                                                             ROOT / "tests/data/txl_battery.wsl"]


def problems(source):
    statements, errors = parse(source)
    return errors + validate(statements)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_examples_round_trip(path):
    statements, errors = parse(path.read_text())
    assert errors == validate(statements) == []
    assert statements
    for st in statements:
        assert parse(print_statement(st, with_label=True)) == ([st], [])


def test_assist_blocks_round_trip_and_canonical_printing():
    source = """model pythia-14m
gauge fold_ln(); center_writers(); ln_to_rms(); rotate(resid, orth)
gauge rotate(resid, orth)
edit rotate(resid, orth, layers=[2], embed=false, unembed=false)
gauge fold_ln(); rotate(resid, orth1)
model: approx vo(L3.H0, orth)
"""
    statements, errors = parse(source)
    assert errors == validate(statements) == []
    for stmt in statements:
        assert parse(print_statement(stmt, with_label=True)) == ([stmt], [])
    assert print_statement(statements[-1], False) == "approx(0.05) vo(L[3].H[0], orth)"


@pytest.mark.parametrize(("source", "code"), [
    ("gauge rotate(resid, orth", "E_SYNTAX"),
    ("gauge frobnicate(L3)", "E_UNKNOWN_OP"), ("edit delete(L[7])", "E_TARGET"),
    ("edit delete(L[mid+3])", "E_TARGET"), ("gauge vo(L3.H[4], gl)", "E_TARGET"),
    ("solve prim(rot90)", "E_MODEL"), ("train modadd(run=A)", "E_MODEL"),
    ("edit swap(L2)", "E_BAD_ARG"), ("gauge permute(L3.heads, perm([0, 0, 1, 2]))", "E_BAD_ARG"),
])
def test_acceptance_errors(source, code):
    err, = problems("model pythia-14m\n" + source)
    assert err["code"] == code
    assert err["line"] == 2
    assert 1 <= err["col"] <= len(source) + 1
    assert set(err) == {"line", "col", "code", "message"}


def test_directive_context_and_line_recovery():
    script = """edit delete(L0)
model neox-tiny
seed 9
wrong: edit delete(L0     # incomplete
seed -1
a: edit delete(L0)
 model unknown
b: edit delete(L0)
model pythia-14m
c: edit delete(L[mid-1])
"""
    st, errors = parse(script)
    assert [(e["line"], e["code"], e["col"]) for e in errors] == [
        (4, "E_SYNTAX", 22), (5, "E_SYNTAX", 1), (7, "E_MODEL", 1)]
    assert [(s.model, s.seed) for s in st] == [(None, 0), ("neox-tiny", 9), (None, 9), ("pythia-14m", 9)]
    assert [(e["line"], e["code"]) for e in validate(st)] == [(1, "E_MODEL"), (8, "E_MODEL")]
    assert st[-1].ops[0].args[0].index == 2


def test_directive_and_label_lexing():
    script = "model grid:task001\nmodel: solve golf()\nseed: solve prim(identity)\nmodel grid:task400"
    st, err = parse(script)
    assert [s.label for s in st] == ["model", "seed"]
    assert err == validate(st) == []
    for directive in ("model grid:task000", "model grid:task401", " model grid:task01"):
        assert parse(directive)[1][0]["code"] == "E_MODEL"


def test_ranges():
    assert expand_range(1, 0.5, -0.1) == pytest.approx([1, .9, .8, .7, .6, .5], abs=1e-12)
    assert expand_range(0, 1, .25) == [0, .25, .5, .75, 1]
    assert expand_range(0, 0.3, .1) == pytest.approx([0, .1, .2, .3])
    assert expand_range(0, .91, .25) == [0, .25, .5, .75]
    for step in ("0", "-0.5"):
        line = f"path lerp(ckpt(A, 0), ckpt(A, 1), t=[0:1:{step}])"
        st, errors = parse("model modadd\n" + line)
        assert not errors
        error, = validate(st)
        assert (error["code"], error["col"]) == ("E_BAD_ARG", line.index("[") + 1)


@pytest.mark.parametrize(("model", "source", "code"), [
    ("neox-tiny", "L1: solve missing()", "E_BAD_ARG"),
    ("neox-tiny", "line7: edit delete(L99)", "E_BAD_ARG"),
    ("neox-tiny", "edit lerp(L0,L1,0.5)", "E_BAD_ARG"),
    ("neox-tiny", "edit delete(prim(rot90))", "E_MODEL"),
    ("neox-tiny", "edit swap(L0, align(L1 -> L2))", "E_BAD_ARG"),
    ("neox-tiny", "edit delete(L99); scale(resid, broken())", "E_UNKNOWN_OP"),
    ("neox-tiny", "gauge rotate(resid, gl)", "E_BAD_ARG"),
    ("neox-tiny", "gauge permute(L0.heads, perm([1,0.0]))", "E_BAD_ARG"),
    ("neox-tiny", "gauge rotate(resid, orth, layers=[10])", "E_TARGET"),
    ("neox-tiny", "gauge rotate(resid, orth, layers=[true])", "E_BAD_ARG"),
    ("neox-tiny", "gauge rotate(resid, orth, layers=all, layers=all)", "E_BAD_ARG"),
    ("neox-tiny", "gauge rotate(resid, layers=all, orth)", "E_BAD_ARG"),
    ("neox-tiny", "gauge rotate(resid, orth, unknown=true)", "E_BAD_ARG"),
    ("neox-tiny", 'edit gridop(rot90, block("gpt_neox.embed_in.weight",0,2,0,3))', "E_SHAPE"),
    ("neox-tiny", 'edit gridop(rot90, block("gpt_neox.embed_in.weight",-1,2,0,3))', "E_TARGET"),
    ("neox-tiny", 'edit gridop(rot90, block("embed_out.bias",0,2,0,3))', "E_TARGET"),
    ("neox-tiny", 'edit gridop(rot90, block("gpt_neox.final_layer_norm.weight",0,2,0,3))', "E_TARGET"),
    ("neox-tiny", 'edit gridop(rot90, block("unknown",2,1,0,2))', "E_BAD_ARG"),
    ("modadd", "path modadd(run=A)", "E_BAD_ARG"),
    ("modadd", "train modadd(run=H1)", "E_BAD_ARG"),
    ("modadd", "train modadd(run=A, steps=1, save=[2])", "E_BAD_ARG"),
    ("modadd", "train modadd(run=A, intervene=[at(1,scale(c=1))])", "E_BAD_ARG"),
    ("modadd", "path cosine(ckpt(A, 2.0), ckpt(A,0))", "E_BAD_ARG"),
    ("modadd", "path lerp(ckpt(A,0), ckpt(A,1), t=[0,1])", "E_BAD_ARG"),
    ("grid:task008", "solve golf(); missing()", "E_BAD_ARG"),
    ("grid:task008", "solve chain()", "E_BAD_ARG"),
    ("grid:task008", "solve chain(prim(rot90), golf())", "E_BAD_ARG"),
    ("grid:task008", "solve prim(frob)", "E_BAD_ARG"),
    ("grid:task008", 'solve prim("rot90")', "E_BAD_ARG"),
    ("grid:task008", "solve prim(ckpt(A,0))", "E_MODEL"),
])
def test_multiphase_validation(model, source, code):
    assert [e["code"] for e in problems("model " + model + "\n" + source)] == [code]


def test_labels_checked_before_family_and_name():
    st, errors = parse("x: edit delete(L0)\nmodel modadd\n   x: gauge frob()")
    assert errors == []
    assert [(e["line"], e["col"], e["code"]) for e in validate(st)] == [
        (1, 4, "E_MODEL"), (3, 4, "E_LABEL")]


def test_literals_round_trip():
    src = r'edit foo(2, 2., -1E-3, .25, "a\"b\\c#d", [0.5:1:.1], [true, [1, 2]])'
    st, err = parse(src)
    assert not err and len(st) == 1
    assert parse(print_statement(st[0])) == (st, [])
    assert print_statement(st[0]).startswith("edit foo(2, 2.0, -0.001, 0.25,")


def test_torch_free_in_clean_process():
    subprocess.run([sys.executable, "-c", """
from wsbench.lang.parser import parse
from wsbench.lang.validate import validate
from wsbench.lang.hints import hints
import sys
s, e = parse('model neox-tiny\\ngauge rotate(resid, orth)')
assert not e and not validate(s) and hints(s)
assert 'torch' not in sys.modules
"""], check=True)


def test_unrepresentable_float_has_a_json_safe_diagnostic():
    src = "approx(-1e999) fold_ln()"
    st, errors = parse("model neox-tiny\n" + src)
    assert not st
    assert [(e["code"], e["col"]) for e in errors] == [("E_BAD_ARG", src.index("-") + 1)]


# ---------------------------------------------------------------- Transformer-XL

def test_txl_models_parse_with_their_family_and_geometry():
    from wsbench.lang.parser import valid_model
    assert valid_model("txl-tiny") and valid_model("transfo-xl-wt103") and not valid_model("transfo-xl")
    script = """model transfo-xl-wt103
a: edit delete(L[mid]); swap(L[mid-1], L[mid+1])
model txl-tiny
b: gauge qk(L[mid].H[1], gl)
model modadd
c: path cosine(ckpt(A, 0), ckpt(A, 1))
model grid:task001
d: solve golf()
model pythia-14m
e: gauge fold_ln()
"""
    statements, errors = parse(script)
    assert errors == validate(statements) == []
    assert [s.family for s in statements] == ["txl", "txl", "modadd", "grid", "neox"]
    wt103, tiny = statements[:2]
    assert [op.args[0].index for op in wt103.ops] == [9, 8] and wt103.ops[1].args[1].index == 10
    assert tiny.ops[0].args[0].index == 2
    assert parse("model transfo-xl\nedit delete(L0)")[1][0]["code"] == "E_MODEL"


UNDEFINED_ON_TXL = [
    ("gauge fold_ln()", "fold_ln", "fold_ln"),
    ("reparam center_writers()", "center_writers", "center_writers"),
    ("edit ln_to_rms()", "ln_to_rms", "ln_to_rms"),
    ("gauge rotate(resid, orth1)", "rotate", "rotate_resid"),
    ("edit rotate(resid, orth, layers=[1], embed=false, unembed=false)", "rotate", "rotate_resid"),
    ("approx(0.01) scale(resid, 2)", "scale", "scale_resid"),
    ("gauge rope_pairs(L1.H0, phis=[0.1, 0.2], scale=2)", "rope_pairs", "rope_pairs"),
    ("gauge qk(L1.H0, gl, dims=pass)", "dims", "qk_pass"),
    ("gauge vo(L1.H0, gl); qk(L1.H1, orth, dims=pass)", "dims", "qk_pass"),
    ("edit delete(L1); fold_ln()", "fold_ln", "fold_ln"),
    ("edit delete(fold_ln())", "fold_ln", "fold_ln"),
]


@pytest.mark.parametrize(("source", "at", "key"), UNDEFINED_ON_TXL)
@pytest.mark.parametrize("model", ["txl-tiny", "transfo-xl-wt103"])
def test_neox_only_operations_are_undefined_on_txl(model, source, at, key):
    from wsbench.models import txl
    err, = problems(f"model {model}\n" + source)
    assert (err["line"], err["code"], err["col"]) == (2, "E_MODEL", source.index(at) + 1)
    assert err["message"] == txl.UNDEFINED[key]
    assert parse("model neox-tiny\n" + source)[1] == []


@pytest.mark.parametrize(("source", "code", "message"), [
    ("edit delete(L4)", "E_TARGET", "layer 4 out of range (txl-tiny has L0..L3)"),
    ("gauge vo(L0.H2, gl)", "E_TARGET", "head 2 out of range"),
    ("gauge permute(L0.mlp, perm([1, 0]))", "E_BAD_ARG", "expected a permutation of 0..63"),
    ("gauge qk(L0.H0, gl, dims=rotary)", "E_BAD_ARG", "expected dims in qk"),
    ("solve prim(rot90)", "E_MODEL", "solve is invalid for txl-tiny"),
    ("edit golf()", "E_MODEL", "golf belongs to another model family"),
    ('edit gridop(rot90, block("crit.out_layers.1.weight", 0, 2, 0, 2))', "E_TARGET",
     "crit.out_layers.1.weight is tied to transformer.word_emb.emb_layers.1.weight: use that name "
     "(one tensor, used by the embedding and the softmax)"),
    ('edit gridop(rot90, block("crit.out_projs.3", 0, 2, 0, 2))', "E_TARGET",
     "crit.out_projs.3 is tied to transformer.word_emb.emb_projs.3: use that name "
     "(one tensor, used by the embedding and the softmax)"),
    ('edit gridop(rot90, block("transformer.pos_emb.inv_freq", 0, 2, 0, 2))', "E_TARGET",
     "unknown or non-matrix parameter transformer.pos_emb.inv_freq"),
    ('edit gridop(rot90, block("crit.cluster_bias", 0, 2, 0, 2))', "E_TARGET",
     "unknown or non-matrix parameter crit.cluster_bias"),
    ('edit gridop(rot90, block("transformer.word_emb.emb_layers.3.weight", 0, 2, 0, 3))', "E_TARGET",
     "block outside 16 by 2 tensor"),
    ('edit gridop(rot90, block("crit.out_projs.0", 0, 2, 0, 3))', "E_SHAPE", "rot90 requires a square block"),
])
def test_txl_targets_blocks_and_claims(source, code, message):
    err, = problems("model txl-tiny\n" + source)
    assert (err["code"], err["message"]) == (code, message)


def test_txl_blocks_follow_the_real_geometry():
    ok = """model transfo-xl-wt103
edit gridop(hmirror, block("transformer.word_emb.emb_layers.0.weight", 0, 20000, 0, 1024))
edit gridop(rot90, block("transformer.word_emb.emb_layers.3.weight", 67719, 67735, 0, 16))
edit gridop(vmirror, block("transformer.layers.17.dec_attn.qkv_net.weight", 0, 3072, 0, 1024))
edit gridop(dmirror, block("transformer.layers.0.pos_ff.CoreNet.3.weight", 0, 1024, 3072, 4096))
edit gridop(identity, block("crit.out_projs.0", 0, 1024, 0, 1024)); gridop(rot180, block("crit.cluster_weight", 0, 3, 0, 1024))
gauge permute(L17.heads, perm([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])); vo(L[mid].H[15], gl)
"""
    statements, errors = parse(ok)
    assert errors == validate(statements) == []
    for source in ('edit gridop(hmirror, block("transformer.word_emb.emb_layers.0.weight", 0, 20001, 0, 1024))',
                   'edit gridop(rot90, block("transformer.word_emb.emb_layers.1.weight", 0, 300, 0, 300))',
                   "gauge vo(L17.H16, gl)", "edit delete(L[mid+9])"):
        assert problems("model transfo-xl-wt103\n" + source)[0]["code"] == "E_TARGET", source


def test_txl_statements_print_canonically_and_round_trip():
    source = """model txl-tiny
a: gauge permute(L[mid].mlp,randperm); scale(L2.mlp, uniform(.5, 2.)); qk(L0.H[1], orth)
b: edit L[mid] := lerp(L[mid-1], align(L3 -> L[mid-1]), .25); swap(L0,L1)
c: edit gridop(rot90, block("transformer.word_emb.emb_layers.0.weight", 0, 4, 0, 4))
d: approx(0.5) L1 := neg(neg_writers(L2))
"""
    statements, errors = parse(source)
    assert errors == validate(statements) == []
    printed = [print_statement(s) for s in statements]
    assert printed == [
        "a: gauge permute(L[mid].mlp, randperm); scale(L[2].mlp, uniform(0.5, 2.0)); qk(L[0].H[1], orth)",
        "b: edit L[mid] := lerp(L[mid-1], align(L[3] -> L[mid-1]), 0.25); swap(L[0], L[1])",
        'c: edit gridop(rot90, block("transformer.word_emb.emb_layers.0.weight", 0, 4, 0, 4))',
        "d: approx(0.5) L[1] := neg(neg_writers(L[2]))",
    ]
    for stmt, text in zip(statements, printed):
        again, errors = parse("model txl-tiny\n" + text)
        assert errors == [] and again == [stmt] and again[0].ops[0] == stmt.ops[0]


def test_txl_validation_is_torch_free_in_clean_process():
    subprocess.run([sys.executable, "-c", """
from wsbench.lang.parser import parse
from wsbench.lang.validate import validate
from wsbench.lang.hints import hints
import sys
s, e = parse('model transfo-xl-wt103\\ngauge rotate(L[mid].mlp, orth)\\ngauge fold_ln()')
assert not e and [x['code'] for x in validate(s)] == ['E_MODEL'] and [h['code'] for h in hints(s)] == ['H_RELU_ROTATE']
assert 'torch' not in sys.modules
"""], check=True)


def test_the_card_names_every_txl_matrix_for_block():
    """Assist, Explore and the agent see only the card, so its block() list for Transformer-XL must name every 2-D
    parameter (and the tied names that are not parameters)."""
    import re

    from wsbench.lang.card import LANGUAGE_CARD
    from wsbench.models.registry import SHAPES, param_shapes, txl_tied

    section = LANGUAGE_CARD.split("Tensors for block()")[1].split("Not defined (E_MODEL)")[0]
    written = {    # each family of matrix names -> how the card writes it
        r"transformer\.layers\.\d+\.dec_attn\.(qkv_net|o_net|r_net)\.weight":
            "transformer.layers.{i}.dec_attn.{qkv_net,o_net,r_net}.weight",
        r"transformer\.layers\.\d+\.dec_attn\.r_[wr]_bias": ".{r_w_bias,r_r_bias}",
        r"transformer\.layers\.\d+\.pos_ff\.CoreNet\.[03]\.weight": "transformer.layers.{i}.pos_ff.CoreNet.{0,3}.weight",
        r"transformer\.word_emb\.emb_layers\.\d+\.weight": "transformer.word_emb.emb_layers.{c}.weight",
        r"transformer\.word_emb\.emb_projs\.\d+": "transformer.word_emb.emb_projs.{c}",
        r"crit\.out_projs\.0": "crit.out_projs.0",
        r"crit\.cluster_weight": "crit.cluster_weight",
    }
    for model in ("txl-tiny", "transfo-xl-wt103"):
        assert SHAPES[model]["tie_projs"] == (False, True, True, True)     # what the card says about the tying
        for name, shape in param_shapes(SHAPES[model]).items():
            assert (len(shape) == 2) == any(re.fullmatch(p, name) for p in written), name
        assert {re.sub(r"\.\d+\.", ".{c}.", n) for n in txl_tied(SHAPES[model]) if "out_layers" in n} == {
            "crit.out_layers.{c}.weight"}
    for text in [*written.values(), "crit.out_layers.{c}.weight", "crit.out_projs.{c} for c >= 1", "E_TARGET"]:
        assert text in section, text
    statements, errors = parse('model txl-tiny\nedit '
                               'gridop(rot180, block("transformer.layers.1.dec_attn.r_w_bias", 0, 2, 0, 8)); '
                               'gridop(hmirror, block("transformer.word_emb.emb_projs.1", 0, 16, 0, 8)); '
                               'gridop(rot90, block("crit.out_projs.0", 0, 16, 0, 16)); '
                               'gridop(hmirror, block("crit.cluster_weight", 0, 3, 0, 16))')
    assert errors == validate(statements) == []
