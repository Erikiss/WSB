from pathlib import Path
import subprocess
import sys

import pytest

from wsbench.lang.parser import parse, expand_range
from wsbench.lang.printer import print_statement
from wsbench.lang.validate import validate


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "wsbench/examples").glob("*.wsl")) + [ROOT / "tests/data/tiny_battery.wsl"]


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
