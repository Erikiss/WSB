import pytest

from wsbench.lang.hints import hints
from wsbench.lang.parser import parse


@pytest.mark.parametrize(("code", "positive", "negative"), [
    ("H_GELU_SCALE", "gauge scale(L2.mlp,uniform(.5,2))", "edit scale(L2.mlp,uniform(.5,2))"),
    ("H_MLP_ROTATE", "gauge rotate(L2.mlp,orth)", "edit rotate(L2.mlp,orth)"),
    ("H_RESID_NOFOLD", "gauge rotate(resid,orth1)", "gauge fold_ln(); rotate(resid,orth1)"),
    ("H_RESID_ONES", "gauge fold_ln();rotate(resid,orth)", "gauge fold_ln();rotate(resid,orth1)"),
    ("H_RESID_PARTIAL", "gauge rotate(resid,orth1,layers=[1])", "gauge rotate(resid,orth1,layers=all)"),
    ("H_ROPE_QK", "gauge qk(L2.H0,gl,dims=all)", "gauge qk(L2.H0,gl)"),
    ("H_RMS_ORDER", "edit ln_to_rms(); center_writers()", "edit center_writers(); ln_to_rms()"),
    ("H_LERP_ALIGN", "edit L0 := lerp(L1,L[mid],.5)", "edit L0 := lerp(L2,L[mid],.5)"),
    ("H_EDIT_GAUGE", "edit permute(L2.heads,randperm);qk(L1.H0,gl)", "edit qk(L1.H0,gl,dims=all)"),
])
def test_hints_are_conditional(code, positive, negative):
    for source, expected in [(positive, True), (negative, False)]:
        statements, errors = parse("model neox-tiny\n" + source)
        assert errors == []
        warnings = hints(statements)
        assert (code in [h["code"] for h in warnings]) is expected
        for h in warnings:
            assert set(h) == {"line", "col", "code", "message"}
            assert 0 < h["col"] <= len(source) and h["line"] == 2
            if h["code"] == "H_GELU_SCALE":
                assert "permute(L[2].mlp, randperm)" in h["message"]


def test_table_order_deduplication_and_invalid_lines():
    line = "gauge rotate(resid,orth,embed=false); rotate(resid,orth); scale(L0.mlp, uniform(.1,1))"
    script = "model neox-tiny\n" + line + "\ngauge rotate(resid, orth); delete(L99)"
    warnings = hints(parse(script)[0])
    assert [h["code"] for h in warnings] == ["H_GELU_SCALE", "H_RESID_NOFOLD", "H_RESID_ONES", "H_RESID_PARTIAL"]
    assert [h["col"] for h in warnings] == [line.index("scale")+1, 7, 7, 7]
