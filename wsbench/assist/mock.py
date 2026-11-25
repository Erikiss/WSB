SOURCE_TEXT = """SliceGPT shows that the residual stream of a transformer can be rotated by any orthogonal matrix without changing its outputs, once LayerNorm is converted to RMSNorm. Rotating only one layer's weights, however, breaks the model."""

TRANSLATE = """```wsl
model pythia-14m
# claim: a global residual rotation is a symmetry after converting LayerNorm (SliceGPT).
gauge fold_ln(); center_writers(); ln_to_rms(); rotate(resid, orth)
# claim: without converting LayerNorm, the same rotation is not a symmetry.
gauge rotate(resid, orth)
# claim: rotating only one layer breaks the model.
edit rotate(resid, orth, layers=[2], embed=false, unembed=false)
```"""

BROKEN_TRANSLATE = """```wsl
model pythia-14m
gauge rotate(resid, orth
```"""

CRITIC = """```json
[{"n": 1, "predict": "exact", "reason": "Computational invariance after LayerNorm is converted.", "fix": ""},
 {"n": 2, "predict": "edit", "reason": "LayerNorm gains and centring pin the basis.", "fix": "gauge fold_ln(); rotate(resid, orth1)"},
 {"n": 3, "predict": "edit", "reason": "The other layers' readers and writers are not compensated.", "fix": ""}]
```"""

ALTERNATIVES = """```wsl
model pythia-14m
# control: rotation restricted to matrices fixing the all-ones vector, LayerNorm kept.
gauge fold_ln(); rotate(resid, orth1)
```"""


def reply(role: str, messages: list[dict]) -> str:
    if (role == "translate" and len(messages) == 1
            and "[[mock-syntax-error]]" in messages[0]["content"]):
        return BROKEN_TRANSLATE
    return {"translate": TRANSLATE, "critic": CRITIC, "alternatives": ALTERNATIVES}[role]
