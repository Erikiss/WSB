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


PLAN_FIRST = """```json
{"question": "A global rotation of the residual stream is a symmetry once LayerNorm is converted to RMSNorm. Without the conversion it is not, and rotating one layer alone breaks the model.", "note": "Direct test of the rotation idea.", "ask_user": "", "done": false}
```"""

PLAN_DONE = """```json
{"question": "", "note": "The first round answered the idea.", "ask_user": "", "done": true}
```"""

PLAN_ASK = """```json
{"question": "", "note": "", "ask_user": "Rotation or permutation first?", "done": false}
```"""

REPORT = "Confirmed: the global rotation after the LayerNorm conversion. Refuted: the rotation without it. Open: larger models."


def reply(role: str, messages: list[dict]) -> str:
    if (role == "translate" and len(messages) == 1
            and "[[mock-syntax-error]]" in messages[0]["content"]):
        return BROKEN_TRANSLATE
    if role == "planner":
        content = messages[0]["content"]
        if "[[mock-ask]]" in content and "The researcher answered:" not in content:
            return PLAN_ASK
        return PLAN_DONE if "Round 1 question:" in content else PLAN_FIRST
    if role == "report":
        return REPORT
    return {"translate": TRANSLATE, "critic": CRITIC, "alternatives": ALTERNATIVES}[role]
