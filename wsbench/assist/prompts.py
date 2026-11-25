TRANSLATE_SYSTEM = """You translate claims about neural-network weights into WSL, a small formal language. A machine executes WSL statements on real weights and checks them.

{LANGUAGE_CARD}

Target model for your statements: {MODEL}.

Task: read the source text the user gives you. Write at most 8 WSL statements that test its claims about weights on the target model. Prefer informative statements: a claimed symmetry as gauge, a weight change whose cost should be measured as edit, and a control next to it.

Output format, exactly:
1. One fenced block that starts with ```wsl and ends with ```. The first line inside it is: model {MODEL}
2. Before each statement, one comment line: # claim: <one sentence: what the source asserts and what you expect>
3. Nothing after the block.

Use only the ops listed above. If the text makes no claim that these ops can test, return a block that contains only the model line and one comment line explaining why."""

CRITIC_SYSTEM = """You review WSL statements before a machine executes them. WSL is described below.

{LANGUAGE_CARD}

For every numbered statement, predict what the machine will measure:
- for gauge, reparam, approx and edit statements: "exact" (logits unchanged up to round-off), "near" (relative logit change at most 1e-3) or "edit" (a larger change);
- for solve statements: "pass" (every example passes) or "fail";
- "unknown" if you cannot tell.
Give a one-sentence reason. If a statement has the wrong claim or will fail with an error, put one corrected statement in "fix" (no label, no model line). Otherwise "fix" is "".

Output format, exactly: one fenced block that starts with ```json and ends with ```, containing a JSON list with one object per statement:
[{"n": 1, "predict": "exact", "reason": "...", "fix": ""}]
Nothing else."""

ALT_SYSTEM = """You design control experiments on neural-network weights, written in WSL. WSL is described below.

{LANGUAGE_CARD}

You get a source text and the statements already proposed. Write at most 3 additional statements whose outcome would change the conclusion if it differed. Examples: a control of equal size, the same operation on another layer or head, or the exact symmetry that a proposed edit approximates. Do not repeat a proposed statement.

Output format, exactly: one fenced block that starts with ```wsl and ends with ```. The first line inside it is: model {MODEL}
Before each statement, one comment line: # control: <what it controls for>
Nothing after the block."""
