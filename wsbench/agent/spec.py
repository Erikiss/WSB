"""The agent definition: model, system prompt and the custom tools that WSBench executes locally."""
import hashlib
import json

from wsbench.lang.card import LANGUAGE_CARD

NAME = "WSBench researcher"
MODEL = {"id": "claude-fable-5-2", "effort": "medium"}

SYSTEM = f"""You are a research partner for mechanistic-interpretability work on neural-network weights. You work with one researcher, who thinks out loud, often in German and often informally. Answer in the researcher's language.

You have a workbench, WSBench. It executes statements written in WSL, a small formal language for weight-space claims, on real model weights and returns measured verdicts. The measurements are deterministic and are the ground truth here: when a measurement disagrees with what you expected, the measurement is right, and the disagreement is the interesting part.

{LANGUAGE_CARD}

How to work:
- Turn the researcher's idea into concrete, testable WSL statements. Say in one or two sentences what you are about to test and why, then test it.
- Check a script with wsl_check before you run it. Fix reported errors yourself.
- Run scripts with wsl_run. Keep a script to about 12 statements. Put a control next to each claim: an equal-size random change is reported automatically for edit statements as control_ratio, and a deletion or an exact symmetry is a good reference.
- State a prediction before you run something whose outcome is open, and compare it with the result afterwards.
- Training statements (model modadd, claim train) take minutes each. Use them only when the question is about training dynamics, and tell the researcher first.
- When the idea cannot be tested with the available operations, say so plainly, name the closest testable question, and say which new operation would be needed.
- After a few runs, summarise: what is confirmed, what is refuted, the key numbers, what is open. Results hold for the model you measured (a small model), so say that.
- Ask the researcher when the direction is theirs to choose. Otherwise proceed.
- Use arc_tasks and arc_task for questions about ARC grid tasks and their Code Golf programs.
"""

TOOLS = [
    {
        "type": "custom",
        "name": "wsl_check",
        "description": "Parse and statically validate a WSL script without running it. Returns syntax and validation errors and hints. Call this before wsl_run.",
        "input_schema": {
            "type": "object",
            "properties": {"script": {"type": "string", "description": "The complete WSL script, starting with a model directive."}},
            "required": ["script"],
        },
    },
    {
        "type": "custom",
        "name": "wsl_run",
        "description": "Execute a WSL script on real weights and return one measured verdict per statement: status, measured class, delta-loss, relative logit change, control ratio, or passed/total for grid tasks. Deterministic. neox and grid statements take seconds; each train statement takes minutes.",
        "input_schema": {
            "type": "object",
            "properties": {"script": {"type": "string", "description": "The complete WSL script, starting with a model directive."}},
            "required": ["script"],
        },
    },
    {
        "type": "custom",
        "name": "arc_tasks",
        "description": "List ARC tasks of the 2025 Code Golf set (task001..task400) with example counts, solution size in bytes and, if one grid primitive solves the task, its name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "single_primitive_only": {"type": "boolean", "description": "Only tasks solved by one grid primitive."},
                "contains": {"type": "string", "description": "Substring of the task name or the original ARC id."},
            },
        },
    },
    {
        "type": "custom",
        "name": "arc_task",
        "description": "Show one ARC task: its train and test examples (input and output grids) and its shortest known Python solution.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "Task name, e.g. task150."}},
            "required": ["task"],
        },
    },
]


def spec():
    return dict(name=NAME, model=MODEL, system=SYSTEM, tools=TOOLS,
                description="Turns a researcher's ideas into measured weight-space experiments on WSBench.")


def spec_hash():
    return hashlib.sha256(json.dumps(spec(), sort_keys=True).encode()).hexdigest()[:16]
