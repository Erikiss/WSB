"""Explore loop: a planner model turns a researcher's idea into a series of measured assist rounds.

Each round: the planner writes one testable question, the assist pipeline translates, predicts and measures it,
and the planner sees a digest of the measured verdicts before it writes the next question. The loop stops when
the planner says it is done, after the requested number of rounds, when the budget is spent, or on cancel.
"""
import json
import time

from wsbench.lang.card import LANGUAGE_CARD
from wsbench.models.registry import SHAPES

from . import openrouter
from .pipeline import AssistRequestError, _block, assist
from .prompts import EXPLORE_PLAN_SYSTEM, EXPLORE_REPORT_SYSTEM

MAX_ROUNDS = 8
MAX_BUDGET_USD = 5.0


def parameters(request):
    text, model = request.get("text"), request.get("model")
    if not isinstance(text, str) or not text.strip() or not isinstance(model, str):
        raise AssistRequestError("text and model must be non-empty strings")
    if SHAPES.get(model, {}).get("family") not in {"neox", "grid"}:
        raise AssistRequestError("model must be a NeoX model or grid:task001 through grid:task400")
    rounds = request.get("rounds", 3)
    if type(rounds) is not int or not 1 <= rounds <= MAX_ROUNDS:
        raise AssistRequestError(f"rounds must be an integer from 1 to {MAX_ROUNDS}")
    budget = request.get("budget_usd", 1.0)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= MAX_BUDGET_USD:
        raise AssistRequestError(f"budget_usd must be a number above 0 and at most {MAX_BUDGET_USD}")
    roles = request.get("roles", {})
    if not isinstance(roles, dict) or any(
        k not in openrouter.DEFAULT_ROLES or not isinstance(v, str) or not v.strip() for k, v in roles.items()
    ):
        raise AssistRequestError("roles must map assist roles to model IDs")
    roles = {**openrouter.DEFAULT_ROLES, **roles}
    planner = request.get("planner", roles["translate"])
    if not isinstance(planner, str) or not planner.strip():
        raise AssistRequestError("planner must be a model ID")
    return text, model, rounds, float(budget), roles, planner


def _number(value):
    return None if value is None else float(f"{value:.4g}")


def digest(verdicts):
    """What the planner sees of a measured script: statements and numbers, no model prose."""
    rows = []
    for v in verdicts:
        m = v.get("metrics") or {}
        row = {"label": v["label"], "statement": v["source"], "status": v["status"]}
        if v.get("measured") is not None:
            row["measured"] = v["measured"]
        for key in ("dloss", "rel"):
            if m.get(key) is not None:
                row[key] = _number(m[key])
        if (m.get("control") or {}).get("ratio") is not None:
            row["control_ratio"] = _number(m["control"]["ratio"])
        if m.get("total") is not None:
            row["passed"] = f"{m['passed']}/{m['total']}"
        if v.get("predicted") is not None:
            row["predicted"], row["agree"] = v["predicted"], v["agree"]
        if v.get("error"):
            row["error"] = v["error"]["code"] + ": " + v["error"]["message"]
        rows.append(row)
    return rows


def _history_text(idea, rounds, answers):
    parts = [f"Researcher's idea:\n<<<\n{idea}\n>>>"]
    for r in rounds:
        parts.append(f"Round {r['n']} question:\n{r['question']}")
        if r.get("verdicts") is not None:
            parts.append(f"Round {r['n']} measured results (JSON):\n" + json.dumps(digest(r["verdicts"]), ensure_ascii=False))
        else:
            parts.append(f"Round {r['n']} produced no measurement: {r.get('problem') or 'nothing testable'}")
    for a in answers:
        parts.append(f"You asked the researcher: {a['question']}\nThe researcher answered: {a['answer']}")
    return "\n\n".join(parts)


def _plan(text):
    block = _block(text, "json")
    if block is None:
        raise ValueError("the planner reply has no json block")
    plan = json.loads(block)
    if not isinstance(plan, dict):
        raise ValueError("the planner reply is not a JSON object")
    out = {key: plan.get(key, "") for key in ("question", "note", "ask_user")}
    if not all(isinstance(v, str) for v in out.values()):
        raise ValueError("question, note and ask_user must be strings")
    out["done"] = plan.get("done") is True
    return out


def explore(request, submit, snapshot, cancel_run, *, on_update=lambda state: None, should_stop=lambda: False,
            ask=lambda question: None, sleep=time.sleep):
    """Run the loop synchronously and return the final state. `submit`, `snapshot` and `cancel_run` are the job
    queue's functions. `ask(question)` blocks until the researcher answers and returns the answer (or None)."""
    idea, model, max_rounds, budget, roles, planner = parameters(request)
    planners = list(dict.fromkeys([planner, roles["critic"]]))     # second entry: fallback if the first refuses
    state = dict(idea=idea, model=model, max_rounds=max_rounds, budget_usd=budget, cost_usd=0.0, rounds=[],
                 answers=[], report=None, stopped=None, planner_calls=[])

    def call(role, system, user):
        system = system.replace("{LANGUAGE_CARD}", LANGUAGE_CARD).replace("{MODEL}", model)
        reply = None
        for model_id in planners:
            reply = openrouter.chat(role, model_id, system, [{"role": "user", "content": user}])
            state["cost_usd"] += reply["cost_usd"]
            state["planner_calls"].append({k: reply[k] for k in ("model", "ok", "error", "cost_usd")}
                                          | {"role": role, "provider": reply.get("provider")})
            if reply["ok"]:
                return reply
        return reply

    def stop(reason):
        state["stopped"] = reason
        on_update(state)

    asked = 0
    round_cost = 0.0           # cost of the most expensive round so far (planner call included)
    while len(state["rounds"]) < max_rounds:
        if should_stop():
            stop("cancelled")
            break
        if state["cost_usd"] + round_cost > budget:     # the next round would probably exceed the budget
            stop("budget")
            break
        n = len(state["rounds"]) + 1
        cost_before = state["cost_usd"]
        reply = call("planner", EXPLORE_PLAN_SYSTEM,
                     _history_text(idea, state["rounds"], state["answers"])
                     + f"\n\nThis is round {n} of at most {max_rounds}. Write the next question.")
        if not reply["ok"]:
            stop("planner failed: " + str(reply["error"]))
            break
        try:
            plan = _plan(reply["text"])
        except (ValueError, TypeError) as exc:
            stop(f"planner reply not usable: {exc}")
            break
        if plan["ask_user"].strip() and asked < 2:
            asked += 1
            state["waiting"] = plan["ask_user"].strip()
            on_update(state)
            answer = ask(state["waiting"])
            question = state.pop("waiting")
            if answer is None:
                stop("cancelled")
                break
            state["answers"].append({"question": question, "answer": answer})
            on_update(state)
            continue
        if plan["done"] and state["rounds"]:
            stop("planner finished")
            break
        if not plan["question"].strip():
            stop("planner gave no question")
            break
        entry = dict(n=n, question=plan["question"].strip(), note=plan["note"].strip(), status="calling models",
                     assist=None, run_id=None, verdicts=None, problem=None, cost_usd=0.0)
        state["rounds"].append(entry)
        on_update(state)

        result = assist({"text": entry["question"], "model": model, "roles": roles}, submit)
        entry.update(assist=result, run_id=result["run_id"], cost_usd=result["cost_usd"])
        state["cost_usd"] += result["cost_usd"]
        if result["run_id"] is None:
            messages = [p["message"] for p in result["parse_errors"]] or ["no statements"]
            entry.update(status="no measurement", problem="; ".join(messages))
            round_cost = max(round_cost, state["cost_usd"] - cost_before)
            on_update(state)
            continue
        entry["status"] = "measuring"
        on_update(state)
        while True:
            run = snapshot(result["run_id"])
            entry["verdicts"] = run["verdicts"]
            if run["status"] in ("done", "cancelled"):
                break
            if should_stop():
                cancel_run(result["run_id"])
            on_update(state)
            sleep(0.3)
        entry["status"] = "done"
        round_cost = max(round_cost, state["cost_usd"] - cost_before)
        on_update(state)
    else:
        state["stopped"] = "all rounds used"

    if any(r["verdicts"] for r in state["rounds"]) and state["stopped"] != "cancelled":
        reply = call("report", EXPLORE_REPORT_SYSTEM, _history_text(idea, state["rounds"], state["answers"])
                     + "\n\nWrite the final report.")
        state["report"] = reply["text"] if reply["ok"] else None
        if not reply["ok"]:
            state["report_error"] = str(reply["error"])
    on_update(state)
    return state
