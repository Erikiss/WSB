"""Synchronous, line-ordered WSL execution."""
import json
from copy import deepcopy
from time import perf_counter

from .verdict import agreement, failed, prepare


def _check_metrics(metrics):
    # A numeric overflow belongs to its statement, before any JSON API update.
    try:
        json.dumps(metrics, allow_nan=False)
    except ValueError as exc:
        raise ArithmeticError("non-finite measurement") from exc


def _measure_neox(stmt, verdict, on_state):
    from .resolve import execute_neox
    metrics, weights, kind = execute_neox(stmt)
    _check_metrics(metrics)
    rel = metrics["rel"]
    measured = "exact" if rel <= 1e-9 else "near" if rel <= 1e-3 else "edit"
    verdict.update(metrics=metrics, measured=measured)
    if stmt.claim == "edit":
        verdict["status"] = "MEASURED"
        if measured == "exact":
            verdict["warnings"].append({"code": "W_NOOP", "message": "this edit is a symmetry: it does not change the function"})
    else:
        confirmed = rel <= stmt.tol if stmt.claim == "approx" else measured == "exact"
        verdict["status"] = "CONFIRMED" if confirmed else "REFUTED"
    if on_state is not None:
        on_state(stmt.display_label, stmt.line, weights, kind)


def run_script(script: str, on_update=None, should_stop=None, on_state=None, predictions=None) -> list[dict]:
    statements, verdicts = prepare(script, predictions)
    def publish():
        if on_update is not None:
            on_update(deepcopy(verdicts))
    publish()
    for verdict in verdicts:
        if verdict["status"] == "ERROR":
            continue
        if should_stop is not None and should_stop():
            failed(verdict, "E_RUNTIME", "cancelled")
            agreement(verdict, predictions)
            publish()
            continue
        stmt = statements[verdict["line"]]
        verdict["status"] = "RUNNING"
        start = perf_counter()
        publish()
        try:
            if stmt.family == "neox":
                _measure_neox(stmt, verdict, on_state)
            elif stmt.family == "grid":
                from wsbench.arc.data import execute_grid
                metrics = execute_grid(stmt)
                status = "CONFIRMED" if metrics["passed"] == metrics["total"] else "REFUTED"
                verdict.update(metrics=metrics, status=status)
            else:
                from .runs import execute_modadd
                def on_curves(curves):
                    _check_metrics(curves)
                    verdict["metrics"]["curves"] = curves
                    publish()
                metrics, warnings = execute_modadd(stmt, on_curves, should_stop)
                _check_metrics(metrics)
                verdict.update(metrics=metrics, warnings=warnings, status="MEASURED")
        except Exception as exc:
            failed(verdict, getattr(exc, "code", "E_RUNTIME"), str(exc))
        verdict["elapsed_ms"] = round(1000 * (perf_counter() - start))
        agreement(verdict, predictions)
        publish()
    return verdicts
