"""Modular-addition experiments and persistent named checkpoint runs."""

import json
import math
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn.functional import cosine_similarity

from wsbench.lang.parser import expand_range
from wsbench.models import modadd
from wsbench.paths import home


class RunError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


# The reference trainer sets PyTorch's process-wide seed and thread count.
_training = threading.Lock()
_disk = threading.RLock()


@contextmanager
def _store_lock(root):
    """Keep all checkpoint reads in a path on the same generation of each run."""
    with _disk:
        if os.name == "posix":
            import fcntl

            descriptor = os.open(root, os.O_RDONLY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                os.close(descriptor)
        else:
            yield


def _run_directory(root, name):
    target = root / name
    # Recover the old generation if the process stopped between renames.
    backup = root / f".previous-{name}"
    if backup.exists():
        if target.exists():
            shutil.rmtree(backup)
        else:
            backup.rename(target)
    return target


def _publish(name, summary, arguments, checkpoints, should_stop):
    root = home() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".training-", dir=root) as temp:
        staged = Path(temp) / "completed"
        staged.mkdir()
        for step, weights in checkpoints.items():
            torch.save(weights, staged / f"step{step}.pt")
        (staged / "run.json").write_text(
            json.dumps({"args": arguments, **summary}, allow_nan=False), encoding="utf-8"
        )
        with _store_lock(root):
            if should_stop and should_stop():
                raise RunError("E_RUNTIME", "cancelled")
            dest = _run_directory(root, name)
            overwrite = dest.exists()
            backup = root / f".previous-{name}"
            if overwrite:
                dest.rename(backup)
            try:
                staged.rename(dest)
            except OSError:
                if overwrite:
                    backup.rename(dest)
                raise
            if overwrite:
                shutil.rmtree(backup)

    return [{"code": "W_OVERWRITE", "message": f"Overwrote run {name}."}] if overwrite else []


def _train(op, on_curves, should_stop):
    args = {"p": 97, "frac": 0.5, "steps": 3000, "seed": 0, "lr": 0.001,
            "wd": 1.0, "filter": "none", "save": [], "intervene": [], **op.kwargs}
    filt = args["filter"]
    if filt == "none":
        filt = None
    else:
        filt = ("grokfast", *map(float, filt.args))
    interventions = [(at.args[0], "scale", float(at.args[1].args[0])) for at in args["intervene"]]
    name = args["run"]
    saves = sorted({0, args["steps"], *args["save"]})
    args.update(filter=filt, intervene=interventions)
    progress = {}

    def on_eval(evaluation):
        _finite(evaluation)
        if on_curves:
            for key, value in evaluation.items():
                progress.setdefault(key, []).append(value)
            on_curves({key: values.copy() for key, values in progress.items()})

    if should_stop and should_stop():
        raise RunError("E_RUNTIME", "cancelled")
    start = time.monotonic()
    result = modadd.train(
        p=args["p"], frac=float(args["frac"]), steps=args["steps"], seed=args["seed"],
        lr=float(args["lr"]), wd=float(args["wd"]), filt=filt, interventions=interventions,
        save_steps=saves, eval_every=50, threads=max(1, (os.cpu_count() or 1) - 1),
        on_eval=on_eval, should_stop=should_stop,
    )
    if should_stop and should_stop():
        raise RunError("E_RUNTIME", "cancelled")
    summary = {"run": name, "curves": result["curves"], "first": result["first"],
               "saved_steps": sorted(result["checkpoints"]), "wall_s": time.monotonic() - start}
    warnings = _publish(name, summary, args, result["checkpoints"], should_stop)
    return summary, warnings


def _operands(op):
    root = home() / "runs"
    metadata = {}
    cache = {}

    def read(expr):
        if expr.name == "diff":
            left, right = (read(arg) for arg in expr.args)
            _check_shapes(left, right)
            return {key: left[key] - right[key] for key in left}
        name, step = expr.args
        if (name, step) in cache:
            return cache[name, step]
        directory = _run_directory(root, name)
        if name not in metadata:
            try:
                metadata[name] = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise RunError("E_MISSING_RUN", f"unknown run {name}") from None
        if step not in metadata[name]["saved_steps"]:
            raise RunError("E_MISSING_RUN", f"run {name} has no saved step {step}")
        try:
            weights = torch.load(directory / f"step{step}.pt", map_location="cpu", weights_only=True)
        except FileNotFoundError:
            raise RunError("E_MISSING_RUN", f"run {name} has no saved step {step}") from None
        cache[name, step] = weights
        return weights

    if not root.exists():
        raise RunError("E_MISSING_RUN", "no saved runs")
    with _store_lock(root):
        operands = [read(expr) for expr in op.args]
    if len(operands) == 2:
        _check_shapes(*operands)
    first = next(iter(metadata.values()))["args"]
    return operands, first["p"], first["frac"]


def _check_shapes(a, b):
    if a.keys() != b.keys() or any(a[key].shape != b[key].shape for key in a):
        raise RunError("E_SHAPE", "checkpoint shapes differ")


def _finite(metrics):
    for name, value in metrics.items():
        if not math.isfinite(value):
            raise RunError("E_RUNTIME", f"non-finite {name} during evaluation")


def _path(op):
    operands, p, frac = _operands(op)
    a = operands[0]
    if op.name == "cosine":
        va, vb = (torch.cat([weights[key].flatten() for key in sorted(weights)]) for weights in operands)
        return {"cosine": cosine_similarity(va, vb, dim=0).clamp(-1, 1).item(),
                "norm_a": va.norm().item(), "norm_b": vb.norm().item()}

    axis = "c" if op.name == "scale" else "t"
    points = []
    for value in expand_range(*op.kwargs[axis]):
        if op.name == "scale":
            weights = {key: value * a[key] for key in a}
        elif op.name == "lerp":
            weights = {key: (1 - value) * a[key] + value * operands[1][key] for key in a}
        else:
            weights = {key: a[key] + value * operands[1][key] for key in a}
        metrics = modadd.eval_state(weights, p, frac)
        _finite(metrics)
        points.append({axis: value, **metrics})
    return {"points": points}


def execute_modadd(stmt, on_curves=None, should_stop=None):
    """Execute one validated statement; return its metrics and warning dictionaries."""
    with _training:
        if stmt.claim == "train":
            previous_threads = torch.get_num_threads()
            try:
                return _train(stmt.ops[0], on_curves, should_stop)
            finally:
                torch.set_num_threads(previous_threads)
        return _path(stmt.ops[0]), []
