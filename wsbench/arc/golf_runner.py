"""Run a pinned Code Golf solution in a fresh Python process for each task."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


_RUNNER = r'''
import contextlib
import copy
import importlib.util
import json
import re
import sys
import numpy as np

# A pinned solution is imported afresh for each example. Keep its output off
# the JSON protocol channel and avoid writing pyc files beside the solution.
sys.dont_write_bytecode = True
examples = json.load(sys.stdin)
results = []
first_failure = True
for example in examples:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location("golf_solution", sys.argv[1])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not callable(getattr(module, "p", None)):
                raise TypeError("solution must define a callable p")
            raw = module.p(copy.deepcopy(example["input"]))
        encoded = json.dumps(raw).replace("true", "1").replace("false", "0")
        if not re.fullmatch(r"[0-9,\[\]\s\.]+", encoded):
            raise ValueError("invalid output characters")
        got = json.loads(encoded)
        ok = np.array_equal(np.array(got), np.array(example["output"]))
    except Exception as exc:
        ok = False
        got = f"error: {type(exc).__name__}: {exc}"
    record = {"ok": bool(ok)}
    if not ok and first_failure:
        record["got"] = got
        first_failure = False
    results.append(record)

json.dump({"results": results}, sys.stdout)
'''


def _failed(examples, message):
    return [{"ok": False, "got": message} for _ in examples]


def run_golf(solution_path, examples):
    solution = Path(solution_path).resolve()
    with tempfile.TemporaryDirectory(prefix="wsbench-golf-") as workdir:
        runner = Path(workdir, "runner.py")
        runner.write_text(_RUNNER, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runner), str(solution)],
                cwd=workdir,
                env={"PATH": os.environ.get("PATH", os.defpath), "PYTHONHASHSEED": "0"},
                input=json.dumps(examples),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=True,
            )
            results = json.loads(completed.stdout)["results"]
            if len(results) != len(examples):
                raise ValueError("unexpected number of results from golf runner")
            return results
        except subprocess.TimeoutExpired:
            return _failed(examples, "error: timeout")
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            return _failed(examples, f"error: {type(exc).__name__}: {exc}")
