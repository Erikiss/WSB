"""Pinned Code Golf data, task browsing, and evaluation of WSL solve statements."""
import csv
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np

from wsbench.paths import home
from .golf_runner import run_golf
from .grid import PRIMITIVES, CompiledNet, check, dense_weight_from_src


ARCHIVE = "https://codeload.github.com/SakanaAI/google-code-golf-2025/tar.gz/8ccc4b44b65d933527c205c304cb918883d12270"
TASKS_DIR = "google-code-golf-2025"
SPLITS = ("train", "test", "arc-gen")
FETCH_MESSAGE = "run: wsbench fetch arc"
_FETCH_LOCK = threading.Lock()


class ArcError(ValueError):
    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _home_dir():
    return home() / "golf"


def _task_dir():
    override = os.environ.get("WSB_GOLF_DIR")
    return Path(override).expanduser() if override is not None else _home_dir() / TASKS_DIR


def _task_files(directory):
    return sorted(directory.glob("task[0-9][0-9][0-9].json"))


def arc_local():
    return next(_task_dir().glob("task[0-9][0-9][0-9].json"), None) is not None


def _manifest():
    names = {"LICENSE", "data/id_map.csv"}
    for n in range(1, 401):
        task = f"task{n:03}"
        names.update((f"{TASKS_DIR}/{task}.json", f"compression/raw/{task}.py", f"solutions/{task}.py"))
    return names


def _unpack(archive, destination, names):
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ArcError("E_DOWNLOAD", "unsafe path in Code Golf archive")
            relative = "/".join(path.parts[1:])
            if relative not in names:
                continue
            if not member.isfile() or relative in seen:
                raise ArcError("E_DOWNLOAD", "invalid Code Golf archive member")
            seen.add(relative)
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as src, output.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    if seen != names:
        raise ArcError("E_DOWNLOAD", "incomplete Code Golf archive")


def fetch_arc():
    """Publish only verified archive contents. A failed download leaves prior data intact."""
    root = _home_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    names = _manifest()
    with _FETCH_LOCK:
        try:
            with tempfile.TemporaryDirectory(dir=root.parent, prefix=".golf-") as directory:
                temp = Path(directory)
                archive = temp / "archive.tar.gz"
                with urllib.request.urlopen(ARCHIVE, timeout=90) as response, archive.open("wb") as out:
                    shutil.copyfileobj(response, out)
                staged = temp / "golf"
                _unpack(archive, staged, names)
                _write_index(staged)
                if not root.exists():
                    try:
                        staged.rename(root)
                        return
                    except OSError:
                        # A concurrent fetch can publish the entire directory first.
                        if not root.is_dir():
                            raise
                for name in [*sorted(names), "index.json"]:
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (staged / name).replace(target)
        except (OSError, tarfile.TarError, ValueError) as exc:
            if isinstance(exc, ArcError):
                raise
            raise ArcError("E_DOWNLOAD", f"Code Golf download failed: {exc}") from exc


def _size(path):
    return path.stat().st_size if path.is_file() else None


def _solution_paths(root, task):
    return root / "compression" / "raw" / f"{task}.py", root / "solutions" / f"{task}.py"


def _id_map(root):
    if root is None or not (root / "data/id_map.csv").is_file():
        return {}
    try:
        with (root / "data/id_map.csv").open(encoding="utf-8", newline="") as mapping:
            return {row["task_id"]: row["original_id"] for row in csv.DictReader(mapping)}
    except (OSError, UnicodeError, KeyError, csv.Error) as exc:
        raise ArcError("E_DOWNLOAD", f"invalid id_map.csv. {FETCH_MESSAGE}") from exc


def _first_primitive(examples):
    candidates = list(PRIMITIVES.keys())[1:]
    for example in examples:
        grid = np.array(example["input"])
        expected = np.array(example["output"])
        candidates = [name for name in candidates if np.array_equal(PRIMITIVES[name](grid), expected)]
        if not candidates:
            return None
    return candidates[0] if candidates else None


def _examples_at(path):
    try:
        data = json.loads(path.read_bytes())
        for split in SPLITS:
            examples = data[split]
            if not isinstance(examples, list) or (split == "train" and not examples):
                raise ValueError("expected a list of examples and a nonempty train split")
            for example in examples:
                for key in ("input", "output"):
                    grid = example[key]
                    arr = np.asarray(grid)
                    if (arr.ndim != 2 or min(arr.shape) < 1
                            or arr.dtype.kind not in "iu" or arr.min() < 0 or arr.max() > 9):
                        raise ValueError("invalid ARC grid")
        return data
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        hint = "check WSB_GOLF_DIR" if "WSB_GOLF_DIR" in os.environ else FETCH_MESSAGE
        raise ArcError("E_DOWNLOAD", f"invalid ARC data in {path.name}: {exc}. {hint}") from exc


def _build_index(files, root=None):
    mapping = _id_map(root)
    rows = []
    for path in files:
        task = path.stem
        data = _examples_at(path)
        examples = [e for split in SPLITS for e in data[split]]
        raw, submitted = (None, None) if root is None else map(_size, _solution_paths(root, task))
        rows.append({
            "task": task, "original_id": mapping.get(task), "n_train": len(data["train"]),
            "n_test": len(data["test"]), "n_arcgen": len(data["arc-gen"]), "bytes_raw": raw,
            "bytes_submitted": submitted, "single_primitive": _first_primitive(examples),
        })
    return rows


def _write_index(root):
    rows = _build_index(_task_files(root / TASKS_DIR), root)
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".index-", delete=False) as output:
        json.dump(rows, output)
    Path(output.name).replace(root / "index.json")
    return rows


@lru_cache(maxsize=4)
def _override_index(signatures):
    return _build_index([path for path, _ in signatures])


def list_tasks():
    files = _task_files(_task_dir())
    if not files:
        raise ArcError("E_DOWNLOAD", FETCH_MESSAGE)
    if "WSB_GOLF_DIR" in os.environ:
        return _override_index(tuple((p, p.stat().st_mtime_ns) for p in files))
    root = _home_dir()
    cached = root / "index.json"
    if cached.is_file():
        try:
            index = json.loads(cached.read_bytes())
            if isinstance(index, list) and len(index) == len(files) and all(
                isinstance(row, dict) and row.get("task") == path.stem and "single_primitive" in row
                for row, path in zip(index, files)
            ):
                return index
        except (OSError, ValueError):
            pass  # The derived index can be regenerated from the task files.
    return _write_index(root)


def _read_task(task):
    if not arc_local():
        raise ArcError("E_DOWNLOAD", FETCH_MESSAGE)
    if not re.fullmatch(r"task[0-9]{3}", task):
        raise ArcError("E_NOT_FOUND", "unknown task", 404)
    path = _task_dir() / (task + ".json")
    if not path.is_file():
        raise ArcError("E_NOT_FOUND", "unknown task", 404)
    return _examples_at(path)


def task_detail(task):
    data = _read_task(task)
    root = None if "WSB_GOLF_DIR" in os.environ else _home_dir()
    source = raw_size = submitted_size = None
    if root is not None:
        raw, submitted = _solution_paths(root, task)
        raw_size, submitted_size = _size(raw), _size(submitted)
        if raw_size is not None:
            try:
                source = raw.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ArcError("E_DOWNLOAD", f"unreadable solution for {task}. {FETCH_MESSAGE}") from exc
    return {
        "task": task, "original_id": _id_map(root).get(task), "train": data["train"],
        "test": data["test"], "arc_gen_first20": data["arc-gen"][:20],
        "n_arcgen": len(data["arc-gen"]), "solution_raw": source,
        "bytes_raw": raw_size, "bytes_submitted": submitted_size,
    }


def _compile(data, prim, size, source):
    if prim not in PRIMITIVES or size not in ("fixed", "gated") or source not in ("all", "train"):
        raise ArcError("E_BAD_REQUEST", "invalid primitive, size, or source")
    examples = data["train"] if source == "train" else [e for split in SPLITS for e in data[split]]
    net = CompiledNet(prim, examples, size)
    sizes = [list(np.shape(examples[0]["input"]))] if size == "fixed" else list(map(list, net.table))
    return net, sizes


def compiled_preview(task, prim, size="fixed", source="all", shape=None):
    net, sizes = _compile(_read_task(task), prim, size, source)
    if size == "fixed":
        selected, src = sizes[0], net.table["*"]
    else:
        selected = sizes[0]
        if shape is not None:
            match = re.fullmatch(r"([0-9]{1,2})x([0-9]{1,2})", shape)
            if not match:
                raise ArcError("E_BAD_REQUEST", "unknown input shape")
            selected = list(map(int, match.groups()))
        if tuple(selected) not in net.table:
            raise ArcError("E_BAD_REQUEST", "unknown input shape")
        src = net.table[tuple(selected)]
    preview = dense_weight_from_src(src).reshape(300, 3, 300, 3).mean(axis=(1, 3))
    return {"sizes": sizes, "shape": selected, "src_grid": src.reshape(30, 30).tolist(),
            "P_preview": preview.tolist()}


def _golf_metrics(task, data, metrics):
    raw, submitted = _solution_paths(_home_dir(), task)
    if "WSB_GOLF_DIR" in os.environ or not raw.is_file():
        raise ArcError("E_DOWNLOAD", FETCH_MESSAGE)
    metrics.update(bytes_raw=_size(raw), bytes_submitted=_size(submitted))
    examples = [e for split in SPLITS for e in data[split]]
    results = iter(run_golf(raw, examples))
    for split in SPLITS:
        passed = 0
        for idx, example in enumerate(data[split]):
            result = next(results)
            if result["ok"]:
                passed += 1
            elif metrics["first_failure"] is None:
                metrics["first_failure"] = {"split": split, "index": idx, "input": example["input"],
                                            "expected": example["output"], "got": result["got"]}
        metrics["by_split"][split] = [passed, len(data[split])]


def execute_grid(stmt):
    task = stmt.model.split(":", 1)[1]
    try:
        data = _read_task(task)
    except ArcError as exc:
        if exc.code == "E_NOT_FOUND":
            raise ArcError("E_DOWNLOAD", FETCH_MESSAGE) from exc
        raise
    op = stmt.ops[0]
    metrics = {"by_split": {}, "first_failure": None}
    if op.name == "golf":
        _golf_metrics(task, data, metrics)
    else:
        if op.name == "compile":
            fn, sizes = _compile(data, op.args[0], op.kwargs.get("size", "fixed"), op.kwargs.get("from", "all"))
            metrics["compiled"] = {"sizes": sizes, "params_dense": fn.n_params_dense(),
                                   "nnz": sum(int((src >= 0).sum()) for src in fn.table.values())}
        else:
            names = [op.args[0]] if op.name == "prim" else [p.args[0] for p in op.args]
            def fn(grid):
                result = np.array(grid)
                for name in names:
                    result = PRIMITIVES[name](result)
                return result.tolist()

        for split in SPLITS:
            passed, total, failure = check(fn, data[split])
            metrics["by_split"][split] = [passed, total]
            if failure is not None and metrics["first_failure"] is None:
                metrics["first_failure"] = {"split": split, **failure}
    metrics["passed"] = sum(pair[0] for pair in metrics["by_split"].values())
    metrics["total"] = sum(pair[1] for pair in metrics["by_split"].values())
    return metrics
