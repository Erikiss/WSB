import io
from pathlib import Path
import tarfile

import numpy as np
import pytest

from wsbench.arc.data import ArcError, _unpack, compiled_preview, fetch_arc, list_tasks, task_detail
from wsbench.arc.golf_runner import run_golf
from wsbench.arc.grid import decode, onehot
from wsbench.arc.render import tensor_view


FIXTURES = Path(__file__).parent / "fixtures" / "golf_mini"
# task, primitive, total, fixed passes, raw bytes (Appendix F.4)
PRIMITIVE_TASKS = [
    ("task087", "rot180", 266, 266, 36),
    ("task140", "rot180", 265, 265, 36),
    ("task150", "vmirror", 266, 39, 30),
    ("task155", "hmirror", 266, 51, 18),
    ("task179", "dmirror", 267, 267, 21),
    ("task241", "dmirror", 266, 32, 21),
    ("task380", "rot270", 267, 267, 27),
]


def solve(task, expression):
    from wsbench.exec.executor import run_script

    verdict, = run_script(f"model grid:{task}\nsolve {expression}")
    assert verdict["error"] is None, verdict
    return verdict


@pytest.mark.parametrize(("task", "expr", "passes", "total"), [
    ("task087", "compile(rot180, size=fixed)", 8, 8),
    ("task150", "compile(vmirror, size=fixed)", 1, 7),
    ("task150", "compile(vmirror, size=gated)", 7, 7),
    ("task150", "prim(vmirror)", 7, 7),
    ("task087", "chain(prim(vmirror), prim(hmirror))", 8, 8),
])
def test_mini(monkeypatch, task, expr, passes, total):
    monkeypatch.setenv("WSB_GOLF_DIR", str(FIXTURES))
    verdict = solve(task, expr)
    assert verdict["status"] == ("CONFIRMED" if passes == total else "REFUTED")
    m = verdict["metrics"]
    assert (m["passed"], m["total"]) == (passes, total)
    assert sum(pair[0] for pair in m["by_split"].values()) == passes
    if passes == total:
        assert m["first_failure"] is None
    else:
        assert (m["first_failure"]["split"], m["first_failure"]["index"]) == ("train", 1)


def test_onehot_roundtrip():
    rng = np.random.default_rng(416)
    for _ in range(200):
        shape = rng.integers(1, 31, 2)
        grid = rng.integers(0, 10, size=shape).tolist()
        assert decode(onehot(grid)) == grid
    assert decode(np.zeros((10, 30, 30))) is None
    assert decode(onehot([[0]])) == [[0]]


def test_mini_index_and_preview(monkeypatch, tmp_path):
    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    monkeypatch.setenv("WSB_GOLF_DIR", str(FIXTURES))
    index = list_tasks()
    assert [(r["task"], r["single_primitive"]) for r in index] == [
        ("task087", "rot180"), ("task150", "vmirror")]
    assert all(r["bytes_raw"] is r["bytes_submitted"] is r["original_id"] is None for r in index)
    assert task_detail("task150")["solution_raw"] is None
    assert not (tmp_path / "golf" / "index.json").exists()
    preview = compiled_preview("task150", "vmirror", "gated", "train", "4x4")
    assert preview["sizes"] == [[4, 4], [7, 7], [6, 6]]
    assert preview["src_grid"][0][:5] == [3, 2, 1, 0, -1]
    assert np.shape(preview["P_preview"]) == (300, 300)
    assert np.sum(preview["P_preview"]) == pytest.approx(16 / 9)
    with pytest.raises(ArcError, match="unknown input shape"):
        compiled_preview("task150", "vmirror", "gated", "train", "5x5")


def test_unreadable_task_and_derived_index_recovery(monkeypatch, tmp_path):
    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    monkeypatch.setenv("WSB_GOLF_DIR", str(tmp_path))
    task = tmp_path / "task150.json"
    task.write_text("{")
    for operation in (list_tasks, lambda: task_detail("task150"),
                      lambda: compiled_preview("task150", "vmirror")):
        with pytest.raises(ArcError, match="task150.json") as exc:
            operation()
        assert exc.value.code == "E_DOWNLOAD"
    monkeypatch.delenv("WSB_GOLF_DIR")
    root = tmp_path / "golf"
    tasks = root / "google-code-golf-2025"
    tasks.mkdir(parents=True)
    (tasks / task.name).write_bytes((FIXTURES / task.name).read_bytes())
    index_file = root / "index.json"
    index_file.write_text("{")
    assert list_tasks()[0]["single_primitive"] == "vmirror"
    assert index_file.read_text().startswith("[")


def test_subprocess_golf(tmp_path):
    solution = tmp_path / "golf.py"
    solution.write_text("def p(g):\n    g[0][0] = True\n    return g\n")
    examples = [{"input": [[0]], "output": [[v]]} for v in (1, 2, 3)]
    results = run_golf(solution, examples)
    assert results == [{"ok": True}, {"ok": False, "got": [[1]]}, {"ok": False}]
    assert all(e["input"] == [[0]] for e in examples)
    solution.write_text("p = lambda g: 1 / 0")
    assert run_golf(solution, examples)[0]["got"] == "error: ZeroDivisionError: division by zero"


def test_archive_rejects_unsafe_and_incomplete_members(tmp_path):
    archive = tmp_path / "data.tar.gz"

    def write_archive(path):
        with tarfile.open(archive, "w:gz") as stream:
            info = tarfile.TarInfo(path)
            info.size = 1
            stream.addfile(info, io.BytesIO(b"x"))

    for path in ("/root/file", "root/../file"):
        write_archive(path)
        with pytest.raises(ArcError, match="unsafe path"):
            _unpack(archive, tmp_path / "stage", {"LICENSE"})
    write_archive("repository/ignored.txt")
    with pytest.raises(ArcError, match="incomplete"):
        _unpack(archive, tmp_path / "stage", {"LICENSE"})
    assert not (tmp_path / "stage").exists()


def test_tensor_views():
    first = np.arange(15, dtype=float).reshape(3, 5)
    last = first + 2
    view = tensor_view("w", first, last, "after", 2)
    assert view == {"name": "w", "shape": [3, 5], "shown": [2, 2], "block": [2, 3], "scale": 14,
                    "data": [[5.5, 8.0], [13.0, 15.5]]}
    assert tensor_view("w", first, last, "diff", 2)["data"] == [[2, 2], [2, 2]]
    assert tensor_view("b", None, np.array([1., 3., 7.]), "before", 2)["scale"] == 7
    flat = tensor_view("b", None, np.zeros(3), "after", 2)
    assert flat["shape"] == [3] and flat["shown"] == [1, 2] and flat["scale"] == 1.
    arc = tensor_view("w", first, last, "diff", view="arc", r0=0, r1=2, c0=0, c1=3)
    assert arc == {"quantiles": [2.] * 9, "grid": [[0] * 3, [0] * 3]}


@pytest.fixture(scope="module")
def dataset():
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("WSB_GOLF_DIR", raising=False)
        fetch_arc()
        yield list_tasks()


@pytest.mark.network
def test_full_dataset(dataset):
    assert len(dataset) == 400
    indexed = [(r["task"], r["single_primitive"], r["bytes_raw"]) for r in dataset if r["single_primitive"]]
    assert indexed == [(task, prim, size) for task, prim, _, _, size in PRIMITIVE_TASKS]
    assert all(r["bytes_raw"] and r["bytes_submitted"] and r["original_id"] for r in dataset)
    for task, prim, total, fixed, _ in PRIMITIVE_TASKS:
        for expr, expected in [(f"prim({prim})", total), (f"compile({prim}, size=fixed)", fixed),
                               (f"compile({prim}, size=gated)", total)]:
            m = solve(task, expr)["metrics"]
            assert (m["passed"], m["total"]) == (expected, total)
            if task == "task150" and "fixed" in expr:
                assert m["by_split"] == {"train": [1, 3], "test": [0, 1], "arc-gen": [38, 262]}
                assert m["compiled"]["nnz"] == 16
            if task == "task150" and "gated" in expr:
                assert len(m["compiled"]["sizes"]) == 7 and m["compiled"]["nnz"] == 280
    assert solve("task150", "compile(vmirror, size=gated, from=train)")["metrics"]["passed"] == 112
    assert solve("task150", "prim(hmirror)")["metrics"]["passed"] == 0


@pytest.mark.network
def test_all_golf_solutions(dataset):
    failed = {}
    for row in dataset:
        verdict = solve(row["task"], "golf()")
        if verdict["status"] != "CONFIRMED":
            failure = verdict["metrics"]["first_failure"]
            assert failure is not None and failure["got"] is not None, verdict
            failed[row["task"]] = failure["got"]
    print(f"Golf solutions confirmed: {len(dataset) - len(failed)} / {len(dataset)}; failures: {failed}")
    assert len(dataset) - len(failed) >= 390, failed
