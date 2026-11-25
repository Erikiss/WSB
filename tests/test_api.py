"""Offline HTTP acceptance tests, including the asynchronous run lifecycle."""
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from uuid import UUID

import numpy as np
import pytest
from fastapi.testclient import TestClient

from wsbench.lang.card import LANGUAGE_CARD
from wsbench.server.app import app


FIXTURES = Path(__file__).parent / "fixtures" / "golf_mini"
VERDICT_FIELDS = set("line label source model claim tol status measured metrics hints warnings error elapsed_ms predicted agree".split())


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    monkeypatch.setenv("WSB_GOLF_DIR", str(FIXTURES))
    monkeypatch.setenv("WSB_LLM", "mock")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with TestClient(app) as test_client:
        yield test_client


def submit(client, script):
    response = client.post("/api/run", json={"script": script})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"run_id"}
    assert UUID(hex=data["run_id"]).version == 4
    return data["run_id"]


def finish(client, run_id):
    url = "/api/runs/" + run_id
    deadline = monotonic() + 30
    while True:
        response = client.get(url)
        assert response.status_code == 200
        job = response.json()
        assert set(job) == {"run_id", "status", "verdicts", "current_line", "script", "source", "created"}
        assert job["run_id"] == run_id
        if job["status"] in ("done", "cancelled"):
            assert job["current_line"] is None
            assert all(set(v) == VERDICT_FIELDS for v in job["verdicts"])
            return job
        assert monotonic() < deadline, job
        sleep(.015)


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    return body["error"]


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"version": "0.1.0", "ok": True, "llm": "mock",
                               "arc_local": True, "models_local": {"pythia-14m": False}}


def test_page_language_examples_palette(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "WSBench" in page.text
    for asset in ("app.js", "style.css"):
        assert client.get("/static/" + asset).status_code == 200
    data = client.get("/api/language").json()
    assert set(data) == {"card", "ops", "examples"}
    assert data["card"] == LANGUAGE_CARD
    names = ["rotate_middle_layer", "antisymmetric_pair", "grokking_steerable", "arc_mirror"]
    assert [e["name"] for e in data["examples"]] == [n + ".wsl" for n in names]
    for entry in data["examples"]:
        assert client.post("/api/parse", json={"script": entry["script"]}).json()["errors"] == []
    ops = data["ops"]
    assert [op["family"] for op in ops] == ["neox"] * 16 + ["modadd"] * 7 + ["grid"] * 4
    assert all(set(op) == {"family", "call", "args", "doc"} for op in ops)
    assert ops[0] == {"family": "neox", "call": "fold_ln()", "args": "", "doc": "reparam (adds embed_out.bias)"}
    assert ops[16]["doc"].startswith("Only valid with claim")
    assert ops[-1]["call"] == "golf()"


def test_parse_and_recovering_run(client):
    script = """# offline example
model neox-tiny
x: gauge rotate(L1.mlp,orth)
x: edit delete(L0)
Broken: edit swap(L1   # comment
model invalid
bad: edit delete(L1)
model grid:task150
solve prim(vmirror)
"""
    parsed = client.post("/api/parse", json={"script": script}).json()
    assert set(parsed) == {"statements", "errors", "hints"}
    assert all(set(s) == {"line", "claim", "family", "source", "label"} for s in parsed["statements"])
    assert parsed["statements"][0] == {"line": 3, "claim": "gauge", "family": "neox",
                                       "source": "gauge rotate(L[1].mlp, orth)", "label": "x"}
    assert [(e["line"], e["code"]) for e in parsed["errors"]] == [
        (4, "E_LABEL"), (5, "E_SYNTAX"), (6, "E_MODEL"), (7, "E_MODEL")]
    assert [h["code"] for h in parsed["hints"]] == ["H_MLP_ROTATE"]
    run_id = submit(client, script)
    job = finish(client, run_id)
    rows = job["verdicts"]
    assert job["status"] == "done"
    assert [v["line"] for v in rows] == [3, 4, 5, 6, 7, 9]
    assert [v["status"] for v in rows] == ["REFUTED"] + ["ERROR"] * 4 + ["CONFIRMED"]
    assert rows[0]["hints"] == parsed["hints"]
    assert rows[2]["label"] == "Broken" and rows[2]["source"] == "Broken: edit swap(L1"
    assert rows[2]["model"] == "neox-tiny" and rows[3]["model"] is None
    assert rows[2]["claim"] is rows[3]["claim"] is None
    assert rows[-1]["label"] == "line9" and rows[-1]["metrics"]["passed"] == 7
    assert client.post(f"/api/runs/{run_id}/cancel").json() == {"ok": True}
    assert finish(client, run_id) == job


def test_tensor_heatmaps_strips_arc_and_invalid_requests(client):
    run_id = submit(client, "model neox-tiny\nA: edit scale(resid,2)\nA: edit delete(L0)\nreparam fold_ln()")
    job = finish(client, run_id)
    assert [v["status"] for v in job["verdicts"]] == ["MEASURED", "ERROR", "CONFIRMED"]
    name = "gpt_neox.embed_in.weight"
    params = {"run_id": run_id, "line": 2, "name": name}

    def tensor(which="before", **kwargs):
        return client.get("/api/tensor", params={**params, "which": which, **kwargs})

    before, after, diff = (tensor(which).json() for which in ("before", "after", "diff"))
    for view in (before, after, diff):
        assert set(view) == {"shape", "shown", "scale", "block", "name", "data"}
        assert view["shape"] == view["shown"] == [64, 16]
        assert view["block"] == [1, 1] and view["name"] == name
    assert before["scale"] == after["scale"]
    np.testing.assert_array_equal(before["data"], diff["data"])
    np.testing.assert_allclose(after["data"], 2 * np.array(before["data"]), rtol=1e-5)
    coarse = tensor(max=7).json()
    assert coarse["block"] == [10, 3] and coarse["shown"] == [7, 6]
    # Check the partial block in the last row and column against the unrounded model.
    from wsbench.models.registry import load_model
    weights = load_model("neox-tiny").W0[name].numpy()
    assert coarse["data"][-1][-1] == float(f"{weights[60:, 15:].mean():.6g}")
    for which, factor in [("before", 1), ("after", 2), ("diff", 1)]:
        block = factor * weights[3:7, 1:4]
        arc = tensor(which, view="arc", r0=3, r1=7, c0=1, c1=4).json()
        assert set(arc) == {"grid", "quantiles"}
        qs = np.quantile(block, np.arange(1, 10) / 10)
        np.testing.assert_allclose(arc["quantiles"], qs)
        assert arc["grid"] == (block[..., None] > qs).sum(axis=2).tolist()

    # Unchanged keys are read from the baseline and new keys have an implicit zero baseline.
    untouched = tensor("diff", name="gpt_neox.final_layer_norm.weight").json()
    assert untouched["shape"] == [16] and untouched["data"] == [[0] * 16]
    key = {"line": 4, "name": "embed_out.bias"}
    strip = tensor(**key).json()
    assert strip["shape"] == [64] and strip["shown"] == [1, 64]
    assert strip["data"] == [[0] * 64]
    assert tensor("diff", **key).json()["data"] == tensor("after", **key).json()["data"]
    assert strip["scale"] == tensor("after", **key).json()["scale"]

    for invalid in ({"max": 0}, {"view": "arc"}, {"view": "images"},
                    {"view": "arc", "r0": 0, "r1": 31, "c0": 0, "c1": 1},
                    {"view": "arc", "r0": -1, "r1": 1, "c0": 0, "c1": 1},
                    {"view": "arc", "r0": 0, "r1": 1, "c0": 0, "c1": 1, **key}):
        assert_error(tensor(**invalid), 400, "E_BAD_REQUEST")
    assert_error(tensor("other"), 400, "E_BAD_REQUEST")
    assert_error(client.get("/api/tensor", params=params), 400, "E_BAD_REQUEST")
    assert_error(tensor(line=3), 404, "E_NOT_FOUND")
    assert_error(tensor(name="missing"), 404, "E_NOT_FOUND")


def test_arc_index_examples_and_compile_views(client, tmp_path):
    index = client.get("/api/arc/tasks").json()
    assert [r["task"] for r in index] == ["task087", "task150"]
    assert index[1] == {"task": "task150", "original_id": None, "bytes_raw": None, "bytes_submitted": None,
                        "n_train": 3, "n_test": 1, "n_arcgen": 3, "single_primitive": "vmirror"}
    assert not (tmp_path / "golf/index.json").exists()
    detail = client.get("/api/arc/task/task150").json()
    assert set(detail) == {"task", "original_id", "bytes_raw", "bytes_submitted", "solution_raw",
                           "train", "test", "arc_gen_first20", "n_arcgen"}
    assert detail["solution_raw"] is None and len(detail["arc_gen_first20"]) == detail["n_arcgen"] == 3
    assert len(detail["train"]) == 3 and len(detail["test"]) == 1
    assert_error(client.get("/api/arc/task/task001"), 404, "E_NOT_FOUND")
    params = {"task": "task150", "prim": "vmirror", "size": "gated", "from": "all"}
    response = client.get("/api/arc/compiled", params={**params, "shape": "3x3"})
    assert response.status_code == 200
    view = response.json()
    assert set(view) == {"sizes", "shape", "src_grid", "P_preview"}
    assert [3, 3] in view["sizes"] and view["shape"] == [3, 3]
    assert np.shape(view["src_grid"]) == (30, 30) and np.shape(view["P_preview"]) == (300, 300)
    assert view["src_grid"][0][:4] == [2, 1, 0, -1]
    assert_error(client.get("/api/arc/compiled", params={**params, "shape": "1x1"}), 400, "E_BAD_REQUEST")
    fixed = client.get("/api/arc/compiled", params={**params, "size": "fixed", "shape": "1x1"}).json()
    assert fixed["sizes"] == [fixed["shape"]] == [[4, 4]]
    for key, value in [("prim", "unknown"), ("size", "unknown"), ("from", "unknown")]:
        assert_error(client.get("/api/arc/compiled", params={**params, key: value}), 400, "E_BAD_REQUEST")
    row, = finish(client, submit(client, "model grid:task150\nsolve golf()"))["verdicts"]
    assert row["error"] == {"code": "E_DOWNLOAD", "message": "run: wsbench fetch arc"}


def test_missing_data(client, monkeypatch):
    monkeypatch.delenv("WSB_GOLF_DIR")
    for path in ("/api/arc/tasks", "/api/arc/task/task150", "/api/arc/compiled?task=task150&prim=vmirror"):
        assert assert_error(client.get(path), 400, "E_DOWNLOAD")["message"] == "run: wsbench fetch arc"
    job = finish(client, submit(client, "model grid:task150\nsolve prim(vmirror)"))
    assert job["verdicts"][0]["error"]["code"] == "E_DOWNLOAD"


def test_assist_endpoint_executes_mock_script(client):
    request = {"text": "Global and partial residual rotation", "model": "neox-tiny"}
    data = client.post("/api/assist", json=request).json()
    assert set(data) == {"run_id", "script", "roles", "predictions", "parse_errors", "cost_usd"}
    assert data["cost_usd"] == 0 and not data["parse_errors"]
    assert data["script"].startswith("model neox-tiny\n")
    assert set(data["roles"]) == {"translate", "critic", "alternatives"}
    rows = finish(client, data["run_id"])["verdicts"]
    assert [v["label"] for v in rows] == ["t1", "t2", "t3", "fix2", "alt1"]
    assert [v["agree"] for v in rows] == ["agree", "agree", "agree", "n/a", "n/a"]


def test_bad_requests_and_unknown_ids(client):
    for url in ("/api/parse", "/api/run"):
        assert_error(client.post(url, json={}), 400, "E_BAD_REQUEST")
    assert_error(client.get("/api/runs/unknown"), 404, "E_NOT_FOUND")
    assert_error(client.post("/api/runs/unknown/cancel"), 404, "E_NOT_FOUND")
    assert_error(client.get("/api/arc/compiled"), 400, "E_BAD_REQUEST")
    assert_error(client.get("/api/tensor"), 400, "E_BAD_REQUEST")
    assert_error(client.post("/api/assist", json={"model": "modadd", "text": "test"}), 400, "E_BAD_REQUEST")


def test_queued_and_running_cancellation(client, monkeypatch):
    # Block one real neox op after it reaches RUNNING so that both queue states
    # can be inspected without depending on scheduler or CPU timing.
    from wsbench.exec import resolve
    original = resolve.execute_neox
    entered, release = Event(), Event()

    def blocked(stmt):
        entered.set()
        if not release.wait(30):
            raise TimeoutError("test did not release running operation")
        return original(stmt)

    monkeypatch.setattr(resolve, "execute_neox", blocked)
    first = submit(client, "model neox-tiny\nedit delete(L0)")
    try:
        assert entered.wait(15)
        status = client.get(f"/api/runs/{first}").json()
        assert status["status"] == "running" and status["current_line"] == 2
        assert status["verdicts"][0]["status"] == "RUNNING"
        queued = submit(client, "model grid:task087\nsolve prim(rot180)\nsolve bad()\nwrong model")
        status = client.get(f"/api/runs/{queued}").json()
        assert status["status"] == "queued" and status["current_line"] is None
        assert [v["status"] for v in status["verdicts"]] == ["PENDING", "ERROR", "ERROR"]
        for job in (queued, first):
            assert client.post(f"/api/runs/{job}/cancel").json() == {"ok": True}
        cancelled = finish(client, queued)
        assert cancelled["status"] == "cancelled"
        assert [v["error"]["code"] for v in cancelled["verdicts"]] == ["E_RUNTIME", "E_UNKNOWN_OP", "E_SYNTAX"]
        assert cancelled["verdicts"][0]["error"]["message"] == "cancelled"
    finally:
        release.set()
    completed = finish(client, first)
    assert completed["status"] == "cancelled"
    assert completed["verdicts"][0]["status"] == "MEASURED"
    params = {"run_id": first, "line": 2, "name": "gpt_neox.embed_in.weight", "which": "after"}
    assert client.get("/api/tensor", params=params).status_code == 200


def test_tensor_and_run_retention_boundaries(client):
    first = submit(client, "model neox-tiny\nedit delete(L1)")
    finish(client, first)
    params = {"run_id": first, "line": 2, "name": "embed_out.weight", "which": "before"}
    for _ in range(4):
        submit(client, "")
    assert client.get("/api/tensor", params=params).status_code == 200
    submit(client, "")
    assert assert_error(client.get("/api/tensor", params=params), 410, "E_EXPIRED")["message"] == "state expired; re-run the script"
    for _ in range(44):
        submit(client, "")
    assert client.get(f"/api/runs/{first}").status_code == 200
    last = submit(client, "")
    finish(client, last)
    assert_error(client.get(f"/api/runs/{first}"), 404, "E_NOT_FOUND")
    assert_error(client.get("/api/tensor", params=params), 404, "E_NOT_FOUND")


def test_invalid_statement_preserves_quoted_hash_and_context(client):
    source = 'A: edit gridop(rot90, block("key#part", 0, 1, 0, 1)'
    run_id = submit(client, "model neox-tiny\nmodel#invalid directive\n" + source + "  # unfinished")
    first, second = finish(client, run_id)["verdicts"]
    assert first["model"] == second["model"] == "neox-tiny"
    assert second["source"] == source
    assert first["error"]["code"] == second["error"]["code"] == "E_SYNTAX"


def test_runs_are_stored_and_survive_a_server_restart(client, tmp_path):
    from wsbench.exec import jobs as jobs_module

    script = "model neox-tiny\nK1: gauge permute(L[1].mlp, randperm)\n"
    run_id = submit(client, script)
    done = finish(client, run_id)
    assert done["script"] == script and done["source"] == "editor"
    stored = tmp_path / "history" / f"{run_id}.json"
    assert stored.exists()

    listing = client.get("/api/runs").json()
    assert listing[0]["run_id"] == run_id
    assert set(listing[0]) == {"run_id", "status", "source", "created", "script"}

    # a new process has an empty in-memory table and must answer from the stored file
    fresh = jobs_module.Jobs()
    again = fresh.snapshot(run_id)
    assert again["status"] == "done"
    assert [v["status"] for v in again["verdicts"]] == ["CONFIRMED"]
    assert fresh.recent()[0]["run_id"] == run_id
    with pytest.raises(jobs_module.JobError) as expired:
        fresh.tensors(run_id, 2, "gpt_neox.layers.1.mlp.dense_h_to_4h.weight")
    assert expired.value.code == "E_EXPIRED"
    assert fresh.cancel(run_id) == {"ok": True}


def test_a_run_interrupted_by_a_restart_is_reported_as_cancelled(client, tmp_path):
    import json
    from wsbench.exec import jobs as jobs_module
    from wsbench.exec.verdict import prepare

    script = "model neox-tiny\nA: gauge fold_ln()\nB: edit delete(L[1])\n"
    _, verdicts = prepare(script)
    verdicts[0]["status"] = "CONFIRMED"
    verdicts[1]["status"] = "RUNNING"
    run_id = "0" * 31 + "1"
    folder = tmp_path / "history"
    folder.mkdir()
    (folder / f"{run_id}.json").write_text(json.dumps(dict(
        run_id=run_id, status="running", verdicts=verdicts, current_line=3, script=script, source="editor",
        created=1.0, predictions=None)))
    job = client.get(f"/api/runs/{run_id}").json()
    assert job["status"] == "cancelled" and job["current_line"] is None
    assert job["verdicts"][0]["status"] == "CONFIRMED"
    assert job["verdicts"][1]["status"] == "ERROR"
    assert job["verdicts"][1]["error"]["message"].startswith("interrupted")


def test_run_ids_cannot_escape_the_history_folder(client, tmp_path):
    (tmp_path / "secret.json").write_text("{}")
    assert_error(client.get("/api/runs/..%2Fsecret"), 404, "E_NOT_FOUND")
    assert_error(client.get("/api/runs/not-a-run"), 404, "E_NOT_FOUND")


def test_arc_and_assist_runs_are_tagged(client):
    arc = client.post("/api/run", json={"script": "model grid:task087\nsolve prim(rot180)\n", "source": "arc"}).json()["run_id"]
    assert finish(client, arc)["source"] == "arc"
    data = client.post("/api/assist", json={"text": "rotate the residual stream", "model": "neox-tiny"}).json()
    assert finish(client, data["run_id"])["source"] == "assist"


def _wait_assist(client, assist_id):
    deadline = monotonic() + 30
    while True:
        task = client.get(f"/api/assist/{assist_id}").json()
        if task["status"] != "running":
            return task
        assert monotonic() < deadline
        sleep(0.05)


def test_assist_runs_in_the_background_and_is_stored(client, tmp_path, monkeypatch):
    from wsbench.assist import openrouter, tasks as tasks_module

    release = Event()
    real_chat = openrouter.chat

    def slow_chat(role, model_id, system, messages):
        release.wait(10)
        return real_chat(role, model_id, system, messages)

    monkeypatch.setattr(openrouter, "chat", slow_chat)
    started = monotonic()
    response = client.post("/api/assist/start", json={"text": "rotate the residual stream", "model": "neox-tiny"})
    assert response.status_code == 200 and monotonic() - started < 2     # answers before the models do
    assist_id = response.json()["assist_id"]
    assert client.get(f"/api/assist/{assist_id}").json()["status"] == "running"
    assert client.get("/api/assist/latest").json()["assist"]["assist_id"] == assist_id
    release.set()

    task = _wait_assist(client, assist_id)
    assert task["status"] == "done" and task["request"]["model"] == "neox-tiny"
    result = task["result"]
    assert set(result["roles"]) == {"translate", "critic", "alternatives"} and result["roles"]["translate"]["text"]
    assert finish(client, result["run_id"])["source"] == "assist"

    # a new server process answers from the stored file, including the model replies
    fresh = tasks_module.AssistTasks()
    assert fresh.get(assist_id)["result"]["script"] == result["script"]
    assert fresh.latest()["assist_id"] == assist_id


def test_assist_start_rejects_bad_requests_and_unknown_ids(client):
    assert_error(client.post("/api/assist/start", json={"text": "x", "model": "modadd"}), 400, "E_BAD_REQUEST")
    assert_error(client.get("/api/assist/" + "0" * 32), 404, "E_NOT_FOUND")
    assert_error(client.get("/api/assist/..%2Fsecret"), 404, "E_NOT_FOUND")


def test_an_assist_request_interrupted_by_a_restart_reports_an_error(client, tmp_path):
    import json

    folder = tmp_path / "assist_history"
    folder.mkdir()
    assist_id = "0" * 31 + "2"
    (folder / f"assist-{assist_id}.json").write_text(json.dumps(dict(
        assist_id=assist_id, status="running", created=1.0, result=None, error=None, request={"text": "x", "model": "neox-tiny"})))
    task = client.get(f"/api/assist/{assist_id}").json()
    assert task["status"] == "error" and task["error"]["message"].startswith("interrupted")


def test_assist_explains_when_nothing_is_testable(client, monkeypatch):
    from wsbench.assist import openrouter

    def chat(role, model_id, system, messages):
        text = "```wsl\nmodel neox-tiny\n# claim: the text is about floating-point rounding, not about weights.\n```"
        return dict(ok=True, text=text, model=model_id, prompt_tokens=1, completion_tokens=1, cost_usd=0.0,
                    error=None, finish_reason="stop")

    monkeypatch.setattr(openrouter, "chat", chat)
    data = client.post("/api/assist", json={"text": "KL divergence is never negative", "model": "neox-tiny"}).json()
    assert data["run_id"] is None and data["roles"]["critic"] is None
    assert data["parse_errors"][0]["message"].startswith("no testable statement")


def _wait_explore(client, explore_id, until=("done", "cancelled", "error")):
    deadline = monotonic() + 60
    while True:
        task = client.get(f"/api/explore/{explore_id}").json()
        if task["status"] in until:
            return task
        assert monotonic() < deadline, task["status"]
        sleep(0.05)


def test_explore_loop_measures_rounds_and_writes_a_report(client, tmp_path):
    from wsbench.assist import explore_tasks as module

    response = client.post("/api/explore/start", json={"text": "rotation idea", "model": "neox-tiny", "rounds": 3})
    explore_id = response.json()["explore_id"]
    task = _wait_explore(client, explore_id)
    assert task["status"] == "done" and task["error"] is None
    state = task["state"]
    assert state["stopped"] == "planner finished" and len(state["rounds"]) == 1
    first = state["rounds"][0]
    assert first["status"] == "done" and first["question"].startswith("A global rotation")
    assert [v["status"] for v in first["verdicts"]] == ["CONFIRMED", "REFUTED", "MEASURED", "CONFIRMED", "CONFIRMED"]
    assert [v["agree"] for v in first["verdicts"]][:3] == ["agree", "agree", "agree"]
    assert state["report"].startswith("Confirmed")
    assert client.get(f"/api/runs/{first['run_id']}").json()["source"] == "explore"
    assert client.get("/api/explore/latest").json()["explore"]["explore_id"] == explore_id
    assert module.ExploreTasks().get(explore_id)["state"]["report"] == state["report"]     # stored on disk


def test_explore_waits_for_the_researchers_answer(client):
    explore_id = client.post("/api/explore/start", json={"text": "[[mock-ask]] idea", "model": "neox-tiny"}).json()["explore_id"]
    task = _wait_explore(client, explore_id, until=("waiting", "done", "error"))
    assert task["status"] == "waiting" and task["state"]["waiting"] == "Rotation or permutation first?"
    assert client.post(f"/api/explore/{explore_id}/answer", json={"answer": "rotation"}).json() == {"ok": True}
    task = _wait_explore(client, explore_id)
    assert task["status"] == "done"
    assert task["state"]["answers"] == [{"question": "Rotation or permutation first?", "answer": "rotation"}]
    assert len(task["state"]["rounds"]) == 1
    assert_error(client.post(f"/api/explore/{explore_id}/answer", json={"answer": "again"}), 400, "E_BAD_REQUEST")


def test_explore_can_be_cancelled_while_it_waits(client):
    explore_id = client.post("/api/explore/start", json={"text": "[[mock-ask]] idea", "model": "neox-tiny"}).json()["explore_id"]
    _wait_explore(client, explore_id, until=("waiting",))
    assert client.post(f"/api/explore/{explore_id}/cancel").json() == {"ok": True}
    task = _wait_explore(client, explore_id)
    assert task["status"] == "cancelled" and task["state"]["rounds"] == [] and task["state"]["report"] is None


def test_explore_stops_at_the_budget_and_after_all_rounds(client, monkeypatch):
    from wsbench.assist import mock, openrouter

    def chat(role, model_id, system, messages):
        text = mock.PLAN_FIRST if role == "planner" else mock.reply(role, messages)
        return dict(ok=True, text=text, model=model_id, prompt_tokens=1, completion_tokens=1, cost_usd=0.2,
                    error=None, finish_reason="stop")

    monkeypatch.setattr(openrouter, "chat", chat)
    explore_id = client.post("/api/explore/start", json={"text": "idea", "model": "neox-tiny", "rounds": 5,
                                                         "budget_usd": 1.0}).json()["explore_id"]
    state = _wait_explore(client, explore_id)["state"]
    assert state["stopped"] == "budget" and len(state["rounds"]) == 1       # 0.8 per round; a second would exceed 1.0
    assert state["cost_usd"] <= 1.0 + 0.2                                   # only the report call comes on top
    explore_id = client.post("/api/explore/start", json={"text": "idea", "model": "neox-tiny", "rounds": 2,
                                                         "budget_usd": 5.0}).json()["explore_id"]
    state = _wait_explore(client, explore_id)["state"]
    assert state["stopped"] == "all rounds used" and len(state["rounds"]) == 2 and state["report"]


def test_explore_falls_back_to_a_second_planner_model(client, monkeypatch):
    from wsbench.assist import mock, openrouter

    def chat(role, model_id, system, messages):
        refused = role in ("planner", "report") and model_id == "anthropic/claude-fable-5.2"
        return dict(ok=not refused, text="" if refused else mock.reply(role, messages), model=model_id,
                    prompt_tokens=1, completion_tokens=1, cost_usd=0.0,
                    error="content_filter" if refused else None, finish_reason="stop")

    monkeypatch.setattr(openrouter, "chat", chat)
    explore_id = client.post("/api/explore/start", json={"text": "idea", "model": "neox-tiny"}).json()["explore_id"]
    state = _wait_explore(client, explore_id)["state"]
    assert state["stopped"] == "planner finished" and state["report"]
    assert [c["ok"] for c in state["planner_calls"][:2]] == [False, True]


def test_explore_rejects_bad_requests(client):
    for body in ({"text": "", "model": "neox-tiny"}, {"text": "x", "model": "modadd"},
                 {"text": "x", "model": "neox-tiny", "rounds": 9}, {"text": "x", "model": "neox-tiny", "budget_usd": 50}):
        assert_error(client.post("/api/explore/start", json=body), 400, "E_BAD_REQUEST")
    assert_error(client.get("/api/explore/" + "0" * 32), 404, "E_NOT_FOUND")
    assert_error(client.post("/api/explore/" + "0" * 32 + "/cancel"), 404, "E_NOT_FOUND")
