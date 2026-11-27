"""Offline HTTP acceptance tests, including the asynchronous run lifecycle."""
import re
import shutil
import subprocess
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from uuid import UUID

from json import dumps as json_dumps, loads as json_loads

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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
    assert response.json() == {"version": "0.1.0", "ok": True, "llm": "mock", "arc_local": True,
                               "models_local": {"pythia-14m": False, "transfo-xl-wt103": False},
                               "anthropic": False, "agent": True}


def test_page_language_examples_palette(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "WSBench" in page.text
    for asset in ("app.js", "style.css"):
        assert client.get("/static/" + asset).status_code == 200
    data = client.get("/api/language").json()
    assert set(data) == {"card", "ops", "examples"}
    assert data["card"] == LANGUAGE_CARD
    names = ["rotate_middle_layer", "antisymmetric_pair", "grokking_steerable", "arc_mirror", "transformer_xl"]
    assert [e["name"] for e in data["examples"]] == [n + ".wsl" for n in names]
    for entry in data["examples"]:
        assert client.post("/api/parse", json={"script": entry["script"]}).json()["errors"] == []
    ops = data["ops"]
    assert [op["family"] for op in ops] == ["neox"] * 16 + ["txl"] * 10 + ["modadd"] * 7 + ["grid"] * 4
    assert all(set(op) == {"family", "call", "args", "doc"} for op in ops)
    assert ops[0] == {"family": "neox", "call": "fold_ln()", "args": "", "doc": "reparam (adds embed_out.bias)"}
    assert ops[26]["doc"].startswith("Only valid with claim")
    txl_calls = [op["call"] for op in ops if op["family"] == "txl"]
    assert "qk(L[i].H[h], A, dims=all)" in txl_calls and not any(c.startswith(("fold_ln", "rope_pairs")) for c in txl_calls)
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


def test_txl_parse_reports_families_reasons_and_hints(client):
    script = """model transfo-xl-wt103
A: gauge permute(L[mid].heads, randperm)
B: gauge rotate(L[mid].mlp, orth)
C: gauge fold_ln(); rotate(resid, orth1)
model txl-tiny
D: edit L[mid] := lerp(L1, L3, 0.5)
E: gauge qk(L0.H0, gl, dims=pass)
"""
    parsed = client.post("/api/parse", json={"script": script}).json()
    assert [(s["label"], s["family"]) for s in parsed["statements"]] == [
        ("A", "txl"), ("B", "txl"), ("C", "txl"), ("D", "txl"), ("E", "txl")]
    assert parsed["statements"][0]["source"] == "gauge permute(L[mid].heads, randperm)"
    assert [(e["line"], e["code"], e["message"]) for e in parsed["errors"]] == [
        (4, "E_MODEL", "fold_ln is not defined for post-LayerNorm Transformer-XL"),
        (7, "E_MODEL", "qk(..., dims=pass) is not defined for Transformer-XL: it has no rotary dimensions; use dims=all")]
    assert [(h["line"], h["code"]) for h in parsed["hints"]] == [(3, "H_RELU_ROTATE"), (6, "H_LERP_ALIGN")]


def test_txl_run_through_the_queue_and_its_tensor_views(client):
    script = "model txl-tiny\nG: gauge vo(L1.H0, gl)\nD: edit delete(L[mid])\nS: edit swap(L0, L3)\nT: gauge fold_ln()"
    run_id = submit(client, script)
    job = finish(client, run_id)
    rows = job["verdicts"]
    assert [(v["label"], v["status"], v["measured"]) for v in rows] == [
        ("G", "CONFIRMED", "exact"), ("D", "MEASURED", "edit"), ("S", "MEASURED", "edit"), ("T", "ERROR", None)]
    assert all(v["model"] == "txl-tiny" for v in rows) and rows[3]["error"]["code"] == "E_MODEL"
    deleted = rows[1]["metrics"]
    assert deleted["rel"] > 1e-3 and deleted["control"]["ratio"] is not None
    assert {t["name"] for t in deleted["changed_tensors"]} == {
        "transformer.layers.2.dec_attn.o_net.weight", "transformer.layers.2.pos_ff.CoreNet.3.weight",
        "transformer.layers.2.pos_ff.CoreNet.3.bias"}

    def tensor(line, name, which, **kwargs):
        return client.get("/api/tensor", params={"run_id": run_id, "line": line, "name": name, "which": which, **kwargs})

    from wsbench.models.registry import load_model
    W0 = load_model("txl-tiny").W0
    name = "transformer.layers.2.pos_ff.CoreNet.3.weight"
    before, after, diff = (tensor(3, name, which).json() for which in ("before", "after", "diff"))
    assert before["shape"] == before["shown"] == [16, 64] and before["block"] == [1, 1]
    np.testing.assert_allclose(before["data"], W0[name].numpy(), rtol=1e-5)
    assert after["data"] == [[0] * 64] * 16
    np.testing.assert_allclose(diff["data"], -W0[name].numpy(), rtol=1e-5)
    bias = tensor(3, "transformer.layers.2.pos_ff.CoreNet.3.bias", "after").json()
    assert bias["shape"] == [16] and bias["shown"] == [1, 16] and bias["data"] == [[0] * 16]
    untouched = tensor(3, "transformer.word_emb.emb_layers.0.weight", "diff").json()
    assert untouched["shape"] == [16, 16] and untouched["data"] == [[0] * 16] * 16
    arc = tensor(3, name, "before", view="arc", r0=0, r1=4, c0=0, c1=6).json()
    assert np.shape(arc["grid"]) == (4, 6) and len(arc["quantiles"]) == 9
    # the swap stored the baseline's own tensors: after(L0) is before(L3)
    swapped = tensor(4, "transformer.layers.0.dec_attn.qkv_net.weight", "after").json()
    assert swapped["data"] == tensor(4, "transformer.layers.3.dec_attn.qkv_net.weight", "before").json()["data"]
    assert_error(tensor(3, "crit.out_layers.0.weight", "after"), 404, "E_NOT_FOUND")     # tied: not a separate tensor
    assert_error(tensor(5, name, "after"), 404, "E_NOT_FOUND")                           # an invalid line has no state


def test_assist_and_explore_accept_txl_targets(client):
    data = client.post("/api/assist", json={"text": "rotation idea", "model": "txl-tiny"}).json()
    assert data["script"] == "model txl-tiny" and data["run_id"] is None        # the mock translator writes NeoX ops
    assert {e["message"] for e in data["parse_errors"] if e["code"] == "E_MODEL"} >= {
        "fold_ln is not defined for post-LayerNorm Transformer-XL"}
    for model in ("txl-tiny", "transfo-xl-wt103"):
        assert client.post("/api/assist/start", json={"text": "x", "model": model, "execute": False}).status_code == 200
    explore_id = client.post("/api/explore/start", json={"text": "idea", "model": "transfo-xl-wt103", "rounds": 1}).json()
    assert set(explore_id) == {"explore_id"}
    assert _wait_explore(client, explore_id["explore_id"])["status"] == "done"


def test_tensor_states_stay_within_the_memory_budget(client, monkeypatch):
    from wsbench.exec import jobs as jobs_module
    monkeypatch.setattr(jobs_module, "STATE_BYTES", 25_000)        # two txl-tiny deletes (10,368 bytes each) fit
    name = "transformer.layers.{}.pos_ff.CoreNet.3.weight"

    def view(run_id, line, layer):
        return client.get("/api/tensor", params={"run_id": run_id, "line": line, "name": name.format(layer), "which": "after"})

    first = submit(client, "model txl-tiny\nedit delete(L0)")
    finish(client, first)
    assert view(first, 2, 0).status_code == 200
    second = submit(client, "model txl-tiny\nedit delete(L1)\nedit delete(L2)")
    finish(client, second)
    assert assert_error(view(first, 2, 0), 410, "E_EXPIRED")["message"] == "state expired; re-run the script"
    assert view(second, 2, 1).status_code == view(second, 3, 2).status_code == 200
    third = submit(client, "model txl-tiny\nedit delete(L0)\nedit delete(L1)\nedit delete(L2)")
    finish(client, third)
    assert_error(view(second, 2, 1), 410, "E_EXPIRED")             # older runs go first
    assert_error(view(third, 2, 0), 410, "E_EXPIRED")              # then the earliest lines of the run itself
    assert view(third, 3, 1).status_code == view(third, 4, 2).status_code == 200


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


class _FakeAnthropic:
    """Stands in for anthropic.Anthropic(): records the request and returns a canned Message-like object."""

    def __init__(self, stop_reason="end_turn", model="claude-fable-5-2", text="hello"):
        from types import SimpleNamespace as NS
        self.calls = []
        self._reply = NS(model=model, stop_reason=stop_reason, stop_details=NS(category="cyber"),
                         content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
                         usage=NS(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0))
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._reply


def test_claude_models_use_the_anthropic_api_when_its_key_is_set(client, monkeypatch):
    from wsbench.assist import anthropic_api, openrouter

    monkeypatch.delenv("WSB_LLM")
    assert client.get("/api/health").json()["anthropic"] is False
    assert openrouter.chat("translate", "anthropic/claude-fable-5.2", "s", [])["error"] == "no-key"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake = _FakeAnthropic()
    monkeypatch.setattr(anthropic_api, "_client", fake)
    health = client.get("/api/health").json()
    assert health["anthropic"] is True and health["llm"] == "live"

    reply = openrouter.chat("planner", "anthropic/claude-fable-5.2", "system text", [{"role": "user", "content": "hi"}])
    assert reply["ok"] and reply["text"] == "hello" and reply["provider"] == "anthropic"
    assert reply["cost_usd"] == pytest.approx((1000 * 8 + 500 * 40) / 1e6)
    sent = fake.calls[0]
    assert sent["model"] == "claude-fable-5-2" and sent["system"] == "system text"
    assert sent["output_config"] == {"effort": "low"} and "thinking" not in sent and "temperature" not in sent
    assert sent["fallbacks"] == "default" and sent["betas"] == ["server-side-fallback-2026-07-01"]

    # other vendors still go to OpenRouter, which has no key here
    assert openrouter.chat("critic", "openai/gpt-6.2-sol", "s", [])["error"] == "no-key"


def test_anthropic_refusals_and_fallback_models_are_reported(monkeypatch):
    from wsbench.assist import anthropic_api

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic_api, "_client", _FakeAnthropic(stop_reason="refusal"))
    reply = anthropic_api.chat("planner", "anthropic/claude-fable-5.2", "s", [{"role": "user", "content": "x"}])
    assert not reply["ok"] and reply["error"] == "refusal (cyber)"

    monkeypatch.setattr(anthropic_api, "_client", _FakeAnthropic(model="claude-opus-4-8"))
    reply = anthropic_api.chat("planner", "anthropic/claude-fable-5.2", "s", [{"role": "user", "content": "x"}])
    assert reply["ok"] and reply["served_by"] == "claude-opus-4-8"
    assert reply["cost_usd"] == pytest.approx((1000 * 5 + 500 * 25) / 1e6)      # billed at the fallback model's rate


class _FakePlatform:
    """Stands in for the Managed Agents API: scripted events per turn, records what the client sends."""

    def __init__(self, turns):
        from types import SimpleNamespace as NS
        self.NS, self.turns, self.sent, self.created = NS, list(turns), [], {"env": 0, "agent": 0, "update": 0, "session": []}
        self.history = []
        self.send_fail = None
        self._turn = -1
        outer = self

        class Stream:
            def __init__(self, events): self.events = events
            def __enter__(self): return iter(self.events)
            def __exit__(self, *a): return False

        def stream(session_id):                     # opened before the message is sent
            return Stream(list(outer.turns[outer._turn + 1]))

        def send(session_id, events):
            if outer.send_fail is not None:
                outer.send_fail(events)
            outer.sent.extend(events)
            if events[0]["type"] == "user.message":
                outer._turn += 1

        def env_create(**kw): outer.created["env"] += 1; return NS(id="env_1")
        def agent_create(**kw): outer.created["agent"] += 1; outer.agent_spec = kw; return NS(id="agent_1", version=1)
        def agent_update(agent_id, **kw): outer.created["update"] += 1; return NS(id=agent_id, version=2)
        def session_create(**kw): outer.created["session"].append(kw); return NS(id="sesn_1")
        def session_retrieve(session_id): return NS(usage=NS(list_cost=NS(amount="37", currency="USD")))

        self.beta = NS(environments=NS(create=env_create), agents=NS(create=agent_create, update=agent_update),
                       sessions=NS(create=session_create, retrieve=session_retrieve,
                                   events=NS(stream=stream, send=send, list=lambda session_id, **kw: list(outer.history))))


def _ev(kind, n, **fields):
    from types import SimpleNamespace as NS
    return NS(type=kind, id=f"sevt_{n}", **fields)


def _wait_agent(client, conv_id):
    deadline = monotonic() + 30
    while True:
        record = client.get(f"/api/agent/{conv_id}").json()
        if record["status"] != "running":
            return record
        assert monotonic() < deadline
        sleep(0.05)


def test_platform_agent_runs_wsbench_tools_and_keeps_a_transcript(client, tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from wsbench.agent import sessions as module

    monkeypatch.delenv("WSB_LLM")
    assert_error(client.post("/api/agent/start", json={"text": "idea"}), 400, "E_NO_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    script = "model neox-tiny\nP1: gauge permute(L[1].mlp, randperm)\nP2: gauge rotate(L[1].mlp, orth)\n"
    idle = NS(type="end_turn")
    platform = _FakePlatform([
        [_ev("agent.message", 1, content=[NS(type="text", text="Ich teste Permutation gegen Rotation.")]),
         _ev("agent.custom_tool_use", 2, name="wsl_check", input={"script": script}),
         _ev("session.status_idle", 3, stop_reason=NS(type="requires_action")),
         _ev("agent.custom_tool_use", 4, name="wsl_run", input={"script": script}),
         _ev("agent.message", 5, content=[NS(type="text", text="Permutation ist exakt, Rotation nicht.")]),
         _ev("session.status_idle", 6, stop_reason=idle)],
        [_ev("agent.message", 7, content=[NS(type="text", text="Gern.")]),
         _ev("session.status_idle", 8, stop_reason=idle)],
    ])
    monkeypatch.setattr(module.sessions, "_client", platform)

    conv_id = client.post("/api/agent/start", json={"text": "Sind Neuronen vertauschbar?", "budget_usd": 1.5}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "idle" and record["cost_usd"] == 0.37 and record["session_id"] == "sesn_1"
    assert [item["kind"] for item in record["items"]] == ["user", "agent", "tool", "tool", "agent"]
    check, run = record["items"][2], record["items"][3]
    assert json_loads(check["result"])["errors"] == []
    verdicts = json_loads(run["result"])["verdicts"]
    assert [v["status"] for v in verdicts] == ["CONFIRMED", "REFUTED"]
    assert client.get(f"/api/runs/{run['run_id']}").json()["source"] == "agent"

    # setup happened once, with a budget in cents and only WSBench's custom tools
    assert platform.created["env"] == 1 and platform.created["agent"] == 1
    assert platform.created["session"][0]["budget"]["max_list_cost"] == {"amount": "150", "currency": "USD"}
    assert {t["name"] for t in platform.agent_spec["tools"]} == {"wsl_check", "wsl_run", "arc_tasks", "arc_task", "arxiv_paper", "lesswrong_post", "github_repo"}
    assert platform.agent_spec["model"]["id"] == "claude-fable-5-2"
    results = [e for e in platform.sent if e["type"] == "user.custom_tool_result"]
    assert [e["custom_tool_use_id"] for e in results] == ["sevt_2", "sevt_4"]

    # a second message continues the same session; the agent is not created again
    assert client.post(f"/api/agent/{conv_id}/message", json={"text": "Danke"}).json() == {"ok": True}
    record = _wait_agent(client, conv_id)
    assert record["items"][-1] == {"kind": "agent", "text": "Gern."} and len(platform.created["session"]) == 1
    assert platform.created["agent"] == 1 and (tmp_path / "agent.json").exists()
    assert client.get("/api/agent/latest").json()["agent"]["agent_conv_id"] == conv_id
    stored = module.AgentSessions().get(conv_id)                           # a new process reads it from disk
    assert stored["status"] == "idle" and stored["session_id"] == "sesn_1" and "seen" not in stored


def test_platform_agent_reports_budget_and_errors(client, monkeypatch):
    from types import SimpleNamespace as NS
    from wsbench.agent import sessions as module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    platform = _FakePlatform([[_ev("session.status_idle", 1, stop_reason=NS(type="budget_reached"))]])
    monkeypatch.setattr(module.sessions, "_client", platform)
    conv_id = client.post("/api/agent/start", json={"text": "idea"}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "budget" and record["items"][-1]["kind"] == "note"
    assert_error(client.post(f"/api/agent/{conv_id}/message", json={"text": "more"}), 400, "E_BAD_REQUEST")
    assert_error(client.post("/api/agent/start", json={"text": "idea", "budget_usd": 500}), 400, "E_BAD_REQUEST")
    assert_error(client.get("/api/agent/" + "0" * 32), 404, "E_NOT_FOUND")

    def boom(**kw): raise RuntimeError("platform down")
    platform.beta.sessions.create = boom
    conv_id = client.post("/api/agent/start", json={"text": "idea"}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "error" and "platform down" in record["error"]


def test_arxiv_tool_reads_only_arxiv_and_pages_the_text(tmp_path, monkeypatch):
    from wsbench.agent import arxiv, tools

    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    calls = []
    atom = ("<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>A  Test\nPaper</title><summary>Short abstract.</summary>"
            "<published>2026-11-23T15:18:30Z</published><author><name>A. Author</name></author></entry></feed>")
    page = ("<html><head><style>x{}</style></head><body><nav>menu</nav><h2>1 Intro</h2><p>Hello <math alttext='x^2'>"
            "<mi>x</mi></math> world.</p><script>bad()</script><h2>2 Method</h2><p>" + "word " * 4000 + "</p></body></html>")

    def get(url):
        calls.append(url)
        return atom if "export.arxiv.org" in url else page

    overview = arxiv.read("https://arxiv.org/abs/2611.30835v1", 0, get)
    assert overview["title"] == "A Test Paper" and overview["abstract"] == "Short abstract." and overview["published"] == "2026-11-23"
    assert overview["headings"] == ["1 Intro", "2 Method"] and overview["text_parts"] == 2
    first = arxiv.read("2611.30835", 1, get)
    assert first["text"].startswith("1 Intro\nHello $x^2$ world.") and "menu" not in first["text"] and "bad()" not in first["text"]
    assert len(calls) == 2 and all("arxiv.org" in url and "2611.30835" in url for url in calls)      # cached after the first read
    with pytest.raises(ValueError):
        arxiv.read("2611.30835", 3, get)
    for bad in ("https://example.com/paper.pdf", "../../etc/passwd", ""):
        assert "not an arXiv identifier" in tools.execute("arxiv_paper", {"paper": bad}, None)


def test_lesswrong_tool_reads_posts_by_id(tmp_path, monkeypatch):
    from wsbench.agent import lesswrong, tools

    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    asked = []

    def fetch(post_id):
        asked.append(post_id)
        return {"data": {"post": {"result": {"title": "A Post", "postedAt": "2026-10-17T16:35:07Z", "baseScore": 42,
                                             "user": {"displayName": "someone"},
                                             "htmlBody": "<h2>Claim</h2><p>Layers <b>commute</b>.</p><script>x()</script>"}}}}

    post_id = "3375SZNGA5dKHQMie"
    overview = lesswrong.read(f"https://www.alignmentforum.org/posts/{post_id}/a-post?x=1", 0, fetch)
    assert (overview["title"], overview["author"], overview["posted"], overview["karma"]) == ("A Post", "someone", "2026-10-17", 42)
    assert overview["headings"] == ["Claim"] and overview["text_parts"] == 1
    assert lesswrong.read(post_id, 1, fetch)["text"] == "Claim\nLayers commute."
    assert asked == [post_id]                                           # cached after the first read
    for bad in ("https://example.com/posts/3375SZNGA5dKHQMie/x", "short", "../../etc"):
        assert "not a LessWrong post" in tools.execute("lesswrong_post", {"post": bad}, None)


def test_github_tool_reads_and_searches_a_repository_without_running_it(tmp_path, monkeypatch):
    import subprocess
    from wsbench.agent import github, tools

    monkeypatch.setenv("WSB_HOME", str(tmp_path / "home"))
    source = tmp_path / "remote" / "owner" / "demo"
    (source / "pkg").mkdir(parents=True)
    (source / "README.md").write_text("# Demo\nA predictive coding toy.\n")
    (source / "pkg" / "core.py").write_text("def step(x):\n    return x + 1  # Hebbian update\n")
    (source / "pkg" / "blob.bin").write_bytes(b"\0\1\2")
    git = ["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    base = str(tmp_path / "remote") + "/"

    overview = github.read("https://github.com/owner/demo", base=base)
    assert overview["entries"] == ["README.md", "pkg/ (2 files)"] and overview["readme"].startswith("# Demo")
    assert overview["file_types"][".py"] == 1 and overview["files_below"] == 3
    assert github.read("owner/demo", "pkg", base=base)["entries"] == ["blob.bin", "core.py"]
    assert "Hebbian update" in github.read("owner/demo", "pkg/core.py", base=base)["text"]
    assert github.read("owner/demo", search="hebbian", base=base)["matches"] == ["pkg/core.py:2: return x + 1  # Hebbian update"]
    for path, message in (("pkg/blob.bin", "binary file"), ("../../home", "leaves the repository"), (".git/config", "not readable"),
                          ("missing.py", "no such path")):
        with pytest.raises(ValueError, match=message):
            github.read("owner/demo", path, base=base)
    for bad in ("../x", "https://evil.example/a/b", "a/b; rm -rf /", "owner"):
        assert "not a GitHub repository" in tools.execute("github_repo", {"repo": bad}, None)


def test_session_titles_are_single_line_plain_text(client, monkeypatch):
    from types import SimpleNamespace as NS
    from wsbench.agent import sessions as module

    pasted = "Wrong, Not Just Weird: Reasoning Models\u2019\nSelf-Checks\tFlag Va\u00adlues\u200b That\r\nContradict \U0001d438 = 0.19"
    title = module.session_title(pasted)
    assert title == "WSBench: Wrong, Not Just Weird: Reasoning Models\u2019 Self-Checks Flag Va"
    assert module.session_title("Va\u00adlues\u200b ok\x00\x1f!") == "WSBench: Values ok !"
    import unicodedata
    assert all(unicodedata.category(ch)[0] != "C" for ch in title) and len(title) <= len("WSBench: ") + 60
    assert module.session_title("\n\n  \n") == "WSBench: "

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    platform = _FakePlatform([[_ev("session.status_idle", 1, stop_reason=NS(type="end_turn"))]])
    monkeypatch.setattr(module.sessions, "_client", platform)
    conv_id = client.post("/api/agent/start", json={"text": pasted}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "idle" and record["items"][0]["text"] == pasted.strip()      # the message itself is sent unchanged
    assert platform.created["session"][0]["title"] == title
    assert [e for e in platform.sent if e["type"] == "user.message"][0]["content"][0]["text"] == pasted.strip()


def _platform_client(client, monkeypatch, turns, **kw):
    from wsbench.agent import sessions as module
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(module, "_sleep", lambda seconds: None)
    platform = _FakePlatform(turns)
    monkeypatch.setattr(module.sessions, "_client", platform)
    return module, platform


def test_platform_agent_reports_refusals_and_exhausted_retries_and_stays_usable(client, monkeypatch):
    from types import SimpleNamespace as NS
    module, platform = _platform_client(client, monkeypatch, [
        [_ev("session.status_idle", 1, stop_reason=NS(type="refusal"), stop_details=NS(category="cyber", explanation="not allowed"))],
        [_ev("session.status_idle", 2, stop_reason=NS(type="retries_exhausted"))],
        [_ev("agent.message", 3, content=[NS(type="text", text="ok"), NS(type="redacted")]),
         _ev("session.status_idle", 4, stop_reason=NS(type="end_turn"))],
    ])
    conv_id = client.post("/api/agent/start", json={"text": "idea"}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "idle" and "declined to answer this turn (cyber): not allowed" in record["items"][-1]["text"]
    client.post(f"/api/agent/{conv_id}/message", json={"text": "again"})
    record = _wait_agent(client, conv_id)
    assert record["status"] == "idle" and "gave up on this turn" in record["items"][-1]["text"]
    client.post(f"/api/agent/{conv_id}/message", json={"text": "once more"})
    record = _wait_agent(client, conv_id)
    assert record["items"][-1] == {"kind": "agent", "text": "ok[withheld by the platform]"}


def test_a_tool_result_that_failed_to_send_is_sent_again_after_the_reconnect(client, monkeypatch):
    from types import SimpleNamespace as NS
    events = [
        _ev("agent.custom_tool_use", 1, name="wsl_check", input={"script": "model neox-tiny\nX: gauge fold_ln()\n"}),
        _ev("session.status_idle", 2, stop_reason=NS(type="requires_action")),
        _ev("agent.message", 3, content=[NS(type="text", text="done")]),
        _ev("session.status_idle", 4, stop_reason=NS(type="end_turn")),
    ]
    module, platform = _platform_client(client, monkeypatch, [events, events])          # the reconnect opens a new stream
    platform.history = [_ev("agent.custom_tool_use", 1, name="wsl_check", input={})]
    failures = []

    def fail_once(events):
        if events[0]["type"] == "user.custom_tool_result" and not failures:
            failures.append(1)
            raise RuntimeError("connection reset")
    platform.send_fail = fail_once
    conv_id = client.post("/api/agent/start", json={"text": "idea"}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "idle" and failures == [1]
    results = [e for e in platform.sent if e["type"] == "user.custom_tool_result"]
    assert len(results) == 1 and results[0]["custom_tool_use_id"] == "sevt_1" and '"errors": []' in results[0]["content"][0]["text"]
    assert [i["kind"] for i in record["items"]].count("tool") == 1            # executed once, delivered once
    assert any(i["kind"] == "note" and "reconnecting" in i["text"] for i in record["items"])


def test_the_event_history_is_read_through_all_pages(client, monkeypatch):
    module, platform = _platform_client(client, monkeypatch, [[]])
    calls = []

    def listing(session_id, **kw):
        calls.append(kw)
        return (event for event in [_ev("agent.message", n, content=[]) for n in range(3)])      # an iterator: pages follow
    platform.beta.sessions.events.list = listing
    assert len(module.sessions._history("sesn_1")) == 3 and calls == [{"limit": 1000}]


def test_a_failed_turn_and_a_server_restart_can_be_continued(client, tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    module, platform = _platform_client(client, monkeypatch, [
        [_ev("session.status_idle", 1, stop_reason=NS(type="end_turn"))],
        [_ev("agent.message", 2, content=[NS(type="text", text="back")]), _ev("session.status_idle", 3, stop_reason=NS(type="end_turn"))],
    ])
    # the first user message fails to send: the conversation is in error, but its session exists
    platform.send_fail = lambda events: (_ for _ in ()).throw(RuntimeError("platform down"))
    conv_id = client.post("/api/agent/start", json={"text": "idea"}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "error" and record["session_id"] == "sesn_1" and "platform down" in record["error"]
    platform.send_fail = None
    client.post(f"/api/agent/{conv_id}/message", json={"text": "retry"})
    assert _wait_agent(client, conv_id)["status"] == "idle"

    # a new server process: the stored conversation is re-attached; an unanswered call is settled and a missed message shown
    platform.history = [_ev("agent.message", 90, content=[NS(type="text", text="said while away")]),
                        _ev("agent.custom_tool_use", 91, name="wsl_run", input={}),
                        _ev("agent.custom_tool_use", 92, name="wsl_check", input={}),
                        _ev("user.custom_tool_result", 93, custom_tool_use_id="sevt_92")]
    fresh = module.AgentSessions()
    fresh._client = platform
    monkeypatch.setattr(module, "sessions", fresh)
    import wsbench.server.app as app_module
    monkeypatch.setattr(app_module, "agent_sessions", fresh)
    before = len(platform.sent)
    assert fresh.get(conv_id)["status"] == "idle"
    assert client.post(f"/api/agent/{conv_id}/message", json={"text": "welcome back"}).json() == {"ok": True}
    record = _wait_agent(client, conv_id)
    assert record["status"] == "idle" and {"kind": "agent", "text": "said while away"} in record["items"]
    sent = platform.sent[before:]
    assert [e["type"] for e in sent] == ["user.custom_tool_result", "user.message"]       # settle first, then the message
    assert sent[0]["custom_tool_use_id"] == "sevt_91" and "did not run" in sent[0]["content"][0]["text"]
    assert_error(client.post("/api/agent/" + "0" * 32 + "/message", json={"text": "x"}), 404, "E_NOT_FOUND")


def test_stop_during_a_tool_call_ends_the_turn_and_interrupts_again(client, monkeypatch):
    from types import SimpleNamespace as NS
    from wsbench.agent import sessions as module
    module_, platform = _platform_client(client, monkeypatch, [[
        _ev("agent.custom_tool_use", 1, name="wsl_check", input={"script": "model neox-tiny\n"}),
        _ev("session.status_idle", 2, stop_reason=NS(type="requires_action")),
        _ev("session.status_idle", 3, stop_reason=NS(type="end_turn")),
    ]])
    monkeypatch.setattr(module, "_is_bad_request", lambda exc: isinstance(exc, ValueError))
    # the platform closed the pending call when the interrupt arrived and now refuses the late result
    platform.send_fail = lambda events: (_ for _ in ()).throw(ValueError("no pending call")) if events[0]["type"] == "user.custom_tool_result" else None
    original = module.agent_tools.execute

    def stopping(name, arguments, jobs, should_stop, on_run):
        module.sessions._control[list(module.sessions._control)[-1]]["stop"].set()      # Stop pressed during the tool (newest conversation)
        return original(name, arguments, jobs, should_stop, on_run)
    monkeypatch.setattr(module.agent_tools, "execute", stopping)
    conv_id = client.post("/api/agent/start", json={"text": "idea"}).json()["agent_conv_id"]
    record = _wait_agent(client, conv_id)
    assert record["status"] == "idle", record["error"]
    assert not any(i["kind"] == "note" and "reconnecting" in i["text"] for i in record["items"])
    assert [e["type"] for e in platform.sent].count("user.interrupt") == 1


def test_sending_events_retries_overload_and_stored_ids_follow_the_key(client, tmp_path, monkeypatch):
    from wsbench.agent import sessions as module
    monkeypatch.setattr(module, "_sleep", lambda seconds: None)
    monkeypatch.setattr(module, "_transient", lambda exc: isinstance(exc, ConnectionResetError))
    platform = _FakePlatform([[]])
    attempts = []
    platform.send_fail = lambda events: attempts.append(1) or (_ for _ in ()).throw(ConnectionResetError()) if len(attempts) < 3 else None
    sessions = module.AgentSessions()
    sessions._client = platform
    sessions._send("sesn_1", [{"type": "user.interrupt"}])
    assert len(attempts) == 3 and platform.sent == [{"type": "user.interrupt"}]
    platform.send_fail = lambda events: (_ for _ in ()).throw(ConnectionResetError())
    with pytest.raises(ConnectionResetError):
        sessions._send("sesn_1", [{"type": "user.interrupt"}])

    # live mode: IDs are stored with a fingerprint of the key and not trusted for another key
    monkeypatch.delenv("WSB_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-one")
    platform = _FakePlatform([[]])
    sessions = module.AgentSessions()
    sessions._client = platform
    ids = sessions.ensure_agent()
    assert ids["agent_id"] == "agent_1" and (tmp_path / "agent.json").exists()
    sessions.ensure_agent()
    assert platform.created["agent"] == 1 and platform.created["env"] == 1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-two")
    sessions.ensure_agent()
    assert platform.created["agent"] == 2 and platform.created["env"] == 2

    # an agent or environment that is gone: the session create is retried once with fresh resources
    monkeypatch.setattr(module, "_gone", lambda exc: isinstance(exc, LookupError))
    real_create, tries = platform.beta.sessions.create, []

    def create(**kw):
        tries.append(kw)
        if len(tries) == 1:
            raise LookupError("agent not found")
        return real_create(**kw)
    platform.beta.sessions.create = create
    assert sessions._create_session("idea", 1.0).id == "sesn_1" and len(tries) == 2 and platform.created["agent"] == 3

    # mock mode never reads or writes the stored IDs
    (tmp_path / "agent.json").unlink()
    monkeypatch.setenv("WSB_LLM", "mock")
    module.AgentSessions().ensure_agent()
    assert not (tmp_path / "agent.json").exists()
    monkeypatch.setenv("WSB_ANTHROPIC_WORKSPACE", "wrkspc_01")
    assert module.console_url("sesn_9") == "https://platform.claude.com/workspaces/wrkspc_01/sessions/sesn_9"


@pytest.mark.skipif(shutil.which("node") is None, reason="runs two functions of app.js in node")
def test_tensor_panel_inserts_a_valid_gridop_for_every_matrix():
    """'Insert gridop(...)' in the tensor panel: the default block is square, so the rot90 it writes validates for every
    2-D parameter of every weight model (some Transformer-XL matrices have a side below 16); a non-square block gets
    rot180, which fits any block."""
    from wsbench.lang.parser import parse
    from wsbench.lang.validate import validate
    from wsbench.models.registry import SHAPES, WEIGHT_MODELS, param_shapes

    source = (Path(__file__).parents[1] / "wsbench" / "server" / "static" / "app.js").read_text(encoding="utf-8")
    functions = [re.search(rf"^function {name}\(.*?^}}$", source, re.M | re.S)
                 for name in ("defaultBlock", "gridopStatement")]
    matrices = {model: {name: shape for name, shape in param_shapes(SHAPES[model]).items() if len(shape) == 2}
                for model in WEIGHT_MODELS}
    program = "\n".join(match.group(0) for match in functions) + f"""
const matrices = {json_dumps(matrices)};
const out = {{nonsquare: gridopStatement('crit.cluster_weight', [0, 3, 0, 16])}};
for (const [model, tensors] of Object.entries(matrices))
  out[model] = Object.entries(tensors).map(([name, shape]) => gridopStatement(name, defaultBlock(shape)));
process.stdout.write(JSON.stringify(out));
"""
    out = json_loads(subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True).stdout)
    assert out.pop("nonsquare") == 'edit gridop(rot180, block("crit.cluster_weight", 0, 3, 0, 16))'
    assert 'edit gridop(rot90, block("crit.cluster_weight", 0, 3, 0, 3))' in out["txl-tiny"]
    assert 'edit gridop(rot90, block("gpt_neox.embed_in.weight", 0, 16, 0, 16))' in out["pythia-14m"]
    for model, statements in out.items():
        assert len(statements) == len(matrices[model])
        parsed, errors = parse(f"model {model}\n" + "\n".join(statements))
        assert errors == validate(parsed) == [], model
