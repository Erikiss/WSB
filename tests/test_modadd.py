import json
import time
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "wsbench" / "examples"


def test_checkpoint_lifecycle(tmp_path, monkeypatch):
    from wsbench.exec.executor import run_script
    import torch

    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    script = """model modadd
a: train modadd(run=A, p=5, steps=3, save=[1, 3])
b: path scale(ckpt(A, 1), c=[1:0.5:-0.25])
c: path add(ckpt(A, 0), diff(ckpt(A, 3), ckpt(A, 0)), t=[0:1:1])
d: path lerp(ckpt(A, 0), ckpt(A, 3), t=[0:1:1])
e: path cosine(ckpt(A, 1), ckpt(A, 1))
f: path scale(ckpt(A, 2), c=[1:1:1])
g: path scale(ckpt(B, 1), c=[1:1:1])
"""
    verdicts = run_script(script)
    assert [v["status"] for v in verdicts] == ["MEASURED"] * 5 + ["ERROR"] * 2
    assert [v["error"]["code"] for v in verdicts[5:]] == ["E_MISSING_RUN"] * 2
    train, scale, add, lerp, cosine = [row["metrics"] for row in verdicts[:5]]
    assert train["saved_steps"] == [0, 1, 3]
    assert train["curves"]["step"] == [0, 3]
    assert [pt["c"] for pt in scale["points"]] == [1.0, 0.75, 0.5]
    assert lerp["points"][0] == pytest.approx(add["points"][0])
    assert lerp["points"][-1]["test_loss"] == pytest.approx(add["points"][-1]["test_loss"])
    assert cosine["cosine"] == pytest.approx(1.0, abs=2e-5)
    assert cosine["norm_a"] == cosine["norm_b"]
    folder = tmp_path / "runs/A"
    desc = json.loads((folder / "run.json").read_text())
    assert desc["saved_steps"] == train["saved_steps"]
    checkpoint = torch.load(folder / "step1.pt", weights_only=True)
    assert all(v.dtype == torch.float32 for v in checkpoint.values())

    updated, = run_script("model modadd\ntrain modadd(run=A, p=5, steps=2)")
    assert [warning["code"] for warning in updated["warnings"]] == ["W_OVERWRITE"]
    assert sorted(p.name for p in folder.iterdir()) == ["run.json", "step0.pt", "step2.pt"]


def test_cancelled_training_keeps_prior_run(tmp_path, monkeypatch):
    from wsbench.exec.executor import run_script

    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    script = "model modadd\ntrain modadd(run=A, p=5, steps=1)"
    assert run_script(script)[0]["status"] == "MEASURED"
    folder = tmp_path / "runs/A"
    old_files = {p.name: p.read_bytes() for p in folder.iterdir()}
    stop = False

    def on_update(rows):
        nonlocal stop
        if rows[0]["metrics"].get("curves", {}).get("step"):
            stop = True

    results = run_script(script + "\ntrain modadd(run=B, p=5, steps=1)",
                         on_update=on_update, should_stop=lambda: stop)
    assert [row["error"] for row in results] == [{"code": "E_RUNTIME", "message": "cancelled"}] * 2
    assert results[0]["metrics"]["curves"]["step"] == [0]
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == old_files
    assert not (tmp_path / "runs/B").exists()


@pytest.mark.slow
def test_grokking_example(tmp_path, monkeypatch):
    from wsbench.exec.executor import run_script

    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    results = run_script((EXAMPLES / "grokking_steerable.wsl").read_text())
    assert all(row["status"] == "MEASURED" for row in results), results
    metrics = {row["label"]: row["metrics"] for row in results}
    base, filtered = metrics["G1"], metrics["G2"]
    assert base["first"]["train_acc>=0.99"] <= 300
    assert 800 <= base["first"]["test_acc>=0.9"] <= 2500
    assert base["curves"]["test_acc"][-1] >= 0.99
    assert filtered["first"]["test_acc>=0.9"] <= 2500
    speedup = base["first"]["test_acc>=0.9"] / filtered["first"]["test_acc>=0.9"]
    print(f"Grokfast speed-up {speedup:.4f}x")
    line = {pt["t"]: pt for pt in metrics["P1"]["points"]}
    assert line[1.0]["test_acc"] >= 0.99
    assert line[0.25]["train_acc"] < 0.95
    assert 0.4 <= line[0]["test_acc"] <= 0.6
    transplant = {pt["t"]: pt for pt in metrics["P2"]["points"]}
    assert transplant[1.0]["test_acc"] < 0.2
    assert abs(metrics["C1"]["cosine"]) < 0.05


@pytest.mark.slow
def test_cancel_running_training_over_http(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from wsbench.server.app import app

    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    with TestClient(app) as client:
        run_id = client.post("/api/run", json={
            "script": "model modadd\ntrain modadd(run=running, p=5, steps=50000)"
        }).json()["run_id"]
        path = f"/api/runs/{run_id}"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            row = client.get(path).json()["verdicts"][0]
            steps = row["metrics"].get("curves", {}).get("step", [])
            if row["status"] == "RUNNING" and len(steps) >= 2:
                break
            time.sleep(0.025)
        else:
            client.post(path + "/cancel")
            pytest.fail(f"Training did not publish two evaluations: {row}")
        assert client.post(path + "/cancel").json() == {"ok": True}
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snapshot = client.get(path).json()
            if snapshot["status"] == "cancelled":
                break
            time.sleep(0.01)
        else:
            pytest.fail("Training did not stop within two seconds")
    row = snapshot["verdicts"][0]
    assert row["status"] == "ERROR" and row["error"]["message"] == "cancelled"
    assert len(row["metrics"]["curves"]["step"]) >= 2
    assert not (tmp_path / "runs" / "running").exists()
