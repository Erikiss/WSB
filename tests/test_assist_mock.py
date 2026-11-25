import json
import os
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from wsbench.assist import mock, openrouter
from wsbench.assist.pipeline import assist
from wsbench.assist.prompts import TRANSLATE_SYSTEM
from wsbench.lang.card import LANGUAGE_CARD


@pytest.fixture
def mock_client(monkeypatch):
    from wsbench.server.app import app

    monkeypatch.setenv("WSB_LLM", "mock")
    with TestClient(app) as client:
        yield client


def test_assist_end_to_end(mock_client):
    response = mock_client.post("/api/assist", json={"text": mock.SOURCE_TEXT, "model": "neox-tiny"})
    assert response.status_code == 200
    data = response.json()
    assert data["cost_usd"] == 0 and data["parse_errors"] == []
    assert all(r["ok"] for r in data["roles"].values())
    assert data["roles"]["translate"]["text"] == mock.TRANSLATE
    assert data["script"].splitlines() == [
        "model neox-tiny",
        "t1: gauge fold_ln(); center_writers(); ln_to_rms(); rotate(resid, orth)",
        "t2: gauge rotate(resid, orth)",
        "t3: edit rotate(resid, orth, layers=[2], embed=false, unembed=false)",
        "fix2: gauge fold_ln(); rotate(resid, orth1)",
        "alt1: gauge fold_ln(); rotate(resid, orth1)",
    ]
    assert data["predictions"] == {"t1": "exact", "t2": "edit", "t3": "edit"}
    deadline = time.monotonic() + 45
    while True:
        run = mock_client.get("/api/runs/" + data["run_id"]).json()
        if run["status"] == "done":
            break
        assert time.monotonic() < deadline, run
        time.sleep(0.02)
    assert [(r["label"], r["status"], r["measured"], r["predicted"], r["agree"])
            for r in run["verdicts"]] == [
        ("t1", "CONFIRMED", "exact", "exact", "agree"),
        ("t2", "REFUTED", "edit", "edit", "agree"),
        ("t3", "MEASURED", "edit", "edit", "agree"),
        ("fix2", "CONFIRMED", "exact", None, "n/a"),
        ("alt1", "CONFIRMED", "exact", None, "n/a"),
    ]


def test_single_repair(mock_client, monkeypatch):
    calls = []
    real_chat = openrouter.chat

    def observed(role, model_id, system, messages):
        calls.append((role, system, messages))
        return {**real_chat(role, model_id, system, messages), "cost_usd": 0.25}

    monkeypatch.setattr(openrouter, "chat", observed)
    text = mock.SOURCE_TEXT + " [[mock-syntax-error]]"
    data = mock_client.post("/api/assist", json={"text": text, "model": "neox-tiny",
                                                 "execute": False}).json()
    assert [role for role, _, _ in calls] == ["translate", "translate", "critic", "alternatives"]
    assert calls[0][1] == calls[1][1]
    assert calls[0][2] == [{"role": "user", "content": f"Source text:\n<<<\n{text}\n>>>"}]
    repair = calls[1][2]
    assert repair[:2] == [*calls[0][2], {"role": "assistant", "content": mock.BROKEN_TRANSLATE}]
    assert repair[2]["content"].startswith("Your WSL did not parse. Errors:\nline 3 col ")
    assert repair[2]["content"].endswith("Return the corrected wsl block only.")
    assert data["parse_errors"] == [] and data["cost_usd"] == 1
    assert data["roles"]["translate"]["text"] == mock.TRANSLATE
    assert data["run_id"] is None


def test_no_key_and_invalid_target(mock_client, monkeypatch):
    monkeypatch.delenv("WSB_LLM")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    data = mock_client.post("/api/assist", json={"text": "x", "model": "grid:task150"}).json()
    assert data["roles"]["translate"]["error"] == "no-key"
    assert data["roles"]["critic"] is data["roles"]["alternatives"] is None
    assert data["script"] == "model grid:task150" and data["run_id"] is None
    assert data["parse_errors"][0]["role"] == "translate"
    invalid = mock_client.post("/api/assist", json={"text": "x", "model": "modadd"})
    assert invalid.status_code == 400 and invalid.json()["error"]["code"] == "E_BAD_REQUEST"


def test_keeps_valid_lines_and_ignores_malformed_critic(monkeypatch):
    replies = dict(translate="```wsl\nmodel grid:task150\ngauge fold_ln()\ngauge bad()\n```",
                   critic="```json\n{}\n```", alternatives="")

    def chat(role, model_id, system, messages):
        return dict(ok=role != "alternatives", text=replies[role], cost_usd=0, model=model_id,
                    error="refused" if role == "alternatives" else None)

    monkeypatch.setattr(openrouter, "chat", chat)
    result = assist({"text": "", "model": "neox-tiny", "execute": False}, None)
    assert result["script"] == "model neox-tiny\nt1: gauge fold_ln()"
    assert [(err["role"], err["line"], err["code"]) for err in result["parse_errors"]] == [
        ("translate", 4, "E_UNKNOWN_OP"), ("critic", None, "E_SYNTAX")]
    assert result["predictions"] == {} and result["roles"]["alternatives"]["error"] == "refused"


def test_client_request_and_fallback_price(monkeypatch):
    monkeypatch.delenv("WSB_LLM", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")

    def post(url, *, headers, json, timeout):
        assert url == openrouter.URL and timeout == 180
        assert headers["Authorization"] == "Bearer test-only" and headers["X-Title"] == "wsbench"
        assert json == {
            "model": openrouter.DEFAULT_ROLES["critic"], "messages": [
                {"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "max_tokens": 6000, "reasoning": {"effort": "low", "exclude": True},
            "usage": {"include": True},
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 30}})

    monkeypatch.setattr(httpx, "post", post)
    result = openrouter.chat("critic", openrouter.DEFAULT_ROLES["critic"], "s", [{"role": "user", "content": "u"}])
    assert result["ok"] and result["cost_usd"] == pytest.approx(0.0005)


def test_invalid_provider_cost(monkeypatch):
    monkeypatch.delenv("WSB_LLM", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    response = httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}],
                                         "usage": {"cost": "NaN"}})
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
    result = openrouter.chat("translate", "custom", "system", [])
    assert result["ok"] is False and result["cost_usd"] == 0
    assert "non-finite usage cost" in result["error"]


def test_model_scope_applies_to_every_role(monkeypatch):
    fence = "`" * 3
    wsl = "model modadd\ntrain modadd(run=OtherModel)\nedit delete(L0)"
    critic = [{"n": 9}, {"n": 1, "predict": "unsupported"},
              {"n": 1, "predict": "edit", "reason": "", "fix": "model modadd\ntrain modadd(run=OtherModel)"}]
    replies = {"translate": f"{fence}wsl\n{wsl}\n{fence}", "critic": f"{fence}json\n{json.dumps(critic)}\n{fence}",
               "alternatives": f"{fence}wsl\n{wsl}\n{fence}"}
    monkeypatch.setattr(openrouter, "chat", lambda role, *args: dict(ok=True, text=replies[role], cost_usd=0,
                                                                    error=None))
    result = assist({"text": "", "model": "neox-tiny", "execute": False}, None)
    assert result["script"].splitlines() == ["model neox-tiny", "t1: edit delete(L[0])", "alt1: edit delete(L[0])"]
    assert result["predictions"] == {"t1": "edit"}
    assert [(err["role"], err["code"]) for err in result["parse_errors"]] == [
        ("translate", "E_MODEL"), ("critic", "E_MODEL"), ("alternatives", "E_MODEL")]


@pytest.mark.live
@pytest.mark.parametrize("model_id", openrouter.DEFAULT_ROLES.values())
def test_live_translation(model_id, monkeypatch):
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("requires OPENROUTER_API_KEY")
    monkeypatch.delenv("WSB_LLM", raising=False)
    system = TRANSLATE_SYSTEM.replace("{LANGUAGE_CARD}", LANGUAGE_CARD).replace("{MODEL}", "pythia-14m")
    result = openrouter.chat("translate", model_id, system, [{
        "role": "user", "content": f"Source text:\n<<<\n{mock.SOURCE_TEXT}\n>>>",
    }])
    if result["ok"]:
        outcome = "completion"
    elif result["finish_reason"] == "content_filter":
        outcome = "content_filter"
    else:
        # Keep provider response bodies and model output out of public CI logs.
        prefix = (result["error"] or "").partition(":")[0]
        http_code = prefix.removeprefix("HTTP ")
        outcome = f"HTTP {http_code}" if http_code.isdigit() else "no usable response"
    print(f"{model_id} | {outcome} | cost USD {result['cost_usd']:.6f}")
    if outcome not in {"completion", "content_filter"}:
        pytest.fail(f"{model_id}: {outcome}", pytrace=False)
