"""Tests for the agent's tools (arXiv, GitHub, result size and text hygiene) and for the model clients' cut-off and cost handling."""
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest

from wsbench.agent import arxiv, github, tools
from wsbench.assist import anthropic_api, openrouter


# ---- model clients ----------------------------------------------------------------------------------------------------

def _usage(**kw):
    return NS(**{"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, **kw})


class _Anthropic:
    def __init__(self, stop_reason="end_turn", text="hello", model="claude-fable-5-2", usage=None):
        self.reply = NS(model=model, stop_reason=stop_reason, stop_details=None, usage=usage or _usage(),
                        content=[NS(type="thinking", thinking=""), NS(type="text", text=text)])
        self.beta = NS(messages=NS(create=lambda **kw: self.reply))


def _chat(monkeypatch, fake):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic_api, "_client", fake)
    return anthropic_api.chat("planner", "anthropic/claude-fable-5.2", "s", [{"role": "user", "content": "x"}])


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
def test_a_reply_cut_off_by_a_limit_is_an_error_not_a_result(monkeypatch, stop_reason):
    reply = _chat(monkeypatch, _Anthropic(stop_reason=stop_reason, text="model neox-tiny\nP1: gauge permute("))
    assert reply["ok"] is False and "cut off" in reply["error"] and stop_reason in reply["error"] and "incomplete" in reply["error"]
    assert reply["text"].startswith("model neox-tiny")            # the partial text stays available for diagnosis
    assert _chat(monkeypatch, _Anthropic())["ok"] is True          # a normal end of turn is still fine


def test_openrouter_reports_a_reply_cut_off_at_the_length_limit(monkeypatch):
    monkeypatch.delenv("WSB_LLM", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")

    def post(finish):
        return lambda *a, **kw: httpx.Response(200, json={"choices": [{"message": {"content": "half an answ"}, "finish_reason": finish}],
                                                          "usage": {"prompt_tokens": 10, "completion_tokens": 10}})
    monkeypatch.setattr(httpx, "post", post("length"))
    reply = openrouter.chat("critic", openrouter.DEFAULT_ROLES["critic"], "s", [{"role": "user", "content": "u"}])
    assert reply["ok"] is False and "cut off" in reply["error"] and reply["text"] == "half an answ"
    monkeypatch.setattr(httpx, "post", post("stop"))
    assert openrouter.chat("critic", openrouter.DEFAULT_ROLES["critic"], "s", [{"role": "user", "content": "u"}])["ok"] is True


def test_cost_counts_every_attempt_of_a_server_side_fallback(monkeypatch):
    declined = NS(type="message", model="claude-fable-5-2", input_tokens=1000, output_tokens=0,
                  cache_read_input_tokens=0, cache_creation_input_tokens=0)
    served = NS(type="fallback_message", model="claude-opus-4-8", input_tokens=1000, output_tokens=500,
                cache_read_input_tokens=0, cache_creation_input_tokens=0)
    compaction = NS(type="compaction", model="claude-opus-4-8", input_tokens=10 ** 6, output_tokens=0)    # not part of a plain message call
    fake = _Anthropic(model="claude-opus-4-8", usage=_usage(iterations=[declined, served, compaction]))
    reply = _chat(monkeypatch, fake)
    assert reply["ok"] and reply["served_by"] == "claude-opus-4-8"
    assert reply["cost_usd"] == pytest.approx((1000 * 8 + (1000 * 5 + 500 * 25)) / 1e6)
    assert (reply["prompt_tokens"], reply["completion_tokens"]) == (1000, 500)          # tokens stay those of the answer

    # without iterations (no fallback happened) the top-level usage is priced as before, and unknown models get the default price
    assert _chat(monkeypatch, _Anthropic())["cost_usd"] == pytest.approx((1000 * 8 + 500 * 40) / 1e6)
    unknown = _usage(iterations=[NS(type="message", model="claude-future-9", input_tokens=1000, output_tokens=500,
                                    cache_read_input_tokens=0, cache_creation_input_tokens=0)])
    assert _chat(monkeypatch, _Anthropic(usage=unknown))["cost_usd"] == pytest.approx((1000 * 8 + 500 * 40) / 1e6)


def test_cost_reads_the_real_sdk_usage_types(monkeypatch):
    beta = pytest.importorskip("anthropic.types.beta")
    hop = {"cache_creation": None, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    usage = beta.BetaUsage.model_validate({
        "input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "cache_creation": None, "inference_geo": None, "server_tool_use": None, "service_tier": None, "speed": None,
        "fallback_credit": None, "output_tokens_details": None,
        "iterations": [{"type": "message", "model": "claude-fable-5-2", "input_tokens": 1000, "output_tokens": 0, **hop},
                       {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1000, "output_tokens": 500, **hop}]})
    reply = _chat(monkeypatch, _Anthropic(model="claude-opus-4-8", usage=usage))
    assert reply["cost_usd"] == pytest.approx((1000 * 8 + 1000 * 5 + 500 * 25) / 1e6)


# ---- arXiv: a failed download must not be remembered --------------------------------------------------------------------

ATOM = ("<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>A Test Paper</title><summary>Short abstract.</summary>"
        "<published>2026-11-23T15:18:30Z</published><author><name>A. Author</name></author></entry></feed>")
PAGE = "<html><body><h2>1 Intro</h2><p>Hello world.</p></body></html>"


def _status_error(code):
    request = httpx.Request("GET", "https://arxiv.org/html/2611.30835")
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=httpx.Response(code, request=request))


class _Arxiv:
    """A `get` that answers the metadata request and plays a script for the HTML request."""

    def __init__(self, html):
        self.html, self.calls = html, []

    def __call__(self, url):
        self.calls.append(url)
        if "export.arxiv.org" in url:
            return ATOM
        outcome = self.html if not isinstance(self.html, list) else self.html.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def html_calls(self):
        return sum("arxiv.org/html" in url for url in self.calls)


def _cache(home, arxiv_id="2611.30835"):
    return Path(home) / "papers" / f"{arxiv_id}.json"


@pytest.mark.parametrize("failure", [_status_error(503), _status_error(429), httpx.ConnectTimeout("slow")])
def test_an_arxiv_outage_is_not_cached(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    get = _Arxiv([failure, PAGE])
    first = arxiv.read("2611.30835", 0, get)
    assert first["text_parts"] == 0 and "try again later" in first["note"].lower()
    assert not _cache(tmp_path).exists()
    second = arxiv.read("2611.30835", 0, get)                       # arXiv is back: the text arrives
    assert second["text_parts"] == 1 and second["headings"] == ["1 Intro"]
    assert get.html_calls() == 2 and json.loads(_cache(tmp_path).read_text())["html"] == "ok"
    assert arxiv.read("2611.30835", 1, get)["text"].startswith("1 Intro\nHello world.") and get.html_calls() == 2


def test_a_paper_without_html_is_asked_again_after_six_hours(tmp_path, monkeypatch):
    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    get = _Arxiv([_status_error(404), PAGE])
    first = arxiv.read("2611.30835", 0, get)
    assert first["text_parts"] == 0 and "No HTML full text is available" in first["note"]
    stored = json.loads(_cache(tmp_path).read_text())
    assert stored["html"] == "missing" and abs(stored["fetched"] - time.time()) < 60
    arxiv.read("2611.30835", 0, get)
    assert get.html_calls() == 1                                    # still inside the window: served from the cache
    stored["fetched"] -= arxiv.MISSING_TTL_S + 1
    _cache(tmp_path).write_text(json.dumps(stored))
    assert arxiv.read("2611.30835", 0, get)["text_parts"] == 1 and get.html_calls() == 2    # the HTML version has appeared


def test_poisoned_and_empty_cache_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("WSB_HOME", str(tmp_path))
    # entry written by the old code during an outage: no text and no record of why
    _cache(tmp_path).parent.mkdir(parents=True)
    _cache(tmp_path).write_text(json.dumps(dict(id="2611.30835", title="A Test Paper", abstract="x", authors=[], published="", text="", headings=[])))
    get = _Arxiv([PAGE])
    assert arxiv.read("2611.30835", 0, get)["text_parts"] == 1 and get.html_calls() == 1
    # a page that was fetched fine but has no visible text is final too: it must not be fetched on every call
    get = _Arxiv(["<html><body><script>x()</script></body></html>"])
    assert arxiv.read("2611.30836", 0, get)["text_parts"] == 0 and arxiv.read("2611.30836", 0, get)["text_parts"] == 0
    assert get.html_calls() == 1
    # an old entry that has text stays valid
    _cache(tmp_path, "2611.30837").write_text(json.dumps(dict(id="2611.30837", title="T", abstract="a", authors=[], published="", text="body", headings=[])))
    get = _Arxiv([])
    assert arxiv.read("2611.30837", 1, get)["text"] == "body" and get.calls == []


def test_a_text_part_leaves_room_for_the_json_escaping():
    assert arxiv.PART_CHARS == 12_000
    heavy = ("a \"quoted\"\tword\n" * 800)[:arxiv.PART_CHARS]              # quotes, tabs and newlines all double in JSON
    out = tools._json(dict(id="1", part=1, text_parts=3, text=heavy))
    assert len(out) < tools.MAX_RESULT_CHARS and json.loads(out)["text"] == heavy


# ---- GitHub -----------------------------------------------------------------------------------------------------------

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]


def _make_remote(root, owner="owner", name="demo", files=None):
    source = Path(root) / owner / name
    source.mkdir(parents=True, exist_ok=True)
    for relative, content in (files or {"README.md": "# Demo\n", "pkg/core.py": "def step(x):\n    return x + 1  # Hebbian update\n"}).items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    subprocess.run([*GIT, "-C", str(source), "init", "-q"], check=True)
    subprocess.run([*GIT, "-C", str(source), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(source), "commit", "-q", "-m", "init"], check=True)
    return str(root) + "/"


@pytest.fixture
def repo_home(tmp_path, monkeypatch):
    monkeypatch.setenv("WSB_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_a_killed_clone_leaves_no_checkout_that_looks_finished(repo_home, tmp_path, monkeypatch):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    held = []
    threading.Thread(target=lambda: [held.append(server.accept()[0]) for _ in range(3)], daemon=True).start()
    monkeypatch.setattr(github, "CLONE_TIMEOUT_S", 2)
    started = time.monotonic()
    with pytest.raises(ValueError) as caught:
        github.read("owner/demo", base=f"http://127.0.0.1:{server.getsockname()[1]}/")
    server.close()
    assert time.monotonic() - started < 30
    assert str(caught.value) == "clone timed out after 2 s; try again or ask for a specific folder"
    assert str(repo_home) not in str(caught.value)
    assert list((repo_home / "repos").iterdir()) == []              # neither the target nor the temporary clone is left

    base = _make_remote(tmp_path / "remote")                         # the remote works now: the next call gets a real checkout
    overview = github.read("owner/demo", base=base)
    assert overview["files_below"] == 2 and overview["commit"]
    assert "Hebbian update" in github.read("owner/demo", "pkg/core.py", base=base)["text"]


def test_debris_of_an_earlier_killed_clone_is_replaced(repo_home, tmp_path):
    debris = repo_home / "repos" / "owner__demo"
    (debris / ".git").mkdir(parents=True)                           # what the old code left behind after a timeout
    (repo_home / "repos" / "owner__demo.tmp-deadbeef").mkdir()
    old = time.time() - 10_000
    os.utime(repo_home / "repos" / "owner__demo.tmp-deadbeef", (old, old))
    base = _make_remote(tmp_path / "remote")
    assert github.read("owner/demo", base=base)["files_below"] == 2
    assert sorted(p.name for p in (repo_home / "repos").iterdir()) == ["owner__demo"]


def test_a_folder_inside_another_repository_is_not_taken_for_a_checkout(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run([*GIT, "-C", str(outer), "init", "-q"], check=True)
    (outer / "x").write_text("x")
    subprocess.run([*GIT, "-C", str(outer), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(outer), "commit", "-q", "-m", "x"], check=True)
    monkeypatch.setenv("WSB_HOME", str(outer / "wsb"))               # WSB_HOME inside a git repository
    (outer / "wsb" / "repos" / "owner__demo").mkdir(parents=True)    # no .git of its own
    assert github._complete(outer / "wsb" / "repos" / "owner__demo") is False
    assert github.read("owner/demo", base=_make_remote(tmp_path / "remote"))["files_below"] == 2


def test_two_callers_clone_a_repository_once_and_case_does_not_matter(repo_home, tmp_path, monkeypatch):
    base = _make_remote(tmp_path / "remote")
    clones, real_run = [], subprocess.run

    def counting(command, *a, **kw):
        if command[:2] == ["git", "clone"]:
            clones.append(command)
            time.sleep(0.4)                                          # a slow clone: the second caller arrives meanwhile
        return real_run(command, *a, **kw)
    monkeypatch.setattr(github.subprocess, "run", counting)
    results = {}

    def read(label, reference):
        results[label] = github.read(reference, base=base)
    threads = [threading.Thread(target=read, args=(label, ref)) for label, ref in (("a", "owner/demo"), ("b", "owner/demo"), ("c", "Owner/Demo"))]
    for t in threads:
        t.start()
        time.sleep(0.1)
    for t in threads:
        t.join(60)
    assert len(clones) == 1
    assert all(r["files_below"] == 2 and r["commit"] for r in results.values()) and len(results) == 3
    assert [p.name for p in (repo_home / "repos").iterdir()] == ["owner__demo"]


def test_a_failed_clone_names_the_repository_not_the_server_paths(repo_home, tmp_path):
    with pytest.raises(ValueError, match="could not clone owner/missing") as caught:
        github.read("owner/missing", base=str(tmp_path / "nowhere") + "/")
    assert str(repo_home) not in str(caught.value)
    assert list((repo_home / "repos").iterdir()) == []


@pytest.fixture
def demo(repo_home, tmp_path):
    return _make_remote(tmp_path / "remote", files={
        "README.md": "# Demo\nA predictive coding toy.\n", "pkg/core.py": "def step(x):\n    return x + 1  # NEEDLE here\n",
        "pkg/deep/more.py": "NEEDLE again\n", "top.py": "nothing\n"})


@pytest.mark.parametrize("path", ["pkg", "pkg/", "./pkg", "/pkg", "./pkg/", "pkg/./deep/..", "//pkg"])
def test_paths_are_normalised_for_listing_and_search(demo, path):
    listing = github.read("owner/demo", path, base=demo)
    assert listing["path"] == "pkg" and listing["entries"] == ["core.py", "deep/ (1 files)"] and listing["files_below"] == 2
    assert github.read("owner/demo", path, search="needle", base=demo)["matches"] == [
        "pkg/core.py:2: return x + 1  # NEEDLE here", "pkg/deep/more.py:1: NEEDLE again"]


@pytest.mark.parametrize("path", ["", ".", "./", "/", "pkg/.."])
def test_the_root_can_be_named_in_several_ways(demo, path):
    listing = github.read("owner/demo", path, base=demo)
    assert listing["path"] == "/" and listing["files_below"] == 4 and listing["commit"] and listing["readme"].startswith("# Demo")
    assert len(github.read("owner/demo", path, search="needle", base=demo)["matches"]) == 2


def test_search_inside_one_file_and_empty_results_say_what_was_searched(demo):
    assert github.read("owner/demo", "pkg/core.py", search="needle", base=demo)["matches"] == ["pkg/core.py:2: return x + 1  # NEEDLE here"]
    empty = github.read("owner/demo", "pkg", search="absent", base=demo)
    assert empty["matches"] == [] and empty["note"] == "no match in the 2 files searched below pkg"
    assert github.read("owner/demo", search="absent", base=demo)["note"] == "no match in the 4 files searched"
    assert github.read("owner/demo", "./pkg/core.py", base=demo)["path"] == "pkg/core.py"          # reading accepts './' as before


@pytest.mark.parametrize("path", ["PKG", "./nothing", "pkg/missing.py", "pkg/core.py/x"])
def test_an_unknown_path_is_an_error_for_listing_reading_and_search(demo, path):
    for kwargs in ({}, {"search": "needle"}):
        with pytest.raises(ValueError, match="no such path"):
            github.read("owner/demo", path, base=demo, **kwargs)


def test_paths_that_leave_the_repository_are_still_refused(demo):
    for path, message in (("../..", "leaves the repository"), ("pkg/../../x", "leaves the repository"), (".git", "not readable"),
                          ("./.git/config", "not readable")):
        with pytest.raises(ValueError, match=message):
            github.read("owner/demo", path, search="x", base=demo)


# ---- text that cannot be sent -----------------------------------------------------------------------------------------

def test_a_file_name_that_is_not_utf8_cannot_reach_the_platform(repo_home, tmp_path, monkeypatch):
    base = _make_remote(tmp_path / "remote", files={"README.md": "x\n", "ok.py": "NEEDLE\n"})
    source = tmp_path / "remote" / "owner" / "demo"
    (source / os.fsdecode(b"caf\xe9.py")).write_text("NEEDLE in a latin-1 name\x1b[0m\n")       # name is not valid UTF-8
    subprocess.run([*GIT, "-C", str(source), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(source), "commit", "-q", "-m", "latin"], check=True)

    overview = github.read("owner/demo", base=base)
    assert not any("\udce9" in entry for entry in overview["entries"]) and any(entry.startswith("caf") for entry in overview["entries"])
    matches = github.read("owner/demo", search="needle", base=base)["matches"]
    assert len(matches) == 2 and all("\udce9" not in m for m in matches)
    out = tools._json(overview)
    out.encode("utf-8")                                                  # the SDK encodes with strict UTF-8
    json.loads(out)
    tools._json(github.read("owner/demo", search="needle", base=base)).encode("utf-8")
    monkeypatch.setattr(github, "BASE", base)
    result = tools.execute("github_repo", {"repo": "owner/demo", "search": "needle"}, None)
    result.encode("utf-8")
    assert "\x1b" not in result


def test_tool_results_are_cleaned(monkeypatch):
    out = tools._json({"text": "a\ud800b\x00c\x1b[31md", "nested": [{"k\udce9": "v\udcff"}]})
    out.encode("utf-8")
    assert json.loads(out) == {"text": "a?bc[31md", "nested": [{"k?": "v?"}]}


# ---- result size ------------------------------------------------------------------------------------------------------

def test_results_are_compact_json():
    assert tools._json({"a": [1, 2], "b": {"c": "d"}}) == '{"a":[1,2],"b":{"c":"d"}}'
    assert tools._json({"text": "é ü"}) == '{"text":"é ü"}'


def test_an_oversize_text_stays_valid_json_and_says_what_was_cut():
    out = tools._json({"id": "1", "part": 1, "text": "word " * 9000})
    data = json.loads(out)
    assert len(out) <= tools.MAX_RESULT_CHARS
    assert data["truncated"] is True and data["truncated_field"] == "text" and data["returned_chars"] == len(data["text"])
    assert data["id"] == "1" and data["text"].startswith("word word") and 10_000 < len(data["text"]) < 20_000


def test_an_oversize_list_loses_items_from_the_end():
    rows = [{"n": i, "label": "x" * 40} for i in range(1000)]
    data = json.loads(tools._json({"count": 1000, "tasks": rows}))
    assert data["count"] == 1000 and data["tasks"] == rows[:len(data["tasks"])]
    assert data["omitted"] == 1000 - len(data["tasks"]) > 0 and data["omitted_from"] == "tasks" and data["truncated"] is True
    assert len(tools._json({"count": 1000, "tasks": rows})) <= tools.MAX_RESULT_CHARS


def test_the_largest_field_is_cut_and_other_values_survive():
    value = {"train": [[[0] * 30 for _ in range(30)] for _ in range(3)], "solution_raw": "x" * 30_000, "bytes_raw": 30_000}
    out = tools._json(value)
    data = json.loads(out)
    assert len(out) <= tools.MAX_RESULT_CHARS and data["train"] == value["train"] and data["bytes_raw"] == 30_000
    assert data["truncated_field"] == "solution_raw" and 0 < len(data["solution_raw"]) < 30_000


def test_an_absurd_result_still_returns_valid_json():
    for value in ({"deep": {"a": ["x" * 100] * 1000}}, ["y" * 30_000], {"k" + str(i): "z" * 100 for i in range(500)}):
        out = tools._json(value)
        json.loads(out)
        assert len(out) <= tools.MAX_RESULT_CHARS


def test_an_arc_task_that_was_cut_before_now_fits_with_its_solution():
    """task054, task074, task110 and task255 of the real data are 21k-28k characters with spaces after the separators."""
    grid = lambda n: [[(i * j + n) % 10 for j in range(30)] for i in range(30)]
    detail = dict(task="task054", original_id="x", train=[{"input": grid(1), "output": grid(2)} for _ in range(3)],
                  test=[{"input": grid(3), "output": grid(4)}], n_arcgen=262, solution_raw="import numpy\n" * 100, bytes_raw=1300)
    spaced = len(json.dumps(detail, ensure_ascii=False))
    out = tools._json(detail)
    assert spaced > tools.MAX_RESULT_CHARS > len(out)
    assert json.loads(out) == detail                                  # nothing lost, including solution_raw
