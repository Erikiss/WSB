"""HTTP entry points for the local WSBench application."""
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from wsbench import __version__
from wsbench.arc.data import ArcError, arc_local, compiled_preview, list_tasks, task_detail
from wsbench.arc.render import tensor_view
from wsbench.assist.pipeline import AssistRequestError, assist
from wsbench.assist.tasks import tasks as assist_tasks
from wsbench.exec.jobs import JobError, jobs
from wsbench.lang.card import LANGUAGE_CARD
from wsbench.lang.hints import hints
from wsbench.lang.parser import parse
from wsbench.lang.printer import print_statement
from wsbench.lang.validate import validate
from wsbench.models.fetch import pythia_local
from ._catalog import OPS

app = FastAPI(title="WSBench", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
STATIC = Path(__file__).parent / "static"
EXAMPLES = Path(__file__).parents[1] / "examples"


class Script(BaseModel):
    script: str
    source: str = "editor"


def error(status, code, message):
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


@app.exception_handler(JobError)
@app.exception_handler(ArcError)
@app.exception_handler(AssistRequestError)
async def domain_error(request, exc):
    return error(getattr(exc, "status_code", 400), exc.code, str(exc))


@app.exception_handler(RequestValidationError)
async def bad_request(request, exc):
    return error(400, "E_BAD_REQUEST", "; ".join(e["msg"] for e in exc.errors()))


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return error(exc.status_code, "E_NOT_FOUND" if exc.status_code == 404 else "E_BAD_REQUEST", str(exc.detail))


@app.get("/api/health")
def health():
    llm = "mock" if os.getenv("WSB_LLM") == "mock" else "live" if os.getenv("OPENROUTER_API_KEY") else "no-key"
    return {"ok": True, "version": __version__, "llm": llm,
            "models_local": {"pythia-14m": pythia_local()}, "arc_local": arc_local()}


@app.get("/api/language")
def language():
    names = ("rotate_middle_layer", "antisymmetric_pair", "grokking_steerable", "arc_mirror")
    return {"card": LANGUAGE_CARD, "ops": OPS,
            "examples": [{"name": name + ".wsl", "script": (EXAMPLES / (name + ".wsl")).read_text("utf-8")}
                         for name in names]}


@app.post("/api/parse")
def check(payload: Script):
    statements, errors = parse(payload.script)
    return {"statements": [dict(line=s.line, label=s.display_label, claim=s.claim,
                                source=print_statement(s, with_label=False), family=s.family) for s in statements],
            "errors": sorted(errors + validate(statements), key=lambda e: (e["line"], e["col"])),
            "hints": hints(statements)}


@app.post("/api/run")
def run(payload: Script):
    source = payload.source if payload.source in ("editor", "arc") else "editor"
    return {"run_id": jobs.submit(payload.script, source=source)}


@app.get("/api/runs")
def recent_runs():
    return jobs.recent()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return jobs.snapshot(run_id)


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str):
    return jobs.cancel(run_id)


@app.get("/api/tensor")
def tensor(run_id: str, line: int, name: str, which: str, max_size: int = Query(256, alias="max"),
           view: str = "heatmap", r0: int | None = None, r1: int | None = None,
           c0: int | None = None, c1: int | None = None):
    before, after = jobs.tensors(run_id, line, name)
    return tensor_view(name, before, after, which, max_size, view, r0, r1, c0, c1)


@app.get("/api/arc/tasks")
def arc_tasks():
    return list_tasks()


@app.get("/api/arc/task/{task}")
def arc_task(task: str):
    return task_detail(task)


@app.get("/api/arc/compiled")
def arc_compiled(task: str, prim: str, size: str = "fixed", source: str = Query("all", alias="from"),
                 shape: str | None = None):
    return compiled_preview(task, prim, size, source, shape)


@app.post("/api/assist")
def assist_request(payload: dict):
    return assist(payload, _assist_submit)


def _assist_submit(script, predictions=None):
    return jobs.submit(script, predictions, source="assist")


@app.post("/api/assist/start")
def assist_start(payload: dict):
    return {"assist_id": assist_tasks.start(payload, _assist_submit)}


@app.get("/api/assist/latest")
def assist_latest():
    return {"assist": assist_tasks.latest()}


@app.get("/api/assist/{assist_id}")
def assist_status(assist_id: str):
    record = assist_tasks.get(assist_id)
    if record is None:
        return error(404, "E_NOT_FOUND", "unknown assist request")
    return record


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
