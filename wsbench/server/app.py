"""Health and run lifecycle endpoints."""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from wsbench import __version__
from wsbench.exec.jobs import JobError, jobs
from wsbench.models.fetch import pythia_local
from wsbench.paths import home


app = FastAPI(title="WSBench", version=__version__, docs_url=None, redoc_url=None)


class Script(BaseModel):
    script: str


@app.exception_handler(JobError)
async def job_error(request, exc):
    return JSONResponse({"error": {"code": exc.code, "message": str(exc)}}, status_code=exc.status_code)


@app.get("/api/health")
def health():
    golf = Path(os.environ.get("WSB_GOLF_DIR", home() / "golf" / "google-code-golf-2025"))
    llm = "mock" if os.getenv("WSB_LLM") == "mock" else "live" if os.getenv("OPENROUTER_API_KEY") else "no-key"
    return {"ok": True, "version": __version__, "llm": llm,
            "models_local": {"pythia-14m": pythia_local()}, "arc_local": any(golf.glob("task[0-9][0-9][0-9].json"))}


@app.post("/api/run")
def run(payload: Script):
    return {"run_id": jobs.submit(payload.script)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return jobs.snapshot(run_id)


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str):
    return jobs.cancel(run_id)
