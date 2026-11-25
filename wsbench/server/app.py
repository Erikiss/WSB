import os

from fastapi import FastAPI

from wsbench import __version__
from wsbench.paths import home

app = FastAPI(title="WSBench", version=__version__, docs_url=None, redoc_url=None)


@app.get("/api/health")
def health():
    model_dir = home() / "models" / "pythia-14m"
    files = ("config.json", "model.safetensors", "tokenizer.json")
    golf = os.environ.get("WSB_GOLF_DIR", home() / "golf" / "google-code-golf-2025")
    from pathlib import Path
    llm = "mock" if os.getenv("WSB_LLM") == "mock" else "live" if os.getenv("OPENROUTER_API_KEY") else "no-key"
    return {"ok": True, "version": __version__, "llm": llm,
            "models_local": {"pythia-14m": all((model_dir / f).is_file() for f in files)},
            "arc_local": any(Path(golf).glob("task[0-9][0-9][0-9].json"))}
