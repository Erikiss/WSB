import os
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("WSB_HOME", "~/.cache/wsbench")).expanduser()
