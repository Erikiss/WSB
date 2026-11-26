"""Let the agent read the code of a public GitHub repository.

The repository is cloned once (shallow) into WSB_HOME/repos/ and then only read as text: listed, paged and searched.
Nothing from a repository is ever executed or imported.
"""
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

from wsbench.paths import home

from .arxiv import PART_CHARS

REPO = re.compile(r"(?:https?://)?(?:www\.)?(?:github\.com/)?([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9._-]{1,100}?)(?:\.git)?(?:/.*)?")
BASE = "https://github.com/"
MAX_FILE_BYTES = 1_000_000
MAX_LIST = 300
MAX_MATCHES = 60
CLONE_TIMEOUT_S = 180


def parse_repo(reference):
    match = REPO.fullmatch(str(reference).strip())
    if match is None or match.group(2) in (".", ".."):
        raise ValueError("not a GitHub repository; expected owner/name or https://github.com/owner/name")
    return match.group(1), match.group(2)


def checkout(reference, base=BASE):
    owner, name = parse_repo(reference)
    target = home() / "repos" / f"{owner}__{name}"
    if (target / ".git").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    done = subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--no-tags", f"{base}{owner}/{name}", str(target)],
                          capture_output=True, text=True, timeout=CLONE_TIMEOUT_S, env=env)
    if done.returncode != 0:
        raise ValueError(f"could not clone {owner}/{name}: {done.stderr.strip()[-300:] or 'git failed'}")
    return target


def _files(root):
    out = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for name in sorted(files):
            path = Path(folder) / name
            if not path.is_symlink():
                out.append(path.relative_to(root).as_posix())
    return out


def _inside(root, relative):
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError("path leaves the repository")
    if ".git" in path.relative_to(root.resolve()).parts:
        raise ValueError("the .git folder is not readable")
    return path


def _text(path):
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file is larger than {MAX_FILE_BYTES} bytes")
    data = path.read_bytes()
    if b"\0" in data[:8000]:
        raise ValueError("binary file")
    return data.decode("utf-8", errors="replace")


def read(reference, path="", part=0, search="", base=BASE):
    """Overview (no path), a directory listing, one file in parts, or a text search across the repository."""
    root = checkout(reference, base)
    owner, name = parse_repo(reference)
    repo = f"{owner}/{name}"
    files = _files(root)
    if search:
        needle, matches = search.lower(), []
        scope = [f for f in files if f.startswith(path.strip("/") + "/")] if path.strip("/") else files
        for relative in scope:
            target = root / relative
            try:
                if target.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = target.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8000]:
                continue
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if needle in line.lower():
                    matches.append(f"{relative}:{number}: {line.strip()[:200]}")
                    if len(matches) >= MAX_MATCHES:
                        return dict(repo=repo, search=search, matches=matches, note=f"first {MAX_MATCHES} matches only")
        return dict(repo=repo, search=search, matches=matches)
    relative = path.strip("/")
    target = _inside(root, relative) if relative else root
    if not target.exists():
        raise ValueError(f"no such path in {repo}: {relative}")
    if target.is_dir():
        prefix = relative + "/" if relative else ""
        inside = [f for f in files if f.startswith(prefix)]
        children = Counter()
        for f in inside:
            head, _, rest = f[len(prefix):].partition("/")
            children[head + ("/" if rest else "")] += 1
        listing = [f"{child} ({count} files)" if child.endswith("/") else child for child, count in sorted(children.items())]
        result = dict(repo=repo, path=relative or "/", entries=listing[:MAX_LIST], files_below=len(inside))
        if not relative:
            commit = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %cs"], capture_output=True, text=True)
            result["commit"] = commit.stdout.strip()
            result["file_types"] = dict(Counter(Path(f).suffix or "(none)" for f in files).most_common(8))
            readme = next((f for f in files if "/" not in f and f.lower().startswith("readme")), None)
            if readme:
                result["readme"] = _text(root / readme)[:6000]
            result["note"] = "Pass path to list a folder or read a file, or search to find text. Code is read, never run."
        return result
    text = _text(target)
    parts = max(1, -(-len(text) // PART_CHARS))
    part = part or 1
    if not 1 <= part <= parts:
        raise ValueError(f"part must be between 1 and {parts}")
    return dict(repo=repo, path=relative, part=part, parts=parts, text=text[(part - 1) * PART_CHARS: part * PART_CHARS])
