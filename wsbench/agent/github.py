"""Let the agent read the code of a public GitHub repository.

The repository is cloned once (shallow) into WSB_HOME/repos/ and then only read as text: listed, paged and searched.
Nothing from a repository is ever executed or imported.
"""
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from threading import Lock
from uuid import uuid4

from wsbench.paths import home

from .arxiv import PART_CHARS
from .text import clean

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


_locks, _guard = {}, Lock()


def _complete(target):
    """A finished checkout has a .git folder of its own whose HEAD resolves. (Pinning the git dir matters: without a
    .git folder git would look at the parent folders and report on some other repository.)"""
    git_dir = target / ".git"
    if not git_dir.is_dir():
        return False
    done = subprocess.run(["git", f"--git-dir={git_dir}", "rev-parse", "--verify", "-q", "HEAD"], capture_output=True)
    return done.returncode == 0


def checkout(reference, base=BASE):
    """The local shallow clone of a repository, cloned on first use. Cloning goes into a temporary folder that is
    published with one rename, so a clone that was killed (timeout, server stop) never leaves a checkout that looks
    finished. One clone per repository at a time; a second caller waits and then finds the finished checkout."""
    owner, name = parse_repo(reference)
    key = f"{owner}__{name}".lower()                       # GitHub names are case-insensitive
    folder = home() / "repos"
    target = folder / key
    with _guard:
        lock = _locks.setdefault(key, Lock())
    with lock:
        if _complete(target):
            return target
        folder.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target, ignore_errors=True)          # debris of an earlier killed clone
        for stale in folder.glob(f"{key}.tmp-*"):
            if time.time() - stale.stat().st_mtime > 2 * CLONE_TIMEOUT_S:
                shutil.rmtree(stale, ignore_errors=True)
        tmp = folder / f"{key}.tmp-{uuid4().hex[:8]}"
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
        try:
            done = subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--no-tags", f"{base}{owner}/{name}", str(tmp)],
                                  capture_output=True, text=True, errors="replace", timeout=CLONE_TIMEOUT_S, env=env)
            if done.returncode != 0:
                detail = done.stderr.replace(str(tmp), name).replace(str(folder), "").strip()[-300:]
                raise ValueError(f"could not clone {owner}/{name}: {clean(detail) or 'git failed'}")
            os.replace(tmp, target)
        except subprocess.TimeoutExpired:
            raise ValueError(f"clone timed out after {CLONE_TIMEOUT_S} s; try again or ask for a specific folder") from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
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


def _locate(root, path, repo):
    """The repository-relative path (posix, '' for the root) and the absolute path for a path as the agent writes it:
    './pkg', 'pkg/', '.' and '/pkg' all work; a path that does not exist is an error, never an empty result."""
    parts = [p for p in str(path).strip().split("/") if p not in ("", ".")]
    if not parts:
        return "", root
    target = _inside(root, "/".join(parts))
    if not target.exists():
        raise ValueError(f"no such path in {repo}: {'/'.join(parts)}")
    relative = target.relative_to(root.resolve()).as_posix()
    return ("" if relative == "." else relative), target


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
    relative, target = _locate(root, path, repo)
    if search:
        needle, matches = search.lower(), []
        scope = [relative] if target.is_file() else [f for f in files if f.startswith(relative + "/")] if relative else files
        for found in scope:
            file = root / found
            try:
                if file.stat().st_size > MAX_FILE_BYTES:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8000]:
                continue
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if needle in line.lower():
                    matches.append(clean(f"{found}:{number}: {line.strip()[:200]}"))
                    if len(matches) >= MAX_MATCHES:
                        return dict(repo=repo, search=search, matches=matches, note=f"first {MAX_MATCHES} matches only")
        result = dict(repo=repo, search=search, matches=matches)
        if not matches:
            result["note"] = f"no match in the {len(scope)} files searched" + (f" below {relative}" if relative else "")
        return result
    if target.is_dir():
        prefix = relative + "/" if relative else ""
        inside = [f for f in files if f.startswith(prefix)]
        children = Counter()
        for f in inside:
            head, _, rest = f[len(prefix):].partition("/")
            children[head + ("/" if rest else "")] += 1
        listing = [clean(f"{child} ({count} files)" if child.endswith("/") else child) for child, count in sorted(children.items())]
        result = dict(repo=repo, path=relative or "/", entries=listing[:MAX_LIST], files_below=len(inside))
        if not relative:
            commit = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %cs"], capture_output=True, text=True, errors="replace")
            result["commit"] = commit.stdout.strip()
            result["file_types"] = dict(Counter(clean(Path(f).suffix) or "(none)" for f in files).most_common(8))
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
