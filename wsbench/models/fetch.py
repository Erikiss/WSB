"""Fetch the pinned model snapshots and check each file before use."""
import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from wsbench.paths import home

try:
    import fcntl
except ImportError:          # Windows: fetch() takes no lock
    fcntl = None


@dataclass(frozen=True)
class Snapshot:
    repo: str        # Hugging Face repository
    revision: str    # pinned commit
    sha256: dict     # file name -> SHA-256, in download order

    @property
    def base_url(self):
        return f'https://huggingface.co/{self.repo}/resolve/{self.revision}'


SNAPSHOTS = {
    'pythia-14m': Snapshot('EleutherAI/pythia-14m', 'cf967c0a9a04383db6f7b1108d86b2962634b4ac', {
        'config.json': 'f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17',
        'model.safetensors': '116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1',
        'tokenizer.json': '870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2',
    }),
    # Transformer-XL large on WikiText-103 (HF TransfoXLLMHeadModel). pytorch_model.bin is the legacy torch format and
    # is only read with torch.load(weights_only=True) after its hash check; the vocabulary comes from vocab.json. The
    # pickled files of the repository (vocab.pkl, vocab.bin, corpus.bin) are never fetched.
    'transfo-xl-wt103': Snapshot('transfo-xl/transfo-xl-wt103', '40a186da79458c9f9de846edfaea79c412137f97', {
        'config.json': '5ccc2e5ae36a26fa0edccfc9fd16aba633bde30aa3cadca17e2394480b823ada',
        'pytorch_model.bin': 'edd5e62f5edb8ef65703ef827ab0bcd16ab29f2b853cfd9d11d81faa1bc7a0fb',
        'vocab.json': '24e5bbda51446a21610bd58bb1ebdefee85766d0b3715f35da09babead595243',
    }),
}

# The Pythia snapshot under the names it had before the table.
REVISION = SNAPSHOTS['pythia-14m'].revision
BASE_URL = SNAPSHOTS['pythia-14m'].base_url
SHA256 = SNAPSHOTS['pythia-14m'].sha256


class DownloadError(Exception):
    code = 'E_DOWNLOAD'


def _snapshot(name):
    if name not in SNAPSHOTS:
        raise ValueError(f'no pinned snapshot for {name!r}')
    return SNAPSHOTS[name]


def model_dir(name):
    _snapshot(name)
    return home() / 'models' / name


def local(name):
    return all((model_dir(name) / file).is_file() for file in _snapshot(name).sha256)


def _sha256(path: Path):
    with path.open('rb') as fp:
        return hashlib.file_digest(fp, 'sha256').hexdigest()


def _delete(path: Path, file: str, name: str):
    """Delete a file with the wrong hash. If that fails (a read-only model directory), the error is still E_DOWNLOAD
    and names the file."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise DownloadError(f'{file}: SHA-256 mismatch (cannot delete it: {exc}); run: wsbench fetch {name}') from exc


def _check(path: Path, expected: str, name: str = 'pythia-14m', file: str | None = None):
    """Raise DownloadError if `path` does not have the pinned hash, after deleting it. The message names `file`, the
    snapshot file (by default the name of `path`)."""
    if _sha256(path) != expected:
        file = file or path.name
        _delete(path, file, name)
        raise DownloadError(f'{file}: SHA-256 mismatch; run: wsbench fetch {name}')


def verify(name, directory=None):
    """Check every file of the snapshot: first that all of them are there, then their hashes. A file with the wrong
    hash is deleted."""
    directory = model_dir(name) if directory is None else Path(directory)
    files = _snapshot(name).sha256
    if not all((directory / file).is_file() for file in files):
        raise DownloadError(f'run: wsbench fetch {name}')     # before hashing 1.1 GB that cannot be used anyway
    for file, digest in files.items():
        _check(directory / file, digest, name)


def partial_path(directory, file):
    """The download of `file` in progress. The name is fixed, so a fetch that was killed (SIGKILL, SIGHUP, SIGTERM)
    leaves at most one partial file per snapshot file. The next fetch overwrites and then removes it, or only removes
    it when the file is already there with the right hash."""
    return Path(directory) / f'.{file}.part'


# Temporary files of earlier versions (NamedTemporaryFile in the model directory), left behind by a killed fetch.
_LEGACY_TMP = re.compile(r'tmp[a-z0-9_]{8}')


@contextmanager
def _locked(directory: Path):
    """One fetch per model directory at a time, also across processes: an exclusive flock on the directory, as in
    exec/runs.py. The kernel drops it when the process ends, also on SIGKILL. Without fcntl there is no lock."""
    if fcntl is None:
        yield
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _discard(path: Path):
    """Remove a leftover of a killed fetch. One that cannot be removed (a read-only directory) is harmless and stays."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def fetch(name, progress=None):
    """Download the missing files of the snapshot and check the ones that are already there; a file with the wrong
    hash is deleted and downloaded again. Each download goes to partial_path(directory, file) and is renamed after
    its hash check. A second fetch of the same snapshot waits for the first and then finds its files present.

    progress(file, received_bytes, total_bytes or None) is called after each received chunk and at the end of each
    file. Only the call at the end can have received_bytes = total_bytes; when the server sent no size, total_bytes is
    then the size received. A callback that raises OSError (EIO once the terminal has hung up) is not called again,
    and the download goes on."""
    import httpx

    snapshot = _snapshot(name)
    directory = model_dir(name)
    file = None

    def report(*args):
        nonlocal progress
        if progress is not None:
            try:
                progress(*args)
            except OSError:
                progress = None

    try:
        directory.mkdir(parents=True, exist_ok=True)
        with _locked(directory), httpx.Client(follow_redirects=True, timeout=90) as client:
            for path in directory.iterdir():
                if _LEGACY_TMP.fullmatch(path.name) and path.is_file():
                    _discard(path)
            for file, expected in snapshot.sha256.items():
                dest, part = directory / file, partial_path(directory, file)
                if dest.is_file():
                    if _sha256(dest) == expected:
                        _discard(part)                  # left by an earlier fetch that was killed
                        continue
                    _delete(dest, file, name)           # the wrong hash: download it again
                try:
                    with client.stream('GET', f'{snapshot.base_url}/{file}') as response:
                        response.raise_for_status()
                        total = response.headers.get('content-length')
                        total = int(total) if total and total.isdigit() else None
                        with part.open('wb') as fp:
                            for chunk in response.iter_bytes(1024 * 1024):
                                fp.write(chunk)
                                # The byte count can reach the total before the last chunk: the end comes below
                                if total is None or response.num_bytes_downloaded < total:
                                    report(file, response.num_bytes_downloaded, total)
                        received = response.num_bytes_downloaded
                        report(file, received, received if total is None else total)
                    _check(part, expected, name, file)
                    part.replace(dest)
                finally:
                    part.unlink(missing_ok=True)
    except httpx.HTTPStatusError as exc:
        raise DownloadError(f'{file}: HTTP {exc.response.status_code}; run: wsbench fetch {name}') from exc
    except (OSError, httpx.RequestError) as exc:
        raise DownloadError(f'{exc}; run: wsbench fetch {name}') from exc


def pythia_dir():
    return model_dir('pythia-14m')


def pythia_local():
    return local('pythia-14m')


def verify_pythia(directory):
    verify('pythia-14m', directory)


def fetch_pythia():
    fetch('pythia-14m')
