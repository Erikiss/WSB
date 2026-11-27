"""Fetch the pinned model snapshots and check each file before use."""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from wsbench.paths import home


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


def _check(path: Path, expected: str, name: str = 'pythia-14m'):
    with path.open('rb') as fp:
        digest = hashlib.file_digest(fp, 'sha256').hexdigest()
    if digest != expected:
        path.unlink(missing_ok=True)
        raise DownloadError(f'SHA-256 mismatch; run: wsbench fetch {name}')


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
    leaves at most one partial file per snapshot file, and the next fetch overwrites and then removes it."""
    return Path(directory) / f'.{file}.part'


# Temporary files of earlier versions (NamedTemporaryFile in the model directory), left behind by a killed fetch.
_LEGACY_TMP = re.compile(r'tmp[a-z0-9_]{8}')


def fetch(name, progress=None):
    """Download the missing files of the snapshot and check the ones that are already there; a file with the wrong
    hash is deleted and downloaded again. Each download goes to partial_path(directory, file) and is renamed after
    its hash check. progress(file, received_bytes, total_bytes or None) is called after each received chunk."""
    import httpx

    snapshot = _snapshot(name)
    directory = model_dir(name)
    file = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if _LEGACY_TMP.fullmatch(path.name) and path.is_file():
                path.unlink(missing_ok=True)
        with httpx.Client(follow_redirects=True, timeout=90) as client:
            for file, expected in snapshot.sha256.items():
                dest = directory / file
                if dest.is_file():
                    try:
                        _check(dest, expected, name)
                        continue
                    except DownloadError:
                        pass                    # _check deleted the corrupt file: download it again
                part = partial_path(directory, file)
                try:
                    with client.stream('GET', f'{snapshot.base_url}/{file}') as response:
                        response.raise_for_status()
                        total = response.headers.get('content-length')
                        total = int(total) if total and total.isdigit() else None
                        with part.open('wb') as fp:
                            for chunk in response.iter_bytes(1024 * 1024):
                                fp.write(chunk)
                                if progress is not None:
                                    progress(file, response.num_bytes_downloaded, total)
                    _check(part, expected, name)
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
