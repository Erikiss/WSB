"""Fetch the pinned Pythia snapshot and check each file before use."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

from wsbench.paths import home


REVISION = 'cf967c0a9a04383db6f7b1108d86b2962634b4ac'
BASE_URL = f'https://huggingface.co/EleutherAI/pythia-14m/resolve/{REVISION}'
SHA256 = {
    'config.json': 'f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17',
    'model.safetensors': '116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1',
    'tokenizer.json': '870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2',
}


class DownloadError(Exception):
    code = 'E_DOWNLOAD'


def pythia_dir():
    return home() / 'models' / 'pythia-14m'


def pythia_local():
    return all((pythia_dir() / name).is_file() for name in SHA256)


def _check(path: Path, expected: str):
    with path.open('rb') as fp:
        digest = hashlib.file_digest(fp, 'sha256').hexdigest()
    if digest != expected:
        path.unlink(missing_ok=True)
        raise DownloadError('SHA-256 mismatch; run: wsbench fetch pythia-14m')


def verify_pythia(directory):
    for name, digest in SHA256.items():
        path = directory / name
        if not path.is_file():
            raise DownloadError('run: wsbench fetch pythia-14m')
        _check(path, digest)


def fetch_pythia():
    import httpx

    directory = pythia_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=90) as client:
            for name, expected in SHA256.items():
                dest = directory / name
                if dest.is_file():
                    _check(dest, expected)
                    continue
                tmp = None
                try:
                    with client.stream('GET', f'{BASE_URL}/{name}') as response:
                        response.raise_for_status()
                        with NamedTemporaryFile(dir=directory, delete=False) as fp:
                            tmp = Path(fp.name)
                            for chunk in response.iter_bytes(1024 * 1024):
                                fp.write(chunk)
                    _check(tmp, expected)
                    tmp.replace(dest)
                finally:
                    if tmp is not None:
                        tmp.unlink(missing_ok=True)
    except httpx.HTTPStatusError as exc:
        raise DownloadError(f'{name}: HTTP {exc.response.status_code}; run: wsbench fetch pythia-14m') from exc
    except (OSError, httpx.RequestError) as exc:
        raise DownloadError(f'{exc}; run: wsbench fetch pythia-14m') from exc
