import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """An offline stand-in for the Hugging Face hub, for fetch() in WSB_HOME = tmp_path.

    hub(name) serves stand-in files for the snapshot `name` and pins their hashes in place of the real ones; with
    sizes (file name -> bytes) the stand-ins have those sizes instead of about 150 kB. It returns a namespace: name,
    directory (the model directory), files (file name -> the stand-in bytes), serve (what the hub answers: file name
    -> bytes, or a callable request -> httpx.Response) and requests (requested file names).
    """
    import httpx
    from wsbench.models import fetch

    monkeypatch.setenv('WSB_HOME', str(tmp_path))
    serve, requests = {}, []

    class Body(httpx.SyncByteStream):          # streamed in pieces like a network response (content= is pre-read)
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            for start in range(0, len(self.data), 40_000):
                yield self.data[start:start + 40_000]

    def handler(request):
        file = request.url.path.rsplit('/', 1)[1]
        requests.append(file)
        body = serve.get(file)
        if callable(body):
            return body(request)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, headers={'content-length': str(len(body))}, stream=Body(body))

    real_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))

    def stand_in(name, sizes=None):
        def data(file):
            text = f'stand-in for {file} of {name} '.encode()
            return text * 3000 if sizes is None else (text * (sizes[file] // len(text) + 1))[:sizes[file]]

        files = {file: data(file) for file in fetch.SNAPSHOTS[name].sha256}
        digests = {file: hashlib.sha256(data).hexdigest() for file, data in files.items()}
        monkeypatch.setitem(fetch.SNAPSHOTS, name, replace(fetch.SNAPSHOTS[name], sha256=digests))
        serve.clear()
        serve.update(files)
        return SimpleNamespace(name=name, directory=fetch.model_dir(name), files=files, serve=serve, requests=requests)

    return stand_in
