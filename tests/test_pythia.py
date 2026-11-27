import hashlib
import subprocess
from pathlib import Path
from time import perf_counter

import pytest

from wsbench.exec.executor import run_script
from wsbench.models import fetch
from wsbench.models.registry import load_model


EXAMPLES = Path(__file__).parents[1] / 'wsbench' / 'examples'


def test_corrupt_model_fails_before_use(hub):
    stand_in = hub('pythia-14m')                  # offline: fetch() talks to the stand-in hub of conftest.py
    directory = fetch.pythia_dir()
    directory.mkdir(parents=True)
    for name in fetch.SHA256:
        (directory / name).write_bytes(b'corrupt')
    first, second = run_script('model pythia-14m\ngauge fold_ln()\ngauge fold_ln()')
    assert first['status'] == second['status'] == 'ERROR'
    assert first['error'] == {'code': 'E_DOWNLOAD', 'message': 'SHA-256 mismatch; run: wsbench fetch pythia-14m'}
    assert not (directory / 'config.json').exists()                 # deleted before anything read it
    assert second['error'] == {'code': 'E_DOWNLOAD', 'message': 'run: wsbench fetch pythia-14m'}
    fetch.fetch_pythia()               # downloads the deleted file and, after their hash checks, the corrupt ones
    assert stand_in.requests == list(fetch.SHA256)
    assert all((directory / name).read_bytes() == data for name, data in stand_in.files.items())
    fetch.verify_pythia(directory)


@pytest.mark.network
def test_pythia_reference():
    subprocess.run(['wsbench', 'fetch', 'pythia-14m'], check=True)
    for name, expected in fetch.SHA256.items():
        with (fetch.pythia_dir() / name).open('rb') as fp:
            assert hashlib.file_digest(fp, 'sha256').hexdigest() == expected
    model = load_model('pythia-14m')
    assert len(model.W0) == 76
    assert model.ids.shape == (1, 149)
    assert model.loss0 == pytest.approx(4.3177, abs=1e-4)
    assert model.max_logit0 == pytest.approx(25.85, abs=0.01)
    rotation = (EXAMPLES / 'rotate_middle_layer.wsl').read_text()
    pair = (EXAMPLES / 'antisymmetric_pair.wsl').read_text()
    gridop = (EXAMPLES / 'arc_mirror.wsl').read_text().split('model pythia-14m')[1]
    start = perf_counter()
    verdicts = run_script(rotation + '\n' + pair + '\n' + gridop)
    seconds = perf_counter() - start
    assert len(verdicts) == 28
    for verdict in verdicts:
        assert verdict['error'] is None, verdict
    rows = {v['label']: v for v in verdicts}
    exact = {1, 4, 6, 7, 9, 10, 12, 14, 15}
    for n in range(1, 18):
        row = rows[f'R{n}']
        assert row['measured'] == ('exact' if n in exact else 'edit')
        assert row['status'] == ('MEASURED' if n == 5 else 'CONFIRMED' if n in exact or n == 17 else 'REFUTED')
    for n in (16, 17):
        m = rows[f'R{n}']['metrics']
        assert m['rel'] == pytest.approx(0.0175, rel=0.001)
        assert m['dloss'] == pytest.approx(0.0002, abs=0.0005)
        assert m['max_abs_dlogits'] == pytest.approx(0.4519, abs=0.00005)
    deltas = [2.5106, 2.1577, 4.0639, 6.2921, 4.9923, 4.0429, 3.2747, 2.9584, 0.0, 2.6678]
    for n, dloss in enumerate(deltas, 1):
        row = rows[f'A{n}']
        assert row['status'] == ('CONFIRMED' if n == 9 else 'MEASURED')
        assert row['measured'] == ('exact' if n == 9 else 'edit')
        assert row['metrics']['dloss'] == pytest.approx(dloss, abs=0.0005)
    assert rows['W1']['status'] == 'MEASURED'
    assert rows['W1']['measured'] == 'edit'
    assert rows['W1']['metrics']['dloss'] == pytest.approx(0.1764, abs=0.0005)
    print(f'28 Pythia statements: {seconds:.2f} s')
    assert seconds < 30, f'28 Pythia statements took {seconds:.2f} seconds'
