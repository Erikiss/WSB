import hashlib
from pathlib import Path

import pytest

from wsbench.exec.executor import run_script
from wsbench.models.registry import load_model


BATTERY = Path(__file__).parent / 'data' / 'tiny_battery.wsl'


def weight_hashes(weights):
    return {key: hashlib.sha256(tensor.numpy().tobytes()).digest() for key, tensor in weights.items()}


@pytest.fixture(scope='module')
def battery():
    model = load_model('neox-tiny')
    before = weight_hashes(model.W0)
    script = BATTERY.read_text()
    first = run_script(script)
    second = run_script(script)
    assert weight_hashes(model.W0) == before
    assert [v['metrics']['max_abs_dlogits'] for v in first] == [v['metrics']['max_abs_dlogits'] for v in second]
    return {v['label']: v for v in first}


def test_tiny_battery(battery):
    assert load_model('neox-tiny').loss0 == pytest.approx(5.9561, abs=1e-4)
    assert list(battery) == [f'T{n}' for n in range(1, 20)]
    measured_edits = {5, 18, 19}
    exact = {1, 4, 6, 7, 9, 10, 12, 14, 17, 18}
    for n, verdict in enumerate(battery.values(), 1):
        assert verdict['line'] == n + 2
        assert verdict['error'] is None
        status = 'MEASURED' if n in measured_edits else 'CONFIRMED' if n in exact or n == 15 else 'REFUTED'
        assert verdict['status'] == status
        measured = 'exact' if n in exact else 'near' if n in (15, 16) else 'edit'
        assert verdict['measured'] == measured
        metrics = verdict['metrics']
        if n in measured_edits:
            assert metrics['param_dist'] > 0
            assert set(metrics['control']) == {'max_abs_dlogits', 'dloss', 'ratio'}
        else:
            assert metrics['control'] is None
    for label in ('T15', 'T16'):
        assert battery[label]['metrics']['rel'] == pytest.approx(7.1558e-6, rel=0.001)
    assert battery['T19']['metrics']['dloss'] == pytest.approx(-0.3300, abs=0.0005)
    assert battery['T14']['metrics']['new_params'] == ['embed_out.bias']
    assert battery['T14']['metrics']['changed_tensors'][-1]['rel_change'] is None
    assert battery['T14']['metrics']['norm_kind'] == 'rms'
    assert [w['code'] for w in battery['T18']['warnings']] == ['W_NOOP']


def test_zero_distance_control_and_rng_index():
    script = 'model neox-tiny\nedit permute(L1.heads, perm([0, 1]))'
    direct = run_script(script)[0]
    assert direct['metrics']['changed_tensors'] == []
    assert direct['metrics']['param_dist'] == 0
    assert direct['metrics']['control'] is None
    with_identity = run_script(script + '; rotate(resid, orth)')[0]
    without_identity = run_script('model neox-tiny\nedit rotate(resid, orth)')[0]
    assert with_identity['metrics'] == without_identity['metrics']


def test_numeric_overflow_is_local_and_updates_are_snapshots():
    updates = []
    results = run_script("model neox-tiny\nedit scale(resid, 1e300)\ngauge fold_ln()", on_update=updates.append)
    assert [v["status"] for v in results] == ["ERROR", "CONFIRMED"]
    assert results[0]["error"]["code"] == "E_RUNTIME"
    assert [tuple(v["status"] for v in event) for event in updates] == [
        ("PENDING", "PENDING"), ("RUNNING", "PENDING"), ("ERROR", "PENDING"),
        ("ERROR", "RUNNING"), ("ERROR", "CONFIRMED"),
    ]
