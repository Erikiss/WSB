"""The executor on txl-tiny: verdicts, metrics, controls, copy-on-write and the thread policy."""
import hashlib
from pathlib import Path

import pytest
import torch

from wsbench.exec import resolve
from wsbench.exec.executor import run_script
from wsbench.lang.parser import parse
from wsbench.models import txl
from wsbench.models.registry import load_model


BATTERY = Path(__file__).parent / 'data' / 'txl_battery.wsl'
L = txl.P.format


def weight_hashes(weights):
    return {key: hashlib.sha256(tensor.numpy().tobytes()).digest() for key, tensor in weights.items()}


@pytest.fixture(scope='module')
def battery():
    model = load_model('txl-tiny')
    before = weight_hashes(model.W0)
    script = BATTERY.read_text()
    first = run_script(script)
    second = run_script(script)
    assert weight_hashes(model.W0) == before                       # the baseline is never written to
    assert [v['metrics'] for v in first] == [v['metrics'] for v in second]
    return {v['label']: v for v in first}


def test_txl_battery(battery):
    assert list(battery) == [f'X{n}' for n in range(1, 25)]
    exact = set(range(1, 11)) | {14, 23}
    edits = set(range(15, 25))
    for n, verdict in enumerate(battery.values(), 1):
        assert verdict['line'] == n + 2 and verdict['model'] == 'txl-tiny' and verdict['error'] is None
        assert verdict['status'] == ('MEASURED' if n in edits else 'CONFIRMED' if n in exact else 'REFUTED')
        assert verdict['measured'] == ('exact' if n in exact else 'edit')
        metrics = verdict['metrics']
        assert metrics['loss0'] == pytest.approx(6.1730, abs=1e-4) and metrics['norm_kind'] == 'ln'
        assert metrics['new_params'] == metrics['removed_params'] == []
        if n in exact:
            assert metrics['rel'] <= 1e-14 and metrics['changed_tensors']
        if n in edits:
            assert metrics['param_dist'] > 0
            assert set(metrics['control']) == {'max_abs_dlogits', 'dloss', 'ratio'}
        else:
            assert metrics['control'] is None
    dloss = {'X13': 0.5632, 'X15': 0.5632, 'X16': 0.1812, 'X17': -0.4343, 'X18': 0.5243, 'X20': -0.4268, 'X24': 0.0045}
    for label, value in dloss.items():
        assert battery[label]['metrics']['dloss'] == pytest.approx(value, abs=5e-4), label
    assert [w['code'] for w in battery['X23']['warnings']] == ['W_NOOP']
    assert [h['code'] for h in battery['X11']['hints']] == ['H_RELU_ROTATE']
    assert [h['code'] for h in battery['X18']['hints']] == ['H_LERP_ALIGN']
    assert [h['code'] for h in battery['X23']['hints']] == ['H_EDIT_GAUGE']


def test_changed_tensors_name_exactly_what_each_edit_replaced(battery):
    def names(label):
        return {t['name'] for t in battery[label]['metrics']['changed_tensors']}
    assert names('X15') == {L(2) + w for w in txl.WRITERS}                          # delete: the writers only
    assert names('X16') == {L(i) + k for i in (1, 2) for k in txl.LAYER_KEYS}       # swap: both layers
    assert names('X17') == {L(3) + k for k in txl.LAYER_KEYS}
    assert names('X20') == {L(2) + k for k in txl.LAYER_KEYS if 'layer_norm' not in k}
    assert names('X21') == {L(0) + txl.QKV} and names('X22') == {txl.EMB.format(0)}
    rel = {t['name']: t['rel_change'] for t in battery['X15']['metrics']['changed_tensors']}
    assert rel == {name: pytest.approx(1.0) for name in names('X15')}               # zeroed: 100 % change


def test_tensor_views_share_unchanged_tensors():
    model = load_model('txl-tiny')
    states = {}
    run_script('model txl-tiny\nedit swap(L1, L2)\nedit delete(L3)', on_state=lambda label, line, W, kind: states.update({line: W}))
    swapped, deleted = states[2], states[3]
    assert all(swapped[L(1) + k] is model.W0[L(2) + k] for k in txl.LAYER_KEYS)    # swap moves tensors, copies none
    assert sum(deleted[k] is not model.W0[k] for k in model.W0) == len(txl.WRITERS)


def test_unchanged_tensors_are_not_compared_entry_by_entry(monkeypatch):
    model = load_model('txl-tiny')
    after = txl.zero_writers(model.W0, [1])
    calls = []
    real_equal = torch.equal
    monkeypatch.setattr(torch, 'equal', lambda a, b: calls.append(1) or real_equal(a, b))
    changes = resolve._changes(model.W0, after)
    assert len(calls) == len(txl.WRITERS) and len(changes['changed_tensors']) == len(txl.WRITERS)


def test_identity_ops_change_nothing_and_leave_the_control_alone():
    script = 'model txl-tiny\nedit permute(L1.heads, perm([0, 1]))'
    direct = run_script(script)[0]
    assert direct['metrics']['changed_tensors'] == [] and direct['metrics']['param_dist'] == 0
    assert direct['metrics']['control'] is None and direct['status'] == 'MEASURED' and direct['measured'] == 'exact'
    with_identity = run_script(script + '; delete(L2)')[0]
    without_identity = run_script('model txl-tiny\nedit delete(L2)')[0]
    assert with_identity['metrics'] == without_identity['metrics']


def test_the_control_draws_each_tensor_from_its_own_generator(monkeypatch):
    """The noise on a tensor depends only on the statement's seed and the tensor's name, not on the other edits."""
    model = load_model('txl-tiny')
    seen = []
    real_forward = txl.forward
    monkeypatch.setattr(txl, 'forward', lambda cfg, W, ids, *a, **k: seen.append(W) or real_forward(cfg, W, ids, *a, **k))
    seed = 1234
    direction = {}
    for layers in ([1], [1, 3]):
        after = txl.zero_writers(model.W0, layers)
        metrics = {'dloss': 0.1, **resolve._changes(model.W0, after)}
        control = resolve._txl_control(model, after, 'ln', seed, metrics)
        assert set(control) == {'dloss', 'max_abs_dlogits', 'ratio'}
        base = seen.pop()
        moved = {k for k in base if base[k] is not model.W0[k]}
        assert moved == {L(i) + w for i in layers for w in txl.WRITERS}                 # noise on changed tensors only
        delta = {k: base[k] - model.W0[k] for k in moved}
        norm = sum(d.square().sum() for d in delta.values()).sqrt().item()
        assert norm == pytest.approx(metrics['param_dist'], rel=1e-12)                # the edit's norm
        key = L(1) + txl.O
        direction[tuple(layers)] = delta[key] / delta[key].norm()
        generator = torch.Generator().manual_seed(resolve.tensor_seed(seed, key))
        expected = torch.randn(model.W0[key].shape, generator=generator, dtype=torch.float64)
        assert torch.allclose(direction[tuple(layers)], expected / expected.norm(), atol=1e-12)
    assert torch.allclose(direction[(1,)], direction[(1, 3)], atol=1e-12)
    assert resolve.tensor_seed(seed, key) != resolve.tensor_seed(seed + 1, key) != resolve.tensor_seed(seed, L(2) + txl.O)
    assert 0 <= resolve.tensor_seed(10 ** 30, key) < 2 ** 64


def test_thread_policy(monkeypatch):
    seen = []
    monkeypatch.setattr(resolve, '_execute', lambda stmt: seen.append(torch.get_num_threads()))
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        resolve.execute_txl(parse('model txl-tiny\ngauge vo(L0.H0, gl)')[0][0])
        assert seen == [1] and torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(previous)
    threads = {model: resolve._threads(parse(f'model {model}\nedit delete(L0)')[0][0])
               for model in ('neox-tiny', 'pythia-14m', 'txl-tiny', 'transfo-xl-wt103')}
    assert threads == {'neox-tiny': 1, 'pythia-14m': 2, 'txl-tiny': 1, 'transfo-xl-wt103': 4}


def test_missing_checkpoint_is_a_statement_error(tmp_path, monkeypatch):
    monkeypatch.setenv('WSB_HOME', str(tmp_path))
    first, second = run_script('model transfo-xl-wt103\ngauge permute(L[mid].heads, randperm)\nmodel txl-tiny\n'
                               'gauge permute(L[mid].heads, perm([1, 0]))')
    assert first['status'] == 'ERROR'
    assert first['error'] == {'code': 'E_DOWNLOAD', 'message': 'run: wsbench fetch transfo-xl-wt103'}
    assert second['status'] == 'CONFIRMED'
