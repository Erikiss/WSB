"""Transformer-XL: agreement with the Hugging Face reference, symmetries, copy-on-write, loading and fetching."""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pytest
import torch
import torch.nn.functional as F

from wsbench.exec.executor import run_script
from wsbench.lang.card import LANGUAGE_CARD
from wsbench.models import fetch, registry, txl
from wsbench.models.neox import rand_gl, rand_orth, rand_uniform
from wsbench.models.registry import SHAPES, load_model, param_shapes


DATA = Path(__file__).parent / 'data'
CFG = txl.TINY_CFG
IDS = torch.tensor([txl.TINY_IDS])
L = txl.P.format
# config.json of transfo-xl/transfo-xl-wt103 at the pinned revision (the keys the implementation reads or checks)
WT103_CONFIG = {
    'adaptive': True, 'attn_type': 0, 'clamp_len': 1000, 'cutoffs': [20000, 40000, 200000], 'd_embed': 1024,
    'd_head': 64, 'd_inner': 4096, 'd_model': 1024, 'div_val': 4, 'layer_norm_epsilon': 1e-05, 'mem_len': 1600,
    'model_type': 'transfo-xl', 'n_head': 16, 'n_layer': 18, 'pre_lnorm': False, 'same_length': True,
    'sample_softmax': -1, 'tie_projs': [False, True, True, True], 'tie_weight': True, 'untie_r': True,
    'vocab_size': 267735,
}


def gen(seed):
    return torch.Generator().manual_seed(seed)


def weight_hashes(weights):
    return {key: hashlib.sha256(tensor.numpy().tobytes()).digest() for key, tensor in weights.items()}


def max_diff(a, b):
    return (a - b).abs().max().item()


@pytest.fixture(scope='module')
def base():
    W = txl.make_tiny(seed=0)
    return W, txl.forward(CFG, W, IDS)


def rel_change(W, logprobs0):
    return max_diff(txl.forward(CFG, W, IDS), logprobs0) / logprobs0.abs().max().item()


# ---------------------------------------------------------------- forward

def test_forward_matches_hf_reference():
    # tests/data/txl_tiny_reference.json: HF transformers 4.46.3 TransfoXLLMHeadModel, float64, same weights
    ref = json.loads((DATA / 'txl_tiny_reference.json').read_text())
    assert ref['ids'] == txl.TINY_IDS
    W = txl.make_tiny(seed=0)
    logprobs = txl.forward(CFG, W, IDS)
    assert logprobs.shape == (1, 32, 64) and logprobs.dtype == torch.float64
    assert max_diff(logprobs[0], torch.tensor(ref['logprobs'], dtype=torch.float64)) <= 1e-12
    assert logprobs.logsumexp(-1).abs().max().item() <= 1e-12
    zero = txl.forward(CFG, W, IDS, zero_mems=CFG.mem_len)          # HF called without mems
    assert max_diff(txl.next_token_nll(zero, IDS), torch.tensor(ref['zero_mems_nll'], dtype=torch.float64)) <= 1e-12


def test_batch_rows_are_independent_and_causal(base):
    W, logprobs0 = base
    other = torch.tensor([[(5 * i + 11) % 64 for i in range(32)]])
    both = txl.forward(CFG, W, torch.cat([IDS, other]))
    assert max_diff(both[:1], logprobs0) <= 1e-12
    assert max_diff(both[1:], txl.forward(CFG, W, other)) <= 1e-12
    changed = IDS.clone()
    changed[0, -1] = (changed[0, -1] + 20) % 64
    assert max_diff(txl.forward(CFG, W, changed)[0, :-1], logprobs0[0, :-1]) <= 1e-12


def test_loss_and_residual_stream(base):
    W, logprobs0 = base
    logprobs, resid = txl.forward(CFG, W, IDS, return_resid=True)
    assert len(resid) == CFG.n_layers + 1 and all(r.shape == (1, 32, CFG.d) for r in resid)
    nll = txl.next_token_nll(logprobs, IDS)
    assert nll.shape == (31,) and txl.next_token_loss(logprobs, IDS) == pytest.approx(nll.mean().item(), abs=1e-15)
    assert txl.next_token_loss(logprobs0, IDS) == pytest.approx(6.1730, abs=1e-4)
    with pytest.raises(ValueError, match='token id'):
        txl.forward(CFG, W, torch.tensor([[0, 64]]))


def test_delete_keeps_the_post_layernorms(base):
    W, _ = base
    _, resid = txl.forward(CFG, txl.zero_writers(W, [2]), IDS, return_resid=True)
    p = L(2)
    ln = lambda x, n: F.layer_norm(x, (CFG.d,), W[p + n + 'weight'], W[p + n + 'bias'], CFG.eps)
    assert max_diff(resid[3], ln(ln(resid[2], txl.LN1), txl.LN2)) <= 1e-12      # x -> LN2(LN1(x))
    assert max_diff(resid[3], resid[2]) > 0.1                                     # not the identity


# ---------------------------------------------------------------- symmetries and edits

def align_into(target, x, y):     # L[target] := align(L[x] -> L[y])
    return lambda W: txl.set_layer(W, target, txl.align_layer(CFG, txl.get_layer(W, x), txl.get_layer(W, y)))


SYMMETRIES = {
    'permute(L1.mlp, randperm)': lambda W: txl.perm_mlp(CFG, W, 1, torch.randperm(CFG.d_mlp, generator=gen(1))),
    'scale(L0.mlp, uniform(0.25, 4))': lambda W: txl.scale_mlp(CFG, W, 0, rand_uniform(CFG.d_mlp, 0.25, 4.0, gen(2))),
    'permute(L3.heads, perm([1, 0]))': lambda W: txl.perm_heads(CFG, W, 3, [1, 0]),
    'vo(L1.H0, gl)': lambda W: txl.vo_gauge(CFG, W, 1, 0, rand_gl(CFG.dh, gen(3))),
    'vo(L2.H1, orth)': lambda W: txl.vo_gauge(CFG, W, 2, 1, rand_orth(CFG.dh, gen(4))),
    'qk(L0.H1, gl, dims=all)': lambda W: txl.qk_gauge(CFG, W, 0, 1, rand_gl(CFG.dh, gen(5))),
    'qk(L3.H0, orth, dims=all)': lambda W: txl.qk_gauge(CFG, W, 3, 0, rand_orth(CFG.dh, gen(6))),
    'L1 := align(L1 -> L2)': align_into(1, 1, 2),
}
EDITS = {
    'rotate(L1.mlp, orth)': lambda W: txl.rotate_mlp(CFG, W, 1, rand_orth(CFG.d_mlp, gen(7))),
    'delete(L2)': lambda W: txl.zero_writers(W, [2]),
    'swap(L1, L2)': lambda W: txl.swap_layers(W, 1, 2),
    'L1 := L2': lambda W: txl.set_layer(W, 1, txl.get_layer(W, 2)),
    'L1 := lerp(L1, L2, 0.5)':
        lambda W: txl.set_layer(W, 1, txl.lerp_layer(txl.get_layer(W, 1), txl.get_layer(W, 2), 0.5)),
    'L2 := neg_writers(L2)': lambda W: txl.set_layer(W, 2, txl.neg_writers(txl.get_layer(W, 2))),
    'L2 := neg(L2)': lambda W: txl.set_layer(W, 2, txl.neg_all(txl.get_layer(W, 2))),
    'L1 := align(L2 -> L1)': align_into(1, 2, 1),
    'gridop(rot90, block(qkv_net L0))': lambda W: txl.gridop_block(W, 'rot90', L(0) + txl.QKV, 0, 8, 0, 8),
    'gridop(hmirror, block(emb_layers.0))': lambda W: txl.gridop_block(W, 'hmirror', txl.EMB.format(0), 0, 16, 0, 16),
}
LAYER = lambda i, keys=txl.LAYER_KEYS: {L(i) + k for k in keys}
MLP = (txl.FF0W, txl.FF0B, txl.FF3W)
CHANGED = {   # exactly the tensors each operation replaces; every other tensor stays the same object
    'permute(L1.mlp, randperm)': LAYER(1, MLP), 'scale(L0.mlp, uniform(0.25, 4))': LAYER(0, MLP),
    'permute(L3.heads, perm([1, 0]))': LAYER(3, (txl.QKV, txl.R, txl.RW, txl.RR, txl.O)),
    'vo(L1.H0, gl)': LAYER(1, (txl.QKV, txl.O)), 'vo(L2.H1, orth)': LAYER(2, (txl.QKV, txl.O)),
    'qk(L0.H1, gl, dims=all)': LAYER(0, (txl.QKV, txl.R, txl.RW, txl.RR)),
    'qk(L3.H0, orth, dims=all)': LAYER(3, (txl.QKV, txl.R, txl.RW, txl.RR)),
    'L1 := align(L1 -> L2)': None,                       # whatever the found permutations move
    'rotate(L1.mlp, orth)': LAYER(1, MLP), 'delete(L2)': LAYER(2, txl.WRITERS),
    'swap(L1, L2)': LAYER(1) | LAYER(2), 'L1 := L2': LAYER(1), 'L1 := lerp(L1, L2, 0.5)': LAYER(1),
    'L2 := neg_writers(L2)': LAYER(2, txl.WRITERS),
    'L2 := neg(L2)': LAYER(2, [k for k in txl.LAYER_KEYS if 'layer_norm' not in k]),
    'L1 := align(L2 -> L1)': LAYER(1),
    'gridop(rot90, block(qkv_net L0))': {L(0) + txl.QKV}, 'gridop(hmirror, block(emb_layers.0))': {txl.EMB.format(0)},
}


@pytest.mark.parametrize('name', SYMMETRIES)
def test_exact_symmetries(base, name):
    W, logprobs0 = base
    out = SYMMETRIES[name](W)
    assert any(not torch.equal(out[k], W[k]) for k in W), 'the symmetry changed no weight'
    assert rel_change(out, logprobs0) <= 1e-9


def test_composed_symmetries(base):
    W, logprobs0 = base
    out = W
    for op in SYMMETRIES.values():
        out = op(out)
    assert rel_change(out, logprobs0) <= 1e-9


@pytest.mark.parametrize('name', EDITS)
def test_edits_change_the_function(base, name):
    W, logprobs0 = base
    assert rel_change(EDITS[name](W), logprobs0) > 1e-3


def test_align_undoes_a_permutation(base):
    W, _ = base
    moved = txl.perm_mlp(CFG, txl.perm_heads(CFG, W, 2, [1, 0]), 2, torch.randperm(CFG.d_mlp, generator=gen(8)))
    aligned = txl.align_layer(CFG, txl.get_layer(moved, 2), txl.get_layer(W, 2))
    assert all(torch.equal(aligned[k], W[L(2) + k]) for k in txl.LAYER_KEYS)
    same = txl.align_layer(CFG, txl.get_layer(W, 2), txl.get_layer(W, 2))
    assert all(same[k] is W[L(2) + k] for k in txl.LAYER_KEYS)          # identity permutations copy nothing


@pytest.mark.parametrize('name', {**SYMMETRIES, **EDITS})
def test_copy_on_write(base, name):
    W, _ = base
    before = weight_hashes(W)
    out = {**SYMMETRIES, **EDITS}[name](W)
    assert weight_hashes(W) == before                                     # the input is never written to
    assert list(out) == list(W)
    replaced = {k for k in W if out[k] is not W[k]}
    if CHANGED[name] is not None:
        assert replaced == CHANGED[name]
    assert replaced and all(not torch.equal(out[k], W[k]) for k in replaced)


def test_layer_values_are_checked(base):
    W, _ = base
    layer = txl.get_layer(W, 1)
    assert set(layer) == set(txl.LAYER_KEYS) and all(layer[k] is W[L(1) + k] for k in layer)
    with pytest.raises(ValueError, match='exactly the parameters of one layer'):
        txl.set_layer(W, 0, {k: v for k, v in layer.items() if k != txl.O})
    with pytest.raises(ValueError, match='E_SHAPE'):
        txl.gridop_block(W, 'x', L(0) + txl.QKV, 0, 2, 0, 4, grid_fn=lambda b: b.T)
    assert set(txl.UNDEFINED) == {'fold_ln', 'center_writers', 'ln_to_rms', 'rotate_resid', 'scale_resid',
                                  'rope_pairs', 'qk_pass'}
    assert txl.UNDEFINED['fold_ln'] == 'fold_ln is not defined for post-LayerNorm Transformer-XL'


# ---------------------------------------------------------------- geometry, configuration, loading

def test_param_shapes_match_make_tiny():
    W = txl.make_tiny()
    assert list(W) == list(param_shapes(SHAPES['txl-tiny'])) == txl.param_names(CFG)
    assert {k: tuple(v.shape) for k, v in W.items()} == param_shapes(SHAPES['txl-tiny'])
    assert txl.geometry(CFG) == SHAPES['txl-tiny']
    assert torch.equal(txl.make_tiny(seed=0)[L(3) + txl.QKV], W[L(3) + txl.QKV])     # deterministic
    tied = dict(txl.tied_pairs(CFG))
    assert tied == {f'crit.out_layers.{c}.weight': txl.EMB.format(c) for c in range(4)} | {
        f'crit.out_projs.{c}': txl.EPROJ.format(c) for c in (1, 2, 3)}
    assert not set(tied) & set(W) and 'crit.out_projs.0' in W


def test_wt103_geometry_and_config():
    shapes = param_shapes(SHAPES['transfo-xl-wt103'])
    assert len(shapes) == 249 and sum(math.prod(s) for s in shapes.values()) == 285_205_322
    cfg = txl.cfg_from_hf_config(WT103_CONFIG)
    assert txl.geometry(cfg) == SHAPES['transfo-xl-wt103']
    assert (cfg.eps, cfg.clamp_len, cfg.mem_len, cfg.same_length, cfg.inv_freq) == (1e-5, 1000, 1600, True, None)
    assert cfg.d_embs == [1024, 256, 64, 16] and cfg.bounds[-1] == (200000, 267735)
    shared = txl.cfg_from_hf_config({k: v for k, v in WT103_CONFIG.items() if k != 'tie_projs'})
    assert shared.tie_projs == (False, True, True, True)       # HF default: proj_share_all_but_first
    with pytest.raises(ValueError, match='it needs'):
        txl.cfg_from_hf_config({k: v for k, v in WT103_CONFIG.items() if k != 'd_head'})


@pytest.mark.parametrize('change, message', [
    ({'pre_lnorm': True}, 'pre_lnorm'), ({'attn_type': 1}, 'attn_type'), ({'untie_r': False}, 'untie_r'),
    ({'div_val': 1}, 'div_val'), ({'adaptive': False}, 'adaptive'), ({'tie_weight': False}, 'tied'),
    ({'model_type': 'gpt2'}, 'model_type'), ({'mem_len': 0}, 'mem_len'), ({'cutoffs': [40000, 20000]}, 'cutoffs'),
    ({'tie_projs': [False, True]}, 'tie_projs'),
])
def test_unsupported_configs(change, message):
    with pytest.raises(ValueError, match=message):
        txl.cfg_from_hf_config({**WT103_CONFIG, **change})


def hf_state(W, cfg=CFG):
    """make_tiny's weights as a TransfoXLLMHeadModel state dict: float32, tied tensors under both names, the buffer."""
    sd = {k: v.float() for k, v in W.items()}
    for dup, kept in txl.tied_pairs(cfg):
        sd[dup] = sd[kept]
    sd[txl.INV_FREQ] = 1 / (10000 ** (torch.arange(0.0, cfg.d, 2.0) / cfg.d))   # computed in float32, as HF does
    return sd


def test_from_hf_state():
    W = txl.make_tiny()
    sd = hf_state(W)
    table = sd[txl.INV_FREQ].double()
    out, cfg = txl.from_hf_state(CFG, sd)
    assert not sd                                                          # consumed while converting
    assert list(out) == list(W) and all(v.dtype == torch.float64 for v in out.values())
    assert all(torch.equal(out[k], W[k].float().double()) for k in W)
    assert replace(cfg, inv_freq=None) == CFG and torch.equal(torch.tensor(cfg.inv_freq, dtype=torch.float64), table)
    assert torch.equal(txl._pos_emb(cfg, 3)[:, :CFG.d // 2], torch.outer(torch.tensor([2.0, 1.0, 0.0]), table).sin())


@pytest.mark.parametrize('spoil, message', [
    (lambda sd: sd.update({'crit.out_layers.1.weight': sd['crit.out_layers.1.weight'] + 1}), 'is not tied'),
    (lambda sd: sd.update({'crit.out_projs.2': sd['crit.out_projs.2'] * 2}), 'is not tied'),
    (lambda sd: sd.update({txl.INV_FREQ: sd[txl.INV_FREQ] * 1.001}), 'sinusoid'),
    (lambda sd: sd.pop('crit.cluster_bias'), 'missing'),
    (lambda sd: sd.update({'transformer.r_w_bias': torch.zeros(2, 8)}), 'unexpected'),
])
def test_from_hf_state_rejects(spoil, message):
    sd = hf_state(txl.make_tiny())
    spoil(sd)
    with pytest.raises(ValueError, match=message):
        txl.from_hf_state(CFG, sd)


def test_load_tiny_model():
    model = load_model('txl-tiny')
    assert model.cfg == CFG and model.ids.tolist() == [txl.TINY_IDS]
    assert list(model.W0) == list(txl.make_tiny())
    assert max_diff(model.logits0, txl.forward(CFG, model.W0, model.ids)) <= 1e-12
    assert model.loss0 == pytest.approx(6.1730, abs=1e-4)
    assert model.max_logit0 == model.logits0.abs().max().item()


@pytest.fixture
def fresh_model_cache():
    registry._load_model.cache_clear()
    yield registry._load_model
    registry._load_model.cache_clear()           # drop the stand-in models below


def test_the_model_cache_holds_every_weight_model(tmp_path, monkeypatch, fresh_model_cache):
    """Using the other weight models must not evict transfo-xl-wt103 (a reload re-hashes 1.1 GB and rebuilds 2.4 GiB).
    Offline: pythia-14m and transfo-xl-wt103 are stood in for by the tiny models of their family."""
    monkeypatch.setenv('WSB_HOME', str(tmp_path))
    assert registry.WEIGHT_MODELS == ('neox-tiny', 'pythia-14m', 'txl-tiny', 'transfo-xl-wt103')
    assert fresh_model_cache.cache_info().maxsize == len(registry.WEIGHT_MODELS)
    loads = []

    def stand_in(load, tiny):
        def loader(name, directory):
            loads.append(name)
            return load(tiny if directory is not None else name, None)
        return loader

    monkeypatch.setattr(registry, '_load_neox', stand_in(registry._load_neox, 'neox-tiny'))
    monkeypatch.setattr(registry, '_load_txl', stand_in(registry._load_txl, 'txl-tiny'))
    monkeypatch.setitem(SHAPES, 'pythia-14m', SHAPES['neox-tiny'])
    monkeypatch.setitem(SHAPES, 'transfo-xl-wt103', SHAPES['txl-tiny'])
    order = ['transfo-xl-wt103', 'pythia-14m', 'txl-tiny', 'neox-tiny']   # wt103 first: the others come after it
    first = {name: load_model(name) for name in order}
    assert load_model('transfo-xl-wt103') is first['transfo-xl-wt103']
    assert load_model('pythia-14m') is first['pythia-14m']
    assert loads == order
    info = fresh_model_cache.cache_info()
    assert (info.misses, info.hits) == (4, 2)


# ---------------------------------------------------------------- pinned snapshots

def test_snapshot_table_keeps_the_pythia_names():
    assert fetch.pythia_dir() == fetch.model_dir('pythia-14m')
    pythia = fetch.SNAPSHOTS['pythia-14m']
    assert fetch.SHA256 is pythia.sha256 and fetch.REVISION == pythia.revision
    assert fetch.BASE_URL == f'https://huggingface.co/EleutherAI/pythia-14m/resolve/{fetch.REVISION}'
    txl_snapshot = fetch.SNAPSHOTS['transfo-xl-wt103']
    assert txl_snapshot.base_url == ('https://huggingface.co/transfo-xl/transfo-xl-wt103/resolve/'
                                     '40a186da79458c9f9de846edfaea79c412137f97')
    assert list(txl_snapshot.sha256) == ['config.json', 'pytorch_model.bin', 'vocab.json']
    with pytest.raises(ValueError, match='no pinned snapshot'):
        fetch.model_dir('txl-tiny')


@pytest.fixture
def wt103_hub(hub):
    """transfo-xl-wt103 stand-in files on the offline hub of conftest.py (in WSB_HOME = tmp_path)."""
    return hub('transfo-xl-wt103')


def test_fetch_helpers(tmp_path, monkeypatch, wt103_hub):
    hub = wt103_hub
    name, directory, files = hub.name, hub.directory, hub.files
    pinned = fetch.SNAPSHOTS[name]
    monkeypatch.setitem(fetch.SNAPSHOTS, name, replace(pinned, sha256={f: '0' * 64 for f in pinned.sha256}))
    assert directory == tmp_path / 'models' / name and not fetch.local(name)
    with pytest.raises(fetch.DownloadError, match='^run: wsbench fetch transfo-xl-wt103$'):
        fetch.verify(name)
    directory.mkdir(parents=True)
    for file, data in files.items():
        (directory / file).write_bytes(data)
    assert fetch.local(name)
    with pytest.raises(fetch.DownloadError) as info:              # the hashes do not match the files
        fetch.verify(name)
    assert str(info.value) == 'SHA-256 mismatch; run: wsbench fetch transfo-xl-wt103'
    assert info.value.code == 'E_DOWNLOAD'
    assert not (directory / 'config.json').exists() and not fetch.local(name)
    with pytest.raises(fetch.DownloadError, match='^run: wsbench fetch transfo-xl-wt103$'):
        load_model(name)

    monkeypatch.setitem(fetch.SNAPSHOTS, name, pinned)            # the hub's hashes again
    (directory / 'config.json').write_bytes(files['config.json'])
    fetch.verify(name)
    fetch.fetch(name)                                             # every file present and correct: nothing to download
    assert hub.requests == []


def test_verify_finds_a_missing_file_before_it_hashes_the_others(wt103_hub, monkeypatch):
    hub = wt103_hub
    hub.directory.mkdir(parents=True)
    for file in ('config.json', 'pytorch_model.bin'):             # vocab.json is missing (an interrupted fetch)
        (hub.directory / file).write_bytes(hub.files[file])
    hashed = []
    monkeypatch.setattr(fetch, '_check', lambda path, *args: hashed.append(path.name))
    with pytest.raises(fetch.DownloadError, match='^run: wsbench fetch transfo-xl-wt103$'):
        fetch.verify(hub.name)
    assert hashed == []                                           # no statement hashes 1.1 GB it cannot use


def test_fetch_downloads_a_corrupt_file_again_and_leaves_no_partial_files(wt103_hub):
    hub = wt103_hub
    directory, files = hub.directory, hub.files
    directory.mkdir(parents=True)
    (directory / 'config.json').write_bytes(files['config.json'])                 # correct: kept, not downloaded
    (directory / 'vocab.json').write_bytes(b'corrupt')                            # wrong hash: downloaded again
    fetch.partial_path(directory, 'pytorch_model.bin').write_bytes(b'left by a fetch killed with SIGKILL')
    (directory / 'tmpq8_x2kd0').write_bytes(b'left by a killed fetch of an earlier version')
    (directory / 'tmp-notes.txt').write_bytes(b'not a temporary file of fetch()')
    seen = []
    fetch.fetch(hub.name, progress=lambda *args: seen.append(args))
    assert hub.requests == ['pytorch_model.bin', 'vocab.json']
    assert sorted(p.name for p in directory.iterdir()) == sorted([*files, 'tmp-notes.txt'])
    assert all((directory / file).read_bytes() == data for file, data in files.items())
    fetch.verify(hub.name)
    sizes = {file: len(data) for file, data in files.items()}
    assert {file for file, _, _ in seen} == {'pytorch_model.bin', 'vocab.json'}
    assert all(total == sizes[file] and 0 < received <= total for file, received, total in seen)
    assert seen[-1] == ('vocab.json', sizes['vocab.json'], sizes['vocab.json'])


def test_an_interrupted_download_leaves_no_partial_file(wt103_hub):
    import httpx

    hub = wt103_hub

    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield hub.files['pytorch_model.bin'][:1000]
            raise httpx.ReadError('connection lost')

    hub.serve['pytorch_model.bin'] = lambda request: httpx.Response(200, stream=Broken())
    with pytest.raises(fetch.DownloadError, match='^connection lost; run: wsbench fetch transfo-xl-wt103$'):
        fetch.fetch(hub.name)
    assert [p.name for p in hub.directory.iterdir()] == ['config.json']          # downloaded before the failure
    hub.serve['pytorch_model.bin'] = hub.files['pytorch_model.bin']
    fetch.fetch(hub.name)
    assert sorted(p.name for p in hub.directory.iterdir()) == sorted(hub.files)
    assert hub.requests == ['config.json', 'pytorch_model.bin', 'pytorch_model.bin', 'vocab.json']


def test_a_mismatching_download_is_not_kept(wt103_hub):
    hub = wt103_hub
    hub.serve['vocab.json'] = b'not the pinned vocabulary'
    with pytest.raises(fetch.DownloadError, match='^SHA-256 mismatch; run: wsbench fetch transfo-xl-wt103$'):
        fetch.fetch(hub.name)
    assert sorted(p.name for p in hub.directory.iterdir()) == ['config.json', 'pytorch_model.bin']


def test_the_fetch_command_shows_progress_only_on_a_terminal(monkeypatch, capsys):
    import sys
    from wsbench import cli

    calls = []
    monkeypatch.setattr(fetch, 'fetch', lambda name, progress=None: calls.append((name, progress)))
    assert cli.main(['fetch', 'transfo-xl-wt103']) == 0
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    assert cli.main(['fetch', 'transfo-xl-wt103']) == 0
    assert calls == [('transfo-xl-wt103', None), ('transfo-xl-wt103', cli._progress)]
    cli._progress('config.json', 856, 856)                                       # below 1 MiB: no line
    cli._progress('pytorch_model.bin', 512 * 2**20, 1088 * 2**20)
    cli._progress('pytorch_model.bin', 1088 * 2**20, 1088 * 2**20)
    assert capsys.readouterr().err == ('\rpytorch_model.bin: 512.0 MiB of 1088.0 MiB'
                                       '\rpytorch_model.bin: 1088.0 MiB of 1088.0 MiB\n')


# ---------------------------------------------------------------- the real checkpoint

@pytest.mark.network
def test_wt103_reference():
    fetch.fetch('transfo-xl-wt103')            # checks the three pinned files (downloads 1.1 GB if one is missing)
    ref = json.loads((DATA / 'txl_wt103_probe.json').read_text())   # HF transformers 4.46.3, float64
    model = load_model('transfo-xl-wt103')
    cfg, W0, ids = model.cfg, model.W0, model.ids
    assert len(W0) == 249 and sum(v.numel() for v in W0.values()) == 285_205_322
    assert len(cfg.inv_freq) == 512 and ids.tolist() == [ref['ids']] and ids.shape == (1, 122)
    assert max_diff(txl.next_token_nll(model.logits0, ids), torch.tensor(ref['nll'], dtype=torch.float64)) <= 1e-9
    assert model.loss0 == pytest.approx(ref['loss'], abs=1e-9)
    assert model.loss0 == pytest.approx(4.724939573, abs=1e-8)
    zero = txl.forward(cfg, W0, ids, zero_mems=cfg.mem_len)       # HF's default call, mems=None
    assert txl.next_token_loss(zero, ids) == pytest.approx(4.135668769365873, abs=1e-9)
    assert max_diff(txl.next_token_nll(zero, ids), torch.tensor(ref['zero_mems_nll'], dtype=torch.float64)) <= 1e-9
    del zero
    moved = txl.perm_heads(cfg, W0, 9, torch.randperm(16, generator=gen(0)))
    assert sum(moved[k] is not W0[k] for k in W0) == 5
    rel = max_diff(txl.forward(cfg, moved, ids), model.logits0) / model.max_logit0
    assert rel <= 1e-9


@pytest.mark.network
def test_wt103_example_reproduces_the_language_card():
    """wsbench/examples/transformer_xl.wsl on the real checkpoint: the verdicts and the numbers of the card's FACTS."""
    fetch.fetch('transfo-xl-wt103')
    script = (Path(__file__).parents[1] / 'wsbench' / 'examples' / 'transformer_xl.wsl').read_text()
    start = perf_counter()
    rows = {v['label']: v for v in run_script(script)}
    seconds = perf_counter() - start
    assert list(rows) == [f'X{n}' for n in range(1, 17)] and all(v['error'] is None for v in rows.values())
    for n in range(1, 7):                        # exact symmetries
        row = rows[f'X{n}']
        assert (row['status'], row['measured']) == ('CONFIRMED', 'exact') and row['metrics']['rel'] <= 1e-13
    for n in (7, 8):                             # rotating MLP neurons and neg(L) are not symmetries
        assert (rows[f'X{n}']['status'], rows[f'X{n}']['measured']) == ('REFUTED', 'edit')
    deltas = {'X7': 4.7240, 'X8': 4.7256, 'X9': -0.0498, 'X10': 3.5286, 'X11': 4.5838, 'X12': -0.0083, 'X13': 0.7036,
              'X14': 0.9440, 'X15': 4.7314, 'X16': -0.0048}
    for label, dloss in deltas.items():
        metrics = rows[label]['metrics']
        assert metrics['dloss'] == pytest.approx(dloss, abs=5e-4), label
        if label not in ('X7', 'X8'):
            assert rows[label]['status'] == 'MEASURED' and metrics['control']['ratio'] is not None
    for label in ('X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15'):
        assert f"{rows[label]['metrics']['dloss']:+.2f}" in LANGUAGE_CARD, label
    assert rows['X9']['metrics']['control']['dloss'] == pytest.approx(0.1509, abs=5e-4)     # NOTES.md: control +0.15
    assert f"base loss {rows['X1']['metrics']['loss0']:.4f}" in LANGUAGE_CARD
    swap = rows['X10']['metrics']['changed_tensors']
    assert len(swap) == 26 and {t['name'].split('.')[2] for t in swap} == {'8', '9'}
    print(f'16 transfo-xl-wt103 statements: {seconds:.1f} s')
