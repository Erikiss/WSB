"""Static model geometry and lazy, immutable CPU model baselines."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from .neox import NeoxCfg
    from .txl import TxlCfg


SHAPES = {
    'neox-tiny': dict(family='neox', n_layers=4, n_heads=2, d=16, d_mlp=64, dh=8, rd=4, vocab=64),
    'pythia-14m': dict(family='neox', n_layers=6, n_heads=4, d=128, d_mlp=512, dh=32, rd=8, vocab=50304),
    'txl-tiny': dict(family='txl', n_layers=4, n_heads=2, d=16, d_mlp=64, dh=8, vocab=64, d_embed=16,
                     cutoffs=(16, 32, 48), div_val=2, tie_projs=(False, True, True, True)),
    'transfo-xl-wt103': dict(family='txl', n_layers=18, n_heads=16, d=1024, d_mlp=4096, dh=64, vocab=267735,
                             d_embed=1024, cutoffs=(20000, 40000, 200000), div_val=4,
                             tie_projs=(False, True, True, True)),
    'modadd': dict(family='modadd'),
}
SHAPES.update({f'grid:task{i:03}': dict(family='grid') for i in range(1, 401)})


def param_shapes(S):
    """Parameter name -> shape for a SHAPES entry, in state-dict order."""
    if S.get('family') == 'txl':
        return _txl_shapes(S)
    d, m, vocab = S['d'], S['d_mlp'], S['vocab']
    out = {
        'gpt_neox.embed_in.weight': (vocab, d),
        'embed_out.weight': (vocab, d),
        'gpt_neox.final_layer_norm.weight': (d,),
        'gpt_neox.final_layer_norm.bias': (d,),
    }
    for i in range(S['n_layers']):
        prefix = f'gpt_neox.layers.{i}.'
        for ln in ('input_layernorm', 'post_attention_layernorm'):
            out[prefix + ln + '.weight'] = (d,)
            out[prefix + ln + '.bias'] = (d,)
        for name, rows, cols in [
            ('attention.query_key_value', 3 * d, d),
            ('attention.dense', d, d),
            ('mlp.dense_h_to_4h', m, d),
            ('mlp.dense_4h_to_h', d, m),
        ]:
            out[prefix + name + '.weight'] = (rows, cols)
            out[prefix + name + '.bias'] = (rows,)
    return out


def _txl_shapes(S):
    """Transformer-XL (HF TransfoXLLMHeadModel names). Tied tensors appear once, under the embedding's name: the
    softmax weights crit.out_layers.{c}.weight and the tied projections crit.out_projs.{c} are not keys."""
    d, dh, h, m, de, div = S['d'], S['dh'], S['n_heads'], S['d_mlp'], S['d_embed'], S['div_val']
    ends = (0, *S['cutoffs'], S['vocab'])
    clusters = range(len(ends) - 1)
    rows = [ends[c + 1] - ends[c] for c in clusters]
    width = [de // div ** c for c in clusters]
    out = {f'transformer.word_emb.emb_layers.{c}.weight': (rows[c], width[c]) for c in clusters}
    out.update({f'transformer.word_emb.emb_projs.{c}': (d, width[c]) for c in clusters})
    for i in range(S['n_layers']):
        a, f = f'transformer.layers.{i}.dec_attn.', f'transformer.layers.{i}.pos_ff.'
        out.update({
            a + 'r_r_bias': (h, dh), a + 'r_w_bias': (h, dh), a + 'qkv_net.weight': (3 * h * dh, d),
            a + 'o_net.weight': (d, h * dh), a + 'layer_norm.weight': (d,), a + 'layer_norm.bias': (d,),
            a + 'r_net.weight': (h * dh, d),
            f + 'CoreNet.0.weight': (m, d), f + 'CoreNet.0.bias': (m,), f + 'CoreNet.3.weight': (d, m),
            f + 'CoreNet.3.bias': (d,), f + 'layer_norm.weight': (d,), f + 'layer_norm.bias': (d,),
        })
    if len(ends) > 2:
        out['crit.cluster_weight'] = (len(ends) - 2, de)
        out['crit.cluster_bias'] = (len(ends) - 2,)
    out.update({f'crit.out_layers.{c}.bias': (rows[c],) for c in clusters})
    out.update({f'crit.out_projs.{c}': (d, width[c]) for c in clusters if not S['tie_projs'][c]})
    return out


def txl_tied(S):
    """Transformer-XL: HF state-dict name of each tied duplicate -> the name its tensor is stored under (once)."""
    clusters = range(len(S['cutoffs']) + 1)
    tied = {f'crit.out_layers.{c}.weight': f'transformer.word_emb.emb_layers.{c}.weight' for c in clusters}
    tied.update({f'crit.out_projs.{c}': f'transformer.word_emb.emb_projs.{c}' for c in clusters if S['tie_projs'][c]})
    return tied


_POST_LN = 'is not defined for post-LayerNorm Transformer-XL'
# WSL operations without a Transformer-XL meaning. Validation rejects them with E_MODEL and this message; the keys
# name the operation (rotate_resid = rotate(resid, ...), qk_pass = qk(..., dims=pass)). Torch-free, for wsbench.lang.
TXL_UNDEFINED = {
    'fold_ln': f'fold_ln {_POST_LN}',
    'center_writers': f'center_writers {_POST_LN}',
    'ln_to_rms': f'ln_to_rms {_POST_LN}',
    'rotate_resid': f'rotate(resid, ...) {_POST_LN}: a LayerNorm acts on the residual stream after every sublayer',
    'scale_resid': f'scale(resid, ...) {_POST_LN}: a LayerNorm acts on the residual stream after every sublayer',
    'rope_pairs': 'rope_pairs is not defined for Transformer-XL: it has relative sinusoidal positions, not RoPE',
    'qk_pass': 'qk(..., dims=pass) is not defined for Transformer-XL: it has no rotary dimensions; use dims=all',
}


PROBE_TEXT = ("The river leaves the mountains in late spring, when the snow begins to melt and the water turns "
        "grey with silt. Farmers in the valley wait for this moment, because the flood brings soil to the "
        "fields and fills the canals that run between the villages. In the old town, the market opens at "
        "dawn. Traders set out baskets of bread, cheese and dried fish, and children carry water from the "
        "well to the stalls. By noon the square is crowded, and the sound of bargaining fills the narrow "
        "streets. When the sun goes down, the lamps are lit one by one, and the town falls quiet again. "
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n")
# The same prose in WikiText-103 tokenisation (tokens separated by single spaces), for Transformer-XL.
TXL_PROBE_TEXT = ("The river leaves the mountains in late spring , when the snow begins to melt and the water turns "
        "grey with silt . Farmers in the valley wait for this moment , because the flood brings soil to the fields "
        "and fills the canals that run between the villages . In the old town , the market opens at dawn . Traders "
        "set out baskets of bread , cheese and dried fish , and children carry water from the well to the stalls . "
        "By noon the square is crowded , and the sound of bargaining fills the narrow streets . When the sun goes "
        "down , the lamps are lit one by one , and the town falls quiet again .")


@dataclass(frozen=True)
class Model:
    W0: dict[str, 'Tensor']
    cfg: 'NeoxCfg | TxlCfg'
    ids: 'Tensor'
    logits0: 'Tensor'        # logits; for Transformer-XL the log-probabilities of its adaptive softmax
    loss0: float
    max_logit0: float


_load_lock = RLock()
# Every weight model has a cache slot, so no model evicts another: a reload of transfo-xl-wt103 re-hashes 1.1 GB and
# rebuilds its 2.4 GiB baseline. The two tiny models take kilobytes, so the most this holds is what two slots could
# already hold: pythia-14m and transfo-xl-wt103.
WEIGHT_MODELS = tuple(name for name, S in SHAPES.items() if S['family'] in ('neox', 'txl'))


def load_model(name: str) -> Model:
    from .fetch import SNAPSHOTS, model_dir

    directory = model_dir(name) if name in SNAPSHOTS else None
    with _load_lock:
        return _load_model(name, directory)


@lru_cache(maxsize=len(WEIGHT_MODELS))
def _load_model(name: str, directory: Path | None) -> Model:
    family = SHAPES.get(name, {}).get('family')
    if family == 'neox':
        from . import neox as module
        W0, cfg, ids = _load_neox(name, directory)
    elif family == 'txl':
        from . import txl as module
        W0, cfg, ids = _load_txl(name, directory)
    else:
        raise ValueError(f'unknown NeoX model {name!r}')

    expected = SHAPES[name]
    assert all(getattr(cfg, key) == size for key, size in expected.items() if key != 'family')
    assert {key: tuple(t.shape) for key, t in W0.items()} == param_shapes(expected)
    logits = module.forward(cfg, W0, ids)
    return Model(W0, cfg, ids, logits, module.next_token_loss(logits, ids), logits.abs().max().item())


def _load_neox(name, directory):
    import torch
    from . import neox

    if name == 'neox-tiny':
        return neox.make_tiny(seed=0), neox.TINY_CFG, torch.tensor([neox.TINY_IDS])
    if name != 'pythia-14m':
        raise ValueError(f'unknown NeoX model {name!r}')
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from .fetch import verify_pythia

    verify_pythia(directory)
    cfg = neox.cfg_from_hf_config(json.loads((directory / 'config.json').read_text()))
    W0 = {key: tensor.to(torch.float64) for key, tensor in load_file(directory / 'model.safetensors').items()}
    tokens = Tokenizer.from_file(str(directory / 'tokenizer.json')).encode(PROBE_TEXT).ids
    assert len(tokens) == 149, 'unexpected Pythia probe token count'
    return W0, cfg, torch.tensor([tokens])


def _load_txl(name, directory):
    import torch
    from . import txl

    if name == 'txl-tiny':
        return txl.make_tiny(seed=0), txl.TINY_CFG, torch.tensor([txl.TINY_IDS])
    from .fetch import verify

    verify(name, directory)    # SHA-256 of every file before anything is parsed
    cfg = txl.cfg_from_hf_config(json.loads((directory / 'config.json').read_text('utf-8')))
    state = torch.load(directory / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    W0, cfg = txl.from_hf_state(cfg, state)     # empties `state` while it converts to float64
    del state
    vocab = json.loads((directory / 'vocab.json').read_text('utf-8'))      # word -> id; has non-ASCII words
    tokens = TXL_PROBE_TEXT.split(' ')
    assert len(tokens) == 122 and all(t in vocab for t in tokens), 'unexpected Transformer-XL probe tokenisation'
    return W0, cfg, torch.tensor([[vocab[t] for t in tokens]])
