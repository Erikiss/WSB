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


SHAPES = {
    'neox-tiny': dict(family='neox', n_layers=4, n_heads=2, d=16, d_mlp=64, dh=8, rd=4, vocab=64),
    'pythia-14m': dict(family='neox', n_layers=6, n_heads=4, d=128, d_mlp=512, dh=32, rd=8, vocab=50304),
    'modadd': dict(family='modadd'),
}
SHAPES.update({f'grid:task{i:03}': dict(family='grid') for i in range(1, 401)})


def param_shapes(S):
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


PROBE_TEXT = ("The river leaves the mountains in late spring, when the snow begins to melt and the water turns "
        "grey with silt. Farmers in the valley wait for this moment, because the flood brings soil to the "
        "fields and fills the canals that run between the villages. In the old town, the market opens at "
        "dawn. Traders set out baskets of bread, cheese and dried fish, and children carry water from the "
        "well to the stalls. By noon the square is crowded, and the sound of bargaining fills the narrow "
        "streets. When the sun goes down, the lamps are lit one by one, and the town falls quiet again. "
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n")


@dataclass(frozen=True)
class Model:
    W0: dict[str, 'Tensor']
    cfg: 'NeoxCfg'
    ids: 'Tensor'
    logits0: 'Tensor'
    loss0: float
    max_logit0: float


_load_lock = RLock()


def load_model(name: str) -> Model:
    from .fetch import pythia_dir

    directory = pythia_dir() if name == 'pythia-14m' else None
    with _load_lock:
        return _load_model(name, directory)


@lru_cache(maxsize=2)
def _load_model(name: str, directory: Path | None) -> Model:
    import torch
    from . import neox

    if name == 'neox-tiny':
        W0 = neox.make_tiny(seed=0)
        cfg = neox.TINY_CFG
        ids = torch.tensor([neox.TINY_IDS])
    elif name == 'pythia-14m':
        from safetensors.torch import load_file
        from tokenizers import Tokenizer
        from .fetch import verify_pythia

        verify_pythia(directory)
        cfg = neox.cfg_from_hf_config(json.loads((directory / 'config.json').read_text()))
        W0 = {key: tensor.to(torch.float64) for key, tensor in load_file(directory / 'model.safetensors').items()}
        tokens = Tokenizer.from_file(str(directory / 'tokenizer.json')).encode(PROBE_TEXT).ids
        assert len(tokens) == 149, 'unexpected Pythia probe token count'
        ids = torch.tensor([tokens])
    else:
        raise ValueError(f'unknown NeoX model {name!r}')

    expected = SHAPES[name]
    assert all(getattr(cfg, key) == size for key, size in expected.items() if key != 'family')
    assert {key: tuple(t.shape) for key, t in W0.items()} == param_shapes(expected)
    logits = neox.forward(cfg, W0, ids)
    return Model(W0, cfg, ids, logits, neox.next_token_loss(logits, ids), logits.abs().max().item())
