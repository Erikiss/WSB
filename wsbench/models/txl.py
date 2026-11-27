"""Reference: Transformer-XL forward pass and weight-space operators, float64, plain torch.

The function is that of Hugging Face's deprecated TransfoXLLMHeadModel (attn_type 0: relative positions with learned
biases; post-LayerNorm; ReLU MLP; adaptive input embedding and adaptive softmax with tied weights) on ONE segment
WITHOUT memory, i.e. the first segment of the original Transformer-XL evaluation. HF itself, called without `mems`,
prepends cfg.mem_len all-zero memory slots instead; forward(..., zero_mems=cfg.mem_len) reproduces that for
comparisons with the reference only.

A model state W maps the state-dict names of TransfoXLLMHeadModel to float64 tensors, each tensor once. The softmax
weight crit.out_layers.{c}.weight IS the embedding table transformer.word_emb.emb_layers.{c}.weight, and a tied softmax
projection crit.out_projs.{c} (tie_projs[c]) IS transformer.word_emb.emb_projs.{c}; only the transformer.word_emb names
are keys, so an edit of a tied tensor changes the embedding and the softmax together. The positional frequency table
is not a parameter: it is TxlCfg.inv_freq.

Every operator is a pure function W -> W and copy-on-write: it returns a new dict that shares every tensor it leaves
alone and holds a new tensor for each one it changes, so a 285M-parameter state is never copied as a whole. Nothing
here writes into a tensor it did not create; callers must not either, because the baseline shares its tensors.
"""
import itertools
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F

from .neox import SQUARE_ONLY, TORCH_PRIMITIVES, rand_gl, rand_orth, rand_perm, rand_uniform  # noqa: F401 (shared)
from .registry import TXL_UNDEFINED, txl_tied

EMB = 'transformer.word_emb.emb_layers.{}.weight'   # [vocab_c, d_emb_c]  token table of cluster c = its softmax weight
EPROJ = 'transformer.word_emb.emb_projs.{}'          # [d, d_emb_c]        embedding projection; softmax one if tied
OPROJ = 'crit.out_projs.{}'                          # [d, d_emb_c]        softmax projection of an untied cluster
OBIAS = 'crit.out_layers.{}.bias'                    # [vocab_c]
CLUSTER_W, CLUSTER_B = 'crit.cluster_weight', 'crit.cluster_bias'   # [n_clusters, d_embed], [n_clusters]
INV_FREQ = 'transformer.pos_emb.inv_freq'            # HF buffer, checked at load and kept as TxlCfg.inv_freq

P = 'transformer.layers.{}.'
QKV = 'dec_attn.qkv_net.weight'    # [3*H*dh, d]: all q rows, all k rows, all v rows; head h owns dh rows in each
O = 'dec_attn.o_net.weight'        # [d, H*dh]   (no bias; head h owns dh columns)
R = 'dec_attn.r_net.weight'        # [H*dh, d]   keys of the relative positions (no bias; head h owns dh rows)
RW, RR = 'dec_attn.r_w_bias', 'dec_attn.r_r_bias'   # [H, dh]  query biases of the content and the position term
LN1 = 'dec_attn.layer_norm.'       # post-LN after attention: x = LN(x + attn)
FF0W, FF0B = 'pos_ff.CoreNet.0.weight', 'pos_ff.CoreNet.0.bias'   # [d_mlp, d], [d_mlp]
FF3W, FF3B = 'pos_ff.CoreNet.3.weight', 'pos_ff.CoreNet.3.bias'   # [d, d_mlp], [d]
LN2 = 'pos_ff.layer_norm.'         # post-LN after the MLP: x = LN(x + mlp)
LAYER_KEYS = (RR, RW, QKV, O, LN1 + 'weight', LN1 + 'bias', R, FF0W, FF0B, FF3W, FF3B, LN2 + 'weight', LN2 + 'bias')
WRITERS = (O, FF3W, FF3B)          # add into the residual stream (before the post-LayerNorm)
GEOMETRY = ('n_layers', 'n_heads', 'd', 'dh', 'd_mlp', 'vocab', 'd_embed', 'cutoffs', 'div_val', 'tie_projs')

UNDEFINED = TXL_UNDEFINED   # WSL operations without a Transformer-XL meaning (validation: E_MODEL with this message)


@dataclass(frozen=True)
class TxlCfg:
    n_layers: int        # n_layer
    n_heads: int         # n_head
    d: int               # d_model
    dh: int              # d_head
    d_mlp: int           # d_inner
    vocab: int           # vocab_size
    d_embed: int         # width of the cluster-0 embedding; cluster c has d_embed // div_val**c
    cutoffs: tuple       # cluster boundaries without 0 and vocab: cluster c holds the ids [ends[c], ends[c+1])
    div_val: int
    tie_projs: tuple     # per cluster: the softmax projection is the embedding projection
    eps: float           # layer_norm_epsilon
    clamp_len: int       # relative distances above clamp_len share one position (<= 0: no clamp)
    mem_len: int         # with same_length, a query sees at most mem_len keys
    same_length: bool
    inv_freq: tuple | None = field(default=None, repr=False)   # the checkpoint's float32 table; None: the formula

    def __post_init__(self):
        ends = (0, *self.cutoffs, self.vocab)
        if any(a >= b for a, b in zip(ends, ends[1:])):
            raise ValueError(f'cutoffs must increase strictly inside (0, vocab): {self.cutoffs}')
        if len(self.tie_projs) != len(self.cutoffs) + 1:
            raise ValueError('tie_projs needs one entry per cluster')
        if self.div_val < 2 or self.d_embed // self.div_val ** len(self.cutoffs) < 1:
            raise ValueError('div_val must be >= 2 and leave every cluster at least one embedding dimension')
        if self.d % 2:
            raise ValueError('d_model must be even (sin | cos position embedding)')
        if self.same_length and self.mem_len < 1:
            raise ValueError('same_length needs mem_len >= 1 (with mem_len 0 HF masks every key)')
        if self.inv_freq is not None and len(self.inv_freq) != self.d // 2:
            raise ValueError('inv_freq needs d/2 entries')

    @property
    def n_clusters(self):  # tail clusters; cluster 0 is the head (shortlist)
        return len(self.cutoffs)

    @property
    def bounds(self):      # [(lo, hi)] token-id range of each cluster
        ends = (0, *self.cutoffs, self.vocab)
        return list(zip(ends[:-1], ends[1:]))

    @property
    def d_embs(self):      # embedding width of each cluster
        return [self.d_embed // self.div_val ** c for c in range(self.n_clusters + 1)]


def cfg_from_hf_config(c):
    """TxlCfg from a TransfoXLConfig dict (config.json); ValueError for anything this implementation does not cover."""
    def need(ok, what):
        if not ok:
            raise ValueError(f'unsupported Transformer-XL config: {what}')
    sizes = ('n_layer', 'n_head', 'd_model', 'd_head', 'd_inner', 'vocab_size', 'd_embed', 'cutoffs', 'div_val')
    need(all(k in c for k in sizes), f'it needs {", ".join(sizes)}')
    need(c.get('model_type', 'transfo-xl') == 'transfo-xl', 'model_type must be transfo-xl')
    need(c.get('attn_type', 0) == 0, 'attn_type must be 0 (relative positions with learned biases)')
    need(c.get('pre_lnorm', False) is False, 'pre_lnorm must be false (post-LayerNorm)')
    need(c.get('untie_r', True) is True, 'untie_r must be true (r_w_bias and r_r_bias in every layer)')
    need(c.get('adaptive', True) is True, 'adaptive must be true')
    need(c['div_val'] > 1, 'div_val must be > 1 (with div_val 1 the softmax is one matrix; not implemented)')
    need(c.get('tie_weight', True) is True and c.get('tie_word_embeddings', True) is True,
         'the softmax weights must be tied to the embedding tables')
    need(c.get('sample_softmax', -1) <= 0, 'sample_softmax must be off')
    cutoffs = tuple(c['cutoffs'])
    tie = c.get('tie_projs')
    if tie is None:          # HF: proj_share_all_but_first (default true) ties every projection but the first
        tie = [False] + [bool(c.get('proj_share_all_but_first', True))] * len(cutoffs)
    return TxlCfg(n_layers=c['n_layer'], n_heads=c['n_head'], d=c['d_model'], dh=c['d_head'], d_mlp=c['d_inner'],
                  vocab=c['vocab_size'], d_embed=c['d_embed'], cutoffs=cutoffs, div_val=c['div_val'],
                  tie_projs=tuple(bool(t) for t in tie), eps=c.get('layer_norm_epsilon', 1e-5),
                  clamp_len=c.get('clamp_len', 1000), mem_len=c.get('mem_len', 1600),
                  same_length=bool(c.get('same_length', True)))


# A tiny model that exercises every path: four clusters of widths 16, 8, 4, 2, an untied first softmax projection,
# relative distances clamped at 16 and (same_length) a 24-key window on the 32 probe tokens.
TINY_CFG = TxlCfg(n_layers=4, n_heads=2, d=16, dh=8, d_mlp=64, vocab=64, d_embed=16, cutoffs=(16, 32, 48), div_val=2,
                  tie_projs=(False, True, True, True), eps=1e-5, clamp_len=16, mem_len=24, same_length=True)
TINY_IDS = [(7 * i + 3) % 64 for i in range(32)]   # every cluster occurs as input and as target


def geometry(cfg):
    """The registry's SHAPES entry for this configuration (family and GEOMETRY fields)."""
    return {'family': 'txl', **{k: getattr(cfg, k) for k in GEOMETRY}}


def param_names(cfg):
    """The keys of W in state-dict order: every tensor once, without the tied duplicates and the buffer."""
    from .registry import param_shapes
    return list(param_shapes(geometry(cfg)))


def tied_pairs(cfg):
    """(name in the HF state dict, name it is stored under in W) for every tied tensor."""
    return list(txl_tied(geometry(cfg)).items())


def make_tiny(seed=0, cfg=TINY_CFG):
    """Random weights for offline tests, in the layout above. Deterministic given the seed."""
    g = torch.Generator().manual_seed(seed)
    n = lambda *s, std=0.2: torch.randn(*s, generator=g, dtype=torch.float64) * std
    D, H, DH, DM = cfg.d, cfg.n_heads, cfg.dh, cfg.d_mlp
    W = {}
    # Tables 0.4 and projections de**-0.5: the embedding (times sqrt(d)) and the logits both have std ~1.6.
    for c, ((lo, hi), de) in enumerate(zip(cfg.bounds, cfg.d_embs)):
        W[EMB.format(c)] = n(hi - lo, de, std=0.4)
    for c, de in enumerate(cfg.d_embs):
        W[EPROJ.format(c)] = n(D, de, std=de ** -0.5)
    for i in range(cfg.n_layers):
        p = P.format(i)
        W[p + RR] = n(H, DH, std=0.3)
        W[p + RW] = n(H, DH, std=0.3)
        W[p + QKV] = n(3 * H * DH, D, std=D ** -0.5)
        W[p + O] = n(D, H * DH, std=(H * DH) ** -0.5)
        W[p + LN1 + 'weight'] = 1 + n(D, std=0.1)
        W[p + LN1 + 'bias'] = n(D, std=0.1)
        W[p + R] = n(H * DH, D, std=D ** -0.5)
        W[p + FF0W] = n(DM, D, std=D ** -0.5)
        W[p + FF0B] = n(DM, std=0.1)
        W[p + FF3W] = n(D, DM, std=DM ** -0.5)
        W[p + FF3B] = n(D, std=0.1)
        W[p + LN2 + 'weight'] = 1 + n(D, std=0.1)
        W[p + LN2 + 'bias'] = n(D, std=0.1)
    if cfg.n_clusters:
        W[CLUSTER_W] = n(cfg.n_clusters, cfg.d_embed, std=0.4)
        W[CLUSTER_B] = n(cfg.n_clusters, std=0.3)
    for c, (lo, hi) in enumerate(cfg.bounds):
        W[OBIAS.format(c)] = n(hi - lo, std=0.1)
    for c, de in enumerate(cfg.d_embs):
        if not cfg.tie_projs[c]:
            W[OPROJ.format(c)] = n(D, de, std=de ** -0.5)
    return W


def inv_freq_formula(d):
    return 1 / (10000 ** (torch.arange(0.0, d, 2.0, dtype=torch.float64) / d))


def from_hf_state(cfg, state):
    """(W, cfg) from a TransfoXLLMHeadModel state dict as torch.load returns it.

    Checks that every tied tensor equals the one it is tied to and that the buffer transformer.pos_emb.inv_freq is
    1/10000^(arange(0, d, 2)/d) to float32 precision, then drops the duplicates and the buffer. The returned cfg
    carries the stored table (as float64) in inv_freq, so the forward pass uses the frequencies HF evaluates the
    checkpoint with (the float64 formula would move the wt103 log-probabilities by up to 6e-6). Consumes `state`:
    entries are popped while they are converted, so the float32 copy is released as the float64 one is built.
    """
    names = param_names(cfg)
    expected = set(names) | {dup for dup, _ in tied_pairs(cfg)} | {INV_FREQ}
    if set(state) != expected:
        raise ValueError(f'unexpected Transformer-XL state dict: missing {sorted(expected - set(state))}, '
                         f'unexpected {sorted(set(state) - expected)}')
    for dup, kept in tied_pairs(cfg):
        if not torch.equal(state[dup], state[kept]):
            raise ValueError(f'{dup} is not tied to {kept}')
        del state[dup]
    inv = state.pop(INV_FREQ).to(torch.float64)
    ref = inv_freq_formula(cfg.d)
    if inv.shape != ref.shape or not torch.allclose(inv, ref, rtol=1e-6, atol=0):
        raise ValueError(f'{INV_FREQ} is not the sinusoid table 1/10000^(2k/d)')
    W = {name: state.pop(name).to(torch.float64) for name in names}
    return W, replace(cfg, inv_freq=tuple(inv.tolist()))


# ---------------------------------------------------------------- forward

def _embed(cfg, W, ids):
    """Adaptive input embedding: the row of the token's cluster table times that cluster's projection^T, * sqrt(d)."""
    if int(ids.min()) < 0 or int(ids.max()) >= cfg.vocab:
        raise ValueError('token id out of range')
    flat = ids.reshape(-1)
    x = torch.zeros(flat.numel(), cfg.d, dtype=torch.float64)
    for c, (lo, hi) in enumerate(cfg.bounds):
        sel = ((flat >= lo) & (flat < hi)).nonzero().squeeze(1)
        if sel.numel():
            x[sel] = F.linear(W[EMB.format(c)][flat[sel] - lo], W[EPROJ.format(c)])
    return (x * cfg.d ** 0.5).view(*ids.shape, cfg.d)


def _pos_emb(cfg, K):
    """[K, d] = [sin | cos] of the relative distances K-1, ..., 0 (clamped at clamp_len) times inv_freq."""
    pos = torch.arange(K - 1, -1, -1, dtype=torch.float64)
    if cfg.clamp_len > 0:
        pos = pos.clamp(max=cfg.clamp_len)
    inv = inv_freq_formula(cfg.d) if cfg.inv_freq is None else torch.tensor(cfg.inv_freq, dtype=torch.float64)
    s = torch.outer(pos, inv)
    return torch.cat([s.sin(), s.cos()], -1)


def _mask(cfg, T, M):
    """[T, M+T], True = hidden (the first M keys are memory slots): causal; with same_length each query sees at most
    mem_len keys, exactly as HF builds it."""
    ones = torch.ones(T, M + T, dtype=torch.bool)
    mask = ones.triu(1 + M)
    if cfg.same_length:
        over = M + T - cfg.mem_len
        mask = mask | ones.tril(-(T - over if over > 0 else T))
    return mask


def _rel_shift(x):
    """x[..., i, j] scores query i against position row j (distance K-1-j). Returns y with y[..., i, j] the score of
    key j, i.e. x[..., i, j + T-1-i], wherever key j is visible (Transformer-XL's zero-pad trick)."""
    *lead, T, K = x.shape
    return F.pad(x, (1, 0)).view(*lead, K + 1, T)[..., 1:, :].reshape(*lead, T, K)


def _out_proj(cfg, W, c):
    return W[EPROJ.format(c)] if cfg.tie_projs[c] else W[OPROJ.format(c)]


def _log_softmax(cfg, W, x):
    """Adaptive softmax, FULL log-probabilities [..., V]: the head scores the shortlist and one entry per tail cluster;
    a tail token gets its cluster's head log-prob plus its log-softmax inside the cluster."""
    h = x.reshape(-1, cfg.d)
    ph = h @ _out_proj(cfg, W, 0)
    head = F.linear(ph, W[EMB.format(0)], W[OBIAS.format(0)])
    if cfg.n_clusters:     # concatenate logits, not weights: no copy of the shortlist table per call
        head = torch.cat([head, F.linear(ph, W[CLUSTER_W], W[CLUSTER_B])], -1)
    head = F.log_softmax(head, -1)
    short = cfg.bounds[0][1]
    out = h.new_empty(h.shape[0], cfg.vocab)
    out[:, :short] = head[:, :short]
    for c in range(1, cfg.n_clusters + 1):
        lo, hi = cfg.bounds[c]
        tail = F.log_softmax(F.linear(h @ _out_proj(cfg, W, c), W[EMB.format(c)], W[OBIAS.format(c)]), -1)
        out[:, lo:hi] = tail.add_(head[:, short + c - 1, None])
    return out.view(*x.shape[:-1], cfg.vocab)


def forward(cfg, W, ids, return_resid=False, zero_mems=0):
    """ids: LongTensor [B,T]. Returns log-probabilities [B,T,V] (float64); optionally the residual stream after the
    embedding and after each layer. One segment without memory. zero_mems=m prepends m all-zero memory slots to every
    layer (keys and values 0, scored by position only), which is what HF's TransfoXL does when called without mems
    (m = cfg.mem_len); it exists to compare with that reference."""
    B, T = ids.shape
    D, H, DH, M = cfg.d, cfg.n_heads, cfg.dh, zero_mems
    K = M + T
    x = _embed(cfg, W, ids)
    pos = _pos_emb(cfg, K)
    hidden = _mask(cfg, T, M)
    lowest, scale = torch.finfo(torch.float64).min, 1 / DH ** 0.5
    resid = [x]
    for i in range(cfg.n_layers):
        p = P.format(i)
        q, k, v = (t.view(B, T, H, DH).transpose(1, 2) for t in F.linear(x, W[p + QKV]).chunk(3, -1))
        if M:
            z = x.new_zeros(B, H, M, DH)
            k, v = torch.cat([z, k], 2), torch.cat([z, v], 2)
        r = F.linear(pos, W[p + R]).view(K, H, DH).transpose(0, 1)                # [H, K, DH]
        ac = (q + W[p + RW][:, None]) @ k.transpose(-1, -2)                       # content term [B, H, T, K]
        bd = _rel_shift((q + W[p + RR][:, None]) @ r.transpose(-1, -2))           # position term
        a = torch.softmax(((ac + bd) * scale).masked_fill(hidden, lowest), -1) @ v
        a = F.linear(a.transpose(1, 2).reshape(B, T, H * DH), W[p + O])
        x = F.layer_norm(x + a, (D,), W[p + LN1 + 'weight'], W[p + LN1 + 'bias'], cfg.eps)
        f = F.linear(F.relu(F.linear(x, W[p + FF0W], W[p + FF0B])), W[p + FF3W], W[p + FF3B])
        x = F.layer_norm(x + f, (D,), W[p + LN2 + 'weight'], W[p + LN2 + 'bias'], cfg.eps)
        resid.append(x)
    logprobs = _log_softmax(cfg, W, x)
    return (logprobs, resid) if return_resid else logprobs


def next_token_nll(logprobs, ids):
    """Per-position negative log-likelihood of ids[0, 1:], shape [T-1]."""
    return -logprobs[0, :-1].gather(1, ids[0, 1:, None]).squeeze(1)


def next_token_loss(logprobs, ids):
    return next_token_nll(logprobs, ids).mean().item()


# ---------------------------------------------------------------- operators (pure and copy-on-write: W -> W)

def _cow(W, changes):
    """A new dict with `changes` applied; every other tensor is shared with W."""
    out = dict(W)
    out.update(changes)
    return out


def perm_mlp(cfg, W, i, perm):
    """Permute the MLP neurons of layer i (rows of CoreNet.0 and its bias, columns of CoreNet.3). Exact."""
    p, perm = P.format(i), torch.as_tensor(perm, dtype=torch.long)
    return _cow(W, {p + FF0W: W[p + FF0W][perm], p + FF0B: W[p + FF0B][perm], p + FF3W: W[p + FF3W][:, perm]})


def scale_mlp(cfg, W, i, c):
    """Per-neuron positive rescale: input side * c, output side / c. Exact here: ReLU is positively homogeneous."""
    p = P.format(i)
    return _cow(W, {p + FF0W: W[p + FF0W] * c[:, None], p + FF0B: W[p + FF0B] * c, p + FF3W: W[p + FF3W] / c[None, :]})


def rotate_mlp(cfg, W, i, Q):
    """Orthogonal mix of the hidden units (not a symmetry through ReLU)."""
    p = P.format(i)
    return _cow(W, {p + FF0W: Q @ W[p + FF0W], p + FF0B: Q @ W[p + FF0B], p + FF3W: W[p + FF3W] @ Q.T})


def head_rows(cfg, h, part):
    """qkv_net layout: part 0 = q, 1 = k, 2 = v fills rows [part*H*dh, (part+1)*H*dh); head h owns dh of them."""
    s = part * cfg.n_heads * cfg.dh + h * cfg.dh
    return slice(s, s + cfg.dh)


def perm_heads(cfg, W, i, perm):
    """Permute the heads of layer i: their q, k and v rows, r_net rows, r_w_bias and r_r_bias rows, o_net columns."""
    p, H, DH = P.format(i), cfg.n_heads, cfg.dh
    perm = torch.as_tensor(perm, dtype=torch.long)
    rows = (perm[:, None] * DH + torch.arange(DH)).reshape(-1)     # head-major rows of one H*dh block
    qkv = torch.cat([rows + part * H * DH for part in range(3)])
    return _cow(W, {p + QKV: W[p + QKV][qkv], p + R: W[p + R][rows], p + RW: W[p + RW][perm],
                    p + RR: W[p + RR][perm], p + O: W[p + O][:, rows]})


def vo_gauge(cfg, W, i, h, N):
    """v -> N v, o_net -> o_net N^-1 for head h of layer i. Exact."""
    p, cols = P.format(i), slice(h * cfg.dh, (h + 1) * cfg.dh)
    qkv, o = W[p + QKV].clone(), W[p + O].clone()
    r = head_rows(cfg, h, 2)
    qkv[r] = N @ qkv[r]
    o[:, cols] = o[:, cols] @ torch.linalg.inv(N)
    return _cow(W, {p + QKV: qkv, p + O: o})


def qk_gauge(cfg, W, i, h, A):
    """q -> A q with both query biases (r_w_bias, r_r_bias) -> A b, k -> A^-T k and the head's r_net rows -> A^-T r, on
    all dh dims of head h: (q + r_w_bias).k and (q + r_r_bias).r are unchanged. Exact."""
    p, blk = P.format(i), slice(h * cfg.dh, (h + 1) * cfg.dh)
    qkv, rnet, rw, rr = (W[p + k].clone() for k in (QKV, R, RW, RR))
    AinvT = torch.linalg.inv(A).T
    rq, rk = head_rows(cfg, h, 0), head_rows(cfg, h, 1)
    qkv[rq] = A @ qkv[rq]
    qkv[rk] = AinvT @ qkv[rk]
    rnet[blk] = AinvT @ rnet[blk]
    rw[h] = A @ rw[h]
    rr[h] = A @ rr[h]
    return _cow(W, {p + QKV: qkv, p + R: rnet, p + RW: rw, p + RR: rr})


# ---------------------------------------------------------------- layer values

def get_layer(W, i):
    """A layer value: dict of that layer's parameters, keys relative to the layer prefix. Shares the tensors."""
    pre = P.format(i)
    return {k[len(pre):]: v for k, v in W.items() if k.startswith(pre)}


def set_layer(W, i, layer):
    if set(layer) != set(LAYER_KEYS):
        raise ValueError('a Transformer-XL layer value needs exactly the parameters of one layer')
    return _cow(W, {P.format(i) + k: v for k, v in layer.items()})


def swap_layers(W, i, j):
    a, b = get_layer(W, i), get_layer(W, j)
    return set_layer(set_layer(W, i, b), j, a)


def zero_writers(W, layers):
    """delete: zero what each layer adds to the residual stream (o_net, CoreNet.3 and its bias). The two post-
    LayerNorms stay, so a deleted layer maps x to LN2(LN1(x)), not to x."""
    keys = [P.format(i) + w for i in layers for w in WRITERS]
    return _cow(W, {k: torch.zeros_like(W[k]) for k in keys})


def lerp_layer(X, Y, t):
    return {k: (1 - t) * X[k] + t * Y[k] for k in X}


def neg_writers(X):
    return {k: (-v if k in WRITERS else v) for k, v in X.items()}


def neg_all(X):
    """Negate every parameter but the LayerNorms. Leaves the attention output unchanged; the MLP changes (ReLU)."""
    return {k: (v if k.startswith((LN1, LN2)) else -v) for k, v in X.items()}


def _layer_as_W(X):
    return {P.format(0) + k: v for k, v in X.items()}


def align_layer(cfg, X, Y):
    """Return X permuted (heads, then MLP neurons) to best match Y. A within-layer symmetry: exact. A head signature
    is its q, k, v and r_net rows, both biases and its o_net columns; a neuron's is its CoreNet.0 row, bias and
    CoreNet.3 column. An identity permutation leaves the tensors shared."""
    from scipy.optimize import linear_sum_assignment
    H, DH = cfg.n_heads, cfg.dh

    def head_sig(L_, h):
        blk = slice(h * DH, (h + 1) * DH)
        parts = [L_[QKV][head_rows(cfg, h, part)] for part in range(3)] + [L_[R][blk], L_[O][:, blk]]
        return torch.cat([t.flatten() for t in parts] + [L_[RW][h], L_[RR][h]])

    sim = torch.stack([head_sig(Y, j) for j in range(H)]) @ torch.stack([head_sig(X, k) for k in range(H)]).T
    if H <= 8:
        best = max(itertools.permutations(range(H)), key=lambda pm: sum(float(sim[j, pm[j]]) for j in range(H)))
    else:
        _, col = linear_sum_assignment(-sim.numpy())
        best = tuple(int(c) for c in col)
    if best != tuple(range(H)):
        X = get_layer(perm_heads(cfg, _layer_as_W(X), 0, list(best)), 0)
    fy = torch.cat([Y[FF0W], Y[FF0B][:, None], Y[FF3W].T], 1)
    fx = torch.cat([X[FF0W], X[FF0B][:, None], X[FF3W].T], 1)
    _, col = linear_sum_assignment(-(fy @ fx.T).numpy())
    if (col != range(cfg.d_mlp)).any():
        X = get_layer(perm_mlp(cfg, _layer_as_W(X), 0, torch.as_tensor(col)), 0)
    return X


def gridop_block(W, name, key, r0, r1, c0, c1, grid_fn=None):
    """Replace the block W[key][r0:r1, c0:c1] by grid_fn(block) (an ARC primitive of TORCH_PRIMITIVES by default).
    Only W[key] is copied; the block is cloned before and after the primitive, so views such as .T never alias."""
    grid_fn = TORCH_PRIMITIVES[name] if grid_fn is None else grid_fn
    t = W[key].clone()
    blk = t[r0:r1, c0:c1].clone()
    out = grid_fn(blk).clone()
    if out.shape != blk.shape:
        raise ValueError(f'E_SHAPE {name} changes the block shape {tuple(blk.shape)} -> {tuple(out.shape)}')
    t[r0:r1, c0:c1] = out
    return _cow(W, {key: t})
