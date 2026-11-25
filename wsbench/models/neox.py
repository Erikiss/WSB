"""Reference: GPT-NeoX forward pass and weight-space operators, float64, plain torch.

A model state is (W, norm_kind): W maps parameter names to float64 tensors, norm_kind is 'ln' or 'rms'.
Every operator is a pure function W -> W (it returns a new dict and never mutates its input).
"""
import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

P = 'gpt_neox.layers.{}.'
READERS = ['attention.query_key_value', 'mlp.dense_h_to_4h']   # read the residual stream through a norm
WRITERS = ['attention.dense', 'mlp.dense_4h_to_h']             # add into the residual stream


@dataclass(frozen=True)
class NeoxCfg:
    d: int            # hidden_size
    n_heads: int      # num_attention_heads
    n_layers: int     # num_hidden_layers
    d_mlp: int        # intermediate_size
    vocab: int        # vocab_size
    rotary_pct: float
    rope_base: float  # rotary_emb_base
    eps: float        # layer_norm_eps

    @property
    def dh(self):     # head dimension
        return self.d // self.n_heads

    @property
    def rd(self):     # rotary dimensions per head (the first rd dims of each head)
        return int(self.dh * self.rotary_pct)


def cfg_from_hf_config(c):
    assert c['use_parallel_residual'] is True and c['hidden_act'] == 'gelu'
    return NeoxCfg(d=c['hidden_size'], n_heads=c['num_attention_heads'], n_layers=c['num_hidden_layers'],
                   d_mlp=c['intermediate_size'], vocab=c['vocab_size'], rotary_pct=c['rotary_pct'],
                   rope_base=c.get('rotary_emb_base', c.get('rope_theta', 10000)), eps=c['layer_norm_eps'])


TINY_CFG = NeoxCfg(d=16, n_heads=2, n_layers=4, d_mlp=64, vocab=64, rotary_pct=0.5, rope_base=10000.0, eps=1e-5)


def make_tiny(seed=0, cfg=TINY_CFG):
    """Random weights for offline tests. Deterministic given the seed."""
    g = torch.Generator().manual_seed(seed)
    n = lambda *s, std=0.2: torch.randn(*s, generator=g, dtype=torch.float64) * std
    D, V, DM = cfg.d, cfg.vocab, cfg.d_mlp
    W = {'gpt_neox.embed_in.weight': n(V, D, std=1.0), 'embed_out.weight': n(V, D, std=0.5),
         'gpt_neox.final_layer_norm.weight': 1 + n(D, std=0.1), 'gpt_neox.final_layer_norm.bias': n(D, std=0.1)}
    for i in range(cfg.n_layers):
        p = P.format(i)
        for ln in ('input_layernorm', 'post_attention_layernorm'):
            W[p + ln + '.weight'] = 1 + n(D, std=0.1)
            W[p + ln + '.bias'] = n(D, std=0.1)
        W[p + 'attention.query_key_value.weight'] = n(3 * D, D, std=D ** -0.5)
        W[p + 'attention.query_key_value.bias'] = n(3 * D, std=0.1)
        W[p + 'attention.dense.weight'] = n(D, D, std=D ** -0.5)
        W[p + 'attention.dense.bias'] = n(D, std=0.1)
        W[p + 'mlp.dense_h_to_4h.weight'] = n(DM, D, std=D ** -0.5)
        W[p + 'mlp.dense_h_to_4h.bias'] = n(DM, std=0.1)
        W[p + 'mlp.dense_4h_to_h.weight'] = n(D, DM, std=DM ** -0.5)
        W[p + 'mlp.dense_4h_to_h.bias'] = n(D, std=0.1)
    return W


TINY_IDS = [(7 * i + 3) % 64 for i in range(32)]


# ---------------------------------------------------------------- forward

def _norm(cfg, x, w, b, kind):
    if kind == 'ln':
        return F.layer_norm(x, (cfg.d,), w, b, cfg.eps)
    if kind == 'rms':
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + cfg.eps) * w + b
    raise ValueError(kind)


def _rope(cfg, T):
    rd = cfg.rd
    inv = 1.0 / (cfg.rope_base ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    fr = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    emb = torch.cat([fr, fr], -1)
    return emb.cos(), emb.sin()


def _apply_rope(cfg, x, cos, sin):          # x [B,H,T,DH]; rotate the first rd dims, pass the rest
    rd = cfg.rd
    xr, xp = x[..., :rd], x[..., rd:]
    a, b = xr[..., :rd // 2], xr[..., rd // 2:]
    rot_half = torch.cat([-b, a], -1)
    return torch.cat([xr * cos + rot_half * sin, xp], -1)


def forward(cfg, W, ids, kind='ln', return_resid=False):
    """ids: LongTensor [B,T]. Returns logits [B,T,V] (float64); optionally the residual stream after each layer."""
    B, T = ids.shape
    D, H, DH = cfg.d, cfg.n_heads, cfg.dh
    cos, sin = _rope(cfg, T)
    mask = torch.full((T, T), float('-inf'), dtype=torch.float64).triu(1)
    x = W['gpt_neox.embed_in.weight'][ids]
    resid = [x]
    for i in range(cfg.n_layers):
        p = P.format(i)
        h = _norm(cfg, x, W[p + 'input_layernorm.weight'], W[p + 'input_layernorm.bias'], kind)
        qkv = h @ W[p + 'attention.query_key_value.weight'].T + W[p + 'attention.query_key_value.bias']
        qkv = qkv.view(B, T, H, 3 * DH).permute(0, 2, 1, 3)       # per head: [q | k | v], each DH wide
        q, k, v = qkv[..., :DH], qkv[..., DH:2 * DH], qkv[..., 2 * DH:]
        q, k = _apply_rope(cfg, q, cos, sin), _apply_rope(cfg, k, cos, sin)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(DH) + mask, -1) @ v
        a = a.permute(0, 2, 1, 3).reshape(B, T, D)
        attn = a @ W[p + 'attention.dense.weight'].T + W[p + 'attention.dense.bias']
        h2 = _norm(cfg, x, W[p + 'post_attention_layernorm.weight'], W[p + 'post_attention_layernorm.bias'], kind)
        mlp = F.gelu(h2 @ W[p + 'mlp.dense_h_to_4h.weight'].T + W[p + 'mlp.dense_h_to_4h.bias'])
        mlp = mlp @ W[p + 'mlp.dense_4h_to_h.weight'].T + W[p + 'mlp.dense_4h_to_h.bias']
        x = x + attn + mlp                                         # parallel residual
        resid.append(x)
    x = _norm(cfg, x, W['gpt_neox.final_layer_norm.weight'], W['gpt_neox.final_layer_norm.bias'], kind)
    logits = x @ W['embed_out.weight'].T
    if 'embed_out.bias' in W:                                      # exists only after fold_ln
        logits = logits + W['embed_out.bias']
    return (logits, resid) if return_resid else logits


def next_token_loss(logits, ids):
    return F.cross_entropy(logits[0, :-1], ids[0, 1:]).item()


# ---------------------------------------------------------------- random group elements

def rand_orth(n, g):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def rand_orth_fixing_ones(n, g):
    """Random R in O(n) with R @ ones = ones."""
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    A[:, 0] = 1.0
    Hb, _ = torch.linalg.qr(A)
    if Hb[:, 0].sum() < 0:
        Hb = -Hb
    Q = torch.eye(n, dtype=torch.float64)
    Q[1:, 1:] = rand_orth(n - 1, g)
    return Hb @ Q @ Hb.T


def rand_gl(n, g):
    """Random well-conditioned invertible matrix."""
    return torch.randn(n, n, generator=g, dtype=torch.float64) / math.sqrt(n) + 2 * torch.eye(n, dtype=torch.float64)


def rand_perm(n, g):
    return torch.randperm(n, generator=g)


def rand_uniform(n, lo, hi, g):
    return lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------- operators (pure: W -> W)

def cp(W):
    return {k: v.clone() for k, v in W.items()}


def _ln_of(reader):
    return 'input_layernorm' if reader.startswith('attention') else 'post_attention_layernorm'


def fold_ln(cfg, W):
    """Move every norm's gamma/beta into the following linear map. Adds the new parameter embed_out.bias."""
    W = cp(W)
    D = cfg.d
    for i in range(cfg.n_layers):
        p = P.format(i)
        for r in READERS:
            g_, b_ = W[p + _ln_of(r) + '.weight'], W[p + _ln_of(r) + '.bias']
            W[p + r + '.bias'] = W[p + r + '.bias'] + W[p + r + '.weight'] @ b_
            W[p + r + '.weight'] = W[p + r + '.weight'] * g_
        for n in ('input_layernorm', 'post_attention_layernorm'):
            W[p + n + '.weight'] = torch.ones(D, dtype=torch.float64)
            W[p + n + '.bias'] = torch.zeros(D, dtype=torch.float64)
    g_, b_ = W['gpt_neox.final_layer_norm.weight'], W['gpt_neox.final_layer_norm.bias']
    W['embed_out.bias'] = W.get('embed_out.bias', torch.zeros(W['embed_out.weight'].shape[0], dtype=torch.float64)) \
        + W['embed_out.weight'] @ b_
    W['embed_out.weight'] = W['embed_out.weight'] * g_
    W['gpt_neox.final_layer_norm.weight'] = torch.ones(D, dtype=torch.float64)
    W['gpt_neox.final_layer_norm.bias'] = torch.zeros(D, dtype=torch.float64)
    return W


def center_writers(cfg, W):
    """Remove the all-ones component of everything written to the residual stream."""
    W = cp(W)
    C = torch.eye(cfg.d, dtype=torch.float64) - 1.0 / cfg.d
    W['gpt_neox.embed_in.weight'] = W['gpt_neox.embed_in.weight'] @ C
    for i in range(cfg.n_layers):
        p = P.format(i)
        for w in WRITERS:
            W[p + w + '.weight'] = C @ W[p + w + '.weight']
            W[p + w + '.bias'] = C @ W[p + w + '.bias']
    return W


def rotate_resid(cfg, W, R, layers=None, embed=True, unembed=True):
    """x -> R x on the residual stream: writers W -> R W (and biases), readers W -> W R^T."""
    W = cp(W)
    if embed:
        W['gpt_neox.embed_in.weight'] = W['gpt_neox.embed_in.weight'] @ R.T
    for i in (range(cfg.n_layers) if layers is None else layers):
        p = P.format(i)
        for r in READERS:
            W[p + r + '.weight'] = W[p + r + '.weight'] @ R.T
        for w in WRITERS:
            W[p + w + '.weight'] = R @ W[p + w + '.weight']
            W[p + w + '.bias'] = R @ W[p + w + '.bias']
    if unembed:
        W['embed_out.weight'] = W['embed_out.weight'] @ R.T
    return W


def scale_resid(cfg, W, c):
    """Multiply the embedding and every writer (weights and biases) by c."""
    W = cp(W)
    W['gpt_neox.embed_in.weight'] = W['gpt_neox.embed_in.weight'] * c
    for i in range(cfg.n_layers):
        p = P.format(i)
        for w in WRITERS:
            W[p + w + '.weight'] = W[p + w + '.weight'] * c
            W[p + w + '.bias'] = W[p + w + '.bias'] * c
    return W


def perm_mlp(cfg, W, i, perm):
    W = cp(W); p = P.format(i)
    W[p + 'mlp.dense_h_to_4h.weight'] = W[p + 'mlp.dense_h_to_4h.weight'][perm]
    W[p + 'mlp.dense_h_to_4h.bias'] = W[p + 'mlp.dense_h_to_4h.bias'][perm]
    W[p + 'mlp.dense_4h_to_h.weight'] = W[p + 'mlp.dense_4h_to_h.weight'][:, perm]
    return W


def scale_mlp(cfg, W, i, c):
    """Per-neuron positive rescale: input side * c, output side / c. Exact only for homogeneous activations (not GELU)."""
    W = cp(W); p = P.format(i)
    W[p + 'mlp.dense_h_to_4h.weight'] = W[p + 'mlp.dense_h_to_4h.weight'] * c[:, None]
    W[p + 'mlp.dense_h_to_4h.bias'] = W[p + 'mlp.dense_h_to_4h.bias'] * c
    W[p + 'mlp.dense_4h_to_h.weight'] = W[p + 'mlp.dense_4h_to_h.weight'] / c[None, :]
    return W


def rotate_mlp(cfg, W, i, Q):
    """Orthogonal mix of the hidden units (not a symmetry through GELU)."""
    W = cp(W); p = P.format(i)
    W[p + 'mlp.dense_h_to_4h.weight'] = Q @ W[p + 'mlp.dense_h_to_4h.weight']
    W[p + 'mlp.dense_h_to_4h.bias'] = Q @ W[p + 'mlp.dense_h_to_4h.bias']
    W[p + 'mlp.dense_4h_to_h.weight'] = W[p + 'mlp.dense_4h_to_h.weight'] @ Q.T
    return W


def qkv_rows(cfg, h, part):
    """Fused QKV layout: head h owns rows [h*3*DH, (h+1)*3*DH); inside, part 0 = q, 1 = k, 2 = v (DH rows each)."""
    s = h * 3 * cfg.dh + part * cfg.dh
    return slice(s, s + cfg.dh)


def perm_heads(cfg, W, i, perm):
    W = cp(W); p = P.format(i); DH = cfg.dh
    qw, qb = W[p + 'attention.query_key_value.weight'], W[p + 'attention.query_key_value.bias']
    ow = W[p + 'attention.dense.weight']
    W[p + 'attention.query_key_value.weight'] = torch.cat([qw[h * 3 * DH:(h + 1) * 3 * DH] for h in perm])
    W[p + 'attention.query_key_value.bias'] = torch.cat([qb[h * 3 * DH:(h + 1) * 3 * DH] for h in perm])
    W[p + 'attention.dense.weight'] = torch.cat([ow[:, h * DH:(h + 1) * DH] for h in perm], 1)
    return W


def vo_gauge(cfg, W, i, h, N):
    """v -> N v, W_O -> W_O N^-1 for head h of layer i."""
    W = cp(W); p = P.format(i); r = qkv_rows(cfg, h, 2); DH = cfg.dh
    W[p + 'attention.query_key_value.weight'][r] = N @ W[p + 'attention.query_key_value.weight'][r]
    W[p + 'attention.query_key_value.bias'][r] = N @ W[p + 'attention.query_key_value.bias'][r]
    o = W[p + 'attention.dense.weight']
    o[:, h * DH:(h + 1) * DH] = o[:, h * DH:(h + 1) * DH] @ torch.linalg.inv(N)
    return W


def qk_gauge(cfg, W, i, h, A, dims):
    """q_d -> A q_d, k_d -> A^-T k_d on the head dims listed in `dims` (LongTensor)."""
    W = cp(W); p = P.format(i)
    qw, qb = W[p + 'attention.query_key_value.weight'], W[p + 'attention.query_key_value.bias']
    rq, rk = qkv_rows(cfg, h, 0), qkv_rows(cfg, h, 1)
    iq = torch.arange(rq.start, rq.stop)[dims]
    ik = torch.arange(rk.start, rk.stop)[dims]
    AinvT = torch.linalg.inv(A).T
    qw[iq] = A @ qw[iq]; qb[iq] = A @ qb[iq]
    qw[ik] = AinvT @ qw[ik]; qb[ik] = AinvT @ qb[ik]
    return W


def rope_pairs(cfg, W, i, h, phis, scale):
    """The same 2x2 rotation phi_j on each RoPE pair (j, j + rd/2) of q AND k, then q * scale, k / scale."""
    assert len(phis) == cfg.rd // 2
    W = cp(W); p = P.format(i)
    qw, qb = W[p + 'attention.query_key_value.weight'], W[p + 'attention.query_key_value.bias']
    for part, s in ((0, scale), (1, 1.0 / scale)):
        base = qkv_rows(cfg, h, part).start
        for j, phi in enumerate(phis):
            a, b = base + j, base + j + cfg.rd // 2
            c, sn = math.cos(phi), math.sin(phi)
            for M in (qw, qb):
                ra, rb = M[a].clone(), M[b].clone()
                M[a] = s * (c * ra - sn * rb)
                M[b] = s * (sn * ra + c * rb)
        # non-rotary dims of this head are scaled too, so that q.k is unchanged on them
        rows = qkv_rows(cfg, h, part)
        for M in (qw, qb):
            M[rows.start + cfg.rd:rows.stop] = M[rows.start + cfg.rd:rows.stop] * s
    return W


# ---------------------------------------------------------------- layer values

def get_layer(W, i):
    """A layer value: dict of that layer's parameters, keys relative to the layer prefix."""
    pre = P.format(i)
    return {k[len(pre):]: W[k].clone() for k in W if k.startswith(pre)}


def set_layer(W, i, layer):
    W = cp(W)
    for k, v in layer.items():
        W[P.format(i) + k] = v.clone()
    return W


def zero_writers(W, layers):
    W = cp(W)
    for i in layers:
        for w in WRITERS:
            W[P.format(i) + w + '.weight'].zero_()
            W[P.format(i) + w + '.bias'].zero_()
    return W


def lerp_layer(X, Y, t):
    return {k: (1 - t) * X[k] + t * Y[k] for k in X}


def neg_writers(X):
    return {k: (-v if any(k.startswith(w) for w in WRITERS) else v.clone()) for k, v in X.items()}


def neg_all(X):
    return {k: (-v if 'layernorm' not in k else v.clone()) for k, v in X.items()}


def _layer_as_W(cfg, X):
    return {P.format(0) + k: v for k, v in X.items()}


def align_layer(cfg, X, Y):
    """Return X permuted (heads, then MLP neurons) to best match Y. A within-layer gauge move."""
    from scipy.optimize import linear_sum_assignment
    DH, H = cfg.dh, cfg.n_heads

    def head_sig(L_, h):
        w = L_['attention.query_key_value.weight'][h * 3 * DH:(h + 1) * 3 * DH]
        return torch.cat([w.flatten(), L_['attention.dense.weight'][:, h * DH:(h + 1) * DH].flatten()])

    sim = torch.tensor([[float(head_sig(Y, j) @ head_sig(X, k)) for k in range(H)] for j in range(H)])
    if H <= 8:
        best = max(itertools.permutations(range(H)), key=lambda pm: sum(float(sim[j, pm[j]]) for j in range(H)))
    else:
        _, col = linear_sum_assignment(-sim.numpy())
        best = tuple(int(c) for c in col)
    Wx = perm_heads(cfg, _layer_as_W(cfg, X), 0, list(best))
    Xh = get_layer(Wx, 0)
    fy = torch.cat([Y['mlp.dense_h_to_4h.weight'], Y['mlp.dense_h_to_4h.bias'][:, None], Y['mlp.dense_4h_to_h.weight'].T], 1)
    fx = torch.cat([Xh['mlp.dense_h_to_4h.weight'], Xh['mlp.dense_h_to_4h.bias'][:, None], Xh['mlp.dense_4h_to_h.weight'].T], 1)
    _, col = linear_sum_assignment(-(fy @ fx.T).numpy())
    Wx = perm_mlp(cfg, _layer_as_W(cfg, Xh), 0, torch.as_tensor(col))
    return get_layer(Wx, 0)


# ARC grid primitives on 2-D torch tensors (same semantics as PRIMITIVES in arc/grid.py; rot90 is clockwise).
TORCH_PRIMITIVES = {
    'identity': lambda b: b,
    'rot90':    lambda b: torch.rot90(b, -1, (0, 1)),
    'rot180':   lambda b: torch.rot90(b, 2, (0, 1)),
    'rot270':   lambda b: torch.rot90(b, 1, (0, 1)),
    'hmirror':  lambda b: b.flip(0),
    'vmirror':  lambda b: b.flip(1),
    'dmirror':  lambda b: b.T,
    'cmirror':  lambda b: torch.rot90(b, 2, (0, 1)).T,
}
SQUARE_ONLY = ('rot90', 'rot270', 'dmirror', 'cmirror')


def gridop_block(W, name, key, r0, r1, c0, c1, grid_fn=None):
    """Replace the block W[key][r0:r1, c0:c1] by grid_fn(block). The block is cloned before and after the
    primitive, so views such as .T never alias the tensor that is written to."""
    grid_fn = TORCH_PRIMITIVES[name] if grid_fn is None else grid_fn
    W = cp(W)
    blk = W[key][r0:r1, c0:c1].clone()
    out = grid_fn(blk).clone()
    if out.shape != blk.shape:
        raise ValueError(f'E_SHAPE {name} changes the block shape {tuple(blk.shape)} -> {tuple(out.shape)}')
    W[key][r0:r1, c0:c1] = out
    return W
