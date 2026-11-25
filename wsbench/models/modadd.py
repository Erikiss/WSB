"""Reference: modular-addition grokking model (Nanda et al. 2023, 1-layer, no LayerNorm), trainer, Grokfast-EMA filter."""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_data(p=97, frac=0.5, split_seed=1234):
    """All pairs (a, b) with label (a + b) mod p. Token p is '='. Fixed split independent of the model seed."""
    X = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    X = torch.cat([X, torch.full((p * p, 1), p)], 1)
    Y = (X[:, 0] + X[:, 1]) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(split_seed))
    n_tr = int(frac * p * p)
    return X[perm[:n_tr]], Y[perm[:n_tr]], X[perm[n_tr:]], Y[perm[n_tr:]]


class OneLayer(nn.Module):
    """1-layer attention + ReLU MLP transformer, untied unembed, readout at the '=' position only."""

    def __init__(s, vocab, d=128, H=4, dmlp=512, n_ctx=3):
        super().__init__()
        s.H, s.dh = H, d // H
        s.E = nn.Parameter(torch.randn(vocab, d) / math.sqrt(d))
        s.P = nn.Parameter(torch.randn(n_ctx, d) / math.sqrt(d))
        s.WQ = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.WK = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.WV = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.WO = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.Win = nn.Parameter(torch.randn(d, dmlp) / math.sqrt(d))
        s.bin = nn.Parameter(torch.zeros(dmlp))
        s.Wout = nn.Parameter(torch.randn(dmlp, d) / math.sqrt(dmlp))
        s.bout = nn.Parameter(torch.zeros(d))
        s.U = nn.Parameter(torch.randn(d, vocab - 1) / math.sqrt(vocab))

    def forward(s, tok):                                   # tok [B,3] -> logits [B, p]
        x = s.E[tok] + s.P
        B = x.shape[0]
        q = (x[:, -1] @ s.WQ).view(B, 1, s.H, s.dh)
        k = (x @ s.WK).view(B, 3, s.H, s.dh)
        v = (x @ s.WV).view(B, 3, s.H, s.dh)
        att = torch.softmax((q * k).sum(-1) / math.sqrt(s.dh), dim=1)
        r = x[:, -1] + (att.unsqueeze(-1) * v).sum(1).reshape(B, -1) @ s.WO
        r = r + F.relu(r @ s.Win + s.bin) @ s.Wout + s.bout
        return r @ s.U


def evaluate(model, data):
    Xtr, Ytr, Xte, Yte = data
    with torch.no_grad():
        lo_tr, lo_te = model(Xtr), model(Xte)
        return dict(train_acc=(lo_tr.argmax(-1) == Ytr).float().mean().item(),
                    test_acc=(lo_te.argmax(-1) == Yte).float().mean().item(),
                    train_loss=F.cross_entropy(lo_tr, Ytr).item(),
                    test_loss=F.cross_entropy(lo_te, Yte).item(),
                    wnorm=math.sqrt(sum((q ** 2).sum().item() for q in model.parameters())))


def gradfilter_ema(model, ema, alpha, lamb):
    """Grokfast-EMA (Lee et al. 2024): g <- g + lamb * EMA_alpha(g). Returns the updated EMA state."""
    if ema is None:
        ema = {n: q.grad.detach().clone() for n, q in model.named_parameters() if q.grad is not None}
    for n, q in model.named_parameters():
        if q.grad is not None:
            ema[n].mul_(alpha).add_(q.grad.detach(), alpha=1 - alpha)
            q.grad.add_(ema[n], alpha=lamb)
    return ema


def train(p=97, frac=0.5, steps=3000, seed=0, lr=1e-3, wd=1.0, filt=None, interventions=(), save_steps=(),
          eval_every=50, threads=1, on_eval=None, should_stop=None):
    """Full-batch AdamW. filt: None or ('grokfast', alpha, lamb). interventions: list of (step, 'scale', c), applied
    to all parameters (theta <- c * theta) after the evaluation/checkpoint of that step index and before its
    gradient computation.
    Returns dict(curves, first, checkpoints) where checkpoints maps step -> state_dict (CPU tensors)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    data = make_data(p, frac)
    Xtr, Ytr = data[0], data[1]
    m = OneLayer(p + 1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    ema = None
    curves = {k: [] for k in ('step', 't', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'wnorm')}
    first, ckpts = {}, {}
    thresholds = [('train_acc>=0.99', 'train_acc', 0.99), ('test_acc>=0.5', 'test_acc', 0.5),
                  ('test_acc>=0.9', 'test_acc', 0.9), ('test_acc>=0.99', 'test_acc', 0.99)]
    iv = {int(s): (kind, c) for s, kind, c in interventions}
    t0 = time.time()
    for step in range(steps + 1):
        if step in save_steps:
            ckpts[step] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if step % eval_every == 0 or step == steps:
            ev = evaluate(m, data)
            ev['step'], ev['t'] = step, time.time() - t0
            for k in curves:
                curves[k].append(ev[k])
            for name, key, thr in thresholds:
                if ev[key] >= thr and name not in first:
                    first[name] = step
            if on_eval:
                on_eval(ev)
        if step == steps or (should_stop and should_stop()):
            break
        if step in iv and iv[step][0] == 'scale':
            with torch.no_grad():
                for q in m.parameters():
                    q.mul_(iv[step][1])
        loss = F.cross_entropy(m(Xtr), Ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if filt is not None and filt[0] == 'grokfast':
            ema = gradfilter_ema(m, ema, filt[1], filt[2])
        opt.step()
    return dict(curves=curves, first=first, checkpoints=ckpts)


def eval_state(state_dict, p=97, frac=0.5):
    m = OneLayer(p + 1)
    m.load_state_dict(state_dict)
    return evaluate(m, make_data(p, frac))
