"""Resolve validated weight-space operations (GPT-NeoX and Transformer-XL) against a fresh view of the baseline."""
import math
import zlib
from itertools import count

import torch

from wsbench.lang.ast import Assignment, Call, Target
from wsbench.models import neox, txl
from wsbench.models.registry import load_model

CONTROL_SALT = 999_983


def _layer(expr, cfg, weights, module=neox):
    """A layer value. neox and txl share the layer-value API (get_layer, neg_all, neg_writers, align_layer, ...)."""
    if isinstance(expr, Target):
        return module.get_layer(weights, expr.index)
    args = expr.args
    first = _layer(args[0], cfg, weights, module)
    if expr.name == 'neg':
        return module.neg_all(first)
    if expr.name == 'neg_writers':
        return module.neg_writers(first)
    second = _layer(args[1], cfg, weights, module)
    if expr.name == 'align':
        return module.align_layer(cfg, first, second)
    return module.lerp_layer(first, second, float(args[2]))


def _operation(op, cfg, weights, group):
    if isinstance(op, Assignment):
        return neox.set_layer(weights, op.target.index, _layer(op.expr, cfg, weights))
    args, kw = op.args, op.kwargs
    if op.name == 'fold_ln':
        return neox.fold_ln(cfg, weights)
    if op.name == 'center_writers':
        return neox.center_writers(cfg, weights)
    target = args[0]
    if op.name == 'gridop':
        return neox.gridop_block(weights, target, *args[1].args)
    if target == 'resid':
        if op.name == 'scale':
            return neox.scale_resid(cfg, weights, float(args[1]))
        layers = kw.get('layers', 'all')
        return neox.rotate_resid(cfg, weights, group(args[1], cfg.d),
                                 None if layers == 'all' else layers,
                                 kw.get('embed', True), kw.get('unembed', True))
    i = target.index
    if op.name == 'delete':
        return neox.zero_writers(weights, [i])
    if op.name == 'swap':
        j = args[1].index
        a, b = neox.get_layer(weights, i), neox.get_layer(weights, j)
        return neox.set_layer(neox.set_layer(weights, i, b), j, a)
    if op.name == 'permute':
        if target.part == 'heads':
            return neox.perm_heads(cfg, weights, i, group(args[1], cfg.n_heads).tolist())
        return neox.perm_mlp(cfg, weights, i, group(args[1], cfg.d_mlp))
    if op.name == 'scale':
        return neox.scale_mlp(cfg, weights, i, group(args[1], cfg.d_mlp))
    if op.name == 'rotate':
        return neox.rotate_mlp(cfg, weights, i, group(args[1], cfg.d_mlp))
    if op.name == 'vo':
        return neox.vo_gauge(cfg, weights, i, target.head, group(args[1], cfg.dh))
    if op.name == 'qk':
        dims = torch.arange(cfg.rd if kw.get('dims', 'pass') == 'pass' else 0, cfg.dh)
        return neox.qk_gauge(cfg, weights, i, target.head, group(args[1], len(dims)), dims)
    if op.name == 'rope_pairs':
        return neox.rope_pairs(cfg, weights, i, target.head, kw['phis'], float(kw.get('scale', 1.0)))
    raise ValueError(f'unsupported NeoX op: {op.name}')


def _txl_operation(op, cfg, weights, group):
    """One Transformer-XL op. The txl operators are copy-on-write: unchanged tensors stay shared with the baseline."""
    if isinstance(op, Assignment):
        return txl.set_layer(weights, op.target.index, _layer(op.expr, cfg, weights, txl))
    args = op.args
    target = args[0]
    if op.name == 'gridop':
        return txl.gridop_block(weights, target, *args[1].args)
    i = target.index
    if op.name == 'delete':
        return txl.zero_writers(weights, [i])
    if op.name == 'swap':
        return txl.swap_layers(weights, i, args[1].index)
    if op.name == 'permute':
        if target.part == 'heads':
            return txl.perm_heads(cfg, weights, i, group(args[1], cfg.n_heads))
        return txl.perm_mlp(cfg, weights, i, group(args[1], cfg.d_mlp))
    if op.name == 'scale':
        return txl.scale_mlp(cfg, weights, i, group(args[1], cfg.d_mlp))
    if op.name == 'rotate':
        return txl.rotate_mlp(cfg, weights, i, group(args[1], cfg.d_mlp))
    if op.name == 'vo':
        return txl.vo_gauge(cfg, weights, i, target.head, group(args[1], cfg.dh))
    if op.name == 'qk':        # dims=all is the only form on Transformer-XL (validation rejects dims=pass)
        return txl.qk_gauge(cfg, weights, i, target.head, group(args[1], cfg.dh))
    raise ValueError(f'unsupported Transformer-XL op: {op.name}')


def _apply(stmt, model, seed):
    index = count()

    def group(value, size):
        if isinstance(value, Call) and value.name == 'perm':
            return torch.tensor(value.args[0], dtype=torch.long)
        generator = torch.Generator().manual_seed(seed + next(index))
        if isinstance(value, Call):
            return neox.rand_uniform(size, *map(float, value.args), generator)
        generate = {'orth': neox.rand_orth, 'orth1': neox.rand_orth_fixing_ones,
                    'gl': neox.rand_gl, 'randperm': neox.rand_perm}
        return generate[value](size, generator)

    operation = _txl_operation if stmt.family == 'txl' else _operation
    weights, kind = model.W0, 'ln'
    for op in stmt.ops:
        if isinstance(op, Call) and op.name == 'ln_to_rms':
            kind = 'rms'
        else:
            weights = operation(op, model.cfg, weights, group)
    return weights, kind


def _changes(before, after):
    changed = []
    sq_dist = 0.
    for key in sorted(before.keys() & after.keys()):
        a, b = before[key], after[key]
        if a is b or torch.equal(a, b):      # copy-on-write operators share every tensor they leave alone
            continue
        dist = torch.linalg.vector_norm(b - a).item()
        sq_dist += dist ** 2
        changed.append({'name': key, 'shape': list(b.shape),
                        'rel_change': dist / max(1e-12, torch.linalg.vector_norm(a).item())})
    changed.sort(key=lambda entry: entry['rel_change'], reverse=True)
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    for key in added:
        changed.append({'name': key, 'shape': list(after[key].shape), 'rel_change': None})
    return {
        'changed_tensors': changed, 'param_dist': math.sqrt(sq_dist),
        'new_params': added, 'removed_params': removed,
    }


def _control_result(module, model, base, kind_args, metrics):
    logits = module.forward(model.cfg, base, model.ids, *kind_args)
    dloss = module.next_token_loss(logits, model.ids) - model.loss0
    return {
        'dloss': dloss, 'max_abs_dlogits': logits.sub_(model.logits0).abs_().max().item(),
        'ratio': metrics['dloss'] / dloss if abs(dloss) >= 1e-12 else None,
    }


def _control(model, after, kind, seed, metrics):
    before = model.W0
    generator = torch.Generator().manual_seed(seed + CONTROL_SALT)
    changed_keys = {entry['name'] for entry in metrics['changed_tensors']}
    noise = {}
    delta_sq = metrics['param_dist'] ** 2
    noise_sq = 0.
    base = before.copy()
    for key in sorted(before.keys() | after.keys()):
        if key not in before:
            base[key] = torch.zeros_like(after[key])
            delta_sq += torch.linalg.vector_norm(after[key]).item() ** 2
        # Draw for unchanged keys too, so the generator sequence stays in key order.
        eps = torch.randn(base[key].shape, generator=generator, dtype=torch.float64)
        if key not in changed_keys and key in after:
            continue
        if key in after:
            mask = after[key] != base[key]
        else:
            mask = base[key] != 0
            delta_sq += torch.linalg.vector_norm(base[key]).item() ** 2
        noise[key] = eps.mul_(mask)
        noise_sq += torch.linalg.vector_norm(eps).item() ** 2

    factor = math.sqrt(delta_sq / noise_sq)
    for key, eps in noise.items():
        base[key] = base[key] + eps.mul_(factor)
    return _control_result(neox, model, base, (kind,), metrics)


def tensor_seed(seed, name):
    """Seed of the control noise for one tensor, a 32-bit value: the CRC-32 of the tensor's name, started from the
    statement's seed (plus CONTROL_SALT) mod 2**32.

    torch's CPU generator (mt19937) uses only the low 32 bits of a seed, so both inputs have to be mixed into those
    bits. For one name, distinct seeds mod 2**32 give distinct generator seeds (CRC-32 is a bijection of its start
    value), as for the NeoX control, which seeds with seed + CONTROL_SALT."""
    return zlib.crc32(name.encode('utf-8'), (seed + CONTROL_SALT) % 2 ** 32)


def _txl_control(model, after, kind, seed, metrics):
    """The random control of an edit: noise of the edit's norm on exactly the entries the edit changed.

    Unlike the NeoX control, which draws one generator sequence for every tensor in key order (about 285M normal
    draws per edit on transfo-xl-wt103), this draws only for the changed tensors, each from its own generator seeded
    by tensor_seed(seed, name). The noise of a tensor therefore depends only on the statement's seed and the tensor's
    name. Transformer-XL operators never add or remove tensors, and every unchanged tensor stays shared. Each noise
    buffer becomes its control tensor in place, so the control holds one extra copy of the changed tensors, not two."""
    before = model.W0
    base = dict(before)
    noise, noise_sq = {}, 0.
    for entry in metrics['changed_tensors']:
        name = entry['name']
        generator = torch.Generator().manual_seed(tensor_seed(seed, name))
        eps = torch.randn(before[name].shape, generator=generator, dtype=torch.float64)
        eps.mul_(after[name] != before[name])
        noise[name] = eps
        noise_sq += torch.linalg.vector_norm(eps).item() ** 2
    factor = metrics['param_dist'] / math.sqrt(noise_sq)
    for name in list(noise):
        base[name] = noise.pop(name).mul_(factor).add_(before[name])     # = before + factor * eps, bit for bit
    return _control_result(txl, model, base, (), metrics)


def _threads(stmt):
    """Torch threads for one statement: 1 for the tiny models, 2 for Pythia, up to 4 for transfo-xl-wt103."""
    if stmt.model in ('neox-tiny', 'txl-tiny'):
        return 1
    return 4 if stmt.family == 'txl' else 2


def _run(stmt):
    previous_threads = torch.get_num_threads()
    threads = min(previous_threads, _threads(stmt))
    if threads == previous_threads:
        return _execute(stmt)
    torch.set_num_threads(threads)
    try:
        return _execute(stmt)
    finally:
        torch.set_num_threads(previous_threads)


@torch.no_grad()
def execute_neox(stmt):
    return _run(stmt)


@torch.no_grad()
def execute_txl(stmt):
    return _run(stmt)


def _execute(stmt):
    model = load_model(stmt.model)
    seed = stmt.seed * 1_000_003 + stmt.line * 1009
    weights, kind = _apply(stmt, model, seed)
    if stmt.family == 'txl':       # log-probabilities of the adaptive softmax take the place of logits
        logits = txl.forward(model.cfg, weights, model.ids)
        loss = txl.next_token_loss(logits, model.ids)
    else:
        logits = neox.forward(model.cfg, weights, model.ids, kind)
        loss = neox.next_token_loss(logits, model.ids)
    max_diff = logits.sub_(model.logits0).abs_().max().item()
    del logits
    metrics = {
        'max_abs_dlogits': max_diff, 'rel': max_diff / max(1e-12, model.max_logit0),
        'loss0': model.loss0, 'loss1': loss, 'dloss': loss - model.loss0,
        **_changes(model.W0, weights), 'norm_kind': kind, 'control': None,
    }
    if stmt.claim == 'edit' and metrics['param_dist'] > 0:
        control = _txl_control if stmt.family == 'txl' else _control
        metrics['control'] = control(model, weights, kind, seed, metrics)
    return metrics, weights, kind
