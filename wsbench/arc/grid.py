"""Reference: ARC grid primitives, NeuroGolf-style one-hot encoding, primitive -> permutation-network compiler."""
import numpy as np

# Official ARC colours (index 0..9) as used by the 2025 Google Code Golf utilities (Apache-2.0).
ARC_PALETTE = [(0, 0, 0), (30, 147, 255), (250, 61, 49), (78, 204, 48), (255, 221, 0),
               (153, 153, 153), (229, 59, 163), (255, 133, 28), (136, 216, 241), (147, 17, 49)]

# Semantics identical to michaelhodel/arc-dsl (rot90 is CLOCKWISE). g is a 2-D numpy array (rows x cols).
PRIMITIVES = {
    'identity': lambda g: g,
    'rot90':    lambda g: np.rot90(g, -1),
    'rot180':   lambda g: np.rot90(g, 2),
    'rot270':   lambda g: np.rot90(g, 1),
    'hmirror':  lambda g: g[::-1, :],          # flip top <-> bottom
    'vmirror':  lambda g: g[:, ::-1],          # flip left <-> right
    'dmirror':  lambda g: g.T,                 # transpose (main diagonal)
    'cmirror':  lambda g: np.rot90(g, 2).T,    # counter-diagonal
}
SHAPE_PRESERVING_ON_SQUARE = list(PRIMITIVES)   # all of them keep an n x n block n x n


def upscale(g, k):
    return np.kron(g, np.ones((k, k), dtype=g.dtype))


def onehot(grid):
    """ARC grid (list of lists, values 0..9, at most 30x30) -> float32 array (10, 30, 30); grid at the top-left,
    padding cells are all-zero in every channel."""
    g = np.array(grid, dtype=np.int64)
    h, w = g.shape
    assert h <= 30 and w <= 30
    x = np.zeros((10, 30, 30), dtype=np.float32)
    x[g, np.arange(h)[:, None], np.arange(w)[None, :]] = 1.0
    return x


def decode(y):
    """(10, 30, 30) -> grid (list of lists) or None if not decodable. A cell is active if any channel > 0.
    Height/width = extent of active cells from the top-left. Every cell inside must be active with exactly one
    channel > 0, every cell outside must be inactive."""
    pos = y > 0
    active = pos.any(0)
    if not active.any():
        return None
    rows = np.where(active.any(1))[0]
    cols = np.where(active.any(0))[0]
    h, w = rows.max() + 1, cols.max() + 1
    inside = np.zeros_like(active)
    inside[:h, :w] = True
    if (active != inside).any() or (pos[:, :h, :w].sum(0) != 1).any():
        return None
    return pos[:, :h, :w].argmax(0).tolist()


def compile_permutation(h, w, prim):
    """Index map src[900]: output cell j (= r*30 + c) copies input cell src[j]; -1 means 'stays empty'.
    Built for an input of exactly h x w placed top-left. The network is y[k, j] = x[k, src[j]] for every
    colour channel k, i.e. the weight matrix kron(I_10, P) with P[j, src[j]] = 1."""
    idx = np.arange(900).reshape(30, 30)
    out = PRIMITIVES[prim](idx[:h, :w])
    src = np.full(900, -1, dtype=np.int64)
    oh, ow = out.shape
    for r in range(oh):
        for c in range(ow):
            src[r * 30 + c] = out[r, c]
    return src


def apply_permutation_net(src, x):
    """x (10, 30, 30) -> y (10, 30, 30)."""
    xf = x.reshape(10, 900)
    y = np.zeros_like(xf)
    m = src >= 0
    y[:, m] = xf[:, src[m]]
    return y.reshape(10, 30, 30)


def dense_weight_from_src(src):
    """Dense 900 x 900 matrix P (for display only; 810,000 entries, 1 per output cell)."""
    P = np.zeros((900, 900), dtype=np.float32)
    for j, s in enumerate(src):
        if s >= 0:
            P[j, s] = 1.0
    return P


class CompiledNet:
    """size='fixed': one permutation compiled for the first example's input size.
    size='gated': one permutation per input size seen in the task's examples (a size gate selects it)."""

    def __init__(self, prim, examples, size='fixed'):
        self.prim, self.size = prim, size
        shapes = [np.array(e['input']).shape for e in examples]
        if size == 'fixed':
            self.table = {'*': compile_permutation(*shapes[0], prim)}
        elif size == 'gated':
            self.table = {s: compile_permutation(*s, prim) for s in dict.fromkeys(shapes)}
        else:
            raise ValueError(size)

    def __call__(self, grid):
        shape = np.array(grid).shape
        src = self.table.get('*', self.table.get(shape))
        if src is None:
            return None
        return decode(apply_permutation_net(src, onehot(grid)))

    def n_params_dense(self):
        return len(self.table) * 900 * 900   # with colour-channel sharing


def check(fn, examples):
    """Official Code Golf rule: the output must equal the expected grid exactly. Returns (passed, total, first_fail)."""
    passed, first_fail = 0, None
    for i, e in enumerate(examples):
        try:
            got = fn([row[:] for row in e['input']])
            ok = got is not None and np.array_equal(np.array(got), np.array(e['output']))
        except Exception as exc:   # noqa: BLE001
            ok, got = False, f'error: {exc!r}'
        if ok:
            passed += 1
        elif first_fail is None:
            first_fail = dict(index=i, input=e['input'], expected=e['output'], got=got)
    return passed, len(examples), first_fail
