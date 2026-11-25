"""Numeric views for the heatmap and the decile-coloured ARC block."""
import numpy as np

from .data import ArcError


def _as_array(tensor):
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().numpy()
    return np.asarray(tensor, dtype=np.float64)


def _block_means(matrix, max_size):
    block = [(length + max_size - 1) // max_size for length in matrix.shape]
    result = matrix
    for axis, stride in enumerate(block):
        if stride > 1:
            length = result.shape[axis]
            starts = np.arange(0, length, stride)
            counts = np.minimum(length - starts, stride)
            divisor = counts[:, None] if axis == 0 else counts
            result = np.add.reduceat(result, starts, axis=axis) / divisor
    return result, block


def tensor_view(name, before, after, which, max_size=256, view="heatmap",
                r0=None, r1=None, c0=None, c1=None):
    if before is None and after is None:
        raise ArcError("E_NOT_FOUND", "unknown tensor", 404)
    if which not in ("before", "after", "diff"):
        raise ArcError("E_BAD_REQUEST", "which must be before, after, or diff")
    if max_size < 1 or view not in ("heatmap", "arc"):
        raise ArcError("E_BAD_REQUEST", "invalid tensor view or maximum")
    if before is None:
        after = _as_array(after)
        before = np.zeros_like(after)
    else:
        before = _as_array(before)
        after = np.zeros_like(before) if after is None else _as_array(after)

    shape = list(before.shape)
    if view == "arc":
        if len(shape) != 2 or None in (r0, r1, c0, c1):
            raise ArcError("E_BAD_REQUEST", "ARC view requires a 2-D tensor and four block bounds")
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]
                and r1 - r0 <= 30 and c1 - c0 <= 30):
            raise ArcError("E_BAD_REQUEST", "invalid ARC block bounds")
        before = before[r0:r1, c0:c1]
        after = after[r0:r1, c0:c1]

    if which == "diff":
        values = after - before
    else:
        values = before if which == "before" else after

    if view == "arc":
        quantiles = np.quantile(values, np.arange(1, 10) / 10)
        grid = np.searchsorted(quantiles, values, side="left")
        return {"grid": grid.tolist(), "quantiles": quantiles.tolist()}

    scale = float(np.abs(values if which == "diff" else before).max())
    if scale == 0:
        scale = float(np.abs(after).max()) or 1.0
    matrix = values[None, :] if len(shape) == 1 else values
    shown, block = _block_means(matrix, max_size)
    data = [[float(f"{v:.6g}") for v in row] for row in shown]
    return {"name": name, "shape": shape, "shown": list(shown.shape),
            "block": block, "scale": scale, "data": data}
