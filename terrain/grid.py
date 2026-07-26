"""Grid helpers. The world wraps in x (cylinder) and clamps in y (poles)."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

# 8-neighbour offsets (dy, dx) and their step lengths.
NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIST8 = [np.sqrt(2), 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


_SHIFT_IDX = {}


def _shift_index(shape, dy, dx):
    key = (shape, dy, dx)
    idx = _SHIFT_IDX.get(key)
    if idx is None:
        h, w = shape
        idx = (np.clip(np.arange(h) + dy, 0, h - 1)[:, None],
               ((np.arange(w) + dx) % w)[None, :])
        _SHIFT_IDX[key] = idx
    return idx


def shift(a, dy, dx):
    """Sample a[y+dy, x+dx]: wrap in x, replicate the edge row in y.

    One fancy index does both axes in a single allocation. Rolling then
    concatenating costs two copies, and the erosion loop calls this eight times
    per iteration on the full grid, so the copies dominate.
    """
    return a[_shift_index(a.shape, dy, dx)]


def min3x3(a, scratch, out):
    """3x3 neighbourhood minimum (wrap in x, clamp in y), into caller buffers.

    Separable: minimum over x, then over y. scipy's minimum_filter does the
    same thing with a general algorithm and costs ~9x more here, and the
    depression fill runs this hundreds of times on the full grid.
    """
    np.minimum(a[:, :-1], a[:, 1:], out=scratch[:, :-1])
    np.minimum(a[:, -1], a[:, 0], out=scratch[:, -1])
    np.minimum(scratch[:, 1:], a[:, :-1], out=scratch[:, 1:])
    np.minimum(scratch[:, 0], a[:, -1], out=scratch[:, 0])
    np.minimum(scratch[:-1], scratch[1:], out=out[:-1])
    out[-1] = scratch[-1]
    np.minimum(out[1:], scratch[:-1], out=out[1:])
    return out


def gradient(a):
    """Central-difference (dy, dx) with the grid's wrap/clamp conventions."""
    dy = (shift(a, 1, 0) - shift(a, -1, 0)) * 0.5
    dx = (shift(a, 0, 1) - shift(a, 0, -1)) * 0.5
    return dy, dx


def blur(a, sigma):
    if sigma <= 0:
        return a
    return gaussian_filter(a.astype(float), sigma, mode=["nearest", "wrap"])


def edt(mask, return_indices=False):
    """Distance to the nearest False cell of `mask`, wrapping in x.

    Tiles the map three times horizontally so the transform can see across the
    seam, then crops back. Returned indices are folded into the real map.
    """
    w = mask.shape[1]
    tiled = np.concatenate([mask, mask, mask], axis=1)
    if not return_indices:
        return distance_transform_edt(tiled)[:, w:2 * w]
    dist, idx = distance_transform_edt(tiled, return_indices=True)
    iy = idx[0][:, w:2 * w]
    ix = idx[1][:, w:2 * w] % w
    return dist[:, w:2 * w], (iy, ix)


def normalize(a, lo=0.0, hi=1.0):
    amin, amax = float(a.min()), float(a.max())
    if amax - amin < 1e-12:
        return np.full_like(a, lo, dtype=float)
    return lo + (hi - lo) * (a - amin) / (amax - amin)


def stamp(arr, y, x, amp, sigma):
    """Add a gaussian bump centred at (y, x), wrapping in x."""
    h, w = arr.shape
    r = int(max(2, sigma * 3))
    ys = np.arange(int(round(y)) - r, int(round(y)) + r + 1)
    xs = np.arange(int(round(x)) - r, int(round(x)) + r + 1)
    keep = (ys >= 0) & (ys < h)
    ys = ys[keep]
    if ys.size == 0:
        return
    dy = (ys - y)[:, None]
    dx = (xs - x)[None, :]
    bump = amp * np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
    np.add.at(arr, (ys[:, None], xs[None, :] % w), bump)
