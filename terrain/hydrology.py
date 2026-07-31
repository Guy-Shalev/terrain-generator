"""Stage 3: erosion and drainage.

Order matters: thermal creep knocks impossible slopes down, then the surface is
depression-filled so water has somewhere to go, D8 flow is routed, discharge is
accumulated, and stream power carves proportionally to it. Repeat a few passes
so valleys deepen and their divides sharpen.

Leftover depressions are deliberately not all filled - they are where lakes go.
"""
import numpy as np

from . import grid


def thermal(h, iters=30, talus=0.035, rate=0.35, mask=None):
    """Creep: any slope steeper than `talus` sheds material to its downhill neighbours.

    All eight transfers are summed into one `delta` and applied at the end of
    the sweep, so every direction still reads the surface as it was at the
    start. Holding the eight excess fields instead, as the direct reading of
    the rule does, is 48 MB live at 1024x768 and the loop becomes memory-bound.
    """
    h = h.copy()
    # `rate / 8` (the share each neighbour gets) and the land mask fold into a
    # single factor, so the inner loop does one multiply rather than three.
    factor = rate / 8.0
    if mask is not None:
        factor = mask.astype(h.dtype) * factor
    delta = np.empty_like(h)
    # Two padded buffers instead of sixteen gathers a sweep: one to read the
    # neighbours of `h` out of, one to hand `give` back to the cell it came
    # from. Same values either way, so the result is unchanged.
    around = grid.Halo(h.shape, h.dtype)
    sent = grid.Halo(h.shape, h.dtype)
    give = np.empty_like(h)
    for _ in range(iters):
        delta[:] = 0.0
        around.load(h)
        for (dy, dx), dist in zip(grid.NEIGH8, grid.DIST8):
            np.subtract(h, around.at(dy, dx), out=give)
            np.subtract(give, talus * dist, out=give)
            np.clip(give, 0, None, out=give)
            np.multiply(give, factor, out=give)
            np.subtract(delta, give, out=delta)
            np.add(delta, sent.load(give).at(-dy, -dx), out=delta)
        if not delta.any():     # nothing anywhere exceeds the angle of repose
            break
        h += delta
    return h


def _pool(a, op):
    """2x2 block reduce, edge-padded to even dimensions."""
    h, w = a.shape
    if h % 2:                       # duplicate, never drop: dropping a row
        a = np.concatenate([a, a[-1:]], axis=0)      # would lose a peak and
    if w % 2:                       # break the upper-bound guarantee
        a = np.concatenate([a, a[:, -1:]], axis=1)
    b = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2)
    return op(op(b, axis=3), axis=1)


def _warm_start(h, outlet, min_size=64):
    """An upper bound on the filled surface, solved cheaply on a coarse grid.

    Planchon-Darboux only ever lowers its estimate, so starting from anything
    at or above the true surface converges to the same answer - it just gets
    there sooner. Solving on block-*maxima* with block-*AND* outlets is a valid
    bound: any coarse escape route corresponds to a fine one through the same
    blocks, whose peak can only be lower. Without this the estimate starts at
    "everything is a mountain" and the correct level has to crawl in from the
    coastline one cell per iteration, which is ~165 iterations at 1024x768.
    """
    if min(h.shape) <= min_size:
        return None
    hc, oc = _pool(h, np.max), _pool(outlet, np.all)
    if not oc.any():
        return None
    coarse = _solve(hc, oc, 0.0, _warm_start(hc, oc, min_size))
    return coarse.repeat(2, 0).repeat(2, 1)[:h.shape[0], :h.shape[1]]


def _solve(h, outlet, eps, init, max_iters=4000):
    """Iterate w = max(h, min(neighbours) + eps) until nothing moves.

    Every array here is preallocated and every step is in-place: this loop runs
    hundreds of times over the full grid, and at 1024x768 a single temporary
    per step is 6 MB of allocation and page faults.
    """
    filled = np.where(outlet, h, h.max() + 1.0 if init is None else init)
    work = np.empty_like(filled)
    best = np.empty_like(filled)
    lower = np.empty(filled.shape, dtype=bool)
    for _ in range(max_iters):
        # Including the centre cell in the minimum is harmless: it can only
        # tie, never pull the result below h.
        grid.min3x3(filled, work, best)
        np.add(best, eps, out=best)
        np.minimum(best, filled, out=best)
        np.maximum(best, h, out=best)
        np.copyto(best, filled, where=outlet)
        # The iteration is monotone decreasing, so "did anything drop" is the
        # whole convergence test. np.allclose costs more than the step itself.
        np.less(best, filled, out=lower)
        done = not lower.any()
        filled, best = best, filled
        if done:
            break
    return filled


def fill_depressions(h, sea_level=0.0, eps=1e-5, max_iters=400):
    """Planchon-Darboux filling. Ocean cells and the map's y edges are outlets.

    Stops early on convergence; capping the iterations leaves the deepest
    interior basins unfilled, which is what we want for future lakes.
    """
    outlet = h <= sea_level
    outlet[0, :] = True
    outlet[-1, :] = True
    init = _warm_start(h, outlet)
    if init is not None:
        # The coarse solve runs without the epsilon tilt, so pad by the most
        # the tilt could ever add before using it as an upper bound.
        init = np.maximum(h, init + eps * sum(h.shape))
    return _solve(h, outlet, eps, init, max_iters)


def flow_routing(filled):
    """D8 steepest descent. Returns (receiver index, distance to receiver)."""
    h, w = filled.shape
    rec = np.arange(h * w).reshape(h, w).copy()
    rec_d = np.ones_like(filled)
    best = np.zeros_like(filled)
    # The neighbour heights come out of one padded copy, and the index grid a
    # neighbour maps to depends only on the shape and the offset, so it is
    # built once and cached rather than gathered on every call.
    around = grid.Halo(filled.shape, filled.dtype).load(filled)
    slope = np.empty_like(filled)
    for (dy, dx), dist in zip(grid.NEIGH8, grid.DIST8):
        np.subtract(filled, around.at(dy, dx), out=slope)
        np.divide(slope, dist, out=slope)
        take = slope > best
        best[take] = slope[take]
        rec[take] = grid.neighbour_index(filled.shape, dy, dx)[take]
        rec_d[take] = dist
    return rec, rec_d, best


def accumulate(filled, rec, weights=None, gate=None):
    """Drainage area: every cell's water, plus everything draining into it.

    Peeled a level at a time (Kahn's algorithm) rather than walked cell by cell
    in elevation order: same topological order requirement, but each level is
    one vectorised scatter instead of hundreds of thousands of interpreted
    steps. Both orders sum the same exact integer counts.

    `gate` is `(mask, fn)`. A cell the mask selects has its total replaced by
    `fn(indices, totals)` before that total moves downstream, which is how a
    lake that evaporates what reaches it stops the river below it. The
    replacement is applied the moment the cell is finalised - Kahn's front is
    exactly the set of cells with nothing left upstream - so `fn` sees the
    lake's whole inflow, and sees it once.
    """
    n = filled.size
    acc = np.ones(n) if weights is None else weights.ravel().astype(float).copy()
    r = rec.ravel()
    flows = r != np.arange(n)              # false at sinks: sea, edges, pits
    indeg = np.bincount(r[flows], minlength=n)
    front = np.flatnonzero(indeg == 0)     # ridge cells, nothing upstream
    gate_mask, gate_fn = gate if gate else (None, None)
    while front.size:
        if gate_mask is not None:
            # Before the sink filter: a gate on a pit still has to fire, or the
            # lake sitting in it never gets told how much water arrived.
            g = front[gate_mask[front]]
            if g.size:
                acc[g] = gate_fn(g, acc[g])
        front = front[flows[front]]
        if front.size == 0:
            break
        tgt = r[front]
        np.add.at(acc, tgt, acc[front])
        np.subtract.at(indeg, tgt, 1)
        front = np.unique(tgt[indeg[tgt] == 0])
    return acc.reshape(filled.shape)


def stream_power(h, sea_level=0.0, passes=4, k=0.06, m=0.5, n=1.0,
                 thermal_iters=12, talus=0.045):
    """Alternate hillslope creep and fluvial incision. Returns (h, filled, acc)."""
    for p in range(passes):
        land = h > sea_level - 0.02
        h = thermal(h, iters=thermal_iters, talus=talus, mask=land)
        filled = fill_depressions(h, sea_level)
        rec, rec_d, _ = flow_routing(filled)
        acc = accumulate(filled, rec)

        flat = h.ravel()
        slope = np.clip((flat - flat[rec.ravel()]) / rec_d.ravel(), 0, None)
        incision = k * (acc.ravel() ** m) * (slope ** n)
        # Never cut a cell below its own receiver: that would invert the flow.
        drop = np.minimum(incision, (flat - flat[rec.ravel()]) * 0.5)
        drop = np.where(land.ravel(), np.clip(drop, 0, None), 0.0)
        h = (flat - drop).reshape(h.shape)
        h = np.where(land, grid.blur(h, 0.4), h)
    filled = fill_depressions(h, sea_level)
    rec, rec_d, _ = flow_routing(filled)
    acc = accumulate(filled, rec)
    return h, filled, acc, rec
