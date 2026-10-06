"""Stage 4: lakes and rivers as real features rather than a flow threshold.

Four things happen here, in order, because each depends on the last:

    water balance    every closed basin holds the water surface its own
                     catchment can keep against evaporation - its spill level if
                     there is plenty, a smaller sheet with no outflow at all if
                     there is not - and a depth field under it
    outlet carving   the pour point of each lake that still spills is notched
                     and a channel cut downstream, so the lake has an outflow
                     and the terrain shows the gorge it drains through
    polylines        the D8 network is traced into head-to-mouth paths and
                     smoothed, which removes the eight-direction staircase
    meander + width  each path is displaced sideways by noise along its own
                     arc length and rasterised with a width taken from
                     discharge, then the bed is incised under it

Carving changes the surface, so filling and routing are redone after it.
"""
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import grid, hydrology


@dataclass
class Water:
    filled: np.ndarray        # depression-filled surface
    flow: np.ndarray          # drainage area, in cells
    receivers: np.ndarray
    lake_id: np.ndarray       # 0 = not a lake
    lake_level: np.ndarray    # flat water surface elevation, 0 outside lakes
    lake_depth: np.ndarray    # level - bed
    width: np.ndarray         # river width in cells, 0 where there is no river
    polylines: list = field(default_factory=list)   # (n, 2) arrays of (y, x)
    widths: list = field(default_factory=list)      # channel width per path node

    @property
    def lake_mask(self):
        return self.lake_id > 0

    @property
    def river_mask(self):
        return self.width > 0

    def surface(self, h):
        """Elevation of whatever is on top: lake surfaces are flat, land is bed."""
        return np.where(self.lake_mask, self.lake_level, h)


# --------------------------------------------------------------------------
# lakes


def _label_lakes(h, filled, cfg, sea_level=0.0):
    """Closed basins, dropped if they are shallower or smaller than the cutoffs."""
    raw = (filled - h > cfg.lake_min_depth) & (h > sea_level)
    # ponytail: labelling does not wrap in x, so a lake straddling the seam
    # becomes two. Tile-and-merge if seam lakes ever matter.
    lbl, n = ndimage.label(raw, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(lbl), 0
    keep = np.zeros(n + 1, bool)
    keep[1:] = np.bincount(lbl.ravel(), minlength=n + 1)[1:] >= cfg.lake_min_area
    remap = np.cumsum(keep) * keep
    return remap[lbl], int(remap.max())


_Routed = namedtuple("_Routed", "filled rec plain flow lake_id n level outflow")


def _lake_balance(h, filled, rec, lbl, n, cfg, evap=None):
    """Give each lake the water surface its catchment can hold, not its rim.

    A lake in steady state loses `lake_evap` per cell of water surface and gains
    its inflow, so the area it settles at is inflow / evaporation - independent
    of how deep the basin happens to be. Sort a basin's bed heights and that
    area names the level directly: with `k` cells wet, the surface stands at the
    k-th lowest bed in the basin. A basin whose inflow covers its whole spill
    area fills to the rim and spills the surplus, as before; one that cannot
    sits part full with **no outflow at all**.

    That second case is why the accumulation has to be gated rather than run and
    then patched: a lake that keeps its water is the end of the line, and the
    channel below it has to come out dry, not merely narrow.

    Under uniform rain a basin cannot go *completely* dry: it catches at least
    its own footprint, so its area never falls below `cells / lake_evap`. Playas
    need evaporation to outrun the rain locally, which is a spatial field, not
    this scalar - see `accumulate`'s `weights`.

    Returns `(rec, gate, level, outflow)`. Every cell of the basin is pointed at
    its exit, the lowest cell of the filled surface within it, so the whole
    inflow is finalised at one gate cell. That is not where D8 already sends it:
    the fill's epsilon tilt drains a basin through the shallow rim strip that
    `lake_min_depth` leaves *outside* the label, so a labelled basin has 3 to 24
    cells whose receiver is not in it (measured over 30 basins on six seeds) and
    gating any one of them would see a fraction of the water. The exit is
    strictly below every cell now pointed at it, so the graph stays acyclic.

    ponytail: the wet cells are the k lowest in the basin, which need not be
    connected - a basin with two hollows in it can come out as two ponds under
    one id. Fine as long as lakes are drawn from a mask; walk the hypsometry per
    connected component if a lake ever needs its own outline.
    """
    idx = np.arange(1, n + 1)
    cells = np.flatnonzero(lbl.ravel() > 0)
    ids = lbl.ravel()[cells]
    # Grouped by lake and rising within each group, so the k-th entry of a
    # group is exactly the level that wets k of its cells.
    order = np.lexsort((h.ravel()[cells], ids))
    cells, ids = cells[order], ids[order]
    beds = h.ravel()[cells]
    bounds = np.searchsorted(ids, np.arange(1, n + 2))    # lake i is [i-1:i]
    spill = np.concatenate([[0.0], np.atleast_1d(
        ndimage.minimum(filled, lbl, index=idx))])

    pos = np.atleast_2d(ndimage.minimum_position(filled, lbl, index=idx))
    exits = np.zeros(n + 1, int)
    exits[1:] = pos[:, 0] * h.shape[1] + pos[:, 1]

    rec = rec.copy()
    rf = rec.ravel()
    tgt = exits[ids]
    move = cells != tgt
    rf[cells[move]] = tgt[move]

    level, outflow = np.zeros(n + 1), np.zeros(n + 1)
    # One evaporation rate per lake, averaged over its own surface: the same
    # basin is a full lake in a wet belt and a salt pan in a rain shadow.
    per = np.zeros(n + 1)
    per[1:] = (float(cfg.lake_evap) if evap is None
               else np.atleast_1d(ndimage.mean(evap, lbl, index=idx)))

    def gate(cell, inflow):
        lake = lbl.ravel()[cell]
        beg, size = bounds[lake - 1], bounds[lake] - bounds[lake - 1]
        ev = per[lake]
        wet = np.where(ev <= 0, size,
                       np.minimum((inflow / np.maximum(ev, 1e-9)).astype(int), size))
        full = wet >= size
        level[lake] = np.where(full, spill[lake],
                               beds[beg + np.minimum(wet, size - 1)])
        outflow[lake] = np.where(full, np.maximum(inflow - ev * size, 0.0), 0.0)
        return outflow[lake]

    mask = np.zeros(h.size, bool)
    mask[exits[1:]] = True
    return rec, (mask, gate), level, outflow


def _shift_local(a, dy, dx):
    """`grid.shift` without the wrap, for working inside one basin's bounding box."""
    out = np.zeros_like(a)
    hh, ww = a.shape
    out[max(0, -dy):hh - max(0, dy), max(0, -dx):ww - max(0, dx)] = \
        a[max(0, dy):hh - max(0, -dy), max(0, dx):ww - max(0, -dx)]
    return out


def _draw_receivers(rec, basin, lake, lbl):
    """Receivers for *drawing*: inside a basin, step towards the water.

    The accounting graph points every cell of a basin straight at the basin's
    exit, which is what lets the balance finalise the whole inflow at one gate -
    but it is a shortcut across the lake bed, and a path traced along it draws a
    ruler-straight river a dozen cells long. Following the plain D8 receivers
    instead is no better: the fill leaves an epsilon tilt that drains a basin
    towards its *spill*, so about 40% of the rivers reaching a part-full lake
    would skirt the pond and run past it to the rim.

    So a basin's dry floor gets its own receivers: a breadth-first tree grown
    out from that basin's own lake, each ring pointing back at the one before
    it. Every step is a single cell, so nothing can teleport, and every step
    goes down a ring, so the graph cannot contain a cycle.

    Both of those were learned the hard way from the obvious version - step to
    whichever neighbour is nearest open water. Measured against water of *any*
    kind, a basin whose rim runs near the coast has its gradient pointing over
    that rim at the sea, so a path climbs out of the basin, misses the pond and
    stops on the far slope. Restricting it to the basin's own lake instead
    leaves cells with no improving neighbour at all, which fall back to their
    plain receiver - and a plain receiver pointing back at a rerouted cell is a
    two-cell loop that swallows the river silently. Seed 1 at 512x384 had one at
    (239, 60) and (239, 61).

    Grown per basin inside its bounding box: the rings are tens of cells across
    where the map is hundreds, and the whole-grid version of this is 400 shifts
    of a 0.8M-cell array.
    """
    if not basin.any() or not lake.any():
        return rec
    rec = rec.copy()
    W = rec.shape[1]
    for sl in ndimage.find_objects(lbl):
        if sl is None:
            continue
        ys = slice(max(0, sl[0].start - 1), min(rec.shape[0], sl[0].stop + 1))
        xs = slice(max(0, sl[1].start - 1), min(W, sl[1].stop + 1))
        todo, frontier = basin[ys, xs].copy(), lake[ys, xs]
        if not (todo.any() and frontier.any()):
            continue
        gidx = (np.arange(ys.start, ys.stop)[:, None] * W +
                np.arange(xs.start, xs.stop)[None, :])
        sub = rec[ys, xs]
        while frontier.any() and todo.any():
            found = np.zeros_like(todo)
            for dy, dx in grid.NEIGH8:
                near = _shift_local(frontier, dy, dx) & todo & ~found
                if not near.any():
                    continue
                sub[near] = _shift_local(gidx, dy, dx)[near]
                found |= near
            todo &= ~found
            frontier = found
        rec[ys, xs] = sub
    return rec


def _route_water(h, cfg, sea_level=0.0, filled=None, rec=None,
                 runoff=None, evap=None):
    """Fill, route, label lakes, and accumulate under each lake's water balance.

    One helper for both halves of this stage: carving needs to know which lakes
    still spill and how hard, and the finished map needs the same answer plus
    the surfaces themselves.

    `runoff` is water contributed per cell and `evap` what a cell of lake
    surface loses, both from `climate`; left out, every cell contributes one and
    every lake evaporates `lake_evap`, which is the uniform-rain world.

    `filled` and `rec` may be handed in when they are already known for exactly
    this surface. The drainage area cannot be, because it is the thing the
    balance changes.
    """
    if filled is None:
        filled = hydrology.fill_depressions(h, sea_level)
    if rec is None:
        rec, _, _ = hydrology.flow_routing(filled)
    lbl, n = _label_lakes(h, filled, cfg, sea_level)
    if n == 0:
        flow = hydrology.accumulate(filled, rec, weights=runoff)
        return _Routed(filled, rec, rec, flow, lbl, 0, np.zeros(1), np.zeros(1))
    # `plain` is D8 as routed, before the balance points every basin cell at its
    # exit. The rewire is right for accounting and wrong for geometry; see
    # `_draw_receivers`.
    plain = rec
    rec, gate, level, outflow = _lake_balance(h, filled, rec, lbl, n, cfg, evap)
    flow = hydrology.accumulate(filled, rec, weights=runoff, gate=gate)
    return _Routed(filled, rec, plain, flow, lbl, n, level, outflow)


def carve_outlets(h, cfg, sea_level=0.0, routed=None, runoff=None, evap=None):
    """Notch each lake's pour point and cut a channel downstream from it.

    Without this a basin fills to its rim and the water has no modelled way
    out: the map shows a lake with no outflowing river and no valley below it.

    A gorge is only cut where the outflow is big enough to become a river.
    Carving is not gated by `river_threshold` otherwise, and the two then
    disagree: the gorge is real terrain but no river is drawn in it, so a very
    high threshold leaves dry trenches winding across the map with no water in
    them. At 384x288 that is 1% of the dug cells at the default threshold and
    30% at six times it. The lake's own outflow is the right test because it is
    exactly what would flow down the gorge - and under a water balance it is
    zero for a lake that keeps everything reaching it, so such a lake gets no
    gorge and no river below it.

    `routed` is an optional (filled, receivers, drainage area) already computed
    for `h` as it arrives. The erosion stage finishes by filling, routing and
    accumulating the very surface handed here and its caller throws that away,
    so the first pass would otherwise redo the fill and the routing - the
    expensive two thirds of it. Only those two are taken: the drainage area is
    recomputed because the balance gates it, and it is recomputed every pass
    because draining a lake changes which lakes there are.
    """
    thresh = cfg.river_threshold * h.size
    for _ in range(cfg.outlet_carve_passes):
        known = routed[:2] if routed is not None else (None, None)
        routed = None
        r = _route_water(h, cfg, sea_level, *known, runoff=runoff, evap=evap)
        lbl, n = r.lake_id, r.n
        if n == 0:
            break

        # Rim = cells just outside a lake. The lowest one is where it spills.
        rim = np.zeros_like(lbl)
        for dy, dx in grid.NEIGH8:
            nb = grid.shift(lbl, dy, dx)
            take = (lbl == 0) & (nb > 0) & (rim == 0)
            rim[take] = nb[take]
        if not rim.any():
            break
        idx = np.arange(1, n + 1)
        pours = ndimage.minimum_position(h, rim, index=idx)

        flat, recf = h.ravel(), r.rec.ravel()
        for lake, pos in enumerate(np.atleast_2d(pours), start=1):
            c = int(pos[0]) * h.shape[1] + int(pos[1])
            if r.outflow[lake] < thresh:   # too little water to cut, or to draw
                continue
            cur = flat[c] - cfg.outlet_carve_depth
            for _ in range(cfg.outlet_carve_len):
                flat[c] = min(flat[c], cur)
                nxt = recf[c]
                if nxt == c or flat[nxt] <= sea_level:
                    break
                cur -= cfg.outlet_carve_slope
                if flat[nxt] < cur:      # a natural channel takes over from here
                    break
                c = nxt
        h = flat.reshape(h.shape)
    return h


# --------------------------------------------------------------------------
# network geometry


def polylines(flow, rec, channel, cross=None):
    """Trace the channel network into paths, each running head -> mouth/junction.

    Heads come from `channel` alone, so no path starts on a `cross` cell.
    """
    hh, w = flow.shape
    up = np.bincount(rec.ravel()[channel.ravel()], minlength=flow.size)
    heads = np.flatnonzero(channel.ravel() & (up == 0))
    heads = heads[np.argsort(-flow.ravel()[heads])]  # longest rivers claim first
    recf, chan = rec.ravel(), channel.ravel()
    # Cells a path may run through without being channel cells themselves: the
    # floor of a basin, where the accounting graph leaves a discharge of about
    # one cell's worth even though a whole river is crossing it. Without this
    # the path stops one step into the basin and the river ends in open ground
    # several cells short of the lake.
    crossf = np.zeros(flow.size, bool) if cross is None else cross.ravel()
    visited = np.zeros(flow.size, bool)
    out = []
    for start in heads:
        c, path = int(start), []
        while True:
            path.append(c)
            visited[c] = True
            nxt = int(recf[c])
            if nxt == c:
                break
            path_end = visited[nxt] or not (chan[nxt] or crossf[nxt])
            if path_end:
                path.append(nxt)     # touch the trunk (or the water) and stop
                break
            c = nxt
        if len(path) >= 3:
            out.append(np.array([(p // w, p % w) for p in path], dtype=float))
    return out


def _unwrap_x(pts, w):
    """Undo the x seam inside one path so smoothing does not average across it."""
    dx = np.diff(pts[:, 1])
    pts[1:, 1] += np.cumsum(-w * (dx > w / 2) + w * (dx < -w / 2))
    return pts


def _smooth(pts, passes=2):
    """Moving average over the interior; D8 only knows eight directions."""
    for _ in range(passes):
        if len(pts) < 5:
            break
        mid = (pts[:-2] + 2 * pts[1:-1] + pts[2:]) / 4.0
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def _fbm1(n, rng, period, octaves=4):
    """1-D fractal noise along a path's arc length."""
    t = np.arange(n, dtype=float)
    out, amp, norm, p = np.zeros(n), 1.0, 0.0, float(period)
    for _ in range(octaves):
        k = int(np.ceil(n / max(2.0, p))) + 2
        ctrl = rng.normal(0, 1, k + 1)
        x = t / max(2.0, p)
        i = np.minimum(x.astype(int), k - 1)
        f = x - i
        f = f * f * (3 - 2 * f)
        out += amp * (ctrl[i] * (1 - f) + ctrl[i + 1] * f)
        norm += amp
        amp *= 0.5
        p *= 0.5
    return out / norm


def meander(pts, width, cfg, rng):
    """Push the path sideways by noise, hardest in the middle of its run.

    Amplitude grows with width because real meander belts scale with the
    channel; the taper keeps the head and the junction/mouth pinned in place so
    tributaries still meet their trunk.

    Curvature-driven migration replaced this for a while - Howard and Knutson's
    model, bends leaning and crawling downstream, with neck cutoffs leaving
    oxbow lakes. It was reverted: at any rate that made the bends visible the
    map read as wrigglier than this does, and the migrated channels wandered far
    enough from the D8 course they were traced from to sit oddly in their
    valleys. Worth another attempt only with a topographic term holding the
    channel to its valley floor, rather than the blunt corridor cap it used.
    """
    n = len(pts)
    if n < 6:
        return pts
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=-1)
    taper = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / cfg.meander_taper, 0, 1)
    amp = cfg.meander_amp * np.sqrt(np.maximum(width, 0.4)) * taper
    return pts + normal * (amp * _fbm1(n, rng, cfg.meander_period))[:, None]


# --------------------------------------------------------------------------
# rasterising


def _disc(width_map, y, x, chan_w):
    """Paint a channel of width `chan_w` cells centred on (y, x), keeping the max."""
    hh, w = width_map.shape
    r = max(chan_w * 0.5, 0.5)
    ri = int(np.ceil(r))
    ys = np.arange(int(round(y)) - ri, int(round(y)) + ri + 1)
    xs = np.arange(int(round(x)) - ri, int(round(x)) + ri + 1)
    ys = ys[(ys >= 0) & (ys < hh)]
    if ys.size == 0:
        return
    sel = np.hypot((ys - y)[:, None], (xs - x)[None, :]) <= r
    if not sel.any():
        # A point at exactly half a cell in *both* axes is 0.707 from all four
        # of its neighbours, so the narrowest channels - anything under a cell
        # wide, which floors `r` at 0.5 - paint nothing at all there and the
        # line comes out with a hole in it. Rare on a D8 course, which starts on
        # cell centres; unmissable on a smoothed one, where averaging integers
        # lands on halves all the time. Take the nearest cell instead of none.
        sel[np.argmin(np.abs(ys - y)), np.argmin(np.abs(xs - x))] = True
    yy, xx = np.nonzero(sel)
    np.maximum.at(width_map, (ys[yy], xs[xx] % w), chan_w)


def _connect_strays(width, lake_mask, sea, draw, cfg, budget=400):
    """Water a reader can follow: every channel must end at a lake or the sea.

    A backstop, not a mechanism. The drawing receivers are what actually keep
    rivers running to the water, and this should have nothing to do on a default
    map - `test_rivers_reach_water` asserts that it finds nothing left to fix.
    It exists because a stranded river is the one defect that reads as broken
    from across the map, and there are several ways to strand one: the map-edge
    cut, a discharge gate, a lake that shrank away from a channel drawn before it
    was known.

    Each body of water is one component of `river | lake | sea`, labelled on a
    3x horizontal tiling so the seam does not split one in two. A component
    touching neither lake nor sea is walked downstream from its lowest cell and
    painted at its own width until it reaches water; if the budget runs out, it
    is erased rather than left hanging.
    """
    river = width > 0
    if not river.any():
        return width
    w = width.shape[1]
    lab, n = ndimage.label(np.concatenate([river | lake_mask | sea] * 3, axis=1),
                           structure=np.ones((3, 3)))
    if n == 0:
        return width
    mid = lab[:, w:2 * w]
    reaches = np.zeros(n + 1, bool)
    reaches[np.unique(lab[np.concatenate([lake_mask | sea] * 3, axis=1)])] = True
    stray = river & ~reaches[mid]
    if not stray.any():
        return width
    width = width.copy()
    wf, drawf = width.ravel(), draw.ravel()
    # Reaching the sea, a lake, or a channel that itself reaches one. Joining a
    # second stray is not an escape: both ends still hang, and taking any river
    # cell as the target leaves them chained together in mid-slope.
    water = ((lake_mask | sea) | (river & reaches[mid])).ravel()
    for i in np.unique(mid[stray]):
        cells = np.flatnonzero((mid == i).ravel() & (wf > 0))
        c = int(cells[np.argmax(wf[cells])])     # widest cell: the downstream end
        chan_w = float(wf[c])
        trail = []
        for _ in range(budget):
            nxt = int(drawf[c])
            if nxt == c:
                break
            c = nxt
            if water[c]:
                for t in trail:
                    _disc(width, t // w, t % w, chan_w)
                trail = None
                break
            trail.append(c)
        if trail is not None:                     # never found water: drop it
            wf[cells] = 0.0
    return width


def rasterize(paths, widths, shape):
    """Stamp every path into a width field, sampling densely enough to stay joined."""
    width_map = np.zeros(shape)
    for pts, wv in zip(paths, widths):
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            steps = max(1, int(np.ceil(np.hypot(*(b - a)) / 0.4)))
            for s in range(steps + 1):
                t = s / steps
                y, x = a + (b - a) * t
                _disc(width_map, y, x, wv[i] + (wv[i + 1] - wv[i]) * t)
    return width_map


# --------------------------------------------------------------------------


def build(h, cfg, rng, sea_level=0.0, routed=None, runoff=None, evap=None):
    """Run the whole stage. Returns (incised height, Water).

    `routed` is (filled, receivers, drainage area) and goes straight to
    `carve_outlets`; see its note. `runoff` and `evap` come from `climate` and
    are what make one basin a lake and its neighbour in a rain shadow a pan.
    """
    h = carve_outlets(h, cfg, sea_level, routed, runoff, evap)

    r = _route_water(h, cfg, sea_level, runoff=runoff, evap=evap)
    filled, rec, flow = r.filled, r.rec, r.flow
    level = r.level[r.lake_id]
    depth = np.where(r.lake_id > 0, np.maximum(level - h, 0.0), 0.0)
    # A lake covers what its water reaches, not the basin holding it: under a
    # balance a basin can stand well below its spill point, and one whose inflow
    # all evaporates holds nothing. Renumber so the basins that came out dry
    # leave no gaps behind - a caller counting lakes off `lake_id.max()` would
    # otherwise ask about ids that no longer exist.
    lake_id = r.lake_id * (depth > 0)
    # One sheet of water, not the k lowest cells wherever they happen to lie.
    # The level is a contour, and a contour through a basin with any texture in
    # its floor picks out tendrils and specks along the whole of it: a
    # part-full lake drew as a spider rather than a pond. Keep the piece holding
    # the deepest cell of each basin and drop the rest, which is what a lake
    # standing at that level would actually cover.
    if lake_id.any():
        piece, npieces = ndimage.label(lake_id > 0, structure=np.ones((3, 3)))
        if npieces > 1:
            keep_piece = np.zeros(npieces + 1, bool)
            deep = ndimage.maximum_position(depth, r.lake_id,
                                            index=np.arange(1, r.n + 1))
            for pos in np.atleast_2d(deep):
                keep_piece[piece[int(pos[0]), int(pos[1])]] = True
            keep_piece[0] = False
            lake_id = lake_id * keep_piece[piece]
    depth = np.where(lake_id > 0, depth, 0.0)
    keep = np.zeros(r.n + 1, bool)
    keep[np.unique(lake_id)] = True
    keep[0] = False
    lake_id = (np.cumsum(keep) * keep)[lake_id]
    level = np.where(lake_id > 0, level, 0.0)

    thresh = cfg.river_threshold * h.size
    # A channel stops at the water's edge, not at the rim of the basin holding
    # it. Basin cells are crossed instead: they are handed to the tracer as
    # `cross` and walked along `draw`, which descends to the lake rather than to
    # the spill. Cutting them out of `channel` instead - which is what this did
    # while the geometry still followed the balance's shortcut - left a quarter
    # of the river cells on seed 1 at 512x384 in the middle of open ground, a
    # median of 4 to 7 cells short of the pond they were running into.
    lake_mask = lake_id > 0
    sea = h <= sea_level
    # Anything the fill had to raise is crossable, not just the basins big and
    # deep enough to be lakes: `lake_min_depth` and `lake_min_area` drop the
    # small ones, and a river running into one of those then stopped dead in a
    # dimple that holds no water. Every stray left after the rest of this was in
    # one - five of the five remaining pieces at 1024x768 on seed 3.
    hollow = (filled - h > 1e-9) & ~lake_mask & ~sea
    draw = _draw_receivers(r.plain, (r.lake_id > 0) & ~lake_mask, lake_mask, r.lake_id)
    channel = (flow >= thresh) & ~sea & ~lake_mask
    # The top and bottom rows are drainage outlets, so water pools along them
    # and draws a ruler-straight river down the map edge. Cut them out.
    channel[:2] = channel[-2:] = False

    paths, widths = [], []
    for pts in polylines(flow, draw, channel, cross=hollow):
        q = flow[pts[:, 0].astype(int), pts[:, 1].astype(int)]
        # Running maximum, because discharge only grows downstream and the field
        # does not say so everywhere: on a basin floor the accounting graph has
        # already sent the water to the basin's exit, so the cells a river
        # crosses to reach the lake read about one cell's worth. Undamped, a
        # trunk narrows to a thread for the last few cells of its run.
        q = np.maximum.accumulate(q)
        # Hydraulic geometry: width goes as a power of discharge. The classic
        # exponent is a half; a little under that holds the trunks in without
        # touching the headwaters, which sit on the 0.7 floor either way.
        # Capping instead only bites on the top percentile and flattens all of
        # it to one width, and lowering `river_width` narrows the mid-sized
        # channels along with the big ones.
        #
        # Measured against `river_width_ref`, never against `river_threshold`.
        # The threshold is the "how many rivers" slider, and dividing by it
        # coupled the two: asking for more rivers made *every* river wider, so
        # at the top of the slider the whole network saturated at
        # `river_width_max` and drew as a mat of equally fat channels rather
        # than as a network with a trunk in it. Discharge is discharge.
        wv = np.clip(cfg.river_width *
                     (q / (cfg.river_width_ref * h.size)) ** cfg.river_width_exp,
                     0.7, cfg.river_width_max)
        pts = _unwrap_x(pts, h.shape[1])
        paths.append(meander(_smooth(pts), wv, cfg, rng))
        widths.append(wv)

    width = rasterize(paths, widths, h.shape)

    # Incise the bed under the channel so rivers sit in valleys, not on top of
    # the land. Lakes and sea keep their own surface.
    cut = cfg.river_incision * np.clip(width / max(1e-6, cfg.river_width_max), 0, 1) ** 0.5
    cut = grid.blur(cut, 0.7) * (h > sea_level) * (lake_id == 0)
    h = h - cut


    # A river is a land feature, so it stops at the water's edge. The raster
    # runs past it in both directions - a mouth is a disc of channel width
    # centred on the last path point, half of which lands in the sea, and a
    # path into a lake is traced to the shore and then painted over the water.
    # Nothing downstream cared until the three got their own colours; now the
    # overrun draws a stripe of the wrong blue across the sea and every lake it
    # feeds. Masked after the incision, which already excluded both and should
    # keep cutting the bed right up to the shoreline.
    width = width * ((h > sea_level) & (lake_id == 0))
    width = _connect_strays(width, lake_mask, h <= sea_level, draw, cfg)

    return h, Water(filled=filled, flow=flow, receivers=rec, lake_id=lake_id,
                    lake_level=level, lake_depth=depth, width=width, polylines=paths,
                    widths=widths)
