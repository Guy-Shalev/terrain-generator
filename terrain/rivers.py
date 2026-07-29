"""Stage 4: lakes and rivers as real features rather than a flow threshold.

Four things happen here, in order, because each depends on the last:

    lake levelling   every closed basin gets one flat water surface at its
                     spill elevation, and a depth field under it
    outlet carving   the pour point of each lake is notched and a channel is
                     cut downstream, so the lake actually has an outflow and
                     the terrain shows the gorge it drains through
    polylines        the D8 network is traced into head-to-mouth paths and
                     smoothed, which removes the eight-direction staircase
    meander + width  each path is displaced sideways by noise along its own
                     arc length and rasterised with a width taken from
                     discharge, then the bed is incised under it

Carving changes the surface, so filling and routing are redone after it.
"""
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


def _levels(lbl, n, filled):
    """One elevation per lake: the lowest fill value in it, i.e. its spill point.

    Planchon-Darboux leaves a tiny epsilon gradient across a filled basin, so
    taking the minimum recovers the flat surface the water would actually sit at.
    """
    level = np.zeros(lbl.shape)
    if n:
        per = ndimage.minimum(filled, lbl, index=np.arange(1, n + 1))
        level = np.concatenate([[0.0], np.atleast_1d(per)])[lbl]
    return level


def carve_outlets(h, cfg, sea_level=0.0, routed=None):
    """Notch each lake's pour point and cut a channel downstream from it.

    Without this a basin fills to its rim and the water has no modelled way
    out: the map shows a lake with no outflowing river and no valley below it.

    A gorge is only cut where the outflow is big enough to become a river.
    Carving is not gated by `river_threshold` otherwise, and the two then
    disagree: the gorge is real terrain but no river is drawn in it, so a very
    high threshold leaves dry trenches winding across the map with no water in
    them. At 384x288 that is 1% of the dug cells at the default threshold and
    30% at six times it. Discharge at the pour point is the right test because
    it is exactly what would flow down the gorge - and it is measured on the
    *filled* surface, where the lake's whole catchment already routes through
    its spill point.

    `routed` is an optional (filled, receivers, drainage area) already computed
    for `h` as it arrives. The erosion stage finishes by filling, routing and
    accumulating the very surface handed here and its caller throws that away,
    so the first pass would otherwise redo it - a third of a second at
    1024x768. The fill and routing are used once and dropped, because after the
    first notch the surface is no longer the one they describe; the drainage
    area is kept for every pass, since carving deepens channels without moving
    the catchments that feed them.
    """
    thresh = cfg.river_threshold * h.size
    flow = None
    for _ in range(cfg.outlet_carve_passes):
        if routed is not None:
            filled, rec, flow = routed
            routed = None
        else:
            filled = hydrology.fill_depressions(h, sea_level)
            rec = None
        lbl, n = _label_lakes(h, filled, cfg, sea_level)
        if n == 0:
            break
        if rec is None:
            rec, _, _ = hydrology.flow_routing(filled)
        if flow is None:
            flow = hydrology.accumulate(filled, rec)

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

        flat, recf, flowf = h.ravel(), rec.ravel(), flow.ravel()
        for pos in np.atleast_2d(pours):
            c = int(pos[0]) * h.shape[1] + int(pos[1])
            if flowf[c] < thresh:   # too little water to cut, or to draw
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


def polylines(flow, rec, channel):
    """Trace the channel network into paths, each running head -> mouth/junction."""
    hh, w = flow.shape
    up = np.bincount(rec.ravel()[channel.ravel()], minlength=flow.size)
    heads = np.flatnonzero(channel.ravel() & (up == 0))
    heads = heads[np.argsort(-flow.ravel()[heads])]  # longest rivers claim first
    recf, chan = rec.ravel(), channel.ravel()
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
            path_end = visited[nxt] or not chan[nxt]
            if path_end:
                path.append(nxt)     # touch the trunk (or the sea) and stop
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
        return
    yy, xx = np.nonzero(sel)
    np.maximum.at(width_map, (ys[yy], xs[xx] % w), chan_w)


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


def build(h, cfg, rng, sea_level=0.0, routed=None):
    """Run the whole stage. Returns (incised height, Water).

    `routed` is (filled, receivers, drainage area) and goes straight to
    `carve_outlets`; see its note.
    """
    h = carve_outlets(h, cfg, sea_level, routed)

    filled = hydrology.fill_depressions(h, sea_level)
    rec, _, _ = hydrology.flow_routing(filled)
    flow = hydrology.accumulate(filled, rec)
    lake_id, n_lakes = _label_lakes(h, filled, cfg, sea_level)
    level = _levels(lake_id, n_lakes, filled)
    depth = np.where(lake_id > 0, np.maximum(level - h, 0.0), 0.0)

    thresh = cfg.river_threshold * h.size
    channel = (flow >= thresh) & (h > sea_level) & (lake_id == 0)
    # The top and bottom rows are drainage outlets, so water pools along them
    # and draws a ruler-straight river down the map edge. Cut them out.
    channel[:2] = channel[-2:] = False

    paths, widths = [], []
    for pts in polylines(flow, rec, channel):
        q = flow[pts[:, 0].astype(int), pts[:, 1].astype(int)]
        # Hydraulic geometry: width goes as a power of discharge. The classic
        # exponent is a half; a little under that holds the trunks in without
        # touching the headwaters, which sit on the 0.7 floor either way.
        # Capping instead only bites on the top percentile and flattens all of
        # it to one width, and lowering `river_width` narrows the mid-sized
        # channels along with the big ones.
        wv = np.clip(cfg.river_width * (q / thresh) ** cfg.river_width_exp,
                     0.7, cfg.river_width_max)
        pts = _unwrap_x(pts, h.shape[1])
        pts = meander(_smooth(pts), wv, cfg, rng)
        paths.append(pts)
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

    return h, Water(filled=filled, flow=flow, receivers=rec, lake_id=lake_id,
                    lake_level=level, lake_depth=depth, width=width, polylines=paths)
