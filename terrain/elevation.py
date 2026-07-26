"""Stage 2: turn plate strain into an elevation field.

Everything is expressed as "amplitude times a falloff in distance from the
nearest boundary", which is what gives belts their width:

    collision      both sides continental, converging -> broad high range
    subduction     trench on the down-going side, volcanic arc inland of the
                   overriding side, plus a cordillera if that side is continent
    rift           continental: valley with raised shoulders
                   oceanic: mid-ocean ridge, and seafloor deepens with age
    transform      narrow shear scar, slight en-echelon ridging

Then hotspot island chains, fractal texture, sea level, and a frayed coastline.
"""
import numpy as np

from . import grid, noise, tectonics


def _ramp(d, width):
    """1 at the boundary, decaying inland."""
    return np.exp(-d / max(1e-6, width))


def _band(d, offset, width):
    """A ridge sitting `offset` cells away from the boundary."""
    return np.exp(-((d - offset) ** 2) / (2 * max(1e-6, width) ** 2))


def tectonic_relief(t, cfg, rng):
    """Elevation from crust thickness plus boundary tectonics, before texture."""
    d = t.dist
    conv, rift, shear = t.conv, t.rift, t.shear
    crust = t.crust                    # local crust thickness, gates everything
    ocean = 1.0 - crust
    # Pairing is a plate-scale property: which two kinds of plate meet here.
    both_cont = t.cont_self * t.cont_other
    subduction = np.clip(np.abs(t.cont_self - t.cont_other) +
                         (1 - t.cont_self) * (1 - t.cont_other), 0, 1)
    over = t.overriding

    # Isostasy: thick continental crust floats high, thin oceanic crust sits low.
    h = cfg.ocean_base + (cfg.cont_base - cfg.ocean_base) * crust

    # Interior variation so cratons are not featureless slabs.
    hh, w = h.shape
    h += cfg.upland_h * noise.fbm(hh, w, rng, periods=cfg.upland_periods, octaves=5) * crust

    # Continent-continent collision: one wide range straddling the suture.
    h += cfg.collision_h * conv * both_cont * crust * _ramp(d, cfg.collision_w)

    # Subduction: trench under the down-going plate, arc over the overriding one.
    down = (1.0 - over) * subduction * conv
    up = over * subduction * conv
    h -= cfg.trench_h * down * _ramp(d, cfg.trench_w)
    # An arc is a line of volcanoes, not a wall: break it up along strike so an
    # intra-oceanic one surfaces as an island chain instead of a seam.
    segs = np.clip(noise.fbm(hh, w, rng, periods=cfg.arc_break_periods, octaves=4)
                   * 2.4 + cfg.arc_break_bias, 0, 1)
    h += cfg.arc_h * up * segs * _band(d, cfg.arc_offset, cfg.arc_w)
    h += cfg.cordillera_h * up * crust * _ramp(d, cfg.cordillera_w)
    # Fore-arc bulge, only where the overriding plate is continental - in open
    # ocean it just draws a second soft ridge parallel to the trench.
    h += 0.15 * cfg.arc_h * up * crust * _band(d, cfg.arc_offset * 2.2, cfg.arc_w * 2)

    # Rifting: a graben on land, a spreading ridge at sea.
    cont_rift = rift * crust
    h -= cfg.rift_depth * cont_rift * _ramp(d, cfg.rift_w)
    h += cfg.rift_shoulder * cont_rift * _band(d, cfg.rift_w * 1.6, cfg.rift_w)
    h += cfg.ridge_h * rift * ocean * _ramp(d, cfg.ridge_w)

    # Transform faults: narrow scar, plus low ridging along strike.
    h -= cfg.transform_h * shear * _ramp(d, cfg.transform_w)
    h += 0.4 * cfg.transform_h * shear * _band(d, cfg.transform_w * 3, cfg.transform_w * 2)

    # Ocean floor subsides as it ages away from spreading centres.
    ridge_src = (t.rift > cfg.ridge_age_thresh) & t.boundary & ~t.is_cont
    if ridge_src.any():
        # Distance from the spreading centres, warped: unwarped, its contours
        # are smooth arcs bounded by the medial axis between ridges, and the
        # abyssal plain ends up tinted in visibly plate-shaped zones.
        d_ridge = grid.edt(~ridge_src)
        d_ridge = noise.warp(d_ridge, *tectonics.warp_field(
            hh, w, rng, cfg.age_warp_periods, cfg.age_warp))
        h -= cfg.age_depth * (1 - np.exp(-d_ridge / cfg.age_scale)) * ocean

    return h


def hotspot_chains(h, t, cfg, rng):
    """Intraplate volcanism smeared into a chain by the plate's own motion."""
    hh, w = h.shape
    for _ in range(cfg.n_hotspots):
        # Oceanic crust only: on a continent the same stamp reads as a row of
        # round craters, and what this is for is island chains.
        for _try in range(40):
            y = rng.uniform(hh * 0.1, hh * 0.9)
            x = rng.uniform(0, w)
            if t.crust[int(y), int(x)] < cfg.hotspot_max_crust:
                break
        else:
            continue
        p = t.plate[int(y), int(x)]
        vy, vx = t.vel[p]
        step = cfg.hotspot_spacing
        n_isl = rng.integers(cfg.hotspot_chain[0], cfg.hotspot_chain[1] + 1)
        amp = cfg.hotspot_h * rng.uniform(0.7, 1.3)
        for i in range(n_isl):
            # Islands trail behind the drifting plate and erode as they go.
            cy = y - vy * step * i + rng.normal(0, step * 0.25)
            cx = x - vx * step * i + rng.normal(0, step * 0.25)
            if not (0 <= cy < hh):
                break
            decay = amp * (0.55 ** (i * 0.8)) * rng.uniform(0.7, 1.15)
            # Three offset lobes rather than one dome: a single gaussian reads
            # as a suspiciously perfect circle once it breaks the surface.
            for _ in range(3):
                s = cfg.hotspot_sigma * rng.uniform(0.55, 1.15)
                grid.stamp(h, cy + rng.normal(0, s * 0.6), cx + rng.normal(0, s * 0.6),
                           decay * rng.uniform(0.35, 0.6), s)
    return h


def texture(h, t, cfg, rng):
    """Fractal detail, weighted so young mountains are rough and abyss is smooth."""
    hh, w = h.shape
    relief = grid.normalize(grid.blur(np.abs(h - grid.blur(h, 12)), 6))
    tectonic_act = grid.normalize(grid.blur(t.conv + 0.6 * t.rift + 0.4 * t.shear, 3))
    rough = np.clip(0.35 + 1.3 * np.maximum(relief, tectonic_act * _ramp(t.dist, 40)), 0, 1.6)

    ridged = noise.fbm(hh, w, rng, periods=cfg.texture_periods, octaves=7, ridged=True)
    fine = noise.fbm(hh, w, rng, periods=cfg.texture_periods * 2, octaves=5)
    # Warping the ridged field bends crests instead of leaving them parallel.
    wy = noise.fbm(hh, w, rng, periods=3, octaves=3) * cfg.texture_warp
    wx = noise.fbm(hh, w, rng, periods=3, octaves=3) * cfg.texture_warp
    ridged = noise.warp(ridged, wy, wx)

    land_ish = np.clip((h - cfg.ocean_base * 0.5) * 2, 0, 1)
    h = h + cfg.texture_h * rough * ridged * land_ish
    h = h + cfg.texture_h * 0.45 * fine
    h = h + cfg.abyss_h * fine * (1 - land_ish)
    return h


def coastal_plain(h, cfg, rng):
    """Bound how fast the land may climb as it leaves the shore.

    Boundary uplift is placed by distance from the plate boundary and crust
    thins towards the margin, so at an ocean-continent boundary both peak in
    the same place: the range crests a few cells from the water and the shelf
    drops away directly beneath it. Capping the rise per cell near the sea
    gives that range a plain and foothills in front of it instead of a wall.

    The cap is self-gating - a coast that already rises gently is under it and
    is left alone - and the allowed slope is noise-modulated along the shore,
    so some coasts still come down steeply into the water. That contrast is
    the point; a uniform cap draws a terrace round every continent.

    Building land *out* into the sea instead was the obvious alternative and
    does not work: any offset keyed on distance from the shore raises a bench
    of near-constant width, which reads as a bright ribbon traced round each
    landmass and grows a shore-parallel river down the middle of it.
    """
    land = h > 0.0
    if not land.any():
        return h
    # edt measures to the nearest False cell, so pass `land`: on a land cell
    # that is the distance out to the sea. Passing ~land reads zero everywhere.
    inland = grid.edt(land)
    n = noise.fbm(*h.shape, rng, periods=cfg.coast_slope_periods, octaves=4)
    allowed = cfg.coast_slope * (0.5 + 1.5 * np.clip(0.5 + 0.5 * n, 0, 1) ** 1.5)
    # Confined to a coastal zone, and faded out across it. A cap that applies
    # at any distance inland binds wherever a peak is under `slope * inland`,
    # which on a continent a couple of hundred cells wide is most of it - the
    # whole world flattens instead of just the shoreline.
    zone = np.exp(-(inland / cfg.coast_plain_zone) ** 2)
    return np.where(land, h + (np.minimum(h, allowed * inland) - h) * zone, h)


def fray_coast(h, cfg, rng):
    """Break up the shoreline where it would otherwise be a smooth contour.

    Two scales, because a coast is ragged in two ways: bays, headlands and
    offshore islands at tens of cells, and a fretted edge at two or three. The
    band restricts both to elevations near sea level, so inland terrain and the
    deep ocean are untouched.
    """
    hh, w = h.shape
    big = noise.fbm(hh, w, rng, periods=cfg.coast_periods * 0.4, octaves=4)
    fine = noise.fbm(hh, w, rng, periods=cfg.coast_periods * 2, octaves=4)
    band = np.exp(-(h / cfg.coast_band) ** 2)
    return h + band * (cfg.coast_h * big + cfg.coast_h_fine * fine)
