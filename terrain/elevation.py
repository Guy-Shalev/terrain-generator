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
Continent-ocean margins get their own shoreline pass: they sit on a much
steeper slope than the rest of the coast, so elevation noise hardly moves them.
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


def erode_margins(h, t, cfg, rng):
    """Fray the coast where a continental plate meets an oceanic one.

    Those margins come out visibly smoother than the rest of the shoreline -
    long clean sweeps where an open coast has inlets, lobes and offshore
    islands - and the cause is the slope, not missing noise. The crust step
    across a plate-type boundary is several times steeper than a noise-drawn
    coast, so the shoreline there sits on a much steeper slope. `fray_coast`
    perturbs *elevation*, and that moves the zero contour by amplitude/slope,
    so the same noise buys about half the displacement on a margin. The noise
    is present; the contour just does not travel far enough to show it.

    So restore the budget rather than the amplitude: scale the fray by the
    local slope, measured against the median coastal slope of this map, and
    the displacement comes out the same on both kinds of coast by
    construction. Taking the reference from the map keeps that true at any
    resolution, where a fixed slope constant would not.

    Displacing the field sideways instead was the obvious alternative and does
    not work. A warp moves the contour by the displacement itself whatever the
    slope, which is the appealing part, but the displacement is coherent over
    its own wavelength: long waves slide the whole margin across as a smooth
    arc, and short ones at a useful amplitude fold the field. Measured across
    four seeds it barely shifted the shoreline's roughness at any frequency.

    The distance gate is not optional. `cont_self` and `cont_other` are pulled
    from the nearest boundary and so are defined everywhere, which means a
    continental interior far from any ocean still reads as a full type
    mismatch; without the gate this fires on coasts that are not margins.
    """
    hh, w = h.shape
    land = h > 0.0
    if not land.any():
        return h

    # Distance to the shoreline, negative at sea. Masking by *elevation* is the
    # obvious choice and is a trap: a band of fixed elevation is only
    # band/slope cells wide on the ground, about five on a margin, and the
    # shoreline cannot travel past the point where its own mask has faded. It
    # caps the stage at the very coasts it exists to fix - measured, ten times
    # the amplitude bought 0.8 of a cell. A distance corridor is the same width
    # everywhere whatever the slope, so amplitude turns into travel again.
    sd = grid.edt(land) - grid.edt(~land)
    near = (np.abs(t.cont_self - t.cont_other)       # continent meets ocean
            * _ramp(t.dist, cfg.margin_zone))        # near that boundary

    # Blur first: the gain should follow the margin's overall steepness, not
    # whatever single-cell texture noise happens to sit under each shore cell.
    slope = np.hypot(*grid.gradient(grid.blur(h, cfg.margin_slope_blur)))
    ref = float(np.median(slope[np.abs(sd) < 2.0]))
    # Only ever boost. Below the reference the coast is already frayed enough,
    # and scaling down there would flatten the gentle coasts to match.
    gain = np.clip(slope / max(ref, 1e-9), 1.0, cfg.margin_gain_max)

    # A fresh noise pair, finer than the global fray: the shortfall measured on
    # these coasts is in small-scale structure, and reusing the same field
    # would only deepen the fray already there instead of adding detail.
    #
    # Both are divided by their own spread. fBm of these octave counts has a
    # standard deviation near 0.18, not 1, so an amplitude knob used raw is
    # about five times weaker than it reads and no setting of it does anything
    # visible. Normalised, `margin_h` is elevation per standard deviation and
    # the shoreline travels roughly margin_h * gain / slope cells.
    def unit(periods, octaves=4, ridged=False, cap=None):
        f = noise.fbm(hh, w, rng, periods=periods, octaves=octaves, ridged=ridged)
        # Centred as well as scaled: the ridged variant is not zero-mean, and a
        # threshold in standard deviations should mean the same for both.
        u = (f - f.mean()) / max(float(f.std()), 1e-9)
        return u if cap is None else np.clip(u, -cap, cap)

    # Capped like the cut term, and for the same reason. Uncapped, fBm reaches
    # about 3.5 standard deviations and the slope gain multiplies it by up to
    # `margin_gain_max`, so the fray could add over a unit of elevation - and
    # it does that precisely where the coast is steepest, raising mountains
    # along the shoreline instead of moving it. The clip costs a little travel
    # at the extremes and nothing anywhere else.
    #
    # `margin_periods` is deliberately coarse. Fine noise near the zero contour
    # does not bend the coastline, it perforates it: at periods 14 this left 18
    # water pockets cut off from the sea against a baseline of 3. At 8 that is
    # back to 3, and the shoreline travels further for it - the wanted effect
    # is a coast that wanders, not one that is speckled.
    h = h + (near * np.exp(-(sd / cfg.margin_reach) ** 2) * gain
             * (cfg.margin_h * unit(cfg.margin_periods, cap=cfg.margin_cap)
                + cfg.margin_h_fine * unit(cfg.margin_periods * 3,
                                           cap=cfg.margin_cap)))

    # Drowned inlets cut into the coast. Ridged noise, not plain fBm, because
    # its crests run in lines and a line cut into a coast is a ria; thresholding
    # plain fBm digs round pits, which read as craters. Clipped non-negative and
    # subtracted, so this only ever removes land.
    #
    # Both the coarseness and the window placement matter more than the depth.
    # At `margin_cut_periods` 22 and centred 5 cells inland this cut 21 water
    # pockets off from the sea against a baseline of 3 - holes in the coast
    # rather than inlets into it. Coarser, shallower, and centred on the
    # shoreline itself, so a cut opens into the ocean instead of stranding a
    # basin behind it, that falls to 4 and the shoreline still moves further.
    cut = np.clip(unit(cfg.margin_cut_periods, octaves=5, ridged=True)
                  - cfg.margin_cut_thresh, 0.0, cfg.margin_cut_cap)
    return h - cfg.margin_cut_h * near * _band(sd, cfg.margin_cut_offset,
                                               cfg.margin_cut_w) * cut


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
