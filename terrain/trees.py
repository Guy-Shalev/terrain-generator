"""Trees: the one stage that reads the finished world and changes nothing in it.

Everything upstream is a mechanism. This is the end product - what the plates,
the rain and the rivers add up to on the ground - so it runs last, reads only
finished fields, and feeds nothing back. Its own generator off the seed, for the
same reason the climate stage takes one: drawing from the shared stream here
would move the river meanders on every seed the generator has ever made.

Two things come out. A `cover` field, which is canopy fraction per cell and is
what an engine wants to scatter from, and a *point set*, which is what makes a
forest look like trees rather than like a green wash. The field is the
intermediate; the points are the product. Same move `rivers` already makes by
keeping polylines: a scatter is positions, and rasterising it is the lossy step.

Nothing here is derivable from `biome.png` alone, and that is the point of the
`cover` field being continuous. The biome grid is a thirteen-way cut - a cell is
temperate forest or it is steppe - and every clearing, timberline and gallery
strip in the world lives in the ground between those two answers.
"""
from dataclasses import dataclass

import numpy as np

from . import climate, grid, noise

# Tree kinds. Four, because that is how many silhouettes read apart at map
# scale: a spire, a dome, a flat crown on a bare stem, and a dot.
CONIFER, BROADLEAF, PALM, SCRUB = range(4)
KIND_NAMES = ["conifer", "broadleaf", "palm", "scrub"]

# What kind grows in each `climate` biome, indexed by biome id. Not a climate
# model - a look-up, because the climate that would decide it has already been
# run and banded, and re-deriving leaf habit from temperature would be the same
# cut drawn twice. Temperate rainforest is coniferous on purpose: the real ones
# are, from Sitka to Valdivia. Hot desert gets palms because the only trees a
# desert has stand where the water is, and the gallery term is what puts them
# there.
KIND_BY_BIOME = np.array([
    SCRUB,      # ocean - never drawn, cover is zero there
    SCRUB,      # ice
    SCRUB,      # tundra
    CONIFER,    # taiga
    SCRUB,      # cold desert
    SCRUB,      # steppe
    BROADLEAF,  # temperate forest
    CONIFER,    # temperate rainforest
    SCRUB,      # shrubland
    PALM,       # hot desert
    PALM,       # savanna - flat-crowned, an acacia read at three pixels
    PALM,       # tropical seasonal forest
    BROADLEAF,  # tropical rainforest
], dtype=np.int8)


@dataclass
class Trees:
    cover: np.ndarray   # canopy fraction per cell, 0..1
    pos: np.ndarray     # (n, 2) float, (y, x) in cells
    kind: np.ndarray    # (n,) index into KIND_NAMES
    size: np.ndarray    # (n,) relative, around 1.0

    def __len__(self):
        return len(self.kind)


def canopy(world, cfg, rng):
    """Canopy fraction per cell. Four terms, every one of them continuous.

    Moisture sets it, the timberline cuts it off, rivers carry it into country
    too dry for it, and noise clumps it so that what grows comes out in stands
    with clearings between them rather than as an even lawn.
    """
    mi = climate.moisture_index(world.temp, world.swing, world.runoff, cfg)

    # Groves. The noise goes into the moisture index *before* the curve, not
    # onto the cover after it: the curve saturates at both ends, so noise
    # applied to the output is flattened everywhere except the middle band and
    # a closed forest comes out with no clearings in it at all.
    n = noise.fbm(*mi.shape, rng, periods=cfg.tree_grove_periods, octaves=4)
    mi = mi * (1.0 + cfg.tree_grove * n / max(1e-9, float(n.std())))

    # Moisture. Trees close a canopy somewhere between steppe and forest: below
    # `tree_mi_open` they are scattered individuals, above `tree_mi_closed` the
    # canopy is already continuous and more rain cannot add to it.
    t = np.clip((mi - cfg.tree_mi_open) /
                max(1e-9, cfg.tree_mi_closed - cfg.tree_mi_open), 0.0, 1.0)
    cover = t * t * (3.0 - 2.0 * t)                 # smoothstep

    # The timberline, as a fade rather than the hard cut `biomes` needs. What
    # `TREE_C` gives a classifier is one answer per cell; what a mountain
    # actually has is a band where the forest thins out and then stops.
    warmest = world.temp + world.swing
    treeline = np.clip((warmest - climate.TREE_C) / cfg.tree_line_fade, 0.0, 1.0)
    cover *= treeline

    # Gallery forest. A river carries trees through country whose rainfall
    # cannot keep them - the Nile and the Okavango are green lines drawn on tan.
    # Scaled by (1 - cover) so it does nothing in a rainforest and everything in
    # a savanna: it stands in for water the rain did not supply, and where the
    # rain already did there is nothing left for it to add. Gated by the
    # timberline too, or a river would carry forest over the tree line with it.
    near = np.exp(-grid.edt(~(world.rivers | world.lakes)) / cfg.tree_gallery_reach)
    cover += cfg.tree_gallery * near * (1.0 - cover) * treeline

    cover = np.clip(cover, 0.0, 1.0)
    cover[~world.land] = 0.0
    cover[world.lakes] = 0.0
    cover[world.biome == climate.ICE] = 0.0
    return cover


def kind_mix(biome, land, cfg):
    """Probability of each kind per cell, as a blurred local composition.

    `KIND_BY_BIOME` on its own is a stencil: it reads a thirteen-way cut, so
    every tree inside the taiga comes out coniferous and every tree a cell
    outside it does not, and the boundary draws as a hard line between two
    species. Real forests do not do that - conifers thin out through a belt of
    mixed stand and broadleaves come in, over a distance far wider than a cell.

    So the *composition* is blurred, not the colours. This is the mirror of
    `biome_soften`, which blurs a boundary's rendering and changes nothing about
    what is there; this changes what is there and lets every individual tree keep
    its own kind and its own silhouette. A cell in the middle of a transition
    does not grow a half-conifer - it grows both, in the proportion its
    neighbourhood does.

    Masked to land and divided by the blurred mask, for the reason `biome_rgb`
    does it: ocean carries `KIND_BY_BIOME[OCEAN]` and without the mask a coastal
    forest gets a third of its trees turned to scrub by water offshore.

    Defined over land, which is the only place anything samples it - a land cell
    always has itself in the blur, so the four sum to exactly one there. Out at
    sea, beyond the blur's reach of any coast, they sum to zero instead of to
    something arbitrary.
    """
    k = KIND_BY_BIOME[biome]
    m = land.astype(float)
    p = np.stack([grid.blur((k == i) * m, cfg.tree_mix)
                  for i in range(len(KIND_NAMES))], axis=-1)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-9)


def scatter(cover, biome, land, cfg, rng):
    """Turn the density field into individual trees, on a jittered grid.

    One candidate per `tree_spacing` square, offset at random inside its own
    square, kept with probability `cover`. That gives a count proportional to
    the field with no two trees closer than the jitter allows - a Poisson disc
    without the rejection loop, which at these densities buys nothing a stamp
    three pixels wide would show.

    `tree_spacing` is deliberately *not* scaled with map size. Every other pixel
    knob is, so that a landform covers the same fraction of the world at any
    resolution - but a tree is drawn at a fixed stamp size, and holding its
    ground area fixed instead would shrink the canopy's texture as the map grew.
    The thing to keep constant here is how a forest looks, not how many hectares
    a tree owns.
    """
    h, w = cover.shape
    s = max(0.5, cfg.tree_spacing)
    gy, gx = np.meshgrid(np.arange(0, h, s), np.arange(0, w, s), indexing="ij")
    y = gy.ravel() + rng.random(gy.size) * s
    x = gx.ravel() + rng.random(gx.size) * s
    yi = np.clip(y.astype(int), 0, h - 1)
    xi = x.astype(int) % w
    p = cover[yi, xi]
    keep = rng.random(p.size) < p
    yi, xi = yi[keep], xi[keep]
    # Size tracks how good the ground is - a tree on the edge of its range is a
    # smaller tree - with a lognormal jitter on top so a stand is not a stencil.
    size = (0.55 + 0.45 * cover[yi, xi]) * np.exp(
        rng.normal(0.0, cfg.tree_size_var, yi.size))
    # Kind is drawn last on purpose: every earlier draw keeps its place in the
    # stream, so adding this left the positions and sizes of an existing seed
    # exactly where they were.
    c = np.cumsum(kind_mix(biome, land, cfg)[yi, xi], axis=1)
    u = rng.random(yi.size)
    kind = np.minimum((u[:, None] > c).sum(axis=1), len(KIND_NAMES) - 1)
    return Trees(cover=cover, pos=np.stack([y[keep], x[keep]], axis=1),
                 kind=kind.astype(np.int8), size=size)


def build(world, rng):
    return scatter(canopy(world, world.cfg, rng), world.biome, world.land,
                   world.cfg, rng)
