"""Invariant checks for the generator. Run: python test_terrain.py"""
import numpy as np

from terrain import Config, generate
from terrain import grid, hydrology, noise, render, rivers


def test_grid_wraps():
    a = np.arange(12).reshape(3, 4).astype(float)
    assert np.array_equal(grid.shift(a, 0, 1)[:, -1], a[:, 0]), "x must wrap"
    assert np.array_equal(grid.shift(a, -1, 0)[0], a[0]), "y must clamp"
    m = np.ones((8, 8), bool)
    m[4, 0] = False
    d = grid.edt(m)
    assert d[4, 7] == 1.0, "distance must cross the seam"


def test_noise_seamless():
    rng = np.random.default_rng(1)
    n = noise.fbm(64, 64, rng, periods=4, octaves=4)
    seam = np.abs(n[:, 0] - n[:, -1]).max()
    interior = np.abs(np.diff(n, axis=1)).max()
    assert seam <= interior * 1.5, f"seam discontinuity {seam} vs {interior}"
    assert np.abs(n).max() < 1.6


def test_fill_and_flow():
    rng = np.random.default_rng(3)
    h = -np.hypot(*np.mgrid[-16:16, -16:16]) * 0.05 + rng.normal(0, 0.01, (32, 32))
    h[16, 16] = -0.3  # a pit near the summit that must be filled
    filled = hydrology.fill_depressions(h, sea_level=-9.0)
    assert (filled >= h - 1e-9).all(), "filling may only raise the surface"
    assert filled[16, 16] > h[16, 16], "pit was not filled"
    rec, rec_d, _ = hydrology.flow_routing(filled)
    acc = hydrology.accumulate(filled, rec)
    assert acc.min() >= 1.0
    assert acc.max() <= h.size
    # Every drop must end up somewhere: total outflow at sinks == cell count.
    sinks = rec.ravel() == np.arange(h.size)
    assert abs(acc.ravel()[sinks].sum() - h.size) < 1e-6


def _bumpy(shape, seed, scale=4):
    rng = np.random.default_rng(seed)
    return grid.blur(rng.normal(0, 1, shape), scale) * 3.0


def test_fill_warm_start_is_exact():
    """The multigrid warm start may change the cost, never the answer."""
    for shape in ((160, 224), (150, 198)):   # even and odd halvings
        h = _bumpy(shape, 11)
        outlet = h <= 0.0
        outlet[0] = outlet[-1] = True
        exact = hydrology._solve(h, outlet, 1e-5, None, 6000)
        assert np.abs(hydrology.fill_depressions(h) - exact).max() < 1e-12, shape


def test_accumulate_matches_ordered_walk():
    """Level peeling and the elevation-ordered walk are both topological."""
    h = _bumpy((96, 128), 4)
    filled = hydrology.fill_depressions(h, sea_level=-99.0)
    rec, _, _ = hydrology.flow_routing(filled)
    walk = np.ones(filled.size)
    r = rec.ravel()
    for i in np.argsort(filled, axis=None)[::-1]:
        if r[i] != i:
            walk[r[i]] += walk[i]
    assert np.array_equal(hydrology.accumulate(filled, rec), walk.reshape(h.shape))


def test_outlet_carving_drains_basins():
    """Carving must remove impounded volume, never add it."""
    cfg = Config(width=96, height=96, seed=4)
    rng = np.random.default_rng(4)
    y, x = np.mgrid[0:96, 0:96]
    h = 0.4 - 0.004 * np.hypot(y - 48, x - 48) + 0.05 * rng.normal(0, 1, (96, 96))
    h = grid.blur(h, 3)
    h[36:60, 36:60] -= 0.12  # a basin with no way out
    before = (hydrology.fill_depressions(h) - h).sum()
    after_h = rivers.carve_outlets(h.copy(), cfg)
    after = (hydrology.fill_depressions(after_h) - after_h).sum()
    assert after < before, f"carving did not drain anything: {before} -> {after}"
    assert (after_h <= h + 1e-12).all(), "carving may only lower the surface"


def test_lakes_and_rivers():
    w = generate(Config(width=192, height=144, seed=9, erosion_passes=2), verbose=False)
    wat = w.water

    lake = wat.lake_mask
    if lake.any():
        # One flat surface per lake, always at or above its own bed.
        for i in range(1, wat.lake_id.max() + 1):
            sel = wat.lake_id == i
            assert np.ptp(wat.lake_level[sel]) < 1e-9, f"lake {i} surface is not level"
            assert (wat.lake_level[sel] >= w.height[sel] - 1e-9).all()
        assert (wat.lake_depth[lake] > 0).all()
    assert (wat.lake_depth[~lake] == 0).all()

    assert wat.polylines, "no rivers traced"
    for pts in wat.polylines:
        assert len(pts) >= 3
    # Discharge only grows downstream, so width must too.
    for pts in wat.polylines:
        q = wat.flow[np.clip(pts[:, 0].astype(int), 0, w.height.shape[0] - 1),
                     pts[:, 1].astype(int) % w.height.shape[1]]
        assert q[-1] >= q[0] * 0.5, "path is not running downstream"
    assert wat.river_mask.any()
    assert wat.width[wat.river_mask].min() > 0
    assert wat.width.max() <= w.cfg.river_width_max + 1e-9
    # Rivers are cut into the bed, never raised above it.
    assert (w.height <= w.height_eroded + 1e-9).all()


def test_meander_stays_attached():
    rng = np.random.default_rng(0)
    cfg = Config()
    pts = np.stack([np.arange(80.0), np.zeros(80)], axis=-1)
    out = rivers.meander(pts.copy(), np.full(80, 2.0), cfg, rng)
    ends = np.hypot(*(out - pts)[[0, -1]].T)
    assert (ends < 1e-9).all(), "endpoints moved; tributaries would detach"
    assert np.abs(out[:, 1]).max() > 0.2, "no meander at all"
    assert np.abs(out[:, 1]).max() < cfg.meander_amp * np.sqrt(2.0) * 1.5


def test_world():
    w = generate(Config(width=128, height=96, seed=5, erosion_passes=2), verbose=False)
    assert w.height.shape == (96, 128)
    assert np.isfinite(w.height).all()
    assert 0.15 < w.land.mean() < 0.5, f"land fraction off: {w.land.mean()}"
    # y does not wrap, so the poles must close the world with ocean rather than
    # leaving continents sliced off at the top and bottom edges.
    assert not w.land[[0, 1, -2, -1]].any(), "land runs off the top or bottom edge"
    assert w.tect.plate.min() == 0 and w.tect.plate.max() == w.cfg.n_plates - 1
    assert w.tect.boundary.any() and w.tect.conv.max() > 0
    # Erosion must remove more than it adds on land.
    delta = (w.height_eroded - w.height_pre)[w.land]
    assert delta.mean() < 0, "erosion should lower land on average"
    assert w.rivers.any(), "no rivers formed"
    for i in range(len(render.LAYERS)):
        name, img = render.layer(w, i)
        assert img.shape == (96, 128, 3) and img.dtype == np.uint8, name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
