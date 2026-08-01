"""Handing a world to something else: greyscale PNGs plus what they mean.

Five, because a heightmap on its own imports as grey rock. The bed says what
shape the land is, the lake depths say where water sits on it, the biome indices
say what covers it, and the river paths are vectors rather than pixels because
that is what they were before they were rasterised. The fifth is the config,
which is the only one of them that can rebuild the world rather than describe it.

Eight bits is not enough for the terrain. The map spans a couple of height units
and 256 codes across that is a step of ~8 mm at the scale the lapse rate
implies - fine on a cliff, and visible as terracing on every plain and shelf,
which is exactly where a heightmap is looked at flat. Class indices are 8-bit
because they are labels: there are thirteen biomes and nothing between them.

Written by hand rather than through an imaging library. A greyscale PNG with no
interlacing and no per-row filtering is a signature, three chunks and a zlib
stream, and that is a smaller thing to carry than a dependency.
"""
import hashlib
import itertools
import json
import struct
import warnings
import zlib
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import scipy

from . import climate
from .world import Config

MAX = 65535

# What one height unit is worth on the ground. Not measured - chosen, and
# chosen here rather than in `Config` because nothing in the generator reads
# it. `temp_lapse` is calibrated by reading the ~0.5 that land runs to as a
# 6 km range, which puts a unit at 12 km, and every biome on the map is
# downstream of that reading. An importer has to scale the terrain by
# something, and the number the climate was tuned against beats a guess.
METRES_PER_UNIT = 12000.0


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _write_png(path, a, depth):
    """Greyscale, 8- or 16-bit, filter type 0 on every row - all the format needs."""
    h, w = a.shape
    # PNG samples are big-endian regardless of the machine writing them.
    body = a.astype(">u2" if depth == 16 else np.uint8).view(np.uint8).reshape(h, -1)
    raw = np.empty((h, body.shape[1] + 1), np.uint8)
    raw[:, 0] = 0                                   # filter: none
    raw[:, 1:] = body
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, 0, 0, 0, 0)))
        f.write(_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
        f.write(_chunk(b"IEND", b""))


def _sidecar(path, meta):
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def heightmap(h, path, lo=None, hi=None):
    """Write `h` as a 16-bit PNG, and a JSON sidecar saying what the codes mean.

    A heightmap on its own is unitless: nothing in it says where sea level went
    or how tall the tallest code is, and an engine importing it has to be told.
    The sidecar carries `lo`, `hi`, the world height one code is worth, the code
    sea level landed on, and what a unit is in metres, so
    `lo + code * units_per_code` puts the surface back in the generator's own
    units and multiplying by `metres_per_unit` puts it on the ground.

    Pass `lo` and `hi` to hold two exports on one scale. Anything that has to
    line up needs it - the water surface against the bed, a series of seeds,
    adjacent tiles - because each export normalised to its own range puts the
    same height at a different code.

    Returns the sidecar dict.
    """
    lo = float(np.min(h)) if lo is None else float(lo)
    hi = float(np.max(h)) if hi is None else float(hi)
    span = max(1e-9, hi - lo)
    _write_png(path, np.rint(np.clip((h - lo) / span, 0, 1) * MAX).astype(np.uint16), 16)
    return _sidecar(path, {
        "width": int(h.shape[1]), "height": int(h.shape[0]), "bits": 16,
        "lo": lo, "hi": hi, "units_per_code": span / MAX,
        "metres_per_unit": METRES_PER_UNIT,
        # Not clamped: on a field that never reaches sea level this sits outside
        # 0..65535, and saying so is more use than pinning it to an edge it is
        # not on.
        "sea_code": (0.0 - lo) / span * MAX,
        "wrap": "x",    # the map is a cylinder; y clamps
    })


def indexmap(a, path, names):
    """Class indices as an 8-bit PNG, with the class names beside it.

    Greyscale and not a palette. A palette would make the file look right in an
    image viewer and be wrong for the job: what an engine wants is the index, to
    pick a texture or a scatter rule with, and a palette buries that under
    colours it would have to match back. `names` in the sidecar is the legend,
    and it is written from the generator's own list so the two cannot drift.
    """
    a = np.asarray(a)
    if a.min() < 0 or a.max() > 255:
        raise ValueError(f"indices must fit in a byte, got {a.min()}..{a.max()}")
    _write_png(path, a.astype(np.uint8), 8)
    return _sidecar(path, {
        "width": int(a.shape[1]), "height": int(a.shape[0]), "bits": 8,
        "classes": list(names), "wrap": "x",
    })


def river_paths(water, path):
    """Channel centrelines as vectors, which is what they were before rasterising.

    Rasterised, a river is a few cells of a width field and its course has to be
    recovered by tracing them. These are the smoothed, meandered paths the
    rasteriser was handed, with the channel width at every node, so a river can
    carry a spline, a mesh, a sound, or a boundary without being read back out
    of pixels.

    Points are `[x, y]` in cell units, which is the transpose of how the arrays
    hold them, and x can run outside the map: a path crossing the seam keeps
    counting rather than jumping, so the polyline stays continuous. Take it
    modulo the width when placing it.
    """
    paths = [{"points": [[round(float(x), 3), round(float(y), 3)] for y, x in pts],
              "width": [round(float(v), 3) for v in wv]}
             for pts, wv in zip(water.polylines, water.widths)]
    Path(path).write_text(json.dumps(
        {"coords": "xy, in cells", "wrap": "x", "rivers": paths}))
    return paths


def _stamp():
    """What produced a world, beyond the knobs: the code, and what it ran on.

    The config says what the generator was asked for. It does not say which
    generator, and every number in it can be unchanged while an edit to
    `elevation.py` moves every coastline. So the package's own source is
    hashed, alongside the two libraries whose arithmetic the output is made of.

    Conservative in one direction only. Equal means the same code, so the map
    is the same map. Different means unverified, not different - reformatting a
    comment moves the hash and nothing else.
    """
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return f"src:{h.hexdigest()[:12]} numpy:{np.__version__} scipy:{scipy.__version__}"


def config(cfg, path):
    """The seed and every knob, written so that loading it rebuilds this world.

    The rasters are the *result*, and a lossy one: 16 bits is a third of a metre
    at `METRES_PER_UNIT`, and nothing in them carries the plates, the stress, the
    temperature or the flow. This file is the map itself. Everything else is a
    picture of it.

    `ref_width` is overwritten with the map's own width, and that is the whole
    subtlety. `generate` scales every pixel-denominated knob from `ref_width` to
    the width being built and keeps the *scaled* copy on the world, so handing
    that copy back scales it a second time and builds a different world -
    silently, and only at sizes other than the reference. Re-anchoring makes
    `_scale_to_size` a no-op, which is correct: these numbers are already tuned
    at this width.
    """
    meta = dict(asdict(cfg), ref_width=cfg.width, generator=_stamp())
    Path(path).write_text(json.dumps(meta, indent=1))
    return meta


def load_config(path):
    """Read one back. JSON has no tuples, and a few knobs are pairs.

    Unknown keys are dropped rather than raising: a config written by a build
    that has since gained or lost a knob should still rebuild what it can, and
    the stamp below is what says whether to trust the result.
    """
    d = json.loads(Path(path).read_text())
    stamp, here = d.pop("generator", None), _stamp()
    if stamp is not None and stamp != here:
        warnings.warn(f"config was written by [{stamp}], this is [{here}]; "
                      "the rebuilt world may not match")
    known = {f.name for f in fields(Config)}
    if d.keys() - known:
        warnings.warn(f"ignoring unknown config keys: {sorted(d.keys() - known)}")
    return Config(**{k: tuple(v) if isinstance(v, list) else v
                     for k, v in d.items() if k in known})


def _new_dir(parent, name):
    """`name`, or `name-2`, `name-3`... - the first one that does not exist yet.

    `mkdir` without `exist_ok` *is* the check, and that is the point: asking
    whether a directory exists and then creating it is two steps, and an export
    started in between wins the same name. Creating it and catching the failure
    is one step the filesystem does atomically.
    """
    parent = Path(parent)
    for n in itertools.count(1):
        d = parent / (name if n == 1 else f"{name}-{n}")
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            continue


def bundle(world, parent="out/export"):
    """Everything an engine needs, in a folder of its own. Returns that folder.

    A new one every call - `seed7`, then `seed7-2` - because an export is a
    thing you keep, and the reason to press the key twice is usually that the
    second world is worth comparing to the first. Overwriting is the one
    behaviour that cannot be undone from the outside.

    The files inside are plainly named, since the folder carries the seed.

    The lake depths go out on their own scale rather than the bed's. The bed
    spans the abyss to the summit and a lake is metres deep in that, so shared
    it would be a handful of codes wide; on its own the full 16 bits describe
    the water, and it is a depth either way - it needs no common zero with
    anything, only its own.
    """
    d = _new_dir(parent, f"seed{world.cfg.seed}")
    heightmap(world.height, d / "height16.png")
    heightmap(world.water.lake_depth, d / "lakes16.png")
    indexmap(world.biome, d / "biome.png", climate.BIOME_NAMES)
    river_paths(world.water, d / "rivers.json")
    config(world.cfg, d / "config.json")
    return d
