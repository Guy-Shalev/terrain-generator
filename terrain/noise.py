"""Perlin/fBm noise on the world grid, tileable in x."""
import numpy as np
from scipy.ndimage import map_coordinates


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(h, w, py, px, rng):
    """Gradient noise, py by px lattice cells, seamless across the x seam."""
    py, px = max(1, int(py)), max(1, int(px))
    ang = rng.uniform(0, 2 * np.pi, (py + 1, px + 1))
    # Kept as two scalar fields rather than one array of 2-vectors: gathering
    # the stacked form builds an (h, w, 2) temporary and then reads it with a
    # stride of two, which is the slowest part of the whole noise stack.
    #
    # Single precision from here on. The gathers below write a full map each,
    # four of them per octave and six octaves per fBm, so this is the second
    # most expensive thing the generator does and it is bound by how many bytes
    # move rather than by the arithmetic. A gradient is a sine: seven digits is
    # six more than the field needs, and float32 propagates on into every
    # elevation and climate field built out of it.
    gy, gx = np.sin(ang, dtype=np.float32), np.cos(ang, dtype=np.float32)
    gy[:, -1] = gy[:, 0]  # seam
    gx[:, -1] = gx[:, 0]
    ys = np.linspace(0, py, h, endpoint=False)
    xs = np.linspace(0, px, w, endpoint=False)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]

    def dot(dy, dx):
        """Corner gradient dotted with the offset to it, in place."""
        iy, ix = y0 + dy, x0 + dx
        out = gy[iy][:, ix]
        out *= fy - dy
        other = gx[iy][:, ix]
        other *= fx - dx
        out += other
        return out

    u, v = _fade(fx), _fade(fy)
    # Each corner is evaluated once. Written as the plain lerp expression the
    # two left-hand corners get computed twice over, and a corner is a full
    # gather plus two multiplies at map resolution.
    d00, d01, d10, d11 = dot(0, 0), dot(0, 1), dot(1, 0), dot(1, 1)
    for lo, hi, t in ((d00, d01, u), (d10, d11, u), (d01, d11, v)):
        np.subtract(hi, lo, out=hi)
        np.multiply(hi, t, out=hi)
        np.add(hi, lo, out=hi)          # hi now holds lerp(lo, hi, t)
    return np.multiply(d11, 1.42, out=d11)


def fbm(h, w, rng, periods=4, octaves=6, gain=0.5, ridged=False):
    """Fractal sum of perlin octaves. Ridged variant gives sharp crests."""
    total = np.zeros((h, w), np.float32)
    amp, norm, p = 1.0, 0.0, float(periods)
    aspect = h / w
    for _ in range(octaves):
        n = perlin(h, w, round(p * aspect), round(p), rng)
        if ridged:
            n = 1.0 - np.abs(n)
            n = n * n * 2.0 - 1.0
        total += amp * n
        norm += amp
        amp *= gain
        p *= 2.0
    return total / norm


def warp(field, dy, dx):
    """Resample `field` at (y+dy, x+dx); x wraps, y clamps."""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([np.clip(yy + dy, 0, h - 1), (xx + dx) % w])
    return map_coordinates(field, coords, order=1, mode="grid-wrap")
