"""Pygame viewer: pan, zoom, layer switching, regeneration.

    python main.py [--seed N] [--size WxH] [--windowed WxH]

Everything is in the menu bar, each entry with its shortcut key beside it.
"""
import argparse
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import pygame

from terrain import Config, generate, export, render


def _map_height(width):
    """4:3 height for a given width, even so the block pooling halves cleanly."""
    return int(width * 3 / 4) // 2 * 2


# The size slider walks this ladder, so one slider is still enough for world
# size. 4:3 up to 1024, then the 16:9 screen sizes. Ordered by cell count, so
# dragging right is always "bigger and slower" - and it is much slower at the
# top: cost grows faster than area (the erosion and fill loops need more passes
# on a wider grid), so 2.6x the cells is 4.4x the wait - ~5s at 640x480, ~76s at
# 1920x1080, and 4K extrapolates to ~10min and ~4 GB.
SIZES = ([(w, _map_height(w)) for w in range(192, 1025, 64)]
         + [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)])


# "How many rivers" and "how many lakes" are cutoffs in the config: a river
# needs this much drainage area, a lake this much depth, so *raising* either
# knob leaves fewer. Sliders read better counting upwards, so they carry a
# divisor - slider 6 reproduces the config defaults, and the numerators are
# taken from those defaults so the two stay in step.
_C = Config()
RIVER_SCALE = _C.river_threshold * 6      # river_threshold = RIVER_SCALE / v
LAKE_SCALE = _C.lake_min_depth * 6        # lake_min_depth  = LAKE_SCALE / v


def _size_index(cfg):
    """Slider rung for the running config. --size takes any WxH, on the ladder
    or not, so the knob starts on whichever rung is closest by cell count."""
    return min(range(len(SIZES)),
               key=lambda i: abs(SIZES[i][0] * SIZES[i][1] - cfg.width * cfg.height))


def _slider_values(cfg):
    """Where each slider sits for a config, in slider order - `wanted` backwards."""
    return (cfg.n_plates, _size_index(cfg), round(cfg.land_fraction * 100),
            round(cfg.margin_h * 100), round(RIVER_SCALE / cfg.river_threshold),
            round(LAKE_SCALE / cfg.lake_min_depth), round(cfg.temp_offset))


# One menu line. `key` is only the hint drawn on the right - the keys
# themselves are handled in `Viewer.run`. No action makes it a line to read
# (the Help menu); `checked` draws a dot beside it while it returns true.
Item = namedtuple("Item", "label key action checked", defaults=(None, None))

LAYER_KEYS = "1234567890-"   # the first eleven layers; the rest are [ ] away


class Slider:
    """One integer slider. Regeneration is seconds long, so the caller applies
    the value on release, not while dragging."""

    TRACK_W, ROW_H, LABEL_PAD = 150, 26, 10

    def __init__(self, label, lo, hi, value, step=1, fmt=str):
        self.label, self.lo, self.hi, self.step, self.fmt = label, lo, hi, step, fmt
        self.value = value
        self.rect = pygame.Rect(0, 0, self.TRACK_W, 14)
        self.grabbed = False
        self._gutter = None

    def gutter(self, font):
        """How much room this slider's label needs, at its *widest* value.

        Measured over every value it can take, not the current one: sized to
        the value under the cursor, the track would shift left and right as the
        knob is dragged. Measured once and kept - the label set never changes.
        """
        if self._gutter is None:
            self._gutter = self.LABEL_PAD + max(
                font.size(f"{self.label} {self.fmt(v)}")[0]
                for v in range(self.lo, self.hi + 1, self.step))
        return self._gutter

    def layout(self, x, y, gutter):
        self.rect.topleft = (x + gutter, y)

    def hit(self, pos):
        return self.rect.inflate(16, 14).collidepoint(pos)

    def set_from(self, mx):
        t = np.clip((mx - self.rect.x) / max(1, self.rect.w), 0, 1)
        v = self.lo + (self.hi - self.lo) * t
        self.value = int(np.clip(round(v / self.step) * self.step, self.lo, self.hi))

    def draw(self, screen, font):
        t = (self.value - self.lo) / max(1e-9, self.hi - self.lo)
        cy = self.rect.centery
        knob = int(self.rect.x + t * self.rect.w)
        pygame.draw.line(screen, (70, 74, 84), (self.rect.x, cy),
                         (self.rect.right, cy), 3)
        pygame.draw.line(screen, (150, 190, 230), (self.rect.x, cy), (knob, cy), 3)
        pygame.draw.circle(screen, (235, 240, 250) if self.grabbed else (190, 205, 225),
                           (knob, cy), 6)
        img = font.render(f"{self.label} {self.fmt(self.value)}", True, (215, 220, 230))
        screen.blit(img, (self.rect.x - self.gutter(font), cy - img.get_height() // 2))


class Viewer:
    def __init__(self, cfg, window):
        pygame.init()
        pygame.display.set_caption("terrain generator")
        self.screen = pygame.display.set_mode(window, pygame.RESIZABLE)
        self.font = pygame.font.SysFont("consolas,couriernew,monospace", 15)
        self.big = pygame.font.SysFont("consolas,couriernew,monospace", 19, bold=True)
        self.cfg = cfg
        self.layer = 0
        self.arrows = False
        self.open = None       # title of the open menu
        self.titles = []       # (rect, title), laid out by draw
        self.rows = []         # (rect, row) of the open menu, laid out by draw
        self.panel = pygame.Rect(0, 0, 0, 0)
        self.rng = np.random.default_rng()
        self.busy = ""
        self.world = None
        self.surfaces = {}
        self.dir = None
        self.zoom = 1.0
        self.cam = [0.0, 0.0]  # top-left of the view, in map cells
        plates, size, land, rough, rivers, lakes, temp = _slider_values(cfg)
        self.sliders = [
            Slider("plates", 4, 48, plates),
            Slider("map size", 0, len(SIZES) - 1, size,
                   fmt=lambda i: "{}x{}".format(*SIZES[i])),
            Slider("land/sea", 0, 100, land, fmt=lambda v: f"{v}/{100 - v}"),
            # Sliders are integer, so this one carries hundredths.
            Slider("shelf coast roughness", 0, 40, rough,
                   fmt=lambda v: f"{v / 100:.2f}"),
            Slider("rivers", 1, 30, rivers),
            Slider("lakes", 1, 30, lakes),
            Slider("temperature", -20, 20, temp, fmt=lambda v: f"{v:+d}C"),
        ]
        self.menus = {
            "File": [
                Item("Import world...", "I", self.import_config),
                None,
                Item("Save layer picture", "P", self.save),
                Item("Export for engine", "E", self.export),
                None,
                Item("Quit", "Esc",
                     lambda: pygame.event.post(pygame.event.Event(pygame.QUIT))),
            ],
            "Edit": [
                Item("New world", "R", self.new_world),
                Item("Rebuild, same seed", "T", lambda: self.regenerate(self.cfg.seed)),
                None,
                *self.sliders,
            ],
            "View": [
                Item("Plate motion arrows", "G",
                     lambda: setattr(self, "arrows", not self.arrows), lambda: self.arrows),
                None,
                Item("Zoom in", "+", lambda: self.zoom_center(1.25)),
                Item("Zoom out", "Num -", lambda: self.zoom_center(0.8)),
                Item("Fit to window", "Z", self.fit),
            ],
            "Layer": [
                Item(name, LAYER_KEYS[i] if i < len(LAYER_KEYS) else "",
                     lambda i=i: setattr(self, "layer", i), lambda i=i: self.layer == i)
                for i, (name, _) in enumerate(render.LAYERS)
            ],
            "Help": [
                Item("Pan", "drag / WASD / arrows"),
                Item("Zoom", "wheel"),
                Item("Next / previous layer", "] ["),
            ],
        }
        self.regenerate(cfg.seed)

    # ---- generation -------------------------------------------------
    def regenerate(self, seed):
        self.cfg.seed = seed

        def progress(stage):
            self.busy = (f"seed {seed}  {self.cfg.width}x{self.cfg.height}"
                         f"  -  {stage}...")
            self.draw()
            pygame.display.flip()
            pygame.event.pump()   # keep the window alive through a long build

        progress("starting")
        self.world = generate(self.cfg, on_stage=progress)
        self.surfaces = {}
        self.dir = None      # a new world is a new folder, not an overwrite
        self.busy = ""
        self.fit()

    def new_world(self):
        self.regenerate(int(self.rng.integers(0, 10 ** 6)))

    def surface(self, i):
        """Layers are rendered on demand and cached; some are not cheap."""
        if i not in self.surfaces:
            name, img = render.layer(self.world, i)
            surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
            self.surfaces[i] = (name, surf)
        return self.surfaces[i]

    # ---- camera -----------------------------------------------------
    def fit(self):
        sw, sh = self.screen.get_size()
        mh, mw = self.world.height.shape
        # The map sits between the menu bar and the info bar, 28px each.
        self.zoom = min(sw / mw, (sh - 56) / mh)
        self.cam = [(mw - sw / self.zoom) / 2, (mh - (sh - 56) / self.zoom) / 2]

    def clamp(self):
        sw, sh = self.screen.get_size()
        mh, mw = self.world.height.shape
        vw, vh = sw / self.zoom, (sh - 56) / self.zoom
        self.cam[0] %= mw                      # x wraps with the world
        self.cam[1] = min(max(self.cam[1], min(0, mh - vh)), max(0, mh - vh))

    def zoom_at(self, factor, pos):
        mx, my = self.screen_to_map(pos)
        self.zoom = float(np.clip(self.zoom * factor, 0.15, 24.0))
        sx, sy = pos
        self.cam[0] = mx - sx / self.zoom
        self.cam[1] = my - (sy - 28) / self.zoom
        self.clamp()

    def zoom_center(self, factor):
        self.zoom_at(factor, (self.screen.get_width() // 2, self.screen.get_height() // 2))

    def screen_to_map(self, pos):
        return self.cam[0] + pos[0] / self.zoom, self.cam[1] + (pos[1] - 28) / self.zoom

    # ---- drawing ----------------------------------------------------
    def blit_map(self, surf):
        """Draw the visible window of the map, repeating across the x seam."""
        sw, sh = self.screen.get_size()
        mh, mw = self.world.height.shape
        z = self.zoom
        view = pygame.Rect(0, 28, sw, sh - 28)
        y0 = int(np.floor(self.cam[1]))
        rows = int(np.ceil((sh - 28) / z)) + 2
        y0 = max(0, min(y0, mh - 1))
        rows = min(rows, mh - y0)
        x = -((self.cam[0] % mw) - int(self.cam[0] % mw)) * z
        col = int(self.cam[0] % mw)
        dy = (y0 - self.cam[1]) * z + 28
        self.screen.set_clip(view)
        while x < sw:
            cols = min(mw - col, int(np.ceil((sw - x) / z)) + 2)
            src = surf.subsurface(pygame.Rect(col, y0, cols, rows))
            # Round both edges, not the width, or the seam shows a 1px gap.
            x0, x1 = int(round(x)), int(round(x + cols * z))
            dst = pygame.transform.scale(src, (max(1, x1 - x0), max(1, int(round(rows * z)))))
            self.screen.blit(dst, (x0, int(dy)))
            x += cols * z
            col = 0
        self.screen.set_clip(None)

    def draw_arrows(self):
        t = self.world.tect
        for i, (sy, sx) in enumerate(t.seeds):
            p = ((sx - self.cam[0]) % self.world.height.shape[1]) * self.zoom
            q = (sy - self.cam[1]) * self.zoom + 28
            vy, vx = t.vel[i] * 34 * min(self.zoom, 2.0)
            a, b = (p, q), (p + vx, q + vy)
            col = (250, 250, 250) if t.plate_cont[i] else (255, 210, 120)
            pygame.draw.line(self.screen, (0, 0, 0), a, b, 5)
            pygame.draw.line(self.screen, col, a, b, 2)
            pygame.draw.circle(self.screen, col, (int(p), int(q)), 4)

    def draw_titles(self):
        """The menu names, left end of the top bar. Returns where they end."""
        x, mouse = 0, pygame.mouse.get_pos()
        self.titles = []
        for title in self.menus:
            rect = pygame.Rect(x, 0, self.font.size(title)[0] + 20, 28)
            if title == self.open or rect.collidepoint(mouse):
                pygame.draw.rect(self.screen, (50, 70, 100), rect)
            self.text(title, x + 10, 6)
            self.titles.append((rect, title))
            x = rect.right
        return x

    def draw_menu(self):
        """The open menu, laid out fresh each frame. Clicks are tested against
        the rects it leaves in `self.rows`, so what is hit is what was drawn."""
        self.rows = []
        if self.open is None:
            return
        rows = self.menus[self.open]
        # One gutter for all the sliders, so their tracks line up in a column.
        gutter = max((r.gutter(self.font) for r in rows if isinstance(r, Slider)),
                     default=0)

        def width(r):
            if isinstance(r, Slider):
                return gutter + Slider.TRACK_W
            if r is None:
                return 0
            return self.font.size(r.label)[0] + 32 + self.font.size(r.key or "")[0]

        w = 24 + max(map(width, rows)) + 16
        heights = [Slider.ROW_H if isinstance(r, Slider) else 22 if r else 9 for r in rows]
        x = next(rect.x for rect, t in self.titles if t == self.open)
        self.panel = pygame.Rect(x, 28, w, sum(heights) + 8)
        pygame.draw.rect(self.screen, (30, 32, 40), self.panel)
        pygame.draw.rect(self.screen, (70, 74, 84), self.panel, 1)
        y, mouse = 32, pygame.mouse.get_pos()
        for r, h in zip(rows, heights):
            rect = pygame.Rect(x + 1, y, w - 2, h)
            if isinstance(r, Slider):
                r.layout(x + 24, y + 6, gutter)
                r.draw(self.screen, self.font)
            elif r is None:
                pygame.draw.line(self.screen, (70, 74, 84), (x + 8, y + 4), (x + w - 8, y + 4))
            else:
                if r.action and rect.collidepoint(mouse):
                    pygame.draw.rect(self.screen, (50, 70, 100), rect)
                if r.checked and r.checked():
                    pygame.draw.circle(self.screen, (150, 190, 230), (x + 12, rect.centery), 4)
                self.text(r.label, x + 24, y + 3)
                if r.key:
                    self.text(r.key, x + w - 16 - self.font.size(r.key)[0], y + 3,
                              (150, 155, 170))
            self.rows.append((rect, r))
            y += h

    def row_at(self, pos):
        """The open menu's row under `pos`. A slider is only hit on its track:
        a click on its label would otherwise snap it to the bottom."""
        for rect, r in self.rows:
            if r.hit(pos) if isinstance(r, Slider) else r and rect.collidepoint(pos):
                return r

    def wanted(self):
        """The config the sliders are currently asking for.

        Returned as a mapping rather than unpacked positionally: comparing it
        against the live config is what decides whether a rebuild is needed,
        and one dict does that for any number of sliders. Every value here is
        derived from an integer by fixed arithmetic, so the equality test in
        `apply_sliders` is exact.
        """
        plates, size, land, rough, rivers, lakes, temp = (
            s.value for s in self.sliders)
        return {
            "n_plates": plates,
            "width": SIZES[size][0], "height": SIZES[size][1],
            "land_fraction": land / 100,
            "margin_h": rough / 100,
            "river_threshold": RIVER_SCALE / rivers,
            "lake_min_depth": LAKE_SCALE / lakes,
            "temp_offset": float(temp),
        }

    def apply_sliders(self):
        """Push slider values into the config; regenerate only if one changed."""
        want = self.wanted()
        if all(getattr(self.cfg, k) == v for k, v in want.items()):
            return
        for k, v in want.items():
            setattr(self.cfg, k, v)
        self.regenerate(self.cfg.seed)

    def text(self, s, x, y, col=(235, 235, 235), font=None):
        img = (font or self.font).render(s, True, col)
        self.screen.blit(img, (x, y))
        return img.get_width()

    def draw(self):
        self.screen.fill((16, 18, 22))
        if self.world is not None:
            name, surf = self.surface(self.layer)
            self.blit_map(surf)
            if self.arrows:
                self.draw_arrows()
        else:
            name = "-"
        pygame.draw.rect(self.screen, (26, 28, 34), (0, 0, self.screen.get_width(), 28))
        x = self.draw_titles() + 14
        x += self.text(f"[{self.layer + 1}] {name}", x, 4, (255, 226, 150), self.big) + 22
        x += self.text(f"seed {self.cfg.seed}  {self.cfg.width}x{self.cfg.height}"
                       f"  zoom {self.zoom:.2f}x", x, 6) + 22
        # What is under the mouse gets its own bar along the bottom.
        sh = self.screen.get_height()
        pygame.draw.rect(self.screen, (26, 28, 34), (0, sh - 28, self.screen.get_width(), 28))
        if self.world is not None:
            mx, my = self.screen_to_map(pygame.mouse.get_pos())
            mh, mw = self.world.height.shape
            if 0 <= my < mh:
                d = self.world.info_at(int(my), int(mx) % mw)
                extra = ""
                if d["lake_depth"] > 0:
                    extra = f"  lake {d['lake_depth']:.3f} deep"
                elif d["river_w"] > 0:
                    extra = f"  river {d['river_w']:.1f} wide"
                if d["elev"] > 0:
                    extra += f"  {d['biome']}  canopy {d['canopy']:.2f}"
                self.text(f"h {d['elev']:+.3f}  plate {d['plate']:>2} {d['crust'][:4]}"
                          f"  bnd {d['dist_to_boundary']:.0f}px  rain {d['rain']:.2f}"
                          f"  {d['temp']:+.0f}C  flow {d['flow']:.0f}{extra}",
                          12, sh - 22, (170, 210, 255))
        # Bottom left above that bar, where no menu drops down over it mid-rebuild.
        if self.busy:
            self.text(self.busy, 12, sh - 60, (255, 200, 120), self.big)
        self.draw_menu()

    # ---- main loop --------------------------------------------------
    def run(self):
        clock = pygame.time.Clock()
        drag = None
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    title = next((t for rect, t in self.titles if rect.collidepoint(e.pos)), None)
                    row = self.row_at(e.pos)
                    if title:
                        self.open = None if title == self.open else title
                    elif isinstance(row, Slider):
                        row.grabbed = True
                        row.set_from(e.pos[0])
                    elif row and row.action:
                        self.open = None
                        row.action()
                    elif self.open:
                        # A click off the menu closes it, and does only that.
                        if not self.panel.collidepoint(e.pos):
                            self.open = None
                    else:
                        drag = e.pos
                if e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    drag = None
                    if any(s.grabbed for s in self.sliders):
                        for s in self.sliders:
                            s.grabbed = False
                        self.apply_sliders()
                if e.type == pygame.MOUSEMOTION and any(s.grabbed for s in self.sliders):
                    next(s for s in self.sliders if s.grabbed).set_from(e.pos[0])
                elif e.type == pygame.MOUSEMOTION and drag:
                    self.cam[0] -= (e.pos[0] - drag[0]) / self.zoom
                    self.cam[1] -= (e.pos[1] - drag[1]) / self.zoom
                    drag = e.pos
                    self.clamp()
                elif e.type == pygame.MOUSEMOTION and self.open:
                    # With one menu open, sliding along the bar opens the next.
                    self.open = next((t for rect, t in self.titles
                                      if rect.collidepoint(e.pos)), self.open)
                if e.type == pygame.MOUSEWHEEL:
                    self.zoom_at(1.16 ** e.y, pygame.mouse.get_pos())
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE and self.open:
                        self.open = None
                    elif e.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    if pygame.K_1 <= e.key <= pygame.K_9:
                        self.layer = e.key - pygame.K_1
                    if e.key == pygame.K_0:
                        self.layer = 9
                    if e.key == pygame.K_MINUS:
                        self.layer = 10
                    self.layer %= len(render.LAYERS)
                    if e.key == pygame.K_RIGHTBRACKET:
                        self.layer = (self.layer + 1) % len(render.LAYERS)
                    if e.key == pygame.K_LEFTBRACKET:
                        self.layer = (self.layer - 1) % len(render.LAYERS)
                    if e.key == pygame.K_r:
                        self.new_world()
                    if e.key == pygame.K_t:
                        self.regenerate(self.cfg.seed)
                    if e.key == pygame.K_g:
                        self.arrows = not self.arrows
                    if e.key == pygame.K_z:
                        self.fit()
                    if e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.zoom_center(1.25)
                    if e.key == pygame.K_KP_MINUS:
                        self.zoom_center(0.8)
                    if e.key == pygame.K_p:
                        self.save()
                    if e.key == pygame.K_e:
                        self.export()
                    if e.key == pygame.K_i:
                        self.import_config()

            keys = pygame.key.get_pressed()
            step = 12 / self.zoom
            self.cam[0] += step * (keys[pygame.K_d] or keys[pygame.K_RIGHT])
            self.cam[0] -= step * (keys[pygame.K_a] or keys[pygame.K_LEFT])
            self.cam[1] += step * (keys[pygame.K_s] or keys[pygame.K_DOWN])
            self.cam[1] -= step * (keys[pygame.K_w] or keys[pygame.K_UP])
            self.clamp()

            self.draw()
            pygame.display.flip()
            clock.tick(60)

    # ---- getting a world out ----------------------------------------
    def folder(self):
        """Where this world's files go: one folder, made on the first save.

        Per world and not per keypress, so a layer picture and the engine data
        land together - looking at a map and handing it over are the same
        errand, and they were two folders apart. `regenerate` drops it, so the
        next world gets `seed7-2` rather than writing over what is already
        there.

        Made lazily. Pressing neither key should not litter `out/export/` with
        an empty directory per world looked at.
        """
        if self.dir is None:
            self.dir = export.new_dir("out/export", f"seed{self.cfg.seed}")
        return self.dir

    def save(self, i=None):
        """One layer as it is drawn, full map - pan and zoom are for looking."""
        name, surf = self.surface(self.layer if i is None else i)
        path = self.folder() / f"{name.replace(' ', '_').replace('/', '-')}.png"
        pygame.image.save(surf, str(path))
        self.busy = f"saved {path.as_posix()}"
        return path

    def export(self):
        """The engine data, and the main view beside it to say which world."""
        d = self.folder()
        export.bundle(self.world, d=d)
        # Colour, so `bundle` cannot write it: the PNG writer in there is
        # greyscale, and pygame is the viewer's dependency, not the generator's.
        self.save(0)
        self.busy = f"exported to {d.as_posix()}/"

    def import_config(self):
        """Rebuild a world from an export's `config.json`.

        The seed and every knob, not the pictures: the world is generated
        again, so one exported by other code may not come out the same, and
        `load_config` warns when the code differs.
        """
        # tkinter because it ships with Python and Windows draws the dialog.
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)   # or the dialog opens behind the map
        path = filedialog.askopenfilename(
            parent=root, title="Import world", initialdir="out/export",
            filetypes=[("World config", "config.json"), ("JSON", "*.json")])
        root.destroy()
        if not path:
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                cfg = export.load_config(path)
            except Exception as err:    # any file can be picked; say so, don't crash
                self.busy = f"could not import {path}: {err}"
                return
        self.cfg = cfg
        # Or the next slider release writes the old world's values back over it.
        for s, v in zip(self.sliders, _slider_values(cfg)):
            s.value = int(np.clip(v, s.lo, s.hi))
        self.regenerate(cfg.seed)
        self.busy = f"imported {Path(path).parent.name}" + (
            "  -  made by other code, may not match" if caught else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", default="640x448", help="map size, WxH")
    ap.add_argument("--windowed", default="1280x820", help="window size, WxH")
    ap.add_argument("--plates", type=int, default=None)
    a = ap.parse_args(argv)
    mw, mh = (int(v) for v in a.size.lower().split("x"))
    ww, wh = (int(v) for v in a.windowed.lower().split("x"))
    cfg = Config(width=mw, height=mh, seed=a.seed)
    if a.plates:
        cfg.n_plates = a.plates
    Viewer(cfg, (ww, wh)).run()
    pygame.quit()


if __name__ == "__main__":
    main()
