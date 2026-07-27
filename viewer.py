"""Pygame viewer: pan, zoom, layer switching, regeneration.

    python viewer.py [--seed N] [--size WxH] [--windowed WxH]

Controls are listed on screen (F1).
"""
import argparse

import numpy as np
import pygame

from terrain import Config, generate, render


def _map_height(width):
    """Maps keep a 4:3 aspect, so one slider is enough for world size."""
    return int(width * 3 / 4) // 2 * 2


HELP = [
    "left drag / arrows / WASD   pan",
    "sliders (top right)         plates, world size, margin fray",
    "wheel / + -                 zoom      (Z resets)",
    "1..9, 0, -                  layer     ([ ] cycles all)",
    "R                           regenerate, new seed",
    "T                           regenerate, same seed",
    "G                           toggle plate motion arrows",
    "P                           save the current layer to out/",
    "F1                          this help",
    "ESC                         quit",
]


class Slider:
    """One integer slider. Regeneration is seconds long, so the caller applies
    the value on release, not while dragging."""

    LABEL_W, TRACK_W, ROW_H = 152, 150, 26

    def __init__(self, label, lo, hi, value, step=1, fmt=str):
        self.label, self.lo, self.hi, self.step, self.fmt = label, lo, hi, step, fmt
        self.value = value
        self.rect = pygame.Rect(0, 0, self.TRACK_W, 14)
        self.grabbed = False

    def layout(self, x, y):
        self.rect.topleft = (x + self.LABEL_W, y)

    @property
    def width(self):
        return self.LABEL_W + self.TRACK_W

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
        screen.blit(img, (self.rect.x - self.LABEL_W, cy - img.get_height() // 2))


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
        self.show_help = True
        self.busy = ""
        self.world = None
        self.surfaces = {}
        self.zoom = 1.0
        self.cam = [0.0, 0.0]  # top-left of the view, in map cells
        self.sliders = [
            Slider("plates", 4, 48, cfg.n_plates),
            Slider("map size", 192, 1024, cfg.width, step=64,
                   fmt=lambda v: f"{v}x{_map_height(v)}"),
            # Sliders are integer, so this one carries hundredths.
            Slider("margin fray", 0, 40, round(cfg.margin_h * 100),
                   fmt=lambda v: f"{v / 100:.2f}"),
        ]
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
        self.busy = ""
        self.fit()

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
        self.zoom = min(sw / mw, (sh - 28) / mh)
        self.cam = [(mw - sw / self.zoom) / 2, (mh - (sh - 28) / self.zoom) / 2]

    def clamp(self):
        sw, sh = self.screen.get_size()
        mh, mw = self.world.height.shape
        vw, vh = sw / self.zoom, (sh - 28) / self.zoom
        self.cam[0] %= mw                      # x wraps with the world
        self.cam[1] = min(max(self.cam[1], min(0, mh - vh)), max(0, mh - vh))

    def zoom_at(self, factor, pos):
        mx, my = self.screen_to_map(pos)
        self.zoom = float(np.clip(self.zoom * factor, 0.15, 24.0))
        sx, sy = pos
        self.cam[0] = mx - sx / self.zoom
        self.cam[1] = my - (sy - 28) / self.zoom
        self.clamp()

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

    def draw_sliders(self):
        w = max(s.width for s in self.sliders) + 24
        h = Slider.ROW_H * len(self.sliders) + 14
        x0, y0 = self.screen.get_width() - w - 10, 38
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 165))
        self.screen.blit(panel, (x0, y0))
        for i, s in enumerate(self.sliders):
            s.layout(x0 + 12, y0 + 12 + i * Slider.ROW_H)
            s.draw(self.screen, self.font)

    def apply_sliders(self):
        """Push slider values into the config; regenerate only if one changed."""
        plates, size, fray = (s.value for s in self.sliders)
        if (plates == self.cfg.n_plates and size == self.cfg.width
                and fray == round(self.cfg.margin_h * 100)):
            return
        self.cfg.n_plates = plates
        self.cfg.width, self.cfg.height = size, _map_height(size)
        self.cfg.margin_h = fray / 100
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
        x = 8
        x += self.text(f"[{self.layer + 1}] {name}", x, 4, (255, 226, 150), self.big) + 22
        x += self.text(f"seed {self.cfg.seed}  {self.cfg.width}x{self.cfg.height}"
                       f"  zoom {self.zoom:.2f}x", x, 6) + 22
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
                self.text(f"h {d['elev']:+.3f}  plate {d['plate']:>2} {d['crust'][:4]}"
                          f"  bnd {d['dist_to_boundary']:.0f}px  flow {d['flow']:.0f}{extra}",
                          x, 6, (170, 210, 255))
        self.draw_sliders()
        if self.busy:
            self.text(self.busy, 12, 40, (255, 200, 120), self.big)
        if self.show_help:
            y = self.screen.get_height() - 20 * len(HELP) - 12
            panel = pygame.Surface((430, 20 * len(HELP) + 8), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 165))
            self.screen.blit(panel, (8, y - 4))
            for i, line in enumerate(HELP):
                self.text(line, 16, y + 20 * i, (215, 215, 215))

    # ---- main loop --------------------------------------------------
    def run(self):
        clock = pygame.time.Clock()
        drag = None
        rng = np.random.default_rng()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    grabbed = next((s for s in self.sliders if s.hit(e.pos)), None)
                    if grabbed:
                        grabbed.grabbed = True
                        grabbed.set_from(e.pos[0])
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
                if e.type == pygame.MOUSEWHEEL:
                    self.zoom_at(1.16 ** e.y, pygame.mouse.get_pos())
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
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
                        self.regenerate(int(rng.integers(0, 10 ** 6)))
                    if e.key == pygame.K_t:
                        self.regenerate(self.cfg.seed)
                    if e.key == pygame.K_g:
                        self.arrows = not self.arrows
                    if e.key == pygame.K_F1:
                        self.show_help = not self.show_help
                    if e.key == pygame.K_z:
                        self.fit()
                    if e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.zoom_at(1.25, (self.screen.get_width() // 2,
                                            self.screen.get_height() // 2))
                    if e.key in (pygame.K_KP_MINUS,):
                        self.zoom_at(0.8, (self.screen.get_width() // 2,
                                           self.screen.get_height() // 2))
                    if e.key == pygame.K_p:
                        self.save()

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

    def save(self):
        import os
        os.makedirs("out", exist_ok=True)
        name, surf = self.surface(self.layer)
        path = f"out/seed{self.cfg.seed}_{name.replace(' ', '_').replace('/', '-')}.png"
        pygame.image.save(surf, path)
        self.busy = f"saved {path}"


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
