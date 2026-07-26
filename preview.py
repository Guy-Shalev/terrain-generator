"""Render every layer to out/ as PNG. Run: python preview.py [seed] [--crop]"""
import os
import sys

import numpy as np
import pygame

from terrain import Config, generate, render


def save(img, path):
    pygame.image.save(pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2))), path)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    world = generate(Config(seed=seed))
    os.makedirs("out", exist_ok=True)
    for i, _ in enumerate(render.LAYERS):
        name, img = render.layer(world, i)
        save(img, f"out/{i:02d}_{name.replace(' ', '_').replace('/', '-')}.png")
        if "--crop" in sys.argv:  # 3x zoom on the map centre, to judge detail
            h, w = world.height.shape
            c = img[h // 3:h // 3 + h // 3, w // 3:w // 3 + w // 3]
            c = np.repeat(np.repeat(c, 3, 0), 3, 1)
            save(c, f"out/crop_{i:02d}.png")
    print(f"wrote {len(render.LAYERS)} layers to out/")


if __name__ == "__main__":
    main()
