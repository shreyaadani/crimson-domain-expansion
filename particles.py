"""Upward-drifting particle system — draws on glow layer for bloom effect."""

import random
import cv2
import numpy as np

# Glow-layer colors (bright, bloom will spread them)
GLOW_BURGUNDY  = (80,  20,  200)
GLOW_VIOLET    = (255, 160, 220)

_COLORS = [GLOW_BURGUNDY, GLOW_VIOLET]

MAX_PARTICLES = 80
PARTICLE_LIFE = 45   # frames


class ParticleSystem:
    def __init__(self, max_particles: int = MAX_PARTICLES):
        self.max_particles = max_particles
        self.particles: list[dict] = []

    def spawn(self, x: float, y: float, count: int = 1):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            self.particles.append({
                'x':    float(x) + random.uniform(-15, 15),
                'y':    float(y) + random.uniform(-8, 8),
                'dx':   random.uniform(-1.8, 1.8),
                'dy':   random.uniform(-5.5, -2.0),
                'life': PARTICLE_LIFE,
                'color': random.choice(_COLORS),
                'size':  random.randint(2, 5),
            })

    def update(self):
        alive = []
        for p in self.particles:
            p['x'] += p['dx']
            p['y'] += p['dy']
            # Slight horizontal drift (organic feel)
            p['dx'] += random.uniform(-0.1, 0.1)
            p['life'] -= 1
            if p['life'] > 0:
                alive.append(p)
        self.particles = alive

    def draw(self, glow, alpha: float = 1.0):
        """Draw particles on the glow layer (bloom handles the glow spread)."""
        if alpha <= 0 or not self.particles:
            return
        h, w = glow.shape[:2]
        for p in self.particles:
            x, y = int(p['x']), int(p['y'])
            if not (0 <= x < w and 0 <= y < h):
                continue
            # Fade brightness with lifetime
            brightness = (p['life'] / PARTICLE_LIFE) * alpha
            color = tuple(int(c * brightness) for c in p['color'])
            cv2.circle(glow, (x, y), p['size'], color, -1)

    def clear(self):
        self.particles.clear()
