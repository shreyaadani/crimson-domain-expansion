"""Flowing energy ribbon animation — draws on glow layer."""

import cv2
import numpy as np

GLOW_PALE   = (255, 200, 240)
GLOW_VIOLET = (220, 140, 200)
_RNG = np.random.default_rng(42)


class RibbonSystem:
    def __init__(self, num_ribbons: int = 5):
        self.num_ribbons = num_ribbons
        self.phase = 0.0
        self.ribbons = [
            {
                'amplitude':    float(_RNG.uniform(35, 90)),
                'frequency':    float(_RNG.uniform(0.4, 1.6)),
                'y_offset':     float(_RNG.uniform(0.15, 0.85)),
                'thickness':    int(_RNG.integers(1, 4)),
                'phase_offset': float(_RNG.uniform(0, 2 * np.pi)),
                'color':        GLOW_PALE if i % 2 == 0 else GLOW_VIOLET,
            }
            for i in range(num_ribbons)
        ]

    def update(self):
        self.phase += 0.06

    def draw(self, glow, alpha: float = 1.0, num_active: int = 5):
        """Draw ribbons on glow layer."""
        if alpha <= 0 or num_active == 0:
            return

        h, w = glow.shape[:2]
        num_pts = 80

        for ribbon in self.ribbons[:num_active]:
            xs = np.linspace(0, w - 1, num_pts, dtype=np.int32)
            center_y = int(h * ribbon['y_offset'])
            ys = (
                center_y
                + ribbon['amplitude']
                * np.sin(
                    ribbon['frequency'] * np.linspace(0, 2 * np.pi, num_pts)
                    + self.phase + ribbon['phase_offset']
                )
            ).astype(np.int32)
            ys = np.clip(ys, 0, h - 1)
            pts = np.column_stack([xs, ys])

            color = tuple(int(c * alpha * 0.6) for c in ribbon['color'])
            cv2.polylines(glow, [pts], False, color, ribbon['thickness'])
