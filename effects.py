"""
Visual effects with glow-layer rendering — performance-optimized.
Key optimizations: downsampled bloom, cached masks, LUT color grading.
"""

import cv2
import math
import random
import numpy as np

# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------
VOID        = (8,   0,   8)
BURGUNDY    = (32,  0,   107)
DEEP_RED    = (0,   0,   139)
VIOLET      = (130, 0,   75)
PALE_VIOLET = (220, 160, 201)
WHITE_GOLD  = (204, 245, 255)
TORII_BLACK = (12,  4,   18)

# Glow-layer colors (bright — bloom spreads them)
GLOW_RED        = (40,  30,  255)
GLOW_GOLD       = (220, 255, 255)
GLOW_VIOLET     = (255, 180, 240)
GLOW_PALE       = (255, 200, 240)
GLOW_EMBER_RED  = (50,  50,  255)
GLOW_EMBER_GOLD = (180, 240, 255)


# ===================================================================
# CACHED MASKS — computed once per resolution, reused every frame
# ===================================================================
_cache = {}


def _get_vignette_mask(h, w):
    key = ('vig', h, w)
    if key not in _cache:
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2)
        _cache[key] = np.clip((dist - 0.3) / 0.7, 0, 1).astype(np.float32)[:, :, np.newaxis]
    return _cache[key]


def _get_void_base_mask(h, w, cx, cy, r):
    """Cached void mask — only recomputes when center/radius change significantly."""
    # Quantize to avoid recomputing every pixel shift
    qcx, qcy, qr = cx // 8 * 8, cy // 8 * 8, r // 8 * 8
    key = ('void', h, w, qcx, qcy, qr)
    if key not in _cache:
        mask = np.zeros((h, w), dtype=np.float32)
        if qr > 0:
            cv2.circle(mask, (qcx, qcy), qr, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max(qr * 0.18, 1))
        _cache[key] = mask[:, :, np.newaxis]
    return _cache[key]


def _build_color_lut(alpha):
    """Pre-build per-channel LUTs for color grading (O(256) instead of O(pixels))."""
    q = round(alpha * 10) / 10  # quantize to avoid rebuilding every frame
    key = ('lut', q)
    if key not in _cache:
        x = np.arange(256, dtype=np.float32)
        b_lut = np.clip(x + 25 * q, 0, 255).astype(np.uint8)
        g_lut = np.clip(x - 45 * q, 0, 255).astype(np.uint8)
        r_lut = np.clip(x + 55 * q, 0, 255).astype(np.uint8)
        # Contrast boost baked in
        for lut in (b_lut, g_lut, r_lut):
            f = lut.astype(np.float32)
            f = np.clip((f - 128) * (1.0 + 0.15 * q) + 128, 0, 255)
            lut[:] = f.astype(np.uint8)
        _cache[key] = (b_lut, g_lut, r_lut)
    return _cache[key]


# Pre-generated noise tile (reused with random offset each frame)
_NOISE_TILE = None
_NOISE_SIZE = (256, 256)


def _get_noise_tile():
    global _NOISE_TILE
    if _NOISE_TILE is None:
        _NOISE_TILE = np.random.randint(0, 20, (*_NOISE_SIZE, 3), dtype=np.uint8)
    return _NOISE_TILE


# ===================================================================
# POST-PROCESSING
# ===================================================================

def bloom_pass(glow_layer):
    """Efficient bloom: downsample 4x -> blur -> upsample + sharp glow."""
    h, w = glow_layer.shape[:2]
    sh, sw = h // 4, w // 4

    # Downsampled wide bloom (cheap!)
    small = cv2.resize(glow_layer, (sw, sh), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), 10)
    wide = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)

    # Tight glow at half res
    sh2, sw2 = h // 2, w // 2
    med = cv2.resize(glow_layer, (sw2, sh2), interpolation=cv2.INTER_AREA)
    med_blur = cv2.GaussianBlur(med, (0, 0), 5)
    tight = cv2.resize(med_blur, (w, h), interpolation=cv2.INTER_LINEAR)

    # Combine: sharp original + tight glow + wide bloom
    result = cv2.add(glow_layer, tight)
    result = cv2.add(result, wide)
    return result


def apply_vignette(frame, intensity: float = 0.4):
    h, w = frame.shape[:2]
    mask = _get_vignette_mask(h, w)
    return (frame.astype(np.float32) * (1 - mask * intensity)).astype(np.uint8)


def apply_color_grade(frame, alpha: float = 1.0):
    """LUT-based color grading — O(256) instead of O(pixels)."""
    if alpha <= 0:
        return frame

    b_lut, g_lut, r_lut = _build_color_lut(alpha)
    b, g, r = cv2.split(frame)
    b = cv2.LUT(b, b_lut)
    g = cv2.LUT(g, g_lut)
    r = cv2.LUT(r, r_lut)
    graded = cv2.merge([b, g, r])

    # Chromatic aberration
    if alpha > 0.3:
        shift = max(2, int(4 * alpha))
        h, w = graded.shape[:2]
        b2, g2, r2 = cv2.split(graded)
        r_s = np.zeros_like(r2); r_s[:, shift:] = r2[:, :-shift]
        b_s = np.zeros_like(b2); b_s[:, :-shift] = b2[:, shift:]
        aberrated = cv2.merge([b_s, g2, r_s])
        blend = alpha * 0.5
        graded = cv2.addWeighted(graded, 1 - blend, aberrated, blend, 0)

    return graded


def apply_screen_shake(frame, shake_frames: int):
    if shake_frames <= 0:
        return frame
    mag = int(shake_frames * 2.5)
    dx, dy = random.randint(-mag, mag), random.randint(-mag, mag)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_REFLECT_101)


def apply_film_grain(frame, intensity: float = 0.06):
    """Tiled noise — no per-frame random generation."""
    if intensity <= 0:
        return frame
    h, w = frame.shape[:2]
    tile = _get_noise_tile()
    # Tile it across the frame with random offset
    ox, oy = random.randint(0, 127), random.randint(0, 127)
    noise = np.tile(tile, (h // _NOISE_SIZE[0] + 2, w // _NOISE_SIZE[1] + 2, 1))
    noise = noise[oy:oy+h, ox:ox+w]
    scaled = (noise.astype(np.float32) * intensity).astype(np.uint8)
    return cv2.add(frame, scaled)


# ===================================================================
# FRAME-LAYER EFFECTS
# ===================================================================

def draw_void_circle(frame, center, radius: float, alpha: float = 1.0):
    if alpha <= 0 or radius <= 0:
        return frame
    h, w = frame.shape[:2]
    mask = _get_void_base_mask(h, w, int(center[0]), int(center[1]), int(radius))
    a = alpha * 0.85
    scaled_mask = mask * a
    dark = frame.astype(np.float32) * 0.15
    void_tint = np.array(VOID, dtype=np.float32) * 0.5
    return (frame.astype(np.float32) * (1 - scaled_mask) + (dark + void_tint) * scaled_mask).astype(np.uint8)


def draw_torii_gates(frame, alpha: float = 1.0):
    if alpha <= 0:
        return frame
    h, w = frame.shape[:2]
    layer = frame.copy()

    lx = lambda f: int(w * f)
    ly = lambda f: int(h * f)
    left = np.array([
        [0, h], [lx(.04), h], [lx(.04), ly(.38)],
        [lx(.06), ly(.34)], [lx(.12), ly(.34)],
        [lx(.14), ly(.38)], [lx(.14), h], [lx(.18), h],
        [lx(.18), ly(.36)], [lx(.10), ly(.30)], [lx(.02), ly(.32)], [0, ly(.34)],
    ], np.int32)
    right = np.array([
        [w, h], [w-lx(.04), h], [w-lx(.04), ly(.38)],
        [w-lx(.06), ly(.34)], [w-lx(.12), ly(.34)],
        [w-lx(.14), ly(.38)], [w-lx(.14), h], [w-lx(.18), h],
        [w-lx(.18), ly(.36)], [w-lx(.10), ly(.30)], [w-lx(.02), ly(.32)], [w, ly(.34)],
    ], np.int32)

    cv2.fillPoly(layer, [left],  TORII_BLACK)
    cv2.fillPoly(layer, [right], TORII_BLACK)
    cv2.addWeighted(layer, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)
    return frame


def draw_dragon_shards(frame, glow, left_shoulder, right_shoulder,
                       alpha: float = 1.0, pulse: float = 1.0):
    if alpha <= 0:
        return frame

    ls = np.array(left_shoulder, dtype=np.float32)
    rs = np.array(right_shoulder, dtype=np.float32)
    p = pulse

    left_offsets = [
        [(0,0), (-45*p,-85*p), (-25*p,-130*p), (-70*p,-65*p)],
        [(0,0), (-65*p,-45*p), (-110*p,-85*p), (-55*p,-25*p)],
        [(0,0), (-25*p,-110*p), (-60*p,-155*p), (-90*p,-95*p)],
        [(-25*p,-35*p), (-70*p,-110*p), (-40*p,-145*p)],
        [(15*p,-25*p), (-50*p,-80*p), (-20*p,-125*p)],
        [(5*p,-15*p), (-35*p,-55*p), (-75*p,-70*p), (-30*p,-10*p)],
    ]
    right_offsets = [
        [(0,0), (45*p,-85*p), (25*p,-130*p), (70*p,-65*p)],
        [(0,0), (65*p,-45*p), (110*p,-85*p), (55*p,-25*p)],
        [(0,0), (25*p,-110*p), (60*p,-155*p), (90*p,-95*p)],
        [(25*p,-35*p), (70*p,-110*p), (40*p,-145*p)],
        [(-15*p,-25*p), (50*p,-80*p), (20*p,-125*p)],
        [(-5*p,-15*p), (35*p,-55*p), (75*p,-70*p), (30*p,-10*p)],
    ]
    fill_colors = [DEEP_RED, BURGUNDY, DEEP_RED, BURGUNDY, DEEP_RED, VIOLET]
    glow_colors = [GLOW_RED, GLOW_GOLD, GLOW_RED, GLOW_GOLD, GLOW_RED, GLOW_VIOLET]

    overlay = frame.copy()
    for i, (lo, ro) in enumerate(zip(left_offsets, right_offsets)):
        lpts = np.array([[ls[0]+o[0], ls[1]+o[1]] for o in lo], np.int32)
        rpts = np.array([[rs[0]+o[0], rs[1]+o[1]] for o in ro], np.int32)
        cv2.fillPoly(overlay, [lpts], fill_colors[i])
        cv2.fillPoly(overlay, [rpts], fill_colors[i])
        cv2.polylines(glow, [lpts], True, glow_colors[i], 2)
        cv2.polylines(glow, [rpts], True, glow_colors[i], 2)
        if len(lo) >= 4:
            mid_l = ((lpts[0] + lpts[2]) // 2).astype(np.int32)
            mid_r = ((rpts[0] + rpts[2]) // 2).astype(np.int32)
            cv2.line(glow, tuple(lpts[0]), tuple(mid_l), glow_colors[i], 1)
            cv2.line(glow, tuple(rpts[0]), tuple(mid_r), glow_colors[i], 1)

    cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
    return frame


# ===================================================================
# GLOW-LAYER EFFECTS
# ===================================================================

def draw_arc_barrier(glow, center, axes, progress, pulse_alpha=1.0, alpha=1.0):
    if alpha <= 0 or progress <= 0:
        return
    cx, cy = int(center[0]), int(center[1])
    ax, ay = int(axes[0]), int(axes[1])
    start = -90
    end = int(-90 + progress * 360)
    s = alpha * pulse_alpha

    outer_c = tuple(max(0, min(255, int(c * s * 0.4))) for c in GLOW_RED)
    cv2.ellipse(glow, (cx, cy), (ax+6, ay+6), 0, start, end, outer_c, 10)
    mid_c = tuple(max(0, min(255, int(c * s * 0.7))) for c in GLOW_RED)
    cv2.ellipse(glow, (cx, cy), (ax, ay), 0, start, end, mid_c, 4)
    core_c = tuple(max(0, min(255, int(c * s))) for c in GLOW_GOLD)
    cv2.ellipse(glow, (cx, cy), (ax, ay), 0, start, end, core_c, 2)


_RUNE_TYPES = ['triangle', 'cross', 'diamond', 'circle']

def draw_rune_ring(glow, center, progress, angle=0.0, alpha=1.0):
    if alpha <= 0 or progress <= 0:
        return
    cx, cy = int(center[0]), int(center[1])
    radius = 65
    color = tuple(max(0, min(255, int(c * alpha))) for c in GLOW_PALE)
    dim_c = tuple(max(0, min(255, int(c * alpha * 0.5))) for c in GLOW_VIOLET)

    for i in range(28):
        if (i + 0.5) / 28 > progress:
            break
        s = (i / 28) * 360 - 90 + angle
        e = ((i + 0.55) / 28) * 360 - 90 + angle
        cv2.ellipse(glow, (cx, cy), (radius, radius), 0, int(s), int(e), color, 2)
    inner_r = radius - 10
    for i in range(28):
        if (i + 0.5) / 28 > progress:
            break
        s = (i / 28) * 360 - 90 - angle * 0.7
        e = ((i + 0.4) / 28) * 360 - 90 - angle * 0.7
        cv2.ellipse(glow, (cx, cy), (inner_r, inner_r), 0, int(s), int(e), dim_c, 1)

    for i, rune in enumerate(_RUNE_TYPES):
        if (i + 1) / 4 > progress:
            break
        a = math.radians(i * 90 + angle)
        rx = int(cx + radius * math.cos(a))
        ry = int(cy + radius * math.sin(a))
        _draw_rune(glow, rx, ry, rune, 10, color)


def _draw_rune(img, cx, cy, rune, size, color):
    if rune == 'triangle':
        pts = np.array([[cx, cy-size], [cx-size, cy+size], [cx+size, cy+size]], np.int32)
        cv2.polylines(img, [pts], True, color, 2)
    elif rune == 'cross':
        cv2.line(img, (cx-size, cy), (cx+size, cy), color, 2)
        cv2.line(img, (cx, cy-size), (cx, cy+size), color, 2)
    elif rune == 'diamond':
        pts = np.array([[cx, cy-size], [cx+size, cy], [cx, cy+size], [cx-size, cy]], np.int32)
        cv2.polylines(img, [pts], True, color, 2)
    elif rune == 'circle':
        cv2.circle(img, (cx, cy), max(1, size//2), color, 2)


_ORB_OFFSETS = [i * (360 / 7) for i in range(7)]
_ORB_RADII   = [185, 165, 210, 175, 195, 160, 220]
_ORB_COLORS  = [GLOW_EMBER_RED, GLOW_EMBER_GOLD] * 4
_ORB_SIZES   = [7, 5, 6, 7, 5, 6, 6]

def draw_ember_orbs(glow, center, angle, alpha=1.0):
    if alpha <= 0:
        return
    cx, cy = int(center[0]), int(center[1])
    for i in range(7):
        a = math.radians(angle + _ORB_OFFSETS[i])
        ox = int(cx + _ORB_RADII[i] * math.cos(a))
        oy = int(cy + _ORB_RADII[i] * math.sin(a))
        r, c = _ORB_SIZES[i], _ORB_COLORS[i]
        s = alpha
        cv2.circle(glow, (ox, oy), r*3, tuple(max(0,min(255,int(v*s*0.35))) for v in c), -1)
        cv2.circle(glow, (ox, oy), r*2, tuple(max(0,min(255,int(v*s*0.6))) for v in c), -1)
        cv2.circle(glow, (ox, oy), r,   tuple(max(0,min(255,int(v*s))) for v in GLOW_GOLD), -1)


def draw_lightning(glow, wrist, alpha=1.0):
    if alpha <= 0 or wrist is None:
        return
    wx, wy = int(wrist[0]), int(wrist[1])

    for _ in range(random.randint(4, 6)):
        direction = random.uniform(0, 2 * math.pi)
        total_len = random.randint(90, 200)
        num_segs = random.randint(5, 7)
        seg_len = total_len / num_segs
        color = GLOW_VIOLET if random.random() > 0.35 else GLOW_GOLD
        thick = 2 if color == GLOW_VIOLET else 1

        x, y = float(wx), float(wy)
        pts = [(wx, wy)]
        for _ in range(num_segs):
            d = direction + math.radians(random.uniform(-40, 40))
            nx = x + seg_len * math.cos(d)
            ny = y + seg_len * math.sin(d)
            nx += random.uniform(-22, 22)
            ny += random.uniform(-22, 22)
            pts.append((int(nx), int(ny)))
            x, y = nx, ny

        bolt_c = tuple(max(0,min(255,int(c*alpha))) for c in color)
        cv2.polylines(glow, [np.array(pts, np.int32)], False, bolt_c, thick)

        # Branch fork
        if random.random() > 0.6 and len(pts) > 3:
            fp = pts[random.randint(1, len(pts)-2)]
            bp = [fp]
            bx, by = float(fp[0]), float(fp[1])
            for _ in range(3):
                bd = direction + math.radians(random.uniform(-60, 60))
                bx += seg_len * 0.6 * math.cos(bd)
                by += seg_len * 0.6 * math.sin(bd)
                bp.append((int(bx), int(by)))
            dim = tuple(max(0,min(255,int(c*alpha*0.5))) for c in color)
            cv2.polylines(glow, [np.array(bp, np.int32)], False, dim, 1)


_FLASH_ALPHAS = {4: 0.90, 3: 0.55, 2: 0.25, 1: 0.10}

def draw_screen_flash(frame, flash_frames):
    a = _FLASH_ALPHAS.get(flash_frames, 0.0)
    if a <= 0:
        return frame
    overlay = np.full_like(frame, WHITE_GOLD)
    cv2.addWeighted(overlay, a, frame, 1.0 - a, 0, frame)
    return frame
