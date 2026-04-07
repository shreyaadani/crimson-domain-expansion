# Crimson Domain Expansion

Real-time anime-style **Domain Expansion** effect powered by your webcam. Inspired by JJK, Fairy Tail, Frieren, and Genshin Impact — themed around a **Lightning Dragon God** aesthetic.

Raise your hand to charge, clench your fist to activate, and open your palm to release.

## Demo

| State | Gesture | What Happens |
|-------|---------|--------------|
| **IDLE** | — | Clean camera feed, subtle vignette |
| **CHARGING** | Raise hand (hold 0.5s) | Energy ribbons, particles, arc barrier and rune ring build up |
| **ACTIVE** | Clench fist | Full domain — void circle, torii gates, dragon shards, ember orbs, lightning bolts, screen flash + shake |
| **DEACTIVATING** | Open palm | Effects fade out over 1.5s |

## Visual Effects

- Void circle darkening around torso
- Burgundy/violet color grading
- Flowing energy ribbons (glow layer + bloom)
- Upward-drifting particles from shoulders
- Elliptical arc barrier with pulsing halo
- Double counter-rotating rune ring with symbols
- Crystalline dragon shards between shoulders
- Torii gate silhouettes
- Ember orbs orbiting torso
- Jagged lightning bolts from wrist
- Screen flash, screen shake, film grain, vignette

## Setup

**Requirements:** Python 3.9–3.12 (MediaPipe does not support 3.13+)

```bash
# Create virtual environment (use py -3.11 on Windows if default Python is 3.13+)
python -m venv venv

# Activate
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Press **Q** to quit.

## Project Structure

```
main.py            — Main loop: capture, detect, render, display
state_machine.py   — IDLE -> CHARGING -> ACTIVE -> DEACTIVATING state machine
gesture.py         — MediaPipe Hands + Pose gesture detection
effects.py         — All visual effects (bloom, vignette, lightning, shards, etc.)
particles.py       — Upward-drifting particle system
ribbons.py         — Flowing sine-wave energy ribbons
requirements.txt   — Python dependencies
```

## Performance

Processes at **640x480** for speed. Bloom pass uses downsampled blurs. Vignette, void masks, and color LUTs are cached. Targets 20+ FPS on integrated graphics.

## Color Palette

| Role | Color (BGR) |
|------|-------------|
| Burgundy core | `(32, 0, 107)` |
| Dark violet | `(90, 0, 50)` |
| Pale violet glow | `(255, 200, 240)` |
| White-gold flash | `(204, 245, 255)` |

## Tech Stack

- **OpenCV** — webcam capture, rendering, compositing
- **MediaPipe** — hand and pose landmark detection
- **NumPy** — array operations, mask generation
