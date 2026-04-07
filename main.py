"""
Dragon Domain Expansion — main entry point.
Optimized glow-layer pipeline. Process at 640x480 for speed.

Run:  python main.py
Quit: press Q
"""

import math
import time

import cv2
import numpy as np

from state_machine import DomainStateMachine
from gesture       import GestureDetector
from particles     import ParticleSystem
from ribbons       import RibbonSystem
from effects       import (
    bloom_pass,
    apply_vignette,
    apply_color_grade,
    apply_screen_shake,
    apply_film_grain,
    draw_void_circle,
    draw_torii_gates,
    draw_dragon_shards,
    draw_arc_barrier,
    draw_rune_ring,
    draw_ember_orbs,
    draw_lightning,
    draw_screen_flash,
)

# ---------------------------------------------------------------------------
PROC_W, PROC_H   = 640, 480   # processing resolution (fast)
VOID_MAX_RADIUS   = 200        # scaled for 640x480
EMBER_SPEED       = 0.35
RUNE_SPEED        = 0.55
SHARD_PULSE_FREQ  = 28
SPAWN_RATE        = 2
SHAKE_FRAMES      = 6


def _pos(g, h, w):
    fb = {
        'left_shoulder':  (int(w * 0.35), int(h * 0.35)),
        'right_shoulder': (int(w * 0.65), int(h * 0.35)),
        'torso_center':   (int(w * 0.50), int(h * 0.55)),
        'chest_center':   (int(w * 0.50), int(h * 0.42)),
        'wrist':          None,
    }
    return {k: g.get(k) or fb[k] for k in fb}


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Request 640x480 from camera directly if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PROC_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROC_H)

    sm        = DomainStateMachine()
    detector  = GestureDetector()
    particles = ParticleSystem()
    ribbons   = RibbonSystem(num_ribbons=5)

    ember_angle  = 0.0
    rune_angle   = 0.0
    frame_count  = 0
    shake_frames = 0

    fps_t, fps_n, fps_d = time.time(), 0, 0.0

    print("Domain Expansion ready -- press Q to quit")
    print("  Raise hand  -> CHARGING")
    print("  Fist clench -> ACTIVATE")
    print("  Open palm   -> DEACTIVATE")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure processing size
        fh, fw = frame.shape[:2]
        if fw != PROC_W or fh != PROC_H:
            frame = cv2.resize(frame, (PROC_W, PROC_H))

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- Detection ------------------------------------------------
        gestures = detector.detect(frame)
        prev_state = sm.state
        sm.update(gestures)
        pos = _pos(gestures, h, w)

        state = sm.state
        cp    = sm.charge_progress
        alpha = sm.effect_alpha

        torso = pos['torso_center']
        chest = pos['chest_center']
        ls    = pos['left_shoulder']
        rs    = pos['right_shoulder']
        wrist = pos['wrist']

        if state == sm.ACTIVE and prev_state == sm.CHARGING:
            shake_frames = SHAKE_FRAMES

        barrier_axes = (int(w * 0.23), int(h * 0.42))

        # --- Animation ticks -----------------------------------------
        if state != sm.IDLE:
            ribbons.update()
            particles.update()
            particles.spawn(ls[0], ls[1], SPAWN_RATE)
            particles.spawn(rs[0], rs[1], SPAWN_RATE)

        if state in (sm.ACTIVE, sm.DEACTIVATING):
            ember_angle += EMBER_SPEED
            rune_angle  += RUNE_SPEED

        shard_pulse   = 1.0 + 0.04 * math.sin(frame_count / SHARD_PULSE_FREQ * 2 * math.pi)
        barrier_pulse = 0.85 + 0.15 * math.sin(frame_count / 42 * 2 * math.pi)

        # =============================================================
        # RENDER
        # =============================================================
        glow = np.zeros((h, w, 3), dtype=np.uint8)

        # -- Frame-layer effects --
        void_r = VOID_MAX_RADIUS * (cp if state == sm.CHARGING else
                 (1.0 if state in (sm.ACTIVE, sm.DEACTIVATING) else 0))
        frame = draw_void_circle(frame, torso, void_r, alpha)

        if state in (sm.ACTIVE, sm.DEACTIVATING):
            frame = apply_color_grade(frame, alpha)
        elif state == sm.CHARGING and cp > 0.3:
            frame = apply_color_grade(frame, (cp - 0.3) * 0.5)

        if state in (sm.ACTIVE, sm.DEACTIVATING):
            frame = draw_torii_gates(frame, alpha)

        if state in (sm.ACTIVE, sm.DEACTIVATING):
            frame = draw_dragon_shards(frame, glow, ls, rs, alpha, shard_pulse)

        # -- Glow-layer effects --
        if state != sm.IDLE:
            n_rib = max(1, int(cp * 5)) if state == sm.CHARGING else 5
            ribbons.draw(glow, alpha, n_rib)

        particles.draw(glow, alpha)

        if state != sm.IDLE:
            arc_p = cp if state == sm.CHARGING else 1.0
            draw_arc_barrier(glow, torso, barrier_axes, arc_p,
                             barrier_pulse if state == sm.ACTIVE else 1.0, alpha)

        if state != sm.IDLE:
            rune_p = cp if state == sm.CHARGING else 1.0
            draw_rune_ring(glow, chest, rune_p, rune_angle, alpha)

        if state in (sm.ACTIVE, sm.DEACTIVATING):
            draw_ember_orbs(glow, torso, ember_angle, alpha)

        if state in (sm.ACTIVE, sm.DEACTIVATING) and wrist:
            draw_lightning(glow, wrist, alpha)

        # -- Bloom + composite --
        bloomed = bloom_pass(glow)
        frame = cv2.add(frame, bloomed)

        # -- Post-processing --
        vig = 0.30 if state == sm.IDLE else 0.55 * alpha + 0.30
        frame = apply_vignette(frame, vig)

        if state in (sm.ACTIVE, sm.DEACTIVATING):
            frame = apply_film_grain(frame, 0.05 * alpha)

        frame = draw_screen_flash(frame, sm.flash_frames)
        if sm.flash_frames > 0:
            sm.flash_frames -= 1

        if shake_frames > 0:
            frame = apply_screen_shake(frame, shake_frames)
            shake_frames -= 1

        # -- HUD --
        _draw_hud(frame, state, cp)
        fps_n += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps_d = fps_n / (now - fps_t)
            fps_n, fps_t = 0, now
        cv2.putText(frame, f"{fps_d:.0f} fps", (w - 100, h - 12),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (80, 80, 80), 1)

        cv2.imshow('Domain Expansion', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


def _draw_hud(frame, state, cp):
    h, w = frame.shape[:2]
    colors = {
        'IDLE':         (60, 60, 60),
        'CHARGING':     (107, 0, 32),
        'ACTIVE':       (204, 245, 255),
        'DEACTIVATING': (130, 0, 75),
    }
    cv2.putText(frame, state, (12, h - 12),
                cv2.FONT_HERSHEY_PLAIN, 1.2, colors.get(state, (60, 60, 60)), 1)
    if state == 'CHARGING':
        bw = int((w - 24) * cp)
        cv2.rectangle(frame, (12, h - 30), (12 + bw, h - 24), (32, 0, 107), -1)
        cv2.rectangle(frame, (12, h - 30), (w - 12, h - 24), (60, 60, 60), 1)


if __name__ == '__main__':
    main()
