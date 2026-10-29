"""Task 5 — the final smash's orb (docs/extra_features.md §1).

The bouncing orb part (draw_part()), the pop where a punch collected it
(draw_pop()), the four orb slots by each health bar (draw_slots()) and the
glow of a player whose orb is full (draw_body_glow()). The game logic
(interaction/game.py: OrbPart, GamePlayer.orb) only keeps state; every size
and timing here comes from the render: config.
"""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core.types import Motion
from ipcvgame.scene.paint import SHIFT, blob, body_mask, fixed

DARK = (27, 16, 17)         # renderer.py's dark backing (BGR), for outlines
WHITE = (235, 229, 210)     # renderer.py's off-white (BGR)


def _p(xy) -> tuple[int, int]:
    """A point (x, y) px as an OpenCV centre in fixed point (SHIFT bits)."""
    x, y = fixed(xy)
    return int(x), int(y)


def _mix(color, other, k: float) -> tuple[int, ...]:
    """color moved a fraction k towards other."""
    return tuple(int(round(c + k * (o - c))) for c, o in zip(color, other, strict=True))


def _halo(canvas: np.ndarray, x: float, y: float, radius: float, color, alpha: float) -> None:
    """A soft disc of `radius` px round (x, y): three discs, each larger and
    fainter, blended into a small region of the canvas only."""
    h, w = canvas.shape[:2]
    x0, y0 = max(int(x - radius) - 1, 0), max(int(y - radius) - 1, 0)
    x1, y1 = min(int(x + radius) + 2, w), min(int(y + radius) + 2, h)
    if x0 >= x1 or y0 >= y1 or alpha <= 0:
        return
    roi = canvas[y0:y1, x0:x1]
    for k, a in ((1.0, 0.35), (0.8, 0.35), (0.6, 0.3)):
        layer = roi.copy()
        cv2.circle(layer, _p((x - x0, y - y0)), int(k * radius * (1 << SHIFT)), color, -1,
                   cv2.LINE_AA, SHIFT)
        cv2.addWeighted(layer, alpha * a, roi, 1.0 - alpha * a, 0.0, dst=roi)


def draw_part(canvas: np.ndarray, x: float, y: float, r: float, age: float,
              colors: tuple[tuple[int, int, int], tuple[int, int, int]], cfg: dict) -> None:
    """The orb part at (x, y) px, radius r: a ball split into the two team
    colours by an S-curve (a yin-yang: two half discs and a half-size disc
    of each colour on the dividing line), so it belongs to no team yet,
    turning at orb_spin_hz; a pale halo orb_halo radii wide, pulsing at
    orb_pulse_hz so it catches the eye; a dark rim and a white shine at
    the top left, so it reads as a ball."""
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * cfg["orb_pulse_hz"] * age)
    _halo(canvas, x, y, cfg["orb_halo"] * r * (0.9 + 0.1 * pulse), WHITE, 0.5 + 0.5 * pulse)
    a = 2 * np.pi * cfg["orb_spin_hz"] * age
    axis = np.array([np.cos(a), np.sin(a)])
    c, ri = _p((x, y)), int(r * (1 << SHIFT))
    cv2.circle(canvas, c, ri, colors[0], -1, cv2.LINE_AA, SHIFT)
    cv2.ellipse(canvas, c, (ri, ri), float(np.degrees(a)), 0, 180, colors[1], -1, cv2.LINE_AA, SHIFT)
    for side, color in ((1, colors[0]), (-1, colors[1])):
        cv2.circle(canvas, _p(np.array([x, y]) + side * 0.5 * r * axis), ri // 2, color, -1,
                   cv2.LINE_AA, SHIFT)
    cv2.circle(canvas, c, ri, DARK, max(1, int(round(r / 8))), cv2.LINE_AA, SHIFT)
    shine = blob(12, 0.25 * r, [], center=(x - 0.35 * r, y - 0.4 * r))
    cv2.fillPoly(canvas, [fixed(shine)], (255, 255, 255), cv2.LINE_AA, SHIFT)


def draw_pop(canvas: np.ndarray, x: float, y: float, r: float, age: float, life: float,
             color: tuple[int, int, int], cfg: dict) -> None:
    """Where a punch collected a part: a ring in the collector's colour
    growing from the part's size to orb_pop_ring radii, and orb_pop_drops
    drops flying out evenly all round, both shrinking to nothing over life
    (s)."""
    k = age / life
    if not 0 <= k < 1:
        return
    fade = 1.0 - k
    ring = r * (1.0 + (cfg["orb_pop_ring"] - 1.0) * (1.0 - fade ** 2))   # fast, then slowing
    cv2.circle(canvas, _p((x, y)), int(ring * (1 << SHIFT)), color,
               max(1, int(round(0.4 * r * fade))), cv2.LINE_AA, SHIFT)
    n = int(cfg["orb_pop_drops"])
    for i in range(n):
        a = 2 * np.pi * i / n
        d = 0.9 * ring
        drop = blob(10, 0.3 * r * fade, [], center=(x + d * np.cos(a), y + d * np.sin(a)))
        cv2.fillPoly(canvas, [fixed(drop)], color, cv2.LINE_AA, SHIFT)


def draw_slots(canvas: np.ndarray, centers: list[tuple[float, float]], r: float, filled: int,
               color: tuple[int, int, int], glow: float) -> None:
    """The orb slots, one small ball per part: the first `filled` in the
    team colour with a white shine, the rest dark with an off-white
    outline. glow (0-1) brightens the filled ones (a full orb pulses)."""
    lit = _mix(color, (255, 255, 255), 0.45 * glow)
    for i, (cx, cy) in enumerate(centers):
        c, ri = _p((cx, cy)), max(1 << SHIFT, int(r * (1 << SHIFT)))
        if i < filled:
            cv2.circle(canvas, c, ri, lit, -1, cv2.LINE_AA, SHIFT)
            cv2.circle(canvas, _p((cx - 0.3 * r, cy - 0.3 * r)), ri // 3, (255, 255, 255), -1,
                       cv2.LINE_AA, SHIFT)
        else:
            cv2.circle(canvas, c, ri, DARK, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(canvas, c, ri, WHITE, 1, cv2.LINE_AA, SHIFT)


def draw_body_glow(canvas: np.ndarray, torso: np.ndarray, motion: Motion | None,
                   head: tuple[np.ndarray, float] | None, color: tuple[int, int, int],
                   strength: float, cfg: dict) -> None:
    """A player whose orb is full glows: a soft halo round the body and a
    lighter tint on it, in their team colour moved glow_white of the way to
    white, so it reads as light, also on a shirt of the same colour.

    The body is the paint's mask (scene/paint.py body_mask(): torso quad and
    a capsule per arm and leg bone) plus a disc for the head (head: centre
    and radius px; None = no head). It is drawn at glow_scale of the frame's
    resolution and blurred (Gaussian, sigma half of glow_reach_sw shoulder
    widths), so the glow fades out over about that distance from the body's
    edge. The blur is 0.5 at the edge and 1 inside; doubled and capped at
    1, the halo starts at full strength at the edge. Opacity: strength x
    that, cut to glow_inside of it on the body, so the halo outside is
    brighter than the tint on the player. Everything only in the body's box
    widened by twice the reach."""
    h, w = canvas.shape[:2]
    sw = float(np.linalg.norm(torso[1] - torso[0]))
    if sw < 1 or strength <= 0:
        return
    reach = cfg["glow_reach_sw"] * sw
    pts = [torso]
    if motion is not None:
        pts.append(motion.kpts[motion.valid])
    if head is not None:
        (hx, hy), hr = head
        pts.append(np.array([[hx - hr, hy - hr], [hx + hr, hy + hr]]))
    pts = np.concatenate(pts)
    x0, y0 = np.maximum(np.floor(pts.min(axis=0) - 2 * reach).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(pts.max(axis=0) + 2 * reach).astype(int), [w, h])
    if x0 >= x1 or y0 >= y1:
        return
    k = float(cfg["glow_scale"])
    small = (max(1, int(round((y1 - y0) * k))), max(1, int(round((x1 - x0) * k))))
    o = np.array([x0, y0], np.float64)
    m = None if motion is None else Motion((motion.kpts - o) * k, motion.vel, motion.valid, {})
    mask = body_mask(small, (0, 0), (torso - o) * k, m, cfg)
    if head is not None:
        cv2.circle(mask, _p((np.asarray(head[0]) - o) * k), int(head[1] * k * (1 << SHIFT)), 255, -1,
                   cv2.LINE_AA, SHIFT)
    body = mask.astype(np.float32) / 255.0
    halo = cv2.GaussianBlur(body, (0, 0), max(0.5, 0.5 * reach * k))
    alpha = strength * np.minimum(2.0 * halo, 1.0) * (1.0 - (1.0 - cfg["glow_inside"]) * body)
    alpha = cv2.resize(alpha, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    roi = canvas[y0:y1, x0:x1]
    # The colour layer filled by OpenCV: a NumPy broadcast of the colour
    # took ~3.4 ms for a 1000 x 720 px box, the blend itself ~0.3 ms.
    layer = np.empty_like(roi)
    cv2.rectangle(layer, (0, 0), (x1 - x0, y1 - y0), _mix(color, (255, 255, 255), cfg["glow_white"]), -1)
    roi[:] = cv2.blendLinear(layer, roi, alpha, 1.0 - alpha)
