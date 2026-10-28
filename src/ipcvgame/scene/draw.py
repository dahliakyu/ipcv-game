"""Drawing helpers shared by the renderer and the modules' debug overlays."""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk

# BGR colours per player id; index 0 is for "no identity". P1 red and P2 blue
# are the Colour Clash team colours from game.py (Task 4), the one table
# for paint, renderer and debug views.
PLAYER_COLORS = [(200, 200, 200), (55, 55, 235), (235, 110, 40), (0, 200, 0), (200, 0, 200)]


# BGR colours of the drinks' buff categories (consumables): green, orange
# and purple, away from the team colours (red, blue). The one table for the
# drinks' badges, buff icons, pick-up text and --until game.
DRINK_COLORS = {"hp": (80, 200, 60), "attack": (30, 140, 255), "defense": (200, 60, 160)}


def player_color(pid: int | None) -> tuple[int, int, int]:
    return PLAYER_COLORS[(pid or 0) % len(PLAYER_COLORS)]


def pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_skeleton(canvas: np.ndarray, kpts: np.ndarray, valid: np.ndarray,
                  color: tuple[int, int, int], thickness: int = 2) -> None:
    for a, b in sk.EDGES:
        if valid[a] and valid[b]:
            cv2.line(canvas, pt(kpts[a]), pt(kpts[b]), color, thickness, cv2.LINE_AA)
    for i in np.flatnonzero(valid):
        cv2.circle(canvas, pt(kpts[i]), thickness + 1, color, -1, cv2.LINE_AA)


def draw_bbox(canvas: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int],
              label: str | None = None, dashed: bool = False) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if dashed:
        for (ax, ay), (bx, by) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                                   ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
            n = max(1, int(np.hypot(bx - ax, by - ay) // 12))
            for k in range(0, n, 2):
                p = (ax + (bx - ax) * k // n, ay + (by - ay) * k // n)
                q = (ax + (bx - ax) * (k + 1) // n, ay + (by - ay) * (k + 1) // n)
                cv2.line(canvas, p, q, color, 2)
    else:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        draw_text(canvas, label, (x1, max(0, y1 - 8)), color)


def draw_text(canvas: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.6) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_lines(canvas: np.ndarray, lines: list[str], org: tuple[int, int] = (10, 24),
               color: tuple[int, int, int] = (255, 255, 255)) -> None:
    x, y = org
    for line in lines:
        draw_text(canvas, line, (x, y), color, 0.5)
        y += 20


def blend_rgba(canvas: np.ndarray, sprite: np.ndarray, x0: float, y0: float,
               alpha: float = 1.0) -> None:
    """Alpha-blend a BGRA sprite onto a BGR canvas in place, its top-left
    corner at (x0, y0) px; the parts outside the canvas are clipped."""
    h, w = sprite.shape[:2]
    x0, y0 = int(round(x0)), int(round(y0))
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, canvas.shape[1]), min(y0 + h, canvas.shape[0])
    if cx0 >= cx1 or cy0 >= cy1:
        return
    s = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = s[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    roi = canvas[cy0:cy1, cx0:cx1]
    roi[:] = (s[:, :, :3] * a + roi * (1.0 - a)).astype(np.uint8)


def hue_of(bgr: tuple[int, int, int]) -> int:
    """OpenCV hue (0-179) of one BGR colour."""
    return int(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])


def main_hue(images: list[np.ndarray]) -> int:
    """Main colour of BGRA sprites: the mean hue (OpenCV, 0-179) of their
    coloured pixels (opaque, saturation > 60). Hue is an angle, so this is a
    circular mean: reds just above 0 and just below 180 average to red."""
    px = np.concatenate([im.reshape(-1, 4) for im in images])
    hsv = cv2.cvtColor(px[None, :, :3], cv2.COLOR_BGR2HSV)[0]
    coloured = (px[:, 3] > 0) & (hsv[:, 1] > 60)
    angle = hsv[coloured, 0].astype(np.float64) * (2 * np.pi / 180)
    mean = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
    return int(round(mean * 180 / (2 * np.pi))) % 180


def shift_hue(image: np.ndarray, shift: int) -> np.ndarray:
    """Copy of a BGRA sprite with every pixel's hue turned by `shift` (OpenCV
    units, 0-179); brightness, saturation and transparency stay, so white
    stays white and the main colour moves to a new team colour."""
    out = image.copy()
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + shift) % 180
    out[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out
