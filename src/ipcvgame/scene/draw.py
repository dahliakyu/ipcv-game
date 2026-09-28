"""Drawing helpers shared by the renderer and the modules' debug overlays."""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk

# BGR colours per player id; index 0 is for "no identity".
PLAYER_COLORS = [(200, 200, 200), (255, 140, 0), (0, 90, 255), (0, 200, 0), (200, 0, 200)]


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
