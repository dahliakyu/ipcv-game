"""Task 1 — face tracking.

PHASE-1 STUB (owned by Task 1): no face model. Places an "above_head" anchor
above the nose keypoint, scaled by shoulder width. Replace with a real face
landmarker on per-player head crops, keeping the same update() signature and
the Face anchors that Tasks 4 and 5 draw at.
"""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.types import Face, Frame, FrameState, Player
from ipcvgame.scene.draw import draw_text, player_color, pt


def measure(row: np.ndarray, center_drop: float, forehead_rise: float, lip_to_nose: float) -> dict:
    """One YuNet detection -> 4 key points (left side, right side, forehead,
    upper lip) in pixels, plus the face centre and the raw landmarks.

    row: [x, y, w, h, right eye (x, y), left eye, nose, right mouth corner,
    left mouth corner, score]. Distances are in face widths (YuNet box width).
    """
    bw = float(row[2])
    e1, e2 = row[4:6].astype(float), row[6:8].astype(float)
    if e1[0] > e2[0]:                                       # e1 = eye on the image left
        e1, e2 = e2, e1
    nose, mouth = row[8:10].astype(float), (row[10:12] + row[12:14]).astype(float) / 2
    u = e2 - e1
    u = u / max(np.linalg.norm(u), 1e-6)                    # along the eye line
    down = np.array([-u[1], u[0]])                          # face's "down" in image coords
    m = (e1 + e2) / 2
    c = m + center_drop * bw * down                         # face centre at cheek level
    pts = np.array([c - 0.5 * bw * u, c + 0.5 * bw * u,     # left / right side: one face width apart
                    m - forehead_rise * bw * down,          # forehead
                    mouth + lip_to_nose * (nose - mouth)])  # upper lip
    return {"pts": pts, "center": c, "width": bw,
            "landmarks": row[4:14].reshape(5, 2).astype(float)}


def geometry(pts: np.ndarray, hat_drop: float, above_head_rise: float) -> dict:
    """4 key points (left, right, forehead, lip) -> face width, head roll
    and the anchors effects draw at. Offsets are in face widths along the
    face's own "down", so they follow a tilted head."""
    left, right, top, lip = pts
    d = right - left
    w = float(np.linalg.norm(d))
    roll = float(np.arctan2(d[1], d[0]))                    # 0 = upright, positive = clockwise on screen
    down = np.array([-np.sin(roll), np.cos(roll)])
    hat = top + hat_drop * w * down                         # hat brim
    return {"width": w, "roll": roll, "center": (left + right) / 2,
            "anchors": {"forehead": top, "hat": hat, "mouth": lip,
                        "above_head": hat - above_head_rise * w * down}}


class FaceTracker:
    def __init__(self, cfg: dict, kpt_conf_threshold: float = 0.5):
        self.cfg = cfg
        self.above_head_factor = float(cfg.get("above_head_factor", 0.9))
        self.default_scale = float(cfg.get("default_scale", 80.0))
        self.kpt_conf_threshold = kpt_conf_threshold

    def update(self, frame: Frame, players: dict[int, Player]) -> dict[int, Face]:
        faces: dict[int, Face] = {}
        for pid, p in players.items():
            if p.det is None:
                continue
            k, ok = p.det.kpts, p.det.conf >= self.kpt_conf_threshold
            if not ok[sk.NOSE]:
                continue
            nose = k[sk.NOSE].copy()
            if ok[sk.L_SHOULDER] and ok[sk.R_SHOULDER]:
                scale = float(np.linalg.norm(k[sk.L_SHOULDER] - k[sk.R_SHOULDER]))
            else:
                scale = self.default_scale
            faces[pid] = Face(
                pid=pid, valid=True, landmarks=None, center=nose, scale=scale,
                roll=0.0, yaw=0.0,
                anchors={"above_head": nose - np.array([0.0, self.above_head_factor * scale])},
            )
        return faces

    def reset(self, pid: int | None = None) -> None:
        pass  # stateless stub

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Face centre, scale circle and every anchor with its name."""
        for pid, f in state.faces.items():
            color = player_color(pid)
            cv2.circle(canvas, pt(f.center), int(f.scale / 2), color, 1, cv2.LINE_AA)
            for name, a in f.anchors.items():
                cv2.drawMarker(canvas, pt(a), color, cv2.MARKER_CROSS, 14, 2)
                draw_text(canvas, name, (pt(a)[0] + 8, pt(a)[1]), color, 0.45)
