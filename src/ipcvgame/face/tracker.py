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
