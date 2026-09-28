"""Task 1 — face-anchored AR effects.

PHASE-1 STUB (owned by Task 1): draws a simple marker at each player's
"above_head" anchor. Replace with sprite overlays (alpha blending, warpAffine
by roll/scale). The renderer calls draw(canvas, faces).
"""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core.types import Face
from ipcvgame.scene.draw import player_color, pt


class FaceEffects:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def draw(self, canvas: np.ndarray, faces: dict[int, Face]) -> None:
        for pid, f in faces.items():
            if not f.valid or "above_head" not in f.anchors:
                continue
            r = max(6, int(f.scale * 0.15))
            cv2.circle(canvas, pt(f.anchors["above_head"]), r, player_color(pid), -1, cv2.LINE_AA)

    def reset(self, pid: int | None = None) -> None:
        pass
