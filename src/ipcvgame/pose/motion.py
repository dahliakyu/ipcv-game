"""Task 2 — per-player motion filtering and signals.

PHASE-1 STUB (owned by Task 2): passes keypoints through unsmoothed and
computes velocities by finite difference on Frame.t. Replace with per-player
One Euro / Kalman filters (core/filters.py) keeping the same update() signature.
"""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core.types import FrameState, Motion, Player
from ipcvgame.scene.draw import draw_skeleton, player_color, pt


class MotionFilter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kpt_conf_threshold = float(cfg.get("kpt_conf_threshold", 0.5))
        self._prev: dict[int, tuple[np.ndarray, float]] = {}  # pid -> (kpts, t)

    def update(self, players: dict[int, Player], t: float) -> dict[int, Motion]:
        out: dict[int, Motion] = {}
        for pid, p in players.items():
            if p.reentered:
                self.reset(pid)
            if p.det is None:
                continue
            kpts = p.det.kpts.copy()
            valid = p.det.conf >= self.kpt_conf_threshold
            vel = np.zeros_like(kpts)
            prev = self._prev.get(pid)
            if prev is not None and t > prev[1]:
                vel = (kpts - prev[0]) / (t - prev[1])
            self._prev[pid] = (kpts, t)
            # Stub: no signals yet; Task 2 defines them (e.g. "r_wrist_speed").
            out[pid] = Motion(kpts=kpts, vel=vel, valid=valid, signals={})
        return out

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._prev.clear()
        else:
            self._prev.pop(pid, None)

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Smoothed skeleton per player, velocity arrows (0.1 s ahead), signals."""
        for pid, m in state.motion.items():
            color = player_color(pid)
            draw_skeleton(canvas, m.kpts, m.valid, color)
            for i in np.flatnonzero(m.valid):
                cv2.arrowedLine(canvas, pt(m.kpts[i]), pt(m.kpts[i] + 0.1 * m.vel[i]),
                                (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
