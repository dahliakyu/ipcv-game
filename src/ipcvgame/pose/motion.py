"""Task 2 — per-player motion filtering and signals.

Backend "ema" is the Colour Clash prototype's ``KeypointTracker``
(``Project/bodypose.py``), moved here in the gesture merge (docs/merge_log.md,
Task 4). Per player:

- keypoints under ``kpt_conf_threshold`` count as missing this frame;
- a missing keypoint keeps its last position, and stays valid, for ``hold_s``;
- all positions are smoothed by an EMA with time constant ``tau``;
- a keypoint that comes back after more than ``hold_s`` jumps to the detection
  instead of sliding in from a stale spot;
- a frame without a detection resets the player.

Backend "stub" is the phase-1 pass-through. Both compute velocities by finite
difference on Frame.t. Signals are not defined yet (Task 2).
"""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.filters import EMA
from ipcvgame.core.types import FrameState, Motion, Player
from ipcvgame.scene.draw import draw_skeleton, player_color, pt


class _PlayerState:
    """One player's smoothing state (the prototype's ``KeypointTracker``)."""

    def __init__(self, tau: float):
        self.ema = EMA(tau)
        # Time each keypoint was last above the confidence threshold; -inf =
        # never, so the first good detection snaps in place.
        self.last_good = np.full(sk.K, -np.inf)
        self.prev: tuple[np.ndarray, float] | None = None  # (kpts, t) for velocity


class MotionFilter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = cfg.get("backend", "ema")
        if self.backend not in ("ema", "stub"):
            raise ValueError(f"Unknown motion backend {self.backend!r}")
        self.kpt_conf_threshold = float(cfg["kpt_conf_threshold"])
        self.tau = float(cfg["tau"])
        self.hold_s = float(cfg["hold_s"])
        self._state: dict[int, _PlayerState] = {}

    def update(self, players: dict[int, Player], t: float) -> dict[int, Motion]:
        out: dict[int, Motion] = {}
        for pid, p in players.items():
            if p.reentered:
                self.reset(pid)
            if p.det is None:
                if self.backend == "ema":
                    self.reset(pid)
                continue
            st = self._state.setdefault(pid, _PlayerState(self.tau))
            good = p.det.conf >= self.kpt_conf_threshold
            if self.backend == "ema":
                kpts, valid, snap = self._smooth(st, p.det.kpts, good, t)
            else:
                kpts, valid, snap = p.det.kpts.copy(), good, np.zeros(sk.K, bool)
            vel = np.zeros_like(kpts)
            if st.prev is not None and t > st.prev[1]:
                vel = (kpts - st.prev[0]) / (t - st.prev[1])
                vel[snap] = 0.0  # a jump back into place is not motion
            st.prev = (kpts, t)
            # No signals yet; Task 2 defines them (e.g. "r_wrist_speed").
            out[pid] = Motion(kpts=kpts, vel=vel, valid=valid, signals={})
        return out

    def _smooth(self, st: _PlayerState, raw: np.ndarray, good: np.ndarray,
                t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The prototype's ``KeypointTracker.update()``, in seconds."""
        snap = good & (t - st.last_good > self.hold_s)
        st.last_good[good] = t
        # Missing points hold their last position (never blended with a bad guess).
        held = st.ema.x if st.ema.x is not None else raw
        target = np.where(good[:, None], raw, held)
        kpts = st.ema(target, t, snap)
        valid = t - st.last_good <= self.hold_s
        return kpts, valid, snap

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._state.clear()
        else:
            self._state.pop(pid, None)

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Smoothed skeleton per player, velocity arrows (0.1 s ahead), signals."""
        for pid, m in state.motion.items():
            color = player_color(pid)
            draw_skeleton(canvas, m.kpts, m.valid, color)
            for i in np.flatnonzero(m.valid):
                cv2.arrowedLine(canvas, pt(m.kpts[i]), pt(m.kpts[i] + 0.1 * m.vel[i]),
                                (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
