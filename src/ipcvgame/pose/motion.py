"""Task 2 — per-player motion filtering and signals.

Backend "ema" is ``KeypointTracker`` from the Colour Clash code
(``Project/bodypose.py``), moved here in the gesture merge (docs/merge_log.md,
Task 4). Per player:

- keypoints under ``kpt_conf_threshold`` count as missing this frame;
- a missing keypoint keeps its last position, and stays valid, for ``hold_s``;
- all positions are smoothed by an EMA with time constant ``tau``;
- a keypoint that comes back after more than ``hold_s`` jumps to the detection
  instead of sliding in from a stale spot;
- a frame without a detection resets the player.

Backend "stub" is the phase-1 pass-through. Both compute velocities by finite
difference on Frame.t.

Signals (``Motion.signals``): ``l_wrist_speed`` and ``r_wrist_speed``, each
wrist's speed in shoulder widths per second (from the smoothed velocity and
the smoothed shoulder width), so the value does not depend on the distance
to the camera. Only for a wrist that is valid with both shoulders valid and
at least ``min_shoulder_px`` apart; otherwise the key is absent. ``l_`` is
the model's label: on the mirrored frame, the player's own right wrist
(core/skeleton.py).

Logging (``log_fields()``): per player and frame, the raw detection against
the smoothed output: how far the keypoints moved since the last frame in
each (for a player standing still, that is the jitter before and after
smoothing), how far the smoothed points are from the detections (the lag
the smoothing adds), how many keypoints are valid and how many of them are
held, and the wrist speeds.
"""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.filters import EMA
from ipcvgame.core.types import FrameState, Motion, Player
from ipcvgame.scene.draw import draw_skeleton, draw_text, player_color, pt


class _PlayerState:
    """One player's smoothing state (``KeypointTracker`` in bodypose.py)."""

    def __init__(self, tau: float):
        self.ema = EMA(tau)
        # Time each keypoint was last above the confidence threshold; -inf =
        # never, so the first good detection snaps in place.
        self.last_good = np.full(sk.K, -np.inf)
        self.prev: tuple[np.ndarray, float] | None = None  # (kpts, t) for velocity
        # This frame and the last, for log_fields(): raw detection, its
        # trusted keypoints, the smoothed output and its valid keypoints.
        self.frame: dict | None = None
        self.last_frame: dict | None = None


class MotionFilter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = cfg.get("backend", "ema")
        if self.backend not in ("ema", "stub"):
            raise ValueError(f"Unknown motion backend {self.backend!r}")
        self.kpt_conf_threshold = float(cfg["kpt_conf_threshold"])
        self.tau = float(cfg["tau"])
        self.hold_s = float(cfg["hold_s"])
        self.min_shoulder_px = float(cfg["min_shoulder_px"])
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
            st.last_frame, st.frame = st.frame, {"raw": p.det.kpts.astype(np.float64), "good": good,
                                                 "kpts": kpts, "valid": valid}
            out[pid] = Motion(kpts=kpts, vel=vel, valid=valid, signals=self._signals(kpts, vel, valid))
        return out

    def _signals(self, kpts: np.ndarray, vel: np.ndarray, valid: np.ndarray) -> dict[str, float]:
        """Wrist speeds in shoulder widths per second (module doc)."""
        if not (valid[sk.L_SHOULDER] and valid[sk.R_SHOULDER]):
            return {}
        sw = float(np.linalg.norm(kpts[sk.L_SHOULDER] - kpts[sk.R_SHOULDER]))
        if sw < self.min_shoulder_px:
            return {}
        return {f"{sk.NAMES[w]}_speed": float(np.linalg.norm(vel[w])) / sw
                for w in (sk.L_WRIST, sk.R_WRIST) if valid[w]}

    def _smooth(self, st: _PlayerState, raw: np.ndarray, good: np.ndarray,
                t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``KeypointTracker.update()`` from bodypose.py, in seconds."""
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

    def log_fields(self, state: FrameState) -> dict:
        """Per-player CSV columns for the players in this frame's output:

        - motion_p<pid>_valid: valid keypoints; motion_p<pid>_held: of
          those, how many hold their last position (not trusted this frame);
        - motion_p<pid>_raw_step_px / _smooth_step_px: mean distance (px) the
          keypoints moved since the last frame, in the raw detection and in
          the smoothed output, over the keypoints trusted in both frames;
          empty on a player's first frame or without such keypoints;
        - motion_p<pid>_lag_px: mean distance (px) between the smoothed and
          the raw keypoints, over the keypoints trusted this frame;
        - motion_p<pid>_l_wrist_speed / _r_wrist_speed: the signals (sw/s),
          empty when absent."""
        row: dict = {}
        for pid, m in sorted(state.motion.items()):
            st = self._state.get(pid)
            if st is None or st.frame is None:
                continue
            f, last, pre = st.frame, st.last_frame, f"motion_p{pid}_"
            row[pre + "valid"] = int(f["valid"].sum())
            row[pre + "held"] = int((f["valid"] & ~f["good"]).sum())
            raw_step = smooth_step = ""
            if last is not None:
                both = f["good"] & last["good"]
                if both.any():
                    raw_step = float(np.linalg.norm(f["raw"][both] - last["raw"][both], axis=1).mean())
                    smooth_step = float(np.linalg.norm(f["kpts"][both] - last["kpts"][both], axis=1).mean())
            row[pre + "raw_step_px"] = "" if raw_step == "" else round(raw_step, 3)
            row[pre + "smooth_step_px"] = "" if smooth_step == "" else round(smooth_step, 3)
            good = f["good"]
            lag = np.linalg.norm(f["kpts"][good] - f["raw"][good], axis=1)
            row[pre + "lag_px"] = round(float(lag.mean()), 3) if good.any() else ""
            for name in ("l_wrist_speed", "r_wrist_speed"):
                v = m.signals.get(name)
                row[pre + name] = "" if v is None else round(v, 3)
        return row

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Smoothed skeleton per player, velocity arrows (0.1 s ahead), and
        each wrist's speed (sw/s) beside it."""
        for pid, m in state.motion.items():
            color = player_color(pid)
            draw_skeleton(canvas, m.kpts, m.valid, color)
            for i in np.flatnonzero(m.valid):
                cv2.arrowedLine(canvas, pt(m.kpts[i]), pt(m.kpts[i] + 0.1 * m.vel[i]),
                                (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
            for w in (sk.L_WRIST, sk.R_WRIST):
                v = m.signals.get(f"{sk.NAMES[w]}_speed")
                if v is not None:
                    draw_text(canvas, f"{v:.1f} sw/s", (int(m.kpts[w][0]) + 6, int(m.kpts[w][1])),
                              (0, 255, 255), 0.4)
