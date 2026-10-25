"""Task 4 — gesture recognition: motion signals -> GestureEvent.

The rules are the Colour Clash prototype's (``Project/bodypose.py``), moved
here in the gesture merge (docs/merge_log.md, Task 4). They read the smoothed
keypoints and validity from ``Motion``; all distances are in shoulder widths
(``sw``), so they work at any distance from the camera. Every number comes
from the ``gestures:`` config.

``GestureRecognizer`` is still the phase-1 stub (no events) until merge step 3.
"""

from __future__ import annotations

import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.geometry import angle_at
from ipcvgame.core.types import FrameState, GestureEvent, Motion
from ipcvgame.scene.draw import draw_lines


def lower_body_y(kpts: np.ndarray, valid: np.ndarray, shoulder_y: float, sw: float,
                 cfg: dict) -> float:
    """Hip level, or an estimate below the shoulders if hips are not visible."""
    if valid[sk.L_HIP] and valid[sk.R_HIP]:
        return (kpts[sk.L_HIP][1] + kpts[sk.R_HIP][1]) / 2.0
    return shoulder_y + cfg["hip_below_shoulders"] * sw


def detect_hands_up(kpts: np.ndarray, valid: np.ndarray, sw: float, cfg: dict) -> bool:
    """Both wrists above the head."""
    if not (valid[sk.L_WRIST] and valid[sk.R_WRIST]):
        return False

    if valid[sk.NOSE]:
        head_y = kpts[sk.NOSE][1]
    else:
        shoulder_y = (kpts[sk.L_SHOULDER][1] + kpts[sk.R_SHOULDER][1]) / 2.0
        head_y = shoulder_y - cfg["head_above_shoulders"] * sw

    return bool(kpts[sk.L_WRIST][1] < head_y and kpts[sk.R_WRIST][1] < head_y)


def detect_crossed(kpts: np.ndarray, valid: np.ndarray, sw: float, cfg: dict) -> bool:
    """Forearms crossed in front of the chest."""
    if not (valid[sk.L_WRIST] and valid[sk.R_WRIST]
            and valid[sk.L_SHOULDER] and valid[sk.R_SHOULDER]):
        return False

    lw, rw = kpts[sk.L_WRIST], kpts[sk.R_WRIST]
    ls, rs = kpts[sk.L_SHOULDER], kpts[sk.R_SHOULDER]

    # Wrists in the opposite left/right order to the shoulders
    # (independent of mirroring / label convention)
    swapped = (lw[0] - rw[0]) * (ls[0] - rs[0]) < 0

    shoulder_y = (ls[1] + rs[1]) / 2.0
    hip_y = lower_body_y(kpts, valid, shoulder_y, sw, cfg)

    top = shoulder_y - cfg["chest_above_shoulders"] * sw
    in_chest_zone = top < lw[1] < hip_y and top < rw[1] < hip_y

    close = np.linalg.norm(lw - rw) < cfg["cross_max_wrist_dist"] * sw

    return bool(swapped and in_chest_zone and close)


def arm_extended(kpts: np.ndarray, valid: np.ndarray, sw: float, shoulder: int,
                 elbow: int, wrist: int, cfg: dict) -> bool:
    """Arm is straight, reaching out, and at roughly shoulder height."""
    if not (valid[shoulder] and valid[elbow] and valid[wrist]):
        return False

    straight = angle_at(kpts[shoulder], kpts[elbow], kpts[wrist]) > cfg["punch_elbow_angle"]

    reach = np.linalg.norm(kpts[wrist] - kpts[shoulder]) / sw > cfg["punch_min_reach"]

    # Wrist roughly between head and hip height (excludes hands up)
    shoulder_y = (kpts[sk.L_SHOULDER][1] + kpts[sk.R_SHOULDER][1]) / 2.0
    hip_y = lower_body_y(kpts, valid, shoulder_y, sw, cfg)
    at_height = shoulder_y - cfg["punch_above_shoulders"] * sw < kpts[wrist][1] < hip_y

    return bool(straight and reach and at_height)


def arm_bent(kpts: np.ndarray, valid: np.ndarray, shoulder: int, elbow: int, wrist: int,
             cfg: dict) -> bool:
    """Arm is pulled in (guard). Unknown arms count as bent."""
    if not (valid[shoulder] and valid[elbow] and valid[wrist]):
        return True

    return angle_at(kpts[shoulder], kpts[elbow], kpts[wrist]) < cfg["punch_bent_angle"]


def detect_punch(kpts: np.ndarray, valid: np.ndarray, sw: float, cfg: dict) -> str | None:
    """Return 'L', 'R' or None: the model's side label of the punching arm.

    On our mirrored frame the model's 'L' is the player's own right arm (see
    core/skeleton.py). The event drops the side; it is kept here for debugging.
    """
    left = arm_extended(kpts, valid, sw, sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST, cfg)
    right = arm_extended(kpts, valid, sw, sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST, cfg)

    # One hand only: the other arm must be bent
    if left and arm_bent(kpts, valid, sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST, cfg):
        return "L"
    if right and arm_bent(kpts, valid, sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST, cfg):
        return "R"

    return None


class GestureRecognizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def update(self, motion: dict[int, Motion], t: float) -> list[GestureEvent]:
        return []

    def reset(self, pid: int | None = None) -> None:
        pass

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Each player's motion signals and this frame's events."""
        lines = []
        for pid, m in sorted(state.motion.items()):
            sig = "  ".join(f"{k}={v:.0f}" for k, v in m.signals.items())
            lines.append(f"P{pid}: {sig}")
        lines += [f"EVENT P{e.pid} {e.name} ({e.conf:.2f})" for e in state.events]
        draw_lines(canvas, lines, (10, canvas.shape[0] - 20 * len(lines) - 10))
