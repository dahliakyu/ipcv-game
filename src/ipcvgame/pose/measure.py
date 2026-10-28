"""Task 2 — body measurements from the smoothed keypoints, for the pose
calibration (feature 1, docs/extra_features.md).

While a player stands in the neutral pose, the game (interaction/game.py)
collects one sample per frame and keeps the median, which ignores the odd
frame where a keypoint jumps. Both are in pixels at the player's distance
from the camera; their ratio (arm length in shoulder widths) does not depend
on that distance, like the gesture rules (interaction/gestures.py).
"""

from __future__ import annotations

import numpy as np

from ipcvgame.core import skeleton as sk

ARMS = ((sk.L_SHOULDER, sk.L_ELBOW, sk.L_WRIST), (sk.R_SHOULDER, sk.R_ELBOW, sk.R_WRIST))


def shoulder_px(kpts: np.ndarray, valid: np.ndarray) -> float | None:
    """Distance between the shoulders (px), the unit of every gesture rule;
    None unless both are valid."""
    if not (valid[sk.L_SHOULDER] and valid[sk.R_SHOULDER]):
        return None
    return float(np.linalg.norm(kpts[sk.L_SHOULDER] - kpts[sk.R_SHOULDER]))


def arm_px(kpts: np.ndarray, valid: np.ndarray) -> float | None:
    """Arm length (px): shoulder to elbow plus elbow to wrist, so a bent arm
    measures the same as a straight one. The mean of the arms with all three
    keypoints valid; None if neither has them."""
    lengths = [float(np.linalg.norm(kpts[e] - kpts[s]) + np.linalg.norm(kpts[w] - kpts[e]))
               for s, e, w in ARMS if valid[s] and valid[e] and valid[w]]
    return float(np.mean(lengths)) if lengths else None
