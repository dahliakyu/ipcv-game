"""Task 5 — pictures and text for the calibration phase's instructions
(game start, docs/merge_log.md).

Each move is shown as a stick figure drawn with OpenCV lines, so no image
assets are needed. A figure is a few fixed joint positions in shoulder widths
(x to the right, y down, the shoulder midpoint at 0, 0), the same unit the
gesture rules use (interaction/gestures.py): the punch arm reaches 2 shoulder
widths to the side (punch_min_reach is 1), the hands-up wrists are above the
head, the crossed wrists are in front of the chest.

Joints "a" are on the image's left, "b" on its right. The figures face the
viewer like a mirror, as the players see themselves in the mirrored frame.
"""

from __future__ import annotations

import cv2
import numpy as np

# The body, the same in every pose: head centre and radius, shoulders, hips, feet.
HEAD, HEAD_R = (0.0, -0.6), 0.35
BODY = {
    "a_shoulder": (-0.5, 0.0), "b_shoulder": (0.5, 0.0),
    "pelvis": (0.0, 1.6), "a_hip": (-0.3, 1.6), "b_hip": (0.3, 1.6),
    "a_foot": (-0.45, 3.2), "b_foot": (0.45, 3.2),
}
# The arms per pose: (elbow, wrist) for each side.
ARMS = {
    # Arms hanging by the sides (step 4's neutral pose).
    "neutral": {"a": ((-0.6, 0.8), (-0.65, 1.55)), "b": ((0.6, 0.8), (0.65, 1.55))},
    # Arm b straight out to the side at shoulder height, arm a bent in guard.
    "punch": {"a": ((-0.95, 0.8), (-0.35, 0.45)), "b": ((1.25, 0.0), (2.0, 0.0))},
    # Both wrists above the head.
    "hands_up": {"a": ((-0.85, -0.7), (-0.6, -1.45)), "b": ((0.85, -0.7), (0.6, -1.45))},
    # Wrists crossed in front of the chest, each just past the middle. Not
    # the hands on the opposite shoulders: there the pose model puts each
    # wrist back on its own side, and the rule misses the cross
    # (docs/limitations.md).
    "arms_crossed": {"a": ((-0.65, 0.8), (0.25, 0.4)), "b": ((0.65, 0.8), (-0.25, 0.4))},
}
# What each move is called, how to do it and what it does in the game.
MOVE_TEXT = {
    "neutral": ("STAND STILL", "ARMS DOWN, FACE THE CAMERA", "MEASURING YOU"),
    "punch": ("PUNCH", "ONE ARM OUT TO THE SIDE", "FIRES A BOLT"),
    "hands_up": ("HANDS UP", "BOTH HANDS ABOVE YOUR HEAD", "MAGIC BEAM"),
    "arms_crossed": ("CROSS ARMS", "CROSS WRISTS IN FRONT OF CHEST", "SHIELD"),
}


def figure_points(pose: str, mirror: bool = False) -> dict[str, np.ndarray]:
    """All joints of a pose in shoulder widths; mirror flips left and right."""
    pts = dict(BODY)
    for side, (elbow, wrist) in ARMS[pose].items():
        pts[f"{side}_elbow"], pts[f"{side}_wrist"] = elbow, wrist
    pts["head"], pts["neck"] = HEAD, (HEAD[0], HEAD[1] + HEAD_R)
    sign = -1.0 if mirror else 1.0
    return {k: np.array([sign * x, y]) for k, (x, y) in pts.items()}


# The figure's lines: spine, shoulders, hips, legs and arms.
LINES = [("neck", "pelvis"), ("a_shoulder", "b_shoulder"), ("a_hip", "b_hip"),
         ("pelvis", "a_foot"), ("pelvis", "b_foot"),
         ("a_shoulder", "a_elbow"), ("a_elbow", "a_wrist"), ("b_shoulder", "b_elbow"), ("b_elbow", "b_wrist")]


def draw_figure(canvas: np.ndarray, pose: str, center: tuple[float, float], height: float,
                color: tuple[int, int, int], mirror: bool = False) -> None:
    """The pose's stick figure, fitted into a height x height square at center."""
    pts = figure_points(pose, mirror)
    xy = np.array(list(pts.values()))
    lo = xy.min(axis=0) - (0, HEAD_R)               # the head's top is above its centre
    hi = xy.max(axis=0)
    unit = height / max(hi[1] - lo[1], hi[0] - lo[0])
    offset = np.array(center) - (lo + hi) / 2 * unit

    def p(name: str) -> tuple[int, int]:
        x, y = pts[name] * unit + offset
        return int(round(x)), int(round(y))
    thick = max(2, int(round(0.18 * unit)))
    for a, b in LINES:
        cv2.line(canvas, p(a), p(b), color, thick, cv2.LINE_AA)
    cv2.circle(canvas, p("head"), int(round(HEAD_R * unit)), color, thick, cv2.LINE_AA)
    for wrist in ("a_wrist", "b_wrist"):
        cv2.circle(canvas, p(wrist), int(round(0.6 * thick)) + 1, color, -1, cv2.LINE_AA)


def draw_tick(canvas: np.ndarray, center: tuple[float, float], size: float,
              color: tuple[int, int, int], thickness: int) -> None:
    """A tick mark size px wide, centred on center."""
    tick = np.array([(-0.5, 0.05), (-0.12, 0.4), (0.5, -0.4)]) * size + center
    cv2.polylines(canvas, [tick.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
