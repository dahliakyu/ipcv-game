"""Shared data contracts between modules.

This file is the contract between tasks. Do not change it without a heads-up
to the whole team (see docs/phase-1/setup-guide.md).

Conventions:
- All coordinates are pixels in the full-resolution, mirrored frame.
- All filters use Frame.t (seconds, time.perf_counter()), never frame counts.
- Keypoints are in canonical skeleton order (core/skeleton.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    image: np.ndarray          # BGR, already mirrored, full resolution
    frame_id: int
    t: float                   # capture time, time.perf_counter()


@dataclass
class PoseDetection:           # Task 2 (pose detection) output, NO identity
    bbox: np.ndarray           # (4,) x1, y1, x2, y2 in pixels
    kpts: np.ndarray           # (K, 2) pixels, canonical skeleton order
    conf: np.ndarray           # (K,)
    score: float


@dataclass
class Player:                  # Task 3 output
    pid: int                   # 1, 2, ...
    status: str                # "active" | "occluded" | "lost"
    det: PoseDetection | None  # None when occluded or lost
    bbox: np.ndarray           # predicted box when det is None
    pos: np.ndarray            # (x, depth) estimate in metres, stub OK for now
    reentered: bool            # True on the frame a lost player comes back


@dataclass
class Motion:                  # Task 2 (motion filtering) output, per player
    kpts: np.ndarray           # smoothed
    vel: np.ndarray            # px/s
    valid: np.ndarray          # (K,) bool, keypoint trustworthy this frame
    signals: dict[str, float]  # "r_wrist_speed", "hands_above_head", ...


@dataclass
class Face:                    # Task 1 output, per player
    pid: int
    valid: bool
    landmarks: np.ndarray | None
    center: np.ndarray
    scale: float
    roll: float
    yaw: float
    anchors: dict[str, np.ndarray]   # "above_head", "forehead", "mouth"


@dataclass
class GestureEvent:            # Task 4 output
    pid: int
    name: str
    t: float
    conf: float


@dataclass
class FrameState:              # everything for one frame, read by game + renderer
    frame: Frame
    players: dict[int, Player]
    motion: dict[int, Motion]
    faces: dict[int, Face]
    events: list[GestureEvent]
    timings: dict[str, float]
