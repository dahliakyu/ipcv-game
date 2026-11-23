"""Task 1 — head pose: pinhole camera, canonical 3D head, angles and anchors.

Model frame: millimetres, origin between the eyes, x to the image right, y
down, z away from the camera (into the head). A frontal face has R = I.
Angles are as seen on screen: yaw > 0 = face turned towards the image right,
pitch > 0 = chin up, roll > 0 = clockwise (as from arctan2 in image coords).
"""

from __future__ import annotations

import cv2
import numpy as np


def camera_matrix(w: int, h: int, focal_per_width: float) -> np.ndarray:
    """Pinhole intrinsics without calibration: focal length focal_per_width *
    image width (px), principal point in the image centre."""
    f = focal_per_width * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _by_image_x(pair: np.ndarray) -> np.ndarray:
    """A left/right pair of points ordered image-left first."""
    return pair[np.argsort(pair[:, 0])]


def head_pose(landmarks: np.ndarray, model_mm: np.ndarray, K: np.ndarray,
              ears: np.ndarray | None = None, ears_mm: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray] | None:
    """Rotation (Rodrigues vector) and translation that put the canonical
    head onto the 5 YuNet landmarks, and optionally the 2 ear keypoints from
    the pose model (solvePnP, SQPnP solver).

    model_mm rows: image-left eye, image-right eye, nose tip, image-left and
    image-right mouth corner; ears_mm: image-left and image-right ear. Pairs
    are matched by image position, not by the detectors' left/right labels,
    so the mirrored frame works. None when there is no valid solution.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    pts = np.vstack([_by_image_x(lm[:2]), lm[2:3], _by_image_x(lm[3:5])])
    model = model_mm
    if ears is not None:
        pts = np.vstack([pts, _by_image_x(np.asarray(ears, dtype=np.float64))])
        model = np.vstack([model_mm, ears_mm])
    # SQPnP asserts on image points that (nearly) coincide: a face only a few
    # pixels wide carries no pose anyway, so treat it as "no solution"
    if not np.isfinite(pts).all() or np.linalg.norm(pts[1] - pts[0]) < 2.0:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(model, pts, K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok or tvec[2, 0] <= 0:                           # head must be in front of the camera
        return None
    return rvec, tvec


def pose_angles(rvec: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in rad from a head rotation (see the module docstring)."""
    R = cv2.Rodrigues(rvec)[0]                               # R = Rz(roll) Ry(-yaw) Rx(-pitch)
    yaw = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    pitch = float(np.arctan2(R[2, 1], R[2, 2]))
    return -yaw, -pitch


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Head rotation matrix from screen angles; inverse of pose_angles() plus roll."""
    a, b, c = -pitch, -yaw, roll
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mediapipe_angles(matrix: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in rad from MediaPipe's facial transformation matrix.
    MediaPipe's face frame has y up and z towards the camera; flipping both
    axes (C R C with C = diag(1, -1, -1)) gives our model frame."""
    C = np.diag([1.0, -1.0, -1.0])
    R = C @ np.asarray(matrix, dtype=np.float64)[:3, :3] @ C
    return pose_angles(cv2.Rodrigues(R)[0])


def translation(R: np.ndarray, model_mm: np.ndarray, points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Head position (mm) for a known rotation: least squares over the
    pinhole equations u = fx (X + tx) / (Z + tz) + cx (and v likewise), which
    are linear in t once multiplied out. Lets the rotation come from
    MediaPipe while the position fits the landmarks in the full frame."""
    X = model_mm @ R.T
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = points[:, 0] - cx, points[:, 1] - cy
    n = len(points)
    A = np.zeros((2 * n, 3))
    A[:n, 0], A[:n, 2] = fx, -du
    A[n:, 1], A[n:, 2] = fy, -dv
    b = np.concatenate([du * X[:, 2] - fx * X[:, 0], dv * X[:, 2] - fy * X[:, 1]])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def sorted_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """The 5 YuNet landmarks in model_mm order (pairs by image position)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    return np.vstack([_by_image_x(lm[:2]), lm[2:3], _by_image_x(lm[3:5])])


def anchor_points_mm(cfg: dict, model_mm: np.ndarray) -> dict[str, np.ndarray]:
    """The forehead, hat, above_head and cheek anchors as 3D points on the head.

    Their frontal projection reproduces geometry()'s 2D offsets: the same
    forehead_rise, hat_rise, above_head_rise, cheek_out and cheek_drop, in
    YuNet box widths, with the box box_per_eye eye distances wide. The hat
    points sit hat_depth_mm behind the face, so they move like a hat on the
    head when it tilts; the cheeks sit cheek_depth_mm behind the eyes.
    """
    box = cfg["box_per_eye"] * float(np.linalg.norm(model_mm[1] - model_mm[0]))
    forehead_y = -cfg["forehead_rise"] * box
    hat_y = forehead_y - cfg["hat_rise"] * box
    d = cfg["hat_depth_mm"]
    cheek_x, cheek_y = cfg["cheek_out"] * box, cfg["cheek_drop"] * box
    return {"forehead": np.array([0.0, forehead_y, 0.0]),
            "hat": np.array([0.0, hat_y, d]),
            "above_head": np.array([0.0, hat_y - cfg["above_head_rise"] * box, d]),
            "left_cheek": np.array([-cheek_x, cheek_y, cfg["cheek_depth_mm"]]),
            "right_cheek": np.array([cheek_x, cheek_y, cfg["cheek_depth_mm"]])}
