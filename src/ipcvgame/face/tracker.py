"""Task 1 — face tracking.

YuNet face detection (OpenCV) on a head crop per player. The crop comes from
the player's pose keypoints, so every face belongs to the player Task 3
assigned: faces cannot swap between players. The 4 key points from
measure() are smoothed with a One Euro filter per player, and geometry()
turns them into the Face size, roll and the anchors effects draw at.

backend "stub" keeps the phase-1 stand-in (an above_head anchor above the
nose keypoint), which needs no model file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import REPO_ROOT
from ipcvgame.core.filters import OneEuro
from ipcvgame.core.geometry import crop
from ipcvgame.core.types import Face, Frame, FrameState, Player
from ipcvgame.scene.draw import draw_bbox, draw_text, player_color, pt

HEAD_KPTS = [sk.NOSE, sk.L_EYE, sk.R_EYE, sk.L_EAR, sk.R_EAR]


def measure(row: np.ndarray, center_drop: float, forehead_rise: float, lip_to_nose: float) -> dict:
    """One YuNet detection -> 4 key points (left side, right side, forehead,
    upper lip) in pixels, plus the face centre and the raw landmarks.

    row: [x, y, w, h, right eye (x, y), left eye, nose, right mouth corner,
    left mouth corner, score]. Distances are in face widths (YuNet box width).
    """
    bw = float(row[2])
    e1, e2 = row[4:6].astype(float), row[6:8].astype(float)
    if e1[0] > e2[0]:                                       # e1 = eye on the image left
        e1, e2 = e2, e1
    nose, mouth = row[8:10].astype(float), (row[10:12] + row[12:14]).astype(float) / 2
    u = e2 - e1
    u = u / max(np.linalg.norm(u), 1e-6)                    # along the eye line
    down = np.array([-u[1], u[0]])                          # face's "down" in image coords
    m = (e1 + e2) / 2
    c = m + center_drop * bw * down                         # face centre at cheek level
    pts = np.array([c - 0.5 * bw * u, c + 0.5 * bw * u,     # left / right side: one face width apart
                    m - forehead_rise * bw * down,          # forehead
                    mouth + lip_to_nose * (nose - mouth)])  # upper lip
    return {"pts": pts, "center": c, "width": bw,
            "landmarks": row[4:14].reshape(5, 2).astype(float)}


def geometry(pts: np.ndarray, hat_drop: float, above_head_rise: float) -> dict:
    """4 key points (left, right, forehead, lip) -> face width, head roll
    and the anchors effects draw at. Offsets are in face widths along the
    face's own "down", so they follow a tilted head."""
    left, right, top, lip = pts
    d = right - left
    w = float(np.linalg.norm(d))
    roll = float(np.arctan2(d[1], d[0]))                    # 0 = upright, positive = clockwise on screen
    down = np.array([-np.sin(roll), np.cos(roll)])
    hat = top + hat_drop * w * down                         # hat brim
    return {"width": w, "roll": roll, "center": (left + right) / 2,
            "anchors": {"forehead": top, "hat": hat, "mouth": lip,
                        "above_head": hat - above_head_rise * w * down}}


def head_box(p: Player, kpt_conf_threshold: float, crop_size: float) -> np.ndarray | None:
    """Square crop around the player's head from the pose keypoints, or None
    when the player has no detection this frame.

    Centre: mean of the visible head keypoints (nose, eyes, ears), else the
    top of the person box. Side: crop_size shoulder widths, else crop_size
    person-box widths when a shoulder is not visible.
    """
    if p.det is None:
        return None
    k, ok = p.det.kpts, p.det.conf >= kpt_conf_threshold
    x1, y1, x2, y2 = p.det.bbox
    if ok[sk.L_SHOULDER] and ok[sk.R_SHOULDER]:
        side = crop_size * float(np.linalg.norm(k[sk.L_SHOULDER] - k[sk.R_SHOULDER]))
    else:
        side = crop_size * float(x2 - x1)
    head = [i for i in HEAD_KPTS if ok[i]]
    c = k[head].mean(axis=0) if head else np.array([(x1 + x2) / 2, y1 + side / 2])
    return np.array([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2])


@dataclass
class _FaceState:
    """One player's face state between frames."""
    filter: OneEuro                     # on the 4 key points
    width: float = 0.0                  # last detected face width (px), for the size-jump check
    last_seen: float = 0.0              # Frame.t of the last detection
    debug: dict = field(default_factory=dict)   # crop box and raw detection, for draw_debug


class FaceTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = cfg.get("backend", "yunet")
        self.kpt_conf_threshold = float(cfg["kpt_conf_threshold"])
        self._state: dict[int, _FaceState] = {}
        if self.backend == "yunet":
            model = Path(cfg["model"])
            if not model.is_absolute():
                model = REPO_ROOT / model
            if not model.is_file():
                raise FileNotFoundError(f"{model} not found; the README has the download command")
            self.detector = cv2.FaceDetectorYN.create(
                str(model), "", (320, 320), float(cfg["score_threshold"]),
                float(cfg["nms_threshold"]), int(cfg["top_k"]))
        elif self.backend != "stub":
            raise ValueError(f"Unknown face backend {self.backend!r}")

    def update(self, frame: Frame, players: dict[int, Player]) -> dict[int, Face]:
        if self.backend == "stub":
            return self._stub_faces(players)
        cfg, t = self.cfg, frame.t
        for pid in [pid for pid in self._state if pid not in players]:
            del self._state[pid]
        faces: dict[int, Face] = {}
        for pid, p in players.items():
            st = self._state.get(pid)
            if st is not None and t - st.last_seen > cfg["lost_after_s"]:
                st = None                       # gone too long: start fresh, no slide-in
            if st is None:
                st = _FaceState(OneEuro(cfg["filter_min_cutoff"], cfg["filter_beta"],
                                        cfg["filter_d_cutoff"]))
            self._state[pid] = st
            st.debug = {}
            box = head_box(p, self.kpt_conf_threshold, cfg["crop_size"])
            det = self._detect(frame.image, box, st) if box is not None else None
            if det is None:
                continue                        # filter is kept for when the face is found again
            m = measure(det, cfg["center_drop"], cfg["forehead_rise"], cfg["lip_to_nose"])
            st.width, st.last_seen = m["width"], t
            g = geometry(st.filter(m["pts"], t), cfg["hat_drop"], cfg["above_head_rise"])
            faces[pid] = Face(pid=pid, valid=True, landmarks=m["landmarks"], center=g["center"],
                              scale=g["width"], roll=g["roll"], yaw=0.0,   # 5 landmarks give no yaw
                              anchors=g["anchors"], alpha=1.0)
        return faces

    def _detect(self, image: np.ndarray, box: np.ndarray, st: _FaceState) -> np.ndarray | None:
        """Run YuNet on the head crop and return the detection (shifted to
        full-frame pixels) closest to the crop centre, i.e. the head keypoints.
        Another player's face in the same crop is further away and skipped."""
        img, off = crop(image, box)
        st.debug["box"] = box
        h, w = img.shape[:2]
        if min(h, w) < self.cfg["min_face"]:
            return None
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(img)
        if rows is None:
            return None
        rows = rows.astype(np.float64)
        rows[:, [0, 4, 6, 8, 10, 12]] += off[0]    # crop -> full-frame pixels
        rows[:, [1, 5, 7, 9, 11, 13]] += off[1]
        rows = rows[rows[:, 2] >= self.cfg["min_face"]]
        if st.width > 0:                            # implausible size jump -> not this face
            rows = rows[np.abs(rows[:, 2] / st.width - 1) <= self.cfg["max_size_jump"]]
        if len(rows) == 0:
            return None
        centre = (box[:2] + box[2:]) / 2
        dist = np.linalg.norm(rows[:, :2] + rows[:, 2:4] / 2 - centre, axis=1)
        det = rows[int(np.argmin(dist))]
        st.debug["det"] = det
        return det

    def _stub_faces(self, players: dict[int, Player]) -> dict[int, Face]:
        """Phase-1 stand-in: above_head anchor above the nose keypoint."""
        factor, default_scale = float(self.cfg["above_head_factor"]), float(self.cfg["default_scale"])
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
                scale = default_scale
            faces[pid] = Face(
                pid=pid, valid=True, landmarks=None, center=nose, scale=scale,
                roll=0.0, yaw=0.0,
                anchors={"above_head": nose - np.array([0.0, factor * scale])},
            )
        return faces

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._state.clear()
        else:
            self._state.pop(pid, None)

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Head crop (dashed), raw YuNet landmarks and score, face centre,
        scale circle and every anchor with its name."""
        for pid, st in self._state.items():
            color = player_color(pid)
            det = st.debug.get("det")
            if "box" in st.debug:
                label = f"P{pid} {det[14]:.2f}" if det is not None else f"P{pid} no face"
                draw_bbox(canvas, st.debug["box"], color, label, dashed=True)
            if det is not None:
                for x, y in det[4:14].reshape(5, 2):
                    cv2.circle(canvas, pt((x, y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        for pid, f in state.faces.items():
            color = player_color(pid)
            cv2.circle(canvas, pt(f.center), int(f.scale / 2), color, 1, cv2.LINE_AA)
            for name, a in f.anchors.items():
                cv2.drawMarker(canvas, pt(a), color, cv2.MARKER_CROSS, 14, 2)
                draw_text(canvas, name, (pt(a)[0] + 8, pt(a)[1]), color, 0.45)
