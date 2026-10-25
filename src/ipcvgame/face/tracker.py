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


def flow_points(pts: np.ndarray, cfg: dict) -> np.ndarray:
    """Grid of points on the face (eyes to mouth) for optical flow to follow,
    from the 4 key points; grid offsets are in face widths from the centre."""
    left, right = pts[0], pts[1]
    d = right - left
    w = float(np.linalg.norm(d))
    u = d / max(w, 1e-6)                                    # along the eye line
    down = np.array([-u[1], u[0]])
    c = (left + right) / 2
    n = int(cfg["flow_grid_n"])
    return np.array([c + (gx * u + gy * down) * w
                     for gx in np.linspace(*cfg["flow_grid_x"], n)
                     for gy in np.linspace(*cfg["flow_grid_y"], n)])


def estimate_motion(prev_gray: np.ndarray, gray: np.ndarray, pts: np.ndarray, cfg: dict) -> np.ndarray | None:
    """How the face moved between two frames: Lucas-Kanade optical flow on
    `pts`, then a similarity transform (shift + rotation + scale) fitted with
    RANSAC, so a few points that slid onto the background don't matter.
    Returns the 2x3 matrix, or None when the fit is not trustworthy."""
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    iters, eps = cfg["flow_criteria"]
    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None, winSize=(cfg["flow_win"], cfg["flow_win"]),
        maxLevel=cfg["flow_levels"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iters), float(eps)))
    good = status.ravel() == 1
    if good.sum() < cfg["flow_min_points"]:
        return None
    M, inliers = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                             ransacReprojThreshold=cfg["flow_ransac_px"])
    if M is None or inliers.sum() < cfg["flow_min_inliers"]:
        return None
    lo, hi = cfg["flow_scale_range"]
    if not lo < np.hypot(M[0, 0], M[1, 0]) < hi:            # implausible zoom between two frames
        return None
    return M


def fade(blind_s: float, hold_s: float, lost_after_s: float) -> float:
    """Effect opacity after blind_s seconds without an update (detection or
    flow): fully visible for hold_s, then a linear fade to 0 at lost_after_s."""
    if blind_s <= hold_s:
        return 1.0
    return max(0.0, 1.0 - (blind_s - hold_s) / (lost_after_s - hold_s))


@dataclass
class _FaceState:
    """One player's face state between frames."""
    filter: OneEuro                     # on the 4 key points
    kp: np.ndarray | None = None        # current 4 key points (px), unsmoothed; moved by flow
    lk: np.ndarray | None = None        # flow points on the face (px)
    landmarks: np.ndarray | None = None # 5 YuNet landmarks (px); moved by flow
    geo: dict | None = None             # last geometry() output, shown while held
    width: float = 0.0                  # current face width (px), for the size-jump check
    last_det: float = 0.0               # Frame.t of the last detection
    last_update: float = 0.0            # Frame.t of the last detection or flow update
    source: str = ""                    # "detect" | "flow" | "hold", for debug and logs
    debug: dict = field(default_factory=dict)   # crop box and raw detection, for draw_debug


class FaceTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = cfg.get("backend", "yunet")
        self.kpt_conf_threshold = float(cfg["kpt_conf_threshold"])
        self._state: dict[int, _FaceState] = {}
        self._prev_gray: np.ndarray | None = None
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
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)   # raw frame: effects not drawn yet
        prev_gray, self._prev_gray = self._prev_gray, gray
        for pid in [pid for pid in self._state if pid not in players]:
            del self._state[pid]
        faces: dict[int, Face] = {}
        for pid, p in players.items():
            st = self._state.get(pid)
            if st is not None and t - st.last_update > cfg["lost_after_s"]:
                st = None                       # gone too long: start fresh, no slide-in
            if st is None:
                st = _FaceState(OneEuro(cfg["filter_min_cutoff"], cfg["filter_beta"],
                                        cfg["filter_d_cutoff"]))
            self._state[pid] = st
            st.debug = {}
            box = head_box(p, self.kpt_conf_threshold, cfg["crop_size"])
            det = None
            if box is not None:
                st.debug["box"] = box
                det = self._detect(frame.image, box, st)
            if det is not None:
                st.debug["det"] = det           # raw detection, for draw_debug and log_fields
                m = measure(det, cfg["center_drop"], cfg["forehead_rise"], cfg["lip_to_nose"])
                st.kp, st.landmarks, st.lk = m["pts"], m["landmarks"], flow_points(m["pts"], cfg)
                st.last_det = st.last_update = t
                st.source = "detect"
            elif (st.lk is not None and prev_gray is not None
                  and t - st.last_det <= cfg["flow_max_s"]):
                # No detection: follow the face with optical flow instead of freezing
                M = estimate_motion(prev_gray, gray, st.lk, cfg)
                if M is not None:
                    st.kp, st.lk, st.landmarks = (P @ M[:, :2].T + M[:, 2]
                                                  for P in (st.kp, st.lk, st.landmarks))
                    st.last_update = t
                    st.source = "flow"
            if st.kp is None:
                continue                        # face not found yet
            if st.last_update == t:             # new key points this frame: smooth them
                st.width = float(np.linalg.norm(st.kp[1] - st.kp[0]))
                st.geo = geometry(st.filter(st.kp, t), cfg["hat_drop"], cfg["above_head_rise"])
            else:
                st.source = "hold"              # nothing new: show the last position, fading
            alpha = fade(t - st.last_update, cfg["hold_s"], cfg["lost_after_s"])
            g = st.geo
            faces[pid] = Face(pid=pid, valid=alpha > 0, landmarks=st.landmarks, center=g["center"],
                              scale=g["width"], roll=g["roll"], yaw=0.0,   # 5 landmarks give no yaw
                              anchors=g["anchors"], alpha=alpha)
        return faces

    def _detect(self, image: np.ndarray, box: np.ndarray, st: _FaceState) -> np.ndarray | None:
        """Run YuNet on the head crop and return the detection (shifted to
        full-frame pixels) closest to the crop centre, i.e. the head keypoints.
        Another player's face in the same crop is further away and skipped."""
        img, off = crop(image, box)
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
        return rows[int(np.argmin(dist))]

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

    def log_fields(self, state: FrameState) -> dict:
        """Per-player CSV columns face_p<pid>_<field>, read by tools/eval_face.py.

        source: detect | flow | hold, or empty while no face was found.
        mouth_x/y is the smoothed mouth anchor; mouth_raw_x/y the unsmoothed
        upper-lip key point, only on detect frames (it is a measurement there).
        """
        row: dict = {}
        for pid, st in sorted(self._state.items()):
            pre = f"face_p{pid}_"
            f = state.faces.get(pid)
            row[pre + "source"] = st.source if f is not None else ""
            if f is None:
                continue
            det = st.debug.get("det")
            row[pre + "score"] = float(det[14]) if st.source == "detect" and det is not None else ""
            row[pre + "alpha"] = float(f.alpha)
            row[pre + "scale"] = float(f.scale)
            row[pre + "roll_deg"] = float(np.degrees(f.roll))
            row[pre + "mouth_x"], row[pre + "mouth_y"] = (float(v) for v in f.anchors["mouth"])
            if st.source == "detect":
                row[pre + "mouth_raw_x"], row[pre + "mouth_raw_y"] = (float(v) for v in st.kp[3])
            if "box" in st.debug:
                row[pre + "crop_px"] = float(st.debug["box"][2] - st.debug["box"][0])
        return row

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self._state.clear()
        else:
            self._state.pop(pid, None)

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Head crop (dashed) labelled with the source (detect + YuNet score,
        flow, hold) and alpha; landmarks; flow points while flowing; face
        centre, scale circle and every anchor with its name."""
        for pid, st in self._state.items():
            color = player_color(pid)
            det = st.debug.get("det")
            f = state.faces.get(pid)
            if st.source == "detect" and det is not None:
                label = f"P{pid} detect {det[14]:.2f}"
            else:
                label = f"P{pid} {st.source or 'no face'}"
            if f is not None:
                label += f" a={f.alpha:.2f}"
            if "box" in st.debug:
                draw_bbox(canvas, st.debug["box"], color, label, dashed=True)
            elif f is not None:                 # no crop (player not detected): label at the face
                draw_text(canvas, label, pt(f.anchors["above_head"]), color, 0.5)
            if st.source == "flow" and st.lk is not None:
                for p in st.lk:
                    cv2.circle(canvas, pt(p), 2, color, -1, cv2.LINE_AA)
            if st.landmarks is not None and st.source != "hold":
                for p in st.landmarks:
                    cv2.circle(canvas, pt(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
        for pid, f in state.faces.items():
            color = player_color(pid)
            cv2.circle(canvas, pt(f.center), int(f.scale / 2), color, 1, cv2.LINE_AA)
            for name, a in f.anchors.items():
                cv2.drawMarker(canvas, pt(a), color, cv2.MARKER_CROSS, 14, 2)
                draw_text(canvas, name, (pt(a)[0] + 8, pt(a)[1]), color, 0.45)
