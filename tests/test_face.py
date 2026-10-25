"""Face tracker on synthetic YuNet detections and images (no model or camera needed)."""

import cv2
import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Frame, FrameState, Player, PoseDetection
from ipcvgame.face import tracker as ft
from ipcvgame.face.head_pose import (
    anchor_points_mm,
    camera_matrix,
    head_pose,
    mediapipe_angles,
    pose_angles,
    rotation,
    translation,
)
from ipcvgame.face.tracker import (
    FaceTracker,
    estimate_motion,
    fade,
    flow_points,
    geometry,
    head_box,
    measure,
)

CFG = load_config()["face"]
# Tracker tests replace YuNet and need no MediaPipe model file either
NO_LANDMARKER = {"landmarker": False}
PIVOT = np.array([110.0, 110.0])
# Upright face: right eye, left eye, nose, right and left mouth corner (pixels)
POINTS = np.array([[90, 100], [130, 100], [110, 118], [96, 138], [124, 138]], dtype=float)


def yunet_row(roll_deg: float = 0.0, swap_eyes: bool = False) -> np.ndarray:
    """A YuNet output row for the face above, rotated by roll_deg about PIVOT."""
    a = np.radians(roll_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (POINTS - PIVOT) @ R.T + PIVOT
    if swap_eyes:
        pts[[0, 1]] = pts[[1, 0]]
    return np.concatenate([[70, 70, 80, 90], pts.ravel(), [0.95]])


def run(row: np.ndarray) -> dict:
    m = measure(row, CFG["center_drop"], CFG["forehead_rise"], CFG["lip_to_nose"])
    return m | geometry(m["pts"], CFG["hat_rise"], CFG["above_head_rise"])


def test_upright_face():
    g = run(yunet_row())
    assert g["roll"] == pytest.approx(0.0)
    assert g["width"] == pytest.approx(80.0)              # sides are one box width apart
    a = g["anchors"]
    assert a["above_head"][1] < a["hat"][1] < 100 < a["mouth"][1]   # top to bottom, eyes at y=100
    assert a["forehead"][0] == pytest.approx(110.0) and a["hat"][0] == pytest.approx(110.0)
    assert 118 < a["mouth"][1] < 138                      # between nose and mouth
    np.testing.assert_allclose(g["landmarks"], POINTS)


@pytest.mark.parametrize("roll_deg", [10.0, -25.0])
def test_roll_follows_head_tilt(roll_deg):
    g, up = run(yunet_row(roll_deg)), run(yunet_row())
    assert np.degrees(g["roll"]) == pytest.approx(roll_deg)
    assert g["width"] == pytest.approx(up["width"])
    # Anchors turn with the head: same offset from the centre, rotated by the roll
    a = np.radians(roll_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    for name in ("forehead", "hat", "mouth", "above_head"):
        np.testing.assert_allclose(g["anchors"][name] - g["center"],
                                   R @ (up["anchors"][name] - up["center"]), atol=1e-9)


def test_eye_order_does_not_matter():
    a, b = run(yunet_row(10.0)), run(yunet_row(10.0, swap_eyes=True))
    for name in a["anchors"]:
        np.testing.assert_allclose(a["anchors"][name], b["anchors"][name])


def player(conf: float = 0.9, shoulders: bool = True) -> Player:
    """Upright player: head keypoints around (300, 100), shoulders 100 px apart."""
    kpts = np.zeros((sk.K, 2))
    kpts[sk.NOSE], kpts[sk.L_EYE], kpts[sk.R_EYE] = (300, 100), (290, 90), (310, 90)
    kpts[sk.L_EAR], kpts[sk.R_EAR] = (280, 95), (320, 95)
    kpts[sk.L_SHOULDER], kpts[sk.R_SHOULDER] = (250, 160), (350, 160)
    c = np.full(sk.K, conf)
    if not shoulders:
        c[sk.R_SHOULDER] = 0.1
    det = PoseDetection(bbox=np.array([240.0, 60, 360, 400]), kpts=kpts, conf=c, score=0.9)
    return Player(pid=1, status="active", det=det, bbox=det.bbox, pos=np.zeros(2), reentered=False)


def test_head_box_from_head_and_shoulders():
    box = head_box(player(), 0.5, 1.5)
    np.testing.assert_allclose((box[:2] + box[2:]) / 2, (300, 94))   # mean of the 5 head keypoints
    np.testing.assert_allclose(box[2:] - box[:2], (150, 150))        # 1.5 shoulder widths


def test_head_box_falls_back_to_person_box():
    box = head_box(player(shoulders=False), 0.5, 1.5)
    np.testing.assert_allclose(box[2:] - box[:2], (180, 180))        # 1.5 box widths (120 px)
    box = head_box(player(conf=0.1), 0.5, 1.5)                       # no keypoint trusted
    np.testing.assert_allclose((box[:2] + box[2:]) / 2, (300, 60 + 90))   # top of the person box


def test_head_box_none_without_detection():
    p = player()
    p.det = None
    assert head_box(p, 0.5, 1.5) is None


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        FaceTracker(CFG | {"backend": "nope"})


def test_missing_model_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        FaceTracker(CFG | {"model": str(tmp_path / "missing.onnx")})


def test_fade_holds_then_fades_to_zero():
    assert fade(0.05, 0.1, 1.0) == 1.0
    assert fade(0.55, 0.1, 1.0) == pytest.approx(0.5)
    assert fade(1.0, 0.1, 1.0) == 0.0 and fade(2.0, 0.1, 1.0) == 0.0


def test_flow_points_cover_the_face_and_turn_with_it():
    up, tilted = run(yunet_row()), run(yunet_row(30.0))
    n = CFG["flow_grid_n"]
    pts_up, pts_tilted = flow_points(up["pts"], CFG), flow_points(tilted["pts"], CFG)
    assert pts_up.shape == (n * n, 2)
    np.testing.assert_allclose(pts_up.mean(axis=0)[0], up["center"][0])
    assert pts_up[:, 1].min() < 100 < pts_up[:, 1].max()       # from above the eyes (y=100) downward
    a = np.radians(30.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    np.testing.assert_allclose(pts_tilted - tilted["center"], (pts_up - up["center"]) @ R.T, atol=1e-9)


def texture(h: int = 240, w: int = 320) -> np.ndarray:
    """Blurred noise: texture optical flow can follow."""
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8), (0, 0), 2)


def shifted(img: np.ndarray, dx: float, dy: float, scale: float = 1.0) -> np.ndarray:
    M = cv2.getRotationMatrix2D((160, 120), 0, scale)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_estimate_motion_finds_a_shift():
    img, pts = texture(), flow_points(run(yunet_row())["pts"] + (50, 10), CFG)
    M = estimate_motion(img, shifted(img, 4, -2), pts, CFG)
    np.testing.assert_allclose(M, [[1, 0, 4], [0, 1, -2]], atol=0.1)


def test_estimate_motion_rejects_zoom_outside_range():
    img, pts = texture(), flow_points(run(yunet_row())["pts"] + (50, 10), CFG)
    zoomed = shifted(img, 0, 0, scale=1.2)
    M = estimate_motion(img, zoomed, pts, CFG)                        # inside [0.7, 1.4]
    assert np.hypot(M[0, 0], M[1, 0]) == pytest.approx(1.2, abs=0.02)
    assert estimate_motion(img, zoomed, pts, CFG | {"flow_scale_range": [0.9, 1.1]}) is None


@pytest.fixture
def scripted_tracker(tmp_path, monkeypatch):
    """A yunet-backend FaceTracker whose detector is scripted per frame."""
    model = tmp_path / "fake.onnx"
    model.touch()
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *a, **k: None)
    tr = FaceTracker(CFG | NO_LANDMARKER | {"model": str(model)})
    script = {}
    monkeypatch.setattr(tr, "_detect", lambda image, box, st: script.pop(len(script) and min(script), None))
    return tr, script


def test_flow_follows_face_when_detection_misses(scripted_tracker):
    tr, script = scripted_tracker
    img = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
    script[0] = yunet_row() + np.r_[50, 10, 0, 0, [50, 10] * 5, 0]   # face around (160, 120)
    p = {1: player()}
    f0 = tr.update(Frame(img, 0, 0.0), p)[1]
    kp0 = tr._state[1].kp.copy()
    f1 = tr.update(Frame(shifted(img, 5, 3), 1, 1 / 30), p)[1]       # no detection
    assert tr._state[1].source == "flow" and f1.alpha == 1.0 and f1.valid
    np.testing.assert_allclose(tr._state[1].kp - kp0, np.tile([5, 3], (4, 1)), atol=0.3)
    assert f1.center[0] > f0.center[0] and f1.center[1] > f0.center[1]


def test_hold_then_fade_then_gone(scripted_tracker, monkeypatch):
    tr, script = scripted_tracker
    monkeypatch.setattr(ft, "estimate_motion", lambda *a: None)      # flow never succeeds
    img = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
    script[0] = yunet_row() + np.r_[50, 10, 0, 0, [50, 10] * 5, 0]
    p = {1: player()}
    f0 = tr.update(Frame(img, 0, 0.0), p)[1]
    f_hold = tr.update(Frame(img, 1, 0.05), p)[1]
    assert tr._state[1].source == "hold" and f_hold.alpha == 1.0
    np.testing.assert_allclose(f_hold.anchors["hat"], f0.anchors["hat"])   # frozen in place
    f_fade = tr.update(Frame(img, 2, 0.55), p)[1]
    assert f_fade.alpha == pytest.approx(0.5) and f_fade.valid
    assert 1 not in tr.update(Frame(img, 3, 1.2), p)                  # past lost_after_s


def test_log_fields_raw_mouth_only_on_detect(scripted_tracker):
    tr, script = scripted_tracker
    img = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
    script[0] = yunet_row() + np.r_[50, 10, 0, 0, [50, 10] * 5, 0]
    p = {1: player()}
    f0 = tr.update(Frame(img, 0, 0.0), p)
    row0 = tr.log_fields(FrameState(Frame(img, 0, 0.0), p, {}, f0, [], {}))
    assert row0["face_p1_source"] == "detect" and row0["face_p1_score"] == pytest.approx(0.95)
    assert {"face_p1_mouth_raw_x", "face_p1_crop_px", "face_p1_roll_deg"} <= set(row0)
    f1 = tr.update(Frame(shifted(img, 5, 3), 1, 1 / 30), p)
    row1 = tr.log_fields(FrameState(Frame(img, 1, 1 / 30), p, {}, f1, [], {}))
    assert row1["face_p1_source"] == "flow" and "face_p1_mouth_raw_x" not in row1


class CentreFaceDetector:
    """Stands in for YuNet: always 'finds' a face in the middle of its input,
    sized relative to the input, like a real face in a head crop."""

    def setInputSize(self, size):
        self.w, self.h = size

    def detect(self, img):
        assert img.shape[1] == self.w and img.shape[0] == self.h
        w, h = self.w, self.h
        pts = [0.40 * w, 0.45 * h, 0.60 * w, 0.45 * h, 0.50 * w, 0.55 * h,
               0.42 * w, 0.65 * h, 0.58 * w, 0.65 * h]
        return 1, np.array([[0.3 * w, 0.3 * h, 0.4 * w, 0.4 * h, *pts, 0.9]], np.float32)


@pytest.mark.parametrize("limit", [None, 64])
def test_shrunk_crop_maps_back_to_frame_pixels(tmp_path, monkeypatch, limit):
    model = tmp_path / "fake.onnx"
    model.touch()
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *a, **k: CentreFaceDetector())
    tr = FaceTracker(CFG | NO_LANDMARKER | {"model": str(model), "crop_max_px": limit})
    box = np.array([100.0, 50, 300, 250])                   # 200 px crop
    det = tr._detect(np.zeros((480, 640, 3), np.uint8), box, ft._FaceState(None, None))
    if limit is not None:
        assert tr.detector.w == 64                          # YuNet saw the shrunk crop
    # Same face in full-frame pixels either way: box at 30-70% of the crop
    np.testing.assert_allclose(det[:4], [160, 110, 80, 80], atol=0.5)
    np.testing.assert_allclose(det[4:6], [180, 140], atol=0.5)


# ---------- head pose (step 9) ----------
MODEL = np.array([CFG["head_model_mm"][k] for k in
                  ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")], dtype=float)
K = camera_matrix(1280, 720, CFG["focal_per_width"])


def head_rvec(chin_up_deg: float = 0.0, turn_right_deg: float = 0.0) -> np.ndarray:
    """Rotation of a head that lifts its chin (the forehead moves away from the
    camera) and then turns towards the image right (the nose moves right)."""
    a, b = np.radians(-chin_up_deg), np.radians(-turn_right_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return cv2.Rodrigues(Ry @ Rx)[0]


def seen(rvec: np.ndarray, pts_mm: np.ndarray, tvec=(0.0, 0.0, 1500.0)) -> np.ndarray:
    """Project head points (mm) for a head 1.5 m from the camera, a playing distance."""
    return cv2.projectPoints(pts_mm, rvec, np.array(tvec), K, None)[0].reshape(-1, 2)


def row_from(landmarks: np.ndarray) -> np.ndarray:
    """A YuNet row for these landmarks, box width box_per_eye eye distances."""
    eye = np.linalg.norm(landmarks[1] - landmarks[0])
    return np.concatenate([[0, 0, CFG["box_per_eye"] * eye, 0], landmarks.ravel(), [0.9]])


def anchors_2d(landmarks: np.ndarray) -> dict:
    return run(row_from(landmarks))["anchors"]


def anchors_3d(landmarks: np.ndarray) -> dict:
    rvec, tvec = head_pose(landmarks, MODEL, K)
    A = anchor_points_mm(CFG, MODEL)
    return dict(zip(A, seen(rvec, np.array(list(A.values())), tvec.ravel()), strict=True))


@pytest.mark.parametrize("chin_up, turn_right", [(0, 0), (20, 0), (-20, 0), (0, 25), (15, -20)])
def test_head_pose_recovers_angles(chin_up, turn_right):
    lm = seen(head_rvec(chin_up, turn_right), MODEL)
    yaw, pitch = pose_angles(head_pose(lm, MODEL, K)[0])
    assert np.degrees(pitch) == pytest.approx(chin_up, abs=0.5)
    assert np.degrees(yaw) == pytest.approx(turn_right, abs=0.5)


def test_head_pose_ignores_landmark_labels():
    lm = seen(head_rvec(10, 15), MODEL)
    swapped = lm[[1, 0, 2, 4, 3]]                  # mirrored frame: eyes and mouth corners swap
    np.testing.assert_allclose(head_pose(swapped, MODEL, K)[0], head_pose(lm, MODEL, K)[0], atol=1e-6)


def test_frontal_3d_anchors_match_2d_anchors():
    """Facing the camera, the 3D anchors land where the 2D offsets put them,
    except for perspective: the hat points are hat_depth_mm deeper, so they
    project closer to the eye line by a factor Z / (Z + depth)."""
    lm = seen(head_rvec(), MODEL)
    a2, a3 = anchors_2d(lm), anchors_3d(lm)
    width = CFG["box_per_eye"] * np.linalg.norm(lm[1] - lm[0])
    eyes_y = lm[0, 1]
    shrink = 1 - 1500 / (1500 + CFG["hat_depth_mm"])
    np.testing.assert_allclose(a3["forehead"], a2["forehead"], atol=1e-6 * width)
    for name in ("hat", "above_head"):
        expected = (eyes_y - a2[name][1]) * shrink              # px closer to the eyes
        assert a3[name][1] - a2[name][1] == pytest.approx(expected, rel=0.02)
        assert a3[name][0] == pytest.approx(a2[name][0])


@pytest.mark.parametrize("chin_up", [25, -25])
def test_hat_follows_head_tilt(chin_up):
    """Chin up: the top of the head turns away, so the hat comes down towards
    the eyes; the 2D offsets kept it at a fixed height ("flies higher").
    Chin down: the top of the head turns into view, so the hat goes up."""
    lm = seen(head_rvec(chin_up), MODEL)
    eyes_y = (lm[0, 1] + lm[1, 1]) / 2
    height_2d = eyes_y - anchors_2d(lm)["hat"][1]
    height_3d = eyes_y - anchors_3d(lm)["hat"][1]
    assert (height_3d < height_2d) if chin_up > 0 else (height_3d > height_2d)


# ---------- MediaPipe pose and blendshapes (step 10) ----------
def mp_matrix(chin_up_deg: float = 0.0, turn_right_deg: float = 0.0) -> np.ndarray:
    """A MediaPipe facial transformation matrix (face frame: y up, z towards
    the camera). Chin up turns the face's up axis away from the camera; a
    turn to the image right moves the nose (+z) towards +x."""
    a, b = np.radians(-chin_up_deg), np.radians(turn_right_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    M = np.eye(4)
    M[:3, :3] = Ry @ Rx
    return M


@pytest.mark.parametrize("chin_up, turn_right", [(0, 0), (33, 0), (-20, 0), (0, 25), (15, -20)])
def test_mediapipe_angles_in_our_convention(chin_up, turn_right):
    yaw, pitch = mediapipe_angles(mp_matrix(chin_up, turn_right))
    assert np.degrees(pitch) == pytest.approx(chin_up, abs=1e-6)
    assert np.degrees(yaw) == pytest.approx(turn_right, abs=1e-6)


@pytest.mark.parametrize("yaw, pitch, roll", [(0, 0, 0), (20, -10, 0), (-30, 25, 15)])
def test_rotation_inverts_pose_angles(yaw, pitch, roll):
    R = rotation(*np.radians([yaw, pitch, roll]))
    np.testing.assert_allclose(np.degrees(pose_angles(cv2.Rodrigues(R)[0])), [yaw, pitch], atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_translation_for_known_rotation():
    R, t = rotation(*np.radians([15, 20, 5])), np.array([40.0, -25.0, 1500.0])
    lm = seen(cv2.Rodrigues(R)[0], MODEL, t)
    np.testing.assert_allclose(translation(R, MODEL, lm, K), t, rtol=1e-6)


def test_ears_help_pnp_measure_tilt():
    """YuNet's landmarks stay near-frontal at strong tilts. With the real
    nose and mouth replaced by frontal ones (what YuNet reports), the 5-point
    fit misses most of a 33 deg chin-up; adding the ears recovers much more."""
    ears = np.array([CFG["head_model_mm"]["left_ear"], CFG["head_model_mm"]["right_ear"]])
    tilted = seen(head_rvec(33), np.vstack([MODEL, ears]))
    frontal = seen(head_rvec(), MODEL)
    biased = tilted[:5].copy()
    eye_mid, eye_mid_f = tilted[:2].mean(0), frontal[:2].mean(0)
    biased[2:] = frontal[2:] - eye_mid_f + eye_mid                # frontal layout under the real eyes
    pitch5 = np.degrees(pose_angles(head_pose(biased, MODEL, K)[0])[1])
    pitch7 = np.degrees(pose_angles(head_pose(biased, MODEL, K, tilted[5:], ears)[0])[1])
    assert pitch5 < 10 < pitch7


class FakeLandmarker:
    """Stands in for MediaPipe: a chin-up face that opens its mouth."""
    calls = 0

    def __init__(self, cfg):
        pass

    def __call__(self, crop_bgr):
        FakeLandmarker.calls += 1
        return {"blendshapes": {"jawOpen": 0.8, "eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.1},
                "matrix": mp_matrix(chin_up_deg=30)}


def two_players() -> dict:
    a, b = player(), player()
    b.pid = 2
    b.det = PoseDetection(bbox=b.det.bbox + [200, 0, 200, 0], kpts=b.det.kpts + [200, 0],
                          conf=b.det.conf, score=0.9)
    return {1: a, 2: b}


def test_landmarker_takes_turns_and_drives_pose(tmp_path, monkeypatch):
    model = tmp_path / "fake.onnx"
    model.touch()
    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *a, **k: None)
    monkeypatch.setattr(ft, "FaceLandmarker", FakeLandmarker)
    FakeLandmarker.calls = 0
    tr = FaceTracker(CFG | {"model": str(model), "landmarker": True, "landmarker_per_frame": 1})
    rows = {1: yunet_row() + np.r_[190, 0, 0, 0, [190, 0] * 5, 0],   # a face at each player's head
            2: yunet_row() + np.r_[390, 0, 0, 0, [390, 0] * 5, 0]}
    monkeypatch.setattr(tr, "_detect", lambda image, box, st: rows[1 if box[0] < 400 else 2].copy())
    img, ps = np.zeros((480, 640, 3), np.uint8), two_players()
    measured = []
    for i in range(4):
        faces = tr.update(Frame(img, i, i / 30), ps)
        measured.append(sorted(pid for pid, st in tr._state.items() if st.mp_t == i / 30))
    assert FakeLandmarker.calls == 4 and measured == [[1], [2], [1], [2]]      # one player per frame, in turn
    for f in faces.values():
        assert f.blendshapes["jawOpen"] == pytest.approx(0.8)
        assert tr._state[f.pid].pose_src == "mediapipe"
        assert np.degrees(tr._state[f.pid].pitch) > 10                          # smoothed towards 30


@pytest.mark.parametrize("bad", ["nan", "collapsed"])
def test_head_pose_rejects_degenerate_points(bad):
    """Regression: SQPnP raised an assertion (crash in the live game) for
    points that coincide or are not finite; head_pose now returns None."""
    lm = seen(head_rvec(), MODEL)
    if bad == "nan":
        lm[3] = np.nan
    else:
        lm = lm.mean(axis=0) + (lm - lm.mean(axis=0)) * 0.01   # a face ~1 px wide
    assert head_pose(lm, MODEL, K) is None


def test_flow_stops_before_face_shrinks_below_min_face(scripted_tracker, monkeypatch):
    tr, script = scripted_tracker
    shrink = np.array([[0.75, 0, 0], [0, 0.75, 0]])                    # 25% smaller every frame
    monkeypatch.setattr(ft, "estimate_motion", lambda *a: shrink)
    img = cv2.cvtColor(texture(), cv2.COLOR_GRAY2BGR)
    script[0] = yunet_row() + np.r_[50, 10, 0, 0, [50, 10] * 5, 0]  # 80 px face
    p = {1: player()}
    sources = []
    for i in range(8):
        tr.update(Frame(img, i, i / 30), p)
        sources.append(tr._state[1].source)
    assert tr._state[1].width >= CFG["min_face"]
    assert sources[:5] == ["detect", "flow", "flow", "flow", "flow"] and sources[-1] == "hold"
