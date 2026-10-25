"""Face tracker on synthetic YuNet detections and images (no model or camera needed)."""

import cv2
import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Frame, Player, PoseDetection
from ipcvgame.face import tracker as ft
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
    return m | geometry(m["pts"], CFG["hat_drop"], CFG["above_head_rise"])


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
    tr = FaceTracker(CFG | {"model": str(model)})
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
