"""Face tracker geometry on synthetic YuNet detections (no model or camera needed)."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Player, PoseDetection
from ipcvgame.face.tracker import FaceTracker, geometry, head_box, measure

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
