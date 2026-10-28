"""Face effects: sprite loading and pasting (no model or camera needed)."""

import cv2
import numpy as np
import pytest

from ipcvgame.core.config import load_config
from ipcvgame.core.types import Face
from ipcvgame.face.effects import FaceEffects, band_mask, load_sprite, paste, war_paint
from ipcvgame.scene.draw import hue_of, player_color

RED = (0, 0, 255)


def solid_sprite(w: int, h: int, color=RED) -> np.ndarray:
    s = np.zeros((h, w, 4), np.uint8)
    s[:, :, :3] = color
    s[:, :, 3] = 255
    return s


def canvas() -> np.ndarray:
    return np.zeros((100, 100, 3), np.uint8)


def test_pivot_lands_on_anchor():
    img = canvas()
    # 10x10 sprite, pivot bottom-middle at (50, 50): the sprite sits above the anchor
    paste(img, solid_sprite(10, 10), (50, 50), (5, 10), 0.0, 10)
    assert tuple(img[45, 50]) == RED
    assert img[55, 50].sum() == 0
    assert img[45, 40].sum() == 0 and img[45, 60].sum() == 0


def test_width_scales_sprite():
    img = canvas()
    paste(img, solid_sprite(10, 10), (50, 50), (5, 5), 0.0, 40)   # 4x larger
    assert tuple(img[50, 32]) == RED and tuple(img[50, 64]) == RED
    assert img[50, 25].sum() == 0 and img[50, 75].sum() == 0


def test_rotation_turns_wide_sprite_tall():
    img = canvas()
    paste(img, solid_sprite(40, 4), (50, 50), (20, 2), np.pi / 2, 40)
    assert tuple(img[35, 50]) == RED and tuple(img[65, 50]) == RED
    assert img[50, 35].sum() == 0


def test_alpha_blends_halfway():
    img = canvas()
    paste(img, solid_sprite(10, 10), (50, 50), (5, 5), 0.0, 10, alpha=0.5)
    assert 120 <= img[50, 50, 2] <= 130


def test_off_screen_is_clipped_or_skipped():
    img = canvas()
    paste(img, solid_sprite(20, 20), (0, 0), (10, 10), 0.0, 20)       # partly off-screen
    assert tuple(img[2, 2]) == RED
    before = img.copy()
    paste(img, solid_sprite(20, 20), (-100, -100), (10, 10), 0.0, 20)  # fully off-screen
    np.testing.assert_array_equal(img, before)


def test_load_sprite_rejects_missing_and_opaque(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sprite(tmp_path / "nope.png")
    rgb = tmp_path / "rgb.png"
    cv2.imwrite(str(rgb), np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(FileNotFoundError):
        load_sprite(rgb)


def face(valid=True, alpha=1.0, pid=1) -> Face:
    return Face(pid=pid, valid=valid, landmarks=None, center=np.array([50.0, 50.0]), scale=30.0,
                roll=0.0, yaw=0.0, anchors={"hat": np.array([50.0, 40.0]),
                                            "mouth": np.array([50.0, 60.0])}, alpha=alpha)


def test_effects_draw_only_visible_faces():
    fx = FaceEffects(load_config()["face"])
    img = canvas()
    fx.draw(img, {1: face(valid=False), 2: face(alpha=0.0)})
    assert img.sum() == 0
    fx.draw(img, {1: face()})
    assert img[:40].sum() > 0       # hat above its anchor
    assert img[55:65].sum() > 0     # mustache around the mouth


def test_hat_hangs_from_its_brim(tmp_path):
    """A hat PNG with transparent padding below the hat: the visible bottom
    (the brim) lands on the hat anchor, not the bottom of the image."""
    hat = solid_sprite(20, 20)
    hat[12:, :, 3] = 0                                   # bottom 8 rows transparent
    cv2.imwrite(str(tmp_path / "hat.png"), hat)
    cfg = load_config()["face"] | {"hat_sprite": str(tmp_path / "hat.png"), "hat_width": 1.0}
    fx = FaceEffects(cfg)
    img = canvas()
    f = face()
    f.anchors = {"hat": np.array([50.0, 50.0])}
    f.scale = 20.0                                       # sprite drawn at its own size
    fx.draw(img, {1: f})
    assert tuple(img[48, 50]) == RED                     # visible hat just above the anchor
    assert img[52, 50].sum() == 0                        # nothing below it


def hues(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)[:, :, 0].astype(int)


def test_hat_band_takes_the_team_colour():
    """Only the red band of hat.png changes: P1 keeps red, P2's band turns
    to the P2 hue; the straw and the outline stay as they are."""
    fx = FaceEffects(load_config()["face"])
    band = fx.band
    assert 0.05 < band[fx.hat[:, :, 3] > 0].mean() < 0.3   # a stripe, not the whole hat
    np.testing.assert_array_equal(fx.team_hat(1), fx.hat)
    p2 = fx.team_hat(2)
    d = np.abs(hues(p2)[band] - hue_of(player_color(2)))
    assert np.median(np.minimum(d, 180 - d)) <= 4
    np.testing.assert_array_equal(p2[~band], fx.hat[~band])


def test_band_mask_wraps_around_red():
    px = np.zeros((1, 2, 4), np.uint8)
    px[0, 0] = (40, 30, 200, 255)                        # hue ~177: red just below 180
    px[0, 1] = (40, 150, 200, 255)                       # orange
    assert band_mask(px, 0, 8, 80, 100).tolist() == [[True, False]]


def cheek_face(yaw=0.0, roll=0.0) -> Face:
    f = face()
    f.yaw, f.roll = yaw, roll
    f.anchors = {"left_cheek": np.array([30.0, 50.0]), "right_cheek": np.array([70.0, 50.0])}
    return f


def stripe_length(poly: np.ndarray) -> float:
    return float(np.linalg.norm(poly[1] - poly[0]))


def test_war_paint_two_stripes_per_cheek():
    polys = war_paint(cheek_face(), 0.2, 0.05, 0.1)
    assert len(polys) == 4
    np.testing.assert_allclose(np.mean([p.mean(0) for p in polys[:2]], axis=0), (30, 50))
    assert all(stripe_length(p) == pytest.approx(0.2 * 30) for p in polys)


def test_war_paint_shortens_the_cheek_turned_away():
    yaw = np.radians(40)                                 # turned to the image right
    polys = war_paint(cheek_face(yaw=yaw), 0.2, 0.05, 0.1)
    assert stripe_length(polys[0]) == pytest.approx(0.2 * 30)            # left cheek: full
    assert stripe_length(polys[2]) == pytest.approx(0.2 * 30 * np.cos(yaw))
    polys = war_paint(cheek_face(yaw=-yaw), 0.2, 0.05, 0.1)
    assert stripe_length(polys[0]) < stripe_length(polys[2])


def test_war_paint_follows_roll():
    p = war_paint(cheek_face(roll=np.pi / 2), 0.2, 0.05, 0.1)[0]
    d = p[1] - p[0]
    assert abs(d[0]) < 1e-9 and d[1] > 0                 # head on its side: stripes vertical


@pytest.mark.parametrize("pid", [1, 2])
def test_war_paint_drawn_in_the_team_colour(pid):
    fx = FaceEffects(load_config()["face"])
    img = canvas()
    f = cheek_face()
    f.pid = pid
    fx.draw(img, {pid: f})
    px = img[50 - 1, 30]                                 # upper stripe of the left cheek
    assert px.sum() > 0
    assert abs(hue_of(tuple(int(v) for v in px)) - hue_of(player_color(pid))) <= 2
