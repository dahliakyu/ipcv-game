"""Face effects: sprite loading and pasting (no model or camera needed)."""

import cv2
import numpy as np
import pytest

from ipcvgame.core.config import load_config
from ipcvgame.core.types import Face
from ipcvgame.face.effects import (
    FaceEffects,
    band_mask,
    bulge_map,
    load_sprite,
    paste,
    war_paint,
    warp_bulge,
)
from ipcvgame.interaction.game import Hit
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


# ---------- Hit reaction (merge_log.md, "Face effects", step 3) ----------
def hit_face(pid=2) -> Face:
    f = face(pid=pid)
    f.anchors = {"hat": np.array([50.0, 40.0])}
    return f


def knocked(t0=10.0, direction=1) -> FaceEffects:
    fx = FaceEffects(load_config()["face"])
    fx.update([Hit(2, 1, direction, 10)], {2: hit_face()}, t0)
    return fx


def test_hit_knocks_the_hat_off_the_way_the_attack_went():
    fx = knocked(direction=1)
    a = fx._anim[2]
    pos0, _, alpha0 = fx.flying_hat(a, 10.0)
    np.testing.assert_allclose(pos0, (50, 40))
    pos, angle, alpha = fx.flying_hat(a, 10.2)
    assert pos[0] > 50 and pos[1] < 40                   # to the right and up at first
    assert angle > 0 and alpha == alpha0 == 1.0          # spinning the same way, still visible
    assert fx.flying_hat(a, 10.45)[2] < 0.6              # fading in the second half
    pos_l, angle_l, _ = fx.flying_hat(knocked(direction=-1)._anim[2], 10.2)
    assert pos_l[0] < 50 and angle_l < 0
    assert fx.flying_hat(a, 10.01 + fx.cfg["hat_fly_s"]) is None


def test_hat_away_then_pops_back():
    fx = knocked()
    a, fly, pop = fx._anim[2], fx.cfg["hat_fly_s"], fx.cfg["hat_pop_s"]
    assert fx.hat_size(a, 10.1) == 0.0
    assert 0.0 < fx.hat_size(a, 10.0 + fly + pop / 2) < 1.0
    assert fx.hat_size(a, 10.0 + fly + pop) == 1.0
    assert fx.hat_size(None, 10.0) == 1.0


def test_second_hit_restarts_the_flash_but_throws_no_second_hat():
    fx = knocked()
    fx.update([Hit(2, 1, -1, 10)], {2: hit_face()}, 10.3)
    a = fx._anim[2]
    assert a.knock_t == 10.0 and a.knock_dir == 1        # the first hat keeps flying
    assert a.flash_t == 10.3
    fx.update([Hit(2, 1, -1, 10)], {2: hit_face()}, 10.0 + fx.knock_s)
    assert a.knock_t == 10.0 + fx.knock_s                # back on the head: knocked off again


def test_flash_white_then_attacker_colour():
    fx = knocked()
    a, s = fx._anim[2], fx.cfg["flash_s"]
    color, alpha = fx.flash(a, 10.0)
    assert color == (255, 255, 255) and alpha == pytest.approx(fx.cfg["flash_opacity"])
    color, alpha = fx.flash(a, 10.0 + 0.75 * s)
    assert color == player_color(1) and alpha < fx.cfg["flash_opacity"] / 2
    assert fx.flash(a, 10.0 + s) is None


def test_no_reaction_without_a_visible_face():
    fx = FaceEffects(load_config()["face"])
    fx.update([Hit(2, 1, 1, 10)], {2: face(pid=2, alpha=0.0)}, 10.0)
    fx.update([Hit(3, 1, 1, 10)], {}, 10.0)
    assert fx._anim == {}


def test_draw_shows_the_flying_hat_not_the_head_hat():
    fx = knocked()
    img = canvas()
    fx.draw(img, {2: hit_face()}, 10.0 + fx.cfg["flash_s"] + 0.01)
    assert img[30:40, 45:55].sum() == 0                  # no hat above the anchor
    assert img[:, 60:].sum() > 0                         # the hat flew to the right
    still = canvas()
    fx.draw(still, {2: hit_face()})                      # no t (debug views): plain hat
    assert still[30:40, 45:55].sum() > 0


def test_reset_forgets_the_reaction():
    fx = knocked()
    fx.reset(1)
    assert 2 in fx._anim
    fx.reset(2)
    assert fx._anim == {}
    fx = knocked()
    fx.reset()
    assert fx._anim == {}


# ---------- Cheek bulge (merge_log.md, "Face effects", step 5) ----------
def test_bulge_map_magnifies_the_middle_and_keeps_the_rim():
    mx, my = bulge_map((41, 41), (20.0, 20.0), 20.0, 0.5)
    assert (mx[20, 20], my[20, 20]) == (20.0, 20.0)            # centre stays
    assert 20 < mx[20, 25] < 25                                 # 5 px out samples closer in: magnified
    np.testing.assert_allclose(mx[20, 0], 0.0, atol=1e-4)       # on the rim: unchanged
    mx0, my0 = bulge_map((41, 41), (20.0, 20.0), 20.0, 0.0)     # k = 0: identity
    ys, xs = np.mgrid[0:41, 0:41]
    np.testing.assert_allclose(mx0, xs, atol=1e-4)
    np.testing.assert_allclose(my0, ys, atol=1e-4)


def test_warp_bulge_grows_a_spot_and_leaves_the_rest():
    img = np.zeros((100, 100, 3), np.uint8)
    img[48:53, 48:53] = 255                                     # 5 px spot at the centre
    img[5, 5] = 200                                             # far outside the circle
    before = int((img[:, :, 0] > 127).sum())
    warp_bulge(img, (50.0, 50.0), 20.0, 0.5)
    assert int((img[:, :, 0] > 127).sum()) > 2 * before         # the spot is magnified
    assert img[5, 5, 0] == 200


def test_puffing_the_cheeks_does_not_start_a_bulge():
    """Only eating a consumable (bulge(), the consumables feature) starts one."""
    fx = FaceEffects(load_config()["face"])
    f = cheek_face()
    f.blendshapes = {"cheekPuff": 0.9}
    fx.update([], {1: f}, 1.0)
    assert fx.bulge_strength(fx._anim.get(1), 1.5) == 0.0


def test_bulge_rises_and_falls_over_its_time():
    fx = FaceEffects(load_config()["face"])
    fx.bulge(1, 10.0)                                           # e.g. eating a consumable
    a, s, k = fx._anim[1], fx.cfg["bulge_s"], fx.cfg["bulge_k"]
    assert fx.bulge_strength(a, 10.0) == pytest.approx(0.0)
    assert fx.bulge_strength(a, 10.0 + s / 2) == pytest.approx(k)
    assert fx.bulge_strength(a, 10.0 + s) == 0.0
    assert fx.bulge_strength(None, 10.0) == 0.0


def test_draw_warps_the_cheeks_only_while_bulging():
    fx = FaceEffects(load_config()["face"])
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    f = cheek_face()
    plain_img = base.copy()
    fx.draw(plain_img, {1: f}, 10.0)
    fx.bulge(1, 10.0)
    bulged = base.copy()
    fx.draw(bulged, {1: f}, 10.0 + fx.cfg["bulge_s"] / 2)
    r = int(fx.cfg["bulge_radius"] * f.scale)
    diff = (plain_img != bulged).any(axis=2)
    assert diff[50 - r:50 + r, 30 - r:30 + r].any()             # left cheek warped
    assert not diff[:, 30 + r + 1:70 - r - 1].any()             # nothing between the cheeks
    later = base.copy()
    fx.draw(later, {1: f}, 10.0 + fx.cfg["bulge_s"])
    np.testing.assert_array_equal(later, plain_img)             # over: as before


def test_cheeks_bulge_gently_while_a_drink_is_held():
    fx = FaceEffects(load_config()["face"])
    k, ramp = fx.cfg["bulge_hold_k"], fx.cfg["bulge_hold_ramp_s"]
    fx.update([], {}, 0.0, holding=[1])
    a = fx._anim[1]
    assert fx.bulge_strength(a, 0.0) == 0.0                    # starts from nothing
    fx.update([], {}, ramp / 2, holding=[1])
    assert fx.bulge_strength(a, ramp / 2) == pytest.approx(k / 2)
    fx.update([], {}, ramp, holding=[1])
    fx.update([], {}, 3.0, holding=[1])
    assert fx.bulge_strength(a, 3.0) == pytest.approx(k)       # stays while held
    fx.update([], {}, 3.0 + ramp / 2)                           # dropped: lets go smoothly
    assert fx.bulge_strength(a, 3.0 + ramp / 2) == pytest.approx(k / 2)
    fx.update([], {}, 3.0 + ramp)
    assert fx.bulge_strength(a, 3.0 + ramp) == 0.0


def test_drinking_puff_rises_from_the_held_bulge_without_a_dip():
    fx = FaceEffects(load_config()["face"])
    t = 0.0
    for _ in range(20):                                         # held for 1 s at ~20 FPS
        fx.update([], {}, t, holding=[1])
        t += 0.05
    fx.update([], {}, t)                                        # drunk: no longer held
    fx.bulge(1, t)
    a, s = fx._anim[1], fx.cfg["bulge_s"]
    ks = [fx.bulge_strength(a, t + i * 0.01) for i in range(int(s / 2 * 100))]
    assert ks[0] == pytest.approx(fx.cfg["bulge_hold_k"], abs=0.07)
    assert all(b >= a_ - 1e-9 for a_, b in zip(ks, ks[1:], strict=False))   # only rises to the peak
    assert ks[-1] == pytest.approx(fx.cfg["bulge_k"], abs=0.01)
