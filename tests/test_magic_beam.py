"""Magic beam (scene/magic_beam.py) and the sprite helpers it uses."""

import cv2
import numpy as np
import pytest

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.scene.draw import blend_rgba, hue_of, main_hue, player_color, shift_hue
from ipcvgame.scene.magic_beam import MagicBeam

RED, WHITE = (40, 40, 220, 255), (255, 255, 255, 255)


def fake_pack(tmp_path, n_ball=10):
    """Ball frames b-1 ... b-n (20 x 20 px, a red dot; frame k has k white
    pixels, so frames can be told apart) and 3 bands of 12 x 6 px: full,
    middle 2/3, thin 1/3 (transparent rows above and below)."""
    for k in range(1, n_ball + 1):
        f = np.zeros((20, 20, 4), np.uint8)
        f[6:14, 6:14] = RED
        f[0, :k] = WHITE
        cv2.imwrite(str(tmp_path / f"b-{k}.png"), f)
    for k, (top, bottom) in enumerate([(0, 12), (2, 10), (4, 8)], start=1):
        f = np.zeros((12, 6, 4), np.uint8)
        f[top:bottom] = RED
        cv2.imwrite(str(tmp_path / f"band-{k}.png"), f)
    return str(tmp_path / "b-*.png"), str(tmp_path / "band-*.png")


@pytest.fixture
def magic(tmp_path):
    ball, band = fake_pack(tmp_path)
    return MagicBeam(ball, band, ramp_s=0.1)


def test_frames_load_in_natural_order(magic):
    # b-10 must come after b-9, not after b-1.
    assert [int((f[0, :, 3] > 0).sum()) for f in magic.ball] == list(range(1, 11))


def test_ball_plays_once_over_the_life(magic):
    whites = [int((magic.ball_frame(t, 1.0, 20)[0, :, 3] > 0).sum()) for t in (0.0, 0.5, 0.99, 1.5)]
    assert whites == [1, 6, 10, 10]


def test_band_ramps_up_then_down(magic):
    # charge 0.35, life 1.15: thin, middle, full ... middle, thin.
    idx = [magic.band_index(t, 0.35, 1.15) for t in (0.2, 0.36, 0.46, 0.6, 0.97, 1.1)]
    assert idx == [None, 2, 1, 0, 1, 2]


def test_draw_beam_from_ball_to_tip_and_ball_on_top(magic):
    canvas = np.zeros((100, 300, 3), np.uint8)
    magic.draw(canvas, 100, 250, 50, 1, age=0.7, charge_s=0.35, life_s=1.15, band_px=12, ball_px=20)
    assert canvas[50, 200].any() and canvas[50, 249].any()    # beam up to the tip
    assert not canvas[50, 260:].any()                          # not past it
    assert not canvas[30, 200].any()                           # 12 px band only
    assert canvas[50, 100].any()                               # ball at the start


def test_draw_left_and_while_charging(magic):
    canvas = np.zeros((100, 300, 3), np.uint8)
    magic.draw(canvas, 200, 50, 50, -1, age=0.7, charge_s=0.35, life_s=1.15, band_px=12, ball_px=20)
    assert canvas[50, 60].any() and not canvas[50, 220:].any()
    canvas[:] = 0
    magic.draw(canvas, 200, 200, 50, -1, age=0.1, charge_s=0.35, life_s=1.15, band_px=12, ball_px=20)
    assert canvas[50, 200].any() and not canvas[50, :180].any()   # only the ball


def test_main_hue_is_circular_for_red():
    px = np.zeros((1, 2, 4), np.uint8)
    px[0, 0], px[0, 1] = (0, 20, 255, 255), (20, 0, 255, 255)    # hues just above 0, just below 180
    assert main_hue([px]) in (0, 1, 2, 177, 178, 179)            # red, not ~90 (cyan)


def test_shift_hue_keeps_white_and_alpha():
    img = np.array([[RED, WHITE, (0, 0, 0, 0)]], np.uint8)
    blue = shift_hue(img, hue_of(player_color(2)) - hue_of(RED[:3]))
    assert abs(hue_of(tuple(int(v) for v in blue[0, 0, :3])) - hue_of(player_color(2))) <= 1
    assert tuple(blue[0, 1]) == WHITE and tuple(blue[:, :, 3].ravel()) == (255, 255, 0)


def test_blend_rgba_clips_and_blends():
    canvas = np.zeros((10, 10, 3), np.uint8)
    sprite = np.zeros((4, 4, 4), np.uint8)
    sprite[:, :, :3], sprite[:, :, 3] = 200, 255
    blend_rgba(canvas, sprite, -2, 8, alpha=0.5)            # partly off the canvas
    assert canvas[8:, :2].min() == 100 and canvas[:8].max() == 0 and canvas[:, 2:].max() == 0
    blend_rgba(canvas, sprite, 20, 20)                      # fully off: no error


def game_renderer():
    """The renderer with the real config and the stub face effects."""
    from ipcvgame.face.effects import FaceEffects
    from ipcvgame.scene.renderer import Renderer
    full = load_config()
    return Renderer(full["render"], FaceEffects({**full["face"], "backend": "stub"}))


def test_renderer_gives_each_team_its_own_beam_colour():
    cfg = load_config()["render"]
    if not list((REPO_ROOT / "assets/magic_effect").glob("M5025-*.png")):
        pytest.skip("magic_effect pack not in this checkout")
    r = game_renderer()
    owner, other = cfg["beam_effect_owner"], 3 - cfg["beam_effect_owner"]
    base = r.beam_effect(owner)
    assert r.beam_effect(owner) is base and len(base.ball) == 26 and len(base.band) == 3
    for pid in (owner, other):
        diff = abs(r.beam_effect(pid).hue - hue_of(player_color(pid)))
        assert min(diff, 180 - diff) <= 2                       # hue is circular
