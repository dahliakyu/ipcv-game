"""Game view (scene/renderer.py): sprites per team, ring and health-bar geometry."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Face
from ipcvgame.face.effects import FaceEffects
from ipcvgame.interaction.game import Game
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.renderer import Renderer
from tests.test_game import busy_game, frame_state

CFG = load_config()


def renderer(**over) -> Renderer:
    return Renderer({**CFG["render"], **over}, FaceEffects({**CFG["face"], "backend": "stub"}))


@pytest.fixture
def game() -> Game:
    return Game(CFG["game"])


def face_at(x: float, y: float, scale: float) -> Face:
    return Face(pid=1, valid=True, landmarks=None, center=np.array([x, y + scale]),
                scale=scale, roll=0.0, yaw=0.0, anchors={"above_head": np.array([x, y])})


def test_draws_every_object(game):
    busy_game(game)
    state = frame_state(0.1)
    canvas = renderer().draw(state, game)
    assert canvas.shape == state.frame.image.shape and canvas.any()


def test_plain_shapes_without_sprites(game):
    r = renderer(bolt_sprite=None, beam_ball=None, shield_sprite=None)
    busy_game(game)
    state = frame_state(0.6)
    game.update(state)
    assert r.draw(state, game).any()
    assert r.bolt_effect(1, game) is None and r.beam_effect(1) is None and r.shield_effect(1) is None


def test_health_bar_length_ignores_the_arms(game):
    game.update(frame_state(0.0))
    s = frame_state(0.0)
    s.faces[1] = face_at(200.0, 120.0, 40.0)
    r = renderer()
    (c, length) = r.bar_geometry(1, s, game)
    # Under the shoulder midpoint, at the bottom of the body box.
    assert c == (200.0, 900.0) and length == pytest.approx(2.5 * 40)
    game.get_player(1).bounds = np.array([100.0, 50, 600, 900])   # arm out: wide box
    assert r.bar_geometry(1, s, game) == (c, length)


def test_health_bar_without_face_uses_shoulder_width(game):
    game.update(frame_state(0.0))
    (cx, cy), length = renderer().bar_geometry(1, frame_state(0.0), game)
    # Shoulders 100 px apart, centred at x 200; box bottom at y 900.
    assert (cx, cy) == (200.0, 900.0) and length == pytest.approx(1.3 * 100)


def test_shield_ring_sits_on_the_chest(game):
    s = frame_state(0.0, events=[(1, "arms_crossed")])
    game.update(s)
    r = renderer()
    (cx, cy), d = r.ring_geometry(1, game.shields[1], s, game)
    # Shoulders at y 300 and 100 px apart, hips at 600: chest at 450.
    assert (cx, cy) == (200.0, 450.0) and d == pytest.approx(r.cfg["shield_ring_sw"] * 100)


def test_shield_ring_falls_back_without_shoulders(game):
    s = frame_state(0.0, events=[(1, "arms_crossed")])
    game.update(s)
    s.motion[1].valid[sk.R_SHOULDER] = False
    (cx, cy), d = renderer().ring_geometry(1, game.shields[1], s, game)
    assert (cx, cy) == (game.shields[1].x, game.shields[1].y) and d == 200.0   # box width


def test_pixelate_makes_blocks():
    from ipcvgame.scene.renderer import pixelate
    img = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    out = pixelate(img, 2)
    assert (out[0::2, 0::2] == out[1::2, 1::2]).all() and (out[0::2] == out[1::2]).all()
    np.testing.assert_array_equal(pixelate(img, 1), img)


def test_paint_crosses_in_attacker_colour_grow_with_damage(game):
    """box: here the area the paint is drawn in."""
    from ipcvgame.interaction.game import PaintSplat
    r = renderer()
    box = np.array([100.0, 100, 300, 500])

    def painted(coverage):
        canvas = np.zeros((600, 400, 3), np.uint8)
        r._draw_player_paint(canvas, box, [PaintSplat(0.5, 0.5, owner_id=2)], coverage, game.splat_band)
        return canvas
    fresh, beaten = painted(0.0), painted(1.0)
    blue = np.all(fresh == player_color(2), axis=2)
    assert blue.any() and not blue[:100].any() and not blue[:, :100].any()     # inside the box
    assert np.all(beaten == player_color(2), axis=2).sum() > 3 * blue.sum()


def test_health_bar_fill_and_only_for_visible_players(game):
    r = renderer(pixel_size=1, show_timings=False)
    state = frame_state(0.0)
    game.update(state)
    game.get_player(1).health = 50
    state.frame.image[:] = 0
    bars = r.draw(state, game)[850:960]               # box bottoms at y 900: the bars, not the HUD
    red = np.all(bars == player_color(1), axis=2)
    blue = np.all(bars == player_color(2), axis=2)
    assert red.any() and blue.any()
    assert red.sum() < blue.sum()                     # half a bar against a full one
    lost = frame_state(0.0, lost=(2,))
    lost.frame.image[:] = 0
    assert not np.all(r.draw(lost, game)[850:960] == player_color(2), axis=2).any()


def test_hud_and_game_over_hide_objects_in_flight(game):
    r = renderer(pixel_size=1, show_timings=False, bolt_sprite=None, beam_ball=None, shield_sprite=None,
                 game_over_dim=1.0)
    busy_game(game)                                  # a bolt, a beam and a shield on screen
    state = frame_state(0.1)
    state.frame.image[:] = 0
    bolt = game.projectiles[0]
    at_bolt = (int(bolt.y), int(bolt.x))
    playing = r.draw(state, game)
    assert playing[: int(r.cfg["hud_h"] * 1000)].any()                  # top bar drawn
    assert tuple(playing[at_bolt]) == player_color(bolt.owner_id)       # the bolt is drawn
    game.winner_id = 1
    over = r.draw(state, game)
    h, w = over.shape[:2]
    mid = over[h // 2 - 60:h // 2 + 60, w // 2 - 200:w // 2 + 200]
    assert np.all(mid == player_color(1), axis=2).any()                 # panel in the winner's colour
    assert tuple(over[at_bolt]) != player_color(bolt.owner_id)          # the bolt is not
    renderer(pixel_size=1, show_timings=False).draw(state, game)        # dimming runs too


def test_new_round_keeps_players(monkeypatch):
    from ipcvgame.app.pipeline import Pipeline
    p = Pipeline(load_config(overrides=["pose.backend=none", "face.backend=stub", "identity.backend=zones"]))
    p.game.get_player(1).health = 10
    p.identity.players[1] = "kept"
    p.new_round()
    assert p.game.get_player(1).health == p.game.max_health and p.identity.players[1] == "kept"
    p.reset()
    assert p.identity.players == {}


def test_paint_stays_on_the_torso_when_an_arm_goes_out(game):
    state = frame_state(0.0)
    game.update(state)
    r = renderer()
    area = r.paint_area(1, state, game)
    assert area == (150.0, 300.0, 250.0, 600.0)          # shoulders 150-250, shoulder line to hips
    game.get_player(1).bounds = np.array([100.0, 150, 600, 900])   # arm out: the box widens
    assert r.paint_area(1, state, game) == area
    state.motion[1].valid[sk.R_SHOULDER] = False         # no shoulders: the box's splat band
    lo, hi = game.splat_band
    assert r.paint_area(1, state, game) == (100 + lo * 500, 150 + lo * 750, 100 + hi * 500, 150 + hi * 750)


def test_health_bar_length_is_smoothed(game):
    r = renderer(pixel_size=1, bar_length_tau_s=1.0)
    s0 = frame_state(0.0)
    s0.faces[1] = face_at(200.0, 120.0, 40.0)
    game.update(s0)
    r.draw(s0, game)
    s1 = frame_state(0.1)                                 # face missed: raw length jumps to 1.3 sw
    game.update(s1)
    r.draw(s1, game)
    smoothed = r._bar_len[1].x[0]
    assert 2.5 * 40 < smoothed < 2.5 * 40 + 0.2 * (1.3 * 100 - 2.5 * 40)
