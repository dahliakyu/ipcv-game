"""Game view (scene/renderer.py): sprites per team, ring and health-bar geometry."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Face
from ipcvgame.face.effects import FaceEffects
from ipcvgame.interaction.game import Game
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.renderer import WHITE as WHITE_PX
from ipcvgame.scene.renderer import Renderer
from tests.test_game import busy_game, frame_state

CFG = load_config()


def renderer(**over) -> Renderer:
    return Renderer({**CFG["render"], **over}, FaceEffects({**CFG["face"], "backend": "stub"}))


@pytest.fixture
def game() -> Game:
    return Game({**CFG["game"], "intro": False})


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
    game.winner_id, game.phase = 1, "result"
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


def test_text_fits_a_narrow_frame():
    import cv2

    from ipcvgame.scene.renderer import CONTROLS, fit_scale
    scale = fit_scale(CONTROLS, 1.5, 1, 200)
    assert cv2.getTextSize(CONTROLS, cv2.FONT_HERSHEY_PLAIN, scale, 1)[0][0] <= 200
    assert fit_scale("OK", 1.5, 1, 200) == 1.5


def test_health_bars_pinned_to_the_bottom_corners(game):
    r = renderer(pixel_size=1, show_timings=False)
    state = frame_state(0.0, xs=(800.0, 200.0))           # P1 stands on the right, P2 on the left
    game.update(state)
    game.get_player(2).health = 50
    state.frame.image[:] = 0
    h, w = state.frame.image.shape[:2]
    bars = r.draw(state, game)[h - 60:]
    red = np.all(bars == player_color(1), axis=2)
    blue = np.all(bars == player_color(2), axis=2)
    xs_red, xs_blue = np.nonzero(red)[1], np.nonzero(blue)[1]
    assert xs_red.max() < w * 0.35 and xs_blue.min() > w * 0.65  # P1 left, P2 right, wherever they stand
    assert blue.sum() < red.sum()                              # half a bar against a full one
    assert xs_blue.max() > w - 20                               # P2 fills from the right edge
    lost = frame_state(0.0, xs=(800.0, 200.0), lost=(2,))
    lost.frame.image[:] = 0
    assert np.all(r.draw(lost, game)[h - 60:] == player_color(2), axis=2).any()   # still shown


# --- Screens around the round (feature 12) ---------------------------------

def intro_game(**over) -> Game:
    return Game({**CFG["game"], "intro": True, **over})


def plain() -> Renderer:
    return renderer(pixel_size=1, show_timings=False, bolt_sprite=None, beam_ball=None, shield_sprite=None)


def dark(t: float = 0.0, **kw):
    s = frame_state(t, **kw)
    s.frame.image[:] = 0
    return s


def middle(canvas: np.ndarray) -> np.ndarray:
    h, w = canvas.shape[:2]
    return canvas[int(0.3 * h):int(0.7 * h), int(0.42 * w):int(0.58 * w)]


def test_status_text_per_phase():
    from ipcvgame.scene.renderer import CONTROLS, PHASE_TEXT, status_text
    assert status_text("playing", True) == CONTROLS
    assert status_text("playing", False) == "STEP INTO VIEW"
    for phase in ("waiting", "calibration", "countdown", "result"):
        assert status_text(phase, True) == PHASE_TEXT[phase]


def test_start_places_in_each_players_half():
    r = renderer()
    shape = (1000, 1000, 3)
    p1, p2 = r.start_place(1, 2, shape), r.start_place(2, 2, shape)
    assert p1[2] < 500 < p2[0]                                    # P1 left, P2 right
    solo = r.start_place(1, 1, shape)
    assert solo[0] < 500 < solo[2]                                # one player: the middle


def test_waiting_ticks_a_ready_player_and_hides_health_bars(monkeypatch):
    from tests.test_game import with_faces
    r, game = plain(), intro_game()
    bars = []
    monkeypatch.setattr(r, "_draw_player_health", lambda canvas, pid, g: bars.append(pid))
    state = dark()
    game.update(state)
    x1, y1, x2, y2 = r.start_place(1, 2, state.frame.image.shape)

    def p1_pixels(canvas):
        return np.all(canvas[y1:y2, x1:x2] == player_color(1), axis=2).sum()
    waiting = p1_pixels(r.draw(state, game))
    ready = with_faces(dark(), pids=(1,))
    assert p1_pixels(r.draw(ready, game)) > 1.5 * waiting         # solid box and a tick
    assert game.phase == "waiting" and bars == []


def test_countdown_numbers_then_fight():
    from tests.test_game import calibrated
    r, game = plain(), intro_game()
    t = calibrated(game)
    assert np.all(middle(r.draw(dark(t), game)) == WHITE_PX, axis=2).any()        # "3"
    game.update(dark(t + game.countdown_s))
    assert game.phase == "playing"
    assert np.all(middle(r.draw(dark(t + game.countdown_s), game)) == WHITE_PX, axis=2).any()  # FIGHT!
    late = t + game.countdown_s + r.cfg["fight_s"] + 0.1
    game.update(dark(late))
    assert not np.all(middle(r.draw(dark(late), game)) == WHITE_PX, axis=2).any()


def test_result_panel_fills_while_hands_are_up():
    r = renderer(pixel_size=1, show_timings=False, game_over_dim=1.0)
    game = Game({**CFG["game"], "intro": False})
    state = dark()
    game.update(state)
    game.winner_id, game.phase = 1, "result"
    empty = np.all(r.draw(state, game) == WHITE_PX, axis=2).sum()
    game._held = game.cfg["replay_hold_s"]
    assert game.hold_fraction() == 1.0
    assert np.all(r.draw(state, game) == WHITE_PX, axis=2).sum() > empty


# --- Instruction panels (feature 12, step 3) --------------------------------

def figure_kpts(pose: str) -> tuple[np.ndarray, np.ndarray]:
    """A stick figure as skeleton keypoints, 100 px per shoulder width."""
    from ipcvgame.scene.instructions import figure_points
    pts = figure_points(pose)
    k = np.zeros((sk.K, 2))
    k[sk.NOSE] = pts["head"]
    for side, model in (("a", "R"), ("b", "L")):     # image left = the model's R on a mirrored frame
        for joint in ("shoulder", "elbow", "wrist", "hip"):
            k[getattr(sk, f"{model}_{joint.upper()}")] = pts[f"{side}_{joint}"]
    return k * 100 + 500, np.ones(sk.K, bool)


@pytest.mark.parametrize("pose", ["neutral", "punch", "hands_up", "arms_crossed"])
def test_each_picture_is_a_pose_the_rules_detect(pose):
    from ipcvgame.interaction.gestures import detect_all
    from ipcvgame.scene.instructions import MOVE_TEXT
    assert pose in MOVE_TEXT
    k, v = figure_kpts(pose)
    on = {name for name, hit in detect_all(k, v, CFG["gestures"]).items() if hit}
    assert on == (set() if pose == "neutral" else {pose})


def test_every_calibration_move_has_a_picture_and_text():
    from ipcvgame.interaction.game import CALIBRATION_MOVES
    from ipcvgame.scene.instructions import ARMS, MOVE_TEXT
    assert set(CALIBRATION_MOVES) <= set(ARMS) and set(CALIBRATION_MOVES) <= set(MOVE_TEXT)


def test_instruction_panels_in_each_half_and_the_reminder_flashes():
    from tests.test_game import run, with_faces
    r, game = plain(), intro_game()
    t = run(game, 0.0, game.cfg["ready_s"], lambda t: with_faces(dark(t)))
    assert game.phase == "calibration"
    shape = (1000, 1000, 3)
    b1, b2 = r.instruction_box(1, 2, shape), r.instruction_box(2, 2, shape)
    assert b1[2] <= 500 <= b2[0]

    def border(canvas, box):
        x1, y1, x2, _ = box
        return canvas[y1, x1 + 5:x2 - 5]
    canvas = r.draw(dark(t), game)
    assert np.all(border(canvas, b1) == player_color(1), axis=1).all()
    assert np.all(border(canvas, b2) == player_color(2), axis=1).all()
    late = t + r.cfg["remind_s"] + 0.1                      # flash on: int(8.1 s x 2 x 2 Hz) is even
    canvas = r.draw(dark(late), game)
    assert np.all(border(canvas, b1) == WHITE_PX, axis=1).all()
