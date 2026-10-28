"""Game view (scene/renderer.py): sprites per team, ring and health-bar geometry."""

import cv2
import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Face, GestureEvent
from ipcvgame.face.effects import FaceEffects
from ipcvgame.interaction.game import Game
from ipcvgame.scene import renderer as renderer_mod
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.renderer import WHITE as WHITE_PX
from ipcvgame.scene.renderer import Renderer
from tests.test_game import busy_game, frame_state

CFG = load_config()


def renderer(**over) -> Renderer:
    return Renderer({**CFG["render"], **over}, FaceEffects({**CFG["face"], "backend": "stub"}))


@pytest.fixture
def game() -> Game:
    # The made-up frames have no faces: the facing-the-camera rule is off.
    return Game({**CFG["game"], "intro": False, "act_facing_camera": False})


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


# --- Screens around the round (game start) ---------------------------------

def intro_game(**over) -> Game:
    return Game({**CFG["game"], "intro": True, "act_facing_camera": False, **over})


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
    start = t + game.countdown_s + 0.01                    # just past the countdown (float sums of t)
    game.update(dark(start))
    assert game.phase == "playing"
    assert np.all(middle(r.draw(dark(start), game)) == WHITE_PX, axis=2).any()  # FIGHT!
    late = t + game.countdown_s + r.cfg["fight_s"] + 0.1
    game.update(dark(late))
    assert not np.all(middle(r.draw(dark(late), game)) == WHITE_PX, axis=2).any()


def test_result_panel_fills_while_hands_are_up():
    r = renderer(pixel_size=1, show_timings=False, game_over_dim=1.0)
    game = Game({**CFG["game"], "intro": False, "act_facing_camera": False})
    state = dark()
    game.update(state)
    game.winner_id, game.phase = 1, "result"
    empty = np.all(r.draw(state, game) == WHITE_PX, axis=2).sum()
    game._held = game.cfg["replay_hold_s"]
    assert game.hold_fraction() == 1.0
    assert np.all(r.draw(state, game) == WHITE_PX, axis=2).sum() > empty


# --- Instruction panels (game start, step 3) --------------------------------

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


def test_line_up_hint_replaces_the_reminder(monkeypatch):
    """Players not level: the nearer player's panel says to step back, in
    their team colour, instead of "TRY IT NOW!"."""
    from tests.test_game import run, with_faces
    r, game = plain(), intro_game()
    t = run(game, 0.0, game.cfg["ready_s"], lambda t: with_faces(dark(t)))
    late = t + r.cfg["remind_s"] + 0.3                      # reminder on, border not flashing
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: texts.append(text))
    r.draw(dark(late), game)
    assert "TRY IT NOW!" in texts and "STEP BACK TO LINE UP" not in texts
    texts.clear()
    monkeypatch.setattr(game, "line_up_hint", lambda pid: "back" if pid == 1 else "wait")
    r.draw(dark(late), game)
    assert "STEP BACK TO LINE UP" in texts and "WAIT FOR THE OTHER PLAYER" in texts
    assert "TRY IT NOW!" not in texts


def test_neutral_panel_asks_to_stand_level(monkeypatch):
    from tests.test_game import run, with_faces
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: texts.append(text))
    for players, shown in ((2, True), (1, False)):
        r, game = plain(), intro_game(round_players=players)
        t = run(game, 0.0, game.cfg["ready_s"], lambda t: with_faces(dark(t)))
        texts.clear()
        r.draw(dark(t), game)
        assert ("STAND LEVEL WITH THE OTHER PLAYER" in texts) == shown


def test_face_the_camera_shown_after_a_blocked_gesture(monkeypatch):
    from tests.test_game import with_faces
    r = plain()
    game = Game({**CFG["game"], "intro": False})
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: texts.append(text))
    s = dark(0.0)
    s.events.append(GestureEvent(1, "punch", 0.0, 1.0))          # no face: does not count
    game.update(s)
    r.draw(dark(0.5), game)
    assert "FACE THE CAMERA" in texts
    texts.clear()
    r.draw(with_faces(dark(0.5 + r.cfg["face_camera_s"])), game)
    assert "FACE THE CAMERA" not in texts


# ---------- Status above the head (merge_log.md, "Face effects", step 4) ----------
def faces_state(t: float = 0.0):
    from tests.test_game import with_faces
    return with_faces(dark(t))


def has_color(img: np.ndarray, color) -> bool:
    return bool((img == np.array(color, np.uint8)).all(axis=2).any())


def test_name_tags_in_every_phase(monkeypatch):
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: texts.append(text))
    for intro in (True, False):
        texts.clear()
        g = Game({**CFG["game"], "intro": intro, "act_facing_camera": False})
        plain().draw(faces_state(), g)
        assert "P1" in texts and "P2" in texts


def test_face_the_camera_takes_the_tags_place(monkeypatch):
    r = plain()
    game = Game({**CFG["game"], "intro": False})
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: texts.append(text))
    s = dark(0.0)
    s.events.append(GestureEvent(1, "punch", 0.0, 1.0))          # no face: does not count
    game.update(s)
    r.draw(faces_state(0.5), game)
    assert "FACE THE CAMERA" in texts and "P1" not in texts and "P2" in texts


def test_crown_for_the_player_with_more_hp(game):
    r = plain()
    game.update(dark())
    assert r.crowned(game) is None                                # both at 100
    game.players[2].health = 60
    assert r.crowned(game) == 1
    assert has_color(r.draw(faces_state(), game), renderer_mod.GOLD)
    game.players[2].health = 100
    game.players[1].health = 60
    assert r.crowned(game) == 2
    solo = Game({**CFG["game"], "intro": False, "round_players": 1, "act_facing_camera": False})
    solo.update(dark())
    assert r.crowned(solo) is None


def test_dizzy_stars_below_a_quarter_of_health(game):
    r = plain()
    game.update(dark())
    assert not has_color(r.draw(faces_state(), game), renderer_mod.STAR)
    game.players[1].health = game.players[2].health = 20
    assert has_color(r.draw(faces_state(), game), renderer_mod.STAR)
    waiting = Game({**CFG["game"], "act_facing_camera": False})   # intro: health does not count yet
    waiting.update(dark())
    waiting.players[1].health = 20
    assert not has_color(r.draw(faces_state(), waiting), renderer_mod.STAR)


def test_stars_circle_over_time(game):
    r = plain()
    game.update(dark())
    game.players[1].health = 10
    a, b = r.draw(faces_state(0.0), game), r.draw(faces_state(0.2), game)
    stars_a = (a == np.array(renderer_mod.STAR, np.uint8)).all(axis=2)
    stars_b = (b == np.array(renderer_mod.STAR, np.uint8)).all(axis=2)
    assert stars_a.any() and stars_b.any() and (stars_a != stars_b).any()


def test_status_fades_with_the_face(game):
    r = plain()
    game.update(dark())
    game.players[2].health = 60                                   # P1 crowned
    full = r.draw(faces_state(), game)
    s = faces_state()
    s.faces[1].alpha = 0.5
    s.faces[2].alpha = 0.5
    half = r.draw(s, game)
    assert has_color(full, renderer_mod.GOLD) and not has_color(half, renderer_mod.GOLD)
    assert half[:300].max() < full[:300].max()


def test_star_shape():
    from ipcvgame.scene.renderer import star
    pts = star((0, 0), 10.0, 0.0)
    assert pts.shape == (10, 2)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1)[::2], 10.0)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1)[1::2], 4.5)
    np.testing.assert_allclose(pts[0], (0, -10), atol=1e-9)     # first point straight up


# ---------- Steady name tags ----------
def tag_face(y: float, x: float = 300.0, scale: float = 60.0) -> Face:
    return Face(pid=1, valid=True, landmarks=None, center=np.array([x, y + 100]), scale=scale,
                roll=0.0, yaw=0.0, anchors={"above_head": np.array([x, y]),
                                            "hat": np.array([x, y + 60])})


def tag_org(monkeypatch, r, f, t, crowned=False):
    orgs = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, org, *a, **k: orgs.append((org, a[1])))
    r._draw_tag(np.zeros((720, 1280, 3), np.uint8), 1, f, crowned, t)
    monkeypatch.undo()
    return orgs[-1]                                       # (org, font scale) of the coloured text


def test_tag_goes_beside_the_hat_and_back_only_with_room(monkeypatch):
    r = plain()
    top = r.cfg["hud_h"] * 720
    r.cfg["tag_filter_min_cutoff"] = 1e6                  # no smoothing: positions as placed
    (x, _), _ = tag_org(monkeypatch, r, tag_face(top + 60), 0.0)
    assert r._tags[1].side == 0
    tag_org(monkeypatch, r, tag_face(top + 5), 0.1)       # under the top bar: beside the hat
    assert r._tags[1].side == 1                           # left half: towards the middle
    tag_org(monkeypatch, r, tag_face(top + 30), 0.2)      # a little room: stays beside
    assert r._tags[1].side == 1
    tag_org(monkeypatch, r, tag_face(top + 80), 0.3)      # plenty of room: back above
    assert r._tags[1].side == 0


def test_tag_side_follows_the_player_past_the_middle(monkeypatch):
    r = plain()
    top = r.cfg["hud_h"] * 720
    tag_org(monkeypatch, r, tag_face(top, x=600), 0.0)
    assert r._tags[1].side == 1
    tag_org(monkeypatch, r, tag_face(top, x=700), 0.1)    # just past the middle: kept
    assert r._tags[1].side == 1
    tag_org(monkeypatch, r, tag_face(top, x=800), 0.2)    # well past it: other side
    assert r._tags[1].side == -1


def test_tag_size_changes_only_past_the_tolerance(monkeypatch):
    r = plain()
    _, s0 = tag_org(monkeypatch, r, tag_face(300, scale=60), 0.0)
    _, s1 = tag_org(monkeypatch, r, tag_face(300, scale=63), 0.1)       # +5%: same size
    _, s2 = tag_org(monkeypatch, r, tag_face(300, scale=80), 0.2)       # +33%: new size
    assert s1 == s0 and s2 > s0


def test_tag_position_is_smoothed_and_reset(monkeypatch):
    r = plain()
    rng = np.random.default_rng(0)
    xs = []
    for i in range(30):
        (x, _), _ = tag_org(monkeypatch, r, tag_face(300, x=300 + rng.normal(0, 4)), i / 30)
        xs.append(x)
    assert np.std(np.diff(xs[10:])) < 2.0                 # anchor noise 4 px: tag far steadier
    r.reset(1)
    assert 1 not in r._tags


def test_drinking_puffs_the_cheeks(game):
    from ipcvgame.interaction.game import Drunk
    r = plain()
    game.update(dark(1.0))
    game.drunk = [Drunk(1, "hp", 20.0)]
    r.draw(faces_state(1.0), game)
    assert r.face_effects._anim[1].bulge_t == 1.0
    assert 2 not in r.face_effects._anim or r.face_effects._anim[2].bulge_t is None


# ---------- Drinks (consumables, extra_features.md feature 2, step 4) ----------
def drink_game():
    from ipcvgame.interaction.game import Game as G
    g = G({**CFG["game"], "intro": False, "act_facing_camera": False, "drink_every_s": [1e6, 1e6]})
    g.update(dark())
    return g


def color_mask(img: np.ndarray, color) -> np.ndarray:
    return (img == np.array(color, np.uint8)).all(axis=2)


def test_drink_sprites_per_kind_cropped_and_optional():
    r = plain()
    for kind in ("hp", "attack", "defense"):
        sprites = r.drink_sprites(kind)
        assert len(sprites) == 3
        for s in sprites:                                         # cropped: visible on every edge
            a = s[:, :, 3] > 0
            assert a[0].any() and a[-1].any() and a[:, 0].any() and a[:, -1].any()
    assert renderer(drink_sprites=None).drink_sprites("hp") == []


@pytest.mark.parametrize("sprites", [True, False])
def test_falling_drink_drawn_with_its_badge(sprites):
    from ipcvgame.interaction.game import Drink
    r = plain() if sprites else renderer(pixel_size=1, show_timings=False, drink_sprites=None)
    g = drink_game()
    d = Drink(500.0, 500.0, "defense", 1.0, 0.3, 0.0)
    g.drinks.append(d)
    canvas = r.draw(dark(), g)
    h = g.drink_height(d)
    assert canvas[int(500 - h / 2):int(500 + h / 2), int(500 - h / 2):int(500 + h)].any()
    assert color_mask(canvas, renderer_mod.DRINK_COLORS["defense"]).any()       # the badge's shield


def test_pick_up_text_rises_and_ends(monkeypatch):
    from ipcvgame.interaction.game import Drunk
    r = plain()
    g = drink_game()
    texts = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, org, *a, **k: texts.append((text, org)))
    g.drunk = [Drunk(1, "attack", 1.42)]
    r.draw(faces_state(1.0), g)
    g.drunk = []
    r.draw(faces_state(1.0 + r.cfg["popup_s"] / 2), g)
    ys = [org[1] for text, org in texts if text == "ATTACK x1.4"]
    assert len(ys) >= 4 and ys[-1] < ys[0]                        # drawn twice (outline + text), rising
    texts.clear()
    r.draw(faces_state(1.0 + r.cfg["popup_s"]), g)
    assert not any(text == "ATTACK x1.4" for text, _ in texts)
    g.drunk = [Drunk(2, "hp", 17.6), Drunk(1, "defense", 0.35)]
    r.draw(faces_state(5.0), g)
    assert {"+18 HP", "DEFENSE 35%"} <= {text for text, _ in texts}


def test_buff_icons_above_the_health_bars_while_running():
    r = plain()
    g = drink_game()
    color = renderer_mod.DRINK_COLORS["attack"]
    assert not color_mask(r.draw(dark(), g), color).any()
    g.players[1].attack, g.players[1].attack_left = 1.4, 8.0
    canvas = r.draw(dark(), g)
    ys, xs = np.nonzero(color_mask(canvas, color))
    assert xs.max() < canvas.shape[1] / 2 and ys.min() > canvas.shape[0] * 0.8   # bottom left: P1
    g.players[1].attack_left = 0.0
    g.players[2].defense, g.players[2].defense_left = 0.3, 8.0
    ys, xs = np.nonzero(color_mask(r.draw(dark(), g), renderer_mod.DRINK_COLORS["defense"]))
    assert xs.min() > canvas.shape[1] / 2                          # bottom right: P2


def test_buff_icon_fades_at_the_end():
    r = plain()
    g = drink_game()
    color = renderer_mod.DRINK_COLORS["attack"]
    g.players[1].attack, g.players[1].attack_left = 1.4, 8.0
    full = color_mask(r.draw(dark(), g), color).sum()
    g.players[1].attack_left = r.cfg["buff_fade_s"] / 2
    assert color_mask(r.draw(dark(), g), color).sum() < full     # blended: not the pure colour


def test_one_player_sees_the_buff_icon():
    from ipcvgame.interaction.game import Game as G
    solo = G({**CFG["game"], "intro": False, "round_players": 1, "act_facing_camera": False})
    solo.update(dark(xs=(200.0,)))
    solo.players[1].defense, solo.players[1].defense_left = 0.4, 5.0
    assert color_mask(plain().draw(dark(xs=(200.0,)), solo), renderer_mod.DRINK_COLORS["defense"]).any()


def test_faded_blends_the_box():
    img = np.zeros((10, 10, 3), np.uint8)
    with renderer_mod.faded(img, (0, 0, 5, 5), 0.5):
        img[:] = 200
    assert img[2, 2, 0] == 100 and img[8, 8, 0] == 200


def test_held_drink_bulges_the_cheeks():
    from ipcvgame.interaction.game import Drink
    r = plain()
    g = drink_game()
    g.drinks.append(Drink(300.0, 200.0, "hp", 1.0, 20.0, 0.0, held_by=1, held_t=0.0))
    for i in range(5):
        r.draw(faces_state(i * 0.1), g)
    assert r.face_effects._anim[1].hold_level == pytest.approx(r.face_effects.cfg["bulge_hold_k"])
    assert 2 not in r.face_effects._anim


def test_held_drink_at_the_lower_lip_face_sized_without_badge():
    from ipcvgame.interaction.game import Drink
    from tests.test_game import mouths
    r = plain()
    g = drink_game()
    s = mouths(0.0, {1: 0.9})                                  # P1's mouth at (200, 260), face 60 px
    d = Drink(200.0, 300.0, "defense", 1.4, 0.4, 0.0, held_by=1, held_t=0.0)
    g.drinks.append(d)
    assert g.drink_height(d) > r.cfg["drink_held_fw"] * 60     # bigger than a face when falling
    canvas = r.draw(s, g)
    g.drinks.clear()
    without = r.draw(mouths(0.0, {1: 0.9}), g)
    ys, xs = np.nonzero((canvas != without).any(axis=2))      # what the drink drew
    top = 260 + r.cfg["drink_held_drop_fw"] * 60
    assert ys.min() >= top - 1 and ys.max() <= top + r.cfg["drink_held_fw"] * 60 + 1
    assert abs(xs.mean() - 200) < 10                           # centred under the mouth
    assert not color_mask(canvas, renderer_mod.DRINK_COLORS["defense"]).any()   # no badge
