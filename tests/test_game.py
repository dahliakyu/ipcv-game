"""Colour Clash game logic (game.py), in pixels."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Face, Frame, FrameState, GestureEvent, Motion, Player, PoseDetection
from ipcvgame.interaction.game import CALIBRATION_MOVES, Game, PaintSplat, Shield
from ipcvgame.scene.draw import player_color


@pytest.fixture
def game() -> Game:
    """Without the intro: playing from the first frame."""
    return Game({**load_config()["game"], "intro": False})


@pytest.fixture
def arena() -> Game:
    """The setup of test_game.py in a 1000 x 1000 px frame, so its 0-1
    coordinates become px by multiplying by 1000: P1 on the left, P2 right.
    With game.py's bolt speed (1.4), so its test numbers still hold."""
    g = Game({**load_config()["game"], "bolt_speed_w": 1.4})
    g.frame_w, g.frame_h = 1000, 1000
    g.get_player(1).bounds = np.array([100.0, 200, 300, 800])
    g.get_player(2).bounds = np.array([600.0, 200, 800, 800])
    return g


def test_team_colours_are_red_and_blue():
    b, g, r = player_color(1)
    assert r > b and r > g          # P1 red
    b, g, r = player_color(2)
    assert b > r and b > g          # P2 blue


def test_new_player_has_full_health(game):
    p = game.get_player(1)
    assert p.health == game.max_health == 100
    assert game.get_player(1) is p


def test_reset_starts_a_fresh_round(game):
    p2 = game.get_player(2)
    p2.x, p2.bounds = 900.0, np.array([800.0, 100, 1000, 600])
    p2.health = 0
    p2.paint_splats.append(PaintSplat(0.5, 0.5, owner_id=1))
    game.shields[2] = Shield(900.0, 400.0, -1, remaining=1.0)
    game.winner_id = 1

    game.reset()

    assert game.winner_id is None
    assert p2.health == 100 and p2.paint_splats == []
    assert game.projectiles == [] and game.beams == [] and game.shields == {}
    assert p2.x == 900.0  # positions are kept


def test_reset_one_player(game):
    p1, p2 = game.get_player(1), game.get_player(2)
    p1.health, p2.health = 40, 50
    game.shields[1] = Shield(100.0, 400.0, 1, remaining=1.0)
    game.reset(1)
    assert p1.health == 100 and 1 not in game.shields
    assert p2.health == 50


# --- The combat tests from test_game.py, in px -----------------------------

def test_red_bolt_damages_blue_player_and_splats_red_paint(arena):
    arena.attack(300, 500, 1, player_id=1)
    arena.advance(0.3, {1, 2})

    blue = arena.get_player(2)
    assert blue.health == 90
    assert blue.paint_splats
    assert all(s.owner_id == 1 for s in blue.paint_splats)
    assert all(0.38 <= s.x <= 0.62 and 0.38 <= s.y <= 0.62 for s in blue.paint_splats)


def test_blue_bolt_damages_red_player_with_blue_paint(arena):
    arena.attack(600, 500, -1, player_id=2)
    arena.advance(0.3, {1, 2})

    red = arena.get_player(1)
    assert red.health == 90
    assert red.paint_splats[0].owner_id == 2


def test_beam_damages_a_target_only_once(arena):
    arena.beam_attack(300, 500, 1, player_id=1)

    arena.advance(0.35, {1, 2})     # still charging
    assert arena.get_player(2).health == 100
    arena.advance(0.15, {1, 2})
    assert arena.get_player(2).health == 88
    arena.advance(0.1, {1, 2})
    assert arena.get_player(2).health == 88


def test_beam_stops_at_the_frame_edge_and_lasts_its_duration(arena):
    arena.beam_attack(300, 100, 1, player_id=1)       # above both boxes
    arena.advance(0.35, {1, 2})
    arena.advance(0.6, {1, 2})                        # tip would be far past the edge
    (beam,) = arena.beams
    assert beam.x == 1000                             # stopped at the frame edge
    arena.advance(0.21, {1, 2})                       # 1.16 s: past charge + duration
    assert arena.beams == []


def test_no_bolt_while_own_beam_is_alive(arena):
    arena.beam_attack(300, 500, 1, player_id=1)
    assert not arena.attack(300, 500, 1, player_id=1)          # P1 is firing a beam
    assert arena.attack(700, 500, -1, player_id=2)             # P2 is not
    arena.advance(0.35, {1, 2})
    arena.advance(0.8, {1, 2})                                 # beam over
    assert arena.attack(300, 500, 1, player_id=1)


def test_one_beam_per_player_at_a_time(arena):
    assert arena.beam_attack(300, 500, 1, player_id=1)
    assert not arena.beam_attack(320, 450, 1, player_id=1)     # first still alive
    assert arena.beam_attack(700, 500, -1, player_id=2)        # other player: fine
    arena.advance(0.35, {1, 2})
    arena.advance(0.8, {1, 2})                                 # both beams finished
    assert arena.beams == []
    assert arena.beam_attack(300, 500, 1, player_id=1)


def test_bolt_cooldown_when_configured():
    # test_bolt_cooldown_is_longer_than_before from test_game.py, with its 1.8 s.
    g = Game({**load_config()["game"], "bolt_cooldown_s": 1.8})
    g.frame_w, g.frame_h = 1000, 1000
    assert g.attack(300, 500, 1, player_id=1)
    g.advance(1.5, {1, 2})
    assert not g.attack(300, 500, 1, player_id=1)
    g.advance(0.31, {1, 2})
    assert g.attack(300, 500, 1, player_id=1)


def test_no_cooldown_every_punch_fires(arena):
    assert arena.attack(300, 500, 1, player_id=1)
    assert arena.attack(300, 500, 1, player_id=1)
    assert len(arena.projectiles) == 2


def test_fully_painted_player_loses_and_game_stops_attacks(arena):
    arena.get_player(2).health = 10
    arena.attack(300, 500, 1, player_id=1)
    arena.advance(0.3, {1, 2})

    assert arena.winner_id == 1
    assert not arena.attack(300, 500, 1, player_id=1)


def test_bolt_does_not_damage_its_owner(arena):
    arena.attack(200, 500, 1, player_id=1)
    arena.advance(0.1, {1})
    assert arena.get_player(1).health == 100


# --- New: shield blocking and the rest of advance() ------------------------

def test_shield_blocks_a_bolt(arena):
    arena.activate_shield(700, 500, -1, player_id=2)
    arena.attack(300, 500, 1, player_id=1)
    arena.advance(0.3, {1, 2})
    p2 = arena.get_player(2)
    assert p2.health == 100 and p2.paint_splats == []
    assert arena.projectiles == []  # absorbed, not passing through


def test_shield_blocks_a_beam_for_the_rest_of_the_beam(arena):
    arena.activate_shield(700, 500, -1, player_id=2)
    arena.beam_attack(300, 500, 1, player_id=1)
    arena.advance(0.35, {1, 2})
    arena.advance(0.15, {1, 2})     # beam reaches P2 while the shield is up
    arena.advance(0.6, {1, 2})      # shield is gone, beam still alive
    assert arena.get_player(2).health == 100


def test_fading_shield_does_not_block_and_then_disappears(arena):
    cfg = arena.cfg
    arena.activate_shield(700, 500, -1, player_id=2)
    arena.advance(cfg["shield_s"] + 0.01, {1, 2})   # fading out
    assert not arena.shielded(2) and 2 in arena.shields
    arena.attack(300, 500, 1, player_id=1)
    arena.advance(0.3, {1, 2})                       # bolt arrives; fade ends
    assert arena.get_player(2).health == 90
    assert 2 not in arena.shields


def test_invisible_player_is_not_hit(arena):
    arena.attack(300, 500, 1, player_id=1)
    arena.advance(0.3, {1})
    assert arena.get_player(2).health == 100


def test_bolt_leaves_the_frame(arena):
    arena.attack(900, 100, 1, player_id=1)   # above both boxes
    arena.advance(0.1, {1, 2})
    assert arena.projectiles == []


def test_same_hit_gives_same_splats():
    def splats():
        g = Game(load_config()["game"])
        g.frame_w, g.frame_h = 1000, 1000
        g.get_player(2).bounds = np.array([600.0, 200, 800, 800])
        g.attack(300, 500, 1, player_id=1)
        g.advance(0.3, {1, 2})
        return [(s.x, s.y) for s in g.get_player(2).paint_splats]
    assert splats() == splats()


# --- update(state): positions, events -> actions, time ---------------------

W, H = 1000, 1000


def body(cx: float) -> tuple[Player, Motion]:
    """A player centred at x = cx: shoulders at y 300, hips at y 600, wrists
    raised at y 200; body box 200 px wide."""
    k = np.zeros((sk.K, 2))
    k[sk.L_SHOULDER], k[sk.R_SHOULDER] = (cx + 50, 300), (cx - 50, 300)
    k[sk.L_HIP], k[sk.R_HIP] = (cx + 40, 600), (cx - 40, 600)
    k[sk.L_WRIST], k[sk.R_WRIST] = (cx + 60, 200), (cx - 60, 200)
    valid = np.ones(sk.K, bool)
    box = np.array([cx - 100, 150, cx + 100, 900.0])
    det = PoseDetection(bbox=box, kpts=k, conf=np.ones(sk.K), score=0.9)
    player = Player(pid=0, status="active", det=det, bbox=box, pos=np.zeros(2), reentered=False)
    return player, Motion(kpts=k, vel=np.zeros_like(k), valid=valid, signals={})


def frame_state(t: float, events=(), xs=(200.0, 800.0), lost=()) -> FrameState:
    players, motion = {}, {}
    for pid, cx in enumerate(xs, start=1):
        p, m = body(cx)
        p.pid = pid
        if pid in lost:
            p.status, p.det = "lost", None
        else:
            motion[pid] = m
        players[pid] = p
    frame = Frame(image=np.zeros((H, W, 3), np.uint8), frame_id=0, t=t)
    return FrameState(frame=frame, players=players, motion=motion, faces={},
                      events=[GestureEvent(pid, name, t, 1.0) for pid, name in events],
                      timings={})


def test_update_reads_boxes_and_positions(game):
    game.update(frame_state(0.0))
    p1 = game.get_player(1)
    np.testing.assert_allclose(p1.bounds, [100, 150, 300, 900])
    assert (p1.x, p1.y) == (200.0, 600.0)   # shoulder midpoint, hip level
    assert (game.frame_w, game.frame_h) == (W, H)


def test_punch_fires_from_the_punching_hand(game):
    s = frame_state(0.0, events=[(1, "punch"), (2, "punch")])
    s.motion[1].kpts[sk.L_WRIST] = (390, 310)   # P1 punches right: 1.4 sw to the side
    s.motion[2].kpts[sk.R_WRIST] = (610, 320)   # P2 punches left
    game.update(s)
    b1, b2 = sorted(game.projectiles, key=lambda b: b.owner_id)
    assert (b1.x, b1.y, b1.direction) == (390.0, 310.0, 1)
    assert (b2.x, b2.y, b2.direction) == (610.0, 320.0, -1)
    assert game.actions == [(1, "bolt"), (2, "bolt")]


def test_punch_without_wrists_fires_from_the_box_edge(game):
    s = frame_state(0.0, events=[(1, "punch"), (2, "punch")])
    for pid in (1, 2):
        s.motion[pid].valid[[sk.L_WRIST, sk.R_WRIST]] = False
    game.update(s)
    b1, b2 = sorted(game.projectiles, key=lambda b: b.owner_id)
    assert (b1.x, b1.y) == (300.0, 600.0)    # P1 left: right edge, hip height
    assert (b2.x, b2.y) == (700.0, 600.0)    # P2 right: left edge


def test_hands_up_beam_starts_between_the_wrists(game):
    game.update(frame_state(0.0, events=[(1, "hands_up")]))
    (beam,) = game.beams
    assert (beam.start_x, beam.y, beam.direction, beam.owner_id) == (200.0, 200.0, 1, 1)


def test_arms_crossed_raises_shield_at_the_body(game):
    game.update(frame_state(0.0, events=[(2, "arms_crossed")]))
    assert (game.shields[2].x, game.shields[2].y, game.shields[2].direction) == (800.0, 600.0, -1)
    assert game.shielded(2)


def test_time_comes_from_frame_t_and_is_capped(game):
    game.update(frame_state(0.0, events=[(1, "punch")]))
    x0 = game.projectiles[0].x                     # P1's punching wrist, x 260
    game.update(frame_state(0.0))
    assert game.projectiles[0].x == x0             # dt = 0: no movement
    game.update(frame_state(0.05))
    speed = game.cfg["bolt_speed_w"] * W
    assert game.projectiles[0].x == pytest.approx(x0 + speed * 0.05)
    game.update(frame_state(5.0))                  # long stall: capped at 0.1 s
    # 0.05 + 0.1 s of flight from x 260: short of P2's box at 700, still flying.
    assert game.projectiles[0].x == pytest.approx(x0 + speed * 0.15)


def test_bolt_hits_through_update(game):
    game.update(frame_state(0.0, events=[(1, "punch")]))
    for i in range(1, 6):
        game.update(frame_state(i * 0.1))
    assert game.get_player(2).health == 90


def test_lost_player_is_not_hit_and_its_events_are_ignored(game):
    game.update(frame_state(0.0, events=[(1, "punch")], lost=(2,)))
    for i in range(1, 6):
        game.update(frame_state(i * 0.1, events=[(2, "punch")], lost=(2,)))
    assert game.get_player(2).health == 100
    assert not any(b.owner_id == 2 for b in game.projectiles)


def test_event_without_shoulders_is_ignored(game):
    s = frame_state(0.0, events=[(1, "punch")])
    s.motion[1].valid[sk.R_SHOULDER] = False
    game.update(s)
    assert game.projectiles == [] and game.actions == []


# --- Debug view and CSV columns --------------------------------------------

def busy_game(game) -> Game:
    """Every kind of object on screen at once."""
    game.update(frame_state(0.0, events=[(1, "punch"), (1, "hands_up"), (2, "arms_crossed")]))
    game.update(frame_state(0.1))
    game.get_player(2).paint_splats.append(PaintSplat(0.5, 0.5, owner_id=1))
    return game


def test_draw_debug_draws_every_object(game):
    busy_game(game)
    state = frame_state(0.1)
    canvas = state.frame.image.copy()
    game.draw_debug(canvas, state)
    assert canvas.any()
    game.winner_id = 1
    game.draw_debug(canvas, state)        # winner text too, no errors


def test_log_fields(game):
    busy_game(game)
    row = game.log_fields(frame_state(0.1))
    assert row["game_p1_health"] == 100 and row["game_p2_health"] == 100
    assert row["game_p2_shield"] == 1 and row["game_p1_shield"] == 0
    assert row["game_bolts"] == 1 and row["game_beams"] == 1
    assert row["game_winner"] == ""
    game.update(frame_state(0.2, events=[(2, "punch")]))   # P1's beam is alive; P2 is free
    assert game.log_fields(frame_state(0.2))["game_p2_actions"] == "bolt"


def test_bolt_age_drives_its_animation(arena):
    arena.attack(300, 500, 1, player_id=1)
    arena.advance(0.1, {1, 2})
    assert arena.projectiles[0].age == pytest.approx(0.1)


def test_bolt_end_is_kept_where_the_bolt_stopped(arena):
    arena.activate_shield(700, 500, -1, player_id=2)
    arena.attack(300, 500, 1, player_id=1)       # blocked by P2's shield
    arena.attack(300, 100, 1, player_id=1)       # above both boxes: leaves the frame
    arena.advance(0.3, {1, 2})
    assert sorted((round(e.x), e.y) for e in arena.bolt_ends) == [(720, 500)]
    arena.advance(0.3, {1, 2})                   # second bolt reaches the edge
    assert (1000.0, 100) in [(e.x, e.y) for e in arena.bolt_ends]
    arena.advance(arena.cfg["bolt_end_s"], {1, 2})
    assert all(e.y == 100 for e in arena.bolt_ends)      # the first end is over
    arena.reset()
    assert arena.bolt_ends == []


def test_torso_and_drawing_values(game):
    game.update(frame_state(0.0))
    x, shoulder_y, hip_y, sw = game.torso(frame_state(0.0).motion[1])
    assert (x, shoulder_y, hip_y, sw) == (200.0, 300.0, 600.0, 100.0)
    assert game.beam_life_s == pytest.approx(game.beam_charge_s + game.cfg["beam_duration_s"])
    assert game.beam_band_px() == pytest.approx(game.cfg["beam_hit_radius_h"] * H)
    shield = Shield(0, 0, 1, remaining=0.5)
    assert game.shield_opacity(shield) == 1.0
    shield.remaining = -game.shield_fade_s / 2
    assert game.shield_opacity(shield) == pytest.approx(0.5)
    shield.remaining = -game.shield_fade_s * 2
    assert game.shield_opacity(shield) == 0.0


@pytest.mark.parametrize("xs, p1_dir", [
    ((200.0, 800.0), 1),      # P1 left of P2: fires right
    ((800.0, 200.0), -1),     # switched sides: P1 fires left, towards P2
    ((600.0, 900.0), 1),      # both in the right half: P1 still fires right, at P2
    ((400.0, 100.0), -1),     # both in the left half: P1 fires left, at P2
])
def test_attacks_fly_towards_the_other_player(game, xs, p1_dir):
    game.update(frame_state(0.0, events=[(1, "hands_up")], xs=xs))
    assert game.beams[0].direction == p1_dir
    game.update(frame_state(0.1, events=[(2, "hands_up")], xs=xs))
    assert game.beams[1].direction == -p1_dir


def test_alone_attacks_fly_towards_the_middle(game):
    game.update(frame_state(0.0, events=[(1, "hands_up")], xs=(800.0, 200.0), lost=(2,)))
    assert game.beams[0].direction == -1          # right half, no one else in view


# --- Phases (feature 12): waiting -> calibration -> countdown -> playing -> result

@pytest.fixture
def intro() -> Game:
    return Game({**load_config()["game"], "intro": True})


def with_faces(state: FrameState, pids=(1, 2)) -> FrameState:
    """A tracked face for each active player in pids."""
    for pid in pids:
        p = state.players.get(pid)
        if p is not None and p.status == "active":
            c = np.array([p.bbox[0] + 100.0, 250.0])
            state.faces[pid] = Face(pid=pid, valid=True, landmarks=None, center=c, scale=60.0,
                                    roll=0.0, yaw=0.0, anchors={"above_head": c - (0, 80)})
    return state


def hands_up(state: FrameState, pids=(1, 2)) -> FrameState:
    """The nose below the raised wrists (y 200): hands up for pids."""
    for pid in pids:
        state.motion[pid].kpts[sk.NOSE] = (state.motion[pid].kpts[sk.L_SHOULDER][0] - 50, 250)
    return state


def run(game: Game, t0: float, t1: float, make=lambda t: frame_state(t), dt: float = 0.1) -> float:
    """Update the game every dt from t0 up to and including t1; returns t1."""
    for i in range(round((t1 - t0) / dt) + 1):
        game.update(make(t0 + i * dt))
    return t1


def calibrated(game: Game, t: float = 0.0) -> float:
    """Through waiting and calibration into the countdown; returns the time."""
    t = run(game, t, t + game.cfg["ready_s"], lambda t: with_faces(frame_state(t)))
    assert game.phase == "calibration"
    for move in CALIBRATION_MOVES:
        t += 0.1
        game.update(frame_state(t, events=[(1, move), (2, move)]))
    assert game.phase == "countdown"
    return t


def test_intro_waits_until_both_places_are_taken(intro):
    assert intro.phase == "waiting"
    run(intro, 0.0, 3.0)                                            # no faces
    assert intro.phase == "waiting"
    run(intro, 3.1, 6.0, lambda t: with_faces(frame_state(t), pids=(1,)))   # P2 has no face
    assert intro.phase == "waiting"
    run(intro, 6.0, 8.0, lambda t: with_faces(frame_state(t, xs=(600.0, 800.0))))
    assert intro.phase == "waiting"                                 # both in the right half
    run(intro, 6.1, 6.1 + intro.cfg["ready_s"] - 0.2, lambda t: with_faces(frame_state(t)))
    assert intro.phase == "waiting"                                 # not long enough yet
    run(intro, 6.1 + intro.cfg["ready_s"] - 0.1, 6.1 + intro.cfg["ready_s"],
        lambda t: with_faces(frame_state(t)))
    assert intro.phase == "calibration"


def test_hold_survives_short_gaps():
    from ipcvgame.interaction.game import _Hold
    h = _Hold()
    assert h.update(True, 0.0, 0.3) == 0.0
    assert h.update(True, 0.1, 0.3) == pytest.approx(0.1)
    assert h.update(False, 0.2, 0.3) == pytest.approx(0.1)         # a short gap: held, not growing
    assert h.update(True, 0.35, 0.3) == pytest.approx(0.35)        # still the same hold
    h.update(False, 0.5, 0.3)
    assert h.update(False, 0.7, 0.3) == 0.0                         # gap too long: over
    assert h.update(True, 0.8, 0.3) == 0.0                          # starts again


def test_calibration_ticks_moves_in_order_and_fires_nothing(intro):
    t = run(intro, 0.0, intro.cfg["ready_s"], lambda t: with_faces(frame_state(t)))
    assert intro.current_move(1) == intro.current_move(2) == "punch"
    intro.update(frame_state(t + 0.1, events=[(1, "hands_up")]))   # not the current move
    assert intro.current_move(1) == "punch" and intro.actions == []
    intro.update(frame_state(t + 0.2, events=[(1, "punch")]))
    assert intro.current_move(1) == "hands_up" and intro.get_player(1).move_t == t + 0.2
    assert intro.actions == [(1, "learned_punch")]
    for move in ("hands_up", "arms_crossed"):
        intro.update(frame_state(t + 0.3, events=[(1, move)]))
    assert intro.current_move(1) is None and intro.current_move(2) == "punch"
    assert intro.phase == "calibration"                             # P2 not done yet
    assert intro.projectiles == [] and intro.beams == [] and intro.shields == {}


def test_countdown_then_playing(intro):
    t = calibrated(intro)
    intro.update(frame_state(t + 0.1, events=[(1, "punch")]))
    assert intro.projectiles == []                                  # nothing fires in the countdown
    t = run(intro, t + 0.2, t + intro.countdown_s)
    assert intro.phase == "playing"
    intro.update(frame_state(t + 0.1, events=[(1, "punch")]))
    assert intro.actions == [(1, "bolt")]


def test_win_shows_result_and_hands_up_starts_the_next_round(intro):
    t = calibrated(intro)
    t = run(intro, t + 0.1, t + intro.countdown_s)
    intro.get_player(2).health = intro.cfg["bolt_damage"]
    intro.update(frame_state(t + 0.1, events=[(1, "punch")]))
    t = run(intro, t + 0.2, t + 1.0)
    assert intro.phase == "result" and intro.winner_id == 1
    hold = intro.cfg["replay_hold_s"]
    t = run(intro, t + 0.1, t + 0.1 + hold, lambda t: hands_up(frame_state(t), pids=(1,)))
    assert intro.phase == "result"                                  # the winner alone
    t = run(intro, t + 0.1, t + 0.1 + hold - 0.2, lambda t: hands_up(frame_state(t)))
    assert intro.phase == "result"                                  # not long enough yet
    t = run(intro, t + 0.1, t + 0.2, lambda t: hands_up(frame_state(t)))
    assert intro.phase == "countdown"                               # calibration is kept
    assert intro.winner_id is None and intro.get_player(2).health == intro.max_health
    assert intro.current_move(1) is None and intro.current_move(2) is None


def test_new_round_and_reset_with_intro(intro):
    calibrated(intro)
    intro.get_player(1).health = 30
    intro.new_round()
    assert intro.phase == "countdown" and intro.get_player(1).health == intro.max_health
    assert intro.current_move(1) is None                            # calibration kept
    intro.reset()
    assert intro.phase == "waiting" and intro.current_move(1) == "punch"


def test_without_intro_rounds_start_at_once(game):
    assert game.phase == "playing"
    game.new_round()
    assert game.phase == "playing"
    game.reset()
    assert game.phase == "playing"


def test_phase_in_log_and_debug_view(intro):
    t = run(intro, 0.0, intro.cfg["ready_s"], lambda t: with_faces(frame_state(t)))
    row = intro.log_fields(frame_state(t))
    assert row["game_phase"] == "calibration" and row["game_p1_move"] == "punch"
    canvas = np.zeros((H, W, 3), np.uint8)
    intro.draw_debug(canvas, frame_state(t))
    assert canvas.any()


def test_solo_intro_needs_only_p1():
    solo = Game({**load_config()["game"], "intro": True, "round_players": 1})
    alone = lambda t: with_faces(frame_state(t, lost=(2,)))   # noqa: E731
    t = run(solo, 0.0, solo.cfg["ready_s"], alone)
    assert solo.phase == "calibration"
    for move in CALIBRATION_MOVES:
        t += 0.1
        s = alone(t)
        s.events = [GestureEvent(1, move, t, 1.0), GestureEvent(2, move, t, 1.0)]
        solo.update(s)
    assert solo.phase == "countdown"
    assert all(pid == 1 for pid, _ in solo.actions)          # only P1 is calibrated
    run(solo, t + 0.1, t + solo.countdown_s, alone)
    assert solo.phase == "playing"


def test_places_are_taken_by_position_not_by_player_number(intro):
    state = with_faces(frame_state(0.0, xs=(800.0, 200.0)))       # P1 stands right, P2 left
    assert intro.place_taken(state, 1) and intro.place_taken(state, 2)
    state = with_faces(frame_state(0.0, xs=(800.0, 200.0)), pids=(1,))
    assert not intro.place_taken(state, 1) and intro.place_taken(state, 2)   # only P1, on the right
