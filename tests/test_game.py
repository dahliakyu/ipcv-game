"""Colour Clash game logic (game.py), in pixels."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Face, Frame, FrameState, GestureEvent, Motion, Player, PoseDetection
from ipcvgame.interaction.game import CALIBRATION_MOVES, Game, Hit, PaintSplat, Shield
from ipcvgame.scene.draw import player_color

# The made-up frames below have no faces, so the facing-the-camera rule is
# off unless a test turns it on (test_gesture_counts_only_facing_the_camera).
NO_FACING = {"act_facing_camera": False}


@pytest.fixture
def game() -> Game:
    """Without the intro: playing from the first frame."""
    return Game({**load_config()["game"], "intro": False, **NO_FACING})


@pytest.fixture
def arena() -> Game:
    """The setup of test_game.py in a 1000 x 1000 px frame, so its 0-1
    coordinates become px by multiplying by 1000: P1 on the left, P2 right.
    With game.py's bolt speed (1.4), so its test numbers still hold."""
    g = Game({**load_config()["game"], "bolt_speed_w": 1.4, **NO_FACING})
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


# --- Hits for the face effects (merge_log.md, "Face effects") -------------

def test_bolt_hit_is_recorded_with_its_direction(arena):
    arena.attack(300, 500, 1, player_id=1)
    arena.attack(600, 500, -1, player_id=2)
    arena.advance(0.3, {1, 2})
    assert sorted(arena.hits, key=lambda h: h.target_id) == [Hit(1, 2, -1, 10), Hit(2, 1, 1, 10)]
    assert sorted(arena.actions) == [(1, "hit"), (2, "hit")]


def test_beam_hit_is_recorded_once(arena):
    arena.beam_attack(300, 500, 1, player_id=1)
    arena.advance(0.35, {1, 2})
    arena.advance(0.15, {1, 2})
    arena.advance(0.1, {1, 2})
    assert arena.hits == [Hit(2, 1, 1, arena.cfg["beam_damage"])]


def test_blocked_hit_is_not_recorded(arena):
    arena.activate_shield(700, 500, -1, player_id=2)
    arena.attack(300, 500, 1, player_id=1)
    arena.advance(0.3, {1, 2})
    assert arena.hits == []


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


def test_hits_last_one_frame(game):
    game.update(frame_state(0.0, events=[(1, "punch")]))
    hit_frames = []
    for i in range(1, 8):
        game.update(frame_state(i * 0.1))
        if game.hits:
            hit_frames.append(i)
            assert game.hits == [Hit(2, 1, 1, 10)]
            assert game.log_fields(frame_state(i * 0.1))["game_p2_actions"] == "hit"
    assert len(hit_frames) == 1


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


# --- Phases (game start): waiting -> calibration -> countdown -> playing -> result

@pytest.fixture
def intro() -> Game:
    return Game({**load_config()["game"], "intro": True, **NO_FACING})


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


def neutral(state: FrameState, pids=(1, 2)) -> FrameState:
    """The neutral pose for pids: arms hanging down, wrists 1.6 sw below the
    shoulders, standing still. Shoulders 100 px apart, each arm 1.6 sw
    (80.6 + 80.2 px)."""
    for pid in pids:
        k = state.motion[pid].kpts
        cx = k[sk.L_SHOULDER][0] - 50
        k[sk.L_ELBOW], k[sk.R_ELBOW] = (cx + 60, 380), (cx - 60, 380)
        k[sk.L_WRIST], k[sk.R_WRIST] = (cx + 65, 460), (cx - 65, 460)
    return state


def in_place(t: float) -> FrameState:
    return with_faces(frame_state(t))


def standing(t: float) -> FrameState:
    return neutral(with_faces(frame_state(t)))


def to_calibration(game: Game, t: float = 0.0) -> float:
    t = run(game, t, t + game.cfg["ready_s"], in_place)
    assert game.phase == "calibration"
    return t


def measured(game: Game, t: float, make=standing) -> float:
    """Hold the neutral pose until it is measured; returns the time."""
    return run(game, t + 0.1, t + 0.2 + game.cfg["neutral_s"], make)


def calibrated(game: Game, t: float = 0.0) -> float:
    """Through waiting and calibration into the countdown; returns the time."""
    t = measured(game, to_calibration(game, t))
    for move in CALIBRATION_MOVES[1:]:
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
    t = to_calibration(intro)
    assert intro.current_move(1) == intro.current_move(2) == "neutral"
    intro.update(frame_state(t + 0.1, events=[(1, "punch")]))      # not the current move
    assert intro.current_move(1) == "neutral" and intro.actions == []
    t = measured(intro, t + 0.1)
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
    assert intro.get_player(1).shoulder_px == pytest.approx(100.0)
    intro.reset()
    assert intro.phase == "waiting" and intro.current_move(1) == "neutral"
    assert intro.get_player(1).shoulder_px is None and intro.arm_sw(1) is None


def test_without_intro_rounds_start_at_once(game):
    assert game.phase == "playing"
    game.new_round()
    assert game.phase == "playing"
    game.reset()
    assert game.phase == "playing"


def test_phase_in_log_and_debug_view(intro):
    t = to_calibration(intro)
    row = intro.log_fields(frame_state(t))
    assert row["game_phase"] == "calibration" and row["game_p1_move"] == "neutral"
    assert row["game_p1_arm_sw"] == ""
    canvas = np.zeros((H, W, 3), np.uint8)
    intro.draw_debug(canvas, frame_state(t))
    assert canvas.any()
    measured(intro, t)
    row = intro.log_fields(frame_state(t))
    assert row["game_p1_shoulder_px"] == 100.0 and row["game_p1_arm_sw"] == pytest.approx(1.608, abs=1e-3)
    intro.draw_debug(canvas, frame_state(t))


def test_solo_intro_needs_only_p1():
    solo = Game({**load_config()["game"], "intro": True, "round_players": 1})
    alone = lambda t: with_faces(frame_state(t, lost=(2,)))   # noqa: E731
    t = run(solo, 0.0, solo.cfg["ready_s"], alone)
    assert solo.phase == "calibration"
    t = measured(solo, t, lambda t: neutral(alone(t), pids=(1,)))
    for move in CALIBRATION_MOVES[1:]:
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


# --- Pose calibration (game start, step 4) ----------------------------------

def test_neutral_pose_measures_the_medians(intro):
    t = to_calibration(intro)

    def noisy(t):
        s = standing(t)
        if round(t * 10) % 7 == 0:                                  # an odd frame: a wrist jumps
            s.motion[1].kpts[sk.L_WRIST] += (0, 40)
        return s
    measured(intro, t, noisy)
    p1 = intro.get_player(1)
    assert p1.shoulder_px == pytest.approx(100.0)
    assert p1.arm_px == pytest.approx(80.62 + 80.16, abs=0.01)     # the median ignores the odd frames
    assert intro.arm_sw(1) == pytest.approx(1.608, abs=1e-3) and not intro.short_arm(1)
    assert intro.actions == [] or all(a == "learned_neutral" for _, a in intro.actions)


@pytest.mark.parametrize("change", ["wrists_up", "moving", "turned", "no_hips", "no_face"])
def test_not_neutral(intro, change):
    t = to_calibration(intro)

    def make(t):
        s = standing(t)
        m = s.motion[1]
        if change == "wrists_up":
            m.kpts[sk.L_WRIST] = (m.kpts[sk.L_SHOULDER][0], 350)    # only 0.5 sw below the shoulders
        elif change == "moving":
            m.vel[sk.R_WRIST] = (150.0, 0.0)                        # 1.5 sw/s
        elif change == "turned":
            s.faces[1].yaw = np.radians(30)
        elif change == "no_hips":
            m.valid[sk.L_HIP] = False
        elif change == "no_face":
            del s.faces[1]
        return s
    measured(intro, t, make)
    assert intro.current_move(1) == "neutral" and intro.get_player(1).shoulder_px is None
    if change == "no_face":
        # P1 not in view: the line-up cannot be checked, so P2 waits too.
        assert intro.current_move(2) == "neutral" and intro.line_up_hint(2) == "wait"
    else:
        assert intro.current_move(2) == "punch"                     # P2 stood still


def test_neutral_hold_starts_over_after_a_long_break(intro):
    t = to_calibration(intro)
    half = intro.cfg["neutral_s"] / 2
    t = run(intro, t + 0.1, t + 0.1 + half, standing)
    assert 0.4 < intro.neutral_fraction(1) < 0.6
    t = run(intro, t + 0.1, t + 1.0, in_place)                     # arms up for 1 s
    assert intro.neutral_fraction(1) == 0.0
    run(intro, t + 0.1, t + 0.1 + half, standing)
    assert intro.current_move(1) == "neutral"                       # half again is not enough


def test_short_arm_is_flagged(intro):
    t = to_calibration(intro)

    def short(t):
        s = standing(t)
        for pid in (1, 2):
            k = s.motion[pid].kpts
            cx = k[sk.L_SHOULDER][0] - 50
            k[sk.L_ELBOW], k[sk.L_WRIST] = (cx + 50, 360), (cx + 50, 420)   # 0.6 + 0.6 sw
            k[sk.R_ELBOW], k[sk.R_WRIST] = (cx - 50, 360), (cx - 50, 420)
        return s
    measured(intro, t, short)
    assert intro.arm_sw(1) == pytest.approx(1.2) and intro.short_arm(1)


# --- Line-up in the calibration (merge_log.md, "Game start: Modifications")

def depths(state: FrameState, d: dict[int, float]) -> FrameState:
    for pid, depth in d.items():
        state.players[pid].pos = np.array([0.0, depth])
    return state


def test_level_by_depth_allows_measuring(intro):
    t = to_calibration(intro)
    measured(intro, t, lambda t: depths(standing(t), {1: 2.5, 2: 2.7}))
    assert intro.current_move(1) == intro.current_move(2) == "punch"
    assert intro.level and intro.line_gap == pytest.approx(0.2)


def test_not_level_sends_the_nearer_player_back(intro):
    t = to_calibration(intro)
    measured(intro, t, lambda t: depths(standing(t), {1: 2.0, 2: 2.6}))
    assert intro.current_move(1) == intro.current_move(2) == "neutral"
    assert not intro.level and intro.nearer == 1
    assert intro.line_up_hint(1) == "back" and intro.line_up_hint(2) == "wait"
    assert intro.log_fields(standing(t))["game_level"] == 0
    t = measured(intro, t + 3, lambda t: depths(standing(t), {1: 2.4, 2: 2.6}))
    assert intro.level and intro.current_move(1) == intro.current_move(2) == "punch"
    assert intro.line_up_hint(1) is None                                   # nobody needs the line-up now


def test_measured_player_still_told_to_step_back(intro):
    """P1 is measured; P2 still needs the neutral pose and P1 steps forward:
    P1's (punch) panel tells them to step back."""
    t = to_calibration(intro)
    t = measured(intro, t, lambda t: neutral(with_faces(frame_state(t)), pids=(1,)))
    assert intro.current_move(1) == "punch" and intro.current_move(2) == "neutral"
    intro.update(depths(standing(t + 0.1), {1: 1.9, 2: 2.6}))
    assert intro.line_up_hint(1) == "back" and intro.line_up_hint(2) == "wait"


def test_solo_has_no_line_up():
    g = Game({**load_config()["game"], "intro": True, "round_players": 1, **NO_FACING})
    t = run(g, 0.0, g.cfg["ready_s"], in_place)
    measured(g, t, lambda t: depths(standing(t), {1: 1.0, 2: 3.0}))
    assert g.level and g.current_move(1) == "punch"


def test_line_up_hysteresis(intro):
    """Not level at 0.5 m; at 0.35 m (under 0.4 m, but over 0.75 x 0.4 m)
    still not level; at 0.25 m level again."""
    t = to_calibration(intro)
    for gap, level in ((0.5, False), (0.35, False), (0.25, True), (0.35, True)):
        t += 0.1
        intro.update(depths(standing(t), {1: 2.0, 2: 2.0 + gap}))
        assert intro.level == level, gap


# --- Gestures count only facing the camera (merge_log.md, "Game start: Modifications")

def facing_game(**over) -> Game:
    return Game({**load_config()["game"], "intro": False, **over})


def test_gesture_counts_only_facing_the_camera():
    g = facing_game()
    g.update(frame_state(0.0, events=[(1, "punch")]))                  # no face: does not count
    assert g.actions == [(1, "not_facing_punch")] and not g.projectiles
    assert g.not_facing_t == {1: 0.0}
    s = with_faces(frame_state(0.1, events=[(1, "punch"), (2, "punch")]))
    s.faces[2].yaw = np.radians(45)                                    # head turned away
    s.motion[1].kpts[sk.L_WRIST] = (390, 310)
    g.update(s)
    assert g.actions == [(1, "bolt"), (2, "not_facing_punch")]
    assert facing_game(act_facing_camera=False).facing_camera(frame_state(0.0), 1)


def test_body_turn_from_shoulders_against_torso_with_arms_down():
    """Calibrated: shoulders 100 px, torso 300 px. Arms down with shoulders
    50 px: turned 60 degrees, so a gesture does not count; with the arms up
    the shoulders' width is not used."""
    g = facing_game()
    player = g.get_player(1)
    player.shoulder_px, player.torso_px = 100.0, 300.0

    def turned(t, events=(), arms_down=True):
        s = with_faces(frame_state(t, events=events))
        if arms_down:
            neutral(s, pids=(1,))
        k = s.motion[1].kpts
        k[sk.L_SHOULDER, 0], k[sk.R_SHOULDER, 0] = 225, 175               # 50 px apart
        return s
    g.update(turned(0.0))
    assert g.body_yaw[1] == pytest.approx(60.0)
    g.update(turned(0.1, events=[(1, "punch")], arms_down=False))
    assert g.actions == [(1, "not_facing_punch")]
    g.update(neutral(with_faces(frame_state(0.2)), pids=(1,)))           # facing again, arms down
    assert g.body_yaw[1] == pytest.approx(0.0)
    s = with_faces(frame_state(0.3, events=[(1, "punch")]))
    s.motion[1].kpts[sk.L_WRIST] = (390, 310)
    g.update(s)
    assert g.actions == [(1, "bolt")]


def test_body_not_checked_before_calibration():
    g = facing_game()
    g.update(neutral(with_faces(frame_state(0.0)), pids=(1,)))
    assert 1 not in g.body_yaw


def test_calibration_move_counts_only_facing_the_camera():
    g = Game({**load_config()["game"], "intro": True})
    t = run(g, 0.0, g.cfg["ready_s"], in_place)
    t = measured(g, t)
    assert g.current_move(1) == "punch"
    g.update(frame_state(t + 0.1, events=[(1, "punch")]))                # no face
    assert g.current_move(1) == "punch" and g.actions == [(1, "not_facing_punch")]
    g.update(with_faces(frame_state(t + 0.2, events=[(1, "punch")])))
    assert g.current_move(1) == "hands_up"


# --- Drinks (consumables, extra_features.md feature 2) ---------------------

def drink_game(**over) -> Game:
    g = Game({**load_config()["game"], "intro": False, "drink_seed": 1, **NO_FACING, **over})
    g.frame_w, g.frame_h = 1000, 1000
    return g


def test_drinks_fall_one_by_one_up_to_the_maximum():
    g = drink_game(drink_every_s=[1.0, 1.0], drink_fall_s=100.0)
    g.advance(0.9)
    assert g.drinks == []
    g.advance(0.2)
    assert len(g.drinks) == 1
    for _ in range(10):
        g.advance(1.0)
    assert len(g.drinks) == g.cfg["drink_max"]


def test_drink_starts_above_the_frame_and_falls_at_constant_speed():
    g = drink_game(drink_every_s=[0.0, 0.0], drink_max=1)
    g.advance(0.0)
    (d,) = g.drinks
    lo, hi = g.cfg["drink_margin_w"], 1 - g.cfg["drink_margin_w"]
    assert lo * 1000 <= d.x <= hi * 1000
    assert d.y == pytest.approx(-g.drink_height(d) / 2)
    y0 = d.y
    g.advance(0.5)
    g.advance(0.5)
    assert d.y == pytest.approx(y0 + 1000 / g.cfg["drink_fall_s"])


def test_drink_is_gone_below_the_frame():
    g = drink_game(drink_every_s=[0.0, 0.0], drink_max=1)
    g.advance(0.0)
    d = g.drinks[0]
    g._drink_wait = 1e9                                   # no new drinks
    for _ in range(int(g.cfg["drink_fall_s"] * 10) + 20):
        g.advance(0.1)
    assert d not in g.drinks and g.drinks == []


def test_bigger_drinks_give_bigger_values():
    g = drink_game()
    lo, hi = g.cfg["drink_size"]
    seen = {}
    for _ in range(300):
        d = g._spawn_drink()
        v0, v1 = g.cfg[f"drink_{d.kind}"]
        assert d.value == pytest.approx(v0 + (v1 - v0) * (d.size - lo) / (hi - lo))
        assert lo <= d.size <= hi
        seen[d.kind] = seen.get(d.kind, 0) + 1
    assert set(seen) == {"hp", "attack", "defense"}


def test_drinks_only_while_playing_and_cleared_by_a_new_round():
    g = Game({**load_config()["game"], "drink_every_s": [0.0, 0.0], **NO_FACING})   # intro: waiting
    for i in range(5):
        g.update(frame_state(i * 0.1))
    assert g.phase == "waiting" and g.drinks == []
    g = drink_game(drink_every_s=[0.0, 0.0])
    for i in range(3):
        g.update(frame_state(i * 0.1))
    assert g.drinks and g.log_fields(frame_state(0.3))["game_drinks"] == len(g.drinks)
    g.new_round()
    assert g.drinks == []


def test_same_seed_same_drinks():
    def drinks():
        g = drink_game(drink_every_s=[0.5, 2.0])
        for _ in range(40):
            g.advance(0.1)
        return [(round(d.x), d.kind, round(d.size, 3)) for d in g.drinks]
    assert drinks() == drinks()


def test_drinks_fall_for_a_single_player():
    """One player is enough: after the solo intro (--solo) and without the
    intro (--no-intro) with only P1 in view."""
    cfg = {**load_config()["game"], "drink_every_s": [0.5, 0.5], "drink_seed": 2}
    solo = Game({**cfg, "intro": True, "round_players": 1})
    alone = lambda t: with_faces(frame_state(t, lost=(2,)))   # noqa: E731
    t = run(solo, 0.0, solo.cfg["ready_s"], alone)
    t = measured(solo, t, lambda t: neutral(alone(t), pids=(1,)))
    for move in CALIBRATION_MOVES[1:]:
        t += 0.1
        s = alone(t)
        s.events = [GestureEvent(1, move, t, 1.0)]
        solo.update(s)
    t = run(solo, t + 0.1, t + solo.countdown_s, alone)
    assert solo.phase == "playing" and solo.drinks == []
    run(solo, t + 0.1, t + 1.0, alone)
    assert solo.drinks
    no_intro = Game({**cfg, "intro": False, **NO_FACING})
    run(no_intro, 0.0, 1.0, lambda t: frame_state(t, xs=(200.0,)))
    assert no_intro.drinks


# --- Drinking with the mouth ------------------------------------------------

from ipcvgame.interaction.game import Drink  # noqa: E402

NO_SPAWN = {"drink_every_s": [1e6, 1e6]}


def mouths(t: float, jaw: dict, xs=(200.0, 800.0), mouth_y: float = 260.0) -> FrameState:
    """A frame with a face per player in xs; jaw[pid] is its jawOpen score
    (None: not measured this frame). The mouth anchor is at (x, mouth_y)."""
    s = frame_state(t, xs=xs)
    for pid, x in enumerate(xs, start=1):
        b = {} if jaw.get(pid) is None else {"jawOpen": jaw[pid]}
        s.faces[pid] = Face(pid=pid, valid=True, landmarks=None, center=np.array([x, mouth_y - 30.0]),
                            scale=60.0, roll=0.0, yaw=0.0,
                            anchors={"mouth": np.array([x, mouth_y]),
                                     "above_head": np.array([x, mouth_y - 150.0])},
                            blendshapes=b)
    return s


def drink_at(g: Game, x: float, y: float, kind: str = "hp") -> Drink:
    d = Drink(x, y, kind, 1.0, 20.0, 0.0)
    g.drinks.append(d)
    return d


def test_mouth_open_and_closed_with_hysteresis():
    g = drink_game(**NO_SPAWN)
    for jaw, open_ in [(0.3, False), (0.6, True), (0.4, True), (None, True), (0.2, False), (0.4, False)]:
        g.update(mouths(0.0, {1: jaw}))
        assert g.mouth_open[1] is open_
    g.update(mouths(0.0, {1: 0.9}))
    s = mouths(0.1, {1: 0.9})
    s.faces[1].valid = False                              # face lost: closed
    g.update(s)
    assert g.mouth_open[1] is False


def test_drink_caught_by_the_open_mouth_and_drunk_on_closing():
    g = drink_game(**NO_SPAWN)
    g.update(mouths(0.0, {1: 0.9}))
    d = drink_at(g, 200.0, 240.0)
    g.update(mouths(0.1, {1: 0.9}))
    assert d.held_by == 1 and (1, "catch") in g.actions
    g.update(mouths(0.2, {1: 0.9}, mouth_y=300.0))            # follows the mouth
    assert d.y == pytest.approx(300.0 + g.cfg["drink_hold_rim"] * g.drink_height(d))
    g.update(mouths(0.3, {1: 0.1}))                           # closes: drinks it
    assert d not in g.drinks
    assert [(x.pid, x.kind, x.value) for x in g.drunk] == [(1, "hp", 20.0)]
    assert (1, "drink_hp") in g.actions
    g.update(mouths(0.4, {1: 0.1}))
    assert g.drunk == []                                      # for one frame


def test_talking_away_from_a_drink_drinks_nothing():
    g = drink_game(**NO_SPAWN)
    d = drink_at(g, 500.0, 260.0)                             # between the players
    for i, jaw in enumerate([0.9, 0.1, 0.9, 0.1]):
        g.update(mouths(i * 0.1, {1: jaw, 2: jaw}))
        d.y = 260.0                                           # keep it there
    assert d.held_by is None and g.drunk == []
    g = drink_game(**NO_SPAWN)
    d = drink_at(g, 200.0, 240.0)
    g.update(mouths(0.0, {1: 0.1}))                           # closed mouth: no catch
    assert d.held_by is None


def test_drink_held_too_long_falls_on_and_cannot_be_caught_again():
    g = drink_game(**NO_SPAWN)
    g.update(mouths(0.0, {1: 0.9}))
    d = drink_at(g, 200.0, 240.0)
    g.update(mouths(0.1, {1: 0.9}))
    assert d.held_by == 1
    g.update(mouths(0.1 + g.cfg["drink_hold_s"] + 0.05, {1: 0.9}))
    assert d.held_by is None and d.dropped_by == 1
    y = d.y
    g.update(mouths(0.2 + g.cfg["drink_hold_s"] + 0.05, {1: 0.9}))
    assert d.held_by is None and d.y > y                      # falls, not caught again
    d.x, d.y = 800.0, 250.0                                   # the other player can
    g.update(mouths(0.3 + g.cfg["drink_hold_s"] + 0.05, {1: 0.9, 2: 0.9}))
    assert d.held_by == 2


def test_nearer_mouth_wins_and_one_drink_per_mouth():
    g = drink_game(**NO_SPAWN)
    g.update(mouths(0.0, {1: 0.9, 2: 0.9}, xs=(200.0, 260.0)))
    d = drink_at(g, 250.0, 260.0)
    g.update(mouths(0.1, {1: 0.9, 2: 0.9}, xs=(200.0, 260.0)))
    assert d.held_by == 2
    e = drink_at(g, 255.0, 262.0)
    g.update(mouths(0.2, {1: 0.9, 2: 0.9}, xs=(200.0, 260.0)))
    assert e.held_by == 1                                     # P2's mouth is full


def test_face_lost_while_holding_drops_the_drink():
    g = drink_game(**NO_SPAWN)
    g.update(mouths(0.0, {1: 0.9}))
    d = drink_at(g, 200.0, 240.0)
    g.update(mouths(0.1, {1: 0.9}))
    s = mouths(0.2, {1: 0.9})
    del s.faces[1]
    g.update(s)
    assert d.held_by is None and g.drunk == []


def test_one_player_can_drink():
    g = drink_game(**NO_SPAWN)
    g.update(mouths(0.0, {1: 0.9}, xs=(200.0,)))
    d = drink_at(g, 200.0, 240.0, "attack")
    g.update(mouths(0.1, {1: 0.9}, xs=(200.0,)))
    g.update(mouths(0.2, {1: 0.0}, xs=(200.0,)))
    assert [(x.pid, x.kind) for x in g.drunk] == [(1, "attack")] and d not in g.drinks
    assert g.log_fields(mouths(0.2, {1: 0.0}, xs=(200.0,)))["game_p1_mouth_open"] == 0


def test_no_drinking_outside_playing():
    g = Game({**load_config()["game"], **NO_SPAWN, **NO_FACING})        # intro: waiting
    d = drink_at(g, 200.0, 240.0)
    g.update(mouths(0.0, {1: 0.9}))
    g.update(mouths(0.1, {1: 0.9}))
    assert d.held_by is None and g.mouth_open[1]
