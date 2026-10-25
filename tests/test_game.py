"""Colour Clash game logic (game.py), in pixels."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Frame, FrameState, GestureEvent, Motion, Player, PoseDetection
from ipcvgame.interaction.game import Game, PaintSplat, Shield
from ipcvgame.scene.draw import player_color


@pytest.fixture
def game() -> Game:
    return Game(load_config()["game"])


@pytest.fixture
def arena() -> Game:
    """The setup of test_game.py in a 1000 x 1000 px frame, so its 0-1
    coordinates become px by multiplying by 1000: P1 on the left, P2 right."""
    g = Game(load_config()["game"])
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


def test_punch_fires_from_the_box_edge_facing_the_other_player(game):
    game.update(frame_state(0.0, events=[(1, "punch"), (2, "punch")]))
    b1, b2 = sorted(game.projectiles, key=lambda b: b.owner_id)
    assert (b1.x, b1.y, b1.direction) == (300.0, 600.0, 1)    # P1 left: right edge
    assert (b2.x, b2.y, b2.direction) == (700.0, 600.0, -1)   # P2 right: left edge
    assert game.actions == [(1, "bolt"), (2, "bolt")]


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
    assert game.projectiles[0].x == 300.0          # first frame: dt = 0
    game.update(frame_state(0.05))
    assert game.projectiles[0].x == pytest.approx(300 + 1.4 * W * 0.05)
    game.update(frame_state(5.0))                  # long stall: capped at 0.1 s
    # 300 + 70 + 140 = 510, short of P2's box at 700: still flying.
    assert game.projectiles[0].x == pytest.approx(510.0)


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
