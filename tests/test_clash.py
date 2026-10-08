import numpy as np

from ipcvgame.core.config import load_config
from ipcvgame.scene.clash.combat import Combat
from ipcvgame.scene.clash.control import KeyboardControl

CFG = load_config()["clash"]
CHEST = 1.2


def fighting() -> Combat:
    c = Combat(CFG)
    c.update(CFG["countdown"] + 1e-3)
    assert c.state == "fight"
    return c


def palm(c: Combat, pid: int) -> np.ndarray:
    x, y = c.fighters[pid].pos
    return np.array([x, y, CHEST])


def run(c: Combat, seconds: float, dt: float = 1 / 60) -> list:
    events = []
    for _ in range(int(seconds / dt)):
        events += c.update(dt)
    return events


def test_no_fire_during_countdown():
    c = Combat(CFG)
    assert not c.fire(1, palm(c, 1)) and not c.bolts


def test_bolt_in_line_hits_and_damages():
    c = fighting()
    c.move(1, (-2.5, 0.5))
    c.move(2, (2.5, 0.5))
    c.update(0.0)
    assert c.fire(1, palm(c, 1))
    events = run(c, 1.0)
    hits = [e for e in events if e.name == "hit"]
    assert len(hits) == 1 and hits[0].pid == 2
    assert c.fighters[2].damage == CFG["bolt_damage"] and c.fighters[1].damage == 0
    assert not c.bolts


def test_bolt_out_of_line_fizzles_at_the_edge():
    c = fighting()
    c.move(1, (-2.5, -1.5))
    c.move(2, (2.5, 1.5))       # 3 m deeper: well outside the body spheres
    c.update(0.0)
    c.fire(2, palm(c, 2))
    events = run(c, 1.5)
    assert [e.name for e in events] == ["fizzle"] and events[0].pid == 2
    assert c.fighters[1].damage == 0


def test_dodge_mid_flight_is_caught_by_the_swept_test_only_if_in_line():
    """The victim steps out of the line while the bolt is in the air."""
    c = fighting()
    c.move(1, (-2.5, 0.0))
    c.move(2, (2.5, 0.0))
    c.update(0.0)
    c.fire(1, palm(c, 1))
    run(c, 0.2)                 # bolt still ~3 m away
    c.move(2, (2.5, 1.5))
    events = run(c, 1.0)
    assert "hit" not in [e.name for e in events]


def test_fast_bolt_does_not_skip_through_body_in_one_long_frame():
    c = fighting()
    c.move(1, (-2.5, 0.0))
    c.move(2, (2.5, 0.0))
    c.update(0.0)
    c.fire(1, palm(c, 1))
    events = c.update(1.0)      # bolt jumps ~9 m, far past the body
    assert [e.name for e in events] == ["hit"]


def test_cooldown_blocks_refire():
    c = fighting()
    assert c.fire(1, palm(c, 1))
    assert not c.fire(1, palm(c, 1))
    run(c, CFG["bolt_cooldown"] + 0.02)
    assert c.fire(1, palm(c, 1))


def test_full_coverage_wins_and_stops_damage():
    c = fighting()
    shots = int(np.ceil(1.0 / CFG["bolt_damage"] - 1e-9))
    events = []
    for _ in range(shots):
        c.fire(2, palm(c, 2))
        events += run(c, max(CFG["bolt_cooldown"], 1.0))
    assert c.state == "over" and c.winner == 2 and c.fighters[1].damage == 1.0
    assert [e.name for e in events].count("win") == 1
    assert not c.fire(2, palm(c, 2))
    c.reset()
    assert c.state == "countdown" and c.fighters[1].damage == 0 and c.winner is None


def test_players_move_only_in_depth_and_stay_on_the_field():
    c = fighting()
    a = c.arena
    c.move(1, (5.0, 9.0))
    c.move(2, (-5.0, -9.0))
    p1, p2 = c.fighters[1].pos, c.fighters[2].pos
    assert p1[0] == a.start(1)[0] and p2[0] == a.start(2)[0]   # x never changes
    assert p1[1] == a.half_depth - a.edge_margin and p2[1] == -(a.half_depth - a.edge_margin)


def test_keyboard_walks_at_constant_speed_and_is_clamped():
    c = Combat(CFG)
    down = {"w", "a", "d"}     # A/D do nothing: movement is 1D
    kb = KeyboardControl(CFG, lambda k: k in down, c.arena.start(2), lambda p: c.arena.clamp(2, p))
    start = kb.pos.copy()
    p = kb.read(0.1)
    assert np.allclose(p - start, [0.0, CFG["key_speed"] * 0.1])
    for _ in range(200):
        p = kb.read(0.1)
    assert np.allclose(p, c.arena.clamp(2, np.array([0.0, 99.0])))
    down.add("s")              # W + S cancel
    assert np.allclose(kb.read(0.1), p)
