import numpy as np

from ipcvgame.core.config import load_config
from ipcvgame.scene.squash.physics import Ball, Court, aim, height_at, step, swept_hit
from ipcvgame.scene.squash.rally import Rally

CFG = load_config()["squash"]
COURT = Court.from_cfg(CFG)


def fly(ball: Ball, until_y: float, dt: float = 1 / 60, max_t: float = 5.0) -> list[str]:
    events, t = [], 0.0
    while ball.pos[1] > until_y and t < max_t:
        events += step(ball, COURT, dt)
        t += dt
    return events


def test_floor_bounce_loses_energy():
    ball = Ball(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    for _ in range(120):
        step(ball, COURT, 1 / 60)
    peak = 0.0
    for _ in range(120):
        step(ball, COURT, 1 / 60)
        peak = max(peak, ball.pos[2])
    assert peak < 1.0 * COURT.floor_restitution ** 2 + 0.05


def test_front_wall_reflects():
    ball = Ball(np.array([0.0, COURT.wall_y - 0.5, 2.0]), np.array([0.0, 10.0, 0.0]))
    events = []
    for _ in range(30):
        events += step(ball, COURT, 1 / 60)
    assert "front_wall" in events and ball.vel[1] < 0


def test_swept_hit_catches_fast_ball_between_frames():
    hand = np.array([0.0, 0.0, 1.0])
    # ball jumps from 1 m in front to 1 m behind the hand in one frame
    assert swept_hit(np.array([0.0, 1.0, 1.0]), np.array([0.0, -1.0, 1.0]), hand, hand, 0.1)
    assert not swept_hit(np.array([0.5, 1.0, 1.0]), np.array([0.5, -1.0, 1.0]), hand, hand, 0.1)


def test_height_at_matches_free_fall_without_bounce():
    z, n = height_at(2.0, 1.0, 0.3, 0.06, 0.5)
    assert n == 0 and abs(z - (2.0 + 0.3 - 0.5 * 9.81 * 0.09)) < 1e-9


def test_aim_reaches_target_after_one_bounce():
    start = np.array([1.0, COURT.wall_y - COURT.ball_radius, 2.2])
    target = np.array([-0.8, CFG["hit_y"], 1.3])
    ball = Ball(start.copy(), aim(start, target, 4.5, COURT))
    events = fly(ball, target[1])
    assert events.count("floor") == 1
    assert np.linalg.norm(ball.pos - target) < 0.1


def test_still_hand_return_clears_the_tin():
    """A ball returned with zero hand speed must reach the wall above the tin."""
    r = Rally(CFG, seed=0)
    r.serve()
    r.ball.pos = np.array([0.0, CFG["hit_y"], 1.3])
    r._return_ball(np.zeros(3))
    ball = Ball(r.ball.pos.copy(), r.ball.vel.copy())
    events = []
    while "front_wall" not in events and "floor" not in events:
        events += step(ball, COURT, 1 / 240)
    assert "front_wall" in events and ball.pos[2] > CFG["tin_height"]


def run_bot(rally: Rally, seconds: float, track: bool) -> list[str]:
    """Hand follows the ball on the hit plane (track=True) or stays put."""
    hand = np.array([0.0, CFG["hit_y"], 1.3])
    events = []
    for _ in range(int(seconds * 60)):
        new = hand.copy()
        if track and rally.state == "incoming":
            new[0], new[2] = rally.ball.pos[0], rally.ball.pos[2]
        events += rally.update(1 / 60, hand, new, (new - hand) * 60)
        hand = new
    return events


def test_rally_continues_with_perfect_tracking():
    r = Rally(CFG, seed=1)
    events = run_bot(r, 20.0, track=True)
    assert events.count("hit") >= 8 and "miss" not in events


def test_rally_ends_in_a_miss_without_hitting():
    r = Rally(CFG, seed=2)
    hand_far = np.array([10.0, 0.0, 10.0])
    events = []
    for _ in range(int(6 * 60)):
        events += r.update(1 / 60, hand_far, hand_far, np.zeros(3))
    assert "serve" in events and "miss" in events and r.count == 0


def test_serve_after_a_miss_starts_a_clean_rally():
    """Regression: the floor-bounce count of the missed rally leaked into the
    next serve, so its first bounce counted as a second bounce."""
    r = Rally(CFG, seed=3)
    hand_far = np.array([10.0, 0.0, 10.0])
    while r.state != "missed":
        r.update(1 / 60, hand_far, hand_far, np.zeros(3))
    r.serve()
    events = run_bot(r, 20.0, track=True)
    assert events.count("hit") >= 8 and "miss" not in events
