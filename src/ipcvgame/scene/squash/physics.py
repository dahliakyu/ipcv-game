"""Ball physics for the squash prototype. Pure NumPy, no rendering.

World frame (metres, Z up, as in Panda3D): x to the right as seen from behind
the player, y forward towards the front wall, z up. The floor is z = 0, the
front wall is y = court.wall_y, the side walls are x = +-court.half_width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81  # m/s^2


@dataclass
class Court:
    half_width: float
    wall_y: float           # front wall
    back_y: float           # back wall (behind the player)
    ball_radius: float
    floor_restitution: float
    wall_restitution: float

    @classmethod
    def from_cfg(cls, cfg: dict) -> Court:
        return cls(
            half_width=cfg["court_width"] / 2,
            wall_y=cfg["front_wall_y"],
            back_y=cfg["back_wall_y"],
            ball_radius=cfg["ball_radius"],
            floor_restitution=cfg["floor_restitution"],
            wall_restitution=cfg["wall_restitution"],
        )


@dataclass
class Ball:
    pos: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)


def step(ball: Ball, court: Court, dt: float, max_substep: float = 1 / 240) -> list[str]:
    """Advance the ball by dt with gravity and wall/floor bounces.

    Substeps keep a fast ball from passing through a wall in one long frame.
    Returns the surfaces hit, in order: "floor", "front_wall", "side_wall",
    "back_wall".
    """
    events: list[str] = []
    n = max(1, int(np.ceil(dt / max_substep)))
    h = dt / n
    r = court.ball_radius
    for _ in range(n):
        ball.vel[2] -= GRAVITY * h          # semi-implicit Euler: velocity first
        ball.pos += ball.vel * h
        if ball.pos[2] < r and ball.vel[2] < 0:
            ball.pos[2] = 2 * r - ball.pos[2]
            ball.vel[2] *= -court.floor_restitution
            events.append("floor")
        if ball.pos[1] > court.wall_y - r and ball.vel[1] > 0:
            ball.pos[1] = 2 * (court.wall_y - r) - ball.pos[1]
            ball.vel[1] *= -court.wall_restitution
            events.append("front_wall")
        if ball.pos[1] < court.back_y + r and ball.vel[1] < 0:
            ball.pos[1] = 2 * (court.back_y + r) - ball.pos[1]
            ball.vel[1] *= -court.wall_restitution
            events.append("back_wall")
        for sign in (-1.0, 1.0):
            limit = sign * (court.half_width - r)
            if sign * ball.pos[0] > sign * limit and sign * ball.vel[0] > 0:
                ball.pos[0] = 2 * limit - ball.pos[0]
                ball.vel[0] *= -court.wall_restitution
                events.append("side_wall")
    return events


def swept_hit(ball0: np.ndarray, ball1: np.ndarray, hand0: np.ndarray, hand1: np.ndarray,
              radius: float) -> bool:
    """True if ball and hand came within `radius` of each other during the frame.

    Both move linearly from *0 to *1, so their offset is d(s) = d0 + s*(d1 - d0)
    for s in [0, 1]. We minimise |d(s)| in closed form. Testing only the end
    positions would miss a fast ball (or a fast swing) that passes through
    the hand between two frames.
    """
    d0 = ball0 - hand0
    dd = (ball1 - hand1) - d0
    denom = float(dd @ dd)
    s = 0.0 if denom < 1e-12 else float(np.clip(-(d0 @ dd) / denom, 0.0, 1.0))
    return float(np.linalg.norm(d0 + s * dd)) <= radius


def height_at(z0: float, vz0: float, t: float, r: float, e: float) -> tuple[float, int]:
    """Ball-centre height after t seconds of vertical flight with floor bounces.

    Closed form per flight arc: z(s) = z + v*s - g*s^2/2. Each bounce flips
    the impact speed and scales it by the restitution e. Returns the height
    and the number of bounces.
    """
    z, v, rem, bounces = z0, vz0, t, 0
    while True:
        s = (v + np.sqrt(v * v + 2 * GRAVITY * max(z - r, 0.0))) / GRAVITY  # time to reach z = r
        if s >= rem:
            return z + v * rem - 0.5 * GRAVITY * rem * rem, bounces
        rem -= s
        v = e * (GRAVITY * s - v)
        z = r
        bounces += 1
        if v < 1e-3:
            return r, bounces


def aim(pos: np.ndarray, target: np.ndarray, speed_y: float, court: Court,
        want_bounces: int = 1) -> np.ndarray:
    """Velocity that takes the ball from `pos` to `target` with |vy| = speed_y.

    Horizontal motion is uniform, so the flight time is T = |dy| / speed_y and
    vx follows directly. The vertical speed has no closed form once floor
    bounces are involved, so we scan vz on a grid for sign changes of
    height_at(T) - target_z and refine each with bisection. Among the roots we
    prefer one with `want_bounces` floor bounces (squash: one bounce before
    the player), else the one with the fewest bounces.
    """
    dy = float(target[1] - pos[1])
    T = abs(dy) / speed_y
    vx = float(target[0] - pos[0]) / T
    vy = np.sign(dy) * speed_y
    r, e, zt = court.ball_radius, court.floor_restitution, float(target[2])

    def f(vz: float) -> float:
        return height_at(float(pos[2]), vz, T, r, e)[0] - zt

    grid = np.linspace(-15.0, 15.0, 121)
    vals = [f(v) for v in grid]
    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:], strict=True):
        if fa == 0 or fa * fb < 0:
            for _ in range(40):
                m = 0.5 * (a + b)
                if f(a) * f(m) <= 0:
                    b = m
                else:
                    a = m
            vz = 0.5 * (a + b)
            roots.append((vz, height_at(float(pos[2]), vz, T, r, e)[1]))
    if not roots:  # unreachable target height: fall back to a straight lob
        vz = (zt - pos[2] + 0.5 * GRAVITY * T * T) / T
        return np.array([vx, vy, vz])
    vz = min(roots, key=lambda rb: (rb[1] != want_bounces, rb[1]))[0]
    return np.array([vx, vy, vz])
