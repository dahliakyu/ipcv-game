"""Color Clash rules for the bolt prototype: move, fire, hit, win. No rendering.

World frame (metres, Z up, as in Panda3D): x along the field, P1 on the left
(x < 0) fires towards +x, P2 on the right fires towards -x; y is depth (away
from the camera), z up. Each player moves in 1D only, along y, at a fixed x
in their own half. With pose input, a sideways step in front of the webcam
(image x, well tracked) becomes this y; camera depth is never needed.

Round flow:

    countdown --(countdown s)--> fight --(a player reaches 100 % damage)--> over
        ^                                                                     |
        +-------------------------------- restart ----------------------------+

Magic Bolt (design doc 4.1): fast, straight, small, low damage, short cooldown,
dodged by stepping out of its line or by ducking (design doc 4.4: crouch). A
bolt is a ball flying parallel to the field axis at the height of the hand
that fired it: a jab flies at head height, so a crouch lets it pass overhead.
Nobody can fire while their body is down (crouching or getting back up), so
a player has to choose between dodging and attacking.

If the prototype becomes the game, these rules move to interaction/game.py
(Task 4), and pose gestures replace the click / key that calls fire().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ipcvgame.scene.squash.physics import swept_hit

PIDS = (1, 2)


@dataclass
class Arena:
    half_length: float     # field runs x in [-half_length, half_length]
    half_depth: float      # y in [-half_depth, half_depth]
    centre_gap: float      # no-man's land on each side of x = 0; players stand midway
                           # between it and the field end
    edge_margin: float     # players stay this far inside the field edges

    @classmethod
    def from_cfg(cls, cfg: dict) -> Arena:
        return cls(cfg["field_length"] / 2, cfg["field_depth"] / 2,
                   cfg["centre_gap"], cfg["edge_margin"])

    def side(self, pid: int) -> int:
        """-1 for P1 (left half), +1 for P2 (right half)."""
        return -1 if pid == 1 else 1

    def clamp(self, pid: int, pos: np.ndarray) -> np.ndarray:
        """Project a floor position (x, y) onto the player's line of movement:
        x fixed at the player's spot, y kept inside the field."""
        d = self.half_depth - self.edge_margin
        return np.array([self.start(pid)[0], np.clip(pos[1], -d, d)])

    def start(self, pid: int) -> np.ndarray:
        """Spawn point, and the fixed x of the player: middle of their half."""
        return np.array([self.side(pid) * (self.centre_gap + self.half_length) / 2, 0.0])


@dataclass
class Fighter:
    pid: int
    pos: np.ndarray                  # (2,) floor x, y
    prev: np.ndarray                 # (2,) position at the start of the frame (swept hits)
    damage: float = 0.0              # 0 = own colour .. 1 = covered in the opponent's colour
    cooldown: float = 0.0            # seconds until the next bolt is allowed
    jab: float = 0.0                 # seconds left of the arm-extension animation
    crouch: float = 0.0              # 0 = standing .. 1 = fully down
    crouch_held: bool = False        # input this frame


@dataclass
class Bolt:
    owner: int
    pos: np.ndarray                  # (3,)
    vel: np.ndarray                  # (3,)
    trail: list[np.ndarray] = field(default_factory=list)  # recent positions, newest last


@dataclass
class Event:
    name: str                        # "fire" | "hit" | "fizzle" | "win"
    pid: int                         # fire/fizzle: shooter; hit: victim; win: winner
    pos: np.ndarray | None = None


class Combat:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.arena = Arena.from_cfg(cfg)
        # hit volume in body coordinates: (height of centre, radius) per sphere
        self.body_spheres = [tuple(s) for s in cfg["body_spheres"]]
        self.reset()

    # --- control -------------------------------------------------------------

    def reset(self) -> None:
        self.fighters = {pid: Fighter(pid, self.arena.start(pid), self.arena.start(pid)) for pid in PIDS}
        self.bolts: list[Bolt] = []
        self.state = "countdown"
        self.winner: int | None = None
        self.t_state = 0.0

    def hold_crouch(self, pid: int, held: bool) -> None:
        """Crouch input for this frame (key / button held)."""
        self.fighters[pid].crouch_held = held

    def drop(self, pid: int) -> float:
        """How far the player's body is lowered right now, in metres."""
        return self.fighters[pid].crouch * self.cfg["crouch_drop"]

    def move(self, pid: int, target: np.ndarray) -> None:
        """Set where a player stands this frame; clamped to their half."""
        self.fighters[pid].pos = self.arena.clamp(pid, np.asarray(target, dtype=float))

    def fire(self, pid: int, palm: np.ndarray) -> bool:
        """Fire a bolt from the palm towards the opponent's half. Returns
        False (and does nothing) during cooldown, while ducking (also while
        getting back up) or outside the fight."""
        f = self.fighters[pid]
        if self.state != "fight" or f.cooldown > 0 or f.crouch > 0:
            return False
        direction = -self.arena.side(pid)            # P1 (left) fires +x
        vel = np.array([direction * self.cfg["bolt_speed"], 0.0, 0.0])
        self.bolts.append(Bolt(pid, np.asarray(palm, dtype=float).copy(), vel))
        f.cooldown = self.cfg["bolt_cooldown"]
        f.jab = self.cfg["jab_time"]
        return True

    # --- per frame -------------------------------------------------------------

    def update(self, dt: float) -> list[Event]:
        """Advance bolts and timers by dt. Call after move() for both players
        and before the next frame's move(), so prev -> pos spans one frame."""
        self.t_state += dt
        events: list[Event] = []
        for f in self.fighters.values():
            f.cooldown = max(0.0, f.cooldown - dt)
            f.jab = max(0.0, f.jab - dt)
            # going down takes crouch_time, so a duck must be started early
            step = dt / self.cfg["crouch_time"]
            f.crouch = min(1.0, f.crouch + step) if f.crouch_held else max(0.0, f.crouch - step)

        if self.state == "countdown" and self.t_state >= self.cfg["countdown"]:
            self._set("fight")

        alive = []
        for b in self.bolts:
            p0 = b.pos.copy()
            b.pos = b.pos + b.vel * dt
            b.trail = (b.trail + [p0])[-self.cfg["trail_len"]:]
            victim = self.fighters[2 if b.owner == 1 else 1]
            hit_at = self._hits(victim, p0, b.pos) if self.state == "fight" else None
            if hit_at is not None:
                # rounded: 10 x 0.1 sums to 0.999... in floating point and would never win
                victim.damage = min(1.0, round(victim.damage + self.cfg["bolt_damage"], 9))
                events.append(Event("hit", victim.pid, hit_at))
                if victim.damage >= 1.0:
                    self.winner = b.owner
                    self._set("over")
                    events.append(Event("win", b.owner))
            elif abs(b.pos[0]) > self.arena.half_length:
                events.append(Event("fizzle", b.owner, b.pos.copy()))
            else:
                alive.append(b)
        self.bolts = alive

        for f in self.fighters.values():
            f.prev = f.pos.copy()
        return events

    def countdown_left(self) -> float:
        return max(0.0, self.cfg["countdown"] - self.t_state) if self.state == "countdown" else 0.0

    # --- internals ---------------------------------------------------------------

    def _hits(self, f: Fighter, b0: np.ndarray, b1: np.ndarray) -> np.ndarray | None:
        """Swept test of the bolt against each body sphere. Both move during
        the frame (the player may be dodging), so a fast bolt cannot skip
        through a body between two frames. Returns the bolt position on a hit."""
        r = self.cfg["bolt_radius"]
        drop = self.drop(f.pid)
        for z, radius in self.body_spheres:
            c0 = np.array([f.prev[0], f.prev[1], z - drop])
            c1 = np.array([f.pos[0], f.pos[1], z - drop])
            if swept_hit(b0, b1, c0, c1, r + radius):
                return b1.copy()
        return None

    def _set(self, state: str) -> None:
        self.state = state
        self.t_state = 0.0
