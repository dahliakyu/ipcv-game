"""Player input for the bolt prototype: where on the floor each player wants
to stand, and whether they fire this frame.

P1 uses the mouse: the cursor is cast onto the floor, and the body walks
there with a lag (EMA, time constant `mouse_lag`), like a player stepping to
a spot. P2 uses WASD: the keys set a walking velocity that is integrated.
Both return a target floor position; Combat.move() clamps it to the half.

Later the pose pipeline replaces both: the floor x comes from the hip
midpoint across the frame and the depth from Player.pos (Task 3), and a
`punch` GestureEvent (Task 4) replaces the click / key.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ipcvgame.core.filters import EMA


class MouseFloorControl:
    def __init__(self, cfg: dict, floor_point: Callable[[], np.ndarray | None], start: np.ndarray):
        self.floor_point = floor_point          # callable -> cursor on the floor (x, y) or None
        self.body = EMA(tau=cfg["mouse_lag"])
        self.last = np.asarray(start, dtype=float)

    def read(self, t: float) -> np.ndarray:
        p = self.floor_point()
        if p is not None:
            self.last = self.body(p, t)
        return self.last

    def reset(self, start: np.ndarray) -> None:
        self.body.reset()
        self.last = np.asarray(start, dtype=float)


# key -> unit step on the floor; W walks away from the camera (+y, "up" on screen)
WASD = {"w": (0.0, 1.0), "s": (0.0, -1.0), "a": (-1.0, 0.0), "d": (1.0, 0.0)}


class KeyboardControl:
    def __init__(self, cfg: dict, is_down: Callable[[str], bool], start: np.ndarray,
                 clamp: Callable[[np.ndarray], np.ndarray]):
        self.is_down = is_down                  # callable key -> bool
        self.speed = cfg["key_speed"]
        self.clamp = clamp                      # keeps the integrated position in the half
        self.pos = np.asarray(start, dtype=float)

    def read(self, dt: float) -> np.ndarray:
        d = np.sum([WASD[k] for k in WASD if self.is_down(k)] or [(0.0, 0.0)], axis=0)
        n = np.linalg.norm(d)
        if n > 0:  # diagonals walk at the same speed as straight lines
            self.pos = self.clamp(self.pos + self.speed * dt * d / n)
        return self.pos

    def reset(self, start: np.ndarray) -> None:
        self.pos = np.asarray(start, dtype=float)
