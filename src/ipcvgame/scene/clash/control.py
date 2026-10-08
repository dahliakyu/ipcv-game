"""Player input for the bolt prototype: where each player wants to stand.

Players move in 1D, along the field depth y (Combat.move() fixes their x).

P1 uses the mouse: the cursor is cast onto the floor, and the body walks to
the cursor's y with a lag (EMA, time constant `mouse_lag`), like a player
stepping to a spot. P2 uses W/S: the keys set a walking velocity that is
integrated.

Later the pose pipeline replaces both: game y comes from the player's
sideways position in the webcam image (hip-midpoint x, mapped from their side
of the frame to [-half_depth, half_depth]), and a `punch` GestureEvent
(Task 4) replaces the click / key.
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


# key -> direction along y; W walks away from the camera ("up" on screen)
KEYS = {"w": 1.0, "s": -1.0}


class KeyboardControl:
    def __init__(self, cfg: dict, is_down: Callable[[str], bool], start: np.ndarray,
                 clamp: Callable[[np.ndarray], np.ndarray]):
        self.is_down = is_down                  # callable key -> bool
        self.speed = cfg["key_speed"]
        self.clamp = clamp                      # keeps the integrated position on the field
        self.pos = np.asarray(start, dtype=float)

    def read(self, dt: float) -> np.ndarray:
        d = sum(v for k, v in KEYS.items() if self.is_down(k))   # W + S together cancel
        if d:
            self.pos = self.clamp(self.pos + np.array([0.0, self.speed * dt * d]))
        return self.pos

    def reset(self, start: np.ndarray) -> None:
        self.pos = np.asarray(start, dtype=float)
