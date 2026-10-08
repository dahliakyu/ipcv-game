"""Player input for the squash prototype.

The game only sees a `Control`: where the player stands across the court and
where their hitting hand is relative to their body, both normalised to
[-1, 1]. The mouse fills it now; later the pose pipeline fills it from
keypoints, and nothing in rally.py or avatar.py has to change.

A pose-based source would compute, from `Motion` of one player:
    body_x = hip-midpoint x / frame width, mapped to [-1, 1]
    hand   = (wrist - shoulder) / arm length, image y flipped so up is +
(mind the mirrored-frame L/R labels: sk.L_WRIST is the player's own right).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ipcvgame.core.filters import EMA


@dataclass
class Control:
    body_x: float       # -1 = left side wall, +1 = right side wall
    hand: np.ndarray    # (2,) u: -1 left .. +1 right of the body, v: -1 low .. +1 high


class ControlSource(Protocol):
    def read(self, t: float) -> Control | None: ...


class MouseControl:
    """Mouse x/y in [-1, 1] (Panda3D's mouse coordinates) to a Control.

    The body follows the mouse with a lag (EMA, time constant `body_lag`),
    like a player running to the ball. The hand leads in the direction of
    motion and sits `hand_side` to the right of the body when still.
    """

    def __init__(self, cfg: dict, mouse_xy):
        self.mouse_xy = mouse_xy                # callable -> (x, y) or None
        self.body = EMA(tau=cfg["body_lag"])
        self.hand_side = cfg["hand_side"]
        self.hand_gain = cfg["hand_lead_gain"]

    def read(self, t: float) -> Control | None:
        m = self.mouse_xy()
        if m is None:
            return None
        mx, my = m
        body_x = float(self.body(np.array([mx]), t)[0])
        u = np.clip(self.hand_side + self.hand_gain * (mx - body_x), -1.0, 1.0)
        return Control(body_x=body_x, hand=np.array([u, np.clip(my, -1.0, 1.0)]))
