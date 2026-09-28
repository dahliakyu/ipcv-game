"""Task 4 — game logic.

PHASE-1 STUB (owned by Task 4): prints gesture events to the console. The
real game state machine comes once the game design is final.
"""

from __future__ import annotations

import numpy as np

from ipcvgame.core.types import FrameState


class Game:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def update(self, state: FrameState) -> None:
        for e in state.events:
            print(f"[game] t={e.t:.3f} P{e.pid} {e.name} conf={e.conf:.2f}")

    def reset(self, pid: int | None = None) -> None:
        pass

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        pass
