"""Task 4 — gesture recognition: motion signals -> GestureEvent.

PHASE-1 STUB (owned by Task 4): returns no events. Replace with rule-based
gestures (hold times, cooldowns, hysteresis) keeping the same update() signature.
"""

from __future__ import annotations

import numpy as np

from ipcvgame.core.types import FrameState, GestureEvent, Motion
from ipcvgame.scene.draw import draw_lines


class GestureRecognizer:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def update(self, motion: dict[int, Motion], t: float) -> list[GestureEvent]:
        return []

    def reset(self, pid: int | None = None) -> None:
        pass

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Each player's motion signals and this frame's events."""
        lines = []
        for pid, m in sorted(state.motion.items()):
            sig = "  ".join(f"{k}={v:.0f}" for k, v in m.signals.items())
            lines.append(f"P{pid}: {sig}")
        lines += [f"EVENT P{e.pid} {e.name} ({e.conf:.2f})" for e in state.events]
        draw_lines(canvas, lines, (10, canvas.shape[0] - 20 * len(lines) - 10))
