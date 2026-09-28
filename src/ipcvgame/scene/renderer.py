"""Task 5 — game renderer.

Phase 1: OpenCV drawing of the camera frame, per-player boxes, face effects
and a timing HUD. Will move to pygame-ce once the game design is final.
"""

from __future__ import annotations

import numpy as np

from ipcvgame.core.types import FrameState
from ipcvgame.face.effects import FaceEffects
from ipcvgame.interaction.game import Game
from ipcvgame.scene.draw import draw_bbox, draw_lines, draw_skeleton, player_color


class Renderer:
    def __init__(self, cfg: dict, face_effects: FaceEffects):
        self.cfg = cfg
        self.face_effects = face_effects
        self.show_timings = bool(cfg.get("show_timings", True))

    def draw(self, state: FrameState, game: Game) -> np.ndarray:
        canvas = state.frame.image.copy()
        for pid, p in state.players.items():
            if p.status == "active":
                draw_bbox(canvas, p.bbox, player_color(pid), f"P{pid}")
        for pid, m in state.motion.items():
            draw_skeleton(canvas, m.kpts, m.valid, player_color(pid))
        self.face_effects.draw(canvas, state.faces)
        if self.show_timings:
            draw_timings(canvas, state.timings)
        return canvas


def draw_timings(canvas: np.ndarray, timings: dict[str, float]) -> None:
    """Per-stage ms of the previous frame, plus FPS from its total."""
    lines = [f"{k:>9s} {v:6.1f} ms" for k, v in timings.items() if k != "total"]
    if "total" in timings and timings["total"] > 0:
        lines.append(f"{'total':>9s} {timings['total']:6.1f} ms  ({1000 / timings['total']:.0f} FPS)")
    draw_lines(canvas, lines)
