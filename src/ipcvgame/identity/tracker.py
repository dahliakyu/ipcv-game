"""Task 3 — multiplayer identity tracking.

PHASE-1 STUB (owned by Task 3): Player 1 = best detection in the left half of
the screen, Player 2 = best detection in the right half. No motion model, no
appearance model; swaps whenever players cross. Replace with the real tracker
(Kalman + Hungarian + colour histogram) keeping the same update() signature.
"""

from __future__ import annotations

import numpy as np

from ipcvgame.core.geometry import bbox_center
from ipcvgame.core.types import Frame, FrameState, Player, PoseDetection
from ipcvgame.scene.draw import draw_bbox, player_color


class IdentityTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.num_players = int(cfg.get("num_players", 2))
        self.players: dict[int, Player] = {}

    def update(self, dets: list[PoseDetection], frame: Frame) -> dict[int, Player]:
        width = frame.image.shape[1]
        zone_w = width / self.num_players
        best: dict[int, PoseDetection] = {}
        for d in dets:
            zone = min(int(bbox_center(d.bbox)[0] // zone_w), self.num_players - 1)
            pid = zone + 1
            if pid not in best or d.score > best[pid].score:
                best[pid] = d

        for pid in range(1, self.num_players + 1):
            prev = self.players.get(pid)
            det = best.get(pid)
            if det is not None:
                self.players[pid] = Player(
                    pid=pid, status="active", det=det, bbox=det.bbox.copy(),
                    pos=np.zeros(2),  # stub: (x, depth) in metres not estimated yet
                    reentered=prev is not None and prev.status == "lost",
                )
            elif prev is not None:
                self.players[pid] = Player(
                    pid=pid, status="lost", det=None, bbox=prev.bbox.copy(),
                    pos=prev.pos.copy(), reentered=False,
                )
        return dict(self.players)

    def reset(self, pid: int | None = None) -> None:
        if pid is None:
            self.players.clear()
        else:
            self.players.pop(pid, None)

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Zone split line, player boxes (dashed when not active) and status."""
        h, w = canvas.shape[:2]
        for z in range(1, self.num_players):
            x = int(w * z / self.num_players)
            canvas[:, max(0, x - 1):x + 1] = (255, 255, 255)
        for p in state.players.values():
            label = f"P{p.pid} {p.status}" + (" REENTER" if p.reentered else "")
            draw_bbox(canvas, p.bbox, player_color(p.pid), label, dashed=p.status != "active")
