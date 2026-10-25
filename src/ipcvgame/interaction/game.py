"""Task 4 — game logic: Colour Clash.

Moved from the prototype's ``game.py`` in the game merge (docs/merge_log.md,
Task 4). Positions are pixels in the mirrored frame; every number comes from
the ``game:`` config.

So far (merge step 5): the game state. ``update()`` still only prints events;
combat (step 6) and the pipeline wiring (step 7) follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ipcvgame.core.types import FrameState


@dataclass
class PaintSplat:
    x: float          # inside the target's body box, fraction of its width
    y: float          # fraction of its height
    owner_id: int     # the attacker; the colour comes from scene/draw.py


@dataclass
class GamePlayer:
    """One player's game state (the prototype's ``Player``; renamed because
    ``Player`` is the identity type in core/types.py)."""
    health: int
    cooldown: float = 0.0
    x: float = 0.0                       # px, shoulder midpoint
    y: float = 0.0                       # px, hip level
    bounds: np.ndarray | None = None     # body box x1, y1, x2, y2 in px
    paint_splats: list[PaintSplat] = field(default_factory=list)


@dataclass
class Projectile:
    x: float
    y: float
    direction: int    # +1 = to the right, -1 = to the left
    owner_id: int


@dataclass
class PaintBeam:
    start_x: float
    x: float          # moving tip
    y: float
    direction: int
    owner_id: int
    age: float = 0.0
    hit_player_ids: set[int] = field(default_factory=set)


@dataclass
class Shield:
    x: float
    y: float
    direction: int
    remaining: float  # s left; below 0 it fades out for shield_fade_s
    age: float = 0.0


class Game:
    """Track recognized players and their effects across the camera view."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_health = int(cfg["max_health"])
        self.players: dict[int, GamePlayer] = {}
        self.projectiles: list[Projectile] = []
        self.beams: list[PaintBeam] = []
        self.shields: dict[int, Shield] = {}
        self.winner_id: int | None = None

    def get_player(self, player_id: int) -> GamePlayer:
        if player_id not in self.players:
            self.players[player_id] = GamePlayer(health=self.max_health)
        return self.players[player_id]

    def update(self, state: FrameState) -> None:
        for e in state.events:
            print(f"[game] t={e.t:.3f} P{e.pid} {e.name} conf={e.conf:.2f}")

    def reset(self, pid: int | None = None) -> None:
        """None: start a fresh round while preserving current player positions.
        A pid: that player starts fresh (full health, no paint, no shield)."""
        if pid is not None:
            if pid in self.players:
                self._refresh(self.players[pid])
            self.shields.pop(pid, None)
            return
        for player in self.players.values():
            self._refresh(player)
        self.projectiles.clear()
        self.beams.clear()
        self.shields.clear()
        self.winner_id = None

    def _refresh(self, player: GamePlayer) -> None:
        player.health = self.max_health
        player.cooldown = 0.0
        player.paint_splats.clear()

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        pass
