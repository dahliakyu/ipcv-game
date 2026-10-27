"""Task 4 — game logic: Colour Clash.

Moved from the Colour Clash ``game.py`` in the game merge (docs/merge_log.md,
Task 4). Positions are pixels in the mirrored frame; every number comes from
the ``game:`` config.

Combat (``advance()``): bolts and beams fly across the frame and damage the
first other player whose body box they reach, adding paint splats; a raised
shield blocks them. At 0 health the attacker wins and the game freezes.

``update(state)`` is the game part of the ``main.py`` loop: it reads each player's
body box and position from the frame state, turns gesture events into actions
(punch -> bolt, hands up -> beam, arms crossed -> shield), then advances the
game by the real time since the last frame.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.types import FrameState, Motion
from ipcvgame.interaction.gestures import lower_body_y, shoulder_width
from ipcvgame.scene.draw import draw_bbox, draw_lines, draw_skeleton, player_color, pt


@dataclass
class PaintSplat:
    x: float          # inside the target's body box, fraction of its width
    y: float          # fraction of its height
    owner_id: int     # the attacker; the colour comes from scene/draw.py


@dataclass
class GamePlayer:
    """One player's game state (``Player`` in game.py; renamed because
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
    age: float = 0.0  # s since it was fired (drives the fire animation)


@dataclass
class BoltEnd:
    """Where a bolt ended (a hit, a shield or the frame edge): drawing only,
    so its break-up can play there for bolt_end_s."""
    x: float
    y: float
    direction: int
    owner_id: int
    age: float = 0.0


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


def punching_hand(m: Motion) -> tuple[float, float] | None:
    """The wrist of the arm that reaches furthest to the side of its shoulder
    (the punching arm), in px; None if no arm has a valid shoulder and wrist.
    ``hand_x`` / ``hand_y`` in bodypose.py, which picked the arm by punch side."""
    best, reach = None, -1.0
    for shoulder, wrist in ((sk.L_SHOULDER, sk.L_WRIST), (sk.R_SHOULDER, sk.R_WRIST)):
        if m.valid[shoulder] and m.valid[wrist]:
            r = abs(m.kpts[wrist][0] - m.kpts[shoulder][0])
            if r > reach:
                best, reach = m.kpts[wrist], r
    return None if best is None else (float(best[0]), float(best[1]))


class Game:
    """Track recognized players and their effects across the camera view."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_health = int(cfg["max_health"])
        # Frame size in px; speeds and sizes in the config are fractions of it.
        # Set from each frame by update() (step 7).
        self.frame_w, self.frame_h = 0, 0
        self.players: dict[int, GamePlayer] = {}
        self.projectiles: list[Projectile] = []
        self.bolt_ends: list[BoltEnd] = []
        self.beams: list[PaintBeam] = []
        self.shields: dict[int, Shield] = {}
        self.winner_id: int | None = None
        self._t: float | None = None  # Frame.t of the previous update
        # Actions this frame as (pid, "bolt" | "beam" | "shield"), for the CSV.
        self.actions: list[tuple[int, str]] = []
        # Phase lengths and the beam's hit band, for drawing (the renderer
        # reads them here instead of from the game: config).
        self.beam_charge_s = float(cfg["beam_charge_s"])
        self.beam_life_s = self.beam_charge_s + float(cfg["beam_duration_s"])
        self.shield_fade_s = float(cfg["shield_fade_s"])
        self.bolt_end_s = float(cfg["bolt_end_s"])
        self.splat_band = (float(cfg["splat_band"][0]), float(cfg["splat_band"][1]))

    def get_player(self, player_id: int) -> GamePlayer:
        if player_id not in self.players:
            self.players[player_id] = GamePlayer(health=self.max_health)
        return self.players[player_id]

    def attack(self, x: float, y: float, direction: int, player_id: int) -> bool:
        """Fire a bolt from (x, y) px; return if it fired. Not while the
        player's own beam is still alive: one attack at a time."""
        player = self.get_player(player_id)
        if (self.winner_id is not None or player.cooldown > 0
                or any(b.owner_id == player_id for b in self.beams)):
            return False

        self.projectiles.append(Projectile(x, y, 1 if direction >= 0 else -1, player_id))
        player.cooldown = self.cfg["bolt_cooldown_s"]
        return True

    def beam_attack(self, x: float, y: float, direction: int, player_id: int) -> bool:
        """Launch a short-lived paint beam from (x, y) px; return if it fired.
        One beam per player at a time: none while their last one is alive."""
        if self.winner_id is not None or any(b.owner_id == player_id for b in self.beams):
            return False
        self.beams.append(PaintBeam(x, x, y, 1 if direction >= 0 else -1, owner_id=player_id))
        return True

    def activate_shield(self, x: float, y: float, direction: int, player_id: int) -> None:
        """Raise the shield beside the player; it blocks hits for shield_s."""
        self.shields[player_id] = Shield(x, y, 1 if direction >= 0 else -1,
                                         remaining=self.cfg["shield_s"])

    def shielded(self, player_id: int) -> bool:
        """True while the player's shield blocks (not during its fade-out)."""
        shield = self.shields.get(player_id)
        return shield is not None and shield.remaining > 0

    def advance(self, dt: float, visible_player_ids: set[int] | None = None) -> None:
        """Move the game on by dt seconds (``update()`` in game.py). Only
        visible players can be hit."""
        if self.winner_id is not None:
            return
        cfg = self.cfg

        for player in self.players.values():
            player.cooldown = max(0.0, player.cooldown - dt)

        for player_id, shield in list(self.shields.items()):
            shield.age += dt
            shield.remaining -= dt
            if shield.remaining <= -cfg["shield_fade_s"]:
                del self.shields[player_id]

        active_player_ids = set(self.players) if visible_player_ids is None else visible_player_ids
        for end in self.bolt_ends:
            end.age += dt
        self.bolt_ends = [e for e in self.bolt_ends if e.age < cfg["bolt_end_s"]]
        remaining = []
        for projectile in self.projectiles:
            previous_x = projectile.x
            projectile.age += dt
            projectile.x += projectile.direction * cfg["bolt_speed_w"] * self.frame_w * dt
            target = self._find_projectile_target(projectile, previous_x, active_player_ids)
            if target is not None:
                # A shield absorbs the bolt: it disappears without damage.
                if not self.shielded(target):
                    self._apply_damage(target, projectile.owner_id, projectile.x,
                                       projectile.y, cfg["bolt_damage"])
            elif 0.0 <= projectile.x <= self.frame_w:
                remaining.append(projectile)
                continue
            self.bolt_ends.append(BoltEnd(min(max(projectile.x, 0.0), float(self.frame_w)),
                                          projectile.y, projectile.direction, projectile.owner_id))
        self.projectiles = remaining

        charge = cfg["beam_charge_s"]
        speed = cfg["beam_speed_w"] * self.frame_w
        active_beams = []
        for beam in self.beams:
            previous_x = beam.x
            previous_age = beam.age
            beam.age += dt
            if previous_age >= charge:
                beam.x += beam.direction * speed * dt
            elif beam.age > charge:
                beam.x += beam.direction * speed * (beam.age - charge)
            if beam.age >= charge:
                self._check_beam_hits(beam, previous_x, active_player_ids)
            # The tip stops at the frame edge; the beam lasts its full duration.
            beam.x = min(max(beam.x, 0.0), float(self.frame_w))
            if beam.age < charge + cfg["beam_duration_s"]:
                active_beams.append(beam)
        self.beams = active_beams

    def _find_projectile_target(self, projectile: Projectile, previous_x: float,
                                visible_player_ids: set[int]) -> int | None:
        # The path since the last frame, so a fast bolt cannot skip a box.
        segment_left = min(previous_x, projectile.x)
        segment_right = max(previous_x, projectile.x)
        for player_id in visible_player_ids:
            if player_id == projectile.owner_id:
                continue
            bounds = self.get_player(player_id).bounds
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            if top <= projectile.y <= bottom and segment_right >= left and segment_left <= right:
                return player_id
        return None

    def _check_beam_hits(self, beam: PaintBeam, previous_x: float,
                         visible_player_ids: set[int]) -> None:
        radius = self.cfg["beam_hit_radius_h"] * self.frame_h
        beam_left = min(beam.start_x, previous_x, beam.x)
        beam_right = max(beam.start_x, previous_x, beam.x)
        for player_id in visible_player_ids:
            if player_id == beam.owner_id or player_id in beam.hit_player_ids:
                continue
            bounds = self.get_player(player_id).bounds
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            if (beam.y + radius >= top and beam.y - radius <= bottom
                    and beam_right >= left and beam_left <= right):
                # A shield blocks the beam for this player for the rest of the beam.
                if not self.shielded(player_id):
                    self._apply_damage(player_id, beam.owner_id,
                                       min(max(right, beam_left), beam_right), beam.y,
                                       self.cfg["beam_damage"])
                beam.hit_player_ids.add(player_id)

    def _apply_damage(self, target_id: int, owner_id: int, impact_x: float,
                      impact_y: float, damage: int) -> None:
        target = self.get_player(target_id)
        if target.health <= 0 or self.winner_id is not None:
            return

        target.health = max(0, target.health - damage)
        if target.bounds is not None:
            lo, hi = self.cfg["splat_band"]
            spread = self.cfg["splat_spread"]
            left, top, right, bottom = target.bounds
            width = max(right - left, 1.0)
            height = max(bottom - top, 1.0)
            # Impact point as a fraction of the box, kept on the torso band.
            center_x = min(hi, max(lo, (impact_x - left) / width))
            center_y = min(hi, max(lo, (impact_y - top) / height))
            splats = target.paint_splats
            splats.append(PaintSplat(center_x, center_y, owner_id))
            # Seeded, so the same hit always gives the same splats.
            rng = random.Random(target_id * 1000 + len(splats))
            for _ in range(self.cfg["splat_extra"]):
                splats.append(PaintSplat(
                    min(hi, max(lo, center_x + rng.uniform(-spread, spread))),
                    min(hi, max(lo, center_y + rng.uniform(-spread, spread))),
                    owner_id,
                ))

        if target.health == 0:
            self.winner_id = owner_id

    def update(self, state: FrameState) -> None:
        """One frame (the game part of the ``main.py`` loop)."""
        self.frame_h, self.frame_w = state.frame.image.shape[:2]
        t = state.frame.t
        # Real time since the last frame, capped so a stall or pause does not
        # make bolts and beams jump ahead.
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), self.cfg["max_dt_s"])
        self._t = t

        # Only players the identity stage sees this frame can be hit.
        visible = {pid for pid, p in state.players.items() if p.status == "active"}
        positions = {}
        for pid in visible:
            player = self.get_player(pid)
            player.bounds = state.players[pid].bbox.astype(float)
            if pid in state.motion:
                pos = self.body_position(state.motion[pid])
                if pos is not None:
                    player.x, player.y = pos
                    positions[pid] = pos

        self.actions = []
        for e in state.events:
            if e.pid in positions:
                self._act(e.pid, e.name, state.motion[e.pid], positions)

        self.advance(dt, visible)

    def torso(self, m: Motion) -> tuple[float, float, float, float] | None:
        """Shoulder midpoint x, shoulder level y, hip level y and shoulder
        width in px, or None without a usable shoulder width. Hips not
        visible: hip_below_shoulders shoulder widths below the shoulders."""
        k, v = m.kpts, m.valid
        sw = shoulder_width(k, v, self.cfg["min_shoulder_px"])
        if sw is None:
            return None
        shoulder_y = (k[sk.L_SHOULDER][1] + k[sk.R_SHOULDER][1]) / 2.0
        center_x = (k[sk.L_SHOULDER][0] + k[sk.R_SHOULDER][0]) / 2.0
        hip_y = lower_body_y(k, v, shoulder_y, sw, self.cfg)
        return float(center_x), float(shoulder_y), float(hip_y), float(sw)

    def facing(self, pid: int, positions: dict[int, tuple[float, float]]) -> int:
        """Sideways direction of pid's attacks: +1 = right, -1 = left.

        Towards the nearest other player in view (their shoulder midpoint),
        so attacks still fly at the opponent after the players have switched
        sides or stand in the same half. main.py fired towards the middle of
        the frame (right from the left half), which sent attacks away from the
        opponent in those cases; that rule stays as the fallback when no one
        else is in view, or both stand at the same x."""
        x = positions[pid][0]
        others = [p[0] for q, p in positions.items() if q != pid and p[0] != x]
        if others:
            nearest = min(others, key=lambda ox: abs(ox - x))
            return 1 if nearest > x else -1
        return 1 if x < self.frame_w / 2 else -1

    def body_position(self, m: Motion) -> tuple[float, float] | None:
        """Shoulder midpoint x and hip level y in px (``center_x`` and
        ``torso_y`` in bodypose.py), or None without a usable shoulder width."""
        torso = self.torso(m)
        return None if torso is None else (torso[0], torso[2])

    def beam_band_px(self) -> float:
        """Half the height (px) of the band a beam hits (beam_hit_radius_h)."""
        return self.cfg["beam_hit_radius_h"] * self.frame_h

    def shield_opacity(self, shield: Shield) -> float:
        """1 while the shield blocks, then down to 0 over shield_fade_s."""
        return 1.0 if shield.remaining > 0 else max(0.0, 1.0 + shield.remaining / self.shield_fade_s)

    def _act(self, pid: int, gesture: str, m: Motion, positions: dict[int, tuple[float, float]]) -> None:
        """Turn one gesture event into a game action, as ``main.py`` did."""
        player = self.get_player(pid)
        direction = self.facing(pid, positions)
        if gesture == "punch":
            # The bolt starts at the punching hand; without a visible wrist, at
            # the edge of the body box facing the other player, at hip height.
            hand = punching_hand(m)
            if hand is not None:
                x, y = hand
            else:
                left, _, right, _ = player.bounds
                x, y = (right if direction > 0 else left), player.y
            if self.attack(x, y, direction, pid):
                self.actions.append((pid, "bolt"))
        elif gesture == "hands_up":
            if not (m.valid[sk.L_WRIST] and m.valid[sk.R_WRIST]):
                return
            hx, hy = (m.kpts[sk.L_WRIST] + m.kpts[sk.R_WRIST]) / 2.0
            if self.beam_attack(float(hx), float(hy), direction, pid):
                self.actions.append((pid, "beam"))
        elif gesture == "arms_crossed":
            self.activate_shield(player.x, player.y, direction, pid)
            self.actions.append((pid, "shield"))

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
        self.bolt_ends.clear()
        self.beams.clear()
        self.shields.clear()
        self.winner_id = None

    def _refresh(self, player: GamePlayer) -> None:
        player.health = self.max_health
        player.cooldown = 0.0
        player.paint_splats.clear()

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """The game state as plain shapes, for checking the logic (the game's
        look is drawn by the renderer): thin skeletons; body boxes with health
        and shield; splats as dots; bolts as dots with a tail and an x where
        they ended; beams as a line with their hit band (a growing circle while
        charging); shields as an ellipse beside the player (thin while
        fading); this frame's actions and the winner."""
        unit = max(canvas.shape[1] / 100.0, 2.0)  # 1% of the frame width, px
        for pid, m in state.motion.items():
            draw_skeleton(canvas, m.kpts, m.valid, player_color(pid), 1)

        for pid, player in sorted(self.players.items()):
            if player.bounds is None:
                continue
            left, top, right, bottom = player.bounds
            w, h = right - left, bottom - top
            label = f"P{pid} {player.health}" + (" shield" if self.shielded(pid) else "")
            draw_bbox(canvas, player.bounds, player_color(pid), label, dashed=True)
            for splat in player.paint_splats:
                c = (left + splat.x * w, top + splat.y * h)
                cv2.circle(canvas, pt(c), 3, player_color(splat.owner_id), -1, cv2.LINE_AA)

        for end in self.bolt_ends:
            x, y, r = int(end.x), int(end.y), int(unit)
            c = player_color(end.owner_id)
            cv2.line(canvas, (x - r, y - r), (x + r, y + r), c, 2, cv2.LINE_AA)
            cv2.line(canvas, (x - r, y + r), (x + r, y - r), c, 2, cv2.LINE_AA)
        for b in self.projectiles:
            c = player_color(b.owner_id)
            cv2.circle(canvas, pt((b.x, b.y)), int(unit), c, -1, cv2.LINE_AA)
            cv2.line(canvas, pt((b.x, b.y)), pt((b.x - b.direction * 4 * unit, b.y)), c, 2, cv2.LINE_AA)

        band = self.beam_band_px()
        for beam in self.beams:
            c = player_color(beam.owner_id)
            if beam.age < self.beam_charge_s:
                r = int(unit * (1 + 2 * beam.age / self.beam_charge_s))
                cv2.circle(canvas, pt((beam.start_x, beam.y)), r, c, 2, cv2.LINE_AA)
                continue
            cv2.line(canvas, pt((beam.start_x, beam.y)), pt((beam.x, beam.y)), c, 2, cv2.LINE_AA)
            for dy in (-band, band):
                cv2.line(canvas, pt((beam.start_x, beam.y + dy)), pt((beam.x, beam.y + dy)), c, 1,
                         cv2.LINE_AA)

        for pid, shield in self.shields.items():
            player = self.players.get(pid)
            h = (player.bounds[3] - player.bounds[1]) if player and player.bounds is not None else 40 * unit
            center = pt((shield.x + shield.direction * 6 * unit, shield.y - 0.25 * h))
            axes = (int(1.5 * unit), int(0.3 * h))
            cv2.ellipse(canvas, center, axes, 0, 0, 360, player_color(pid),
                        3 if shield.remaining > 0 else 1, cv2.LINE_AA)

        lines = [f"P{pid} {action}" for pid, action in self.actions]
        if self.winner_id is not None:
            lines.append(f"P{self.winner_id} wins")
        draw_lines(canvas, lines, (10, canvas.shape[0] - 20 * len(lines) - 10))

    def log_fields(self, state: FrameState) -> dict:
        """CSV columns: game_p<pid>_health, game_p<pid>_shield (1 while it
        blocks), game_p<pid>_actions (this frame's actions joined by '+'),
        game_bolts and game_beams (in flight), game_winner (pid or empty)."""
        row: dict = {}
        for pid, player in sorted(self.players.items()):
            pre = f"game_p{pid}_"
            row[pre + "health"] = player.health
            row[pre + "shield"] = int(self.shielded(pid))
            row[pre + "actions"] = "+".join(a for p, a in self.actions if p == pid)
        row["game_bolts"] = len(self.projectiles)
        row["game_beams"] = len(self.beams)
        row["game_winner"] = self.winner_id if self.winner_id is not None else ""
        return row
