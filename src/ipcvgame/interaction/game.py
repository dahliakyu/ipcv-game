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
from ipcvgame.scene.draw import draw_lines, draw_skeleton, draw_text, hue_of, player_color, pt
from ipcvgame.scene.fire_bolt import FireBolt
from ipcvgame.scene.fire_ring import FireRing
from ipcvgame.scene.magic_beam import MagicBeam


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
        self._magic: dict[int, MagicBeam] = {}  # beam look per owner, loaded on first draw
        self._fire: dict[int, FireBolt] = {}  # bolt look per owner, loaded on first draw
        self._rings: dict[int, FireRing] = {}  # shield look per owner, loaded on first draw

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
                self._act(e.pid, e.name, state.motion[e.pid])

        self.advance(dt, visible)

    def body_position(self, m: Motion) -> tuple[float, float] | None:
        """Shoulder midpoint x and hip level y in px (``center_x`` and
        ``torso_y`` in bodypose.py), or None without a usable shoulder width."""
        k, v = m.kpts, m.valid
        sw = shoulder_width(k, v, self.cfg["min_shoulder_px"])
        if sw is None:
            return None
        shoulder_y = (k[sk.L_SHOULDER][1] + k[sk.R_SHOULDER][1]) / 2.0
        center_x = (k[sk.L_SHOULDER][0] + k[sk.R_SHOULDER][0]) / 2.0
        return float(center_x), float(lower_body_y(k, v, shoulder_y, sw, self.cfg))

    def _act(self, pid: int, gesture: str, m: Motion) -> None:
        """Turn one gesture event into a game action, as ``main.py`` did."""
        player = self.get_player(pid)
        # Towards the middle: a player on the left half fires to the right.
        direction = 1 if player.x < self.frame_w / 2 else -1
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
        """The game state, in the attacker's or owner's colour, until the
        renderer draws it properly: skeleton (thin), health bar and paint
        splats on each body box, bolts (the fire blast and its break-up where
        it ended, or a dot),
        beams (the magic ball and beam, or a
        line with a growing dot while charging), shields (ellipse beside the
        player; thin while fading; or the fire ring around the chest), this
        frame's actions, and the winner."""
        unit = max(canvas.shape[1] / 100.0, 2.0)  # 1% of the frame width, px
        for pid, m in state.motion.items():
            draw_skeleton(canvas, m.kpts, m.valid, player_color(pid), 1)

        for pid, player in sorted(self.players.items()):
            if player.bounds is None:
                continue
            left, top, right, bottom = player.bounds
            w, h = right - left, bottom - top
            for splat in player.paint_splats:
                c = (left + splat.x * w, top + splat.y * h)
                cv2.circle(canvas, pt(c), max(2, int(0.06 * w)), player_color(splat.owner_id), -1,
                           cv2.LINE_AA)
            # Health bar below the body, kept inside the frame.
            (cx, cy), length = self.bar_geometry(pid, state)
            x0, x1 = int(cx - length / 2), int(cx + length / 2)
            y0 = min(int(cy + 2 * unit), canvas.shape[0] - 10)
            cv2.rectangle(canvas, (x0, y0 - int(unit)), (x1, y0), (40, 40, 40), -1)
            fill = x0 + int((x1 - x0) * player.health / self.max_health)
            cv2.rectangle(canvas, (x0, y0 - int(unit)), (fill, y0), player_color(pid), -1)
            label = f"P{pid} {player.health}"
            if self.shielded(pid):
                label += " shield"
            draw_text(canvas, label, (x0, y0 - int(unit) - 6), player_color(pid))

        bolt_h = int(self.cfg["bolt_sprite_h"] * canvas.shape[0])
        for end in self.bolt_ends:
            fire = self._bolt_effect(end.owner_id)
            k = fire.end_index(end.age) if fire is not None else None
            if k is not None:
                fire.draw(canvas, end.x, end.y, end.direction, k, bolt_h)
        for b in self.projectiles:
            fire = self._bolt_effect(b.owner_id)
            if fire is not None:
                fire.draw(canvas, b.x, b.y, b.direction, fire.flight_index(b.age), bolt_h)
                continue
            c = player_color(b.owner_id)
            cv2.circle(canvas, pt((b.x, b.y)), int(unit), c, -1, cv2.LINE_AA)
            cv2.line(canvas, pt((b.x, b.y)), pt((b.x - b.direction * 4 * unit, b.y)), c, 2, cv2.LINE_AA)

        charge = self.cfg["beam_charge_s"]
        life = charge + self.cfg["beam_duration_s"]
        thickness = max(2, int(2 * self.cfg["beam_hit_radius_h"] * canvas.shape[0]))
        ball_px = int(self.cfg["beam_ball_h"] * canvas.shape[0])
        for beam in self.beams:
            c = player_color(beam.owner_id)
            magic = self._beam_effect(beam.owner_id)
            if magic is not None:
                magic.draw(canvas, beam.start_x, beam.x, beam.y, beam.direction, beam.age,
                           charge, life, thickness, ball_px)
            elif beam.age < charge:
                r = int(unit * (1 + 2 * beam.age / charge))
                cv2.circle(canvas, pt((beam.start_x, beam.y)), r, c, 2, cv2.LINE_AA)
            else:
                cv2.line(canvas, pt((beam.start_x, beam.y)), pt((beam.x, beam.y)), c, thickness,
                         cv2.LINE_AA)

        for pid, shield in self.shields.items():
            ring = self._shield_effect(pid)
            if ring is not None:
                center, diameter = self.ring_geometry(pid, shield, state)
                fade = 1.0 if shield.remaining > 0 else 1.0 + shield.remaining / self.cfg["shield_fade_s"]
                ring.draw(canvas, center, diameter, shield.age, max(fade, 0.0))
                continue
            player = self.players.get(pid)
            h = (player.bounds[3] - player.bounds[1]) if player and player.bounds is not None else 40 * unit
            center = pt((shield.x + shield.direction * 6 * unit, shield.y - 0.25 * h))
            axes = (int(1.5 * unit), int(0.3 * h))
            cv2.ellipse(canvas, center, axes, 0, 0, 360, player_color(pid),
                        3 if shield.remaining > 0 else 1, cv2.LINE_AA)

        lines = [f"P{pid} {action}" for pid, action in self.actions]
        draw_lines(canvas, lines, (10, canvas.shape[0] - 20 * len(lines) - 10))
        if self.winner_id is not None:
            draw_text(canvas, f"P{self.winner_id} wins - press r for a new round",
                      (int(canvas.shape[1] * 0.3), canvas.shape[0] // 2),
                      player_color(self.winner_id), 1.0)

    def _bolt_effect(self, owner_id: int) -> FireBolt | None:
        """The bolt's look for one player (game.bolt_sprite), loaded on first
        use so the game logic runs without it; None = draw a plain dot.

        The frames' own colours belong to player bolt_effect_owner; for any
        other player every hue is turned so it takes their team colour."""
        cfg = self.cfg
        if not cfg.get("bolt_sprite"):
            return None
        if owner_id not in self._fire:
            def load(hue_shift: int = 0) -> FireBolt:
                return FireBolt(cfg["bolt_sprite"], cfg["bolt_sprite_launch"],
                                cfg["bolt_sprite_flight"], cfg["bolt_sprite_fps"],
                                cfg["bolt_end_s"], hue_shift)
            owner = cfg["bolt_effect_owner"]
            base = self._fire.get(owner) or load()
            self._fire[owner] = base
            if owner_id != owner:
                self._fire[owner_id] = load(hue_of(player_color(owner_id)) - base.hue)
        return self._fire[owner_id]

    def ring_geometry(self, pid: int, shield: Shield, state: FrameState) -> tuple[tuple[float, float], float]:
        """Centre and diameter (px) of the shield's fire ring: on the
        player's chest (halfway between shoulders and hips), following them,
        shield_ring_sw shoulder widths across; without a shoulder width, where
        the shield went up and as wide as the body box."""
        m = state.motion.get(pid)
        sw = shoulder_width(m.kpts, m.valid, self.cfg["min_shoulder_px"]) if m is not None else None
        if sw is not None:
            shoulder_y = (m.kpts[sk.L_SHOULDER][1] + m.kpts[sk.R_SHOULDER][1]) / 2.0
            x, hip_y = self.body_position(m)
            return (x, (shoulder_y + hip_y) / 2.0), self.cfg["shield_ring_sw"] * sw
        player = self.players.get(pid)
        width = (player.bounds[2] - player.bounds[0]) if player and player.bounds is not None else 100.0
        return (shield.x, shield.y), float(width)

    def _shield_effect(self, owner_id: int) -> FireRing | None:
        """The shield's look for one player (game.shield_sprite), loaded on
        first use; None = draw the plain ellipse. Hue-turned for players other
        than shield_effect_owner, as for the bolt and the beam."""
        cfg = self.cfg
        if not cfg.get("shield_sprite"):
            return None
        if owner_id not in self._rings:
            def load(hue_shift: int = 0) -> FireRing:
                return FireRing(cfg["shield_sprite"], cfg["shield_form_frames"],
                                cfg["shield_loop_frames"], cfg["shield_form_s"],
                                cfg["shield_sprite_fps"], hue_shift)
            owner = cfg["shield_effect_owner"]
            base = self._rings.get(owner) or load()
            self._rings[owner] = base
            if owner_id != owner:
                self._rings[owner_id] = load(hue_of(player_color(owner_id)) - base.hue)
        return self._rings[owner_id]

    def _beam_effect(self, owner_id: int) -> MagicBeam | None:
        """The beam's look for one player (game.beam_ball, game.beam_band),
        loaded on first use so the game logic runs without it; None = draw a
        plain line.

        The pack's own colours belong to player beam_effect_owner; for any
        other player every hue is turned so it takes their team colour."""
        cfg = self.cfg
        if not cfg.get("beam_ball"):
            return None
        if owner_id not in self._magic:
            def load(hue_shift: int = 0) -> MagicBeam:
                return MagicBeam(cfg["beam_ball"], cfg["beam_band"], cfg["beam_ramp_s"], hue_shift)
            owner = cfg["beam_effect_owner"]
            base = self._magic.get(owner) or load()
            self._magic[owner] = base
            if owner_id != owner:
                self._magic[owner_id] = load(hue_of(player_color(owner_id)) - base.hue)
        return self._magic[owner_id]

    def bar_geometry(self, pid: int, state: FrameState) -> tuple[tuple[float, float], float]:
        """Centre (px) and length (px) of the debug health bar.

        It sits at the bottom of the body box, centred under the shoulders,
        out of the way of beams fired from raised hands. Its length depends
        only on the player's distance to the camera, not on their arms (the
        body box grows with an extended arm): debug_bar_face_widths face
        widths; without a face, debug_bar_shoulder_widths shoulder widths;
        without either, the body box width."""
        player = self.players[pid]
        left, _, right, bottom = player.bounds
        m = state.motion.get(pid)
        sw = shoulder_width(m.kpts, m.valid, self.cfg["min_shoulder_px"]) if m is not None else None
        x = player.x if sw is not None else (left + right) / 2.0
        face = state.faces.get(pid)
        if face is not None and face.scale > 0:
            length = self.cfg["debug_bar_face_widths"] * face.scale
        elif sw is not None:
            length = self.cfg["debug_bar_shoulder_widths"] * sw
        else:
            length = float(right - left)
        return (float(x), float(bottom)), length

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
