"""Task 5 — game renderer, from Colour Clash renderer.py.

The game view, in drawing order: the camera frame (pixelated, as in
renderer.py), each player's paint splats (renderer.py's pixel crosses), the
game's objects drawn with their sprites (fire blast bolts, magic ball and
beam, ring of fire shields), face effects, health bars and a timing HUD. The
game logic (interaction/game.py) only keeps state; this module decides how it
looks (renderer merge, docs/merge_log.md, Task 5).

Sprites are in each team's colour: the frames' own colours belong to one
player (*_effect_owner), and every other player gets a copy with every hue
turned to their team colour (scene/draw.py). They are loaded on first use, so
the game logic and tests run without them; a sprite set to null in the config
is drawn as a plain shape instead.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from ipcvgame.core.types import FrameState
from ipcvgame.face.effects import FaceEffects
from ipcvgame.interaction.game import Game, Shield
from ipcvgame.scene.draw import draw_bbox, draw_lines, draw_skeleton, hue_of, player_color, pt
from ipcvgame.scene.fire_bolt import FireBolt
from ipcvgame.scene.fire_ring import FireRing
from ipcvgame.scene.magic_beam import MagicBeam

WHITE = (235, 229, 210)     # renderer.py's off-white (BGR), for outlines and text


class Renderer:
    def __init__(self, cfg: dict, face_effects: FaceEffects):
        self.cfg = cfg
        self.face_effects = face_effects
        self.show_timings = bool(cfg.get("show_timings", True))
        self.show_skeleton = bool(cfg.get("show_skeleton", False))
        self._fire: dict[int, FireBolt] = {}     # bolt look per player
        self._magic: dict[int, MagicBeam] = {}   # beam look per player
        self._rings: dict[int, FireRing] = {}    # shield look per player

    def draw(self, state: FrameState, game: Game) -> np.ndarray:
        canvas = pixelate(state.frame.image, int(self.cfg.get("pixel_size", 1)))
        if self.show_skeleton:
            for pid, p in state.players.items():
                if p.status == "active":
                    draw_bbox(canvas, p.bbox, player_color(pid), f"P{pid}")
            for pid, m in state.motion.items():
                draw_skeleton(canvas, m.kpts, m.valid, player_color(pid))
        # Paint and health only for players the identity stage sees now
        # (renderer.py drew only detected players).
        visible = [pid for pid, p in sorted(state.players.items())
                   if p.status == "active" and pid in game.players and game.players[pid].bounds is not None]
        for pid in visible:
            player = game.players[pid]
            self._draw_player_paint(canvas, player.bounds, player.paint_splats,
                                    1.0 - player.health / game.max_health)
        self.draw_game(canvas, state, game)
        self.face_effects.draw(canvas, state.faces)
        for pid in visible:
            self._draw_player_health(canvas, pid, state, game)
        if self.show_timings:
            draw_timings(canvas, state.timings)
        return canvas

    # ------------------------------------------------------ paint and health

    def _draw_player_paint(self, canvas: np.ndarray, bounds: np.ndarray, splats: list,
                           coverage: float) -> None:
        """renderer.py: each splat is a pixel cross (5 cells) inside the body
        box, in the attacker's colour. The cells grow as the player's health
        drops (coverage = 1 - health / max_health), so a beaten player looks
        more painted."""
        h, w = canvas.shape[:2]
        left, top, right, bottom = (int(v) for v in bounds)
        box_w, box_h = max(1, right - left), max(1, bottom - top)
        (cw0, cw1), (ch0, ch1) = self.cfg["splat_cell_w"], self.cfg["splat_cell_h"]
        cell_w = max(2, int(box_w * (cw0 + cw1 * coverage)))
        cell_h = max(2, int(box_h * (ch0 + ch1 * coverage)))
        for splat in splats:
            cx, cy = left + int(splat.x * box_w), top + int(splat.y * box_h)
            for ox, oy in ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)):
                x, y = cx + ox * cell_w, cy + oy * cell_h
                x0, y0 = max(left, 0, x), max(top, 0, y)
                x1, y1 = min(right, w, x + cell_w - 1), min(bottom, h, y + cell_h - 1)
                if x0 < x1 and y0 < y1:
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), player_color(splat.owner_id), -1, cv2.LINE_8)

    def _draw_player_health(self, canvas: np.ndarray, pid: int, state: FrameState, game: Game) -> None:
        """renderer.py's health bar (dark backing, "PLAYER N", white outline,
        fill in the team colour), at the place from bar_geometry(): below the
        body, kept inside the frame (Task 4 modifications)."""
        h, w = canvas.shape[:2]
        (cx, cy), length = self.bar_geometry(pid, state, game)
        bar_h = max(4, int(self.cfg["bar_h"] * h))
        font = self.cfg["bar_font_h"] * h / 10.0          # FONT_HERSHEY_PLAIN is ~10 px tall at scale 1
        label_h = int(12 * font)
        left = int(min(max(cx - length / 2, 4), w - length - 4))
        right = left + int(length)
        top = int(min(cy + bar_h, h - bar_h - 4))
        color = player_color(pid)
        cv2.rectangle(canvas, (left - 2, top - label_h - 2), (right + 2, top + bar_h + 2), (27, 16, 17), -1)
        cv2.putText(canvas, f"PLAYER {pid}", (left, top - 3), cv2.FONT_HERSHEY_PLAIN, font, color, 1,
                    cv2.LINE_8)
        cv2.rectangle(canvas, (left, top), (right, top + bar_h), WHITE, 1, cv2.LINE_8)
        fill = int((right - left - 2) * max(0, game.players[pid].health) / game.max_health)
        if fill > 0:
            cv2.rectangle(canvas, (left + 1, top + 1), (left + fill, top + bar_h - 1), color, -1, cv2.LINE_8)

    # ----------------------------------------------------------- game objects

    def draw_game(self, canvas: np.ndarray, state: FrameState, game: Game) -> None:
        """Bolts (fire blast, and its break-up where it ended), beams (magic
        ball and beam) and shields (ring of fire on the chest)."""
        unit = max(canvas.shape[1] / 100.0, 2.0)    # 1% of the frame width, px
        bolt_h = int(self.cfg["bolt_sprite_h"] * canvas.shape[0])
        for end in game.bolt_ends:
            fire = self.bolt_effect(end.owner_id, game)
            k = fire.end_index(end.age) if fire is not None else None
            if k is not None:
                fire.draw(canvas, end.x, end.y, end.direction, k, bolt_h)
        for b in game.projectiles:
            fire = self.bolt_effect(b.owner_id, game)
            if fire is not None:
                fire.draw(canvas, b.x, b.y, b.direction, fire.flight_index(b.age), bolt_h)
            else:
                cv2.circle(canvas, pt((b.x, b.y)), int(unit), player_color(b.owner_id), -1, cv2.LINE_AA)

        thickness = max(2, int(2 * game.beam_band_px()))
        ball_px = int(self.cfg["beam_ball_h"] * canvas.shape[0])
        for beam in game.beams:
            magic = self.beam_effect(beam.owner_id)
            if magic is not None:
                magic.draw(canvas, beam.start_x, beam.x, beam.y, beam.direction, beam.age,
                           game.beam_charge_s, game.beam_life_s, thickness, ball_px)
            elif beam.age >= game.beam_charge_s:
                cv2.line(canvas, pt((beam.start_x, beam.y)), pt((beam.x, beam.y)),
                         player_color(beam.owner_id), thickness, cv2.LINE_AA)

        for pid, shield in game.shields.items():
            ring = self.shield_effect(pid)
            center, diameter = self.ring_geometry(pid, shield, state, game)
            if ring is not None:
                ring.draw(canvas, center, diameter, shield.age, game.shield_opacity(shield))
            elif game.shield_opacity(shield) > 0:
                cv2.circle(canvas, pt(center), int(diameter / 2), player_color(pid), 3, cv2.LINE_AA)

    def ring_geometry(self, pid: int, shield: Shield, state: FrameState,
                      game: Game) -> tuple[tuple[float, float], float]:
        """Centre and diameter (px) of the shield's fire ring: on the
        player's chest (halfway between shoulders and hips), following them,
        shield_ring_sw shoulder widths across; without a shoulder width, where
        the shield went up and as wide as the body box."""
        m = state.motion.get(pid)
        torso = game.torso(m) if m is not None else None
        if torso is not None:
            x, shoulder_y, hip_y, sw = torso
            return (x, (shoulder_y + hip_y) / 2.0), self.cfg["shield_ring_sw"] * sw
        player = game.players.get(pid)
        width = (player.bounds[2] - player.bounds[0]) if player and player.bounds is not None else 100.0
        return (shield.x, shield.y), float(width)

    def bar_geometry(self, pid: int, state: FrameState, game: Game) -> tuple[tuple[float, float], float]:
        """Centre (px) and length (px) of the health bar (Task 4
        modifications).

        It sits at the bottom of the body box, centred under the shoulders,
        out of the way of beams fired from raised hands. Its length depends
        only on the player's distance to the camera, not on their arms (the
        body box grows with an extended arm): bar_face_widths face widths;
        without a face, bar_shoulder_widths shoulder widths; without either,
        the body box width."""
        player = game.players[pid]
        left, _, right, bottom = player.bounds
        m = state.motion.get(pid)
        torso = game.torso(m) if m is not None else None
        x = torso[0] if torso is not None else (left + right) / 2.0
        face = state.faces.get(pid)
        if face is not None and face.scale > 0:
            length = self.cfg["bar_face_widths"] * face.scale
        elif torso is not None:
            length = self.cfg["bar_shoulder_widths"] * torso[3]
        else:
            length = float(right - left)
        return (float(x), float(bottom)), length

    # ---------------------------------------------------------------- sprites

    def _team_copy(self, cache: dict, owner: int, owner_id: int, load: Callable[[int], object]):
        """The effect for owner_id: the frames as they are for `owner`, a
        hue-turned copy in their team colour for everyone else."""
        if owner_id not in cache:
            base = cache.get(owner) or load(0)
            cache[owner] = base
            if owner_id != owner:
                cache[owner_id] = load(hue_of(player_color(owner_id)) - base.hue)
        return cache[owner_id]

    def bolt_effect(self, owner_id: int, game: Game) -> FireBolt | None:
        """The bolt's look for one player (bolt_sprite); None = a plain dot."""
        cfg = self.cfg
        if not cfg.get("bolt_sprite"):
            return None
        return self._team_copy(self._fire, cfg["bolt_effect_owner"], owner_id, lambda shift: FireBolt(
            cfg["bolt_sprite"], cfg["bolt_sprite_launch"], cfg["bolt_sprite_flight"],
            cfg["bolt_sprite_fps"], game.bolt_end_s, shift))

    def beam_effect(self, owner_id: int) -> MagicBeam | None:
        """The beam's look for one player (beam_ball, beam_band); None = a
        plain line."""
        cfg = self.cfg
        if not cfg.get("beam_ball"):
            return None
        return self._team_copy(self._magic, cfg["beam_effect_owner"], owner_id, lambda shift: MagicBeam(
            cfg["beam_ball"], cfg["beam_band"], cfg["beam_ramp_s"], shift))

    def shield_effect(self, owner_id: int) -> FireRing | None:
        """The shield's look for one player (shield_sprite); None = a plain
        circle."""
        cfg = self.cfg
        if not cfg.get("shield_sprite"):
            return None
        return self._team_copy(self._rings, cfg["shield_effect_owner"], owner_id, lambda shift: FireRing(
            cfg["shield_sprite"], cfg["shield_form_frames"], cfg["shield_loop_frames"],
            cfg["shield_form_s"], cfg["shield_sprite_fps"], shift))


def pixelate(image: np.ndarray, size: int) -> np.ndarray:
    """renderer.py's retro look: shrink by `size` and scale back up with the
    nearest pixel, so every size x size block has one colour. 1 = a copy."""
    if size <= 1:
        return image.copy()
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // size), max(1, h // size)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def draw_timings(canvas: np.ndarray, timings: dict[str, float]) -> None:
    """Per-stage ms of the previous frame, plus FPS from its total."""
    lines = [f"{k:>9s} {v:6.1f} ms" for k, v in timings.items() if k != "total"]
    if "total" in timings and timings["total"] > 0:
        lines.append(f"{'total':>9s} {timings['total']:6.1f} ms  ({1000 / timings['total']:.0f} FPS)")
    draw_lines(canvas, lines)
