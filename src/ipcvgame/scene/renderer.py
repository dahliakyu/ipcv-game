"""Task 5 — game renderer, from Colour Clash renderer.py.

The game view, in drawing order: the camera frame (pixelated, as in
renderer.py), each player's paint splats (renderer.py's pixel crosses, on the
torso), the
game's objects drawn with their sprites (fire blast bolts, magic ball and
beam, ring of fire shields), face effects, health bars pinned to the bottom,
renderer.py's top bar
with the controls, the end-of-game panel and a timing HUD. The
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
# renderer.py's controls line, with our three gestures and what they fire.
CONTROLS = "PUNCH: FIRE BOLT   HANDS UP: MAGIC BEAM   CROSS ARMS: SHIELD"
KEYS = "R: NEW ROUND   Q: QUIT"


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
        # Paint only on players the identity stage sees now (renderer.py drew
        # only detected players).
        visible = [pid for pid, p in sorted(state.players.items())
                   if p.status == "active" and pid in game.players and game.players[pid].bounds is not None]
        for pid in visible:
            player = game.players[pid]
            self._draw_player_paint(canvas, self.paint_area(pid, state, game), player.paint_splats,
                                    1.0 - player.health / game.max_health, game.splat_band)
        if game.winner_id is None:
            # After a win the game stops (game.py), so bolts and beams in
            # flight would stand still: they are not drawn.
            self.draw_game(canvas, state, game)
        self.face_effects.draw(canvas, state.faces)
        for pid in sorted(game.players):
            self._draw_player_health(canvas, pid, game)
        top = self._draw_hud(canvas, has_player=bool(visible), game_over=game.winner_id is not None)
        if game.winner_id is not None:
            self._draw_game_over(canvas, game.winner_id)
        if self.show_timings:
            draw_timings(canvas, state.timings, (10, top + 20))
        return canvas

    # -------------------------------------------------------------------- HUD

    def _scale(self, canvas: np.ndarray, key: str) -> float:
        """A renderer.py font scale (for 480p) at this frame height."""
        return self.cfg[key] * canvas.shape[0] / 480.0

    def _draw_hud(self, canvas: np.ndarray, has_player: bool, game_over: bool) -> int:
        """renderer.py's top bar: "COLOUR CLASH" in P1's colour, and the
        controls on the right, or "STEP INTO VIEW" while nobody is seen; the
        keys at the bottom, centred between the health bars. Returns the
        bar's height (px)."""
        h, w = canvas.shape[:2]
        bar = int(self.cfg["hud_h"] * h)
        title, text = self._scale(canvas, "hud_title_scale"), self._scale(canvas, "hud_text_scale")
        thick = max(1, int(round(text)))
        cv2.rectangle(canvas, (0, 0), (w, bar), (27, 16, 17), -1)
        (title_w, _), _ = cv2.getTextSize("COLOUR CLASH", cv2.FONT_HERSHEY_PLAIN, title, thick)
        cv2.putText(canvas, "COLOUR CLASH", (12, int(bar * 0.68)), cv2.FONT_HERSHEY_PLAIN, title,
                    player_color(1), thick, cv2.LINE_8)
        status = CONTROLS if has_player else "STEP INTO VIEW"
        # Shrunk to fit beside the title on a narrow frame.
        scale = fit_scale(status, text, thick, w - title_w - 36)
        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_PLAIN, scale, thick)
        cv2.putText(canvas, status, (w - tw - 12, int(bar * 0.65)), cv2.FONT_HERSHEY_PLAIN, scale, WHITE,
                    thick, cv2.LINE_8)
        if not game_over:
            gap = w - 2 * (int(self.cfg["bar_w"] * w) + 24)     # between the two health bars
            scale = fit_scale(KEYS, text, thick, gap)
            (kw, kh), _ = cv2.getTextSize(KEYS, cv2.FONT_HERSHEY_PLAIN, scale, thick)
            x = (w - kw) // 2
            # Dark backing like the health bars, so the text reads on a bright floor.
            cv2.rectangle(canvas, (x - 6, h - 18 - kh), (x + kw + 6, h - 6), (27, 16, 17), -1)
            cv2.putText(canvas, KEYS, (x, h - 12), cv2.FONT_HERSHEY_PLAIN, scale, WHITE, thick, cv2.LINE_8)
        return bar

    def _draw_game_over(self, canvas: np.ndarray, winner_id: int) -> None:
        """renderer.py's end panel: the frame dimmed, a dark panel in the
        middle framed in the winner's colour, "PLAYER N WINS!" and the keys."""
        h, w = canvas.shape[:2]
        canvas[:] = (canvas * self.cfg["game_over_dim"]).astype(np.uint8)
        k = h / 480.0
        panel_w, panel_h = min(w - 24, int(380 * k)), int(84 * k)
        left, top = (w - panel_w) // 2, h // 2 - panel_h // 2
        color = player_color(winner_id)
        corners = (left, top), (left + panel_w, top + panel_h)
        cv2.rectangle(canvas, *corners, (27, 16, 17), -1)
        cv2.rectangle(canvas, *corners, color, max(2, int(3 * k)), cv2.LINE_8)
        cv2.putText(canvas, f"PLAYER {winner_id} WINS!", (left + int(28 * k), top + int(38 * k)),
                    cv2.FONT_HERSHEY_PLAIN, 2 * k, color, max(1, int(2 * k)), cv2.LINE_8)
        cv2.putText(canvas, KEYS, (left + int(28 * k), top + int(66 * k)),
                    cv2.FONT_HERSHEY_PLAIN, fit_scale(KEYS, k, max(1, int(k)), panel_w - int(40 * k)), WHITE,
                    max(1, int(k)), cv2.LINE_8)

    # ------------------------------------------------------ paint and health

    def paint_area(self, pid: int, state: FrameState, game: Game) -> tuple[float, float, float, float]:
        """Where the player's paint is drawn (x1, y1, x2, y2 px): the torso,
        between the shoulders and from the shoulder line to the hips, so the
        splats stay on the shirt when an arm goes out. renderer.py drew them
        in the body box, which widens with the arms, so they slid off the
        body. Without a shoulder width: the box's splat band."""
        m = state.motion.get(pid)
        torso = game.torso(m) if m is not None else None
        if torso is not None:
            x, shoulder_y, hip_y, sw = torso
            return x - sw / 2, shoulder_y, x + sw / 2, hip_y
        left, top, right, bottom = game.players[pid].bounds
        lo, hi = game.splat_band
        w, h = right - left, bottom - top
        return left + lo * w, top + lo * h, left + hi * w, top + hi * h

    def _draw_player_paint(self, canvas: np.ndarray, area: tuple, splats: list, coverage: float,
                           band: tuple[float, float]) -> None:
        """renderer.py: each splat is a pixel cross (5 cells) in the
        attacker's colour. The cells grow as the player's health drops
        (coverage = 1 - health / max_health), so a beaten player looks more
        painted.

        Changed: the game keeps splats inside a band of the body box
        (game.splat_band); that band is stretched over `area` (the torso), and
        the cells are fractions of the area, not of the box."""
        h, w = canvas.shape[:2]
        left, top, right, bottom = (int(v) for v in area)
        area_w, area_h = max(1, right - left), max(1, bottom - top)
        lo, hi = band
        (cw0, cw1), (ch0, ch1) = self.cfg["splat_cell_w"], self.cfg["splat_cell_h"]
        cell_w = max(2, int(area_w * (cw0 + cw1 * coverage)))
        cell_h = max(2, int(area_h * (ch0 + ch1 * coverage)))
        for splat in splats:
            u = min(max((splat.x - lo) / (hi - lo), 0.0), 1.0)
            v = min(max((splat.y - lo) / (hi - lo), 0.0), 1.0)
            cx, cy = left + int(u * area_w), top + int(v * area_h)
            for ox, oy in ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)):
                x, y = cx + ox * cell_w, cy + oy * cell_h
                x0, y0 = max(left, 0, x), max(top, 0, y)
                x1, y1 = min(right, w, x + cell_w - 1), min(bottom, h, y + cell_h - 1)
                if x0 < x1 and y0 < y1:
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), player_color(splat.owner_id), -1, cv2.LINE_8)

    def _draw_player_health(self, canvas: np.ndarray, pid: int, game: Game) -> None:
        """renderer.py's health bar (dark backing, "PLAYER N", off-white
        outline, fill in the team colour), pinned to the bottom of the frame
        like in a fighting game: P1 bottom left, P2 bottom right, each filling
        from its outer edge, so it drains towards the corner. Other pids have
        no slot. Drawn for every known player, also while out of view."""
        if pid not in (1, 2):
            return
        h, w = canvas.shape[:2]
        bar_h = max(4, int(self.cfg["bar_h"] * h))
        length = int(self.cfg["bar_w"] * w)
        font = self.cfg["bar_font_h"] * h / 10.0          # FONT_HERSHEY_PLAIN is ~10 px tall at scale 1
        label_h = int(12 * font)
        left = 12 if pid == 1 else w - 12 - length
        right = left + length
        top = h - 12 - bar_h
        color = player_color(pid)
        label = f"PLAYER {pid}"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font, 1)
        cv2.rectangle(canvas, (left - 2, top - label_h - 2), (right + 2, top + bar_h + 2), (27, 16, 17), -1)
        cv2.putText(canvas, label, (left if pid == 1 else right - lw, top - 3), cv2.FONT_HERSHEY_PLAIN, font,
                    color, 1, cv2.LINE_8)
        cv2.rectangle(canvas, (left, top), (right, top + bar_h), WHITE, 1, cv2.LINE_8)
        fill = int((length - 2) * max(0, game.players[pid].health) / game.max_health)
        if fill > 0:
            x0, x1 = (left + 1, left + fill) if pid == 1 else (right - fill, right - 1)
            cv2.rectangle(canvas, (x0, top + 1), (x1, top + bar_h - 1), color, -1, cv2.LINE_8)

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


def fit_scale(text: str, scale: float, thickness: int, max_w: float) -> float:
    """The font scale, made smaller if needed so `text` is at most max_w px wide."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thickness)
    return scale if tw <= max_w else scale * max(max_w, 1.0) / tw


def pixelate(image: np.ndarray, size: int) -> np.ndarray:
    """renderer.py's retro look: shrink by `size` and scale back up with the
    nearest pixel, so every size x size block has one colour. 1 = a copy."""
    if size <= 1:
        return image.copy()
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, w // size), max(1, h // size)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def draw_timings(canvas: np.ndarray, timings: dict[str, float], org: tuple[int, int] = (10, 24)) -> None:
    """Per-stage ms of the previous frame, plus FPS from its total."""
    lines = [f"{k:>9s} {v:6.1f} ms" for k, v in timings.items() if k != "total"]
    if "total" in timings and timings["total"] > 0:
        lines.append(f"{'total':>9s} {timings['total']:6.1f} ms  ({1000 / timings['total']:.0f} FPS)")
    draw_lines(canvas, lines, org)
