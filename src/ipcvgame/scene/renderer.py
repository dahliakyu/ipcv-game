"""Task 5 — game renderer, from Colour Clash renderer.py.

The game view, in drawing order: the camera frame (pixelated, as in
renderer.py), each player's paint splats (renderer.py's pixel crosses, on the
torso), the
game's objects drawn with their sprites (fire blast bolts, magic ball and
beam, ring of fire shields), face effects, each player's status above the
head (name tag, dizzy stars, crown; docs/merge_log.md, "Face effects"),
health bars pinned to the bottom,
renderer.py's top bar
with the controls, the end-of-game panel and a timing HUD. The
game logic (interaction/game.py) only keeps state; this module decides how it
looks (renderer merge, docs/merge_log.md, Task 5).

Around the round (game start, docs/merge_log.md), the game's phase
decides what else is drawn: the players' start places while waiting, an
instruction panel per player during calibration, 3-2-1 during the countdown,
"FIGHT!" as a round starts and the result panel after a win. The top bar says
what each phase expects. Health bars are shown from the countdown on.

Sprites are in each team's colour: the frames' own colours belong to one
player (*_effect_owner), and every other player gets a copy with every hue
turned to their team colour (scene/draw.py). They are loaded on first use, so
the game logic and tests run without them; a sprite set to null in the config
is drawn as a plain shape instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from ipcvgame.core.filters import OneEuro
from ipcvgame.core.types import Face, FrameState
from ipcvgame.face.effects import FaceEffects
from ipcvgame.interaction.game import CALIBRATION_MOVES, Game, Shield
from ipcvgame.scene.draw import draw_bbox, draw_lines, draw_skeleton, hue_of, player_color, pt
from ipcvgame.scene.fire_bolt import FireBolt
from ipcvgame.scene.fire_ring import FireRing
from ipcvgame.scene.instructions import MOVE_TEXT, draw_figure, draw_tick
from ipcvgame.scene.magic_beam import MagicBeam

WHITE = (235, 229, 210)     # renderer.py's off-white (BGR), for outlines and text
# renderer.py's controls line, with our three gestures and what they fire.
CONTROLS = "PUNCH: FIRE BOLT   HANDS UP: MAGIC BEAM   CROSS ARMS: SHIELD"
KEYS = "R: NEW ROUND   Q: QUIT"
DARK = (27, 16, 17)         # renderer.py's dark backing (BGR), for bars and panels
GREY = (120, 115, 105)      # moves still to do, in the instruction panels
STAR = (60, 220, 255)       # dizzy stars (BGR yellow)
GOLD = (40, 190, 255)       # the crown (BGR gold)
# Calibration panel while the players do not stand level (Game.line_up_hint()).
LINE_UP_TEXT = {"back": "STEP BACK TO LINE UP", "wait": "WAIT FOR THE OTHER PLAYER"}
# Neutral pose with two players: the standing instruction for the line-up.
STAND_LEVEL_TEXT = "STAND LEVEL WITH THE OTHER PLAYER"
# The top bar's text in the phases around the round (playing: CONTROLS).
PHASE_TEXT = {
    "waiting": "STAND IN YOUR PLACE",
    "calibration": "LEARN THE MOVES",
    "countdown": "GET READY",
    "result": "HANDS UP TO PLAY AGAIN",
}


@dataclass
class _Tag:
    """One player's name tag between frames, so it does not jitter."""
    filter: OneEuro          # on the tag's position (px)
    text_h: float = 0.0      # text height (px) in use; changes only past tag_size_tol
    side: int = 0            # 0 = above the head; +1 / -1 = beside the hat, to the right / left


class Renderer:
    def __init__(self, cfg: dict, face_effects: FaceEffects):
        self.cfg = cfg
        self.face_effects = face_effects
        self.show_timings = bool(cfg.get("show_timings", True))
        self.show_skeleton = bool(cfg.get("show_skeleton", False))
        self._fire: dict[int, FireBolt] = {}     # bolt look per player
        self._magic: dict[int, MagicBeam] = {}   # beam look per player
        self._rings: dict[int, FireRing] = {}    # shield look per player
        self._tags: dict[int, _Tag] = {}         # name tag per player (_draw_tag())

    def reset(self, pid: int | None = None) -> None:
        """Forget the name tags' smoothing (one player on re-entry, or all)."""
        if pid is None:
            self._tags.clear()
        else:
            self._tags.pop(pid, None)

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
        phase = game.phase
        if phase != "result":
            # After a win the game stops (game.py), so bolts and beams in
            # flight would stand still: they are not drawn.
            self.draw_game(canvas, state, game)
        # The game's hits start the face effects' hit reaction; a drink drunk
        # puffs the drinker's cheeks (consumables, extra_features.md feature 2).
        self.face_effects.update(game.hits, state.faces, state.frame.t)
        for d in game.drunk:
            self.face_effects.bulge(d.pid, state.frame.t)
        self.face_effects.draw(canvas, state.faces, state.frame.t)
        self._draw_status(canvas, state, game)
        if phase in ("countdown", "playing", "result"):
            for pid in sorted(game.players):
                self._draw_player_health(canvas, pid, game)
        top = self._draw_hud(canvas, status_text(phase, has_player=bool(visible)),
                             game_over=phase == "result")
        if phase == "waiting":
            for place in game.round_players:
                self._draw_start_place(canvas, place, game.place_taken(state, place), len(game.round_players))
        elif phase == "calibration":
            for pid in game.round_players:
                self._draw_instructions(canvas, pid, game, state.frame.t)
        elif phase == "countdown":
            left = game.countdown_left()
            n = max(1, math.ceil(left))
            # Each number starts large and settles over its second.
            self._draw_big_text(canvas, str(n), 1.3 - 0.3 * (n - left))
        elif phase == "playing" and game.intro and game.phase_elapsed() < self.cfg["fight_s"]:
            self._draw_big_text(canvas, "FIGHT!", 0.6)
        elif phase == "result":
            self._draw_game_over(canvas, game)
        if phase in ("calibration", "playing"):
            self._draw_not_facing(canvas, state, game)
        if self.show_timings:
            draw_timings(canvas, state.timings, (10, top + 20))
        return canvas

    # -------------------------------------------------------------------- HUD

    def _scale(self, canvas: np.ndarray, key: str) -> float:
        """A renderer.py font scale (for 480p) at this frame height."""
        return self.cfg[key] * canvas.shape[0] / 480.0

    def _draw_hud(self, canvas: np.ndarray, status: str, game_over: bool) -> int:
        """renderer.py's top bar: "COLOUR CLASH" in P1's colour, and the
        status on the right (status_text()); the keys at the bottom, centred
        between the health bars. Returns the bar's height (px)."""
        h, w = canvas.shape[:2]
        bar = int(self.cfg["hud_h"] * h)
        title, text = self._scale(canvas, "hud_title_scale"), self._scale(canvas, "hud_text_scale")
        thick = max(1, int(round(text)))
        cv2.rectangle(canvas, (0, 0), (w, bar), DARK, -1)
        (title_w, _), _ = cv2.getTextSize("COLOUR CLASH", cv2.FONT_HERSHEY_PLAIN, title, thick)
        cv2.putText(canvas, "COLOUR CLASH", (12, int(bar * 0.68)), cv2.FONT_HERSHEY_PLAIN, title,
                    player_color(1), thick, cv2.LINE_8)
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
            cv2.rectangle(canvas, (x - 6, h - 18 - kh), (x + kw + 6, h - 6), DARK, -1)
            cv2.putText(canvas, KEYS, (x, h - 12), cv2.FONT_HERSHEY_PLAIN, scale, WHITE, thick, cv2.LINE_8)
        return bar

    def _draw_game_over(self, canvas: np.ndarray, game: Game) -> None:
        """renderer.py's end panel: the frame dimmed, a dark panel in the
        middle framed in the winner's colour, "PLAYER N WINS!" and the keys.
        Added (game start): each player's final health in their colour, and
        "HANDS UP TO PLAY AGAIN" with a bar that fills while everyone holds
        their hands up (game.hold_fraction())."""
        h, w = canvas.shape[:2]
        canvas[:] = (canvas * self.cfg["game_over_dim"]).astype(np.uint8)
        k = h / 480.0
        panel_w, panel_h = min(w - 24, int(380 * k)), int(150 * k)
        left, top = (w - panel_w) // 2, h // 2 - panel_h // 2
        winner = game.winner_id
        color = player_color(winner) if winner is not None else WHITE
        corners = (left, top), (left + panel_w, top + panel_h)
        cv2.rectangle(canvas, *corners, DARK, -1)
        cv2.rectangle(canvas, *corners, color, max(2, int(3 * k)), cv2.LINE_8)
        x, thin, inner = left + int(28 * k), max(1, int(k)), panel_w - int(56 * k)
        title = f"PLAYER {winner} WINS!" if winner is not None else "ROUND OVER"
        cv2.putText(canvas, title, (x, top + int(38 * k)), cv2.FONT_HERSHEY_PLAIN, 2 * k, color,
                    max(1, int(2 * k)), cv2.LINE_8)
        hx = x
        for pid in sorted(game.players):
            label = f"P{pid} {game.players[pid].health} HP"
            cv2.putText(canvas, label, (hx, top + int(64 * k)), cv2.FONT_HERSHEY_PLAIN, 1.2 * k,
                        player_color(pid), thin, cv2.LINE_8)
            hx += cv2.getTextSize(label + "    ", cv2.FONT_HERSHEY_PLAIN, 1.2 * k, thin)[0][0]
        again = PHASE_TEXT["result"]
        cv2.putText(canvas, again, (x, top + int(92 * k)), cv2.FONT_HERSHEY_PLAIN,
                    fit_scale(again, 1.2 * k, thin, inner), WHITE, thin, cv2.LINE_8)
        bar_top, bar_h = top + int(100 * k), max(4, int(6 * k))
        cv2.rectangle(canvas, (x, bar_top), (x + inner, bar_top + bar_h), WHITE, 1, cv2.LINE_8)
        fill = int((inner - 2) * game.hold_fraction())
        if fill > 0:
            cv2.rectangle(canvas, (x + 1, bar_top + 1), (x + fill, bar_top + bar_h - 1), WHITE, -1)
        cv2.putText(canvas, KEYS, (x, top + int(134 * k)), cv2.FONT_HERSHEY_PLAIN,
                    fit_scale(KEYS, k, thin, inner), WHITE, thin, cv2.LINE_8)

    # ---------------------------------------------------------- phase screens

    def start_place(self, pid: int, n: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Where pid should stand while waiting (x1, y1, x2, y2 px): a box
        start_place_w wide in their part of the frame (P1 the left half, P2
        the right; one player: the middle), from start_place_y[0] to [1] of
        the height."""
        h, w = shape[:2]
        half = self.cfg["start_place_w"] * w / 2
        cx = (pid - 0.5) / n * w
        y0, y1 = self.cfg["start_place_y"]
        return int(cx - half), int(y0 * h), int(cx + half), int(y1 * h)

    def _draw_start_place(self, canvas: np.ndarray, pid: int, ready: bool, n: int) -> None:
        """Waiting: start place pid in that team's colour, dashed with
        "PLAYER N / STAND HERE" until someone with a face stands in its half
        of the frame (game.place_taken()), then solid with a tick and "READY".
        The box is a guide; the whole half counts."""
        x1, y1, x2, y2 = self.start_place(pid, n, canvas.shape)
        color = player_color(pid)
        thick = max(2, int(canvas.shape[0] / 240))
        if ready:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick, cv2.LINE_8)
            draw_tick(canvas, ((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1) * 0.3, color, 3 * thick)
        else:
            draw_bbox(canvas, np.array([x1, y1, x2, y2]), color, dashed=True)
        scale = self._scale(canvas, "start_text_scale")
        tthick = max(1, int(round(scale)))
        for text, y, c in ((f"PLAYER {pid}", y1, color), ("READY" if ready else "STAND HERE", y2, WHITE)):
            sc = fit_scale(text, scale, tthick, x2 - x1 - 12)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, sc, tthick)
            tx, ty = (x1 + x2 - tw) // 2, y + th + 10 if y == y1 else y - 10
            cv2.rectangle(canvas, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), DARK, -1)
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_PLAIN, sc, c, tthick, cv2.LINE_8)

    def instruction_box(self, pid: int, n: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Where pid's instruction panel goes (x1, y1, x2, y2 px): at the
        bottom of their part of the frame (as start_place()), over the legs,
        so the face, arms and face effects stay visible."""
        h, w = shape[:2]
        half = self.cfg["instruction_w"] * w / 2
        cx = (pid - 0.5) / n * w
        y0, y1 = self.cfg["instruction_y"]
        return int(cx - half), int(y0 * h), int(cx + half), int(y1 * h)

    def _draw_instructions(self, canvas: np.ndarray, pid: int, game: Game, t: float) -> None:
        """Calibration: pid's panel. A stick figure of their current move
        (scene/instructions.py; mirrored for P2, so the punch points at the
        opponent), its name in the team colour, how to do it and its effect,
        and one dot per move (filled = done, white ring = now, grey = to do).
        Just after a move is done, "NICE!" with a tick for nice_s; when all
        are done, "DONE!". A move not done within remind_s: the frame
        flashes and "TRY IT NOW!" appears. The neutral pose has a bar that
        fills while it is held (game.neutral_fraction()); a player whose arm
        measured short for the punch rule (game.short_arm()) is told to
        stretch it fully on the punch panel. With two players, the neutral
        pose also asks them to stand level; while it waits for that
        (game.line_up_hint()), the nearer player is told to step back and
        the other to wait, instead of that line and the reminder."""
        x1, y1, x2, y2 = self.instruction_box(pid, len(game.round_players), canvas.shape)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, canvas.shape[1]), min(y2, canvas.shape[0])
        if x2 - x1 < 20 or y2 - y1 < 20:
            return
        color = player_color(pid)
        player = game.players.get(pid)
        done = player.move if player is not None else 0
        since = t - player.move_t if player is not None else 0.0
        move = game.current_move(pid)
        nice = done > 0 and since < self.cfg["nice_s"]
        remind = move is not None and not nice and since > self.cfg["remind_s"]
        flash = remind and int(since * 2 * self.cfg["remind_hz"]) % 2 == 0

        roi = canvas[y1:y2, x1:x2]
        a = self.cfg["instruction_alpha"]
        roi[:] = (roi * (1 - a) + np.array(DARK) * a).astype(np.uint8)
        k = canvas.shape[0] / 480.0
        border = max(2, int(2 * k))
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), WHITE if flash else color,
                      2 * border if flash else border, cv2.LINE_8)

        pad = int(8 * k)
        fig = y2 - y1 - 2 * pad                      # the figure's square
        fig_center = (x1 + pad + fig / 2, y1 + pad + fig / 2)
        hint = None
        if nice:
            last = CALIBRATION_MOVES[done - 1]
            title, lines = "NICE!", ["MEASURED" if last == "neutral" else MOVE_TEXT[last][0] + " LEARNED"]
            draw_figure(canvas, CALIBRATION_MOVES[done - 1], fig_center, fig, GREY, mirror=pid == 2)
            draw_tick(canvas, fig_center, fig * 0.6, color, max(3, int(4 * k)))
        elif move is None:
            wait = "WAIT FOR THE OTHER PLAYER" if len(game.round_players) > 1 else "GET READY"
            title, lines = "DONE!", [wait]
            draw_tick(canvas, fig_center, fig * 0.6, color, max(3, int(4 * k)))
        else:
            title, how, effect = MOVE_TEXT[move]
            # With two players the neutral pose asks them to stand level, in
            # place of "= MEASURING YOU" (the bar shows that), so the panel
            # still fits a reminder.
            level = move == "neutral" and len(game.round_players) > 1
            lines = [how] if level else [how, "= " + effect]
            if move == "punch" and game.short_arm(pid):
                lines.append("STRETCH YOUR ARM ALL THE WAY")
            hint = LINE_UP_TEXT.get(game.line_up_hint(pid))
            if hint:
                lines.append(hint)        # more useful than a reminder
            else:
                if level:
                    lines.append(STAND_LEVEL_TEXT)
                if remind:
                    lines.append("TRY IT NOW!")
            draw_figure(canvas, move, fig_center, fig, color, mirror=pid == 2)

        tx, text_w = x1 + 2 * pad + fig, x2 - x1 - 3 * pad - fig
        thin = max(1, int(round(k)))
        scale = fit_scale(title, self._scale(canvas, "instruction_title_scale"), 2 * thin, text_w)
        (_, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_PLAIN, scale, 2 * thin)
        y = y1 + pad + th
        cv2.putText(canvas, title, (tx, y), cv2.FONT_HERSHEY_PLAIN, scale, color, 2 * thin, cv2.LINE_8)
        for line in lines:
            sc = fit_scale(line, self._scale(canvas, "instruction_text_scale"), thin, text_w)
            (_, lh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, sc, thin)
            y += int(lh * 1.9)
            if line == "TRY IT NOW!":
                c = WHITE if flash else color
            else:
                c = color if line == hint else WHITE
            cv2.putText(canvas, line, (tx, y), cv2.FONT_HERSHEY_PLAIN, sc, c, thin, cv2.LINE_8)

        r = max(4, int(6 * k))                       # progress dots along the bottom
        if move == "neutral" and not nice:
            # Under the last line of text, above the dots.
            bar_h = max(4, int(6 * k))
            top = min(y + int(10 * k), y2 - pad - 2 * r - int(6 * k) - bar_h)
            cv2.rectangle(canvas, (tx, top), (tx + text_w, top + bar_h), WHITE, 1, cv2.LINE_8)
            fill = int((text_w - 2) * game.neutral_fraction(pid))
            if fill > 0:
                cv2.rectangle(canvas, (tx + 1, top + 1), (tx + fill, top + bar_h - 1), color, -1)
        for i in range(len(CALIBRATION_MOVES)):
            c = (tx + r + i * 3 * r, y2 - pad - r)
            if i < done:
                cv2.circle(canvas, c, r, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, c, r, WHITE if i == done else GREY, thin + (i == done), cv2.LINE_AA)

    def _draw_not_facing(self, canvas: np.ndarray, state: FrameState, game: Game) -> None:
        """"FACE THE CAMERA" above a player's head for face_camera_s after a
        gesture did not count because they were turned away
        (Game.facing_camera()), in their team colour with a dark outline:
        at the face's above_head anchor, or above the body box without a
        face."""
        t = state.frame.t
        for pid, when in game.not_facing_t.items():
            p = state.players.get(pid)
            if t - when > self.cfg["face_camera_s"] or p is None:
                continue
            face = state.faces.get(pid)
            if face is not None and face.valid and "above_head" in face.anchors:
                cx, cy = face.anchors["above_head"]
            else:
                cx, cy = (p.bbox[0] + p.bbox[2]) / 2.0, p.bbox[1]
            text = "FACE THE CAMERA"
            scale = self._scale(canvas, "face_camera_text_scale")
            thick = max(1, int(round(canvas.shape[0] / 480.0)))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thick)
            x = int(np.clip(cx - tw / 2, 4, canvas.shape[1] - tw - 4))
            y = int(np.clip(cy, th + 4, canvas.shape[0] - 4))
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, DARK, thick + 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, player_color(pid), thick,
                        cv2.LINE_AA)

    def _hint_shown(self, pid: int, t: float, game: Game) -> bool:
        """"FACE THE CAMERA" is above this player's head now (_draw_not_facing())."""
        when = game.not_facing_t.get(pid)
        return (game.phase in ("calibration", "playing") and when is not None
                and t - when <= self.cfg["face_camera_s"])

    # ---------------------------------------------- status above the head

    def crowned(self, game: Game) -> int | None:
        """The round player with the most HP, or None at a draw or with
        fewer than two players."""
        hp = {pid: game.players[pid].health for pid in game.round_players if pid in game.players}
        if len(hp) < 2:
            return None
        best = max(hp.values())
        top = [pid for pid, h in hp.items() if h == best]
        return top[0] if len(top) == 1 else None

    def _draw_status(self, canvas: np.ndarray, state: FrameState, game: Game) -> None:
        """Above each visible face (docs/merge_log.md, "Face effects"): the
        name tag "P1" in the team colour at above_head, a crown above it for the player with more HP
        (_draw_tag()), and dizzy stars around the head below dizzy_hp.
        "FACE THE CAMERA" takes the tag's and crown's place while shown.
        Health only counts from the countdown on. All of it fades with
        Face.alpha."""
        t = state.frame.t
        fight = game.phase in ("countdown", "playing", "result")
        crown = self.crowned(game) if fight else None
        for pid, f in sorted(state.faces.items()):
            if not f.valid or f.alpha <= 0 or "above_head" not in f.anchors:
                continue
            # Fading: draw on a copy and blend it in (only during a face's short fade-out)
            layer = canvas if f.alpha >= 1 else canvas.copy()
            player = game.players.get(pid)
            if fight and player is not None and player.health < self.cfg["dizzy_hp"] * game.max_health:
                self._draw_stars(layer, f, t)
            if not self._hint_shown(pid, t, game):
                self._draw_tag(layer, pid, f, crowned=pid == crown, t=t)
            if layer is not canvas:
                cv2.addWeighted(layer, f.alpha, canvas, 1.0 - f.alpha, 0.0, dst=canvas)

    def _draw_tag(self, canvas: np.ndarray, pid: int, f: Face, crowned: bool, t: float) -> None:
        """"P<pid>", tag_face_h face widths tall (at least tag_min_h of the
        frame height), with the crown above it when `crowned`. Upright, so
        it stays readable when the head tilts.

        Centred on above_head with its baseline there; when that would put
        the tag or crown under the top bar (a head near the top of the
        frame), it goes beside the hat instead, tag_side_fw face widths
        from it on the side towards the middle of the frame.

        Kept steady per player (_Tag): the size changes only when the face
        width has changed by more than tag_size_tol; the tag goes back
        above the head only once there is tag_side_hyst tag heights of room
        to spare, so a head near the limit does not flip it every frame;
        beside the hat, it changes sides only once the player is
        tag_side_flip frame widths past the middle; and its position has
        its own One Euro filter, smoother than the face's, because
        above_head is far from the face and shakes most."""
        cfg = self.cfg
        h, w = canvas.shape[:2]
        st = self._tags.get(pid)
        if st is None:
            st = self._tags[pid] = _Tag(OneEuro(cfg["tag_filter_min_cutoff"], cfg["tag_filter_beta"]))
        text_h = max(cfg["tag_face_h"] * f.scale, cfg["tag_min_h"] * h)
        if st.text_h <= 0 or abs(text_h / st.text_h - 1.0) > cfg["tag_size_tol"]:
            st.text_h = text_h
        scale = st.text_h / 10.0                            # FONT_HERSHEY_PLAIN is ~10 px tall at scale 1
        thick = max(1, int(round(st.text_h / 12.0)))
        text = f"P{pid}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thick)
        crown_h = (0.6 * cfg["crown_per_tag"] + 0.25) * th if crowned else 0.0
        top_limit = cfg["hud_h"] * h + 4
        cx, cy = f.anchors["above_head"]
        hx, hy = f.anchors.get("hat", (cx, cy))
        room = cy - th - crown_h - top_limit                # px to spare above the head
        if st.side == 0 and room < 0:
            st.side = 1 if hx < w / 2 else -1
        elif st.side != 0 and room > cfg["tag_side_hyst"] * th:
            st.side = 0
        elif (hx - w / 2) * st.side > cfg["tag_side_flip"] * w:
            st.side = -st.side                              # walked past the middle: other side
        if st.side != 0:
            cx = hx + st.side * (cfg["tag_side_fw"] * f.scale + tw / 2)
            cy = max(hy, top_limit + th + crown_h)
        cx, cy = st.filter(np.array([cx, cy]), t)
        x = int(np.clip(cx - tw / 2, 4, w - tw - 4))
        y = int(np.clip(cy, th + 4, h - 4))
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, DARK, thick + 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, player_color(pid), thick,
                    cv2.LINE_AA)
        if crowned:
            self._draw_crown(canvas, x + tw / 2, float(y - th), float(th))

    def _draw_crown(self, canvas: np.ndarray, cx: float, top: float, tag_h: float) -> None:
        """A gold crown with three points just above the name tag, sized by the tag."""
        w = self.cfg["crown_per_tag"] * tag_h
        h = 0.6 * w
        b = top - 0.25 * tag_h                              # small gap above the tag
        pts = np.array([[cx - w / 2, b], [cx + w / 2, b], [cx + w / 2, b - h], [cx + w / 4, b - 0.45 * h],
                        [cx, b - h], [cx - w / 4, b - 0.45 * h], [cx - w / 2, b - h]])
        poly = [np.round(pts).astype(np.int32)]
        cv2.fillPoly(canvas, poly, GOLD, cv2.LINE_AA)
        cv2.polylines(canvas, poly, True, DARK, max(1, int(round(tag_h / 12.0))), cv2.LINE_AA)

    def _draw_stars(self, canvas: np.ndarray, f: Face, t: float) -> None:
        """stars_n stars circling the head at stars_hz turns per second, on
        an ellipse round the hat (stars_rise face widths above the brim,
        stars_orbit face widths wide each side, flattened to stars_flat),
        turned with the head's roll. Stars behind the head are drawn first
        and smaller, so the ring reads as going round the head."""
        cfg = self.cfg
        u = np.array([np.cos(f.roll), np.sin(f.roll)])     # along the eye line
        down = np.array([-u[1], u[0]])
        centre = f.anchors.get("hat", f.anchors["above_head"]) - cfg["stars_rise"] * f.scale * down
        rx = cfg["stars_orbit"] * f.scale
        ry = cfg["stars_flat"] * rx
        r = max(cfg["star_size"] * f.scale, 3.0)
        n = int(cfg["stars_n"])
        angles = [2 * np.pi * (cfg["stars_hz"] * t + k / n) for k in range(n)]
        thick = max(1, int(round(r / 6.0)))
        for a in sorted(angles, key=np.sin):              # sin < 0: behind the head, drawn first
            p = centre + rx * np.cos(a) * u + ry * np.sin(a) * down
            size = r * (0.8 + 0.2 * np.sin(a))
            poly = [np.round(star(p, size, a)).astype(np.int32)]
            cv2.fillPoly(canvas, poly, STAR, cv2.LINE_AA)
            cv2.polylines(canvas, poly, True, DARK, thick, cv2.LINE_AA)

    def _draw_big_text(self, canvas: np.ndarray, text: str, size: float) -> None:
        """Text in the middle of the frame, `size` x countdown_h of its height
        tall, off-white with a dark outline so it reads on any background."""
        h, w = canvas.shape[:2]
        scale = size * self.cfg["countdown_h"] * h / 10.0   # FONT_HERSHEY_PLAIN is ~10 px tall at scale 1
        thick = max(2, int(scale))
        scale = fit_scale(text, scale, thick, w - 24)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thick)
        org = ((w - tw) // 2, (h + th) // 2)
        outline = thick + max(4, thick)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_PLAIN, scale, DARK, outline, cv2.LINE_AA)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_PLAIN, scale, WHITE, thick, cv2.LINE_AA)

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
        cv2.rectangle(canvas, (left - 2, top - label_h - 2), (right + 2, top + bar_h + 2), DARK, -1)
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


def status_text(phase: str, has_player: bool) -> str:
    """The top bar's status: what the phase expects; while playing, the
    controls, or "STEP INTO VIEW" while nobody is seen."""
    if phase == "playing":
        return CONTROLS if has_player else "STEP INTO VIEW"
    return PHASE_TEXT[phase]


def star(centre, radius: float, angle: float) -> np.ndarray:
    """A five-pointed star (10 points, px): outer radius `radius`, inner
    0.45 of it, turned by `angle` (rad)."""
    a = angle + np.arange(10) * np.pi / 5 - np.pi / 2
    r = np.where(np.arange(10) % 2 == 0, radius, 0.45 * radius)
    return np.asarray(centre, float) + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


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
