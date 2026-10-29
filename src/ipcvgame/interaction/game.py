"""Task 4 — game logic: Colour Clash.

Moved from the Colour Clash ``game.py`` in the game merge (docs/merge_log.md,
Task 4). Positions are pixels in the mirrored frame; every number comes from
the ``game:`` config.

Combat (``advance()``): bolts and beams fly across the frame and damage the
first other player whose body box they reach, adding a paint splash on the
torso (``PaintSplat``, in torso coordinates, so the paint moves with the
body; docs/merge_log.md, "Normal attack"); a raised shield blocks them.
At 0 health the attacker wins and the game freezes.
Every hit that does damage is also kept in ``Game.hits`` for that frame,
for the face effects' hit reaction (docs/merge_log.md, "Face effects").

Drinks (consumables, docs/merge_log.md, "Consumables"): while playing,
drinks fall from the top of the frame, one every drink_every_s, at most drink_max
at once. Each has a buff category (hp, attack, defense) and a random size;
the bigger the drink, the bigger its buff value. A player drinks one only
with the mouth (_drink_with_mouths()): a drink that touches the open mouth
sticks to it, and closing the mouth drinks it (Game.drunk, for this frame).
Its buff (_apply_drink()): hp heals at once; attack multiplies the damage
the player deals and defense cuts the damage they take, each for
drink_buff_s. A second drink of the same kind replaces the value and
restarts the time.

``update(state)`` is the game part of the ``main.py`` loop: it reads each player's
body box and position from the frame state, turns gesture events into actions
(punch -> bolt, hands up -> beam, arms crossed -> shield), then advances the
game by the real time since the last frame.

Phases (game start, docs/merge_log.md), in ``Game.phase``:

- ``waiting``: until both start places are taken (``ready_s``): someone in
  view with a face stands in each half of the frame (``place_taken()``).
- ``calibration``: each player performs the moves of ``CALIBRATION_MOVES``
  in turn; a move's gesture event ticks it off instead of firing. The first
  move is the neutral pose (``_neutral_pose()``), held for ``neutral_s``
  while the shoulder width and arm length are measured (pose/measure.py);
  the medians are kept in ``GamePlayer`` until ``reset()``.
- ``countdown``: ``countdown_s``, then a fresh round.
- ``playing``: the only phase with combat (actions and ``advance()``).
- ``result``: after a win; both players holding their hands up for
  ``replay_hold_s`` start the next round from the countdown, keeping their
  calibration. ``reset()`` (key R) goes back to ``waiting``.

With ``intro: false`` there is no waiting, calibration or countdown: the game
plays at once, as before (tests, clips recorded without the intro, e.g.
``--set game.intro=false``). With ``round_players: 1`` only P1 is needed, to
test the intro alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.geometry import bilinear
from ipcvgame.core.types import FrameState, Motion
from ipcvgame.interaction.gestures import detect_hands_up, lower_body_y, shoulder_width
from ipcvgame.pose.measure import arm_px, shoulder_px, torso_px
from ipcvgame.scene.draw import DRINK_COLORS, draw_bbox, draw_lines, draw_skeleton, player_color, pt

PHASES = ("waiting", "calibration", "countdown", "playing", "result")
# Buff categories of the drinks; their values are drink_<kind> in the config.
DRINK_KINDS = ("hp", "attack", "defense")
# The moves taught in the calibration phase, in this order: the neutral pose
# (measured by the game), then the gesture event names.
CALIBRATION_MOVES = ("neutral", "punch", "hands_up", "arms_crossed")
# Keypoints the neutral pose needs: the whole upper body and the hips.
NEUTRAL_KPTS = [sk.L_SHOULDER, sk.R_SHOULDER, sk.L_ELBOW, sk.R_ELBOW, sk.L_WRIST, sk.R_WRIST,
                sk.L_HIP, sk.R_HIP]


@dataclass
class _Hold:
    """How long a condition has held (s). Gaps up to gap_s where it fails do
    not restart it, so a keypoint dropout of a frame or two is ignored."""
    since: float | None = None
    last: float = -np.inf

    def update(self, on: bool, t: float, gap_s: float) -> float:
        if on:
            if self.since is None or t - self.last > gap_s:
                self.since = t
            self.last = t
        elif self.since is not None and t - self.last > gap_s:
            self.since = None
        return 0.0 if self.since is None else self.last - self.since


@dataclass
class PaintSplat:
    """One splash of paint from one hit. (u, v) is a point on the torso
    (GamePlayer.torso): u from the screen-left shoulder (0) to the
    screen-right one (1) when facing the camera, v from the shoulder line (0)
    to the hips (1). The renderer maps it back onto the body every frame
    (core.geometry.bilinear()), so the paint moves, leans and scales with
    the player."""
    u: float
    v: float
    owner_id: int       # the attacker; the colour comes from scene/draw.py
    direction: int = 1  # the way the attack flew: +1 to the right, -1 to the left
    seed: int = 0       # the splash's shape (renderer): the same in every frame
    t: float = 0.0      # Frame.t of the hit


@dataclass
class GamePlayer:
    """One player's game state (``Player`` in game.py; renamed because
    ``Player`` is the identity type in core/types.py)."""
    health: int
    cooldown: float = 0.0
    x: float = 0.0                       # px, shoulder midpoint
    y: float = 0.0                       # px, hip level
    bounds: np.ndarray | None = None     # body box x1, y1, x2, y2 in px
    # The torso this frame (torso_corners()), (4, 2) px; None when unknown.
    torso: np.ndarray | None = None
    paint_splats: list[PaintSplat] = field(default_factory=list)
    # Calibration: index of the current move in CALIBRATION_MOVES (all done
    # at len), and the Frame.t it was shown at (for a reminder).
    move: int = 0
    move_t: float = 0.0
    # Pose calibration: median shoulder width, arm length and torso length
    # (px) in the neutral pose; None until measured. Kept for new rounds.
    shoulder_px: float | None = None
    arm_px: float | None = None
    torso_px: float | None = None
    # Drink buffs (consumables): damage dealt times `attack`, damage taken
    # times (1 - `defense`), each for the seconds left; 1 and 0 without one.
    attack: float = 1.0
    attack_left: float = 0.0
    defense: float = 0.0
    defense_left: float = 0.0


@dataclass
class Projectile:
    x: float
    y: float
    direction: int    # +1 = to the right, -1 = to the left
    owner_id: int
    age: float = 0.0  # s since it was fired (drives the fire animation)


@dataclass
class BoltEnd:
    """Where a bolt (the paint blob) ended: drawing only, kept for
    bolt_end_s, so the renderer can play its burst on a shield there."""
    x: float
    y: float
    direction: int
    owner_id: int
    age: float = 0.0
    reason: str = "edge"   # "hit" (a body), "shield" or "edge" (left the frame)


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
class Hit:
    """One hit that did damage this frame, for the face effects."""
    target_id: int
    owner_id: int     # the attacker
    direction: int    # the way the attack travelled: +1 = to the right, -1 = to the left
    damage: int


@dataclass
class Drink:
    """A falling drink (consumables). The renderer picks the sprite."""
    x: float          # centre, px
    y: float          # centre, px
    kind: str         # "hp" | "attack" | "defense"
    size: float       # size factor (drink_size), times drink_h of the frame height
    value: float      # buff value, from the size (drink_<kind>)
    variant: float    # 0-1: which of the category's drinks is drawn
    held_by: int | None = None   # the player whose mouth holds it; it does not fall then
    held_t: float = 0.0          # Frame.t it was caught
    dropped_by: int | None = None   # held too long by this player: they cannot catch it again


@dataclass
class Drunk:
    """A drink a player drank this frame."""
    pid: int
    kind: str         # "hp" | "attack" | "defense"
    value: float


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
        # Actions this frame as (pid, "bolt" | "beam" | "shield" | "hit"), and
        # (pid, "learned_<move>") for a move ticked off in calibration; for the CSV.
        self.actions: list[tuple[int, str]] = []
        # Hits that did damage this frame (cleared by update()), for the face
        # effects; each is also logged as the target's action "hit".
        self.hits: list[Hit] = []
        # Drinks drunk this frame (cleared by update()), for the buffs and the
        # face effects' cheek bulge; each is also the action drink_<kind>.
        self.drunk: list[Drunk] = []
        # Each player's mouth, open or closed (hysteresis on jawOpen).
        self.mouth_open: dict[int, bool] = {}
        self.intro = bool(cfg["intro"])
        self.phase = "waiting" if self.intro else "playing"
        self.phase_t: float | None = None   # Frame.t the phase started at; None = at the next update
        self.countdown_s = float(cfg["countdown_s"])
        # The players a round needs: P1 and P2, or P1 alone (round_players: 1).
        self.round_players = tuple(range(1, int(cfg["round_players"]) + 1))
        self._hold = _Hold()                 # the condition that ends this phase
        self._held = 0.0                     # how long it has held (s), for hold_fraction()
        # Neutral pose per player: how long it has held, and this hold's
        # (shoulder px, arm px) samples, one per frame.
        self._neutral: dict[int, _Hold] = {}
        self._neutral_held: dict[int, float] = {}
        self._samples: dict[int, list[tuple[float, float, float]]] = {}
        # Frame.t of each player's last gesture that did not count because
        # they were not facing the camera (facing_camera()), for the hint.
        self.not_facing_t: dict[int, float] = {}
        # Each player's body turn (degrees) the last time their arms were
        # down (_track_body_yaw()); gestures move the shoulders.
        self.body_yaw: dict[int, float] = {}
        # Line-up this frame (calibration, _check_line_up()): the round
        # players stand level; else the one nearer the camera (None when
        # someone is not in view) and the depth gap (m; None when unknown).
        self.level = True
        self.nearer: int | None = None
        self.line_gap: float | None = None
        # Phase lengths and the beam's hit band, for drawing (the renderer
        # reads them here instead of from the game: config).
        self.beam_charge_s = float(cfg["beam_charge_s"])
        self.beam_life_s = self.beam_charge_s + float(cfg["beam_duration_s"])
        self.shield_fade_s = float(cfg["shield_fade_s"])
        self.bolt_end_s = float(cfg["bolt_end_s"])
        # Drinks (consumables): in the air now, a seedable random generator
        # (drink_seed; null = a different game every time) and the time to
        # the next one (s), counted down while playing.
        self.drinks: list[Drink] = []
        self.drink_buff_s = float(cfg["drink_buff_s"])   # for drawing the time left
        self._rng = random.Random(cfg["drink_seed"])
        self._drink_wait = self._rng.uniform(*cfg["drink_every_s"])

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
            self._tick_buffs(player, dt)

        for player_id, shield in list(self.shields.items()):
            shield.age += dt
            shield.remaining -= dt
            if shield.remaining <= -cfg["shield_fade_s"]:
                del self.shields[player_id]

        self._advance_drinks(dt)

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
                reason = "shield" if self.shielded(target) else "hit"
                if reason == "hit":
                    self._apply_damage(target, projectile.owner_id, projectile.x,
                                       projectile.y, cfg["bolt_damage"], projectile.direction)
            elif 0.0 <= projectile.x <= self.frame_w:
                remaining.append(projectile)
                continue
            else:
                reason = "edge"
            self.bolt_ends.append(BoltEnd(min(max(projectile.x, 0.0), float(self.frame_w)),
                                          projectile.y, projectile.direction, projectile.owner_id,
                                          reason=reason))
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

    def drink_height(self, drink: Drink) -> float:
        """A drink's height on screen (px)."""
        return self.cfg["drink_h"] * self.frame_h * drink.size

    def _spawn_drink(self) -> Drink:
        """A new drink just above the top of the frame, at a random x in the
        play area, of a random category and size. Its value goes linearly
        from the category's smallest (smallest size) to largest value."""
        cfg, rng = self.cfg, self._rng
        lo, hi = cfg["drink_size"]
        size = rng.uniform(lo, hi)
        kind = rng.choice(DRINK_KINDS)
        v0, v1 = cfg[f"drink_{kind}"]
        value = v0 + (v1 - v0) * (size - lo) / (hi - lo)
        margin = cfg["drink_margin_w"]
        drink = Drink(rng.uniform(margin, 1.0 - margin) * self.frame_w, 0.0, kind, size, value,
                      rng.random())
        drink.y = -self.drink_height(drink) / 2
        return drink

    def _advance_drinks(self, dt: float) -> None:
        """Spawn a drink when one is due (and fewer than drink_max are in the
        air; else as soon as there is room), let all fall at a constant
        speed, and drop those below the frame."""
        cfg = self.cfg
        self._drink_wait -= dt
        if self._drink_wait <= 0 and len(self.drinks) < cfg["drink_max"]:
            self.drinks.append(self._spawn_drink())
            self._drink_wait = self._rng.uniform(*cfg["drink_every_s"])
        speed = self.frame_h / cfg["drink_fall_s"]
        for drink in self.drinks:
            if drink.held_by is None:
                drink.y += speed * dt
        self.drinks = [d for d in self.drinks if d.y - self.drink_height(d) / 2 <= self.frame_h]

    def _apply_drink(self, drunk: Drunk) -> None:
        """A drink's buff: hp heals at once (rounded, up to max_health);
        attack and defense take the drink's value and last drink_buff_s,
        replacing a running buff of the same kind."""
        player = self.get_player(drunk.pid)
        if drunk.kind == "hp":
            player.health = min(self.max_health, player.health + int(round(drunk.value)))
        elif drunk.kind == "attack":
            player.attack, player.attack_left = drunk.value, self.cfg["drink_buff_s"]
        elif drunk.kind == "defense":
            player.defense, player.defense_left = drunk.value, self.cfg["drink_buff_s"]

    @staticmethod
    def _tick_buffs(player: GamePlayer, dt: float) -> None:
        """Count the buffs down; an ended buff goes back to no effect."""
        if player.attack_left > 0:
            player.attack_left -= dt
            if player.attack_left <= 0:
                player.attack, player.attack_left = 1.0, 0.0
        if player.defense_left > 0:
            player.defense_left -= dt
            if player.defense_left <= 0:
                player.defense, player.defense_left = 0.0, 0.0

    def _track_mouth(self, state: FrameState, pid: int) -> None:
        """Mouth open or closed from MediaPipe's jawOpen score: open above
        drink_mouth_open, closed again below drink_mouth_closed (hysteresis,
        so a score near one threshold does not flip it every frame). No
        score this frame (MediaPipe runs for each player in turn): unchanged.
        No visible face: closed."""
        f = state.faces.get(pid)
        if f is None or not f.valid or "mouth" not in f.anchors:
            self.mouth_open[pid] = False
            return
        self.mouth_open.setdefault(pid, False)          # a new face starts closed
        jaw = f.blendshapes.get("jawOpen")
        if jaw is None:
            return
        if jaw > self.cfg["drink_mouth_open"]:
            self.mouth_open[pid] = True
        elif jaw < self.cfg["drink_mouth_closed"]:
            self.mouth_open[pid] = False

    def _drink_with_mouths(self, state: FrameState, t: float) -> None:
        """Drinking, once per frame while playing (only with the mouth):

        - A held drink follows its player's mouth: its rim (drink_hold_rim of
          its height above the centre) at the mouth anchor. Closing the
          mouth drinks it; held longer than drink_hold_s, or the face lost,
          it falls on again, and that player cannot catch it again.
        - A free drink that touches an open mouth sticks to it: its centre
          within drink_catch_fw face widths of the mouth anchor plus half
          its height. Each player holds one drink at a time; the nearer
          mouth wins.
        """
        cfg = self.cfg
        for d in list(self.drinks):
            if d.held_by is None:
                continue
            f = state.faces.get(d.held_by)
            if f is None or not f.valid or "mouth" not in f.anchors:
                d.held_by, d.dropped_by = None, d.held_by       # face lost: not a closed mouth
                continue
            if not self.mouth_open.get(d.held_by, False):       # before the time limit: a mouth
                self.drinks.remove(d)                           # closing as time runs out drinks
                self.drunk.append(Drunk(d.held_by, d.kind, d.value))
                self._apply_drink(self.drunk[-1])
                self.actions.append((d.held_by, f"drink_{d.kind}"))
                continue
            if t - d.held_t > cfg["drink_hold_s"]:
                d.held_by, d.dropped_by = None, d.held_by
                continue
            mx, my = f.anchors["mouth"]
            d.x, d.y = float(mx), float(my) + cfg["drink_hold_rim"] * self.drink_height(d)
        holding = {d.held_by for d in self.drinks if d.held_by is not None}
        for d in self.drinks:
            if d.held_by is not None:
                continue
            best, best_dist = None, np.inf
            for pid, open_ in self.mouth_open.items():
                f = state.faces.get(pid)
                if not open_ or pid in holding or pid == d.dropped_by or f is None or not f.valid:
                    continue
                dist = float(np.hypot(d.x - f.anchors["mouth"][0], d.y - f.anchors["mouth"][1]))
                if dist <= cfg["drink_catch_fw"] * f.scale + self.drink_height(d) / 2 and dist < best_dist:
                    best, best_dist = pid, dist
            if best is not None:
                d.held_by, d.held_t = best, t
                holding.add(best)
                self.actions.append((best, "catch"))

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
                                       self.cfg["beam_damage"], beam.direction)
                beam.hit_player_ids.add(player_id)

    def _apply_damage(self, target_id: int, owner_id: int, impact_x: float,
                      impact_y: float, damage: int, direction: int) -> None:
        target = self.get_player(target_id)
        if target.health <= 0 or self.winner_id is not None:
            return

        # Drink buffs: the attacker's attack and the target's defense.
        owner = self.players.get(owner_id)
        damage = int(round(damage * (owner.attack if owner else 1.0) * (1.0 - target.defense)))
        target.health = max(0, target.health - damage)
        self.hits.append(Hit(target_id, owner_id, direction, damage))
        self.actions.append((target_id, "hit"))
        target.paint_splats.append(self._splash(target_id, owner_id, impact_y, direction))

        if target.health == 0:
            self.winner_id = owner_id

    def _splash(self, target_id: int, owner_id: int, impact_y: float,
                direction: int) -> PaintSplat:
        """The splash a hit leaves on the target's torso. Attacks fly
        sideways, so they enter the body box at its near edge, often at an
        outstretched arm: u starts at the torso's side facing the attacker
        (splat_band_u) and steps inward by a random 0-splat_inward_u. v is
        the hit height between the shoulder line and the hips, kept in
        splat_band_v. Without a torso this frame: halfway down, and the
        screen-left shoulder taken as u = 0. Seeded, so the same hit always
        gives the same splash."""
        cfg, target = self.cfg, self.get_player(target_id)
        rng = random.Random(target_id * 1000 + len(target.paint_splats))
        (u_lo, u_hi), (v_lo, v_hi) = cfg["splat_band_u"], cfg["splat_band_v"]
        corners = target.torso
        if corners is None:
            v, u0_on_left = 0.5, True
        else:
            shoulder_y = (corners[0][1] + corners[1][1]) / 2.0
            hip_y = (corners[2][1] + corners[3][1]) / 2.0
            v = (impact_y - shoulder_y) / max(hip_y - shoulder_y, 1.0)
            u0_on_left = corners[0][0] <= corners[1][0]
        step = rng.uniform(0.0, cfg["splat_inward_u"])
        # Flying right, the attack comes in from the screen's left side.
        u = u_lo + step if (direction > 0) == u0_on_left else u_hi - step
        return PaintSplat(u, min(max(v, v_lo), v_hi), owner_id, 1 if direction >= 0 else -1,
                          rng.randrange(2**31), self._t or 0.0)

    def test_hit(self, pid: int = 1) -> bool:
        """Throw a test blob at pid from the far side of the frame (key b),
        in the other team's colour, so one player can see a hit and its paint
        alone. It is a normal bolt: it does damage and a shield blocks it.
        Only while playing, with pid in view; returns if it was thrown."""
        player = self.players.get(pid)
        if (self.phase != "playing" or self.winner_id is not None
                or player is None or player.bounds is None):
            return False
        left, top, right, bottom = player.bounds
        if player.torso is not None:
            y = float(bilinear(player.torso, 0.5, self.cfg["test_hit_v"])[1])
        else:
            y = (top + bottom) / 2.0
        # From the far side: a player on the left gets it from the right edge.
        direction = -1 if (left + right) / 2.0 < self.frame_w / 2.0 else 1
        x = float(self.frame_w) if direction < 0 else 0.0
        self.projectiles.append(Projectile(x, y, direction, 2 if pid == 1 else 1))
        self.actions.append((pid, "test_blob"))
        return True

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
            player.torso = self.torso_corners(state.motion[pid]) if pid in state.motion else None
            if pid in state.motion:
                pos = self.body_position(state.motion[pid])
                if pos is not None:
                    player.x, player.y = pos
                    positions[pid] = pos

        if self.phase_t is None:
            self.phase_t = t
        self.actions = []
        self.hits = []
        self.drunk = []
        for pid in visible:
            self._track_body_yaw(state, pid)
        for pid in state.players:
            self._track_mouth(state, pid)
        if self.phase == "playing":
            for e in state.events:
                if e.pid in positions and self._facing_or_blocked(state, e.pid, e.name, t):
                    self._act(e.pid, e.name, state.motion[e.pid], positions)
            if self.winner_id is None:
                self._drink_with_mouths(state, t)
            self.advance(dt, visible)
            if self.winner_id is not None:
                self._enter("result", t)
        elif self.phase == "calibration":
            for e in state.events:
                if e.name == self.current_move(e.pid) and self._facing_or_blocked(state, e.pid, e.name, t):
                    self._tick(e.pid, e.name, t)
            self._check_line_up(state)
            for pid in self.round_players:
                if self.current_move(pid) == "neutral":
                    self._measure(state, pid, t)
        self._next_phase(state, t)

    # ----------------------------------------------------------------- phases

    def _enter(self, phase: str, t: float | None) -> None:
        self.phase, self.phase_t, self._hold, self._held = phase, t, _Hold(), 0.0
        self.level, self.nearer, self.line_gap = True, None, None
        if phase == "calibration":
            for pid in self.round_players:
                player = self.get_player(pid)
                player.move, player.move_t = 0, t if t is not None else 0.0
            self._neutral.clear()
            self._neutral_held.clear()
            self._samples.clear()

    def _start_round(self, t: float | None) -> None:
        """A fresh round: from the countdown, or at once without the intro."""
        self._clear_round()
        self._enter("countdown" if self.intro else "playing", t)

    def _next_phase(self, state: FrameState, t: float) -> None:
        """Move to the next phase once this one's condition is met."""
        cfg = self.cfg
        if self.phase == "waiting":
            ready = all(self.place_taken(state, place) for place in self.round_players)
            self._held = self._hold.update(ready, t, cfg["hold_gap_s"])
            if self._held >= cfg["ready_s"]:
                self._enter("calibration", t)
        elif self.phase == "calibration":
            if all(self.current_move(pid) is None for pid in self.round_players):
                self._start_round(t)
        elif self.phase == "countdown":
            if self.phase_elapsed() >= self.countdown_s:
                self._enter("playing", t)
        elif self.phase == "result":
            up = all(self._hands_up(state, pid) for pid in self.round_players)
            self._held = self._hold.update(up, t, cfg["hold_gap_s"])
            if self._held >= cfg["replay_hold_s"]:
                self._start_round(t)

    def phase_elapsed(self) -> float:
        """Seconds since the current phase started (0 before the first update)."""
        if self._t is None or self.phase_t is None:
            return 0.0
        return self._t - self.phase_t

    def countdown_left(self) -> float:
        """Seconds left in the countdown (0 in the other phases)."""
        return max(0.0, self.countdown_s - self.phase_elapsed()) if self.phase == "countdown" else 0.0

    def hold_fraction(self) -> float:
        """How far the hold that ends this phase has got, 0-1: everyone in
        view (waiting), or everyone's hands up (result); 0 in the other phases."""
        target = {"waiting": self.cfg["ready_s"], "result": self.cfg["replay_hold_s"]}.get(self.phase)
        return 0.0 if target is None else min(1.0, self._held / target)

    def current_move(self, pid: int) -> str | None:
        """The calibration move pid has to perform next; None when all are done."""
        move = self.players[pid].move if pid in self.players else 0
        return CALIBRATION_MOVES[move] if move < len(CALIBRATION_MOVES) else None

    def _measure(self, state: FrameState, pid: int, t: float) -> None:
        """Calibration, neutral pose: one (shoulder, arm, torso) sample (px)
        per frame in the pose. Once the pose has held neutral_s (gaps up to
        hold_gap_s allowed), the medians are stored and the move is ticked
        off. A hold that starts again starts with no samples. The pose only
        counts while the players stand level (_check_line_up())."""
        hold = self._neutral.setdefault(pid, _Hold())
        on = self.level and self._neutral_pose(state, pid)
        held = self._neutral_held[pid] = hold.update(on, t, self.cfg["hold_gap_s"])
        if hold.since == t:
            self._samples[pid] = []
        if on:
            m = state.motion[pid]
            sample = (shoulder_px(m.kpts, m.valid), arm_px(m.kpts, m.valid), torso_px(m.kpts, m.valid))
            if None not in sample:
                self._samples.setdefault(pid, []).append(sample)
        samples = self._samples.get(pid)
        if held >= self.cfg["neutral_s"] and samples:
            player = self.get_player(pid)
            medians = (float(v) for v in np.median(samples, axis=0))
            player.shoulder_px, player.arm_px, player.torso_px = medians
            self._tick(pid, "neutral", t)

    def _neutral_pose(self, state: FrameState, pid: int) -> bool:
        """pid stands in the neutral pose this frame: in view, head turned at
        most neutral_max_yaw_deg (so the shoulders show their full width),
        upper body and hips seen, both wrists at least neutral_wrist_below_sw
        below the shoulder line (arms down) and slower than
        neutral_max_speed_sw (standing still)."""
        if not self.in_view(state, pid) or pid not in state.motion:
            return False
        cfg, m = self.cfg, state.motion[pid]
        if abs(math.degrees(state.faces[pid].yaw)) > cfg["neutral_max_yaw_deg"]:
            return False
        if not m.valid[NEUTRAL_KPTS].all():
            return False
        sw = shoulder_width(m.kpts, m.valid, cfg["min_shoulder_px"])
        if sw is None:
            return False
        shoulder_y = (m.kpts[sk.L_SHOULDER][1] + m.kpts[sk.R_SHOULDER][1]) / 2.0
        wrists = [sk.L_WRIST, sk.R_WRIST]
        low = (m.kpts[wrists, 1] > shoulder_y + cfg["neutral_wrist_below_sw"] * sw).all()
        still = (np.linalg.norm(m.vel[wrists], axis=1) < cfg["neutral_max_speed_sw"] * sw).all()
        return bool(low and still)

    def _check_line_up(self, state: FrameState) -> None:
        """Calibration: do the two round players stand level, side by side at
        the same distance from the camera? Then a later change in a
        player's depth starts from one line for both.

        - By depth (Player.pos, identity: from the shoulder width in px, the
          players face the camera in the neutral pose): level when less than
          line_up_depth_m apart. The smaller depth is the nearer player.
          Without a depth for both, nothing to check.
        - Someone not in view: not level, and no one to send back.
        - Hysteresis: once not level, the gap must shrink under
          line_up_hysteresis of the tolerance before the players are level
          again, so a gap near the tolerance does not flicker.
        With one round player there is no one to line up with."""
        was_level = self.level
        self.level, self.nearer, self.line_gap = True, None, None
        if len(self.round_players) < 2:
            return
        a, b = self.round_players[:2]
        if not (self.in_view(state, a) and self.in_view(state, b)):
            self.level = False
            return
        da, db = state.players[a].pos[1], state.players[b].pos[1]
        if da <= 0 or db <= 0:
            return
        k = 1.0 if was_level else self.cfg["line_up_hysteresis"]
        self.line_gap = abs(db - da)
        self.level = self.line_gap < k * self.cfg["line_up_depth_m"]
        if not self.level:
            self.nearer = a if da < db else b

    def line_up_hint(self, pid: int) -> str | None:
        """Calibration panel: "back" when pid has to step back to stand
        level, "wait" when pid's neutral pose waits for the line-up; None
        when the players are level or nobody needs the neutral pose any more."""
        if self.phase != "calibration" or self.level:
            return None
        if not any(self.current_move(p) == "neutral" for p in self.round_players):
            return None
        if pid == self.nearer:
            return "back"
        return "wait" if self.current_move(pid) == "neutral" else None

    def neutral_fraction(self, pid: int) -> float:
        """How far pid's neutral-pose hold has got, 0-1 (for the panel)."""
        return min(1.0, self._neutral_held.get(pid, 0.0) / self.cfg["neutral_s"])

    def arm_sw(self, pid: int) -> float | None:
        """pid's calibrated arm length in shoulder widths; None before calibration."""
        player = self.players.get(pid)
        if player is None or player.shoulder_px is None or player.arm_px is None:
            return None
        return player.arm_px / player.shoulder_px

    def short_arm(self, pid: int) -> bool:
        """pid's arm is short for the punch rule (under min_arm_sw)."""
        arm = self.arm_sw(pid)
        return arm is not None and arm < self.cfg["min_arm_sw"]

    def _tick(self, pid: int, gesture: str, t: float) -> None:
        """Calibration: the player's current move is ticked off by its event;
        other events are ignored, and nothing fires."""
        if pid in self.round_players and gesture == self.current_move(pid):
            player = self.get_player(pid)
            player.move, player.move_t = player.move + 1, t
            self.actions.append((pid, f"learned_{gesture}"))

    def body_yaw_deg(self, state: FrameState, pid: int) -> float | None:
        """How far pid's body is turned away from the camera this frame
        (degrees). Turning shrinks the shoulder width in the image by
        cos(angle) but not the torso length, so the shoulder-to-torso ratio
        against the one measured in the neutral pose (facing the camera)
        gives the angle, at any distance. Only meaningful with the arms
        down: crossing the arms rolls the shoulders in and raising them
        lifts the shoulders, both narrowing the shoulder width (on
        session.mp4, 22-38 degrees while facing the camera). None before
        the calibration, or without both shoulders and hips."""
        player, m = self.players.get(pid), state.motion.get(pid)
        if player is None or player.shoulder_px is None or not player.torso_px or m is None:
            return None
        sw, torso = shoulder_px(m.kpts, m.valid), torso_px(m.kpts, m.valid)
        if not sw or not torso:
            return None
        ratio = (sw / torso) / (player.shoulder_px / player.torso_px)
        return math.degrees(math.acos(min(1.0, ratio)))

    def _track_body_yaw(self, state: FrameState, pid: int) -> None:
        """Keep pid's body turn from frames with both wrists below the
        shoulder line by neutral_wrist_below_sw (arms down), where the
        shoulder width is not changed by a gesture. Between two actions the
        gesture rules make the player return to a normal position, so the
        value is from just before the gesture."""
        m = state.motion.get(pid)
        if m is None:
            return
        sw = shoulder_width(m.kpts, m.valid, self.cfg["min_shoulder_px"])
        wrists = [sk.L_WRIST, sk.R_WRIST]
        if sw is None or not m.valid[wrists].all():
            return
        shoulder_y = (m.kpts[sk.L_SHOULDER][1] + m.kpts[sk.R_SHOULDER][1]) / 2.0
        if (m.kpts[wrists, 1] > shoulder_y + self.cfg["neutral_wrist_below_sw"] * sw).all():
            yaw = self.body_yaw_deg(state, pid)
            if yaw is not None:
                self.body_yaw[pid] = yaw

    def facing_camera(self, state: FrameState, pid: int) -> bool:
        """pid faces the camera with head and body, so their gesture counts:
        a tracked face turned at most act_max_head_yaw_deg this frame, and
        the body turned at most act_max_body_yaw_deg the last time the arms
        were down (body_yaw; not checked before the calibration, e.g. with
        --no-intro). Off with act_facing_camera: false."""
        cfg = self.cfg
        if not cfg["act_facing_camera"]:
            return True
        face = state.faces.get(pid)
        if face is None or not face.valid or abs(math.degrees(face.yaw)) > cfg["act_max_head_yaw_deg"]:
            return False
        body = self.body_yaw.get(pid)
        return body is None or body <= cfg["act_max_body_yaw_deg"]

    def _facing_or_blocked(self, state: FrameState, pid: int, gesture: str, t: float) -> bool:
        """True if pid's gesture counts; else it is logged as
        not_facing_<gesture> and the hint time is set."""
        if self.facing_camera(state, pid):
            return True
        self.not_facing_t[pid] = t
        self.actions.append((pid, f"not_facing_{gesture}"))
        return False

    def place_taken(self, state: FrameState, place: int) -> bool:
        """Waiting: someone in view with a face stands in start place `place`
        (1 = P1's, the left half of the frame; 2 = the right half; one player:
        the whole frame), by their shoulder midpoint (box centre without
        shoulders). Whoever it is: identity numbers players in the order it
        first sees them, not by where they stand."""
        n, w = len(self.round_players), state.frame.image.shape[1]
        lo, hi = (place - 1) / n * w, place / n * w
        return any(lo <= self.player_x(state, pid) < hi
                   for pid in state.players if self.in_view(state, pid))

    def player_x(self, state: FrameState, pid: int) -> float:
        """Where pid stands (px): the shoulder midpoint, or the box centre
        without shoulders. Decides the start places, and P1 / P2 when
        calibration starts (app/pipeline.py)."""
        pos = self.body_position(state.motion[pid]) if pid in state.motion else None
        if pos is not None:
            return pos[0]
        bbox = state.players[pid].bbox
        return float(bbox[0] + bbox[2]) / 2.0

    @staticmethod
    def in_view(state: FrameState, pid: int) -> bool:
        """pid is seen by the identity stage and has a face."""
        p, face = state.players.get(pid), state.faces.get(pid)
        return p is not None and p.status == "active" and face is not None and face.valid

    def _hands_up(self, state: FrameState, pid: int) -> bool:
        """Both of pid's wrists above the head this frame (the gesture rule)."""
        p, m = state.players.get(pid), state.motion.get(pid)
        if p is None or p.status != "active" or m is None:
            return False
        sw = shoulder_width(m.kpts, m.valid, self.cfg["min_shoulder_px"])
        return sw is not None and detect_hands_up(m.kpts, m.valid, sw, self.cfg)

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

    def torso_corners(self, m: Motion) -> np.ndarray | None:
        """The torso as four corners (4, 2) px, for paint in torso
        coordinates (PaintSplat): R_SHOULDER, L_SHOULDER, L_HIP, R_HIP, so
        u = 0 is the screen-left side for a player facing the camera (the
        model's L_* is on the screen's right, core/skeleton.py). None
        without a usable shoulder width (torso()). Hips not both visible:
        the shoulders moved down to torso()'s estimated hip level."""
        torso = self.torso(m)
        if torso is None:
            return None
        k, v = m.kpts.astype(np.float64), m.valid
        rs, ls = k[sk.R_SHOULDER], k[sk.L_SHOULDER]
        if v[sk.L_HIP] and v[sk.R_HIP]:
            lh, rh = k[sk.L_HIP], k[sk.R_HIP]
        else:
            drop = np.array([0.0, torso[2] - torso[1]])
            lh, rh = ls + drop, rs + drop
        return np.array([rs, ls, lh, rh])

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

    def new_round(self) -> None:
        """A fresh round with the same players (key r): they keep their
        positions and calibration; with the intro it starts from the countdown."""
        self._start_round(self._t)

    def reset(self, pid: int | None = None) -> None:
        """None: everything starts over (key R): a fresh round, the
        calibration forgotten, and with the intro back to waiting; player
        positions are kept. A pid: that player starts fresh (full health, no
        paint, no shield)."""
        if pid is not None:
            if pid in self.players:
                self._refresh(self.players[pid])
            self.shields.pop(pid, None)
            self.body_yaw.pop(pid, None)
            self.mouth_open.pop(pid, None)
            for d in self.drinks:
                if d.held_by == pid:
                    d.held_by = None         # falls on
            return
        self._clear_round()
        for player in self.players.values():
            player.move, player.move_t = 0, 0.0
            player.shoulder_px = player.arm_px = player.torso_px = None
        self.body_yaw.clear()
        self._enter("waiting" if self.intro else "playing", self._t)

    def _clear_round(self) -> None:
        for player in self.players.values():
            self._refresh(player)
        self.projectiles.clear()
        self.bolt_ends.clear()
        self.beams.clear()
        self.shields.clear()
        self.hits.clear()
        self.drinks.clear()
        self.drunk.clear()
        self._drink_wait = self._rng.uniform(*self.cfg["drink_every_s"])
        self.winner_id = None
        self.not_facing_t.clear()

    def _refresh(self, player: GamePlayer) -> None:
        player.health = self.max_health
        player.cooldown = 0.0
        player.paint_splats.clear()
        player.attack, player.attack_left = 1.0, 0.0
        player.defense, player.defense_left = 0.0, 0.0

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """The game state as plain shapes, for checking the logic (the game's
        look is drawn by the renderer): thin skeletons; body boxes with health
        and shield; the torso (thin) with its splashes as dots; bolts as dots with a tail and an x where
        they ended; beams as a line with their hit band (a growing circle while
        charging); shields as an ellipse beside the player (thin while
        fading); drinks as a circle of their size with the category's
        letter; this frame's actions and the winner."""
        unit = max(canvas.shape[1] / 100.0, 2.0)  # 1% of the frame width, px
        for pid, m in state.motion.items():
            draw_skeleton(canvas, m.kpts, m.valid, player_color(pid), 1)

        for pid, player in sorted(self.players.items()):
            if player.bounds is None:
                continue
            label = f"P{pid} {player.health}" + (" shield" if self.shielded(pid) else "")
            if player.attack_left > 0:
                label += f" atk x{player.attack:.2f} {player.attack_left:.0f}s"
            if player.defense_left > 0:
                label += f" def {player.defense:.0%} {player.defense_left:.0f}s"
            draw_bbox(canvas, player.bounds, player_color(pid), label, dashed=True)
            if player.torso is not None:
                cv2.polylines(canvas, [np.round(player.torso).astype(np.int32)], True,
                              player_color(pid), 1, cv2.LINE_AA)
                for splat in player.paint_splats:
                    c = bilinear(player.torso, splat.u, splat.v)
                    cv2.circle(canvas, pt(c), 4, player_color(splat.owner_id), -1, cv2.LINE_AA)

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

        for d in self.drinks:
            r = int(self.drink_height(d) / 2)
            width = 2 if d.held_by is None else 4                # thick while held at a mouth
            cv2.circle(canvas, pt((d.x, d.y)), r, DRINK_COLORS[d.kind], width, cv2.LINE_AA)
            draw_lines(canvas, [f"{d.kind[0].upper()} {d.value:.2f}"], (int(d.x) - r, int(d.y) + 5))
        for pid, f in state.faces.items():           # mouths: the catch circle, filled while open
            if f.valid and "mouth" in f.anchors:
                r = int(self.cfg["drink_catch_fw"] * f.scale)
                cv2.circle(canvas, pt(f.anchors["mouth"]), r, player_color(pid),
                           -1 if self.mouth_open.get(pid) else 1, cv2.LINE_AA)

        lines = [f"phase {self.phase} {self.phase_elapsed():.1f} s"]
        if self.phase == "calibration":
            for pid in self.round_players:
                move = self.current_move(pid)
                extra = f" {self.neutral_fraction(pid):.0%}" if move == "neutral" else ""
                lines.append(f"P{pid} next: {move or 'done'}{extra}")
            if self.line_gap is not None:
                line = f"line-up: {self.line_gap:.2f} m apart"
                lines.append(line + (" level" if self.level else f", P{self.nearer} step back"))
            elif not self.level:
                lines.append("line-up: waiting for both players")
        for pid in self.round_players:
            arm = self.arm_sw(pid)
            if arm is not None:
                lines.append(f"P{pid} shoulders {self.players[pid].shoulder_px:.0f} px, arm {arm:.2f} sw"
                             + (" SHORT" if self.short_arm(pid) else ""))
        lines += [f"P{pid} {action}" for pid, action in self.actions]
        if self.winner_id is not None:
            lines.append(f"P{self.winner_id} wins")
        draw_lines(canvas, lines, (10, canvas.shape[0] - 20 * len(lines) - 10))

    def log_fields(self, state: FrameState) -> dict:
        """CSV columns: game_phase; game_p<pid>_health, game_p<pid>_shield (1
        while it blocks), game_p<pid>_actions (this frame's actions joined by
        '+'), game_p<pid>_move (next calibration move, empty when all are
        done), game_p<pid>_shoulder_px, game_p<pid>_arm_px,
        game_p<pid>_torso_px and game_p<pid>_arm_sw (the calibration, empty
        before it); game_p<pid>_body_yaw (degrees, the last arms-down
        body_yaw_deg(); empty before the calibration); game_level (1 while the players stand
        level) and game_line_gap (m); game_bolts and game_beams (in flight),
        game_winner (pid or empty)."""
        row: dict = {"game_phase": self.phase}
        for pid, player in sorted(self.players.items()):
            pre = f"game_p{pid}_"
            row[pre + "health"] = player.health
            row[pre + "shield"] = int(self.shielded(pid))
            row[pre + "actions"] = "+".join(a for p, a in self.actions if p == pid)
            row[pre + "move"] = self.current_move(pid) or ""
            arm = self.arm_sw(pid)
            row[pre + "shoulder_px"] = "" if player.shoulder_px is None else round(player.shoulder_px, 1)
            row[pre + "arm_px"] = "" if player.arm_px is None else round(player.arm_px, 1)
            row[pre + "torso_px"] = "" if player.torso_px is None else round(player.torso_px, 1)
            row[pre + "arm_sw"] = "" if arm is None else round(arm, 3)
            body = self.body_yaw.get(pid)
            row[pre + "body_yaw"] = "" if body is None else round(body, 1)
            row[pre + "mouth_open"] = int(self.mouth_open.get(pid, False))
            row[pre + "attack"] = round(player.attack, 3)
            row[pre + "attack_left"] = round(player.attack_left, 2)
            row[pre + "defense"] = round(player.defense, 3)
            row[pre + "defense_left"] = round(player.defense_left, 2)
            row[pre + "holding"] = next((d.kind for d in self.drinks if d.held_by == pid), "")
        row["game_level"] = int(self.level)
        row["game_line_gap"] = "" if self.line_gap is None else round(self.line_gap, 3)
        row["game_bolts"] = len(self.projectiles)
        row["game_beams"] = len(self.beams)
        row["game_drinks"] = len(self.drinks)
        row["game_winner"] = self.winner_id if self.winner_id is not None else ""
        return row
