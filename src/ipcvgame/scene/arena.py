"""Task 5 — the room reacts to the fight (scene processing, brief §3.2.5).

The camera image is the arena. Three things in it respond to the game's
actions (interaction/game.py), all drawn by the renderer:

1. Wall paint. Paint that misses a body lands on the room behind the
   players and stays there for the round, in the shooter's team colour:
   - a blob that flies out of the frame splashes on the side wall at the
     edge where it left (BoltEnd reason "edge");
   - a blob that hits a player also sprays past them onto the wall behind,
     in the flight direction, just past their far shoulder and sized by
     their shoulder width, so it shows around them at any distance ("hit");
   - a blob a shield stops bounces back as a smaller splash ("shield").
   A beam leaves nothing on the wall (2026-11-01: its stripe stayed after
   the beam and read as a laser that never switched off).
   Later paint covers earlier paint, so over a round the wall shows who
   attacked where, and how much of it each team holds (wall_share(),
   logged). A new round starts with a clean wall.
2. Lighting. While a beam charges and fires, the room dims and takes on
   the caster's colour, and lights up again after. The players stay lit,
   each in a soft spotlight: the person mask blurred much more than for the
   paint, so its coarse outline does not show as a hard halo.
3. Impact. Every hit shakes the picture for a moment, along the hit's
   direction (shake_offset()); the health bars and top bar stay still.

"Behind the players" needs a person mask. It comes from MediaPipe's Selfie
Segmenter (scene/segment.py, render.arena_segmenter), which follows the
real outline, hair and clothes included. Without the model file,
person_mask() builds one from the keypoints, the same way as the body
paint's clip mask (scene/paint.py body_mask()), but wider (it must cover
clothes and hair, not stay inside them), plus a disc for the head, softened
by a blur; a player without a torso this frame is covered by their whole
box. That mask is coarse: wall paint stops along its straight capsule
edges, with bare wall between it and a thin arm (docs/project_report.md, section 7).

The wall and the lighting work on the small image the pixel look shrinks
the camera frame to (render.pixel_size), so they share its look and cost a
quarter of the pixels at pixel_size 2. Positions from the game are
full-resolution pixels and are scaled down here.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.types import FrameState, Motion
from ipcvgame.interaction.game import Game
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.paint import body_mask, drip_polygons, fill_polygons, splash_shapes
from ipcvgame.scene.segment import PersonSegmenter

HEAD = [sk.NOSE, sk.L_EYE, sk.R_EYE, sk.L_EAR, sk.R_EAR]


class Arena:
    """The room's state for one round: wall paint, light and shake."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # body_mask() reads the paint's mask_* keys; the arena has its own,
        # wider ones.
        self._mask_cfg = {
            "mask_torso_u": cfg["arena_mask_torso_u"], "mask_torso_v": cfg["arena_mask_torso_v"],
            "mask_arm_sw": cfg["arena_mask_arm_sw"], "mask_leg_sw": cfg["arena_mask_leg_sw"],
        }
        # The Selfie Segmenter, loaded on first use (so the game logic and
        # tests run without it); False = not available, use the keypoints.
        self._segmenter: PersonSegmenter | None | bool = None
        self._silhouette: tuple[float, tuple, np.ndarray] | None = None   # (Frame.t, shape, mask)
        self.reset()

    def reset(self, pid: int | None = None) -> None:
        """A clean wall, full light, no shake. Per player there is nothing to
        forget: the wall belongs to the round, not to a player."""
        if pid is not None:
            return
        self._shape: tuple[int, int] | None = None   # wall size (small image px)
        self._wall_bgr: np.ndarray | None = None
        self._wall_a: np.ndarray | None = None       # paint coverage 0-255
        self._owner: np.ndarray | None = None        # pid whose paint is on top, 0 = none
        self._box: tuple[int, int, int, int] | None = None   # painted area x0, y0, x1, y1
        self.share: dict[int, float] = {}            # wall fraction per pid
        self._ends: set[tuple] = set()               # bolt ends already painted
        self._shakes: list[tuple[float, int, float]] = []   # (Frame.t, direction, amplitude px)
        self._dim = 0.0                              # 0 = full light, 1 = dimmest
        self._dim_color: tuple[int, int, int] = (0, 0, 0)
        self._phase: str | None = None
        self._t: float | None = None
        self.stamps = 0                              # paint stamps so far (tests, log)

    # ------------------------------------------------------------- per frame

    def update(self, state: FrameState, game: Game, small_shape: tuple[int, int]) -> None:
        """Read this frame's game events: paint the new bolt ends onto the
        wall, start a shake per hit, and move the light towards dim (a beam
        is up) or full. small_shape: (h, w) of the image the wall is drawn on."""
        t = state.frame.t
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), 0.1)
        self._t = t
        if self._new_round(game.phase) or self._shape != small_shape:
            self._clear_wall(small_shape)
        h, w = state.frame.image.shape[:2]
        self._scale = small_shape[1] / w

        ends = {(round(e.x, 1), round(e.y, 1), e.owner_id, e.direction, e.reason): e
                for e in game.bolt_ends}
        for key, e in ends.items():
            if key not in self._ends:
                self._paint_end(e.x, e.y, e.direction, e.owner_id, e.reason, w, h, game)
        self._ends = set(ends)

        cfg = self.cfg
        for hit in game.hits:
            self._shakes.append((t, 1 if hit.direction >= 0 else -1, cfg["arena_shake_fw"] * w))
        self._shakes = [s for s in self._shakes if t - s[0] < cfg["arena_shake_s"]]

        if game.beams:
            self._dim_color = player_color(game.beams[-1].owner_id)
        step = dt / max(cfg["arena_dim_fade_s"], 1e-6)
        target = 1.0 if game.beams else 0.0
        self._dim = min(self._dim + step, target) if self._dim < target else max(self._dim - step, target)

    def draw(self, small: np.ndarray, state: FrameState, game: Game) -> None:
        """Wall paint and lighting onto the small camera image, in place,
        both kept off the players (person_mask())."""
        wall = self._box is not None
        if not wall and self._dim <= 0.0:
            return
        h, w = small.shape[:2]
        person = self.person(small, state, game)
        if wall:
            x0, y0, x1, y1 = self._box
            roi = small[y0:y1, x0:x1]
            alpha = (self._wall_a[y0:y1, x0:x1].astype(np.float32)
                     * (self.cfg["arena_wall_opacity"] / 255.0)
                     * (1.0 - person[y0:y1, x0:x1]))
            roi[:] = cv2.blendLinear(self._wall_bgr[y0:y1, x0:x1], roi, alpha, 1.0 - alpha)
        if self._dim > 0.0:
            # small x (1 - dark) + colour x tint, with dark and tint both
            # proportional to the light lost (dim) off the players: the same
            # as blending towards one fixed colour, colour x arena_tint /
            # arena_dim, by w = arena_dim x dim, in one OpenCV call.
            cfg = self.cfg
            k = int(round(cfg["arena_light_blur_fw"] * small.shape[1])) | 1
            lit = cv2.GaussianBlur(person, (k, k), 0) if k > 1 else person
            w = (1.0 - lit) * np.float32(cfg["arena_dim"] * self._dim)
            target = np.empty_like(small)
            tint = cfg["arena_tint"] / max(cfg["arena_dim"], 1e-6)
            target[:] = np.clip(np.float32(self._dim_color) * tint, 0, 255).astype(np.uint8)
            small[:] = cv2.blendLinear(small, target, 1.0 - w, w)

    def silhouette(self, small: np.ndarray, t: float) -> np.ndarray | None:
        """The Selfie Segmenter's person mask of the small image, float32
        0-1 (render.arena_segmenter); None when it is null or its model is
        missing. Run at most once per frame (Frame.t): the wall paint, the
        lighting and the orb glow share it."""
        if self._segmenter is None:
            path = self.cfg.get("arena_segmenter")
            self._segmenter = False
            if path:
                try:
                    self._segmenter = PersonSegmenter(path, self.cfg["arena_segmenter_ramp"])
                except FileNotFoundError as exc:
                    print(f"[arena] {exc}; the wall paint and glow use the keypoint mask")
        if not self._segmenter:
            return None
        if self._silhouette is None or self._silhouette[:2] != (t, small.shape[:2]):
            self._silhouette = (t, small.shape[:2], self._segmenter(small))
        return self._silhouette[2]

    def person(self, small: np.ndarray, state: FrameState, game: Game) -> np.ndarray:
        """Where the people are on the small image, float32 0-1: the
        silhouette(), or the keypoint mask (person_mask()) without the
        segmenter."""
        mask = self.silhouette(small, state.frame.t)
        if mask is not None:
            return mask
        return person_mask(small.shape[:2], self._scale, state, game, self._mask_cfg, self.cfg)

    def shake_offset(self, t: float) -> tuple[float, float]:
        """The picture's offset (px) at time t from the hits' shakes: a
        damped sine along each hit's direction, (1 - age / arena_shake_s)^2
        times arena_shake_fw of the frame width, starting towards where the
        hit went; half as much up and down."""
        cfg = self.cfg
        dx = dy = 0.0
        for t0, direction, amp in self._shakes:
            age = t - t0
            if not 0.0 <= age < cfg["arena_shake_s"]:
                continue
            env = amp * (1.0 - age / cfg["arena_shake_s"]) ** 2
            phase = 2.0 * math.pi * cfg["arena_shake_hz"] * age
            dx += env * direction * math.sin(phase)
            dy += 0.5 * env * math.sin(1.3 * phase + 1.0)
        return dx, dy

    def shake(self, canvas: np.ndarray, t: float) -> np.ndarray:
        """canvas moved by shake_offset(t), the edge pixels repeated into
        the gap; the same array when there is nothing to shake."""
        dx, dy = self.shake_offset(t)
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return canvas
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = canvas.shape[:2]
        return cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    def log_fields(self) -> dict:
        """Wall share per team (%), how much light is gone (0-1)."""
        row = {f"render_wall_p{pid}": round(100.0 * s, 2) for pid, s in self.share.items()}
        row["render_dim"] = round(self._dim, 3)
        return row

    # ----------------------------------------------------------------- wall

    def _new_round(self, phase: str) -> bool:
        """True when the wall must be cleaned: on every phase change except
        playing -> result (the wall stays on the result panel) and
        countdown -> playing (the round is already fresh)."""
        before, self._phase = self._phase, phase
        return before is not None and before != phase and (before, phase) not in (
            ("playing", "result"), ("countdown", "playing"))

    def _clear_wall(self, shape: tuple[int, int]) -> None:
        h, w = shape
        self._shape = shape
        self._wall_bgr = np.zeros((h, w, 3), np.uint8)
        self._wall_a = np.zeros((h, w), np.uint8)
        self._owner = np.zeros((h, w), np.uint8)
        self._box = None
        self.share = {}

    def _paint_end(self, x: float, y: float, direction: int, pid: int, reason: str,
                   w: int, h: int, game: Game) -> None:
        """One splash for a bolt end, placed by its reason (module doc)."""
        cfg = self.cfg
        d = 1 if direction >= 0 else -1
        seed = int(x) * 7919 + int(y) * 31 + pid
        if reason == "edge":
            r = cfg["arena_edge_r_h"] * h
            cx = min(max(x, 0.0), float(w)) - d * cfg["arena_edge_inset_r"] * r
            self._stamp(cx, y, d, pid, r, seed)
        elif reason == "hit":
            body = hit_body(game, x, pid)
            if body is None:                       # no torso: a fixed step past the hit
                self._stamp(x + d * cfg["arena_behind_fw"] * w, y, d, pid, cfg["arena_hit_r_h"] * h, seed)
            else:
                cx, sw = body
                self._stamp(cx + d * cfg["arena_behind_sw"] * sw, y, d, pid, cfg["arena_hit_r_sw"] * sw, seed)
        elif reason == "shield":
            self._stamp(x - d * cfg["arena_bounce_fw"] * w, y, -d, pid, cfg["arena_bounce_r_h"] * h, seed)

    def _stamp(self, x: float, y: float, direction: int, pid: int, radius: float, seed: int) -> None:
        """A splash (scene/paint.py's shapes, drips fully run) of `radius`
        full-res px at (x, y), flying `direction`, onto the wall."""
        s = self._scale
        polys, drips = splash_shapes(seed, self.cfg)
        polys = polys + drip_polygons(drips, 1.0)
        flip = np.array([direction, 1.0])
        pts = [(p * flip * radius + (x, y)) * s for p in polys]
        self._cover(pts, pid, lambda cov, pts_roi: fill_polygons(cov, pts_roi))

    def _cover(self, pts: list[np.ndarray], pid: int, fill) -> None:
        """Draw a coverage shape (fill(cov, points shifted into the region))
        over the region the points span, and put it on the wall in pid's
        colour: alpha compositing "over", so new paint covers old."""
        h, w = self._shape
        allp = np.concatenate(pts)
        x0, y0 = np.maximum(np.floor(allp.min(axis=0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(allp.max(axis=0)).astype(int) + 2, [w, h])
        if x0 >= x1 or y0 >= y1:
            return
        cov = np.zeros((y1 - y0, x1 - x0), np.uint8)
        fill(cov, [p - (x0, y0) for p in pts])
        if not cov.any():
            return
        a = cov.astype(np.float32) / 255.0
        old_a = self._wall_a[y0:y1, x0:x1].astype(np.float32) / 255.0
        out_a = a + old_a * (1.0 - a)
        old = self._wall_bgr[y0:y1, x0:x1].astype(np.float32)
        color = np.float32(player_color(pid))
        mix = ((color * a[:, :, None] + old * (old_a * (1.0 - a))[:, :, None])
               / np.maximum(out_a, 1e-6)[:, :, None])
        self._wall_bgr[y0:y1, x0:x1] = np.clip(mix, 0, 255).astype(np.uint8)
        self._wall_a[y0:y1, x0:x1] = np.round(out_a * 255.0).astype(np.uint8)
        self._owner[y0:y1, x0:x1][cov >= 128] = pid
        self._box = (x0, y0, x1, y1) if self._box is None else (
            min(self._box[0], x0), min(self._box[1], y0), max(self._box[2], x1), max(self._box[3], y1))
        counts = np.bincount(self._owner.ravel(), minlength=3)
        self.share = {p: counts[p] / self._owner.size for p in range(1, len(counts)) if counts[p]}
        self.stamps += 1


def hit_body(game: Game, x: float, shooter: int) -> tuple[float, float] | None:
    """(centre x, shoulder width) px of the player a blob ending at x hit: the
    other player whose box holds x, nearest centre first; None without one
    whose torso is known this frame."""
    best = None
    for pid, gp in game.players.items():
        if pid == shooter or gp.torso is None or gp.bounds is None:
            continue
        left, _, right, _ = gp.bounds
        if not left - 1.0 <= x <= right + 1.0:
            continue
        torso = np.asarray(gp.torso, float)
        cx = float(torso[:, 0].mean())
        if best is None or abs(cx - x) < abs(best[0] - x):
            best = (cx, float(np.linalg.norm(torso[1] - torso[0])))
    return best


def person_mask(shape: tuple[int, int], scale: float, state: FrameState, game: Game,
                mask_cfg: dict, cfg: dict) -> np.ndarray:
    """float32 mask (shape: h, w; 1 = a player) of everyone in view, in an
    image `scale` times the frame's size: body_mask() from the torso and
    limbs with the arena's wider mask_* widths, a head disc of
    arena_head_sw shoulder widths around the head keypoints, or the whole
    box for a player whose torso is not found; then blurred by
    arena_mask_blur_fw of the width, so wall paint fades out at the edge
    instead of cutting along it."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for pid, p in state.players.items():
        if p.status == "lost":
            continue
        gp = game.players.get(pid)
        m = state.motion.get(pid)
        torso = gp.torso if gp is not None and p.status == "active" else None
        if torso is None or m is None:
            x0, y0, x1, y1 = (np.asarray(p.bbox, float) * scale).round().astype(int)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
            continue
        torso_s = np.asarray(torso, float) * scale
        small = Motion(kpts=m.kpts * scale, vel=m.vel, valid=m.valid, signals=m.signals)
        np.maximum(mask, body_mask((h, w), (0, 0), torso_s, small, mask_cfg), out=mask)
        head = [i for i in HEAD if m.valid[i]]
        if head:
            c = small.kpts[head].mean(axis=0)
            sw = float(np.linalg.norm(torso_s[1] - torso_s[0]))
            cv2.circle(mask, (int(round(c[0])), int(round(c[1]))),
                       max(1, int(round(cfg["arena_head_sw"] * sw))), 255, -1, cv2.LINE_AA)
    k = int(round(cfg["arena_mask_blur_fw"] * w)) | 1
    if k > 1:
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask.astype(np.float32) / 255.0
