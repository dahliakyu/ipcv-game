"""Task 1 — face-anchored AR effects.

A hat and a mustache per player, placed at the anchors from FaceTracker:
scaled to the face width (Face.scale), rotated by the head roll (Face.roll)
and faded by Face.alpha. The renderer calls draw(canvas, faces).

The rest came with the face effects (docs/merge_log.md, "Face effects").

Team colour: each player's hat band is turned to their team colour
(scene/draw.py), and two stripes of war paint in that colour sit on each
cheek anchor.

Hit reaction: the renderer passes the game's hits each frame (update()).
A hit knocks the hat off: it flies away the way the attack went, spinning
and falling, fades, then pops back onto the head. The face flashes white,
then in the attacker's colour. The only state kept between frames: one
_Anim record per player, timed with Frame.t.

Cheek bulge: puffed cheeks, chipmunk style, for about a second. A real
image warp, not a sticker: around each cheek anchor the camera image is
magnified by a radial warp with cv2.remap (bulge_map()). Started only by
bulge(pid, t), when the player eats a consumable (extra_features.md,
feature 2); while a drink sits at the open mouth (update(..., holding)),
the cheeks already bulge gently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ipcvgame.core.config import REPO_ROOT
from ipcvgame.core.types import Face
from ipcvgame.scene.draw import hue_of, player_color, shift_hue


def load_sprite(path: str | Path) -> np.ndarray:
    """Load an RGBA sprite; a relative path is taken from the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise FileNotFoundError(f"{path} missing or has no transparency (needs RGBA)")
    return img


def paste(img: np.ndarray, sprite: np.ndarray, at, pivot, angle: float, width: float,
          alpha: float = 1.0) -> None:
    """Alpha-blend `sprite` onto `img` in place.

    Sprite point `pivot` (x, y in sprite pixels) lands on image point `at`.
    The sprite is rotated by `angle` (rad, as from arctan2 in image
    coordinates, so positive is clockwise on screen), scaled to `width` px
    and faded by `alpha` (0-1). Only the box the sprite covers is warped and
    blended, clipped to the frame.
    """
    sh, sw = sprite.shape[:2]
    pivot = np.asarray(pivot, dtype=float)
    M = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), -np.degrees(angle), width / sw)
    M[:, 2] += np.asarray(at, dtype=float) - pivot   # move the pivot onto the anchor
    # Box the sprite covers in the image, clipped to the frame
    corners = cv2.transform(np.float32([[[0, 0], [sw, 0], [0, sh], [sw, sh]]]), M)[0]
    h, w = img.shape[:2]
    x1, y1 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(corners.max(0)).astype(int), (w, h))
    if x2 <= x1 or y2 <= y1:
        return                                       # fully off-screen
    M[:, 2] -= (x1, y1)                              # warp into that box only
    warped = cv2.warpAffine(sprite, M, (int(x2 - x1), int(y2 - y1)))
    a = warped[:, :, 3:4] * (alpha / 255.0)
    roi = img[y1:y2, x1:x2]
    roi[:] = (warped[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def band_mask(sprite: np.ndarray, hue: int, tol: int, min_sat: int, min_val: int) -> np.ndarray:
    """Boolean mask of a sprite's visible pixels less than `tol` from `hue`
    (OpenCV hue, 0-179, measured around the circle so 178 is 2 from 0), at
    least min_sat saturated and min_val bright."""
    hsv = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_BGR2HSV)
    d = np.abs(hsv[:, :, 0].astype(int) - hue)
    d = np.minimum(d, 180 - d)
    return ((sprite[:, :, 3] > 0) & (d < tol)
            & (hsv[:, :, 1] >= min_sat) & (hsv[:, :, 2] >= min_val))


def recolor(sprite: np.ndarray, mask: np.ndarray, hue: int, color: tuple[int, int, int]) -> np.ndarray:
    """Copy of `sprite` with the pixels in `mask` turned from `hue` to the
    hue of `color`; saturation and brightness stay, so the shading stays.
    Already that hue: an exact copy (the HSV round trip would shift colours
    by a few levels)."""
    out = sprite.copy()
    shift = (hue_of(color) - hue) % 180
    if shift:
        out[mask] = shift_hue(sprite, shift)[mask]
    return out


def fill_polys(img: np.ndarray, polys: list[np.ndarray], color: tuple[int, int, int],
               alpha: float) -> None:
    """Fill polygons (px, float) on `img` in place, blended with `alpha`
    (0-1). Only their bounding box is touched, clipped to the frame."""
    pts = np.concatenate(polys)
    h, w = img.shape[:2]
    x1, y1 = np.maximum(np.floor(pts.min(0)).astype(int) - 1, 0)
    x2, y2 = np.minimum(np.ceil(pts.max(0)).astype(int) + 2, (w, h))
    if x2 <= x1 or y2 <= y1:
        return
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    shift = 4                                        # sub-pixel corners: 1/16 px
    cv2.fillPoly(mask, [np.round((p - (x1, y1)) * (1 << shift)).astype(np.int32) for p in polys],
                 255, cv2.LINE_AA, shift)
    a = mask[:, :, None] * (alpha / 255.0)
    roi = img[y1:y2, x1:x2]
    roi[:] = (np.array(color, np.float64) * a + roi * (1 - a)).astype(np.uint8)


def war_paint(f: Face, length: float, thick: float, gap: float) -> list[np.ndarray]:
    """Two stripes (4 corners each, px) on each cheek anchor of `f`: along
    the eye line (roll), one above the other `gap` apart; sizes in face
    widths. The stripes on the cheek turned away from the camera are
    shortened by cos(yaw): yaw > 0 turns the face to the image right, which
    turns the image-right cheek away."""
    u = np.array([np.cos(f.roll), np.sin(f.roll)])  # along the eye line
    down = np.array([-u[1], u[0]])
    polys = []
    for name, side in (("left_cheek", -1), ("right_cheek", 1)):
        if name not in f.anchors:
            continue
        squash = np.cos(f.yaw) if side * f.yaw > 0 else 1.0
        along = 0.5 * length * f.scale * squash * u
        across = 0.5 * thick * f.scale * down
        for k in (-0.5, 0.5):
            c = f.anchors[name] + k * gap * f.scale * down
            polys.append(np.array([c - along - across, c + along - across,
                                   c + along + across, c - along + across]))
    return polys


def ellipse(center, axes, angle: float, n: int = 36) -> np.ndarray:
    """Polygon (n points, px) of an ellipse with half-axes `axes` (along,
    across), its first axis turned by `angle` (rad, clockwise on screen)."""
    u = np.array([np.cos(angle), np.sin(angle)])
    v = np.array([-u[1], u[0]])
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)[:, None]
    return np.asarray(center, float) + axes[0] * np.cos(a) * u + axes[1] * np.sin(a) * v


def bulge_map(size: tuple[int, int], centre, radius: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Sampling maps for cv2.remap: a radial bulge (magnifier) of strength k
    around `centre` (px, in the output's coordinates), for an output of
    `size` (w, h).

    Each output pixel at distance rho < radius from the centre takes the
    source pixel at rho' = rho * (rho / radius) ** e, with e = k (1 - u^2)
    and u = rho / radius. Since rho' < rho, the middle is magnified: the
    cheek swells. The (1 - u^2) factor brings e to 0 at the rim, so rho' =
    rho there and the warp also has slope 1: it blends into the unwarped
    image without a seam. k = 0 changes nothing."""
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - np.float32(centre[0]), ys - np.float32(centre[1])
    u = np.minimum(np.sqrt(dx * dx + dy * dy) / max(radius, 1e-6), 1.0)
    f = np.power(np.maximum(u, 1e-6), k * (1.0 - u * u)).astype(np.float32)
    return np.float32(centre[0]) + dx * f, np.float32(centre[1]) + dy * f


def warp_bulge(img: np.ndarray, centre, radius: float, k: float) -> None:
    """Bulge `img` in place around `centre` (px): only the box of the
    circle is remapped, clipped to the frame."""
    if k <= 0 or radius < 2:
        return
    h, w = img.shape[:2]
    x1, y1 = (max(int(np.floor(c - radius)), 0) for c in centre)
    x2 = min(int(np.ceil(centre[0] + radius)) + 1, w)
    y2 = min(int(np.ceil(centre[1] + radius)) + 1, h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    mx, my = bulge_map((x2 - x1, y2 - y1), (centre[0] - x1, centre[1] - y1), radius, k)
    roi[:] = cv2.remap(roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@dataclass
class _Anim:
    """One player's hit reaction and cheek bulge. Times are Frame.t; None = not running."""
    knock_t: float | None = None     # the hat was knocked off
    knock_at: np.ndarray | None = None   # hat anchor (px) at that moment
    knock_scale: float = 0.0         # face width (px) then, sizes the flight
    knock_roll: float = 0.0          # head roll then (rad)
    knock_dir: int = 1               # the way the attack went: +1 right, -1 left
    flash_t: float | None = None     # the face flash started
    flash_owner: int = 0             # the attacker, for the flash colour
    bulge_t: float | None = None     # the cheek bulge started
    bulge_k0: float = 0.0            # its strength then, so it rises from there
    hold_level: float = 0.0          # the gentle bulge while a drink is at the mouth
    hold_t: float | None = None      # Frame.t hold_level was last moved


class FaceEffects:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hat = load_sprite(cfg["hat_sprite"])
        # The hat hangs from the bottom of its visible part (the brim), not from the
        # bottom of the image: the PNG has transparent padding below the hat.
        self.hat_brim = float(np.flatnonzero(self.hat[:, :, 3].max(axis=1) > 0).max() + 1)
        self.mustache = load_sprite(cfg["mustache_sprite"])
        self.hat_width = float(cfg["hat_width"])
        self.mustache_width = float(cfg["mustache_width"])
        self.band = band_mask(self.hat, int(cfg["hat_band_hue"]), int(cfg["hat_band_hue_tol"]),
                              int(cfg["hat_band_min_sat"]), int(cfg["hat_band_min_val"]))
        self._hats: dict[int, np.ndarray] = {}      # hat with the band in each pid's colour
        self._anim: dict[int, _Anim] = {}            # hit reaction per player
        self.knock_s = float(cfg["hat_fly_s"]) + float(cfg["hat_pop_s"])   # whole hat reaction

    def team_hat(self, pid: int) -> np.ndarray:
        """The hat with its band in the player's team colour, made on first use."""
        if pid not in self._hats:
            self._hats[pid] = recolor(self.hat, self.band, int(self.cfg["hat_band_hue"]),
                                      player_color(pid))
        return self._hats[pid]

    def update(self, hits: Iterable, faces: dict[int, Face], t: float,
               holding: Iterable[int] = ()) -> None:
        """Start the hit reaction for this frame's hits (Game.hits: target_id,
        owner_id, direction). Every hit restarts the flash; the hat is
        knocked off only while it is on the head, so hits in a row never
        throw two hats. A player without a visible face gets neither.

        `holding`: the players with a drink at their mouth now. Their cheeks
        bulge gently (bulge_hold_k), reached and let go over
        bulge_hold_ramp_s, so the warp never jumps."""
        holding = set(holding)
        rate = self.cfg["bulge_hold_k"] / self.cfg["bulge_hold_ramp_s"]
        for pid in holding | {pid for pid, a in self._anim.items() if a.hold_level > 0}:
            a = self._anim.setdefault(pid, _Anim())
            target = self.cfg["bulge_hold_k"] if pid in holding else 0.0
            step = rate * max(0.0, t - a.hold_t) if a.hold_t is not None else 0.0
            a.hold_level += float(np.clip(target - a.hold_level, -step, step))
            a.hold_t = t
        for h in hits:
            f = faces.get(h.target_id)
            if f is None or not f.valid or f.alpha <= 0:
                continue
            a = self._anim.setdefault(h.target_id, _Anim())
            a.flash_t, a.flash_owner = t, h.owner_id
            if "hat" in f.anchors and not self._knocked(a, t):
                a.knock_t, a.knock_at = t, np.array(f.anchors["hat"], float)
                a.knock_scale, a.knock_roll, a.knock_dir = f.scale, f.roll, h.direction

    def bulge(self, pid: int, t: float) -> None:
        """Puff this player's cheeks from time t (Frame.t): called when they
        eat a consumable (extra_features.md, feature 2), the only trigger.
        It rises from the strength the cheeks have now (the gentle bulge of
        a drink at the mouth, or a puff already running), so it starts over
        without a jump."""
        a = self._anim.setdefault(pid, _Anim())
        a.bulge_k0, a.bulge_t = self.bulge_strength(a, t), t

    def bulge_strength(self, a: _Anim | None, t: float) -> float:
        """Warp strength k at time t: the larger of the gentle bulge while a
        drink is at the mouth and the puff after drinking. The puff follows
        half a sine over bulge_s: rising from its start strength (bulge_k0)
        to bulge_k in the first half, falling to 0 in the second."""
        if a is None:
            return 0.0
        puff = 0.0
        if a.bulge_t is not None:
            u = (t - a.bulge_t) / self.cfg["bulge_s"]
            peak = self.cfg["bulge_k"]
            if 0 <= u < 0.5:
                puff = a.bulge_k0 + (peak - a.bulge_k0) * np.sin(np.pi * u)
            elif 0.5 <= u < 1:
                puff = peak * np.sin(np.pi * u)
        return float(max(a.hold_level, puff))

    def _knocked(self, a: _Anim, t: float) -> bool:
        """The hat is flying or popping back."""
        return a.knock_t is not None and t - a.knock_t < self.knock_s

    def flying_hat(self, a: _Anim, t: float) -> tuple[np.ndarray, float, float] | None:
        """The knocked-off hat's brim point (px), angle (rad) and opacity at
        time t, or None once it has faded. Thrown with hat_fly_v (face
        widths/s: sideways, the way the attack went, and up), it falls with
        hat_gravity and spins hat_spin_deg per second the same way; fully
        visible for the first half of hat_fly_s, then fading out."""
        cfg = self.cfg
        if a.knock_t is None or not 0 <= t - a.knock_t < cfg["hat_fly_s"]:
            return None
        dt = t - a.knock_t
        vx, vy = cfg["hat_fly_v"]
        fall = -vy * dt + 0.5 * cfg["hat_gravity"] * dt ** 2           # screen y is down
        pos = a.knock_at + a.knock_scale * np.array([a.knock_dir * vx * dt, fall])
        angle = a.knock_roll + a.knock_dir * np.radians(cfg["hat_spin_deg"]) * dt
        return pos, angle, min(1.0, 2.0 * (1.0 - dt / cfg["hat_fly_s"]))

    def hat_size(self, a: _Anim | None, t: float) -> float:
        """Size of the hat on the head (1 = normal): 0 while it is away, then
        growing back over hat_pop_s (ease-out, so it pops on quickly)."""
        if a is None or not self._knocked(a, t):
            return 1.0
        k = (t - a.knock_t - self.cfg["hat_fly_s"]) / self.cfg["hat_pop_s"]
        return 0.0 if k < 0 else 1.0 - (1.0 - k) ** 2

    def flash(self, a: _Anim | None, t: float) -> tuple[tuple[int, int, int], float] | None:
        """Flash colour and opacity at time t, or None: white for the first
        half of flash_s, then the attacker's colour, fading out linearly."""
        if a is None or a.flash_t is None:
            return None
        k = (t - a.flash_t) / self.cfg["flash_s"]
        if not 0 <= k < 1:
            return None
        color = (255, 255, 255) if k < 0.5 else player_color(a.flash_owner)
        return color, self.cfg["flash_opacity"] * (1.0 - k)

    def draw(self, canvas: np.ndarray, faces: dict[int, Face], t: float | None = None) -> None:
        """Draw every effect. t (Frame.t) runs the hit reaction and the cheek
        bulge; None draws the faces only (debug views, which have no game).
        Order per face: war paint, cheek bulge (so the paint swells with
        the cheek), mustache, flash, hat."""
        hw = self.hat.shape[1]
        mh, mw = self.mustache.shape[:2]
        cfg = self.cfg
        for f in faces.values():
            if not f.valid or f.alpha <= 0:
                continue
            a = self._anim.get(f.pid) if t is not None else None
            polys = war_paint(f, cfg["warpaint_len"], cfg["warpaint_thick"], cfg["warpaint_gap"])
            if polys:
                fill_polys(canvas, polys, player_color(f.pid), cfg["warpaint_opacity"] * f.alpha)
            k = self.bulge_strength(a, t) * f.alpha
            if k > 0:
                for name, side in (("left_cheek", -1), ("right_cheek", 1)):
                    if name in f.anchors:     # the cheek turned away looks smaller, as the war paint
                        squash = np.cos(f.yaw) if side * f.yaw > 0 else 1.0
                        warp_bulge(canvas, f.anchors[name], cfg["bulge_radius"] * f.scale * squash, k)
            if "mouth" in f.anchors:     # pivot: centre of the mustache
                paste(canvas, self.mustache, f.anchors["mouth"], (mw / 2, mh / 2), f.roll,
                      self.mustache_width * f.scale, f.alpha)
            fl = self.flash(a, t)
            if fl is not None:           # over the face, under the hat
                axes = np.array(cfg["flash_axes"], float) * f.scale
                fill_polys(canvas, [ellipse(f.center, axes, f.roll)], fl[0], fl[1] * f.alpha)
            size = self.hat_size(a, t)
            if "hat" in f.anchors and size > 0:   # pivot: middle of the brim
                paste(canvas, self.team_hat(f.pid), f.anchors["hat"], (hw / 2, self.hat_brim), f.roll,
                      size * self.hat_width * f.scale, f.alpha)
        if t is None:
            return
        # Flying hats, also for a player whose face is gone meanwhile
        for pid, a in self._anim.items():
            fly = self.flying_hat(a, t)
            if fly is not None:
                pos, angle, alpha = fly
                paste(canvas, self.team_hat(pid), pos, (hw / 2, self.hat_brim), angle,
                      self.hat_width * a.knock_scale, alpha)

    def reset(self, pid: int | None = None) -> None:
        """Forget the hit reaction and cheek bulge of one player, or of all
        (the face fade itself lives in FaceTracker, Face.alpha)."""
        if pid is None:
            self._anim.clear()
        else:
            self._anim.pop(pid, None)
