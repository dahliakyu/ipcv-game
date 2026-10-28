"""Task 1 — face-anchored AR effects.

A hat and a mustache per player, placed at the anchors from FaceTracker:
scaled to the face width (Face.scale), rotated by the head roll (Face.roll)
and faded by Face.alpha. The renderer calls draw(canvas, faces).

Team colour (feature 1): each player's hat band is turned to their team
colour (scene/draw.py), and two stripes of war paint in that colour sit on
each cheek anchor.
"""

from __future__ import annotations

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

    def team_hat(self, pid: int) -> np.ndarray:
        """The hat with its band in the player's team colour, made on first use."""
        if pid not in self._hats:
            self._hats[pid] = recolor(self.hat, self.band, int(self.cfg["hat_band_hue"]),
                                      player_color(pid))
        return self._hats[pid]

    def draw(self, canvas: np.ndarray, faces: dict[int, Face]) -> None:
        hw = self.hat.shape[1]
        mh, mw = self.mustache.shape[:2]
        cfg = self.cfg
        for f in faces.values():
            if not f.valid or f.alpha <= 0:
                continue
            polys = war_paint(f, cfg["warpaint_len"], cfg["warpaint_thick"], cfg["warpaint_gap"])
            if polys:
                fill_polys(canvas, polys, player_color(f.pid), cfg["warpaint_opacity"] * f.alpha)
            if "hat" in f.anchors:       # pivot: middle of the brim
                paste(canvas, self.team_hat(f.pid), f.anchors["hat"], (hw / 2, self.hat_brim), f.roll,
                      self.hat_width * f.scale, f.alpha)
            if "mouth" in f.anchors:     # pivot: centre of the mustache
                paste(canvas, self.mustache, f.anchors["mouth"], (mw / 2, mh / 2), f.roll,
                      self.mustache_width * f.scale, f.alpha)

    def reset(self, pid: int | None = None) -> None:
        pass  # stateless: the fade lives in FaceTracker (Face.alpha)
