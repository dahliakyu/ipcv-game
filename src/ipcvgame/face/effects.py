"""Task 1 — face-anchored AR effects.

A hat and a mustache per player, placed at the anchors from FaceTracker:
scaled to the face width (Face.scale), rotated by the head roll (Face.roll)
and faded by Face.alpha. The renderer calls draw(canvas, faces).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ipcvgame.core.config import REPO_ROOT
from ipcvgame.core.types import Face


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


class FaceEffects:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hat = load_sprite(cfg["hat_sprite"])
        self.mustache = load_sprite(cfg["mustache_sprite"])
        self.hat_width = float(cfg["hat_width"])
        self.mustache_width = float(cfg["mustache_width"])

    def draw(self, canvas: np.ndarray, faces: dict[int, Face]) -> None:
        hh, hw = self.hat.shape[:2]
        mh, mw = self.mustache.shape[:2]
        for f in faces.values():
            if not f.valid or f.alpha <= 0:
                continue
            if "hat" in f.anchors:       # pivot: bottom-middle of the hat
                paste(canvas, self.hat, f.anchors["hat"], (hw / 2, hh), f.roll,
                      self.hat_width * f.scale, f.alpha)
            if "mouth" in f.anchors:     # pivot: centre of the mustache
                paste(canvas, self.mustache, f.anchors["mouth"], (mw / 2, mh / 2), f.roll,
                      self.mustache_width * f.scale, f.alpha)

    def reset(self, pid: int | None = None) -> None:
        pass  # stateless: the fade lives in FaceTracker (Face.alpha)
