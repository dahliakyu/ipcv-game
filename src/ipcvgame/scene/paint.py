"""Task 5 — paint on the players' bodies (extra_features.md feature 3).

Each player's splashes (game.PaintSplat, in torso coordinates: u from the
screen-left shoulder 0 to the right one 1, v from the shoulder line 0 to the
hips 1) are kept as polygons in torso coordinates, made once per splash
(splash_polygons()). Every frame they go onto the body in three steps:

1. Place: every polygon point goes through bilinear() over the torso's four
   corners, the same map the game uses, so the paint stays at its (u, v)
   while the player moves, leans and turns. Points outside the torso
   (u < 0, v > 1, ...) are extrapolated, so paint can run past the
   shoulders onto the arms. A warp of an image through the four corners
   (cv2.warpPerspective) was tried first: a perspective map treats the
   torso's trapezoid (hips narrower than shoulders) as a slanted rectangle,
   and on a test body it put the middle of the torso 42 px too low.
2. Clip: body_mask() from the keypoints, the torso quad widened a little plus
   a capsule (a thick line) along each upper and lower arm and leg, so paint
   never lands on the background.
3. Shade (partly): the paint is multiplied by (1 - s) + s x the camera's
   brightness relative to its mean under the paint. Folds and shadows show
   through, but a dark shirt does not darken the paint, because only the
   brightness relative to the shirt counts.
"""

from __future__ import annotations

import cv2
import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.geometry import bilinear
from ipcvgame.core.types import Motion
from ipcvgame.scene.draw import player_color

ARMS = [(sk.L_SHOULDER, sk.L_ELBOW), (sk.L_ELBOW, sk.L_WRIST),
        (sk.R_SHOULDER, sk.R_ELBOW), (sk.R_ELBOW, sk.R_WRIST)]
LEGS = [(sk.L_HIP, sk.L_KNEE), (sk.L_KNEE, sk.L_ANKLE),
        (sk.R_HIP, sk.R_KNEE), (sk.R_KNEE, sk.R_ANKLE)]
SHIFT = 4                 # OpenCV fixed-point bits: polygons drawn at 1/16 px


def body_mask(shape: tuple[int, int], origin: tuple[int, int], torso: np.ndarray,
              motion: Motion | None, cfg: dict) -> np.ndarray:
    """uint8 mask (shape: h, w) of the body in an image region whose top-left
    corner is `origin` (x, y) px: the torso quad stretched to mask_torso_u /
    _v (the keypoints are joints, inside the body's outline), and a capsule
    along every arm and leg bone with both keypoints valid, mask_arm_sw /
    mask_leg_sw shoulder widths thick. Edges anti-aliased."""
    mask = np.zeros(shape, np.uint8)
    o = np.asarray(origin, np.float64)
    (u0, u1), (v0, v1) = cfg["mask_torso_u"], cfg["mask_torso_v"]
    quad = bilinear(torso, [u0, u1, u1, u0], [v0, v0, v1, v1]) - o
    cv2.fillConvexPoly(mask, fixed(quad), 255, cv2.LINE_AA, SHIFT)
    if motion is None:
        return mask
    sw = float(np.linalg.norm(torso[1] - torso[0]))
    for bones, width in ((ARMS, cfg["mask_arm_sw"]), (LEGS, cfg["mask_leg_sw"])):
        thick = max(1, int(round(width * sw)))
        for a, b in bones:
            if motion.valid[a] and motion.valid[b]:
                pa, pb = fixed(motion.kpts[[a, b]] - o)
                cv2.line(mask, tuple(pa), tuple(pb), 255, thick, cv2.LINE_AA, SHIFT)
    return mask


def fixed(points: np.ndarray) -> np.ndarray:
    """Points in px to OpenCV's fixed point (SHIFT bits), int32."""
    return np.round(np.asarray(points) * (1 << SHIFT)).astype(np.int32)


def splash_polygons(splat, coverage: float, cfg: dict) -> list[np.ndarray]:
    """The splash's shape as polygons (N, 2) in torso coordinates (u, v).
    Sizes are in torso widths, and v is divided by paint_aspect (the
    torso's height over its width), so a round shape stays about round on
    the body. For now a disc of radius splash_r = [base, growth] torso
    widths, growing as the player's health drops (coverage = 1 - health /
    max_health); step 3 replaces it with the splash generator."""
    r0, r1 = cfg["splash_r"]
    r = r0 + r1 * coverage
    a = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    return [np.stack([splat.u + r * np.cos(a), splat.v + r * np.sin(a) / cfg["paint_aspect"]], axis=1)]


class BodyPaint:
    """One player's paint: the splashes' polygons (made again only when the
    splashes or the coverage change), drawn onto the body every frame."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._key: tuple | None = None
        self._shapes: list[tuple[tuple[int, int, int], list[np.ndarray]]] = []
        self.builds = 0           # how often the shapes were made (tests)

    def update(self, splats: list, coverage: float) -> None:
        """Make the polygons again if the splashes or the coverage changed."""
        key = (tuple((s.seed, s.owner_id, s.u, s.v) for s in splats), round(coverage, 3))
        if key == self._key:
            return
        self._key = key
        self.builds += 1
        self._shapes = [(player_color(s.owner_id), splash_polygons(s, coverage, self.cfg)) for s in splats]

    def draw(self, canvas: np.ndarray, torso: np.ndarray, motion: Motion | None, splats: list,
             coverage: float) -> None:
        """Map the polygons onto the body (torso: the game's four corners),
        clip them to the body mask, shade them and blend them into canvas in
        place. Nothing without splashes, or with a torso too narrow or folded
        to map."""
        if not splats:
            return
        self.update(splats, coverage)
        cfg = self.cfg
        if (np.linalg.norm(torso[1] - torso[0]) < cfg["paint_min_torso_px"]
                or not cv2.isContourConvex(torso.astype(np.float32).reshape(-1, 1, 2))):
            return
        mapped = [(color, [bilinear(torso, p[:, 0], p[:, 1]) for p in polys])
                  for color, polys in self._shapes]
        pts = np.concatenate([p for _, polys in mapped for p in polys])
        h, w = canvas.shape[:2]
        x0, y0 = np.maximum(np.floor(pts.min(axis=0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + 2, [w, h])
        if x0 >= x1 or y0 >= y1:
            return
        o = np.array([x0, y0], np.float64)
        color_img = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        coverage_img = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for color, polys in mapped:
            fp = [fixed(p - o) for p in polys]
            # Colour without anti-aliasing plus a 2 px outline, so the soft
            # edge of the coverage never picks up black from outside.
            cv2.fillPoly(color_img, fp, color, cv2.LINE_8, SHIFT)
            cv2.polylines(color_img, fp, True, color, 2, cv2.LINE_8, SHIFT)
            cv2.fillPoly(coverage_img, fp, 255, cv2.LINE_AA, SHIFT)
        mask = body_mask(coverage_img.shape, (x0, y0), torso, motion, cfg)
        alpha = coverage_img.astype(np.float32) * mask * (cfg["paint_opacity"] / 255.0 / 255.0)
        roi = canvas[y0:y1, x0:x1]
        paint = np.clip(color_img.astype(np.float32) * self._shade(roi, alpha)[:, :, None], 0, 255)
        a = alpha[:, :, None]
        roi[:] = (paint * a + roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)

    def _shade(self, roi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Per-pixel factor for the paint: (1 - s) + s x the camera's
        brightness over its mean under the paint (s = paint_shade), the
        ratio capped at 2, so the factor is 1 - s to 1 + s. +1 keeps a black
        image at a ratio of 1."""
        s = float(self.cfg["paint_shade"])
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        under = alpha > 0.05
        if s <= 0 or not under.any():
            return np.ones_like(gray)
        ratio = np.minimum((gray + 1.0) / (float(gray[under].mean()) + 1.0), 2.0)
        return (1.0 - s) + s * ratio
