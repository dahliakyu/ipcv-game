"""Task 5 — paint on the players' bodies (docs/merge_log.md, "Normal attack").

Each player's splashes (game.PaintSplat, in torso coordinates: u from the
screen-left shoulder 0 to the right one 1, v from the shoulder line 0 to the
hips 1) are generated once per hit as polygons (splash_shapes(), seeded by
the hit, so a splash looks the same in every frame and no two look alike):
a wobbly blob, spikes thrown on in the attack's direction, droplets and
drips that run down over drip_s. Each splash is drawn once into a small
coverage sprite. Every frame the sprites go onto the body in three steps:

1. Place: each sprite is put on the body with the local affine map of
   bilinear() at its splash's centre (the map's value and slopes there), the
   torso coordinates the game uses: the centre lands exactly at its (u, v)
   and the shape follows the torso's local stretch and lean. Across one
   splash the bilinear map hardly bends, so one cv2.warpAffine per splash
   replaces mapping and filling its ~30 polygons every frame (that cost 4-9
   ms per player with 10 splashes; the target is under 2 ms). Paint past the
   torso (u < 0, ...) is extrapolated, so it can run onto the arms. While
   a splash's drips run (drip_s after the hit), they are drawn as polygons
   with the same map, then baked into its sprite. A warp of one image
   through the torso's four corners (cv2.warpPerspective) was tried first: a
   perspective map treats the torso's trapezoid (hips narrower than
   shoulders) as a slanted rectangle, and on a test body it put the middle
   of the torso 42 px too low.
2. Clip: body_mask() from the keypoints, the torso quad widened a little plus
   a capsule (a thick line) along each upper and lower arm and leg, so paint
   never lands on the background.
3. Shade (partly): the paint is multiplied by (1 - s) + s x the camera's
   brightness relative to its mean under the paint. Folds and shadows show
   through, but a dark shirt does not darken the paint, because only the
   brightness relative to the shirt counts.

The paint blob in flight (draw_blob(), the normal attack; it replaced the
fire bolt sprite) and its burst into drops on a shield (draw_burst()) are
here too: they share the blob() outline.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def blob(n: int, radius: float, wobble: list[tuple[float, float]], center=(0.0, 0.0)) -> np.ndarray:
    """A closed outline of n points: a circle whose radius changes with the
    angle, r(a) = radius x (1 + sum of amp x sin(k a + phase)) for k = 2, 3,
    ... (wobble: (amp, phase) per k). Low k bends the whole outline, so it
    reads as liquid, not as a gear."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = np.ones_like(a)
    for k, (amp, phase) in enumerate(wobble, start=2):
        r += amp * np.sin(k * a + phase)
    return np.stack([center[0] + radius * r * np.cos(a), center[1] + radius * r * np.sin(a)], axis=1)


def splash_shapes(seed: int, cfg: dict) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
    """One splash, generated from its seed, in units of the central blob's
    radius: x along the attack's flight (the paint flies on that way), y
    down. Returns the fixed polygons and the drips (x, start y, full
    length, width), which run down over time (drip_polygons()).

    - the central blob: blob() with wobble amplitudes up to splash_wobble / k;
    - splash_spikes spikes: tapering wedges from inside the blob out to
      1 + spike_len radii, a share spike_forward of them within
      spike_spread_rad of the flight direction, the rest anywhere; each with
      a drop at its tip;
    - splash_drops droplets at drop_dist radii, aimed like the spikes, sizes
      from a power law (drop_r = [min, max], exponent drop_power): many
      small, few large, as real spray;
    - drips from the blob's lower edge."""
    rng = np.random.default_rng(seed)
    polys: list[np.ndarray] = []

    def wobble(scale: float) -> list[tuple[float, float]]:
        return [(rng.uniform(0, scale / k), rng.uniform(0, 2 * np.pi)) for k in range(2, 6)]

    def aim() -> float:
        if rng.random() < cfg["spike_forward"]:
            return float(rng.normal(0.0, cfg["spike_spread_rad"]))
        return float(rng.uniform(-np.pi, np.pi))

    polys.append(blob(48, 1.0, wobble(cfg["splash_wobble"])))
    for _ in range(rng.integers(cfg["splash_spikes"][0], cfg["splash_spikes"][1] + 1)):
        ang = aim()
        reach = 1.0 + rng.uniform(*cfg["spike_len"])
        half = rng.uniform(*cfg["spike_width"]) / 2
        d, n = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
        # Base inside the blob, waist at two thirds, a point at the tip.
        polys.append(np.array([0.6 * d + half * n, 0.6 * reach * d + 0.55 * half * n, reach * d,
                               0.6 * reach * d - 0.55 * half * n, 0.6 * d - half * n]))
        polys.append(blob(12, 0.7 * half, wobble(0.2), center=reach * d))
    lo, hi = cfg["drop_r"]
    for _ in range(rng.integers(cfg["splash_drops"][0], cfg["splash_drops"][1] + 1)):
        ang, dist = aim(), rng.uniform(*cfg["drop_dist"])
        # Pareto sizes: r = lo x (1 - U)^(-1/power), cut at hi.
        r = min(lo * (1.0 - rng.random()) ** (-1.0 / cfg["drop_power"]), hi)
        polys.append(blob(12, r, wobble(0.2), center=(dist * np.cos(ang), dist * np.sin(ang))))
    drips = []
    for _ in range(rng.integers(cfg["drips"][0], cfg["drips"][1] + 1)):
        # Starting just inside the blob's lower edge, away from its sides.
        x = rng.uniform(-0.6, 0.6)
        drips.append((x, 0.8 * np.sqrt(1.0 - x * x), rng.uniform(*cfg["drip_len"]),
                      rng.uniform(*cfg["drip_w"])))
    return polys, drips


def drip_polygons(drips: list[tuple[float, float, float, float]], grown: float) -> list[np.ndarray]:
    """The drips at `grown` (0-1) of their full length: a strip narrowing a
    little downwards, with a round drop at its end."""
    polys = []
    for x, y0, length, w in drips:
        y1 = y0 + grown * length
        polys.append(np.array([[x - w / 2, y0], [x + w / 2, y0], [x + 0.35 * w, y1], [x - 0.35 * w, y1]]))
        polys.append(blob(12, 0.55 * w, [], center=(x, y1)))
    return polys


def fill_polygons(img: np.ndarray, polys: list[np.ndarray]) -> None:
    """Fill each polygon (px) in a uint8 coverage image, anti-aliased. One
    call per polygon: fillPoly (and drawContours) with several overlapping
    polygons fill even-odd and leave holes where they overlap."""
    for p in polys:
        cv2.fillPoly(img, [fixed(p)], 255, cv2.LINE_AA, SHIFT)


@dataclass
class _Splash:
    """A splash ready to draw: its shapes in blob radii (splash_shapes()),
    its place, the flight's sign along u, its attacker's colour and hit
    time, and its coverage sprite (drips added once they have run)."""
    u: float
    v: float
    sign: int
    color: tuple[int, int, int]
    t: float
    polys: list[np.ndarray]
    drips: list[tuple[float, float, float, float]]
    sprite: np.ndarray | None = None
    offset: tuple[int, int] = (0, 0)   # the cropped sprite's top-left in full-sprite px
    baked: bool = False                # drips fully run and drawn into the sprite


class BodyPaint:
    """One player's paint: each splash generated and drawn into its sprite
    once, the sprites placed on the body every frame."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._splashes: dict[tuple, _Splash] = {}
        self.builds = 0           # how many splashes were generated (tests)
        k = float(cfg["paint_sprite_px"])
        self._k = k
        self._c = float(int(np.ceil(cfg["paint_sprite_extent"] * k)))     # sprite centre, px
        self._n = 2 * int(self._c) + 1                                      # sprite size, px

    def update(self, splats: list, torso: np.ndarray) -> list[_Splash]:
        """The splashes to draw, generating the new ones. The flight's sign
        along u is fixed when a splash is made, from the torso then: flying
        right goes to +u while the screen-left shoulder is u = 0."""
        out = []
        for s in splats:
            key = (s.seed, s.owner_id, s.u, s.v, s.t)
            if key not in self._splashes:
                self.builds += 1
                polys, drips = splash_shapes(s.seed, self.cfg)
                sign = s.direction * (1 if torso[0][0] <= torso[1][0] else -1)
                self._splashes[key] = _Splash(s.u, s.v, sign, player_color(s.owner_id), s.t, polys, drips)
            out.append(self._splashes[key])
        if len(self._splashes) > len(out):         # forget splashes cleared by a new round
            keep = {id(x) for x in out}
            self._splashes = {k: x for k, x in self._splashes.items() if id(x) in keep}
        return out

    def _sprite(self, sp: _Splash, with_drips: bool) -> None:
        """Draw the splash's coverage into sp.sprite: paint_sprite_px per
        blob radius, paint_sprite_extent radii from the centre each way,
        cropped to its painted part (sp.offset), so placing it every frame
        touches as few pixels as possible."""
        img = np.zeros((self._n, self._n), np.uint8)
        polys = sp.polys + (drip_polygons(sp.drips, 1.0) if with_drips else [])
        fill_polygons(img, [p * self._k + self._c for p in polys])
        x, y, w, h = cv2.boundingRect(img)
        sp.sprite, sp.offset = img[y:y + h, x:x + w], (x, y)

    def _affine(self, sp: _Splash, torso: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """The local map at the splash's centre from its shape units (blob
        radii, x along the flight, y down) to image px: (origin (2,),
        columns (2, 2)). bilinear()'s slopes there: d/du = (1 - v)(q1 - q0)
        + v(q2 - q3), d/dv = (1 - u)(q3 - q0) + u(q2 - q1); x moves u by
        radius x sign, y moves v by radius / paint_aspect."""
        q, u, v = torso, sp.u, sp.v
        du = (1 - v) * (q[1] - q[0]) + v * (q[2] - q[3])
        dv = (1 - u) * (q[3] - q[0]) + u * (q[2] - q[1])
        cols = np.stack([du * radius * sp.sign, dv * radius / self.cfg["paint_aspect"]], axis=1)
        return bilinear(q, u, v), cols

    def draw(self, canvas: np.ndarray, torso: np.ndarray, motion: Motion | None, splats: list,
             coverage: float, t: float | None = None) -> None:
        """Map the splashes onto the body (torso: the game's four corners) at
        time t (Frame.t; None = drips fully run), clip them to the body
        mask, shade them and blend them into canvas in place. Nothing
        without splashes, or with a torso too narrow or folded to map.

        Size: the blob's radius is splash_r = [base, growth] torso widths,
        base + growth x coverage (coverage = 1 - health / max_health), so
        every splash swells as the player loses and a player near 0 HP is
        mostly paint. v is divided by paint_aspect, so round stays round."""
        if not splats:
            return
        cfg = self.cfg
        if (np.linalg.norm(torso[1] - torso[0]) < cfg["paint_min_torso_px"]
                or not cv2.isContourConvex(torso.astype(np.float32).reshape(-1, 1, 2))):
            return
        splashes = self.update(splats, torso)
        r0, r1 = cfg["splash_r"]
        radius = r0 + r1 * coverage
        h, w = canvas.shape[:2]
        # Each splash: its map, the sprite px -> image px affine, and the box
        # its sprite covers on screen.
        k, c = self._k, self._c
        placed = []
        for sp in splashes:
            grown = 1.0 if t is None else min(max((t - sp.t) / cfg["drip_s"], 0.0), 1.0)
            if sp.sprite is None or (grown >= 1.0 and not sp.baked):
                sp.baked = grown >= 1.0
                self._sprite(sp, with_drips=sp.baked)
            origin, cols = self._affine(sp, torso, radius)
            # Cropped-sprite px -> shape units ((px + offset - c) / k) -> image px.
            a = np.hstack([cols / k, (origin + cols @ ((np.array(sp.offset) - c) / k))[:, None]])
            sh, sw = sp.sprite.shape
            corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64)
            box = corners @ a[:, :2].T + a[:, 2]
            placed.append((sp, a, box, origin, cols, grown))
        allbox = np.concatenate([p[2] for p in placed])
        x0, y0 = np.maximum(np.floor(allbox.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(allbox.max(axis=0)).astype(int) + 1, [w, h])
        if x0 >= x1 or y0 >= y1:
            return
        # One coverage image per attacker's colour (later colours over earlier).
        layers: dict[tuple[int, int, int], np.ndarray] = {}
        for sp, a, box, origin, cols, grown in placed:
            cov = layers.setdefault(sp.color, np.zeros((y1 - y0, x1 - x0), np.uint8))
            bx0, by0 = np.maximum(np.floor(box.min(axis=0)).astype(int), [x0, y0])
            bx1, by1 = np.minimum(np.ceil(box.max(axis=0)).astype(int) + 1, [x1, y1])
            if bx0 >= bx1 or by0 >= by1:
                continue
            m = a.copy()
            m[:, 2] -= (bx0, by0)
            warped = cv2.warpAffine(sp.sprite, m, (int(bx1 - bx0), int(by1 - by0)), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            part = cov[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
            np.maximum(part, warped, out=part)
            if not sp.baked:                       # drips still running: drawn as polygons
                eased = 1.0 - (1.0 - grown) ** 2   # fast at first, then slowing
                drips = [origin + p @ cols.T - (x0, y0) for p in drip_polygons(sp.drips, eased)]
                fill_polygons(cov, drips)
        # The per-pixel work below only where there is paint: the sprites'
        # boxes reach far past their splashes.
        union = np.maximum.reduce(list(layers.values()))
        bx, by, bw, bh = cv2.boundingRect(union)
        if bw == 0 or bh == 0:
            return
        layers = {color: cov[by:by + bh, bx:bx + bw] for color, cov in layers.items()}
        x0, y0 = x0 + bx, y0 + by
        coverage_img = union[by:by + bh, bx:bx + bw].astype(np.float32) / 255.0
        shade = self._shade(canvas[y0:y0 + bh, x0:x0 + bw], coverage_img)
        if len(layers) == 1:                       # one attacker: colour x shade, per channel in C
            paint = cv2.merge([cv2.convertScaleAbs(shade, alpha=float(ch)) for ch in next(iter(layers))])
        else:
            color_img = np.zeros((bh, bw, 3), np.float32)
            for color, cov in layers.items():
                a = cov.astype(np.float32)[:, :, None] / 255.0
                color_img = color_img * (1.0 - a) + np.float32(color) * a
            # Colour where any paint is, so a soft edge is not darkened by black.
            color_img /= np.maximum(coverage_img, 1e-6)[:, :, None]
            paint = np.minimum(color_img * shade[:, :, None], 255.0).astype(np.uint8)
        mask = body_mask((bh, bw), (x0, y0), torso, motion, cfg)
        alpha = coverage_img * mask * (cfg["paint_opacity"] / 255.0)
        roi = canvas[y0:y0 + bh, x0:x0 + bw]
        # roi x (1 - alpha) + paint x alpha, per pixel, in one OpenCV call.
        roi[:] = cv2.blendLinear(paint, roi, alpha, 1.0 - alpha)

    def _shade(self, roi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Per-pixel factor for the paint: (1 - s) + s x the camera's
        brightness over its mean under the paint (s = paint_shade), the
        ratio capped at 2, so the factor is 1 - s to 1 + s. +1 keeps a black
        image at a ratio of 1."""
        s = float(self.cfg["paint_shade"])
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        under = (alpha > 0.05).astype(np.uint8)
        if s <= 0 or not under.any():
            return np.ones(gray.shape, np.float32)
        k = s / (cv2.mean(gray, mask=under)[0] + 1.0)
        # (1 - s) + s x min((gray + 1) / (mean + 1), 2), as one multiply-add.
        return np.minimum(gray.astype(np.float32) * k + ((1.0 - s) + k), 1.0 + s)


# ----------------------------------------------------- the blob in flight

def _mix(color: tuple[int, int, int], other: tuple[int, int, int], k: float) -> tuple[int, ...]:
    """color moved a fraction k towards other."""
    return tuple(int(round(c + k * (o - c))) for c, o in zip(color, other, strict=True))


def draw_blob(canvas: np.ndarray, x: float, y: float, direction: int, age: float, radius: float,
              color: tuple[int, int, int], cfg: dict) -> None:
    """The paint blob (the normal attack) at (x, y) px, flying in direction
    (+1 right, -1 left), age s after it was fired.

    - Body: blob() with wobble amplitudes blob_wobble / k for k = 2..4 whose
      phases turn at blob_wobble_hz x k turns a second, so the outline keeps
      changing like a ball of liquid; stretched blob_stretch times along the
      flight.
    - A darker rim (half way to black) so it shows on a background of its
      own colour, and a highlight towards the front top (most of the way
      to white), so it looks wet.
    - blob_trail droplets behind it, each blob_trail_gap radii further back,
      sinking blob_trail_fall x i^2 radii (gravity) and smaller each time.
    Drawn opaque and anti-aliased straight onto the canvas."""
    r = radius
    wobble = [(cfg["blob_wobble"] / k, 2 * np.pi * cfg["blob_wobble_hz"] * k * age + k) for k in (2, 3, 4)]
    body = blob(32, 1.0, wobble) * r
    body[:, 0] *= cfg["blob_stretch"]
    rim = _mix(color, (0, 0, 0), 0.5)
    for i in range(1, int(cfg["blob_trail"]) + 1):
        c = (x - direction * i * cfg["blob_trail_gap"] * r, y + cfg["blob_trail_fall"] * i * i * r)
        drop = blob(12, r * 0.45 * 0.7 ** (i - 1), [], center=c)
        cv2.fillPoly(canvas, [fixed(drop)], color, cv2.LINE_AA, SHIFT)
    pts = fixed(body + (x, y))
    cv2.fillPoly(canvas, [pts], color, cv2.LINE_AA, SHIFT)
    cv2.polylines(canvas, [pts], True, rim, max(1, int(round(r / 8))), cv2.LINE_AA, SHIFT)
    shine = blob(12, 0.22 * r, [], center=(x + direction * 0.3 * r, y - 0.4 * r))
    cv2.fillPoly(canvas, [fixed(shine)], _mix(color, (255, 255, 255), 0.75), cv2.LINE_AA, SHIFT)


def draw_burst(canvas: np.ndarray, x: float, y: float, direction: int, age: float, life: float,
               radius: float, color: tuple[int, int, int], cfg: dict, seed: int) -> None:
    """A blob stopped by a shield at (x, y), bursting into drops: burst_drops
    drops thrown back (against direction) and up, within burst_spread_rad of
    45 degrees up-back, at burst_speed radii a second, falling with
    burst_gravity radii/s^2 and shrinking to nothing over life (s; the
    game's bolt_end_s). The same seed gives the same drops in every frame."""
    rng = np.random.default_rng(seed)
    t, r = age, radius
    shrink = max(0.0, 1.0 - age / life)
    if shrink <= 0:
        return
    base = np.arctan2(-1.0, -float(direction))             # up and back, image y down
    for _ in range(rng.integers(cfg["burst_drops"][0], cfg["burst_drops"][1] + 1)):
        ang = base + rng.normal(0.0, cfg["burst_spread_rad"])
        speed = rng.uniform(*cfg["burst_speed"]) * r
        size = rng.uniform(*cfg["burst_drop_r"]) * r * shrink
        fall = 0.5 * cfg["burst_gravity"] * r * t * t
        c = (x + speed * np.cos(ang) * t, y + speed * np.sin(ang) * t + fall)
        cv2.fillPoly(canvas, [fixed(blob(10, size, [], center=c))], color, cv2.LINE_AA, SHIFT)
