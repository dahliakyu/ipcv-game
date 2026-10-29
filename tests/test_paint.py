"""Paint on the players' bodies (scene/paint.py, extra_features.md feature 3)."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Motion
from ipcvgame.interaction.game import PaintSplat
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.paint import BodyPaint, body_mask, draw_blob, draw_burst, splash_shapes

# Fully opaque, so painted pixels have the exact team colour.
CFG = {**load_config()["render"], "paint_opacity": 1.0}
BLUE = player_color(2)
# A torso 100 px wide, shoulders at y 300, hips at y 470 (aspect 1.7):
# R_SHOULDER, L_SHOULDER, L_HIP, R_HIP.
TORSO = np.array([[150.0, 300], [250, 300], [250, 470], [150, 470]])


def arms_down(torso=TORSO, valid=True) -> Motion:
    """Arms hanging beside the torso (elbows and wrists 15 px outside the
    shoulders), legs straight down from the hips."""
    k = np.zeros((sk.K, 2))
    k[sk.R_SHOULDER], k[sk.L_SHOULDER] = torso[0], torso[1]
    k[sk.L_HIP], k[sk.R_HIP] = torso[2], torso[3]
    k[sk.R_ELBOW], k[sk.R_WRIST] = torso[0] + [-15, 120], torso[0] + [-15, 230]
    k[sk.L_ELBOW], k[sk.L_WRIST] = torso[1] + [15, 120], torso[1] + [15, 230]
    k[sk.L_KNEE], k[sk.L_ANKLE] = torso[2] + [0, 200], torso[2] + [0, 400]
    k[sk.R_KNEE], k[sk.R_ANKLE] = torso[3] + [0, 200], torso[3] + [0, 400]
    v = np.ones(sk.K, bool)
    if not valid:
        v[[sk.R_ELBOW, sk.R_WRIST, sk.L_ELBOW, sk.L_WRIST]] = False
    return Motion(kpts=k, vel=np.zeros_like(k), valid=v, signals={})


def paint(splats, torso=TORSO, motion=None, image=None, coverage=0.0, **over):
    layer = BodyPaint({**CFG, **over})
    canvas = np.full((900, 500, 3), 90, np.uint8) if image is None else image.copy()
    layer.draw(canvas, torso, arms_down(torso) if motion is None else motion, splats, coverage)
    return canvas, layer


def blue(canvas):
    return np.all(np.abs(canvas.astype(int) - BLUE) <= 2, axis=2)


def test_splash_lands_at_its_torso_point_in_the_attackers_colour():
    canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=3)])
    assert blue(canvas)[385, 200]                                 # the blob covers its point
    ys, xs = np.nonzero(blue(canvas))
    # Spray around it: within ~half a torso width.
    assert abs(xs.mean() - 200) < 50 and abs(ys.mean() - 385) < 50


def test_spray_flies_on_in_the_attacks_direction():
    shift = {}
    for direction in (1, -1):
        dx = []
        for seed in range(12):
            canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, direction=direction, seed=seed)])
            dx.append(np.nonzero(blue(canvas))[1].mean() - 200)
        shift[direction] = np.mean(dx)
    # Blob radius 10 px here: the spray leans ~a quarter radius onward.
    assert shift[1] > 1.5 and shift[-1] < -1.5


def test_every_splash_looks_different_but_stays_the_same():
    a, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=1)])
    b, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=2)])
    again, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=1)])
    assert (blue(a) != blue(b)).sum() > 100
    np.testing.assert_array_equal(a, again)


def test_shape_parts_within_their_ranges():
    polys, drips = splash_shapes(7, CFG)
    lo, hi = CFG["splash_spikes"]
    d_lo, d_hi = CFG["splash_drops"]
    # The blob, two polygons per spike (wedge, tip drop), one per droplet.
    assert 1 + 2 * lo + d_lo <= len(polys) <= 1 + 2 * hi + d_hi
    assert CFG["drips"][0] <= len(drips) <= CFG["drips"][1]
    sizes = [np.ptp(p[:, 0]) / 2 for p in polys[1 + 2 * lo:]]     # droplets (and some tip drops)
    assert max(sizes) < 1.5 * CFG["drop_r"][1] + 0.4


def test_drips_run_down_then_stop():
    s = [PaintSplat(0.5, 0.3, owner_id=2, seed=4, t=10.0)]
    layer = BodyPaint(CFG)

    def lowest(t):
        canvas = np.full((900, 500, 3), 90, np.uint8)
        layer.draw(canvas, TORSO, arms_down(), s, 0.0, t)
        return np.nonzero(blue(canvas[:, 150:250]))[0].max()      # over the torso
    at_hit, half, done, later = lowest(10.0), lowest(10.0 + CFG["drip_s"] / 2), lowest(11.0), lowest(15.0)
    assert at_hit < half < done == later

def test_paint_grows_as_health_drops():
    fresh, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], coverage=0.0)
    beaten, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], coverage=1.0)
    assert blue(beaten).sum() > 2 * blue(fresh).sum()


def test_paint_follows_the_torso():
    still, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=5)])
    moved, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=5)], torso=TORSO + [120, 40])
    np.testing.assert_array_equal(np.roll(blue(still), (40, 120), axis=(0, 1)), blue(moved))

def test_paint_runs_onto_the_arm_but_not_the_background():
    beside = [PaintSplat(-0.25, 0.3, owner_id=2)]            # just outside the left shoulder
    on_arm, _ = paint(beside)
    no_arm, _ = paint(beside, motion=arms_down(valid=False))
    assert blue(on_arm)[:, :130].any()                        # on the arm, past the torso's outline
    assert not blue(no_arm)[:, :130].any()                    # no arm seen: clipped at the torso
    far, _ = paint([PaintSplat(1.8, 1.25, owner_id=2)])       # beside the hips, past the arm: nothing there
    assert not blue(far).any()


def test_body_mask_covers_torso_and_limbs():
    m = body_mask((900, 500), (0, 0), TORSO, arms_down(), CFG)
    assert m[385, 200] == 255                                 # torso
    assert m[300 + 175, 135] == 255                           # forearm
    assert m[385, 60] == 0 and m[100, 200] == 0               # background, above the head
    assert m[470 + 100, 245] == 255 and m[470 + 100, 200] == 0   # a thigh, the gap between the legs
    shifted = body_mask((100, 100), (150, 330), TORSO, None, CFG)    # a region, no limbs
    assert shifted[55, 50] == 255


def test_shading_shows_folds_but_keeps_paint_bright_on_dark_cloth():
    stripes = np.full((900, 500, 3), 30, np.uint8)
    stripes[::8] = 60                                          # folds on a dark shirt
    canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=3)], image=stripes)
    centre = canvas[381:390, 196:205].astype(int)
    lum = centre.sum(axis=2)
    assert lum.max() > lum.min() + 50                          # the folds show through
    assert lum.mean() == pytest.approx(sum(BLUE), rel=0.15)    # but the paint is not darkened
    flat, _ = paint([PaintSplat(0.5, 0.5, owner_id=2, seed=3)], image=stripes, paint_shade=0.0)
    assert blue(flat[381:390, 196:205]).all()


def test_shapes_made_once_per_splash():
    s = [PaintSplat(0.5, 0.5, owner_id=2, seed=1)]
    _, layer = paint(s)
    canvas = np.zeros((900, 500, 3), np.uint8)
    layer.draw(canvas, TORSO + 30, arms_down(TORSO + 30), s, 0.5)      # moved, more damage
    assert layer.builds == 1
    two = s + [PaintSplat(0.3, 0.2, owner_id=2, seed=2)]
    layer.draw(canvas, TORSO, arms_down(), two, 0.5)
    layer.draw(canvas, TORSO, arms_down(), two, 0.6)
    assert layer.builds == 2
    layer.draw(canvas, TORSO, arms_down(), [], 0.0)                     # nothing to draw
    layer.draw(canvas, TORSO, arms_down(), two[1:], 0.0)                # a new round: forgotten
    assert len(layer._splashes) == 1


def test_overlapping_parts_leave_no_holes():
    """fillPoly with several overlapping polygons at once leaves holes."""
    splats = [PaintSplat(0.5 + 0.05 * s, 0.5, owner_id=2, seed=s) for s in range(4)]
    together, _ = paint(splats)
    union = np.any([blue(paint([s])[0]) for s in splats], axis=0)
    assert not (union & ~blue(together)).any()

def test_no_paint_side_on_or_without_splashes():
    side_on = TORSO.copy()
    side_on[[1, 2], 0] = side_on[[0, 3], 0] + 5                  # shoulders 5 px apart
    canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], torso=side_on)
    assert not blue(canvas).any()
    canvas, layer = paint([])
    assert not blue(canvas).any() and layer.builds == 0
    edge = TORSO + [-220, 0]                                     # partly out of the frame: no error
    paint([PaintSplat(0.1, 0.5, owner_id=2)], torso=edge)


# --- The paint blob in flight and its burst (step 4) ------------------------

RED = player_color(1)


def blob_at(age, direction=1):
    canvas = np.zeros((400, 600, 3), np.uint8)
    draw_blob(canvas, 300.0, 200.0, direction, age, 26.0, RED, CFG)
    return canvas


def test_blob_is_a_wet_ball_in_the_team_colour():
    c = blob_at(0.0)
    assert tuple(c[200, 300]) == RED                                  # body
    lighter = c[185:195, 300:315].reshape(-1, 3).sum(axis=1).max()
    assert lighter > sum(RED) + 150                                    # highlight, front top
    assert c[150:250, 360:].sum() == 0                                # nothing ahead of it


def test_blob_wobbles_and_trails_droplets_behind():
    a, b = blob_at(0.0), blob_at(0.2)
    assert (np.any(a != b, axis=2)).sum() > 50                         # the outline moves
    red = np.all(a == RED, axis=2)
    assert red[200:215, 240:270].any()                                 # droplets behind (to the left)
    left = np.all(blob_at(0.0, direction=-1) == RED, axis=2)
    assert left[200:215, 330:360].any() and not left[200:215, 240:262].any()


def test_burst_drops_fly_back_up_then_fall_and_vanish():
    def drops(age):
        canvas = np.zeros((400, 600, 3), np.uint8)
        draw_burst(canvas, 300.0, 200.0, 1, age, 0.5, 26.0, RED, CFG, seed=3)
        ys, xs = np.nonzero(np.all(canvas == RED, axis=2))
        return xs, ys
    xs, ys = drops(0.1)
    assert xs.mean() < 300 and ys.mean() < 200                         # thrown back (left) and up
    later_x, later_y = drops(0.4)
    assert later_x.mean() < xs.mean() and later_y.mean() > ys.mean()    # further back, falling
    assert len(drops(0.5)[0]) == 0                                      # gone at the end
    np.testing.assert_array_equal(drops(0.2)[0], drops(0.2)[0])         # the same drops each frame
