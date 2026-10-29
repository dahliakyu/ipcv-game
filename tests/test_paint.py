"""Paint on the players' bodies (scene/paint.py, extra_features.md feature 3)."""

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Motion
from ipcvgame.interaction.game import PaintSplat
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.paint import BodyPaint, body_mask

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
    canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)])
    ys, xs = np.nonzero(blue(canvas))
    assert xs.mean() == pytest.approx(200, abs=1.5) and ys.mean() == pytest.approx(385, abs=1.5)
    # About round on the body: the layer's aspect matches this torso's.
    assert (xs.max() - xs.min()) == pytest.approx(ys.max() - ys.min(), abs=3)


def test_paint_grows_as_health_drops():
    fresh, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], coverage=0.0)
    beaten, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], coverage=1.0)
    assert blue(beaten).sum() > 2 * blue(fresh).sum()


def test_paint_follows_the_torso():
    moved = TORSO + [120, 40]
    canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], torso=moved)
    ys, xs = np.nonzero(blue(canvas))
    assert xs.mean() == pytest.approx(320, abs=1.5) and ys.mean() == pytest.approx(425, abs=1.5)


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
    canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], image=stripes)
    centre = canvas[378:393, 193:208].astype(int)
    lum = centre.sum(axis=2)
    assert lum.max() > lum.min() + 50                          # the folds show through
    assert lum.mean() == pytest.approx(sum(BLUE), rel=0.15)    # but the paint is not darkened
    flat, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], image=stripes, paint_shade=0.0)
    assert blue(flat[378:393, 193:208]).all()


def test_shapes_made_only_on_change():
    s = [PaintSplat(0.5, 0.5, owner_id=2, seed=1)]
    _, layer = paint(s)
    canvas = np.zeros((900, 500, 3), np.uint8)
    layer.draw(canvas, TORSO + 30, arms_down(TORSO + 30), s, 0.0)      # moved: same layer
    assert layer.builds == 1
    layer.draw(canvas, TORSO, arms_down(), s + [PaintSplat(0.3, 0.2, owner_id=2, seed=2)], 0.0)
    layer.draw(canvas, TORSO, arms_down(), s + [PaintSplat(0.3, 0.2, owner_id=2, seed=2)], 0.1)
    assert layer.builds == 3


def test_no_paint_side_on_or_without_splashes():
    side_on = TORSO.copy()
    side_on[[1, 2], 0] = side_on[[0, 3], 0] + 5                  # shoulders 5 px apart
    canvas, _ = paint([PaintSplat(0.5, 0.5, owner_id=2)], torso=side_on)
    assert not blue(canvas).any()
    canvas, layer = paint([])
    assert not blue(canvas).any() and layer.builds == 0
    edge = TORSO + [-220, 0]                                     # partly out of the frame: no error
    paint([PaintSplat(0.1, 0.5, owner_id=2)], torso=edge)
