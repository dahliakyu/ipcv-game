"""Fire ring (scene/fire_ring.py): the shield's look, and where the game draws it."""

import cv2
import numpy as np
import pytest

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.scene.draw import hue_of, player_color
from ipcvgame.scene.fire_ring import FireRing

ORANGE = (40, 140, 255, 255)


@pytest.fixture
def frames_dir(tmp_path):
    """3 forming frames (a growing dot), 2 ring frames (a 20 px ring around
    (25, 15) in 50 x 30 px), then 2 copies of the ring frames that must be
    ignored."""
    for k in range(1, 8):
        f = np.zeros((30, 50, 4), np.uint8)
        if k <= 3:
            cv2.circle(f, (25, 15), k, ORANGE, -1)
        else:
            cv2.circle(f, (25, 15), 10, ORANGE, 2)
        cv2.imwrite(str(tmp_path / f"FX{k:03d}.png"), f)
    return tmp_path


@pytest.fixture
def ring(frames_dir):
    return FireRing(frames_dir, n_form=3, n_loop=2, form_s=0.3, fps=10)


def test_forming_then_loop_and_copies_unused(ring):
    assert len(ring.frames) == 5
    assert [ring.index(t) for t in (0.0, 0.1, 0.25, 0.3, 0.4, 0.5)] == [0, 1, 2, 3, 4, 3]


def test_ring_is_measured(ring):
    assert ring.center == pytest.approx((25, 15), abs=1)
    assert ring.ring_px == pytest.approx(23, abs=2)            # the 2 px circle of radius 10


def test_draw_centres_and_sizes_the_ring(ring):
    canvas = np.zeros((200, 200, 3), np.uint8)
    ring.draw(canvas, (100, 100), diameter=ring.ring_px * 4, age=0.4)
    ys, xs = np.nonzero(canvas.any(axis=2))
    assert (xs.min() + xs.max()) / 2 == pytest.approx(100, abs=3)
    assert xs.max() - xs.min() == pytest.approx(ring.ring_px * 4, abs=6)


def test_fade_lowers_opacity(ring):
    full, faded = np.zeros((100, 100, 3), np.uint8), np.zeros((100, 100, 3), np.uint8)
    ring.draw(full, (50, 50), 40, 0.4)
    ring.draw(faded, (50, 50), 40, 0.4, alpha=0.5)
    assert faded.max() < full.max() and faded.max() > 0


def test_hue_shift_and_too_few_frames(frames_dir, tmp_path_factory):
    base = FireRing(frames_dir, 3, 2, 0.3, 10)
    blue = FireRing(frames_dir, 3, 2, 0.3, 10, hue_shift=hue_of(player_color(2)) - base.hue)
    assert abs(blue.hue - hue_of(player_color(2))) <= 1
    with pytest.raises(ValueError):
        FireRing(frames_dir, n_form=5, n_loop=5, form_s=0.3, fps=10)


def test_game_shield_ring_per_team():
    from ipcvgame.interaction.game import Game
    cfg = load_config()["game"]
    if not (REPO_ROOT / cfg["shield_sprite"]).exists():
        pytest.skip("shield frames not in this checkout")
    g = Game(cfg)
    assert g._shield_effect(1) is g._shield_effect(1)
    assert len(g._shield_effect(1).frames) == cfg["shield_form_frames"] + cfg["shield_loop_frames"]
    assert abs(g._shield_effect(2).hue - hue_of(player_color(2))) <= 1
