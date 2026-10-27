"""Fire bolt (scene/fire_bolt.py): the normal attack's look, by phase."""

import cv2
import numpy as np
import pytest

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.scene.draw import hue_of, player_color
from ipcvgame.scene.fire_bolt import FireBolt

ORANGE = (40, 140, 255, 255)


@pytest.fixture
def frames_dir(tmp_path):
    """6 frames of 10 x 30 px, like the pack: a blast whose head moves right
    (heads at columns 9, 19, 24, 27, 28, 29), 4 px thick around row 5."""
    for k, head in enumerate([9, 19, 24, 27, 28, 29], start=1):
        f = np.zeros((10, 30, 4), np.uint8)
        f[3:7, head - 8:head + 1] = ORANGE
        cv2.imwrite(str(tmp_path / f"f_{k:02d}.png"), f)
    return tmp_path


@pytest.fixture
def fire(frames_dir):
    return FireBolt(frames_dir, n_launch=1, n_flight=2, fps=10, end_s=0.3)


def test_phases(fire):
    assert [fire.flight_index(t) for t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)] == [0, 0, 1, 2, 1, 2]
    assert [fire.end_index(t) for t in (0.0, 0.1, 0.2, 0.29, 0.3)] == [3, 4, 5, 5, None]


def test_each_frame_has_its_head_on_the_bolt(fire):
    for k in range(6):
        canvas = np.zeros((40, 200, 3), np.uint8)
        fire.draw(canvas, 100, 20, 1, k, height=10)
        cols = np.flatnonzero(canvas.any(axis=(0, 2)))
        assert cols.max() == 100 and cols.min() == 92          # head on x, flames behind


def test_left_bolts_are_mirrored(fire):
    canvas = np.zeros((40, 200, 3), np.uint8)
    fire.draw(canvas, 100, 20, -1, 1, height=10)
    cols = np.flatnonzero(canvas.any(axis=(0, 2)))
    assert cols.min() == 100 and cols.max() == 108


def test_blast_is_centred_on_the_bolt_height(fire):
    canvas = np.zeros((40, 200, 3), np.uint8)
    fire.draw(canvas, 100, 20, 1, 1, height=20)                # 2x: rows 6-13 of the frame
    rows = np.flatnonzero(canvas.any(axis=(1, 2)))
    assert rows.min() == 17 and rows.max() == 24


def test_hue_shift_gives_the_team_colour(frames_dir):
    base = FireBolt(frames_dir, 1, 2, 10, 0.3)
    blue = FireBolt(frames_dir, 1, 2, 10, 0.3, hue_shift=hue_of(player_color(2)) - base.hue)
    assert abs(blue.hue - hue_of(player_color(2))) <= 1


def test_too_few_frames_raise(tmp_path):
    for k in range(3):
        cv2.imwrite(str(tmp_path / f"f{k}.png"), np.zeros((4, 4, 4), np.uint8))
    with pytest.raises(ValueError):
        FireBolt(tmp_path, n_launch=1, n_flight=2, fps=10, end_s=0.3)


def game_renderer():
    """The renderer with the real config and the stub face effects."""
    from ipcvgame.face.effects import FaceEffects
    from ipcvgame.scene.renderer import Renderer
    full = load_config()
    return Renderer(full["render"], FaceEffects({**full["face"], "backend": "stub"}))


def test_renderer_gives_each_team_its_own_bolt_colour():
    from ipcvgame.interaction.game import Game
    full = load_config()
    cfg = full["render"]
    if not (REPO_ROOT / cfg["bolt_sprite"]).exists():
        pytest.skip("bolt frames not in this checkout")
    r, g = game_renderer(), Game(full["game"])
    assert r.bolt_effect(1, g) is r.bolt_effect(1, g) and len(r.bolt_effect(1, g).frames) == 6
    for pid in (1, 2):
        if pid != cfg["bolt_effect_owner"]:
            assert abs(r.bolt_effect(pid, g).hue - hue_of(player_color(pid))) <= 1
