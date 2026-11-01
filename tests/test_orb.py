"""The final smash's orb (scene/orb.py; merge_log.md, "Final smash"): orb, pop, slot, body glow."""

import numpy as np

from ipcvgame.core.config import load_config
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.orb import DARK, draw_body_glow, draw_orb, draw_pop, draw_slot
from tests.test_paint import TORSO, arms_down

CFG = load_config()["render"]
RED, BLUE = player_color(1), player_color(2)


def near(img: np.ndarray, color, tol: int = 2) -> np.ndarray:
    return np.all(np.abs(img.astype(int) - color) <= tol, axis=2)


def orb(age: float) -> np.ndarray:
    canvas = np.zeros((200, 200, 3), np.uint8)
    draw_orb(canvas, 100.0, 100.0, 30.0, age, (RED, BLUE), CFG)
    return canvas


def test_orb_shows_both_team_colours_and_turns():
    a = orb(0.0)
    assert near(a, RED).sum() > 300 and near(a, BLUE).sum() > 300
    assert abs(int(near(a, RED).sum()) - int(near(a, BLUE).sum())) < 0.25 * near(a, RED).sum()
    half = orb(0.5 / CFG["orb_spin_hz"])                 # half a turn: the colours swap sides
    probe = (100, 85)                                     # (y, x) inside the ball, left of centre
    assert near(a, RED)[probe] != near(half, RED)[probe]


def test_pop_grows_and_vanishes():
    def pop(age):
        canvas = np.zeros((300, 300, 3), np.uint8)
        draw_pop(canvas, 150.0, 150.0, 20.0, age, CFG["orb_pop_s"], RED, CFG)
        return near(canvas, RED)
    early, late = pop(0.05), pop(0.3)
    ys, xs = np.nonzero(late)
    assert early.any() and np.hypot(xs - 150, ys - 150).max() > 2 * 20   # flies out past the orb
    assert not pop(CFG["orb_pop_s"]).any()


def test_slot_filled_in_team_colour_only_while_held():
    held, empty = np.zeros((40, 40, 3), np.uint8), np.zeros((40, 40, 3), np.uint8)
    draw_slot(held, (20.0, 20.0), 8.0, True, RED, 0.0)
    draw_slot(empty, (20.0, 20.0), 8.0, False, RED, 0.0)
    assert near(held[18:24, 20:24], RED).any() and not near(empty, RED).any()
    assert near(empty[20:21, 20:21], DARK).all()          # an empty slot's middle is dark


def glow(strength: float = 0.8, **over) -> tuple[np.ndarray, np.ndarray]:
    before = np.full((900, 500, 3), 90, np.uint8)
    canvas = before.copy()
    draw_body_glow(canvas, TORSO, arms_down(), None, BLUE, strength, {**CFG, **over})
    return before, canvas


def test_body_glow_is_a_halo_brightest_just_outside_the_body():
    before, canvas = glow()
    change = np.abs(canvas.astype(int) - before).sum(axis=2)
    # Torso 150-250 px wide at y 300-470; the arms hang 15 px outside it.
    inside = change[380, 200]
    near_edge = change[380, 150 - 15 - 25]                # just outside the arm
    far = change[380, 20]                                 # 2 shoulder widths away
    assert near_edge > inside > 0 and far == 0
    tint = canvas[380, 110].astype(int) - 90
    assert tint[0] > 0 and tint[0] > tint[2]              # towards the (BGR) blue


def test_no_glow_at_zero_strength():
    before, canvas = glow(0.0)
    assert (canvas == before).all()


def test_body_glow_follows_the_real_outline_of_its_player_only():
    """A segmenter mask with a wide coat around the keypoint body (torso
    150-250 px) and another person far to the right: the glow tints the coat
    and its halo starts at the coat's edge, not at the keypoint capsules,
    and the other person does not glow."""
    people = np.zeros((900, 500), np.float32)
    people[280:700, 100:300] = 1.0                        # the coat: 35 px past the arm capsules
    people[280:700, 430:480] = 1.0                        # someone else, 1.8 shoulder widths away
    before = np.full((900, 500, 3), 90, np.uint8)
    with_outline, keypoints_only = before.copy(), before.copy()
    draw_body_glow(with_outline, TORSO, arms_down(), None, BLUE, 0.8, CFG, people, 1.0)
    draw_body_glow(keypoints_only, TORSO, arms_down(), None, BLUE, 0.8, CFG)
    change = lambda img: np.abs(img.astype(int) - before).sum(axis=2)   # noqa: E731
    a, b = change(with_outline), change(keypoints_only)
    assert a[500, 105] < a[500, 85]                        # inside the coat: the lighter tint; past it: halo
    assert a[500, 105] > 0 and b[500, 105] > a[500, 105]   # keypoints only: halo already there
    assert a[500, 455] == 0                                # the other person: no glow
