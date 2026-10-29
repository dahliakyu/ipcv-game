"""The final smash's orb (scene/orb.py; extra_features.md §1): part, pop, slots, body glow."""

import numpy as np

from ipcvgame.core.config import load_config
from ipcvgame.scene.draw import player_color
from ipcvgame.scene.orb import DARK, draw_body_glow, draw_part, draw_pop, draw_slots
from tests.test_paint import TORSO, arms_down

CFG = load_config()["render"]
RED, BLUE = player_color(1), player_color(2)


def near(img: np.ndarray, color, tol: int = 2) -> np.ndarray:
    return np.all(np.abs(img.astype(int) - color) <= tol, axis=2)


def part(age: float) -> np.ndarray:
    canvas = np.zeros((200, 200, 3), np.uint8)
    draw_part(canvas, 100.0, 100.0, 30.0, age, (RED, BLUE), CFG)
    return canvas


def test_part_shows_both_team_colours_and_turns():
    a = part(0.0)
    assert near(a, RED).sum() > 300 and near(a, BLUE).sum() > 300
    assert abs(int(near(a, RED).sum()) - int(near(a, BLUE).sum())) < 0.25 * near(a, RED).sum()
    half = part(0.5 / CFG["orb_spin_hz"])                 # half a turn: the colours swap sides
    probe = (100, 85)                                     # (y, x) inside the ball, left of centre
    assert near(a, RED)[probe] != near(half, RED)[probe]


def test_pop_grows_and_vanishes():
    def pop(age):
        canvas = np.zeros((300, 300, 3), np.uint8)
        draw_pop(canvas, 150.0, 150.0, 20.0, age, CFG["orb_pop_s"], RED, CFG)
        return near(canvas, RED)
    early, late = pop(0.05), pop(0.3)
    ys, xs = np.nonzero(late)
    assert early.any() and np.hypot(xs - 150, ys - 150).max() > 2 * 20   # flies out past the part
    assert not pop(CFG["orb_pop_s"]).any()


def test_slots_filled_in_team_colour_in_order():
    canvas = np.zeros((40, 120, 3), np.uint8)
    centres = [(15.0 + 25 * i, 20.0) for i in range(4)]
    draw_slots(canvas, centres, 8.0, 2, RED, 0.0)
    filled = [near(canvas[12:28, int(x) - 3:int(x) + 4], RED).any() for x, _ in centres]
    assert filled == [True, True, False, False]
    assert near(canvas[20:21, 90:91], DARK).all()         # the third slot's middle is dark


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
