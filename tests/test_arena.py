"""The room reacting to the fight (scene/arena.py): wall paint, lighting, shake."""

import numpy as np
import pytest

from ipcvgame.core.config import load_config
from ipcvgame.interaction.game import BoltEnd, Game, Hit, PaintBeam
from ipcvgame.scene.arena import Arena, person_mask
from ipcvgame.scene.draw import player_color
from tests.test_game import H, W
from tests.test_game import frame_state as game_frame_state

CFG = load_config()
# The keypoint person mask: these drawn frames have no real people for the
# Selfie Segmenter (test_segmenter_mask_is_used covers that path).
CFG["render"] = {**CFG["render"], "arena_segmenter": None}
GREY = 128


def frame_state(t: float):
    """tests.test_game's bodies, with only the keypoints it places valid (the
    rest sit at (0, 0) and would draw limbs to the corner in the mask)."""
    state = game_frame_state(t)
    for m in state.motion.values():
        m.valid = m.kpts.any(axis=1)
    return state


@pytest.fixture
def game() -> Game:
    g = Game({**CFG["game"], "intro": False, "act_facing_camera": False})
    g.update(frame_state(0.0))          # players placed, torsos known
    return g


def step(arena: Arena, game: Game, t: float) -> np.ndarray:
    """One frame at full size (scale 1) on a grey room; the drawn image."""
    state = frame_state(t)
    arena.update(state, game, (H, W))
    img = np.full((H, W, 3), GREY, np.uint8)
    arena.draw(img, state, game)
    return img


def close_to(px: np.ndarray, color, opacity: float) -> bool:
    want = np.float32(color) * opacity + GREY * (1 - opacity)
    return bool(np.all(np.abs(px.astype(np.float32) - want) < 12))


def test_blob_off_the_edge_paints_the_side_wall(game):
    arena = Arena(CFG["render"])
    game.bolt_ends.append(BoltEnd(float(W), 450.0, 1, owner_id=1, reason="edge"))
    img = step(arena, game, 0.0)
    r = CFG["render"]["arena_edge_r_h"] * H
    cx = int(W - CFG["render"]["arena_edge_inset_r"] * r)
    assert close_to(img[450, cx], player_color(1), CFG["render"]["arena_wall_opacity"])
    assert (img[450, 800] == GREY).all()            # P2's torso: the wall is behind them
    assert arena.share[1] > 0 and arena.log_fields()["render_wall_p1"] > 0
    step(arena, game, 0.05)                         # the same bolt end is painted once
    assert arena.stamps == 1


def test_hit_sprays_behind_the_target(game):
    """P2's torso is centred at x 800, 100 px shoulder to shoulder: the spray
    is centred arena_behind_sw shoulder widths past it, half behind the body."""
    arena = Arena(CFG["render"])
    game.bolt_ends.append(BoltEnd(700.0, 450.0, 1, owner_id=1, reason="hit"))
    img = step(arena, game, 0.0)
    x = int(800 + CFG["render"]["arena_behind_sw"] * 100)
    assert arena._wall_a[450, x] > 200             # on the wall...
    assert (img[450, 800] == GREY).all()            # ...hidden where the body is
    assert close_to(img[450, x + 15], player_color(1), CFG["render"]["arena_wall_opacity"])


def test_beam_dims_the_room_and_leaves_no_stripe(game):
    arena = Arena(CFG["render"])
    beam = PaintBeam(start_x=300.0, x=300.0, y=450.0, direction=1, owner_id=1)
    game.beams.append(beam)
    for t in (0.0, 0.1, 0.2, 0.3):                  # charging: the light goes down
        img = step(arena, game, t)
    assert arena._dim == 1.0 and arena.stamps == 0
    assert img[100, 500].mean() < GREY * (1 - CFG["render"]["arena_dim"]) + 25
    assert (img[450, 200] == GREY).all()            # P1 keeps their light
    beam.age, beam.x = game.beam_charge_s, 650.0    # firing: the tip moved
    step(arena, game, 0.35)
    assert arena.stamps == 0 and arena._box is None   # nothing on the wall
    game.beams.clear()
    for t in (0.4, 0.5, 0.6, 0.7):
        img = step(arena, game, t)
    assert arena._dim == 0.0
    assert (img[100, 500] == GREY).all() and (img[450, 500] == GREY).all()   # full light, no stripe


def test_wall_kept_for_the_result_and_cleaned_for_a_new_round(game):
    arena = Arena(CFG["render"])
    game.bolt_ends.append(BoltEnd(0.0, 450.0, -1, owner_id=2, reason="edge"))
    step(arena, game, 0.0)
    game.phase = "result"
    step(arena, game, 0.1)
    assert arena.share
    game.phase = "countdown"
    step(arena, game, 0.2)
    assert not arena.share and arena._box is None


def test_hit_shakes_along_its_direction(game):
    arena = Arena(CFG["render"])
    game.hits.append(Hit(target_id=2, owner_id=1, direction=-1, damage=10))
    step(arena, game, 0.0)
    dx, _ = arena.shake_offset(0.02)
    assert dx < 0                                   # towards where the hit went
    assert arena.shake_offset(CFG["render"]["arena_shake_s"] + 0.01) == (0.0, 0.0)
    canvas = np.zeros((H, W, 3), np.uint8)
    assert arena.shake(canvas, 1.0) is canvas       # nothing to shake: no copy


def test_person_mask_covers_bodies_and_scales(game):
    state = frame_state(0.0)
    mask = person_mask((H // 2, W // 2), 0.5, state, game, Arena(CFG["render"])._mask_cfg, CFG["render"])
    assert mask.shape == (H // 2, W // 2)
    assert mask[225, 100] > 0.99 and mask[225, 400] > 0.99    # both torsos, at half size
    assert mask[50, 250] == 0.0                               # the room between them


def test_segmenter_mask_is_used(game):
    """With the Selfie Segmenter, its mask keeps the wall paint off the
    people: here a fake one that marks the right half of the image."""
    arena = Arena(CFG["render"])
    marked = np.zeros((H, W), np.float32)
    marked[:, W // 2:] = 1.0
    arena._segmenter = lambda small: marked
    game.bolt_ends.append(BoltEnd(float(W), 450.0, 1, owner_id=1, reason="edge"))
    img = step(arena, game, 0.0)
    assert arena._wall_a[450, W - 40] > 200 and (img[450, W - 40] == GREY).all()


def test_missing_segmenter_model_falls_back_to_keypoints(game, capsys):
    arena = Arena({**CFG["render"], "arena_segmenter": "models/missing.tflite"})
    state = frame_state(0.0)
    arena.update(state, game, (H, W))
    mask = arena.person(np.zeros((H, W, 3), np.uint8), state, game)
    assert mask[450, 200] > 0.99 and "keypoint mask" in capsys.readouterr().out


def test_segmenter_returns_its_own_copy():
    """The mask must not point into MediaPipe's buffer: a later call must
    not change an earlier result. Skipped without the model file."""
    from ipcvgame.core.config import REPO_ROOT
    from ipcvgame.scene.segment import PersonSegmenter
    path = load_config()["render"]["arena_segmenter"]
    if not path or not (REPO_ROOT / path).is_file():
        pytest.skip("selfie segmenter model not downloaded")
    seg = PersonSegmenter(path, (0.3, 0.7))
    rng = np.random.default_rng(0)
    first = seg(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
    kept = first.copy()
    for _ in range(3):
        seg(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
    assert first.shape == (180, 320) and first.dtype == np.float32
    assert (first == kept).all() and 0.0 <= first.min() and first.max() <= 1.0
