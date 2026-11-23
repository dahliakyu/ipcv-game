"""load_frames (scene/animation.py): frames from a folder or a glob pattern."""

import cv2
import numpy as np
import pytest

from ipcvgame.scene.animation import load_frames


def write(tmp_path, name, k):
    """A 4 x 4 frame with k opaque pixels in its top row, to tell frames apart."""
    f = np.zeros((4, 4, 4), np.uint8)
    f[0, :k] = 255
    cv2.imwrite(str(tmp_path / name), f)


def test_folder_frames_load_in_natural_order(tmp_path):
    for k, name in ((1, "x-1.png"), (2, "x-2.png"), (4, "x-10.png")):
        write(tmp_path, name, k)
    assert [int((f[0, :, 3] > 0).sum()) for f in load_frames(tmp_path)] == [1, 2, 4]


def test_frames_load_from_a_glob_in_natural_order(tmp_path):
    for k in (1, 2, 10):
        write(tmp_path, f"m-{k}.png", min(k, 4))
    write(tmp_path, "other.png", 3)
    frames = load_frames(tmp_path / "m-*.png")
    assert [int((f[0, :, 3] > 0).sum()) for f in frames] == [1, 2, 4]   # 10 last, no other


def test_missing_or_mixed_frames_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frames(tmp_path)
    write(tmp_path, "a.png", 1)
    cv2.imwrite(str(tmp_path / "b.png"), np.zeros((5, 5, 4), np.uint8))
    with pytest.raises(ValueError):
        load_frames(tmp_path)
