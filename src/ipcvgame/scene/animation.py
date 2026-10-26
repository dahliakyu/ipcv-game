"""Sprite frames: loading animation frames from a folder or a glob pattern.

``load_frames()`` reads the PNG files of a folder, or those matching a glob
pattern, in natural order (``x-2`` before ``x-10``); they must all be the
same size and have transparency.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from ipcvgame.core.config import REPO_ROOT


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def load_frames(source: str | Path) -> list[np.ndarray]:
    """BGRA frames from a folder (all its PNGs) or a glob pattern, in natural
    order; a relative path is taken from the repo root."""
    source = Path(source)
    if not source.is_absolute():
        source = REPO_ROOT / source
    files = source.glob("*.png") if source.is_dir() else source.parent.glob(source.name)
    frames = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in sorted(files, key=_natural_key)]
    if not frames:
        raise FileNotFoundError(f"no PNG frames in {source}")
    for f in frames:
        if f is None or f.ndim != 3 or f.shape[2] != 4 or f.shape != frames[0].shape:
            raise ValueError(f"{source}: frames must be RGBA and all the same size")
    return frames
