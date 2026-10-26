"""Fire bolt: the look of the Task 4 normal attack (assets/beam frames).

The frames are a one-shot fire blast drawn to the right: a launch flash,
the blast in flight, then its break-up into sparks. They are split by phase:

- **launch** (the first frames): shown for one frame time after the punch;
- **flight** (the next frames): looped while the bolt flies;
- **end** (the rest): played once where the bolt ends (a hit, a shield or
  the frame edge), over ``end_s``.

Each frame is drawn with its own head (its rightmost opaque column: the
blast moves within the pack's frames) on the bolt's position, so the front of
the fire is always on the bolt and the flames trail behind it; bolts going
left are mirrored. Another team colour is a copy with every hue turned
by the same angle (``hue_shift``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ipcvgame.scene.animation import load_frames
from ipcvgame.scene.draw import blend_rgba, main_hue, shift_hue


class FireBolt:
    def __init__(self, source: str | Path, n_launch: int, n_flight: int, fps: float,
                 end_s: float, hue_shift: int = 0):
        frames = load_frames(source)
        if hue_shift:
            frames = [shift_hue(f, hue_shift) for f in frames]
        if len(frames) <= n_launch + n_flight:
            raise ValueError(f"{source}: needs launch, flight and end frames")
        self.frames, self.fps, self.end_s = frames, fps, end_s
        self.n_launch, self.n_flight = n_launch, n_flight
        self.hue = main_hue(frames)
        # Anchors: each frame's head (rightmost opaque column), and the middle
        # row of the blast in flight, shared so it doesn't jump up and down.
        self.head_x = [float(np.flatnonzero(f[:, :, 3].any(axis=0)).max()) for f in frames]
        flight = np.concatenate([f[:, :, 3] for f in frames[n_launch:n_launch + n_flight]], axis=0) > 0
        self.center_y = float((np.nonzero(flight)[0] % frames[0].shape[0]).mean())
        self._scaled: dict[int, list[np.ndarray]] = {}

    def _frames_at(self, height: int) -> list[np.ndarray]:
        if height not in self._scaled:   # once per size: sizes follow the frame size
            s = height / self.frames[0].shape[0]
            self._scaled[height] = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                                    for f in self.frames]
        return self._scaled[height]

    def flight_index(self, age: float) -> int:
        """Frame shown `age` s after the punch: launch, then flight looped."""
        n = int(age * self.fps)   # frames shown so far
        if n < self.n_launch:
            return n
        return self.n_launch + (n - self.n_launch) % self.n_flight

    def end_index(self, age: float) -> int | None:
        """Frame shown `age` s after the bolt ended, or None once it is over."""
        if age >= self.end_s:
            return None
        n_end = len(self.frames) - self.n_launch - self.n_flight
        return self.n_launch + self.n_flight + int(age / self.end_s * n_end)

    def draw(self, canvas: np.ndarray, x: float, y: float, direction: int, k: int,
             height: int) -> None:
        """Draw frame k with the blast's head on (x, y)."""
        frames = self._frames_at(max(int(height), 1))
        img, s = frames[k], frames[k].shape[0] / self.frames[0].shape[0]
        hx = self.head_x[k] * s
        if direction < 0:
            img, hx = img[:, ::-1], img.shape[1] - 1 - hx
        blend_rgba(canvas, img, x - hx, y - self.center_y * s)
