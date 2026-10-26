"""Fire ring: the look of the Task 4 shield (assets/shield/FX019 frames).

The pack's frames are a ring of fire that forms and then burns: whirling
arcs close into a ring (the forming frames), then the burning ring loops
(the loop frames; the pack repeats them once more, those copies are not
used). Forming plays once over ``form_s``, faster than the pack's preview so
the ring is up while the shield already blocks; the ring then loops at
``fps``. The ring is drawn centred on the player's chest and sized to their
body, and fades out with the shield (``alpha``). Another team colour is a
copy with every hue turned by the same angle (``hue_shift``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ipcvgame.scene.animation import load_frames
from ipcvgame.scene.draw import blend_rgba, main_hue, shift_hue


class FireRing:
    def __init__(self, source: str | Path, n_form: int, n_loop: int, form_s: float, fps: float,
                 hue_shift: int = 0):
        frames = load_frames(source)
        if len(frames) < n_form + n_loop:
            raise ValueError(f"{source}: needs {n_form} forming and {n_loop} loop frames")
        frames = frames[:n_form + n_loop]
        if hue_shift:
            frames = [shift_hue(f, hue_shift) for f in frames]
        self.frames, self.n_form, self.form_s, self.fps = frames, n_form, form_s, fps
        self.hue = main_hue(frames)
        # The ring in the loop frames: a fixed centre (middle of their union, so
        # it does not jitter) and its size (the median height of each frame's
        # flames, which flicker outside the ring now and then).
        loop = np.stack([f[:, :, 3] > 0 for f in frames[n_form:]])
        ys, xs = np.nonzero(loop.any(axis=0))
        self.center = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)
        heights = [np.ptp(np.flatnonzero(m.any(axis=1))) + 1 for m in loop]
        self.ring_px = float(np.median(heights))
        self._scaled: dict[int, list[np.ndarray]] = {}

    def index(self, age: float) -> int:
        """Frame shown `age` s after the shield went up."""
        if age < self.form_s:
            return int(age / self.form_s * self.n_form)
        n_loop = len(self.frames) - self.n_form
        return self.n_form + int((age - self.form_s) * self.fps) % n_loop

    def draw(self, canvas: np.ndarray, center: tuple[float, float], diameter: float, age: float,
             alpha: float = 1.0) -> None:
        """Draw the ring for `age`, `diameter` px across, centred on `center`."""
        height = max(int(round(self.frames[0].shape[0] * diameter / self.ring_px)), 1)
        if height not in self._scaled:   # sizes vary with distance: keep a few
            if len(self._scaled) > 8:
                self._scaled.clear()
            s = height / self.frames[0].shape[0]
            self._scaled[height] = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                                    for f in self.frames]
        img = self._scaled[height][self.index(age)]
        s = img.shape[0] / self.frames[0].shape[0]
        blend_rgba(canvas, img, center[0] - self.center[0] * s, center[1] - self.center[1] * s, alpha)
