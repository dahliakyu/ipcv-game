"""Magic beam: the look of the Task 4 power attack (magic_effect pack).

A magic ball sits on the beam's start point, between the player's raised
hands, and the beam comes out of it, as in the pack's preview (magic.gif):

- **Ball:** its frames play once over the beam's whole life (charge plus
  firing), so it grows while charging, swirls while the beam fires and
  shrinks to a dot as the beam ends.
- **Beam:** the band frames are its cross-section, thickest first (horizontal
  stripes, the same in every column, so a band is stretched to any length
  exactly). After the charge it starts at the thinnest band, thickens one
  step every ``ramp_s`` and thins again in its last steps. The full band is
  as thick as the game's hit band: what you see is what hits.

For another team colour every pixel's hue is turned by the same angle
(``hue_shift``), which keeps brightness and saturation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ipcvgame.scene.animation import load_frames
from ipcvgame.scene.draw import blend_rgba, main_hue, shift_hue


class MagicBeam:
    def __init__(self, ball: str | Path, band: str | Path, ramp_s: float, hue_shift: int = 0):
        self.ball = load_frames(ball)
        # One column per band is enough: every column is the same.
        self.band = [f[:, f.shape[1] // 2:f.shape[1] // 2 + 1] for f in load_frames(band)]
        if hue_shift:
            self.ball = [shift_hue(f, hue_shift) for f in self.ball]
            self.band = [shift_hue(f, hue_shift) for f in self.band]
        self.hue = main_hue(self.ball + self.band)
        self.ramp_s = ramp_s
        self._ball_scaled: dict[int, list[np.ndarray]] = {}

    def ball_frame(self, age: float, life_s: float, size: int) -> np.ndarray:
        """The ball at `age` of a `life_s` beam, scaled to `size` px."""
        if size not in self._ball_scaled:   # once per size: sizes follow the frame size
            s = size / self.ball[0].shape[0]
            self._ball_scaled[size] = [cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                                       for f in self.ball]
        frames = self._ball_scaled[size]
        return frames[min(int(age / life_s * len(frames)), len(frames) - 1)]

    def band_index(self, age: float, charge_s: float, life_s: float) -> int | None:
        """Which band is shown (0 = thickest), or None while charging."""
        if age < charge_s:
            return None
        # Steps since the beam fired, or until it ends, whichever is fewer.
        steps = int(min(age - charge_s, life_s - age) / self.ramp_s)
        return max(len(self.band) - 1 - steps, 0)

    def draw(self, canvas: np.ndarray, start_x: float, tip_x: float, y: float, direction: int,
             age: float, charge_s: float, life_s: float, band_px: float, ball_px: int) -> None:
        """Draw one beam: the band from start_x to tip_x at height y, then the
        ball on top, centred on (start_x, y)."""
        k = self.band_index(age, charge_s, life_s)
        length = int(abs(tip_x - start_x))
        if k is not None and length > 0:
            col = self.band[k]
            h = max(int(round(band_px * col.shape[0] / self.band[0].shape[0])), 1)
            img = np.repeat(cv2.resize(col, (1, h), interpolation=cv2.INTER_NEAREST), length, axis=1)
            blend_rgba(canvas, img, min(start_x, tip_x), y - h / 2)
        ball = self.ball_frame(age, life_s, max(int(ball_px), 1))
        if direction < 0:
            ball = ball[:, ::-1]
        blend_rgba(canvas, ball, start_x - ball.shape[1] / 2, y - ball.shape[0] / 2)
