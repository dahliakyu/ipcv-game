"""Task 5 — where the people are: MediaPipe's Selfie Segmenter.

The arena (scene/arena.py) keeps wall paint and dimming off the players. A
mask built from the keypoints (arena.person_mask()) is a set of capsules
around the bones: wider than thin arms, narrower than a coat, so wall paint
stopped along straight lines with bare wall between it and the body. The
Selfie Segmenter (landscape model, 256 x 144 input) gives a soft person
mask that follows the real outline, hair and clothes included, for
everyone in view at once; no per-player split is needed for the wall.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ipcvgame.core.config import REPO_ROOT


class PersonSegmenter:
    def __init__(self, model_path: str, ramp: tuple[float, float] = (0.0, 1.0)):
        model = Path(model_path)
        if not model.is_absolute():
            model = REPO_ROOT / model
        if not model.is_file():
            raise FileNotFoundError(f"{model} not found; the README has the download command")
        # IMAGE mode: every frame stands alone, as for the face landmarker.
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True, output_category_mask=False))
        self.lo, self.hi = (float(v) for v in ramp)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """float32 (h, w) person mask 0-1 for the BGR image, at its size
        (MediaPipe scales the model's output back up): the confidence ramped
        from 0 at `lo` to 1 at `hi`, so the soft band at the outline (the
        model works at 256 x 144) gets narrower."""
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        r = self.segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        # A copy: numpy_view() points into MediaPipe's buffer, which is freed
        # with the result (a view read back garbage).
        conf = np.array(r.confidence_masks[0].numpy_view()[:, :, 0], dtype=np.float32, copy=True)
        return np.clip((conf - self.lo) / max(self.hi - self.lo, 1e-6), 0.0, 1.0)

    def close(self) -> None:
        self.segmenter.close()
