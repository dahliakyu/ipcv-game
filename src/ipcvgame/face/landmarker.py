"""Task 1 — MediaPipe Face Landmarker on a face crop.

Used next to YuNet: YuNet finds and tracks the face, and
the landmarker reads what YuNet's 5 points cannot give: the head rotation
(reliable also for strong up/down tilt) and the 52 blendshape scores
(jawOpen, eyeBlinkLeft/Right, mouthSmileLeft/Right, cheekPuff, ...).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ipcvgame.core.config import REPO_ROOT


class FaceLandmarker:
    def __init__(self, cfg: dict):
        model = Path(cfg["landmarker_model"])
        if not model.is_absolute():
            model = REPO_ROOT / model
        if not model.is_file():
            raise FileNotFoundError(f"{model} not found; the README has the download command")
        conf = float(cfg["landmarker_min_conf"])
        # IMAGE mode: every call stands alone. The crops come from different
        # players in turn, so MediaPipe's own tracking between calls would mix them.
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=conf, min_face_presence_confidence=conf,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        self.max_side = int(cfg["landmarker_max_side"])

    def __call__(self, crop_bgr: np.ndarray) -> dict | None:
        """Blendshape scores and the 4x4 head transformation matrix for the
        face in the crop, or None when no face is found."""
        h, w = crop_bgr.shape[:2]
        if min(h, w) < 16:
            return None
        s = min(1.0, self.max_side / max(h, w))
        if s < 1.0:                                          # the model works at 256 px anyway
            crop_bgr = cv2.resize(crop_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        r = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not r.face_blendshapes or not r.facial_transformation_matrixes:
            return None
        return {"blendshapes": {b.category_name: float(b.score) for b in r.face_blendshapes[0]},
                "matrix": np.asarray(r.facial_transformation_matrixes[0], dtype=np.float64)}

    def close(self) -> None:
        self.landmarker.close()
