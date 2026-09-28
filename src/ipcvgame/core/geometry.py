"""Small geometry helpers: boxes, crops, points."""

from __future__ import annotations

import numpy as np


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two (4,) boxes in x1, y1, x2, y2."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def clip_bbox(b: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.array([
        np.clip(b[0], 0, width), np.clip(b[1], 0, height),
        np.clip(b[2], 0, width), np.clip(b[3], 0, height),
    ], dtype=np.float64)


def crop(image: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop image to box b. Returns (crop, offset) where offset is the (x, y)
    to add to crop-local coordinates to get full-frame coordinates."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_bbox(b, w, h).astype(int)
    return image[y1:y2, x1:x2], np.array([x1, y1], dtype=np.float64)
