"""Task 2 — pose detection.

PHASE-1 STUB (owned by Task 2). The only real model call in the stub pipeline:
a pretrained Ultralytics YOLO-pose model plus an adapter to the canonical
skeleton. Task 2 owns tuning and may replace the model, as long as detect()
keeps returning list[PoseDetection] in canonical order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import REPO_ROOT
from ipcvgame.core.types import Frame, FrameState, PoseDetection
from ipcvgame.scene.draw import draw_bbox, draw_skeleton


class PoseEstimator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.backend = cfg.get("backend", "yolo")
        self.last: list[PoseDetection] = []
        self.model = None
        if self.backend == "yolo":
            self._load_yolo()
        elif self.backend != "none":
            raise ValueError(f"Unknown pose backend {self.backend!r}")

    def _load_yolo(self) -> None:
        import torch
        from ultralytics import YOLO

        weights = Path(self.cfg["weights"])
        if not weights.is_absolute():
            weights = REPO_ROOT / weights
        weights.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads known weight names into the given path.
        self.model = YOLO(str(weights))
        device = self.cfg.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = bool(self.cfg.get("half", True)) and device.startswith("cuda")

    def detect(self, frame: Frame) -> list[PoseDetection]:
        if self.model is None:
            self.last = []
            return self.last
        result = self.model.predict(
            frame.image, imgsz=self.cfg.get("imgsz", 640), conf=self.cfg.get("det_conf", 0.35),
            max_det=self.cfg.get("max_det", 5), device=self.device,
            quantize=16 if self.half else 32, verbose=False,
        )[0]
        dets: list[PoseDetection] = []
        if result.boxes is not None and len(result.boxes) and result.keypoints is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            kpts = result.keypoints.xy.cpu().numpy()
            kconf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
                     else np.ones(kpts.shape[:2], dtype=np.float32))
            for b, s, k, c in zip(boxes, scores, kpts, kconf, strict=True):
                dets.append(PoseDetection(
                    bbox=b.astype(np.float64),
                    kpts=k[sk.FROM_COCO17].astype(np.float64),
                    conf=c[sk.FROM_COCO17].astype(np.float64),
                    score=float(s),
                ))
        self.last = dets
        return dets

    def reset(self, pid: int | None = None) -> None:
        pass  # stateless

    def draw_debug(self, canvas: np.ndarray, state: FrameState) -> None:
        """Raw detections, no identity: grey boxes, keypoints above debug_kpt_conf."""
        for d in self.last:
            draw_bbox(canvas, d.bbox, (200, 200, 200), f"{d.score:.2f}")
            draw_skeleton(canvas, d.kpts, d.conf >= self.cfg.get("debug_kpt_conf", 0.5), (200, 200, 200))
