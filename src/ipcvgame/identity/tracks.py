"""Task 3 — BoT-SORT track ids on the pose model's own boxes.

Colour Clash main.py ran a second model (yolo11n.pt) with
`person_detector.track(..., tracker="botsort.yaml")`. Here Ultralytics'
BOTSORT class gets the PoseDetection boxes the pose stage already found, so
one model runs per frame. The settings from botsort.yaml are in the
`identity.botsort` config.

BoT-SORT keeps a Kalman filter (constant velocity) per track, matches tracks
to boxes by IoU, and keeps an unmatched track as "lost" for track_buffer
frames, predicting where it moves, before removing it.
"""

from __future__ import annotations

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

from ipcvgame.core.geometry import clip_bbox
from ipcvgame.core.types import Frame, PoseDetection
from ipcvgame.identity.tracker import Tracks


class BotSortTracks:
    def __init__(self, cfg: dict):
        self.bot = BOTSORT(IterableSimpleNamespace(**cfg))

    def update(self, dets: list[PoseDetection], frame: Frame) -> Tracks:
        """active: (track_id, index into dets) for the tracks seen this frame;
        lost: track_id -> predicted box for the tracks BoT-SORT still holds."""
        h, w = frame.image.shape[:2]
        data = np.zeros((len(dets), 6))           # x1, y1, x2, y2, score, class (0 = person)
        for i, d in enumerate(dets):
            data[i, :4] = d.bbox
            data[i, 4] = d.score
        out = self.bot.update(Boxes(data, (h, w)), frame.image)
        active = [(int(row[4]), int(row[7])) for row in out]
        lost = {t.track_id: clip_bbox(t.xyxy, w, h) for t in self.bot.lost_stracks}
        return Tracks(active, lost)

    def reset(self) -> None:
        self.bot.reset()
