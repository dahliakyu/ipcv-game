"""Frame sources: Webcam and VideoFile behind one interface.

Both yield core.types.Frame objects that are already mirrored (if configured),
so every module can be developed on recorded clips exactly as on the live
camera.

    with open_source(cfg["source"]) as src:
        for frame in src:
            ...
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from ipcvgame.core.types import Frame
from ipcvgame.io.recorder import timestamps_path


class Source:
    """Iterable of Frames. Subclasses implement _read() -> (ok, image, t)."""

    def __init__(self, mirror: bool = True):
        self.mirror = mirror
        self._frame_id = 0
        self.cap: cv2.VideoCapture | None = None

    def _read(self) -> tuple[bool, cv2.typing.MatLike | None, float]:
        raise NotImplementedError

    def read(self) -> Frame | None:
        ok, image, t = self._read()
        if not ok or image is None:
            return None
        if self.mirror:
            image = cv2.flip(image, 1)
        frame = Frame(image=image, frame_id=self._frame_id, t=t)
        self._frame_id += 1
        return frame

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    @property
    def size(self) -> tuple[int, int]:
        assert self.cap is not None
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Webcam(Source):
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, mirror: bool = True):
        super().__init__(mirror)
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {index}")
        # MJPG is needed by most UVC webcams to reach 720p at 30 fps.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # Keep only the newest frame so we never process stale ones.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _read(self):
        ok, image = self.cap.read()
        return ok, image, time.perf_counter()


class VideoFile(Source):
    """Recorded clip. With video_time="file", Frame.t follows the clip's own
    time (starting at the open time), so filters see the true dt no matter how
    fast we process. If the clip has a ``<clip>.timestamps.csv`` sidecar (written
    by io/recorder.py), its real capture times are replayed; otherwise
    frame_idx / fps is used."""

    def __init__(self, path: str | Path, mirror: bool = True,
                 video_time: str = "file", loop: bool = False):
        super().__init__(mirror)
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {self.path}")
        if video_time not in ("file", "wall"):
            raise ValueError(f"video_time must be 'file' or 'wall', got {video_time!r}")
        self.video_time = video_time
        self.loop = loop
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._t0 = time.perf_counter()
        self._t_offset = 0.0  # accumulated clip duration when looping
        self._last_video_t = 0.0
        self.timestamps = self._load_timestamps()

    def _load_timestamps(self) -> np.ndarray | None:
        p = timestamps_path(self.path)
        if not p.exists():
            return None
        ts = np.loadtxt(p, delimiter=",", skiprows=1, usecols=1, ndmin=1)
        return ts - ts[0] if len(ts) else None

    def _read(self):
        ok, image = self.cap.read()
        if not ok and self.loop:
            self._t_offset += self._last_video_t + 1.0 / self.fps
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self.cap.read()
        if not ok:
            return False, None, 0.0
        if self.video_time == "wall":
            return True, image, time.perf_counter()
        # Frame index is more reliable than POS_MSEC across codecs.
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if self.timestamps is not None and idx < len(self.timestamps):
            self._last_video_t = float(self.timestamps[idx])
        else:
            self._last_video_t = idx / self.fps
        return True, image, self._t0 + self._t_offset + self._last_video_t


def open_source(cfg: dict, uri: str | int | None = None) -> Source:
    """Build a source from the config's source section. A uri that is an int
    (or digit string) is a webcam index; anything else is a video path."""
    uri = cfg.get("uri", 0) if uri is None else uri
    if isinstance(uri, str) and uri.isdigit():
        uri = int(uri)
    if isinstance(uri, int):
        return Webcam(uri, cfg.get("width", 1280), cfg.get("height", 720),
                      cfg.get("fps", 30), cfg.get("mirror", True))
    return VideoFile(uri, cfg.get("mirror", True), cfg.get("video_time", "file"),
                     cfg.get("loop", False))
