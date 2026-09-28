"""Recording: test clips and per-frame CSV logs.

ClipRecorder
    Writes a video clip plus a sidecar ``<clip>.timestamps.csv`` holding each
    frame's real capture time. Webcam FPS varies, so frame_idx / fps would give
    filters the wrong dt; VideoFile replays the sidecar times instead.
    Clips are stored raw (un-mirrored): Frame.image is mirrored at capture, so
    the recorder flips it back, and VideoFile mirrors again on load. This keeps
    clips identical to what the webcam delivers.

RunLogger
    One CSV per run: frame_id, t, per-stage ms (``ms_<stage>``), and per player
    status, pos and bbox. Owners add their own columns by giving their module a
    ``log_fields(state) -> dict[str, float | str]`` method; the pipeline merges
    them into the row (prefix keys with your module, e.g. ``face_p1_roll``).
    Rows are appended every ``flush_every`` frames and on close(). Columns may
    appear mid-run (e.g. when Player 2 first shows up); the file is then
    rewritten once so it keeps a single header.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ipcvgame.core.types import Frame, FrameState


def timestamps_path(clip_path: str | Path) -> Path:
    p = Path(clip_path)
    return p.with_name(p.stem + ".timestamps.csv")


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class ClipRecorder:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int],
                 mirrored: bool = True, fourcc: str = "mp4v"):
        """size is (width, height); mirrored says whether incoming frames are
        mirrored (then they are flipped back before writing)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.mirrored = mirrored
        # fps is only the container's nominal rate; true times go to the sidecar.
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {self.path}")
        self._ts_file = open(timestamps_path(self.path), "w", newline="")
        self._ts = csv.writer(self._ts_file)
        self._ts.writerow(["frame_idx", "t"])
        self.n = 0

    def write(self, frame: Frame) -> None:
        image = frame.image
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError(f"frame size {image.shape[1]}x{image.shape[0]} != clip size {self.size}")
        self.writer.write(cv2.flip(image, 1) if self.mirrored else image)
        self._ts.writerow([self.n, f"{frame.t:.6f}"])
        self.n += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._ts_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fmt(v: float) -> str:
    return f"{v:.3f}"


class RunLogger:
    def __init__(self, path: str | Path, flush_every: int = 300):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.rows: list[dict] = []
        self.columns: list[str] = []  # first-seen order, stable across flushes
        self._t0: float | None = None
        self._written = 0
        self._header: list[str] | None = None
        self._seen: dict[str, int] = {}

    def log(self, state: FrameState, extra: dict | None = None) -> None:
        f = state.frame
        if self._t0 is None:
            self._t0 = f.t
        row: dict = {"frame_id": f.frame_id, "t": f"{f.t - self._t0:.6f}"}
        for stage, ms in state.timings.items():
            row[f"ms_{stage}"] = _fmt(ms)
        for pid, p in sorted(state.players.items()):
            pre = f"p{pid}_"
            row[pre + "status"] = p.status
            row[pre + "reentered"] = int(p.reentered)
            row[pre + "x_m"], row[pre + "depth_m"] = _fmt(p.pos[0]), _fmt(p.pos[1])
            cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
            row[pre + "cx"], row[pre + "cy"] = _fmt(cx), _fmt(cy)
            row[pre + "score"] = _fmt(p.det.score) if p.det is not None else ""
        row["n_events"] = len(state.events)
        if state.events:
            row["events"] = ";".join(f"p{e.pid}:{e.name}" for e in state.events)
        for k, v in (extra or {}).items():
            row[k] = _fmt(v) if isinstance(v, float | np.floating) else v
        new = [k for k in row if k not in self.columns]
        if new:
            self.columns = sorted(self.columns + new, key=self._column_key)
        self.rows.append(row)
        if self.flush_every and len(self.rows) - self._written >= self.flush_every:
            self.flush()

    _PLAYER_FIELDS = ["status", "reentered", "x_m", "depth_m", "cx", "cy", "score"]

    def _column_key(self, col: str) -> tuple:
        """Stable order: frame_id, t, stage ms, players by pid, events, then
        owners' extra columns in first-seen order."""
        if col in ("frame_id", "t"):
            return (0, col != "frame_id")
        if col.startswith("ms_"):
            return (1, col == "ms_total", self._first_seen(col))
        m = re.fullmatch(r"p(\d+)_(\w+)", col)
        if m and m.group(2) in self._PLAYER_FIELDS:
            return (2, int(m.group(1)), self._PLAYER_FIELDS.index(m.group(2)))
        if col in ("n_events", "events"):
            return (3, col)
        return (4, self._first_seen(col))

    def _first_seen(self, col: str) -> int:
        if col not in self._seen:
            self._seen[col] = len(self._seen)
        return self._seen[col]

    def flush(self) -> None:
        """Append unwritten rows. If new columns appeared since the header was
        written, rewrite the whole file once so it keeps a single header."""
        if self._header != self.columns:
            tmp = self.path.with_suffix(".csv.tmp")
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.columns, restval="")
                w.writeheader()
                w.writerows(self.rows)
            tmp.replace(self.path)
            self._header = list(self.columns)
        else:
            with open(self.path, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.columns, restval="").writerows(
                    self.rows[self._written:])
        self._written = len(self.rows)

    def close(self) -> None:
        if self.rows and self._written != len(self.rows):
            self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

