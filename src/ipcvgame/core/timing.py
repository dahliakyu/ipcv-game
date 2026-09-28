"""Per-stage profiler.

    prof = Profiler()
    with prof("pose"):
        ...
    prof.last()     # {"pose": 12.3, ...} in ms, for the frame just finished
    prof.summary()  # mean ms per stage over the whole run
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self) -> None:
        self._current: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._frame_start: float | None = None
        self._frames = 0

    @contextmanager
    def __call__(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self._current[stage] = self._current.get(stage, 0.0) + ms
            self._totals[stage] += ms
            self._counts[stage] += 1

    def start_frame(self) -> None:
        self._frame_start = time.perf_counter()
        self._current = {}

    def end_frame(self) -> None:
        if self._frame_start is not None:
            ms = (time.perf_counter() - self._frame_start) * 1000.0
            self._current["total"] = ms
            self._totals["total"] += ms
            self._counts["total"] += 1
        self._last = dict(self._current)
        self._frames += 1

    def current(self) -> dict[str, float]:
        """Timings recorded so far in the frame in progress."""
        return dict(self._current)

    def last(self) -> dict[str, float]:
        """Timings of the last completed frame."""
        return dict(self._last)

    def summary(self) -> dict[str, float]:
        return {k: self._totals[k] / self._counts[k] for k in self._totals if self._counts[k]}
