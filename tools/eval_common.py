"""Shared parts of the tools/eval_*.py scripts (how to run them: README.md, "Evaluation").

- ``run_clip()``: run the pipeline headless on a clip and write the per-frame
  log, as ``python -m ipcvgame.app.main --headless --log`` does, plus three
  things the app cannot do: darken and add noise to the frames (``--light``,
  ``--noise``: a repeatable lighting condition), call a collector on every
  frame (for data that is not in the log, e.g. every keypoint), and save a
  run-info file next to the log (start-up time, frame size, settings, and
  optionally CPU/GPU/memory samples).
- ``logs_for_inputs()``: the common command line: clips are run first, CSV
  logs are read as they are.
- Metric helpers shared by the scripts (RMS, second difference, Markdown).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ipcvgame.core.config import REPO_ROOT

VIDEO_EXT = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
LABEL_DIR = REPO_ROOT / "data" / "labels"


# --- metric helpers ---------------------------------------------------------

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def mean(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def pct(x, q: float) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else float("nan")


def ratio(num: float, den: float) -> float:
    """100 * num / den, NaN when den is 0."""
    return 100.0 * num / den if den else float("nan")


def second_difference(xy: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|p[i+1] - 2 p[i] + p[i-1]| (px) for every i whose three consecutive
    frames are all ok. Steady motion cancels out, so what is left is
    frame-to-frame shake. Returns the values and the middle-frame indices."""
    m = ok[:-2] & ok[1:-1] & ok[2:]
    d = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1)
    idx = np.flatnonzero(m) + 1
    return d[m], idx


def player_ids(df: pd.DataFrame, pattern: str = r"p(\d+)_status") -> list[int]:
    return sorted({int(m.group(1)) for c in df.columns if (m := re.fullmatch(pattern, c))})


def markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(dict.fromkeys(k for r in rows for k in r))

    def fmt(v) -> str:
        if isinstance(v, float | np.floating):
            return "–" if np.isnan(v) else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, float("nan"))) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def write_tables(tables: dict[str, list[dict]], md_path: str | None) -> None:
    """Print each table under its heading, and save them all to md_path."""
    parts = [f"### {title}\n\n{markdown(rows)}\n" for title, rows in tables.items()]
    text = "\n".join(parts)
    print(text)
    if md_path:
        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text(text)
        print(f"saved {md_path}", file=sys.stderr)


def event_list(df: pd.DataFrame) -> pd.DataFrame:
    """The log's ``events`` column ("p1:punch;p2:hands_up") as rows of
    frame_id, t, pid, name."""
    rows = []
    if "events" in df:
        for fid, t, ev in df[["frame_id", "t", "events"]].dropna().itertuples(index=False):
            for item in str(ev).split(";"):
                pid, _, name = item.partition(":")
                if pid.startswith("p") and name:
                    rows.append((int(fid), float(t), int(pid[1:]), name))
    return pd.DataFrame(rows, columns=["frame_id", "t", "pid", "name"])


# --- running clips ----------------------------------------------------------

def degrade(image: np.ndarray, light: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    """A darker, noisier copy of the frame: pixel values times ``light``, plus
    Gaussian noise with standard deviation ``noise`` (grey levels). Less light
    at the same exposure darkens the image; the camera's gain then amplifies
    sensor noise, which the noise term stands in for."""
    out = image.astype(np.float32) * light
    if noise > 0:
        out += rng.normal(0.0, noise, image.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


class ResourceSampler:
    """CPU, memory and GPU use of this process, sampled every ``every_s``
    seconds during a clip run. Uses psutil and NVML when installed."""

    def __init__(self, every_s: float = 0.5):
        self.every_s = every_s
        self.samples: list[dict] = []
        self._last = -np.inf
        try:
            import psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)        # the first call only starts the count
        except ImportError:
            self._proc = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 - no NVIDIA driver / library: no GPU columns
            self._nvml = None

    def __call__(self, state, pipeline) -> None:
        now = time.perf_counter()
        if now - self._last < self.every_s:
            return
        self._last = now
        s: dict = {}
        if self._proc is not None:
            # % of one core (can exceed 100 with several threads)
            s["cpu_pct"] = self._proc.cpu_percent(None)
            s["rss_mb"] = self._proc.memory_info().rss / 2**20
        if self._nvml is not None:
            s["gpu_pct"] = float(self._nvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu)
            s["gpu_mem_mb"] = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu).used / 2**20
        try:
            import torch
            if torch.cuda.is_available():
                s["torch_mem_mb"] = torch.cuda.max_memory_reserved() / 2**20
        except ImportError:
            pass
        self.samples.append(s)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        df = pd.DataFrame(self.samples[1:] or self.samples)   # the first CPU sample covers start-up
        out = {}
        for col in df.columns:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_max"] = float(df[col].max())
        return out


def machine_info() -> dict:
    """CPU, RAM and GPU of this machine, for the report's test set-up."""
    import os
    import platform
    info = {"python": platform.python_version(), "cpu_threads": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        info["cpu"] = platform.processor()
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def run_clip(clip: Path, log: Path, until: str | None = None, overrides: list[str] = (),
             light: float = 1.0, noise: float = 0.0, max_frames: int | None = None,
             collectors: list[Callable] = ()) -> dict:
    """Run the pipeline on ``clip`` (headless) and write the per-frame log to
    ``log``, like the app's main loop. Every collector is called as
    ``collector(state, pipeline)`` after each frame. Returns the run info:
    build_s (pipeline construction: model loading and warm-up), startup_s
    (from the start of construction to the end of the first frame), frames,
    frame size and the settings."""
    from ipcvgame.app.pipeline import Pipeline
    from ipcvgame.core.config import load_config
    from ipcvgame.io.recorder import RunLogger
    from ipcvgame.io.source import open_source

    cfg = load_config(None, list(overrides))
    t0 = time.perf_counter()
    pipeline = Pipeline(cfg, until=until)
    build_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)                  # the same noise on every run
    info = {"clip": str(clip), "until": until, "overrides": list(overrides),
            "light": light, "noise": noise, "build_s": build_s}
    n = 0
    with open_source(cfg["source"], str(clip)) as source, \
            RunLogger(log, cfg["recorder"].get("flush_every", 300)) as logger:
        info["width"], info["height"] = source.size
        for frame in source:
            if light != 1.0 or noise > 0:
                frame = replace(frame, image=degrade(frame.image, light, noise, rng))
            state, _ = pipeline.step(frame)
            # state.timings is the previous frame's; log this frame's own.
            logger.log(replace(state, timings=pipeline.prof.last()), pipeline.log_fields(state))
            for c in collectors:
                c(state, pipeline)
            n += 1
            if n == 1:
                info["startup_s"] = time.perf_counter() - t0
            if max_frames is not None and n >= max_frames:
                break
    info["frames"] = n
    info["wall_s"] = time.perf_counter() - t0
    return info


def info_path(log: Path) -> Path:
    return log.with_suffix(".info.json")


def read_info(log: Path) -> dict:
    p = info_path(log)
    return json.loads(p.read_text()) if p.exists() else {}


# --- command line -----------------------------------------------------------

def add_common_args(ap: argparse.ArgumentParser, intro: bool) -> None:
    ap.add_argument("inputs", nargs="+", help="clips (run first) and/or run-log CSVs")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="config override for the clip runs, e.g. motion.tau=0.1")
    ap.add_argument("--tag", default=None, help="suffix for log names and labels, e.g. tau01")
    ap.add_argument("--light", type=float, default=1.0,
                    help="multiply every frame's pixel values by this (e.g. 0.3: darker), "
                         "a repeatable low-light condition")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="add Gaussian noise with this standard deviation (grey levels), "
                         "the camera gain in low light")
    ap.add_argument("--intro", action=argparse.BooleanOptionalAction, default=intro,
                    help="run the game intro (calibration, countdown); --no-intro plays from "
                         f"the first frame (default: {'intro' if intro else 'no intro'})")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--log-dir", default=str(REPO_ROOT / "logs" / "eval"))
    ap.add_argument("--markdown", default=None, metavar="MD", help="also write the tables to this file")


def clip_label(stem: str, args: argparse.Namespace) -> str:
    """Clip name plus the condition: -<tag>, -light<x>, -noise<y>."""
    label = stem
    if args.tag:
        label += f"-{args.tag}"
    if args.light != 1.0:
        label += f"-light{args.light:g}"
    if args.noise > 0:
        label += f"-noise{args.noise:g}"
    return label


def logs_for_inputs(args: argparse.Namespace, prefix: str, until: str | None,
                    make_collectors: Callable[[Path], list] | None = None
                    ) -> list[tuple[str, Path, dict]]:
    """(label, log path, run info) per input. A clip is run first, to
    <log-dir>/<prefix>-<label>.csv; ``make_collectors(log)`` gives the
    per-frame collectors, each with an optional ``save(log)`` called after
    the run. A CSV is used as it is."""
    out = []
    for item in map(Path, args.inputs):
        if item.suffix.lower() in VIDEO_EXT:
            label = clip_label(item.stem, args)
            log = Path(args.log_dir) / f"{prefix}-{label}.csv"
            log.parent.mkdir(parents=True, exist_ok=True)
            overrides = list(args.overrides)
            overrides.append(f"game.intro={'true' if args.intro else 'false'}")
            collectors = make_collectors(log) if make_collectors else []
            print(f"running {item} -> {log}", file=sys.stderr)
            info = run_clip(item, log, until, overrides, args.light, args.noise,
                            args.max_frames, collectors)
            for c in collectors:
                if hasattr(c, "save"):
                    c.save(log)
                if hasattr(c, "summary"):
                    info["resources"] = c.summary()
            info_path(log).write_text(json.dumps(info, indent=2))
        else:
            log = item
            label = re.sub(rf"^{prefix}-", "", item.stem)
            info = read_info(log)
        out.append((label, log, info))
    return out


def clip_stem(label: str, info: dict) -> str:
    """The clip's file stem, for finding its label file."""
    if info.get("clip"):
        return Path(info["clip"]).stem
    return label.split("-")[0]


def read_labels(path: Path | None) -> pd.DataFrame | None:
    """A label file from tools/label_clip.py (frame_id, t, label, x, y), or None."""
    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path)
