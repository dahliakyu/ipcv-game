"""Entry point.

    python -m ipcvgame.app.main                          # webcam from config
    python -m ipcvgame.app.main --source data/clips/crossing.mp4
    python -m ipcvgame.app.main --until identity         # debug one module
    python -m ipcvgame.app.main --headless --max-frames 100 --set pose.backend=none
    python -m ipcvgame.app.main --record data/clips/session.mp4   # also save the raw input

Every run writes a per-frame CSV to logs/ (recorder.log in the config; --no-log
to skip). For recording test clips without the pipeline, use tools/record_clip.py.

Keys: q / Esc quit, p pause, r reset all module state.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import cv2

from ipcvgame.app.pipeline import STAGES, Pipeline
from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.io.recorder import ClipRecorder, RunLogger, run_stamp
from ipcvgame.io.source import open_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IPCV webcam multiplayer game")
    ap.add_argument("--config", default=None, help="override YAML on top of configs/default.yaml")
    ap.add_argument("--source", default=None, help="webcam index or video path (default: config)")
    ap.add_argument("--until", choices=STAGES, default=None,
                    help="run up to this stage and show its debug overlay")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="config override, e.g. pose.det_conf=0.5")
    ap.add_argument("--headless", action="store_true", help="no window (tests, benchmarks)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--log", default=None, metavar="CSV",
                    help="per-frame CSV path (default: <recorder.log_dir>/run-<time>.csv)")
    ap.add_argument("--no-log", action="store_true", help="do not write the per-frame CSV")
    ap.add_argument("--record", default=None, metavar="MP4",
                    help="also record the raw input frames to this clip")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    pipeline = Pipeline(cfg, until=args.until)
    window = cfg["render"].get("window_name", "IPCV game")
    if args.until:
        window += f" [debug: {args.until}]"

    rec_cfg = cfg["recorder"]
    logger = None
    if rec_cfg.get("log", True) and not args.no_log:
        log_dir = Path(rec_cfg.get("log_dir", "logs"))
        log_dir = log_dir if log_dir.is_absolute() else REPO_ROOT / log_dir
        log_path = args.log or log_dir / f"run-{run_stamp()}.csv"
        logger = RunLogger(log_path, rec_cfg.get("flush_every", 300))

    n = 0
    with ExitStack() as stack:
        source = stack.enter_context(open_source(cfg["source"], args.source))
        print(f"source {source.size[0]}x{source.size[1]}")
        if logger is not None:
            stack.enter_context(logger)
        clip = None
        if args.record:
            clip = stack.enter_context(ClipRecorder(
                args.record, cfg["source"].get("fps", 30), source.size,
                mirrored=source.mirror, fourcc=rec_cfg.get("fourcc", "mp4v")))
        paused = False
        frames = iter(source)
        canvas = None
        while True:
            if not paused:
                frame = next(frames, None)
                if frame is None:
                    break
                if clip is not None:
                    clip.write(frame)
                state, canvas = pipeline.step(frame)
                if logger is not None:
                    # state.timings is the previous frame's; log this frame's own.
                    logger.log(replace(state, timings=pipeline.prof.last()),
                               pipeline.log_fields(state))
                n += 1
            if not args.headless and canvas is not None:
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("r"):
                    pipeline.reset()
            if args.max_frames is not None and n >= args.max_frames:
                break

    if not args.headless:
        cv2.destroyAllWindows()
    if logger is not None:
        print(f"log: {logger.path}")
    if args.record:
        print(f"clip: {args.record}")
    summary = pipeline.prof.summary()
    print(f"processed {n} frames; mean ms per stage:")
    for k, v in summary.items():
        print(f"  {k:>9s} {v:7.2f}")


if __name__ == "__main__":
    main()
