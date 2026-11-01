"""Task 5 evaluation: end-to-end frame rate, stage times, resources and stability.

    python tools/eval_pipeline.py data/clips/*.mp4                 # run the whole game on each clip
    python tools/eval_pipeline.py logs/run-*.csv                   # live webcam sessions (no re-run)
    python tools/eval_pipeline.py data/clips/session.mp4 --light 0.3 --noise 8
    python tools/eval_pipeline.py data/clips/session.mp4 --tag nano --set pose.model=models/yolo26n-pose.pt

A clip is run headless through every stage (render included) without the
intro, and its per-frame log is written to logs/eval/pipeline-<label>.csv.
Clip runs also record start-up time and CPU / GPU / memory use (psutil and
NVML) in pipeline-<label>.info.json; logs from the game itself (logs/run-*.csv,
e.g. a live session) have no such file, so those columns stay empty, but
they show the live frame rate and the long-run stability.

Tables: "frame rate" (FPS, latency, freezes, failures, start-up), "stages"
(median and 95th percentile ms per stage), "resources", and "fast motion"
(the fastest wrist motion still tracked). The machine's CPU / GPU are
printed first. Metric definitions: the protocol of Task 5 in docs/project_report.md.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from eval_common import (
    ResourceSampler,
    add_common_args,
    logs_for_inputs,
    machine_info,
    mean,
    pct,
    player_ids,
    write_tables,
)

STAGES = ["pose", "identity", "motion", "face", "gestures", "game", "render"]


def fps_of(ms: np.ndarray) -> float:
    return 1000.0 / ms if ms and not np.isnan(ms) else float("nan")


def frame_rate(df: pd.DataFrame, label: str, info: dict, freeze_ms: float) -> dict:
    ms = df["ms_total"].to_numpy(float) if "ms_total" in df else np.full(len(df), np.nan)
    t = df["t"].to_numpy(float)
    gaps = np.diff(t) * 1000.0
    failed = df["pipeline_failed"].notna() if "pipeline_failed" in df else pd.Series(False, index=df.index)
    fallback = (df["pipeline_fallback"].dropna().iloc[-1]
                if "pipeline_fallback" in df and df["pipeline_fallback"].notna().any() else "")
    duration = float(t[-1] - t[0]) if len(t) > 1 else float("nan")
    return {
        "clip": label, "frames": len(df), "duration s": duration,
        # Processing rate: what the pipeline could keep up with (1000 / ms per frame).
        "FPS mean": fps_of(mean(ms)), "FPS 5% low": fps_of(pct(ms, 95)), "FPS 1% low": fps_of(pct(ms, 99)),
        # Frames per second of capture time: live, the rate the game ran at;
        # on a clip, the clip's own recording rate.
        "capture FPS": (len(t) - 1) / duration if duration and duration > 0 else float("nan"),
        "ms median": pct(ms, 50), "ms p95": pct(ms, 95), "ms max": pct(ms, 100),
        f"frames > {freeze_ms:g} ms": int(np.sum(ms > freeze_ms)),
        f"capture gaps > {freeze_ms:g} ms": int(np.sum(gaps > freeze_ms)),
        "failed frames": int(failed.sum()), "fallback": fallback,
        "build s": info.get("build_s", float("nan")),
        "startup s": info.get("startup_s", float("nan")),
    }


def stages(df: pd.DataFrame, label: str) -> dict:
    row = {"clip": label}
    for s in STAGES + ["total"]:
        if f"ms_{s}" in df:
            v = df[f"ms_{s}"].to_numpy(float)
            row[f"{s} med"] = pct(v, 50)
            row[f"{s} p95"] = pct(v, 95)
    return row


def resources(label: str, info: dict) -> dict:
    r = info.get("resources", {})
    return {"clip": label,
            "CPU % (1 core = 100)": r.get("cpu_pct_mean", float("nan")),
            "CPU % max": r.get("cpu_pct_max", float("nan")),
            "RAM MB max": r.get("rss_mb_max", float("nan")),
            "GPU util %": r.get("gpu_pct_mean", float("nan")),
            "GPU mem MB max": r.get("gpu_mem_mb_max", float("nan")),
            "torch GPU MB max": r.get("torch_mem_mb_max", float("nan"))}


def fast_motion(df: pd.DataFrame, label: str) -> list[dict]:
    """The fastest wrist motion the pipeline still tracked: the wrist speed
    signal (shoulder widths per second, smoothed) exists only while the
    wrist is valid, so its top percentiles are the fastest tracked motion.
    The maximum can be a single keypoint glitch; p99 is the robust figure."""
    rows = []
    for pid in player_ids(df):
        cols = [c for c in (f"motion_p{pid}_l_wrist_speed", f"motion_p{pid}_r_wrist_speed") if c in df]
        if not cols:
            continue
        v = np.concatenate([df[c].to_numpy(float) for c in cols])
        v = v[~np.isnan(v)]
        rows.append({"clip": label, "player": f"P{pid}", "wrist samples": len(v),
                     "wrist p50 sw/s": pct(v, 50), "wrist p95 sw/s": pct(v, 95),
                     "wrist p99 sw/s": pct(v, 99),
                     "wrist max sw/s": float(v.max()) if len(v) else float("nan")})
    return rows


def main() -> None:
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, intro=False)
    ap.add_argument("--freeze-ms", type=float, default=200.0,
                    help="a frame slower than this counts as a freeze (a visible stutter)")
    args = ap.parse_args()

    import ipcvgame.app.pipeline  # noqa: F401 - time the imports (torch, ultralytics, mediapipe)
    import_s = time.perf_counter() - t_start
    print("machine: " + ", ".join(f"{k} {v}" for k, v in machine_info().items()), file=sys.stderr)
    print(f"imports: {import_s:.1f} s (once per process, add to start-up)", file=sys.stderr)

    tables: dict[str, list[dict]] = {"Task 5: frame rate": [], "Task 5: stages": [],
                                     "Task 5: resources": [], "Task 5: fast motion": []}
    for label, log, info in logs_for_inputs(args, "pipeline", None, lambda log: [ResourceSampler()]):
        df = pd.read_csv(log).sort_values("frame_id").reset_index(drop=True)
        tables["Task 5: frame rate"].append(frame_rate(df, label, info, args.freeze_ms))
        tables["Task 5: stages"].append(stages(df, label))
        if info.get("resources"):
            tables["Task 5: resources"].append(resources(label, info))
        tables["Task 5: fast motion"] += fast_motion(df, label)
    write_tables({k: v for k, v in tables.items() if v}, args.markdown)


if __name__ == "__main__":
    main()
