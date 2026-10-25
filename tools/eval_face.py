"""Task 1 evaluation: face tracking quality per clip and player.

    python tools/eval_face.py data/clips/*.mp4                  # run the pipeline on each clip, then evaluate
    python tools/eval_face.py logs/eval/face-crossing.csv       # evaluate existing run logs
    python tools/eval_face.py data/clips/*.mp4 --tag beta0 --set face.filter_beta=0.0   # parameter experiment
    python tools/eval_face.py data/clips/*.mp4 --markdown results/face.md               # also save the table

A clip is run headless up to the face stage, and its per-frame log is written
to logs/eval/face-<clip>[-<tag>].csv. Logs are evaluated from the columns
FaceTracker.log_fields() writes. Metric definitions: docs/evaluation.md.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ipcvgame.core.config import REPO_ROOT

VIDEO_EXT = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
SOURCES = ["detect", "flow", "hold"]


def run_clip(clip: Path, log: Path, overrides: list[str]) -> None:
    """Run the pipeline headless up to the face stage, writing the per-frame log."""
    from ipcvgame.app.main import main
    args = ["--headless", "--until", "face", "--source", str(clip), "--log", str(log)]
    for o in overrides:
        args += ["--set", o]
    main(args)


def player_ids(df: pd.DataFrame) -> list[int]:
    return sorted({int(m.group(1)) for c in df.columns if (m := re.fullmatch(r"face_p(\d+)_source", c))})


def second_difference(xy: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|p[i+1] - 2 p[i] + p[i-1]| (px) for every i whose three consecutive
    frames are all ok. Steady motion cancels out, so what is left is
    frame-to-frame shake. Returns the values and the middle-frame indices."""
    m = ok[:-2] & ok[1:-1] & ok[2:]
    d = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=1)
    idx = np.flatnonzero(m) + 1
    return d[m], idx


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def evaluate(df: pd.DataFrame, label: str) -> list[dict]:
    """One result row per player in the log."""
    df = df.sort_values("frame_id").reset_index(drop=True)
    consecutive = np.r_[True, np.diff(df["frame_id"].to_numpy()) == 1]
    rows = []
    for pid in player_ids(df):
        pre = f"face_p{pid}_"
        present = df[f"p{pid}_status"].notna() if f"p{pid}_status" in df else df[pre + "source"].notna()
        src = df[pre + "source"].fillna("").where(present)
        n = int(present.sum())
        row = {"clip": label, "player": f"P{pid}", "frames": n}
        for s in SOURCES:
            row[f"{s} %"] = 100 * float((src == s).sum()) / n if n else float("nan")
        row["no face %"] = 100 * float((src == "").sum()) / n if n else float("nan")

        # Jitter and lag on detect frames, where the raw lip point is a measurement
        det = (src == "detect").to_numpy() & consecutive
        scale = df[pre + "scale"].to_numpy(float) if pre + "scale" in df else np.full(len(df), np.nan)
        if pre + "mouth_raw_x" in df:
            raw = df[[pre + "mouth_raw_x", pre + "mouth_raw_y"]].to_numpy(float)
            sm = df[[pre + "mouth_x", pre + "mouth_y"]].to_numpy(float)
            j_raw, idx = second_difference(raw, det)
            j_sm, _ = second_difference(sm, det)
            lag = np.linalg.norm(sm - raw, axis=1)[det]
            row |= {
                "jitter raw px": rms(j_raw), "jitter smooth px": rms(j_sm),
                "jitter raw %w": rms(j_raw / scale[idx] * 100),
                "jitter smooth %w": rms(j_sm / scale[idx] * 100),
                "lag px": float(np.mean(lag)) if len(lag) else float("nan"),
                "lag p95 px": float(np.percentile(lag, 95)) if len(lag) else float("nan"),
                "lag %w": float(np.mean(lag / scale[det] * 100)) if len(lag) else float("nan"),
            }
        if pre + "crop_px" in df:
            row["crop px"] = float(df[pre + "crop_px"].mean())
        ms = df["ms_face"].dropna().to_numpy(float) if "ms_face" in df else np.array([])
        row["face ms"] = float(np.mean(ms)) if len(ms) else float("nan")
        row["face ms p95"] = float(np.percentile(ms, 95)) if len(ms) else float("nan")
        rows.append(row)
    return rows


def markdown(rows: list[dict]) -> str:
    cols = list(dict.fromkeys(k for r in rows for k in r))

    def fmt(v) -> str:
        if isinstance(v, float):
            return "–" if np.isnan(v) else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, float("nan"))) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="clips (run first) and/or run-log CSVs")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="config override for the clip runs, e.g. face.filter_beta=0.0")
    ap.add_argument("--tag", default=None, help="suffix for log names and labels, e.g. beta0")
    ap.add_argument("--log-dir", default=str(REPO_ROOT / "logs" / "eval"))
    ap.add_argument("--markdown", default=None, metavar="MD", help="also write the table to this file")
    args = ap.parse_args()

    rows = []
    for item in map(Path, args.inputs):
        label = item.stem + (f"-{args.tag}" if args.tag else "")
        if item.suffix.lower() in VIDEO_EXT:
            log = Path(args.log_dir) / f"face-{label}.csv"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"running {item} -> {log}", file=sys.stderr)
            run_clip(item, log, args.overrides)
        else:
            log = item
            label = re.sub(r"^face-", "", label)
        df = pd.read_csv(log)
        if not player_ids(df):
            print(f"{log}: no face columns (face.backend=stub, or no player found)", file=sys.stderr)
            continue
        rows += evaluate(df, label)

    table = markdown(rows)
    print(table)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(table + "\n")
        print(f"saved {args.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
