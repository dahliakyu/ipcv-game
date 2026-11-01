"""Task 1 evaluation: face tracking quality per clip and player.

    python tools/eval_face.py data/clips/*.mp4                  # run the pipeline on each clip, then evaluate
    python tools/eval_face.py logs/eval/face-crossing.csv       # evaluate existing run logs
    python tools/eval_face.py data/clips/*.mp4 --tag beta0 --set face.filter_beta=0.0   # parameter experiment
    python tools/eval_face.py data/clips/*.mp4 --markdown results/face.md               # also save the table
    python tools/eval_face.py data/clips/session.mp4 --light 0.3 --noise 8              # low light

A clip is run headless up to the face stage, and its per-frame log is written
to logs/eval/face-<clip>[-<tag>].csv. Logs are evaluated from the columns
FaceTracker.log_fields() writes. Metric definitions: the protocol of Task 1 in docs/project_report.md.
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
from eval_common import add_common_args, logs_for_inputs, rms, second_difference, write_tables

SOURCES = ["detect", "flow", "hold"]


def player_ids(df: pd.DataFrame) -> list[int]:
    return sorted({int(m.group(1)) for c in df.columns if (m := re.fullmatch(r"face_p(\d+)_source", c))})


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
        if pre + "pose" in df:                  # detect/flow frames: where the head pose came from
            moving = src.isin(["detect", "flow"])
            pose = df[pre + "pose"].fillna("")[moving]
            m = int(moving.sum())
            row["pose %"] = 100 * float((pose != "").sum()) / m if m else float("nan")
            row["mediapipe pose %"] = 100 * float((pose == "mediapipe").sum()) / m if m else float("nan")

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
        if pre + "hat_x" in df:                 # smoothed hat anchor, same second difference
            j_hat, _ = second_difference(df[[pre + "hat_x", pre + "hat_y"]].to_numpy(float), det)
            row["hat jitter px"] = rms(j_hat)
        if pre + "crop_px" in df:
            row["crop px"] = float(df[pre + "crop_px"].mean())
        ms = df["ms_face"].dropna().to_numpy(float) if "ms_face" in df else np.array([])
        row["face ms"] = float(np.mean(ms)) if len(ms) else float("nan")
        row["face ms p95"] = float(np.percentile(ms, 95)) if len(ms) else float("nan")
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, intro=False)
    args = ap.parse_args()

    rows = []
    for label, log, _ in logs_for_inputs(args, "face", "face"):
        df = pd.read_csv(log)
        if not player_ids(df):
            print(f"{log}: no face columns (face.backend=stub, or no player found)", file=sys.stderr)
            continue
        rows += evaluate(df, label)
    write_tables({"Task 1: face tracking": rows}, args.markdown)


if __name__ == "__main__":
    main()
