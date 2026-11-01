"""Task 3 evaluation: identity tracking per clip and player.

    python tools/label_clip.py data/clips/crossing.mp4 --mode identity   # once per clip: who is who
    python tools/eval_identity.py data/clips/*.mp4                       # run each clip, then evaluate
    python tools/eval_identity.py logs/eval/identity-crossing.csv        # evaluate an earlier run
    python tools/eval_identity.py data/clips/crossing.mp4 --tag th05 --set identity.reid_threshold=0.5
    python tools/eval_identity.py data/clips/crossing.mp4 --light 0.3 --noise 8
    python tools/eval_identity.py data/clips/*.mp4 --true-depth 1=2.4 --true-depth 2=2.5

A clip is run headless up to the game stage with the intro (the game's
calibration renumbers players left to right and refreshes their OSNet
vectors, as in a live game), and its per-frame log is written to
logs/eval/identity-<label>.csv.

Two tables:
- "tracks" (no labels needed): how many players were created, time active /
  occluded / lost, re-entries, how long re-identification took, track
  relinks, sudden jumps, depth stability, timing.
- "labelled" (with data/labels/<clip>.identity.csv from tools/label_clip.py):
  identity accuracy, switches, crossings preserved, re-entries recovered.
Metric definitions: the protocol of Task 3 in docs/project_report.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from eval_common import (
    LABEL_DIR,
    add_common_args,
    clip_stem,
    logs_for_inputs,
    mean,
    pct,
    player_ids,
    ratio,
    read_labels,
    write_tables,
)

from ipcvgame.core.config import load_config

STATUSES = ("active", "occluded", "lost")
# Label names of the two players; any other label (C, D, ...) is someone
# else in view: a bystander, or a newcomer who should not take a player's id.
PLAYERS = ("A", "B")


def renumbered(df: pd.DataFrame, frames: int = 3) -> np.ndarray:
    """Rows right after the game's calibration started: there the pipeline
    renumbers the players left to right (order_left_to_right), which moves
    ids between people on purpose. Those rows are left out of jumps, order
    flips, corrections and relinks."""
    out = np.zeros(len(df), bool)
    if "game_phase" in df:
        ph = df["game_phase"].fillna("").to_numpy()
        for i in np.flatnonzero((ph[1:] == "calibration") & (ph[:-1] == "waiting")) + 1:
            out[i:i + frames] = True
    return out


def changes(values: pd.Series, skip: np.ndarray) -> int:
    """How often the non-empty value changes from one row to the next it
    appears in, not counting changes that land on a skipped row."""
    v = values.where(values.astype(str) != "")
    keep = v.notna().to_numpy()
    x, sk_ = v[keep].astype(float).to_numpy(), skip[keep]
    return int(np.sum((x[1:] != x[:-1]) & ~sk_[1:]))


def track_metrics(df: pd.DataFrame, label: str, width: float, jump_frac: float,
                  expected: int, true_depth: dict[int, float]) -> list[dict]:
    """The label-free table: one row per player in the log."""
    df = df.sort_values("frame_id").reset_index(drop=True)
    t = df["t"].to_numpy(float)
    pids = player_ids(df)
    skip = renumbered(df)
    pending = df["identity_pending"].fillna(0).to_numpy(float) if "identity_pending" in df else None
    ms = df["ms_identity"].to_numpy(float) if "ms_identity" in df else np.full(len(df), np.nan)
    order_flips = 0
    if 1 in pids and 2 in pids:
        both = (df["p1_status"] == "active") & (df["p2_status"] == "active")
        side = np.sign(df["p1_cx"].astype(float) - df["p2_cx"].astype(float)).where(both)
        order_flips = changes(side, skip)
    rows = []
    for pid in pids:
        pre = f"p{pid}_"
        status = df[pre + "status"]
        n = int(status.notna().sum())
        st = status.fillna("").to_numpy()
        prev = np.r_[[""], st[:-1]]
        re = df[pre + "reentered"].fillna(0).astype(int).to_numpy() == 1
        reentries = re & (prev == "lost")
        # reentered without being lost: the tracker swapped two players'
        # links back (OSNet found a track on the wrong person).
        corrections = re & (prev != "lost") & (prev != "") & ~skip

        # Re-identification time: from the start of the run of frames with an
        # unassigned track just before the re-entry, to the re-entry.
        reid_s = []
        if pending is not None:
            for i in np.flatnonzero(reentries):
                j = i
                while j > 0 and pending[j - 1] > 0:
                    j -= 1
                reid_s.append(t[i] - t[j])

        track_col = f"identity_p{pid}_track"
        relinks = changes(df[track_col], skip) if track_col in df else 0

        # Jumps: the box centre moves more than jump_frac of the frame width
        # between two consecutive active frames: no person moves that fast,
        # so it is most likely the id landing on someone else.
        act = st == "active"
        cxy = df[[pre + "cx", pre + "cy"]].to_numpy(float)
        consecutive = np.r_[False, np.diff(df["frame_id"].to_numpy()) == 1]
        step = np.r_[np.nan, np.linalg.norm(np.diff(cxy, axis=0), axis=1)]
        jumps = int(np.sum(act & np.r_[False, act[:-1]] & consecutive & ~skip
                           & (step > jump_frac * width)))

        depth = df[pre + "depth_m"].to_numpy(float)[act] if pre + "depth_m" in df else np.array([])
        row = {"clip": label, "player": f"P{pid}", "frames": n,
               "players created": len(pids), "extra ids": max(0, len(pids) - expected)}
        for s_ in STATUSES:
            row[f"{s_} %"] = ratio(int((st == s_).sum()), n)
        row |= {
            "re-entries": int(reentries.sum()),
            "re-id s": pct(reid_s, 50),
            "corrections": int(corrections.sum()),
            "track relinks": relinks,
            "jumps": jumps,
            "order flips": order_flips,
            # Median and interquartile range: a turned body gives a few far
            # too large depths (narrow shoulders), which would swamp a mean.
            "depth m": pct(depth, 50),
            "depth IQR m": pct(depth, 75) - pct(depth, 25),
        }
        if pid in true_depth:
            row["depth error mm"] = 1000.0 * pct(np.abs(depth - true_depth[pid]), 50)
        row["identity ms"] = mean(ms)
        row["identity ms p95"] = pct(ms, 95)
        rows.append(row)
    return rows


def assign(df: pd.DataFrame, labels: pd.DataFrame, gate: float) -> pd.DataFrame:
    """For every labelled frame and every active player in the log: the
    labelled person nearest to the player's box centre, within gate px, or
    "" when none is that close. Columns frame_id, t, pid, person."""
    people_rows = labels[labels["x"].notna() & (labels["label"] != "-")]
    by_frame = {f: g for f, g in people_rows.groupby("frame_id")}
    log = df.set_index("frame_id")
    pids = player_ids(df)
    rows = []
    for fid in sorted(labels["frame_id"].unique()):
        if fid not in log.index:
            continue
        r = log.loc[fid]
        people = by_frame.get(fid)
        for pid in pids:
            if r.get(f"p{pid}_status") != "active":
                continue
            c = np.array([r[f"p{pid}_cx"], r[f"p{pid}_cy"]], float)
            person = ""
            if people is not None:
                d = np.linalg.norm(people[["x", "y"]].to_numpy(float) - c, axis=1)
                if d.min() <= gate:
                    person = people["label"].iloc[int(np.argmin(d))]
            rows.append((fid, float(r["t"]), pid, person))
    return pd.DataFrame(rows, columns=["frame_id", "t", "pid", "person"])


def labelled_metrics(df: pd.DataFrame, labels: pd.DataFrame, label: str, gate: float,
                     settle_s: float, absent_s: float) -> dict:
    """The labelled table row for one clip."""
    a = assign(df, labels, gate)
    t_of = dict(zip(labels["frame_id"], labels["t"], strict=False))
    frames = sorted(labels["frame_id"].unique())
    # Each player's own person: the player (A or B) it is on most often. A
    # player on anyone else (a bystander, a newcomer) is always wrong.
    on_player = a["person"].isin(PLAYERS)
    own = {pid: g.loc[on_player[g.index], "person"].mode().iloc[0]
           for pid, g in a.groupby("pid") if on_player[g.index].any()}
    correct = sum(int(p == own.get(pid)) for pid, p in zip(a["pid"], a["person"], strict=True))
    on_others = int((a["person"].ne("") & ~on_player).sum())
    switches = 0
    for _, g in a[a["person"] != ""].groupby("pid"):
        p = g.sort_values("frame_id")["person"].to_numpy()
        switches += int((p[1:] != p[:-1]).sum())

    # People without a player: labelled, but no active player within the gate.
    ab = labels[labels["label"].isin(PLAYERS)]
    covered = {(f, p) for f, p in zip(a["frame_id"], a["person"], strict=True) if p}
    unassigned = sum((f, p) not in covered for f, p in zip(ab["frame_id"], ab["label"], strict=True))

    # Crossings: A and B swap sides (x order) between two labelled frames.
    mapping = {f: dict(zip(g["pid"], g["person"], strict=True))
               for f, g in a[a["person"] != ""].groupby("frame_id")}
    xs = ab.pivot_table(index="frame_id", columns="label", values="x")
    crossings = preserved = 0
    if {"A", "B"} <= set(xs.columns):
        xs = xs.dropna()
        side = np.sign(xs["A"] - xs["B"]).to_numpy()
        fids = xs.index.to_numpy()
        for k in np.flatnonzero(side[1:] != side[:-1]) + 1:
            crossings += 1
            before = next((mapping[f] for f in reversed(frames)
                           if f < fids[k] and len(mapping.get(f, {})) >= 2), None)
            after = next((mapping[f] for f in frames
                          if t_of[f] >= t_of[fids[k]] + settle_s and len(mapping.get(f, {})) >= 2), None)
            if before is not None and after is not None:
                preserved += int(all(after.get(pid) == p for pid, p in before.items() if pid in after))

    # Re-entries: a person not labelled for at least absent_s, then back.
    reentries = recovered = 0
    rec_s = []
    present = {p: sorted(ab.loc[ab["label"] == p, "frame_id"]) for p in PLAYERS}
    for person, fl in present.items():
        for f0, f1 in zip(fl[:-1], fl[1:], strict=True):
            if t_of[f1] - t_of[f0] < absent_s:
                continue
            # Only a real absence: frames in between were reviewed without them.
            if not any(f0 < f < f1 for f in frames):
                continue
            pid_before = next((pid for pid, p in mapping.get(f0, {}).items() if p == person), None)
            if pid_before is None:
                continue
            reentries += 1
            back = next((f for f in frames if f >= f1 and person in mapping.get(f, {}).values()), None)
            if back is not None:
                pid_after = next(pid for pid, p in mapping[back].items() if p == person)
                if pid_after == pid_before:
                    recovered += 1
                    rec_s.append(t_of[back] - t_of[f1])
    n = len(a)
    return {"clip": label, "labelled frames": len(frames), "player-frames": n,
            "id accuracy %": ratio(correct, n), "on others %": ratio(on_others, n),
            "unassigned players %": ratio(unassigned, len(ab)),
            "id switches": switches,
            "switches /min": switches / ((max(t_of.values()) - min(t_of.values())) / 60.0)
            if len(t_of) > 1 else float("nan"),
            "crossings": crossings, "crossings kept %": ratio(preserved, crossings),
            "re-entries": reentries, "recovered %": ratio(recovered, reentries),
            "recovery s": pct(rec_s, 50)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, intro=True)
    ap.add_argument("--players", type=int, default=2, help="people in the clips (for 'extra ids')")
    ap.add_argument("--jump", type=float, default=0.15,
                    help="a box centre moving more than this fraction of the frame width in one "
                         "frame counts as a jump")
    ap.add_argument("--gate", type=float, default=0.15,
                    help="labels: a player is on a labelled person when its box centre is within "
                         "this fraction of the frame width of the click")
    ap.add_argument("--settle-s", type=float, default=1.0,
                    help="labels: compare identities this long after a crossing")
    ap.add_argument("--absent-s", type=float, default=1.0,
                    help="labels: a person unlabelled this long has left the view")
    ap.add_argument("--true-depth", action="append", default=[], metavar="PID=METRES",
                    help="measured distance of a player to the camera, for the depth error")
    ap.add_argument("--labels", type=Path, default=None, help="label file (one input only)")
    args = ap.parse_args()
    true_depth = {int(k): float(v) for k, v in (s.split("=") for s in args.true_depth)}
    default_width = float(load_config()["source"].get("width", 1280))

    tables: dict[str, list[dict]] = {"Task 3: tracks": [], "Task 3: labelled": []}
    for label, log, info in logs_for_inputs(args, "identity", "game"):
        df = pd.read_csv(log)
        if not player_ids(df):
            print(f"{log}: no players found", file=sys.stderr)
            continue
        width = float(info.get("width", default_width))
        tables["Task 3: tracks"] += track_metrics(df, label, width, args.jump, args.players, true_depth)
        lab_path = args.labels or LABEL_DIR / f"{clip_stem(label, info)}.identity.csv"
        labels = read_labels(lab_path)
        if labels is None:
            print(f"{label}: no {lab_path}, skipping the labelled metrics", file=sys.stderr)
            continue
        tables["Task 3: labelled"].append(labelled_metrics(df, labels, label, args.gate * width,
                                                           args.settle_s, args.absent_s))
    write_tables({k: v for k, v in tables.items() if v}, args.markdown)


if __name__ == "__main__":
    main()
