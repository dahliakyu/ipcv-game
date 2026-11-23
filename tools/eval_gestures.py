"""Task 4 evaluation: gesture events and game responses per clip and player.

    python tools/label_clip.py data/clips/session.mp4 --mode gestures   # once per clip: intended actions
    python tools/eval_gestures.py data/clips/*.mp4                      # run each clip, then evaluate
    python tools/eval_gestures.py logs/eval/gestures-session.csv        # evaluate an earlier run
    python tools/eval_gestures.py data/clips/shield.mp4 --tag cross12 --set gestures.cross_max_wrist_dist=1.2
    python tools/eval_gestures.py data/clips/session.mp4 --light 0.3 --noise 8

A clip is run headless up to the game stage without the intro (clips are
recorded without it; --intro keeps it), and its per-frame log is written to
logs/eval/gestures-<label>.csv.

Tables:
- "events" (no labels needed): events per gesture and minute, the game's
  responses, time from the event's frame to the screen, simultaneous
  events of both players.
- "recognition" (with data/labels/<clip>.gestures.csv from
  tools/label_clip.py): per gesture, intended actions recognised, missed,
  false triggers, precision, latency from the action's start to the event.
- "both players": labelled actions of both players close together, and
  whether both were recognised.
Metric definitions: the protocol of Task 4 in docs/project_report.md.
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
    event_list,
    logs_for_inputs,
    pct,
    player_ids,
    ratio,
    read_labels,
    write_tables,
)

GESTURES = ("punch", "arms_crossed", "hands_up")
# Game actions that answer a gesture event (game.py _act and the facing check).
RESPONSES = {"bolt": "bolts", "shield": "shields", "beam": "beams", "no_orb": "no orb"}


def actions(df: pd.DataFrame, pid: int) -> list[str]:
    col = f"game_p{pid}_actions"
    if col not in df:
        return []
    return [a for v in df[col].dropna().astype(str) for a in v.split("+") if a]


def event_metrics(df: pd.DataFrame, label: str, sim_s: float) -> list[dict]:
    """The label-free table: one row per player."""
    ev = event_list(df)
    ms_total = df.set_index("frame_id")["ms_total"] if "ms_total" in df else pd.Series(dtype=float)
    rows = []
    for pid in player_ids(df):
        active = df[f"p{pid}_status"] == "active"
        dt = df["t"].diff().where(active).clip(upper=0.5)      # a gap is not time in play
        minutes = float(dt.sum()) / 60.0
        e = ev[ev["pid"] == pid]
        other = ev[ev["pid"] != pid]["t"].to_numpy()
        row = {"clip": label, "player": f"P{pid}", "active min": minutes}
        for g in GESTURES:
            row[g] = int((e["name"] == g).sum())
        row["events /min"] = len(e) / minutes if minutes > 0 else float("nan")
        acts = actions(df, pid)
        for a, name in RESPONSES.items():
            row[name] = acts.count(a)
        row["not facing"] = sum(a.startswith("not_facing_") for a in acts)
        row["event→screen ms"] = pct(ms_total.reindex(e["frame_id"]).to_numpy(float), 50)
        row["with other ±s"] = int(sum(np.any(np.abs(other - t) <= sim_s) for t in e["t"]))
        rows.append(row)
    return rows


def label_pids(df: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """The player each label is on: the active player whose box centre is
    nearest to the click in that frame (NaN when none is active)."""
    log = df.set_index("frame_id")
    pids = player_ids(df)
    out = []
    for fid, x, y in labels[["frame_id", "x", "y"]].itertuples(index=False):
        best, best_d = np.nan, np.inf
        if fid in log.index:
            r = log.loc[fid]
            for pid in pids:
                if r.get(f"p{pid}_status") == "active":
                    d = np.hypot(float(r[f"p{pid}_cx"]) - x, float(r[f"p{pid}_cy"]) - y)
                    if d < best_d:
                        best, best_d = pid, d
        out.append(best)
    return pd.Series(out, index=labels.index)


def match(df: pd.DataFrame, labels: pd.DataFrame, pre_s: float, window_s: float) -> tuple[pd.DataFrame,
                                                                                          pd.DataFrame]:
    """Pair every labelled action with the first unused event of the same
    gesture and player from pre_s before to window_s after its start.
    Returns the labels (with pid, event_t, latency_ms; event_t NaN when
    missed) and the events (with matched, and confused: an unmatched event
    inside another gesture's label window of the same player)."""
    t0 = df["t"].astype(float)
    t_of = dict(zip(df["frame_id"], t0, strict=True))
    lab = labels[labels["label"].isin(GESTURES)].copy()
    lab["t"] = lab["frame_id"].map(t_of)                 # log time, the same clock as the events
    lab = lab.dropna(subset=["t"]).sort_values("t")
    lab["pid"] = label_pids(df, lab)
    ev = event_list(df)
    # A label "_end" marks the last labelled frame: events after it are not
    # scored (e.g. only the move tests at the start of a gameplay clip).
    end = labels.loc[labels["label"] == "_end", "frame_id"]
    if len(end):
        ev = ev[ev["frame_id"] <= int(end.max())].reset_index(drop=True)
    ev["matched"] = False
    lab["event_t"] = np.nan
    for i, r in lab.iterrows():
        cand = ev[(~ev["matched"]) & (ev["pid"] == r["pid"]) & (ev["name"] == r["label"])
                  & (ev["t"] >= r["t"] - pre_s) & (ev["t"] <= r["t"] + window_s)]
        if len(cand):
            j = cand.index[0]
            ev.loc[j, "matched"] = True
            lab.loc[i, "event_t"] = ev.loc[j, "t"]
    lab["latency_ms"] = (lab["event_t"] - lab["t"]) * 1000.0
    ev["confused"] = False
    for j, e in ev[~ev["matched"]].iterrows():
        near = lab[(lab["pid"] == e["pid"]) & (lab["label"] != e["name"])
                   & (e["t"] >= lab["t"] - pre_s) & (e["t"] <= lab["t"] + window_s)]
        ev.loc[j, "confused"] = len(near) > 0
    return lab, ev


def recognition_metrics(df: pd.DataFrame, labels: pd.DataFrame, label: str, pre_s: float,
                        window_s: float, sim_s: float) -> tuple[list[dict], dict]:
    lab, ev = match(df, labels, pre_s, window_s)
    ms = df.set_index("frame_id")["ms_total"] if "ms_total" in df else pd.Series(dtype=float)
    ev_ms = dict(zip(ev["t"], ms.reindex(ev["frame_id"]).to_numpy(float), strict=False))
    rows = []
    for g in GESTURES:
        lg, eg = lab[lab["label"] == g], ev[ev["name"] == g]
        tp = int(lg["event_t"].notna().sum())
        fp = int((~eg["matched"]).sum())
        lat = lg["latency_ms"].dropna().to_numpy()
        screen = lat + np.array([ev_ms.get(t, np.nan) for t in lg["event_t"].dropna()])
        rows.append({
            "clip": label, "gesture": g, "intended": len(lg), "recognised": tp,
            "missed %": ratio(len(lg) - tp, len(lg)),
            "false triggers": fp, "of which other gesture": int(eg.loc[~eg["matched"], "confused"].sum()),
            "precision %": ratio(tp, tp + fp),
            "latency ms": pct(lat, 50), "latency p95 ms": pct(lat, 95),
            "to screen ms": pct(screen, 50),
        })
    # Both players: labelled actions of different players within sim_s.
    trials = ok = 0
    lab = lab.dropna(subset=["pid"])
    used: set[int] = set()
    for i, a in lab.iterrows():
        if i in used:
            continue
        other = lab[(lab["pid"] != a["pid"]) & ((lab["t"] - a["t"]).abs() <= sim_s) & ~lab.index.isin(used)]
        if len(other):
            j = other.index[0]
            used |= {i, j}
            trials += 1
            ok += int(pd.notna(a["event_t"]) and pd.notna(lab.loc[j, "event_t"]))
    both = {"clip": label, "two-player trials": trials, "both recognised %": ratio(ok, trials)}
    return rows, both


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, intro=False)
    ap.add_argument("--window-s", type=float, default=1.0,
                    help="labels: an event up to this long after an action's start belongs to it")
    ap.add_argument("--pre-s", type=float, default=0.2,
                    help="labels: an event up to this long before the labelled start still counts "
                         "(the start frame is a judgement call)")
    ap.add_argument("--sim-s", type=float, default=0.5,
                    help="actions of both players this close in time count as simultaneous")
    ap.add_argument("--labels", type=Path, default=None, help="label file (one input only)")
    args = ap.parse_args()

    tables: dict[str, list[dict]] = {"Task 4: events": [], "Task 4: recognition": [],
                                     "Task 4: both players": []}
    for label, log, info in logs_for_inputs(args, "gestures", "game"):
        df = pd.read_csv(log)
        if not player_ids(df):
            print(f"{log}: no players found", file=sys.stderr)
            continue
        tables["Task 4: events"] += event_metrics(df, label, args.sim_s)
        lab_path = args.labels or LABEL_DIR / f"{clip_stem(label, info)}.gestures.csv"
        labels = read_labels(lab_path)
        if labels is None:
            print(f"{label}: no {lab_path}, skipping the labelled metrics", file=sys.stderr)
            continue
        rows, both = recognition_metrics(df, labels, label, args.pre_s, args.window_s, args.sim_s)
        tables["Task 4: recognition"] += rows
        tables["Task 4: both players"].append(both)
    write_tables({k: v for k, v in tables.items() if v}, args.markdown)


if __name__ == "__main__":
    main()
