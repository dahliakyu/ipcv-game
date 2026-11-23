"""Hand-label a clip for the identity and gesture evaluations.

    python tools/label_clip.py data/clips/crossing.mp4 --mode identity
    python tools/label_clip.py data/clips/session.mp4 --mode gestures

The labels are about the people in the clip, not about what the tracker did,
so one label file scores every run of the clip (other settings, darker
frames). They go to data/labels/<clip>.<mode>.csv (frame_id, t, label, x, y;
x, y in pixels of the mirrored frame, as the pipeline sees it). An existing
file is loaded, so labelling can be done in several sittings.

identity: one frame every --every-s seconds. Click on each person's chest:
    left click = player A, right click = player B (A = whoever is on the left
    in the first frame; keep that choice for the whole clip), middle click =
    anyone else (C: a bystander, or a newcomer who is not one of the two
    players). Mark only the people you can see; a frame you leave with no click means nobody is in
    view. Keys: n / space next frame, b back, x clear this frame, q / Esc save
    and quit.
gestures: every frame. Find the first frame of an intended action, press its
    key, then click on the person doing it: 1 punch, 2 arms_crossed (shield),
    3 hands_up (beam). Label every intended action in the clip: events with
    no label count as false triggers. Keys: d / a one frame forward / back,
    D / A one second, space play / pause, u undo the last label, q / Esc save
    and quit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from ipcvgame.core.config import REPO_ROOT, load_config
from ipcvgame.io.recorder import timestamps_path

GESTURE_KEYS = {ord("1"): "punch", ord("2"): "arms_crossed", ord("3"): "hands_up"}
PERSON_COLOR = {"A": (0, 0, 255), "B": (255, 128, 0), "C": (0, 200, 0)}
PERSON_OF_BUTTON = {cv2.EVENT_LBUTTONDOWN: "A", cv2.EVENT_RBUTTONDOWN: "B", cv2.EVENT_MBUTTONDOWN: "C"}
GESTURE_COLOR = (0, 255, 255)
NOBODY = "-"                               # identity: a frame reviewed with nobody in view


def label_path(clip: Path, mode: str) -> Path:
    return REPO_ROOT / "data" / "labels" / f"{clip.stem}.{mode}.csv"


class Clip:
    """Random access to a clip's frames, mirrored as the pipeline sees them,
    with the same times (the timestamps sidecar, else frame / fps)."""

    def __init__(self, path: Path, mirror: bool):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        self.mirror = mirror
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        ts = timestamps_path(path)
        if ts.exists():
            t = np.loadtxt(ts, delimiter=",", skiprows=1, usecols=1, ndmin=1)
            self.t = t - t[0]
        else:
            self.t = np.arange(self.n) / fps
        self.n = min(self.n, len(self.t))
        self._next = 0                     # index the next sequential read returns

    def read(self, i: int) -> np.ndarray:
        if i != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, image = self.cap.read()
        self._next = i + 1
        if not ok:
            raise SystemExit(f"cannot read frame {i}")
        return cv2.flip(image, 1) if self.mirror else image


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [{"frame_id": int(r["frame_id"]), "t": float(r["t"]), "label": r["label"],
                 "x": float(r["x"]) if r["x"] else None, "y": float(r["y"]) if r["y"] else None}
                for r in csv.DictReader(fh)]


def save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["frame_id"], r["label"]))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame_id", "t", "label", "x", "y"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "t": f"{r['t']:.4f}",
                        "x": "" if r["x"] is None else f"{r['x']:.1f}",
                        "y": "" if r["y"] is None else f"{r['y']:.1f}"})
    print(f"saved {len(rows)} labels to {path}")


def put(canvas: np.ndarray, text: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--mode", choices=["identity", "gestures"], required=True)
    ap.add_argument("--every-s", type=float, default=0.5,
                    help="identity: label one frame every this many seconds")
    ap.add_argument("--width", type=int, default=1280, help="window width (px)")
    ap.add_argument("--out", type=Path, default=None, help="label file (default data/labels/...)")
    args = ap.parse_args()

    cfg = load_config()
    clip = Clip(args.clip, bool(cfg["source"].get("mirror", True)))
    out = args.out or label_path(args.clip, args.mode)
    rows = load(out)
    if args.mode == "identity":
        frames = [int(np.searchsorted(clip.t, s)) for s in np.arange(0.0, clip.t[clip.n - 1], args.every_s)]
        frames = sorted(set(f for f in frames if f < clip.n))
    else:
        frames = list(range(clip.n))
    pos = 0
    if args.mode == "identity" and rows:        # resume after the last labelled frame
        done = {r["frame_id"] for r in rows}
        pos = next((k for k, f in enumerate(frames) if f not in done), len(frames) - 1)

    window = f"label {args.mode}: {args.clip.name}"
    cv2.namedWindow(window)
    state = {"scale": 1.0, "pending": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        if event in PERSON_OF_BUTTON:
            state["click"] = (event, x / state["scale"], y / state["scale"])
    cv2.setMouseCallback(window, on_mouse)

    playing = False
    shown, image = None, None
    while True:
        fid = frames[pos]
        t = float(clip.t[fid])
        if fid != shown:                    # read (and seek) only when the frame changes
            image, shown = clip.read(fid), fid
        state["scale"] = args.width / image.shape[1]

        click = state["click"]
        state["click"] = None
        if click is not None:
            event, x, y = click
            if args.mode == "identity":
                person = PERSON_OF_BUTTON[event]
                rows = [r for r in rows if not (r["frame_id"] == fid and r["label"] in (person, NOBODY))]
                rows.append({"frame_id": fid, "t": t, "label": person, "x": x, "y": y})
            elif state["pending"] is not None:
                rows.append({"frame_id": fid, "t": t, "label": state["pending"], "x": x, "y": y})
                state["pending"] = None

        canvas = cv2.resize(image, None, fx=state["scale"], fy=state["scale"])
        if args.mode == "identity":
            here = [r for r in rows if r["frame_id"] == fid and r["label"] != NOBODY]
            for r in here:
                c = (int(r["x"] * state["scale"]), int(r["y"] * state["scale"]))
                color = PERSON_COLOR.get(r["label"], (0, 200, 0))
                cv2.circle(canvas, c, 10, color, -1)
                cv2.putText(canvas, r["label"], (c[0] + 12, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            put(canvas, f"frame {fid}  t={t:.2f}s  ({pos + 1}/{len(frames)})", 25)
            put(canvas, "L-click A, R-click B, M-click other (C), n/space next, b back, x clear, q save", 50)
        else:
            near = [r for r in rows if abs(r["t"] - t) <= 0.5]
            for r in near:
                c = (int(r["x"] * state["scale"]), int(r["y"] * state["scale"]))
                cv2.circle(canvas, c, 8 if r["frame_id"] == fid else 4, GESTURE_COLOR, -1)
                cv2.putText(canvas, r["label"], (c[0] + 10, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            GESTURE_COLOR, 2)
            pend = f"  -> click the person doing: {state['pending']}" if state["pending"] else ""
            put(canvas, f"frame {fid}/{clip.n - 1}  t={t:.2f}s  labels={len(rows)}{pend}", 25)
            put(canvas, "1 punch 2 arms_crossed 3 hands_up + click | d/a frame, D/A 1 s, "
                        "space play, u undo, q save+quit", 50)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(30 if playing else 20) & 0xFF
        if key in (ord("q"), 27):
            break
        if args.mode == "identity":
            if key in (ord("n"), ord(" ")):
                if not any(r["frame_id"] == fid for r in rows):
                    rows.append({"frame_id": fid, "t": t, "label": NOBODY, "x": None, "y": None})
                pos = min(pos + 1, len(frames) - 1)
            elif key == ord("b"):
                pos = max(pos - 1, 0)
            elif key == ord("x"):
                rows = [r for r in rows if r["frame_id"] != fid]
        else:
            step = {ord("d"): 1, ord("a"): -1}.get(key, 0)
            if key in (ord("D"), ord("A")):
                one_s = max(1, int(np.searchsorted(clip.t, t + 1.0)) - fid)
                step = one_s if key == ord("D") else -one_s
            if key == ord(" "):
                playing = not playing
            if key in GESTURE_KEYS:
                state["pending"] = GESTURE_KEYS[key]
                playing = False
            if key == ord("u") and rows:
                rows.pop()
            if playing and step == 0:
                step = 1
            pos = int(np.clip(pos + step, 0, len(frames) - 1))
    cv2.destroyAllWindows()
    save(out, rows)


if __name__ == "__main__":
    main()
