"""Task 2 evaluation: keypoint detection, smoothing and motion signals per clip and player.

    python tools/eval_motion.py data/clips/*.mp4                       # run each clip, then evaluate
    python tools/eval_motion.py logs/eval/motion-session.csv           # evaluate an earlier run
    python tools/eval_motion.py data/clips/session.mp4 --tag tau01 --set motion.tau=0.1
    python tools/eval_motion.py data/clips/session.mp4 --light 0.3 --noise 8   # low light
    python tools/eval_motion.py data/clips/*.mp4 --markdown results/motion.md

A clip is run headless up to the motion stage. Besides the per-frame log
(logs/eval/motion-<label>.csv), every player's keypoints are saved to
motion-<label>.kpts.npz: the raw detection, its confidences, the smoothed
keypoints and their validity. The metrics need them per keypoint, which the
log (per-player means) does not have; a log without its .npz cannot be
evaluated. Metric definitions: the protocol of Task 2 in docs/project_report.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from eval_common import (
    add_common_args,
    logs_for_inputs,
    mean,
    pct,
    ratio,
    rms,
    second_difference,
    write_tables,
)

from ipcvgame.core import skeleton as sk

# Keypoint groups for the tables: left and right pooled.
GROUPS = {
    "nose": [sk.NOSE], "eyes": [sk.L_EYE, sk.R_EYE], "ears": [sk.L_EAR, sk.R_EAR],
    "shoulders": [sk.L_SHOULDER, sk.R_SHOULDER], "elbows": [sk.L_ELBOW, sk.R_ELBOW],
    "wrists": [sk.L_WRIST, sk.R_WRIST], "hips": [sk.L_HIP, sk.R_HIP],
    "knees": [sk.L_KNEE, sk.R_KNEE], "ankles": [sk.L_ANKLE, sk.R_ANKLE],
}
# The keypoints the gesture rules read (Task 4).
UPPER = [sk.L_SHOULDER, sk.R_SHOULDER, sk.L_ELBOW, sk.R_ELBOW, sk.L_WRIST, sk.R_WRIST]
WRISTS = [sk.L_WRIST, sk.R_WRIST]


class KeypointCollector:
    """Per frame and player with a detection: raw keypoints, confidences,
    smoothed keypoints and validity, from the Player and Motion outputs."""

    def __init__(self):
        self.rows: dict[str, list] = {k: [] for k in ("frame_id", "t", "pid", "raw", "conf",
                                                      "smooth", "valid")}
        self.params: dict = {}

    def __call__(self, state, pipeline) -> None:
        m = pipeline.motion
        self.params = {"thr": getattr(m, "kpt_conf_threshold", np.nan),
                       "hold_s": getattr(m, "hold_s", np.nan), "tau": getattr(m, "tau", np.nan)}
        for pid, mo in state.motion.items():
            p = state.players.get(pid)
            if p is None or p.det is None:
                continue
            r = self.rows
            r["frame_id"].append(state.frame.frame_id)
            r["t"].append(state.frame.t)
            r["pid"].append(pid)
            r["raw"].append(p.det.kpts.astype(np.float32))
            r["conf"].append(p.det.conf.astype(np.float32))
            r["smooth"].append(mo.kpts.astype(np.float32))
            r["valid"].append(mo.valid.astype(bool))

    def save(self, log: Path) -> None:
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        np.savez_compressed(kpts_path(log), **arrays, **{k: np.float64(v) for k, v in self.params.items()})


def kpts_path(log: Path) -> Path:
    return log.with_suffix(".kpts.npz")


def segments(frame_id: np.ndarray) -> np.ndarray:
    """Segment number per row: a new segment wherever a frame is skipped
    (the player was not detected), so differences never span a gap."""
    return np.cumsum(np.r_[True, np.diff(frame_id) != 1])


def centred_mean(xy: np.ndarray, seg: np.ndarray, half: int = 2) -> np.ndarray:
    """Mean over the 2 * half + 1 frames around each frame, within its segment
    (fewer at the segment ends): a reference track with the jitter averaged
    out, for telling still frames from moving ones."""
    out = np.empty_like(xy)
    for s_ in np.unique(seg):
        idx = np.flatnonzero(seg == s_)
        block = xy[idx]
        csum = np.cumsum(np.concatenate([np.zeros((1, *block.shape[1:])), block]), axis=0)
        lo = np.clip(np.arange(len(idx)) - half, 0, len(idx))
        hi = np.clip(np.arange(len(idx)) + half + 1, 0, len(idx))
        out[idx] = (csum[hi] - csum[lo]) / (hi - lo)[:, None, None]
    return out


def gap_runs(good: np.ndarray, valid: np.ndarray, seg: np.ndarray, t: np.ndarray) -> list[tuple[float, bool]]:
    """Dropouts of one keypoint: runs of frames below the confidence
    threshold, between two trusted frames of the same segment. Returns
    (duration s, bridged) per run, where duration runs from the last trusted
    frame to the next one and bridged means the smoothed output stayed valid
    (the hold covered it). Runs at a segment's start or end are left out:
    there the outcome is unknown."""
    out = []
    i, n = 0, len(good)
    while i < n:
        if good[i]:
            i += 1
            continue
        j = i
        while j < n and not good[j] and seg[j] == seg[i]:
            j += 1
        starts_inside = i > 0 and seg[i - 1] == seg[i] and good[i - 1]
        ends_inside = j < n and seg[j] == seg[i] and good[j]
        if starts_inside and ends_inside:
            out.append((float(t[j] - t[i - 1]), bool(valid[i:j].all())))
        i = max(j, i + 1)
    return out


def filter_delay(t: np.ndarray, raw: np.ndarray, smooth: np.ndarray, ok: np.ndarray,
                 seg: np.ndarray, max_ms: float = 250.0, step_ms: float = 2.0) -> float:
    """The time shift (ms) that best lines the smoothed track up with the raw
    one: argmin over d of mean |smooth(t) - raw(t - d)|, with raw linearly
    interpolated in time. An EMA follows a steady motion about its time
    constant tau behind. Only rows where ok (trusted, moving) count."""
    delays = np.arange(0.0, max_ms + step_ms, step_ms) / 1000.0
    err = np.zeros(len(delays))
    cnt = np.zeros(len(delays))
    for s in np.unique(seg):
        m = seg == s
        if m.sum() < 5:
            continue
        ts, rs, ss, oks = t[m], raw[m], smooth[m], ok[m]
        for k, d in enumerate(delays):
            q = ts - d
            use = oks & (q >= ts[0])
            if not use.any():
                continue
            shifted = np.stack([np.interp(q[use], ts, rs[:, c]) for c in range(2)], axis=1)
            err[k] += np.linalg.norm(ss[use] - shifted, axis=1).sum()
            cnt[k] += use.sum()
    if not cnt.any():
        return float("nan")
    e = np.divide(err, cnt, out=np.full(len(err), np.inf), where=cnt > 0)
    return float(delays[int(np.argmin(e))] * 1000.0)


def evaluate(df: pd.DataFrame, kp: dict, label: str, still_sw: float = 0.5) -> dict[str, list[dict]]:
    """Three tables: keypoints (detection and missing-keypoint handling),
    smoothing (jitter, lag, delay, timing) and per keypoint group."""
    thr = float(kp.get("thr", 0.35))
    tables: dict[str, list[dict]] = {"keypoints": [], "smoothing": [], "per keypoint": []}
    pids = np.asarray(kp["pid"])
    ms = {s: df[f"ms_{s}"].to_numpy(float) if f"ms_{s}" in df else np.full(len(df), np.nan)
          for s in ("pose", "identity", "motion")}
    to_signal = ms["pose"] + ms["identity"] + ms["motion"]
    pooled_good = []
    for pid in sorted(set(pids.tolist())):
        sel = pids == pid
        order = np.argsort(np.asarray(kp["frame_id"])[sel], kind="stable")
        fid = np.asarray(kp["frame_id"])[sel][order]
        t = np.asarray(kp["t"], float)[sel][order]
        raw = np.asarray(kp["raw"], float)[sel][order]
        smooth = np.asarray(kp["smooth"], float)[sel][order]
        good = np.asarray(kp["conf"])[sel][order] >= thr
        valid = np.asarray(kp["valid"])[sel][order]
        n = len(fid)
        pooled_good.append(good)
        seg = segments(fid)

        # --- detection and missing keypoints
        # Missing-keypoint handling over the upper body: the keypoints the
        # gestures read (legs are often out of the frame, which is not a dropout).
        gaps = [g for k in UPPER for g in gap_runs(good[:, k], valid[:, k], seg, t)]
        tables["keypoints"].append({
            "clip": label, "player": f"P{pid}", "frames": n,
            "kpt %": 100 * float(good.mean()),
            "upper body %": 100 * float(good[:, UPPER].mean()),
            "wrists %": 100 * float(good[:, WRISTS].mean()),
            "upper held %": 100 * float((valid & ~good)[:, UPPER].mean()),
            "upper lost %": 100 * float((~valid)[:, UPPER].mean()),
            "upper gaps": len(gaps),
            "bridged %": ratio(sum(b for _, b in gaps), len(gaps)),
            "gap median s": pct([d for d, _ in gaps], 50),
            "gap max s": max((d for d, _ in gaps), default=float("nan")),
        })

        # --- still / moving frames: the speed of the raw keypoints averaged
        # over 5 frames (centred), in shoulder widths per second, the largest
        # over the trusted keypoints. Independent of the filter under test, so
        # every filter is scored on the same still and moving frames.
        ref = centred_mean(raw, seg)
        dt = np.r_[np.nan, np.diff(t)]
        dt[np.r_[True, np.diff(seg) != 0]] = np.nan
        speed = np.linalg.norm(np.diff(ref, axis=0, prepend=ref[:1]), axis=2) / dt[:, None]
        sh = good[:, sk.L_SHOULDER] & good[:, sk.R_SHOULDER]
        sw = np.where(sh, np.linalg.norm(ref[:, sk.L_SHOULDER] - ref[:, sk.R_SHOULDER], axis=1), np.nan)
        both = good & np.r_[[np.zeros(sk.K, bool)], good[:-1]]
        body = np.where(both, speed, np.nan)
        with np.errstate(invalid="ignore", all="ignore"):
            top = np.nanmax(np.where(np.isnan(body), -np.inf, body), axis=1) / sw
        top[~np.isfinite(top)] = np.nan
        still = top < still_sw
        moving = top >= still_sw

        # --- jitter: second difference per keypoint over three consecutive
        # frames (same segment), trusted in all three
        same3 = np.r_[False, (seg[2:] == seg[:-2]), False]
        jit = {("still", "raw"): [], ("still", "smooth"): [], ("moving", "raw"): [], ("moving", "smooth"): []}
        for k in range(sk.K):
            ok = good[:, k]
            for name, xy in (("raw", raw[:, k]), ("smooth", smooth[:, k])):
                d, idx = second_difference(xy, ok)
                keep = same3[idx]
                d, idx = d[keep], idx[keep]
                jit[("still", name)].append(d[still[idx]])
                jit[("moving", name)].append(d[moving[idx]])
        jit = {k: np.concatenate(v) if v else np.array([]) for k, v in jit.items()}

        lag = np.linalg.norm(smooth - raw, axis=2)[good]
        # Filter delay on the wrists while moving (the control signal)
        w_t, w_raw, w_sm, w_ok, w_seg = [], [], [], [], []
        for w in WRISTS:
            w_t.append(t)
            w_raw.append(raw[:, w])
            w_sm.append(smooth[:, w])
            w_ok.append(good[:, w] & moving)
            w_seg.append(seg + (w * 10**7))      # separate segments per wrist
        delay = filter_delay(np.concatenate(w_t), np.concatenate(w_raw), np.concatenate(w_sm),
                             np.concatenate(w_ok), np.concatenate(w_seg))
        rows_in_log = df["frame_id"].isin(fid).to_numpy()
        proc = pct(to_signal[rows_in_log], 50)
        tables["smoothing"].append({
            "clip": label, "player": f"P{pid}",
            "still frames": int(np.nansum(still)),
            "jitter still raw px": rms(jit[("still", "raw")]),
            "jitter still smooth px": rms(jit[("still", "smooth")]),
            "jitter moving raw px": rms(jit[("moving", "raw")]),
            "jitter moving smooth px": rms(jit[("moving", "smooth")]),
            "lag px": mean(lag), "lag p95 px": pct(lag, 95),
            "filter delay ms": delay,
            "to signal ms": proc,
            "signal delay ms": delay + proc,
            "pose ms": mean(ms["pose"][rows_in_log]), "pose ms p95": pct(ms["pose"][rows_in_log], 95),
            "motion ms": mean(ms["motion"][rows_in_log]),
        })

    if pooled_good:
        g = np.concatenate(pooled_good)
        tables["per keypoint"].append(
            {"clip": label, "frames": len(g)} |
            {f"{name} %": 100 * float(g[:, idx].mean()) for name, idx in GROUPS.items()})
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, intro=False)
    ap.add_argument("--still-sw", type=float, default=0.5,
                    help="a frame is 'still' when every trusted keypoint moves slower than this "
                         "(shoulder widths per second), else 'moving'")
    args = ap.parse_args()

    tables: dict[str, list[dict]] = {}
    for label, log, _ in logs_for_inputs(args, "motion", "motion", lambda log: [KeypointCollector()]):
        if not kpts_path(log).exists():
            print(f"{log}: no {kpts_path(log).name}; run the clip with this script", file=sys.stderr)
            continue
        kp = dict(np.load(kpts_path(log)))
        if not len(kp["pid"]):
            print(f"{log}: no player detected", file=sys.stderr)
            continue
        for title, rows in evaluate(pd.read_csv(log), kp, label, args.still_sw).items():
            tables.setdefault(f"Task 2: {title}", []).extend(rows)
    write_tables(tables, args.markdown)


if __name__ == "__main__":
    main()
