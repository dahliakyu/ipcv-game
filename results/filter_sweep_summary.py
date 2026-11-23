"""Task 2 filter sweep: EMA against nine One Euro settings, from the logs in
logs/task2_motion/filter_sweep/ (python results/filter_sweep_summary.py).

Still jitter: RMS second difference of every trusted keypoint over all frames
of the two standing clips (face_still_2m, face_still_3m), raw and smoothed.
Fast moves: filter delay and lag on the punch and beam clips (Shoot_test,
Laser_test), as tools/eval_motion.py computes them."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "tools"))
from eval_common import rms, second_difference  # noqa: E402
from eval_motion import evaluate, segments  # noqa: E402

LOGS = ROOT / "logs" / "task2_motion" / "filter_sweep"
TAGS = ["ema"] + [f"oe{m}-{b}" for m in ("0.5", "1.0", "2.0") for b in ("0.003", "0.01", "0.03")]

rows = []
for tag in TAGS:
    raw_d, sm_d = [], []
    for clip in ("face_still_2m", "face_still_3m"):
        kp = dict(np.load(LOGS / f"motion-{clip}-{tag}.kpts.npz"))
        for pid in np.unique(kp["pid"]):
            s = kp["pid"] == pid
            good = kp["conf"][s] >= float(kp["thr"])
            seg = segments(kp["frame_id"][s])
            same = np.r_[False, seg[2:] == seg[:-2], False]
            for k in range(good.shape[1]):
                for arr, out in ((kp["raw"][s][:, k], raw_d), (kp["smooth"][s][:, k], sm_d)):
                    d, idx = second_difference(arr.astype(float), good[:, k])
                    out.append(d[same[idx]])
    raw, sm = rms(np.concatenate(raw_d)), rms(np.concatenate(sm_d))
    fast = []
    for clip in ("Shoot_test", "Laser_test"):
        log = LOGS / f"motion-{clip}-{tag}.csv"
        kp = dict(np.load(log.with_suffix(".kpts.npz")))
        fast += evaluate(pd.read_csv(log), kp, clip)["smoothing"]
    fast = pd.DataFrame(fast)
    name = ("EMA (tau 0.048 s)" if tag == "ema"
            else "One Euro, min cutoff {} Hz, beta {}".format(*tag[2:].split("-")))
    rows.append({"filter": name, "still jitter raw px": raw, "still jitter smoothed px": sm,
                 "change %": 100 * (sm / raw - 1),
                 "fast moves: delay ms": fast["filter delay ms"].mean(),
                 "fast moves: lag px": fast["lag px"].mean(),
                 "fast moves: lag p95 px": fast["lag p95 px"].mean()})
table = pd.DataFrame(rows).round(2)
lines = ["| " + " | ".join(table.columns) + " |", "|" + " --- |" * len(table.columns)]
lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)]
out = "### Task 2: filter sweep\n\n" + "\n".join(lines) + "\n"
(ROOT / "task2_filter_sweep.md").write_text(out)
print(out)
