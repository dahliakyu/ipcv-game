"""tools/eval_motion.py metrics on synthetic keypoints."""

import sys

import numpy as np
import pandas as pd
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import REPO_ROOT
from ipcvgame.core.filters import EMA

sys.path.insert(0, str(REPO_ROOT / "tools"))   # the eval scripts import tools/eval_common.py
import eval_motion  # noqa: E402

DT = 1 / 30


def base_pose() -> np.ndarray:
    """A standing skeleton, shoulders 100 px apart."""
    p = np.zeros((sk.K, 2))
    p[:, 0] = np.linspace(400, 600, sk.K)
    p[:, 1] = np.linspace(100, 700, sk.K)
    p[sk.L_SHOULDER] = (550, 250)
    p[sk.R_SHOULDER] = (450, 250)
    return p


def synthetic(n: int = 30):
    """Frames 0..n-1: standing still, the raw keypoints shake +-1 px in x
    around the smoothed ones. Frames n+1..2n (one frame skipped, so a new
    segment): everything moves 20 px per frame, raw = smoothed. The left
    wrist drops below the confidence threshold for 2 frames while still
    (held: bridged) and for 15 frames while moving (longer than hold_s: lost
    after it)."""
    fid = np.r_[np.arange(n), np.arange(n + 1, 2 * n + 1)]
    t = fid * DT
    base = base_pose()
    raw, smooth, conf, valid = [], [], [], []
    for k, f in enumerate(fid):
        if k < n:
            shake = np.zeros_like(base)
            shake[:, 0] = 1.0 if f % 2 == 0 else -1.0
            raw.append(base + shake)
            smooth.append(base.copy())
        else:
            moved = base + [20.0 * (k - n), 0.0]
            raw.append(moved)
            smooth.append(moved.copy())
        c = np.full(sk.K, 0.9)
        v = np.ones(sk.K, bool)
        if 10 <= k < 12 or n + 5 <= k < n + 20:
            c[sk.L_WRIST] = 0.1
        if n + 15 <= k < n + 20:              # after hold_s (0.33 s = 10 frames): not valid
            v[sk.L_WRIST] = False
        conf.append(c)
        valid.append(v)
    kp = {"frame_id": fid, "t": t, "pid": np.ones(len(fid), int), "raw": np.array(raw),
          "conf": np.array(conf), "smooth": np.array(smooth), "valid": np.array(valid),
          "thr": 0.35, "hold_s": 0.33, "tau": 0.048}
    df = pd.DataFrame({"frame_id": fid, "t": t, "ms_pose": 10.0, "ms_identity": 1.0, "ms_motion": 0.1})
    return df, kp


def test_jitter_still_and_moving():
    df, kp = synthetic()
    (r,) = eval_motion.evaluate(df, kp, "synthetic")["smoothing"]
    assert r["still frames"] == 29                         # frame 0 has no speed yet
    assert r["jitter still raw px"] == pytest.approx(4.0)  # second difference of a +-1 alternation
    assert r["jitter still smooth px"] == pytest.approx(0.0)
    assert r["jitter moving raw px"] == pytest.approx(0.0)  # steady motion cancels out
    assert r["to signal ms"] == pytest.approx(11.1)
    assert r["pose ms"] == pytest.approx(10.0)


def test_detection_and_gaps():
    df, kp = synthetic()
    (r,) = eval_motion.evaluate(df, kp, "synthetic")["keypoints"]
    n_kpts = 60 * sk.K
    assert r["frames"] == 60
    assert r["kpt %"] == pytest.approx(100 * (1 - 17 / n_kpts))
    assert r["upper gaps"] == 2
    assert r["bridged %"] == pytest.approx(50.0)           # the short one is held, the long one is not
    assert r["gap max s"] == pytest.approx(16 * DT)        # last trusted frame to the next trusted one


def test_per_keypoint_table():
    df, kp = synthetic()
    (r,) = eval_motion.evaluate(df, kp, "synthetic")["per keypoint"]
    assert r["nose %"] == pytest.approx(100.0)
    assert r["wrists %"] == pytest.approx(100 * (1 - 17 / 120))


def test_filter_delay_of_a_shifted_track():
    t = np.arange(60) * DT
    raw = np.stack([600.0 * t, np.zeros_like(t)], axis=1)
    smooth = np.stack([600.0 * (t - 0.05), np.zeros_like(t)], axis=1)
    ok = np.ones(len(t), bool)
    assert eval_motion.filter_delay(t, raw, smooth, ok, np.ones(len(t))) == pytest.approx(50.0, abs=2.0)


def test_filter_delay_of_the_ema_matches_theory():
    """A discrete EMA follows a ramp dt (1 - a) / a behind, a = 1 - exp(-dt / tau):
    at 30 FPS and tau = 0.048 s, a = 0.5, so one frame (33 ms)."""
    t = np.arange(90) * DT
    raw = np.stack([600.0 * t, np.zeros_like(t)], axis=1)
    ema = EMA(0.048)
    smooth = np.array([ema(r, ti) for r, ti in zip(raw, t, strict=True)])
    ok = np.r_[np.zeros(15, bool), np.ones(75, bool)]       # after the start-up transient
    a = 1 - np.exp(-DT / 0.048)
    expected = DT * (1 - a) / a * 1000
    assert eval_motion.filter_delay(t, raw, smooth, ok, np.ones(len(t))) == pytest.approx(expected, abs=2.0)
