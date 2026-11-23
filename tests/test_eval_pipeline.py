"""tools/eval_pipeline.py and tools/eval_common.py helpers on synthetic data."""

import argparse
import sys

import numpy as np
import pandas as pd
import pytest

from ipcvgame.core.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))   # the eval scripts import tools/eval_common.py
import eval_common  # noqa: E402
import eval_pipeline  # noqa: E402


def synthetic_log(n: int = 100) -> pd.DataFrame:
    """25 ms per frame at a 30 FPS capture, one 300 ms freeze at frame 50
    (also a capture gap), and a failing face stage on frames 60-61."""
    ms = np.full(n, 25.0)
    ms[50] = 300.0
    t = np.arange(n) / 30.0
    t[50:] += 0.3
    df = pd.DataFrame({"frame_id": np.arange(n), "t": t, "ms_total": ms, "ms_pose": 10.0,
                       "p1_status": "active",
                       "motion_p1_l_wrist_speed": np.linspace(0, 9.9, n), "pipeline_failed": None})
    df = df.astype({"pipeline_failed": object})
    df.loc[60:61, "pipeline_failed"] = "face"
    return df


def test_frame_rate_freezes_and_failures():
    r = eval_pipeline.frame_rate(synthetic_log(), "s", {"build_s": 3.0}, 200.0)
    assert r["frames"] == 100
    assert r["ms median"] == pytest.approx(25.0)
    assert r["FPS mean"] == pytest.approx(1000 / (25 * 0.99 + 3.0))
    assert r["frames > 200 ms"] == 1 and r["capture gaps > 200 ms"] == 1
    assert r["failed frames"] == 2
    assert r["build s"] == 3.0 and np.isnan(r["startup s"])


def test_stages_and_fast_motion():
    df = synthetic_log()
    s = eval_pipeline.stages(df, "s")
    assert s["pose med"] == pytest.approx(10.0) and s["total med"] == pytest.approx(25.0)
    (m,) = eval_pipeline.fast_motion(df, "s")
    assert m["wrist max sw/s"] == pytest.approx(9.9)
    assert m["wrist samples"] == 100


def test_degrade_darkens_and_adds_noise():
    img = np.full((40, 40, 3), 200, np.uint8)
    rng = np.random.default_rng(0)
    assert (eval_common.degrade(img, 0.5, 0.0, rng) == 100).all()
    noisy = eval_common.degrade(img, 0.5, 8.0, rng).astype(float)
    assert noisy.mean() == pytest.approx(100.0, abs=1.0) and noisy.std() == pytest.approx(8.0, abs=1.0)


def test_event_list_and_labels():
    df = pd.DataFrame({"frame_id": [0, 1], "t": [0.0, 0.1], "events": [None, "p1:punch;p2:hands_up"]})
    ev = eval_common.event_list(df)
    assert ev[["pid", "name"]].values.tolist() == [[1, "punch"], [2, "hands_up"]]
    args = argparse.Namespace(tag="tau01", light=0.3, noise=8.0)
    assert eval_common.clip_label("session", args) == "session-tau01-light0.3-noise8"
    assert eval_common.clip_stem("session-tau01", {}) == "session"
