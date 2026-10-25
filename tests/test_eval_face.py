"""tools/eval_face.py metrics on a synthetic run log."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from ipcvgame.core.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("eval_face", REPO_ROOT / "tools" / "eval_face.py")
eval_face = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_face)


def synthetic_log(n: int = 40) -> pd.DataFrame:
    """P1 moves steadily right; the raw mouth point shakes +-1 px around the
    smoothed one. The last 10 frames are flow (6) and hold (4)."""
    i = np.arange(n)
    x = 100 + 2.0 * i
    shake = np.where(i % 2 == 0, 1.0, -1.0)
    src = ["detect"] * (n - 10) + ["flow"] * 6 + ["hold"] * 4
    det = np.array([s == "detect" for s in src])
    return pd.DataFrame({
        "frame_id": i, "ms_face": 2.0, "p1_status": "active",
        "face_p1_source": src, "face_p1_scale": 100.0,
        "face_p1_mouth_x": x, "face_p1_mouth_y": 50.0,
        "face_p1_mouth_raw_x": np.where(det, x + shake, np.nan),
        "face_p1_mouth_raw_y": np.where(det, 50.0, np.nan),
        "face_p1_crop_px": 150.0,
    })


def test_metrics_on_known_signal():
    (r,) = eval_face.evaluate(synthetic_log(), "synthetic")
    assert r["frames"] == 40
    assert (r["detect %"], r["flow %"], r["hold %"], r["no face %"]) == (75.0, 15.0, 10.0, 0.0)
    assert r["jitter raw px"] == pytest.approx(4.0)       # second difference of +-1 alternation
    assert r["jitter smooth px"] == pytest.approx(0.0)    # steady motion cancels out
    assert r["jitter raw %w"] == pytest.approx(4.0)       # face 100 px wide
    assert r["lag px"] == pytest.approx(1.0) and r["lag %w"] == pytest.approx(1.0)
    assert r["face ms"] == pytest.approx(2.0)


def test_missing_frames_count_as_no_face():
    df = synthetic_log()
    df.loc[:4, "face_p1_source"] = np.nan                  # face not found yet
    (r,) = eval_face.evaluate(df, "synthetic")
    assert r["no face %"] == pytest.approx(12.5)


def test_markdown_table():
    table = eval_face.markdown(eval_face.evaluate(synthetic_log(), "synthetic"))
    assert table.splitlines()[0].startswith("| clip | player | frames | detect %")
    assert "| synthetic | P1 | 40 |" in table
