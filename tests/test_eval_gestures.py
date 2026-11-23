"""tools/eval_gestures.py metrics on a synthetic log and labels."""

import sys

import numpy as np
import pandas as pd
import pytest

from ipcvgame.core.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))   # the eval scripts import tools/eval_common.py
import eval_gestures  # noqa: E402


def synthetic_log() -> pd.DataFrame:
    """10 s at 10 FPS; P1 on the left, P2 on the right. Events: P1 punch at
    1.0 s, P2 punch at 1.3 s, P1 arms crossed at 3.0 s, P1 punch at 7.0 s
    (nobody meant it)."""
    n = 100
    df = pd.DataFrame({"frame_id": np.arange(n), "t": np.arange(n) * 0.1, "ms_total": 30.0,
                       "p1_status": "active", "p1_cx": 300.0, "p1_cy": 400.0,
                       "p2_status": "active", "p2_cx": 900.0, "p2_cy": 400.0,
                       "events": None, "game_p1_actions": None, "game_p2_actions": None})
    df = df.astype({"events": object, "game_p1_actions": object, "game_p2_actions": object})
    for f, ev, a1, a2 in [(10, "p1:punch", "bolt", None), (13, "p2:punch", None, "bolt"),
                          (30, "p1:arms_crossed", "shield", None), (70, "p1:punch", "bolt", None)]:
        df.loc[f, "events"] = ev
        df.loc[f, "game_p1_actions"] = a1
        df.loc[f, "game_p2_actions"] = a2
    return df


def labels() -> pd.DataFrame:
    """Intended: P1 punch from 0.9 s, P2 punch from 1.2 s, P1 arms crossed
    from 2.8 s, P1 hands up from 8.0 s (never recognised)."""
    return pd.DataFrame([(9, 0.9, "punch", 310.0, 380.0), (12, 1.2, "punch", 880.0, 390.0),
                         (28, 2.8, "arms_crossed", 290.0, 420.0), (80, 8.0, "hands_up", 300.0, 300.0)],
                        columns=["frame_id", "t", "label", "x", "y"])


def test_events_without_labels():
    p1, p2 = eval_gestures.event_metrics(synthetic_log(), "s", 0.5)
    assert (p1["punch"], p1["arms_crossed"], p1["hands_up"]) == (2, 1, 0)
    assert (p1["bolts"], p1["shields"]) == (2, 1)
    assert p1["active min"] == pytest.approx(9.9 / 60)
    assert p1["event→screen ms"] == pytest.approx(30.0)
    assert p1["with other ±s"] == 1 and p2["with other ±s"] == 1   # the punches 0.3 s apart


def test_recognition_with_labels():
    rows, both = eval_gestures.recognition_metrics(synthetic_log(), labels(), "s", 0.2, 1.0, 0.5)
    r = {row["gesture"]: row for row in rows}
    assert (r["punch"]["intended"], r["punch"]["recognised"], r["punch"]["false triggers"]) == (2, 2, 1)
    assert r["punch"]["precision %"] == pytest.approx(200 / 3)
    assert r["punch"]["latency ms"] == pytest.approx(100.0)
    assert r["punch"]["to screen ms"] == pytest.approx(130.0)
    assert r["arms_crossed"]["latency ms"] == pytest.approx(200.0)
    assert r["hands_up"]["missed %"] == pytest.approx(100.0)
    assert both == {"clip": "s", "two-player trials": 1, "both recognised %": pytest.approx(100.0)}


def test_wrong_gesture_counts_as_confused():
    lab = pd.DataFrame([(9, 0.9, "hands_up", 310.0, 380.0)], columns=["frame_id", "t", "label", "x", "y"])
    rows, _ = eval_gestures.recognition_metrics(synthetic_log(), lab, "s", 0.2, 1.0, 0.5)
    punch = next(r for r in rows if r["gesture"] == "punch")
    assert punch["false triggers"] == 3 and punch["of which other gesture"] == 1


def test_events_after_the_end_label_are_not_scored():
    lab = pd.concat([labels(), pd.DataFrame([(50, 5.0, "_end", np.nan, np.nan)], columns=labels().columns)])
    rows, _ = eval_gestures.recognition_metrics(synthetic_log(), lab, "s", 0.2, 1.0, 0.5)
    punch = next(r for r in rows if r["gesture"] == "punch")
    assert punch["false triggers"] == 0          # the unlabelled punch at 7.0 s is after the end
