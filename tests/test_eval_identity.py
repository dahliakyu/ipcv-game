"""tools/eval_identity.py metrics on synthetic logs and labels."""

import sys

import numpy as np
import pandas as pd
import pytest

from ipcvgame.core.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))   # the eval scripts import tools/eval_common.py
import eval_identity  # noqa: E402

W = 1280.0


def make_log(x1, x2, s1=None, s2=None, **extra) -> pd.DataFrame:
    n = len(x1)
    df = pd.DataFrame({"frame_id": np.arange(n), "t": np.arange(n) * 0.1, "ms_identity": 1.0,
                       "p1_status": s1 or ["active"] * n, "p1_reentered": 0,
                       "p1_cx": x1, "p1_cy": 400.0, "p1_depth_m": 2.0,
                       "p2_status": s2 or ["active"] * n, "p2_reentered": 0,
                       "p2_cx": x2, "p2_cy": 400.0, "p2_depth_m": 3.0})
    for k, v in extra.items():
        df[k] = v
    return df


def labels_from(xa, xb, every: int = 5) -> pd.DataFrame:
    rows = []
    for f in range(0, len(xa), every):
        for person, x in (("A", xa[f]), ("B", xb[f])):
            if x is not None:
                rows.append((f, f * 0.1, person, x, 400.0))
        if xa[f] is None and xb[f] is None:
            rows.append((f, f * 0.1, "-", np.nan, np.nan))
    return pd.DataFrame(rows, columns=["frame_id", "t", "label", "x", "y"])


def test_jump_and_order_flip_without_labels():
    # The ids swap people at frame 20: both boxes jump across the frame.
    x1 = np.r_[np.full(20, 300.0), np.full(20, 900.0)]
    x2 = np.r_[np.full(20, 900.0), np.full(20, 300.0)]
    rows = eval_identity.track_metrics(make_log(x1, x2), "s", W, 0.15, 2, {2: 2.5})
    assert [r["jumps"] for r in rows] == [1, 1]
    assert rows[0]["order flips"] == 1
    assert rows[0]["players created"] == 2 and rows[0]["extra ids"] == 0
    assert rows[0]["depth m"] == pytest.approx(2.0) and rows[0]["depth IQR m"] == pytest.approx(0.0)
    assert rows[1]["depth error mm"] == pytest.approx(500.0)


def test_calibration_renumbering_is_not_a_switch():
    x1 = np.r_[np.full(20, 900.0), np.full(20, 300.0)]
    x2 = np.r_[np.full(20, 300.0), np.full(20, 900.0)]
    phase = ["waiting"] * 20 + ["calibration"] * 20
    rows = eval_identity.track_metrics(make_log(x1, x2, game_phase=phase), "s", W, 0.15, 2, {})
    assert [r["jumps"] for r in rows] == [0, 0]
    assert rows[0]["order flips"] == 0


def test_reentry_and_reid_time():
    n = 40
    s2 = ["active"] * 20 + ["lost"] * 10 + ["active"] * 10
    df = make_log(np.full(n, 300.0), np.full(n, 900.0), s2=s2)
    df.loc[30, "p2_reentered"] = 1
    df["identity_pending"] = 0
    df.loc[26:29, "identity_pending"] = 1          # a new track waited 4 frames for OSNet
    rows = eval_identity.track_metrics(df, "s", W, 0.15, 2, {})
    assert rows[1]["re-entries"] == 1 and rows[1]["corrections"] == 0
    assert rows[1]["re-id s"] == pytest.approx(0.4)
    assert rows[1]["lost %"] == pytest.approx(25.0)


def test_labelled_crossing_kept():
    # A walks from the left to the right, B the other way; the ids follow them.
    xa = list(np.linspace(300, 900, 40))
    xb = list(np.linspace(900, 300, 40))
    df = make_log(np.array(xa), np.array(xb))
    r = eval_identity.labelled_metrics(df, labels_from(xa, xb), "s", 0.15 * W, 1.0, 1.0)
    assert r["crossings"] == 1 and r["crossings kept %"] == pytest.approx(100.0)
    assert r["id accuracy %"] == pytest.approx(100.0) and r["id switches"] == 0


def test_labelled_crossing_swapped():
    # The ids stay on their side of the screen: they swap people in the crossing.
    xa = list(np.linspace(300, 900, 40))
    xb = list(np.linspace(900, 300, 40))
    x1 = np.minimum(xa, xb)
    x2 = np.maximum(xa, xb)
    r = eval_identity.labelled_metrics(make_log(x1, x2), labels_from(xa, xb), "s", 0.15 * W, 1.0, 1.0)
    assert r["crossings"] == 1 and r["crossings kept %"] == pytest.approx(0.0)
    assert r["id switches"] == 2                      # each id changed person once
    assert r["id accuracy %"] < 100.0


def test_labelled_reentry_recovered():
    n = 60
    xb = [900.0] * 20 + [None] * 20 + [900.0] * 20     # B leaves for 2 s
    s2 = ["active"] * 20 + ["lost"] * 20 + ["active"] * 20
    df = make_log(np.full(n, 300.0), np.full(n, 900.0), s2=s2)
    r = eval_identity.labelled_metrics(df, labels_from([300.0] * n, xb), "s", 0.15 * W, 1.0, 1.0)
    assert r["re-entries"] == 1 and r["recovered %"] == pytest.approx(100.0)
    assert r["recovery s"] == pytest.approx(0.0)


def test_player_on_a_bystander_is_wrong():
    # P2 follows C (someone else in view) all clip: never its own person.
    n = 20
    df = make_log(np.full(n, 300.0), np.full(n, 900.0))
    lab = labels_from([300.0] * n, [600.0] * n)
    lab = pd.concat([lab, pd.DataFrame([(f, f * 0.1, "C", 900.0, 400.0) for f in range(0, n, 5)],
                                       columns=lab.columns)])
    r = eval_identity.labelled_metrics(df, lab, "s", 0.15 * W, 1.0, 1.0)
    assert r["id accuracy %"] == pytest.approx(50.0) and r["on others %"] == pytest.approx(50.0)
    assert r["unassigned players %"] == pytest.approx(50.0)     # B never has a player
