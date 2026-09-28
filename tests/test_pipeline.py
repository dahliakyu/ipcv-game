"""End-to-end smoke tests with stub modules and synthetic detections (no model)."""

import numpy as np
import pytest

from ipcvgame.app.pipeline import STAGES, Pipeline
from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Frame, FrameState, PoseDetection

W, H = 640, 480


def fake_det(cx: float) -> PoseDetection:
    kpts = np.zeros((sk.K, 2))
    kpts[:, 0] = cx
    kpts[:, 1] = np.linspace(100, 400, sk.K)
    kpts[sk.L_SHOULDER, 0], kpts[sk.R_SHOULDER, 0] = cx - 40, cx + 40
    return PoseDetection(bbox=np.array([cx - 60, 80, cx + 60, 420.0]), kpts=kpts,
                         conf=np.full(sk.K, 0.9), score=0.9)


@pytest.fixture
def pipeline():
    cfg = load_config(overrides=["pose.backend=none"])
    return Pipeline(cfg)


def frame(i: int) -> Frame:
    return Frame(image=np.zeros((H, W, 3), np.uint8), frame_id=i, t=i / 30)


def test_runs_end_to_end(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline.pose, "detect", lambda f: [fake_det(150), fake_det(500)])
    for i in range(5):
        state, canvas = pipeline.step(frame(i))
    assert isinstance(state, FrameState)
    assert canvas.shape == (H, W, 3)
    assert set(state.players) == {1, 2}
    assert state.players[1].det.bbox[0] < state.players[2].det.bbox[0]
    assert "above_head" in state.faces[1].anchors
    assert state.faces[1].anchors["above_head"][1] < state.faces[1].center[1]
    assert {"pose", "identity", "motion", "face", "render", "total"} <= set(state.timings)


def test_velocity_uses_frame_time(pipeline, monkeypatch):
    xs = iter([150.0, 160.0])
    monkeypatch.setattr(pipeline.pose, "detect", lambda f: [fake_det(next(xs))])
    pipeline.step(Frame(np.zeros((H, W, 3), np.uint8), 0, 0.0))
    state, _ = pipeline.step(Frame(np.zeros((H, W, 3), np.uint8), 1, 0.05))
    np.testing.assert_allclose(state.motion[1].vel[:, 0], 200.0)


def test_lost_and_reentered(pipeline, monkeypatch):
    seq = iter([[fake_det(150)], [], [fake_det(150)]])
    monkeypatch.setattr(pipeline.pose, "detect", lambda f: next(seq))
    s0, _ = pipeline.step(frame(0))
    s1, _ = pipeline.step(frame(1))
    s2, _ = pipeline.step(frame(2))
    assert s0.players[1].status == "active" and not s0.players[1].reentered
    assert s1.players[1].status == "lost" and s1.players[1].det is None
    assert s2.players[1].reentered


@pytest.mark.parametrize("until", STAGES)
def test_until_each_stage(until, monkeypatch):
    p = Pipeline(load_config(overrides=["pose.backend=none"]), until=until)
    monkeypatch.setattr(p.pose, "detect", lambda f: [fake_det(150)])
    state, canvas = p.step(frame(0))
    assert canvas.shape == (H, W, 3)
    assert bool(state.players) == (STAGES.index(until) >= STAGES.index("identity"))
