"""MotionFilter: the prototype's KeypointTracker (gate, hold, EMA, snap back)."""

import math

import numpy as np
import pytest

from ipcvgame.core import skeleton as sk
from ipcvgame.core.config import load_config
from ipcvgame.core.types import Player, PoseDetection
from ipcvgame.pose.motion import MotionFilter


def player(x: float, conf: np.ndarray | None = None, reentered: bool = False,
           det: bool = True) -> dict[int, Player]:
    kpts = np.column_stack([np.full(sk.K, x), np.linspace(100, 400, sk.K)])
    conf = np.full(sk.K, 0.9) if conf is None else conf
    d = PoseDetection(bbox=np.array([x - 50, 80, x + 50, 420.0]), kpts=kpts,
                      conf=conf, score=0.9) if det else None
    return {1: Player(pid=1, status="active" if det else "lost", det=d,
                      bbox=np.array([x - 50, 80, x + 50, 420.0]), pos=np.zeros(2),
                      reentered=reentered)}


@pytest.fixture
def cfg():
    return load_config()["motion"]


def test_first_detection_snaps_in_place(cfg):
    m = MotionFilter(cfg).update(player(100.0), 0.0)[1]
    np.testing.assert_allclose(m.kpts[:, 0], 100.0)
    assert m.valid.all()
    np.testing.assert_allclose(m.vel, 0.0)


def test_ema_moves_by_time_constant(cfg):
    f = MotionFilter(cfg)
    f.update(player(0.0), 0.0)
    m = f.update(player(10.0), cfg["tau"])[1]
    # After one time constant the EMA has covered 1 - 1/e of the step.
    np.testing.assert_allclose(m.kpts[:, 0], 10.0 * (1 - math.exp(-1)))


def test_missing_keypoint_holds_then_invalid_then_snaps(cfg):
    f = MotionFilter(cfg)
    low = np.full(sk.K, 0.9)
    low[sk.L_WRIST] = 0.1
    hold = cfg["hold_s"]
    f.update(player(100.0), 0.0)
    held = f.update(player(300.0, low), hold)[1]
    assert held.valid[sk.L_WRIST]
    assert held.kpts[sk.L_WRIST, 0] == pytest.approx(100.0)  # never pulled to the bad value
    gone = f.update(player(300.0, low), hold + 0.05)[1]
    assert not gone.valid[sk.L_WRIST]
    back = f.update(player(300.0), hold + 0.10)[1]
    assert back.valid[sk.L_WRIST]
    assert back.kpts[sk.L_WRIST, 0] == pytest.approx(300.0)  # snapped, not smoothed
    assert back.vel[sk.L_WRIST, 0] == 0.0


def test_no_detection_resets_player(cfg):
    f = MotionFilter(cfg)
    f.update(player(0.0), 0.0)
    assert f.update(player(0.0, det=False), 0.033) == {}
    m = f.update(player(200.0), 0.066)[1]
    np.testing.assert_allclose(m.kpts[:, 0], 200.0)


def test_reentered_resets_player(cfg):
    f = MotionFilter(cfg)
    f.update(player(0.0), 0.0)
    m = f.update(player(200.0, reentered=True), 0.033)[1]
    np.testing.assert_allclose(m.kpts[:, 0], 200.0)


def test_stub_passes_through(cfg):
    f = MotionFilter({**cfg, "backend": "stub"})
    f.update(player(0.0), 0.0)
    m = f.update(player(10.0), 0.05)[1]
    np.testing.assert_allclose(m.kpts[:, 0], 10.0)
    np.testing.assert_allclose(m.vel[:, 0], 200.0)


def test_unknown_backend_raises(cfg):
    with pytest.raises(ValueError):
        MotionFilter({**cfg, "backend": "kalman"})
