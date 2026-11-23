"""MotionFilter: KeypointTracker from bodypose.py (gate, hold, EMA, snap back)."""

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
    f = MotionFilter({**cfg, "backend": "ema"})
    f.update(player(0.0), 0.0)
    m = f.update(player(10.0), cfg["tau"])[1]
    # After one time constant the EMA has covered 1 - 1/e of the step.
    np.testing.assert_allclose(m.kpts[:, 0], 10.0 * (1 - math.exp(-1)))


@pytest.mark.parametrize("backend", ["ema", "oneeuro"])
def test_missing_keypoint_holds_then_invalid_then_snaps(cfg, backend):
    f = MotionFilter({**cfg, "backend": backend})
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


# --- Wrist speed signals and logging ------------------------------------------

def body(dx_wrist: float = 0.0, conf: np.ndarray | None = None, jitter: float = 0.0) -> dict[int, Player]:
    """A player with shoulders 100 px apart; the screen-right wrist (model's
    L_WRIST) moved dx_wrist px to the right; every keypoint shifted by jitter px."""
    kpts = np.zeros((sk.K, 2))
    kpts[:, 1] = np.linspace(100, 400, sk.K)
    kpts[:, 0] = 200.0
    kpts[sk.L_SHOULDER], kpts[sk.R_SHOULDER] = (250.0, 200.0), (150.0, 200.0)
    kpts[sk.L_WRIST, 0] += dx_wrist
    kpts += jitter
    conf = np.full(sk.K, 0.9) if conf is None else conf
    d = PoseDetection(bbox=np.array([100, 80, 300, 420.0]), kpts=kpts, conf=conf, score=0.9)
    return {1: Player(pid=1, status="active", det=d, bbox=d.bbox, pos=np.zeros(2), reentered=False)}


def test_wrist_speed_in_shoulder_widths_per_second(cfg):
    f = MotionFilter({**cfg, "backend": "stub"})       # no smoothing: exact numbers
    f.update(body(), 0.0)
    m = f.update(body(dx_wrist=20.0), 0.1)[1]          # 20 px in 0.1 s = 200 px/s = 2 sw/s
    assert m.signals["l_wrist_speed"] == pytest.approx(2.0)
    assert m.signals["r_wrist_speed"] == pytest.approx(0.0)


def test_no_wrist_speed_without_shoulders_or_wrist(cfg):
    f = MotionFilter({**cfg, "backend": "stub"})
    low = np.full(sk.K, 0.9)
    low[sk.R_WRIST] = 0.1
    assert set(f.update(body(conf=low), 0.0)[1].signals) == {"l_wrist_speed"}
    low[sk.L_SHOULDER] = 0.1
    assert f.update(body(conf=low), 0.1)[1].signals == {}
    narrow = body()
    narrow[1].det.kpts[sk.L_SHOULDER, 0] = 155.0       # 5 px: under min_shoulder_px
    assert f.update(narrow, 0.2)[1].signals == {}


def state_of(motion):
    from ipcvgame.core.types import FrameState
    return FrameState(frame=None, players={}, motion=motion, faces={}, events=[], timings={})


def test_log_shows_smoothing_cuts_jitter(cfg):
    """A still player whose detection jumps 2 px back and forth every frame:
    the smoothed points move less than the raw ones, and lag behind them."""
    f = MotionFilter(cfg)
    rows = []
    for i in range(30):
        motion = f.update(body(jitter=2.0 * (i % 2)), i / 30)
        rows.append(f.log_fields(state_of(motion)))
    assert rows[0]["motion_p1_raw_step_px"] == ""      # first frame: nothing to compare with
    last = rows[-1]
    assert last["motion_p1_raw_step_px"] == pytest.approx(2.0 * np.sqrt(2), abs=1e-3)
    assert 0 < last["motion_p1_smooth_step_px"] < last["motion_p1_raw_step_px"]
    assert last["motion_p1_lag_px"] > 0
    assert last["motion_p1_valid"] == sk.K and last["motion_p1_held"] == 0
    assert last["motion_p1_l_wrist_speed"] != ""


def test_log_counts_held_keypoints(cfg):
    f = MotionFilter(cfg)
    f.update(body(), 0.0)
    low = np.full(sk.K, 0.9)
    low[[sk.L_WRIST, sk.R_WRIST]] = 0.1
    motion = f.update(body(conf=low), 0.05)
    row = f.log_fields(state_of(motion))
    assert row["motion_p1_valid"] == sk.K and row["motion_p1_held"] == 2


def run(f: MotionFilter, xs, dt: float = 1 / 30) -> np.ndarray:
    """Feed x positions at a fixed frame rate; returns the smoothed x of the nose."""
    return np.array([f.update(player(x), i * dt)[1].kpts[sk.NOSE, 0] for i, x in enumerate(xs)])


def test_oneeuro_smooths_a_still_keypoint_more_than_the_ema(cfg):
    shake = 100.0 + np.where(np.arange(60) % 2 == 0, 1.0, -1.0)       # +-1 px noise, standing still
    ema = run(MotionFilter({**cfg, "backend": "ema"}), shake)[30:]
    one = run(MotionFilter({**cfg, "backend": "oneeuro"}), shake)[30:]
    assert np.std(one) < 0.5 * np.std(ema)


def test_oneeuro_lags_less_than_the_ema_on_a_fast_move(cfg):
    ramp = 1000.0 / 30 * np.arange(60)                                  # 1000 px/s, a punch at 2 m
    lag_ema = (ramp - run(MotionFilter({**cfg, "backend": "ema"}), ramp))[-1]
    lag_one = (ramp - run(MotionFilter({**cfg, "backend": "oneeuro"}), ramp))[-1]
    assert 0 < lag_one < lag_ema
