import numpy as np

from ipcvgame.core.filters import EMA, ConstantVelocityKalman, OneEuro


def test_one_euro_first_sample_passes_through():
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    x = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_allclose(f(x, 0.0), x)


def test_one_euro_reduces_jitter_on_static_signal():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    out = [f(np.array([100.0]) + rng.normal(0, 5, 1), i / 30) for i in range(300)]
    assert np.std(out[100:]) < 2.0


def test_one_euro_is_fps_independent():
    """Same step input sampled at 30 and 60 fps reaches the same value at t=0.5 s."""
    def run(fps):
        f = OneEuro(min_cutoff=1.0)
        f(np.array([0.0]), 0.0)
        y = None
        for i in range(1, int(0.5 * fps) + 1):
            y = f(np.array([1.0]), i / fps)
        return float(y[0])
    assert abs(run(30) - run(60)) < 0.05


def test_ema_converges():
    f = EMA(tau=0.1)
    for i in range(100):
        y = f(np.array([5.0]), i / 30)
    assert abs(y[0] - 5.0) < 1e-3


def test_kalman_tracks_constant_velocity():
    kf = ConstantVelocityKalman(dim=2, q=10.0, r=1.0)
    kf.init(np.array([0.0, 0.0]), 0.0)
    for i in range(1, 60):
        t = i / 30
        kf.predict(t)
        kf.update(np.array([100.0 * t, -50.0 * t]))
    np.testing.assert_allclose(kf.vel, [100.0, -50.0], atol=5.0)
    # Predict through a 0.2 s occlusion.
    p = kf.predict(59 / 30 + 0.2)
    np.testing.assert_allclose(p, [100.0 * (59 / 30 + 0.2), -50.0 * (59 / 30 + 0.2)], atol=3.0)
