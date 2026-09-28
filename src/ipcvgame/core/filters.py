"""Shared smoothing filters for Tasks 1-3.

All filters work on NumPy arrays of any shape (e.g. (K, 2) keypoints) and take
the real timestamp t in seconds, so they stay correct when the FPS varies.
Each instance holds state for ONE signal of ONE player; call reset() when the
player re-enters.
"""

from __future__ import annotations

import math

import numpy as np


class EMA:
    """Exponential moving average with a time constant instead of a fixed alpha.

    alpha = 1 - exp(-dt / tau), so the smoothing strength does not depend on FPS.
    """

    def __init__(self, tau: float):
        self.tau = tau
        self.x: np.ndarray | None = None
        self.t: float | None = None

    def reset(self) -> None:
        self.x = None
        self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.t is None:
            self.x, self.t = x.copy(), t
            return self.x.copy()
        dt = max(t - self.t, 1e-6)
        alpha = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        self.x = self.x + alpha * (x - self.x)
        self.t = t
        return self.x.copy()


def _smoothing_factor(dt: float, cutoff: np.ndarray | float) -> np.ndarray | float:
    r = 2.0 * math.pi * cutoff * dt
    return r / (r + 1.0)


class OneEuro:
    """One Euro filter (Casiez et al., CHI 2012), vectorised over array elements.

    min_cutoff: Hz, lower = smoother when still (less jitter).
    beta: speed coefficient, higher = less lag during fast motion.
    d_cutoff: Hz, cutoff for the derivative estimate.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.t is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x.copy()
        dt = max(t - self.t, 1e-6)
        a_d = _smoothing_factor(dt, self.d_cutoff)
        self.dx = a_d * (x - self.x) / dt + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = _smoothing_factor(dt, cutoff)
        self.x = a * x + (1.0 - a) * self.x
        self.t = t
        return self.x.copy()


class ConstantVelocityKalman:
    """Constant-velocity Kalman filter for a D-dimensional point.

    State is [p (D), v (D)]. Use predict(t) every frame and update(z) when a
    measurement is available; while occluded, only predict.

    q: process noise (acceleration spectral density, px^2/s^3).
    r: measurement noise variance (px^2).
    """

    def __init__(self, dim: int = 2, q: float = 1000.0, r: float = 25.0):
        self.dim = dim
        self.q = q
        self.r = r
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t: float | None = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    @property
    def pos(self) -> np.ndarray:
        assert self.x is not None
        return self.x[: self.dim].copy()

    @property
    def vel(self) -> np.ndarray:
        assert self.x is not None
        return self.x[self.dim:].copy()

    def init(self, z: np.ndarray, t: float, vel_var: float = 1e4) -> None:
        d = self.dim
        self.x = np.concatenate([np.asarray(z, dtype=np.float64), np.zeros(d)])
        self.P = np.diag([self.r] * d + [vel_var] * d)
        self.t = t

    def predict(self, t: float) -> np.ndarray:
        assert self.x is not None and self.P is not None and self.t is not None
        d = self.dim
        dt = max(t - self.t, 0.0)
        F = np.eye(2 * d)
        F[:d, d:] = dt * np.eye(d)
        # Discrete white-noise acceleration model.
        q11, q12, q22 = dt**3 / 3, dt**2 / 2, dt
        Q = self.q * np.block([[q11 * np.eye(d), q12 * np.eye(d)],
                               [q12 * np.eye(d), q22 * np.eye(d)]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t
        return self.pos

    def update(self, z: np.ndarray) -> np.ndarray:
        assert self.x is not None and self.P is not None
        d = self.dim
        H = np.hstack([np.eye(d), np.zeros((d, d))])
        R = self.r * np.eye(d)
        y = np.asarray(z, dtype=np.float64) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2 * d) - K @ H) @ self.P
        return self.pos
