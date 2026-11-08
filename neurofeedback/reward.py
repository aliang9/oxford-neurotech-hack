"""Smoothing and the cumulative reward."""

import math

from . import config as cfg


class EMA:
    """Exponential moving average with time constant tau_s, robust to uneven update intervals."""

    def __init__(self, tau_s=cfg.EMA_TAU_S):
        self.tau = tau_s
        self.value = None

    def update(self, x, dt):
        if self.value is None:
            self.value = x
        else:
            self.value += (1 - math.exp(-dt / self.tau)) * (x - self.value)
        return self.value

    def reset(self):
        self.value = None


class RewardAccumulator:
    """cumulative += max(0, smoothed_z - threshold) * dt, only while active. Never decreases."""

    def __init__(self, threshold=cfg.THRESHOLD):
        self.threshold = threshold
        self.total = 0.0
        self.active_time = 0.0

    def step(self, smoothed_z, dt, active):
        """Returns the reward rate applied at this update (0 when not active)."""
        if not active or smoothed_z is None:
            return 0.0
        rate = max(0.0, smoothed_z - self.threshold)
        self.total += rate * dt
        self.active_time += dt
        return rate

    def reset(self):
        self.total = 0.0
        self.active_time = 0.0
