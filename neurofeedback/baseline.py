"""Eyes-closed resting baseline: mean and std of log alpha power, separately per electrode.

Keeping one baseline per electrode means any electrode can be dropped later (an artifact in
one window, or excluded by hand for the rest of the session) and the remaining electrodes
are still each compared with their own baseline.
"""

from dataclasses import dataclass

import numpy as np

from . import config as cfg


@dataclass
class Baseline:
    mean: np.ndarray     # per channel; NaN where the channel has no usable baseline
    std: np.ndarray
    n: np.ndarray        # clean updates per channel
    floored: np.ndarray  # std was below MIN_BASELINE_STD and was raised to it

    @property
    def available(self):
        return np.isfinite(self.mean)

    def z(self, log_alpha):
        """Per-channel z-scores (NaN for channels without a baseline)."""
        return (np.asarray(log_alpha) - self.mean) / self.std


class BaselineRecorder:
    """Collects per-channel log alpha for `duration_s` of stream time.

    Each update passes one value per channel, NaN for channels rejected in that window.
    Rejected updates still count toward the duration.
    """

    def __init__(self, n_channels, duration_s=cfg.BASELINE_S, step_s=cfg.STEP_S):
        self.n_channels = n_channels
        self.duration, self.step = duration_s, step_s
        self.elapsed = 0.0
        self.values = []

    def add(self, log_alpha, dt):
        self.elapsed += dt
        self.values.append(np.asarray(log_alpha, float))

    @property
    def done(self):
        return self.elapsed >= self.duration - 1e-6  # dt sums carry float error

    def finish(self, names):
        """(Baseline or None, message). A channel with too few clean updates gets no baseline."""
        expected = max(1, int(round(self.duration / self.step)))
        v = np.array(self.values).reshape(-1, self.n_channels)
        n = np.isfinite(v).sum(axis=0)
        ok = n >= max(2, cfg.MIN_BASELINE_VALID * expected)
        mean = np.full(self.n_channels, np.nan)
        std = np.full(self.n_channels, np.nan)
        floored = np.zeros(self.n_channels, bool)
        parts = []
        for c, name in enumerate(names):
            if not ok[c]:
                parts.append(f"{name} failed ({n[c]}/{expected} clean)")
                continue
            x = v[np.isfinite(v[:, c]), c]
            mean[c], std[c] = x.mean(), x.std(ddof=1)
            if std[c] < cfg.MIN_BASELINE_STD:  # near-zero variance would blow up z
                std[c], floored[c] = cfg.MIN_BASELINE_STD, True
            parts.append(f"{name} {mean[c]:.2f}±{std[c]:.2f}{' (floored)' if floored[c] else ''}"
                         f" [{n[c]}/{expected}]")
        if not ok.any():
            return None, "Baseline failed, no channel had enough clean data: " + "; ".join(parts)
        return Baseline(mean, std, n, floored), "Baseline, log10(uV^2): " + "; ".join(parts)
