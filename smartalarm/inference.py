"""Online Bayesian filtering with the trained HSMM.

Every 30 s the stager's probability vector p (or None if the epoch is missing) is turned
into a likelihood, l ∝ p / class_prior, floored slightly so that one overconfident wrong
epoch cannot zero out the truth. The posterior over (stage, elapsed duration) is
predicted one step with the HSMM and multiplied by l. Probabilities are never collapsed to
a hard label.

Time since sleep onset (the model's time bins) is estimated online: onset is the first of
ONSET_RUN consecutive epochs with P(asleep) > 0.5 (asleep = any stage but W). The first
epoch's prior is INITIAL_WAKE_PROB awake, since tracking starts at bedtime.
"""

import numpy as np

from sleep import config as sleep_cfg
from sleep.models import forecast, forecast_bins, time_bin_minutes
from . import config as cfg

K = sleep_cfg.K
EVIDENCE_FLOOR = 1e-3


def likelihood(probs, class_prior=None):
    """Stager output -> evidence vector (ones for a missing epoch)."""
    if probs is None or not np.all(np.isfinite(probs)) or np.sum(probs) <= 0:
        return np.ones(K)
    lik = np.asarray(probs, float)
    if class_prior is not None:
        lik = lik / np.maximum(class_prior, 1e-6)
    lik = lik / lik.sum()
    return (1 - EVIDENCE_FLOOR * K) * lik + EVIDENCE_FLOOR


class OnlineHSMM:
    def __init__(self, model, class_prior=None):
        self.model = model
        self.class_prior = class_prior
        self.reset()

    def reset(self):
        self.alpha = None          # (1, K, D) joint posterior over (stage, elapsed d)
        self.t = -1                # last epoch processed
        self.onset = None          # estimated sleep-onset epoch
        self.marginals = []        # filtered P(stage) per epoch, for the hypnogram
        self.n_missing = 0

    def minutes_since_onset(self, epoch):
        return None if self.onset is None else (epoch - self.onset) / 2.0

    def time_bin(self, epoch):
        if not self.model.time_dependent:
            return 0
        m = self.minutes_since_onset(epoch)
        return int(time_bin_minutes(-1 if m is None or m < 0 else m))

    def observe(self, epoch, probs):
        """Add the observation for `epoch`. Skipped epochs are filled in as missing."""
        while self.t + 1 < epoch:
            self._update(self.t + 1, None)
        self._update(epoch, probs)

    def _update(self, epoch, probs):
        ev = likelihood(probs, self.class_prior)[None]
        if probs is None:
            self.n_missing += 1
        if self.alpha is None:   # tracking starts at bedtime: mostly awake
            w = cfg.INITIAL_WAKE_PROB
            ev = ev * np.array([w] + [(1 - w) / (K - 1)] * (K - 1))
            self.alpha = self.model.init_belief(ev, np.array([self.time_bin(epoch)]))
        else:
            pred = self.model.step(self.alpha, np.array([self.time_bin(epoch - 1)]))
            self.alpha = self.model.update(pred, ev)
        self.t = epoch
        p = self.posterior()
        self.marginals.append(p)
        if self.onset is None and len(self.marginals) >= cfg.ONSET_RUN:
            if all(m[0] < 0.5 for m in self.marginals[-cfg.ONSET_RUN:]):
                self.onset = epoch - cfg.ONSET_RUN + 1

    def posterior(self):
        """Current P(stage), shape (K,)."""
        a = self.model.marginal(self.alpha)[0]
        return a / a.sum()

    def future_bins(self, n):
        """Time bin of each epoch the next n steps start from (t, t+1, ...)."""
        m = self.minutes_since_onset(self.t)
        return forecast_bins([-1.0 if m is None else m], n, self.model.time_dependent)[0]

    def forecast(self, n):
        """(n + 1, K): P(stage) now and 1..n epochs ahead."""
        if n <= 0:
            return self.posterior()[None]
        fc = forecast(self.model, self.alpha, self.future_bins(n)[None], list(range(1, n + 1)))
        return np.vstack([self.posterior()] + [fc[k][0] for k in range(1, n + 1)])
