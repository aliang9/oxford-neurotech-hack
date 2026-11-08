"""Where stage observations come from: one probability vector (W, N1, N2, N3, REM) per
30 s epoch, or None when the epoch is missing.

    ReplaySource   a held-out Sleep-EDF night, as one-hot labels or through a simulated
                   noisy EEG stager. Also holds the true stages, for display and scoring only.
    PushSource     for a live EEG stager later: call push(probs) every 30 s from the stager
                   thread; the session pulls them in order. A gap in pushes = missing epochs.

Every source tells the session the class prior its probabilities were trained with
(`class_prior`), so they can be turned into likelihoods; None means a flat prior.
"""

import collections
import threading
import time

import numpy as np

from sleep import config as sleep_cfg
from . import config as cfg

K = sleep_cfg.K


class Observation(tuple):
    """(epoch index, probs or None)."""
    __slots__ = ()

    def __new__(cls, epoch, probs):
        return super().__new__(cls, (epoch, None if probs is None else np.asarray(probs, float)))

    epoch = property(lambda s: s[0])
    probs = property(lambda s: s[1])


def stager_prior(recordings):
    """Stage frequencies a stager trained on these nights would have learned."""
    st = np.concatenate([r.stages for r in recordings]).astype(int)
    st = st[st >= 0]
    return np.bincount(st, minlength=K) / len(st)


def simulate_stager(stages, prior, seed=0, correct=cfg.SIM_CORRECT, sigma=cfg.SIM_SIGMA,
                    rho=cfg.SIM_RHO, dropout=cfg.SIM_DROPOUT):
    """Noisy posterior-like probabilities for each true stage (None for missing epochs).

    For true stage s the stager's log-likelihoods are log(centre[s]) + sigma * z_t, where
    centre[s] puts `correct` on s and the rest on typical confusions and z_t is AR(1) noise
    (errors persist for a few epochs). The output is that likelihood times the stager's
    class prior, renormalised: what a real classifier's posterior looks like."""
    rng = np.random.default_rng(seed)
    idx = {s: i for i, s in enumerate(cfg.STAGES)}
    centre = np.full((K, K), 1e-3)
    for s, conf in cfg.SIM_CONFUSION.items():
        centre[idx[s], idx[s]] = correct
        for j, w in conf.items():
            centre[idx[s], idx[j]] = (1 - correct) * w
    z = rng.normal(size=K)
    out = []
    for s in stages:
        z = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=K)
        if s < 0 or rng.random() < dropout:
            out.append(None)
            continue
        lg = np.log(centre[s]) + sigma * z
        post = np.exp(lg - lg.max()) * prior
        out.append(post / post.sum())
    return out


class ReplaySource:
    """A recorded night played back epoch by epoch. `truth` is never given to the model."""

    def __init__(self, rec, mode="stager", prior=None, seed=0):
        self.rec = rec
        self.truth = rec.stages.astype(int)
        self.n = len(self.truth)
        if mode == "labels":
            self.obs = [None if s < 0 else np.eye(K)[s] for s in self.truth]
            self.class_prior = None
        else:
            self.class_prior = prior
            self.obs = simulate_stager(self.truth, prior, seed)
        self.pos = 0

    def seek(self, epoch):
        self.pos = int(np.clip(epoch, 0, self.n))

    def pull(self, max_n=1):
        out = []
        while self.pos < self.n and len(out) < max_n:
            out.append(Observation(self.pos, self.obs[self.pos]))
            self.pos += 1
        return out

    @property
    def finished(self):
        return self.pos >= self.n


class PushSource:
    """Thread-safe queue for a live stager: push(probs) once per 30 s epoch.

    Epoch indices come from wall-clock time since start(), so an interrupted stream shows
    up as missing epochs (None) rather than as a jump in the night."""

    def __init__(self, class_prior=None):
        self.class_prior = class_prior
        self.q = collections.deque()
        self.got = {}                       # epoch -> probs, pushed but not yet pulled
        self.lock = threading.Lock()
        self.t0 = None
        self.next_epoch = 0

    def start(self):
        self.t0 = time.time()

    def push(self, probs, t=None):
        with self.lock:
            e = int(((t or time.time()) - self.t0) // cfg.EPOCH_S)
            self.q.append(Observation(e, probs))

    def pull(self, max_n=10 ** 9):
        """Observations up to now, with None filled in for epochs nobody pushed."""
        now_epoch = int((time.time() - self.t0) // cfg.EPOCH_S)
        out = []
        with self.lock:
            while self.q:
                o = self.q.popleft()
                self.got[o.epoch] = o.probs
        while self.next_epoch < now_epoch and len(out) < max_n:
            out.append(Observation(self.next_epoch, self.got.pop(self.next_epoch, None)))
            self.next_epoch += 1
        return out

    finished = False
