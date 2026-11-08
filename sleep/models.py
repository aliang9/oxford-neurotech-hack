"""Supervised sleep-dynamics models: first-order HMM and a hazard-based HSMM.

Stage labels are observed in training, so both models are fitted by counting.

HMM.  P(S_{t+1} = j | S_t = i) = A[i, j], with Dirichlet (Laplace) smoothing. Forecasts are
      b A^k for a belief b over the 5 stages (or a product of per-time-bin matrices).

HSMM. The hidden state is (stage s, elapsed epochs in that stage d), d = 1..MAX_DUR, where
      d = MAX_DUR means "MAX_DUR or longer". Each 30 s step:
        leave with hazard h_s(d) -> next stage s' ~ P(s' | s, d), entering (s', 1)
        stay with 1 - h_s(d)     -> (s, d + 1)
      The duration pmf of stage s is f_s(d) = h_s(d) prod_{j<d} (1 - h_s(j)). The hazard is
      estimated per duration bin from at-risk/exit counts (right-censored runs are handled
      by construction) and shrunk toward the stage's mean (geometric) hazard, so with no
      data it falls back to the HMM. Runs whose start was not observed (the first run of a
      recording, or after an unscored gap) are left out of the duration statistics.

Time dependence. Every parameter can also be indexed by a time-since-sleep-onset bin
(config.TIME_EDGES_H); per-bin estimates are shrunk toward the stationary estimate with
KAPPA_TIME pseudo-counts.

Beliefs. Both models filter a sequence of per-epoch evidence vectors (one-hot labels, a
uniform vector for an unscored epoch, or EEG-classifier stage probabilities, which are
treated as likelihoods), so they work from uncertain current stages and uncertain history.
"""

import numpy as np

from . import config as cfg

K = cfg.K
D = cfg.MAX_DUR


# --- time and duration bins -------------------------------------------------------------

def time_bin_minutes(minutes):
    """Time bin (0 = not asleep yet) for minutes since sleep onset; None/negative -> 0."""
    m = np.asarray(-1.0 if minutes is None else minutes, float)
    edges = np.array(cfg.TIME_EDGES_H) * 60.0
    return np.where(m < 0, 0, np.searchsorted(edges, m, side="right")).astype(int)


def time_bins_epochs(n_epochs, onset):
    """Time bin of each epoch index 0..n-1 of a recording whose first sleep epoch is `onset`."""
    t = np.arange(n_epochs)
    return time_bin_minutes(np.where(t >= onset, (t - onset) / cfg.EPOCHS_PER_MIN, -1.0))


def _bin_index(edges, d):
    """Index of the bin (left edges) containing duration d (1-based epochs)."""
    return np.searchsorted(np.asarray(edges), d, side="right") - 1


DUR_BIN = _bin_index(cfg.DUR_EDGES, np.arange(1, D + 1))      # (D,) hazard bin of each d
DEST_BIN = _bin_index(cfg.DEST_EDGES, np.arange(1, D + 1))    # (D,) destination bin of each d
N_TB = len(cfg.TIME_LABELS)
N_DUR = len(cfg.DUR_EDGES)
N_DEST = len(cfg.DEST_EDGES)


# --- counting ---------------------------------------------------------------------------

def count(recordings):
    """Sufficient statistics, all indexed by the time bin of the epoch a step starts from.

    trans[tb, i, j]      consecutive scored epochs i -> j
    at_risk[tb, s, d-1]  epochs in stage s with known elapsed d whose next epoch is scored
    exits[tb, s, d-1]    ... of which the next epoch is a different stage
    dest[tb, s, b, j]    exits from s (elapsed d in destination bin b) into j
    """
    trans = np.zeros((N_TB, K, K))
    at_risk = np.zeros((N_TB, K, D))
    exits = np.zeros((N_TB, K, D))
    dest = np.zeros((N_TB, K, N_DEST, K))
    for r in recordings:
        st = r.stages.astype(int)
        tb = time_bins_epochs(len(st), r.onset)
        d = None                          # elapsed epochs in the current run, None = unknown
        for t in range(len(st) - 1):
            s = st[t]
            if s < 0:
                d = None
                continue
            if t > 0 and st[t - 1] == s:
                d = None if d is None else d + 1
            elif t > 0 and st[t - 1] >= 0:
                d = 1                     # a transition into s was observed
            else:
                d = None                  # start of record or after a gap
            nxt = st[t + 1]
            if nxt < 0:
                continue
            trans[tb[t], s, nxt] += 1
            if d is not None:
                dc = min(d, D) - 1
                at_risk[tb[t], s, dc] += 1
                if nxt != s:
                    exits[tb[t], s, dc] += 1
                    dest[tb[t], s, DEST_BIN[dc], nxt] += 1
    return dict(trans=trans, at_risk=at_risk, exits=exits, dest=dest)


def _shrink(counts, total, prior, kappa):
    """(counts + kappa * prior) / (total + kappa): a Beta/Dirichlet posterior mean."""
    return (counts + kappa * prior) / (total + kappa)


# --- HMM --------------------------------------------------------------------------------

class HMM:
    """First-order Markov chain on the 5 stages, optionally one matrix per time bin."""

    def __init__(self, A, time_dependent):
        self.A = np.asarray(A)              # (N_TB, K, K); identical slices if stationary
        self.time_dependent = time_dependent

    @classmethod
    def fit(cls, stats, time_dependent=False):
        C = stats["trans"]
        A0 = C.sum(0) + cfg.LAPLACE
        A0 /= A0.sum(1, keepdims=True)
        if not time_dependent:
            return cls(np.repeat(A0[None], N_TB, 0), False)
        A = _shrink(C, C.sum(2, keepdims=True), A0[None], cfg.KAPPA_TIME)
        return cls(A, True)

    # beliefs are (B, K); tb is (B,) time bins of the epoch each step starts from
    def init_belief(self, evidence, tb=None):
        return _normalize(np.asarray(evidence, float))

    def step(self, belief, tb):
        return np.einsum("bi,bij->bj", belief, self.A[tb])

    def update(self, belief, evidence):
        return _normalize(belief * evidence)

    def marginal(self, belief):
        return belief

    def to_dict(self):
        return dict(kind="hmm", A=self.A, time_dependent=self.time_dependent)


# --- HSMM -------------------------------------------------------------------------------

class HSMM:
    """Explicit-duration model with state (stage, elapsed d). See the module docstring."""

    def __init__(self, hazard, dest, time_dependent):
        self.hazard = np.asarray(hazard)    # (N_TB, K, D)  P(leave at d | reached d)
        self.dest = np.asarray(dest)        # (N_TB, K, D, K) P(next = j | leave s at d), j != s
        self.time_dependent = time_dependent

    @classmethod
    def fit(cls, stats, time_dependent=False):
        X, N, Dst = stats["exits"], stats["at_risk"], stats["dest"]
        lo, hi = cfg.HAZARD_CLIP
        offdiag = 1.0 - np.eye(K)

        def per_dur_bin(a):               # (..., D) -> (..., N_DUR) summed within bins
            out = np.zeros(a.shape[:-1] + (N_DUR,))
            np.add.at(out, (..., DUR_BIN), a)
            return out

        # stationary: mean hazard per stage, then per duration bin shrunk toward it
        X0, N0 = X.sum(0), N.sum(0)                                   # (K, D)
        h_mean = (X0.sum(1) + 1) / (N0.sum(1) + 2)                    # (K,)
        h0 = _shrink(per_dur_bin(X0), per_dur_bin(N0), h_mean[:, None], cfg.KAPPA_DUR)
        # destinations: per stage (Laplace), then per destination bin shrunk toward it
        P_mean = Dst.sum((0, 2)) + cfg.LAPLACE * offdiag
        P_mean /= P_mean.sum(1, keepdims=True)                        # (K, K)
        D0 = Dst.sum(0)                                               # (K, N_DEST, K)
        P0 = _shrink(D0, D0.sum(2, keepdims=True), P_mean[:, None], cfg.KAPPA_DEST)

        if time_dependent:
            h = _shrink(per_dur_bin(X), per_dur_bin(N), h0[None], cfg.KAPPA_TIME)
            P = _shrink(Dst, Dst.sum(3, keepdims=True), P0[None], cfg.KAPPA_TIME)
        else:
            h = np.repeat(h0[None], N_TB, 0)
            P = np.repeat(P0[None], N_TB, 0)
        hazard = np.clip(h[..., DUR_BIN], lo, hi)                     # (N_TB, K, D)
        dest = P[:, :, DEST_BIN, :]                                   # (N_TB, K, D, K)
        return cls(hazard, dest, time_dependent)

    @classmethod
    def from_hmm(cls, hmm):
        """The HSMM equivalent to an HMM (geometric durations); used to check the code."""
        A = hmm.A
        stay = np.einsum("tii->ti", A)
        hazard = np.repeat((1 - stay)[:, :, None], D, 2)
        P = A * (1 - np.eye(K)) / (1 - stay)[:, :, None]
        dest = np.repeat(P[:, :, None, :], D, 2)
        return cls(hazard, dest, hmm.time_dependent)

    def elapsed_prior(self, tb):
        """P(d | s) for a stage entered at an unknown time: proportional to survival S_s(d).
        The last state (d >= MAX_DUR) gets the geometric tail S(D) / h(D)."""
        h = self.hazard[tb]                                            # (B, K, D)
        surv = np.concatenate([np.ones(h.shape[:-1] + (1,)),
                               np.cumprod(1 - h[..., :-1], -1)], -1)
        surv[..., -1] /= h[..., -1]
        return surv / surv.sum(-1, keepdims=True)

    def init_belief(self, evidence, tb):
        ev = _normalize(np.asarray(evidence, float))
        return ev[:, :, None] * self.elapsed_prior(tb)

    def step(self, alpha, tb):
        h = self.hazard[tb]                                            # (B, K, D)
        leave = alpha * h
        stay = alpha - leave
        out = np.zeros_like(alpha)
        out[:, :, 1:] = stay[:, :, :-1]
        out[:, :, -1] += stay[:, :, -1]
        out[:, :, 0] = np.einsum("bkd,bkdj->bj", leave, self.dest[tb])
        return out

    def update(self, alpha, evidence):
        a = alpha * evidence[:, :, None]
        return a / np.maximum(a.sum((1, 2), keepdims=True), 1e-300)

    def marginal(self, alpha):
        return alpha.sum(2)

    def duration_pmf(self, tb=0):
        """(K, D) pmf of stage durations in time bin tb (last column = MAX_DUR or longer)."""
        h = self.hazard[tb]
        surv = np.concatenate([np.ones((K, 1)), np.cumprod(1 - h[:, :-1], 1)], 1)
        pmf = surv * h
        pmf[:, -1] = surv[:, -1]
        return pmf

    def to_dict(self):
        return dict(kind="hsmm", hazard=self.hazard, dest=self.dest,
                    time_dependent=self.time_dependent)


# --- persistence baseline ---------------------------------------------------------------

class Persistence:
    """Current stage continues. To have a finite log loss, the current stage gets the
    training-set probability of being unchanged k epochs later, the rest is spread evenly."""

    def __init__(self, p_same):
        self.p_same = dict(p_same)          # {k epochs: P(S_{t+k} == S_t)}

    @classmethod
    def fit(cls, recordings, max_k):
        same = np.zeros(max_k + 1)
        n = np.zeros(max_k + 1)
        for r in recordings:
            st = r.stages.astype(int)
            for k in range(1, max_k + 1):
                a, b = st[:-k], st[k:]
                ok = (a >= 0) & (b >= 0) & (np.arange(len(a)) >= r.onset)
                same[k] += np.sum(a[ok] == b[ok])
                n[k] += ok.sum()
        return cls({k: (same[k] + 1) / (n[k] + 2) for k in range(1, max_k + 1)})

    def forecast(self, current, k):
        p = self.p_same[k]
        out = np.full((len(current), K), (1 - p) / (K - 1))
        out[np.arange(len(current)), current] = p
        return out

    def to_dict(self):
        return dict(kind="persistence", p_same=self.p_same)


def from_dict(d):
    if d["kind"] == "hmm":
        return HMM(d["A"], d["time_dependent"])
    if d["kind"] == "hsmm":
        return HSMM(d["hazard"], d["dest"], d["time_dependent"])
    return Persistence(d["p_same"])


# --- filtering and forecasting ----------------------------------------------------------

def _normalize(p):
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-300)


def evidence_from_labels(stages):
    """(T, K) evidence: one-hot for scored epochs, uniform for unscored ones."""
    st = np.asarray(stages, int)
    ev = np.ones((len(st), K))
    ok = st >= 0
    ev[ok] = np.eye(K)[st[ok]]
    return ev


def filter_states(model, evidence, tb):
    """Causal forward filter over one recording.

    evidence (T, K), tb (T,) time bin of each epoch. Returns the list of filtered states
    (model-specific shape, batch of 1) after each epoch: state t uses evidence 0..t only."""
    state = model.init_belief(evidence[:1], tb[:1])
    states = [state]
    for t in range(1, len(evidence)):
        state = model.update(model.step(state, tb[t - 1:t]), evidence[t:t + 1])
        states.append(state)
    return states


def forecast(model, state, step_bins, horizons):
    """Stage distributions at the given horizons (epochs) from a batch of filtered states.

    state: (B, K) or (B, K, D); step_bins: (B, max_h) time bin of the epoch each forecast
    step starts from. Returns {k: (B, K)}."""
    out = {}
    hs = set(horizons)
    for j in range(max(horizons)):
        state = model.step(state, step_bins[:, j])
        if j + 1 in hs:
            out[j + 1] = _normalize(model.marginal(state))
    return out


def forecast_bins(origin_bins_minutes, max_h, time_dependent):
    """(B, max_h) time bins for forecast steps j = 0..max_h-1 from origins at the given
    minutes since onset (negative = not asleep yet: kept in the pre-onset bin)."""
    m = np.asarray(origin_bins_minutes, float)[:, None]
    if not time_dependent:
        return np.zeros((len(m), max_h), int)
    steps = np.arange(max_h)[None] / cfg.EPOCHS_PER_MIN
    return np.where(m < 0, 0, time_bin_minutes(np.where(m < 0, 0, m + steps)))
