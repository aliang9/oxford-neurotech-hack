"""Wake-or-wait policies.

Cost of waking at epoch u in stage s:  c(s) + sleep_value * minutes(latest - u)
(an awakening-difficulty proxy plus the sleep given up). At `latest` the alarm must fire.

BayesPolicy: optimal stopping by backward induction on a 30 s grid over the HSMM's hidden
state (stage s, elapsed d), with the HSMM's own duration-dependent transitions:

    V_latest(s, d) = c(s)
    C_u(s, d)      = (1 - h_s(d)) V_{u+1}(s, d+1) + h_s(d) sum_j P(j | s, d) V_{u+1}(j, 1)
    V_u(s, d)      = min(stop_u(s), C_u(s, d))     inside the window, C_u before it

At decision time u with posterior alpha over (s, d), wake iff E_alpha[stop_u] <= E_alpha[C_u]:
waking now is no worse than the expected cost of carrying on and deciding again later.
This is the QMDP approximation: the recursion assumes future stages will be known when
those later decisions are made. With a decent stager that is close, and it slightly
overvalues waiting. V and C depend on time, settings and the (online) sleep-onset
estimate only, never on observations, so they are computed once per night by one backward
pass, and each decision is two dot products.

ReactivePolicy: wake at the first epoch in the window with P(W or N1) >= 0.5.
FixedPolicy: wake at the latest time.
"""

import numpy as np

from sleep import config as sleep_cfg
from . import config as cfg

K = sleep_cfg.K


def cost_vector(stage_cost=None):
    sc = stage_cost or cfg.STAGE_COST
    return np.array([sc[s] for s in cfg.STAGES], float)


def stop_cost(costs, u, latest, sleep_value):
    return costs + sleep_value * max(latest - u, 0) / 2.0


class Plan:
    """Backward-induction tables C[u] for u in [start, latest)."""

    def __init__(self, model, costs, sleep_value, earliest, latest, start, bins):
        """bins[i] = time bin of epoch start + i (the bin a step from that epoch uses)."""
        self.start, self.earliest, self.latest = start, earliest, latest
        n = latest - start
        D = model.hazard.shape[-1]
        shift = np.minimum(np.arange(D) + 1, D - 1)
        V = np.repeat(costs[:, None], D, 1)                 # V_latest
        self.C = np.empty((max(n, 0), K, D))
        for i in range(n - 1, -1, -1):
            u = start + i
            h = model.hazard[bins[i]]
            enter = model.dest[bins[i]] @ V[:, 0]           # (K, D): E[V(j, 1)] on leaving
            C = (1 - h) * V[:, shift] + h * enter
            self.C[i] = C
            if u >= earliest:
                V = np.minimum(stop_cost(costs, u, latest, sleep_value)[:, None], C)
            else:
                V = C
        self.costs, self.sleep_value = costs, sleep_value

    def covers(self, u):
        return self.start <= u < self.latest


class BayesPolicy:
    name = "bayes"

    def __init__(self, model, stage_cost=None, sleep_value=cfg.SLEEP_VALUE_PER_MIN):
        self.model = model
        self.costs = cost_vector(stage_cost)
        self.sleep_value = sleep_value
        self.plan = None
        self._key = None

    def _plan(self, inf, earliest, latest):
        key = (inf.onset, earliest, latest)
        if self.plan is None or self._key != key or not self.plan.covers(inf.t):
            start = min(inf.t, earliest)
            bins = inf.future_bins(max(latest - inf.t, 1))
            # epochs before t (only when start < t, never in practice) reuse bin of t
            bins = np.concatenate([np.full(max(inf.t - start, 0), bins[0]), bins])
            self.plan = Plan(self.model, self.costs, self.sleep_value, earliest, latest,
                             start, bins)
            self._key = key
        return self.plan

    def evaluate(self, inf, earliest, latest):
        """Expected cost of waking now and of carrying on, under the current posterior."""
        u = inf.t
        a = inf.alpha[0]
        pa = a.sum(1)
        stop = float(pa @ stop_cost(self.costs, u, latest, self.sleep_value))
        if u >= latest:
            return dict(stop=stop, cont=np.inf)
        plan = self._plan(inf, earliest, latest)
        cont = float((a * plan.C[u - plan.start]).sum())
        return dict(stop=stop, cont=cont)

    def decide(self, inf, earliest, latest):
        v = self.evaluate(inf, earliest, latest)
        in_window = inf.t >= earliest
        v["wake"] = inf.t >= latest or (in_window and v["stop"] <= v["cont"])
        return v


class ReactivePolicy:
    name = "reactive"

    def __init__(self, stages=cfg.REACTIVE_STAGES, threshold=cfg.REACTIVE_THRESHOLD):
        self.idx = [cfg.STAGES.index(s) for s in stages]
        self.threshold = threshold

    def decide(self, inf, earliest, latest):
        p = float(inf.posterior()[self.idx].sum())
        return dict(wake=inf.t >= latest or (inf.t >= earliest and p >= self.threshold),
                    p_light=p)


class FixedPolicy:
    name = "fixed"

    def decide(self, inf, earliest, latest):
        return dict(wake=inf.t >= latest)


def open_loop_costs(forecast_path, t_now, earliest, latest, costs, sleep_value):
    """Expected cost of waking at each epoch of the window if decided now (for display).
    forecast_path[k] = P(stage at t_now + k). Returns (epochs, expected costs)."""
    us = np.arange(max(earliest, t_now), latest + 1)
    ks = us - t_now
    ks = ks[ks < len(forecast_path)]
    us = us[:len(ks)]
    exp = forecast_path[ks] @ costs + sleep_value * (latest - us) / 2.0
    return us, exp


def explain(decision, posterior, t, earliest, latest):
    """One or two plain sentences on the current recommendation."""
    top = np.argsort(posterior)[::-1]
    s0 = f"{cfg.STAGES[top[0]]} {posterior[top[0]]:.0%}"
    if t >= latest:
        return "Latest wake-up time reached: the alarm always fires now."
    if t < earliest:
        mins = (earliest - t) / 2
        return f"Probably {s0}. The wake-up window opens in {mins:.0f} min; until then, sleep on."
    if "stop" not in decision:
        return f"Probably {s0}."
    stop, cont = decision["stop"], decision["cont"]
    if decision["wake"]:
        return (f"Probably {s0}. Waking now costs {stop:.1f}; waiting is expected to cost "
                f"{cont:.1f}, so now is the best moment.")
    return (f"Probably {s0}. Waking now would cost {stop:.1f}; carrying on is expected to "
            f"cost {cont:.1f}, so keep sleeping and check again in 30 s.")
