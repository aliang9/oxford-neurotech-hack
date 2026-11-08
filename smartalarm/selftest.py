"""Checks for the inference and decision logic.

    python -m smartalarm.selftest

1. Bellman recursion: with fully observed stages, the expected cost the plan promises at
   the start equals the mean realised cost of following it on Monte Carlo HSMM trajectories.
2. The optimal policy never expects to do worse than always waking at the deadline.
3. Class-prior correction: a stager output equal to its prior carries no evidence.
4. The deadline fires at `latest` even if the model raises on every epoch, and with a
   stream of nothing but missing epochs.
5. Causality: changing every observation after epoch t leaves all decisions up to t alone.
6. seek() rebuilds exactly the state of an uninterrupted run.
"""

import joblib
import numpy as np

from sleep import data
from sleep.models import from_dict
from . import config as cfg
from .inference import OnlineHSMM, likelihood
from .policy import Plan, cost_vector, stop_cost
from .session import Session
from .sources import Observation, ReplaySource, stager_prior


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return bool(ok)


def simulate_policy(model, plan, alpha0, start, n_traj=20000, seed=0):
    """Mean realised cost of following `plan` with the true (s, d) visible."""
    rng = np.random.default_rng(seed)
    K, D = alpha0.shape
    flat = rng.choice(K * D, n_traj, p=(alpha0 / alpha0.sum()).ravel())
    s, d = flat // D, flat % D
    cost = np.full(n_traj, np.nan)
    alive = np.ones(n_traj, bool)
    for u in range(start, plan.latest + 1):
        stop = stop_cost(plan.costs, u, plan.latest, plan.sleep_value)[s]
        if u == plan.latest:
            wake = alive
        else:
            C = plan.C[u - plan.start][s, d]
            wake = alive & (u >= plan.earliest) & (stop <= C)
        cost[wake] = stop[wake]
        alive &= ~wake
        if u == plan.latest:
            break
        tb = plan.bins[u - plan.start]
        h = model.hazard[tb][s, d]
        leave = rng.random(n_traj) < h
        nxt = np.array([rng.choice(K, p=model.dest[tb][si, di]) for si, di
                        in zip(s[leave], d[leave])], int)
        s = s.copy()
        s[leave] = nxt
        d = np.where(leave, 0, np.minimum(d + 1, D - 1))
    return cost.mean()


def main():
    blob = joblib.load(cfg.MODEL_FILE)
    model = from_dict(blob["models"][cfg.MODEL_NAME])
    recs = data.load()
    test = [r for r in recs if r.subject in set(blob["test_subjects"])]
    prior = stager_prior([r for r in recs if r.subject not in set(blob["test_subjects"])])
    rec = test[0]
    ok = True

    # 1-2: Bellman vs Monte Carlo, from a filtered state 10 min before the window
    latest = rec.onset + 900
    earliest = latest - 60
    inf = OnlineHSMM(model)
    for t, s in enumerate(rec.stages[:earliest - 20]):
        inf.observe(t, None if s < 0 else np.eye(5)[s])
    start = inf.t
    bins = inf.future_bins(latest - start)
    costs = cost_vector()
    plan = Plan(model, costs, cfg.SLEEP_VALUE_PER_MIN, earliest, latest, start, bins)
    plan.bins = bins
    a = inf.alpha[0]
    promised = float((a * plan.C[0]).sum())       # cannot stop yet at `start`
    realised = simulate_policy(model, plan, a, start)
    ok &= check(f"Bellman value {promised:.3f} == Monte Carlo cost {realised:.3f}",
                abs(promised - realised) < 0.05)
    p_latest = inf.forecast(latest - start)[-1]
    ok &= check(f"policy ({promised:.3f}) <= always wake at deadline ({p_latest @ costs:.3f})",
                promised <= p_latest @ costs + 1e-9)

    # 3: class prior
    ok &= check("stager output equal to its prior -> flat evidence",
                np.allclose(likelihood(prior, prior), 0.2))

    # 4: deadline with a broken model / no data
    class Broken:
        time_dependent = True

        def __getattr__(self, _):
            raise RuntimeError("model down")

    s = Session(ReplaySource(rec, "labels"), Broken(), earliest, latest, log=False)
    s.advance(10 ** 6)
    ok &= check("deadline fires at latest with a failing model",
                s.fired and s.fired["epoch"] == latest and s.fired["reason"] == "deadline")

    class Silent:
        class_prior = None
        pos = 0

        def seek(self, e):
            self.pos = e

        def pull(self, n):
            out = [Observation(self.pos + i, None) for i in range(min(n, 2000))]
            self.pos += len(out)
            return out

    s = Session(Silent(), model, earliest, latest, log=False)
    s.advance(10 ** 6)
    ok &= check("deadline fires at latest with only missing epochs",
                s.fired and s.fired["epoch"] == latest)

    # 5: causality
    def decisions(source, upto):
        s = Session(source, model, earliest, latest, log=False)
        out = []
        while s.now < upto and not s.fired:
            s.advance(1)
            out.append((s.now, s.decision.get("wake"), round(s.decision.get("cont", 0), 9)))
        return out

    t_cut = earliest + 20
    src_a = ReplaySource(rec, "stager", prior, seed=3)
    src_b = ReplaySource(rec, "stager", prior, seed=3)
    rng = np.random.default_rng(9)
    for i in range(t_cut + 1, src_b.n):
        src_b.obs[i] = rng.dirichlet(np.ones(5))
    ok &= check("no decision depends on observations after it",
                decisions(src_a, t_cut) == decisions(src_b, t_cut))

    # 6: seek
    s1 = Session(ReplaySource(rec, "stager", prior, seed=1), model, earliest, latest, log=False)
    s1.advance(700)
    s2 = Session(ReplaySource(rec, "stager", prior, seed=1), model, earliest, latest, log=False)
    s2.advance(300)
    s2.seek(700)
    ok &= check("seek() reproduces an uninterrupted run",
                s1.now == s2.now and np.allclose(s1.inf.alpha, s2.inf.alpha))

    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
