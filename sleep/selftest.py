"""Correctness checks for the models. Needs models/sleep/models.joblib (run sleep.train).

    python -m sleep.selftest

1. HMM forecasts equal b A^k (numpy matrix_power) for a soft belief b.
2. An HSMM built from the HMM (geometric durations) gives the same forecasts as the HMM,
   including after filtering a soft, partly missing history. This checks the HSMM step,
   filter and elapsed-duration bookkeeping against an independent implementation.
3. No leakage: changing labels after the origin does not change any forecast.
4. Forecasts are distributions; predict_future returns (horizons, 5).
5. With hard labels, the filtered HSMM elapsed duration equals the observed run length.
"""

import numpy as np

from . import config as cfg
from . import data
from .models import HSMM, K, evidence_from_labels, filter_states, forecast, \
    forecast_bins, time_bins_epochs
from .predict import load_models, predict_future


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok


def main():
    models = load_models(cfg.MODELS_DIR + "/models.joblib")
    hmm, hsmm = models["hmm"], models["hsmm_time"]
    rng = np.random.default_rng(1)
    ok = True

    b = rng.dirichlet(np.ones(K), size=3)
    hs = [1, 10, 60]
    fc = forecast(hmm, b, np.zeros((3, 60), int), hs)
    ok &= check("HMM forecast == b A^k",
                all(np.allclose(fc[k], b @ np.linalg.matrix_power(hmm.A[0], k)) for k in hs))

    for name in ["hmm", "hmm_time"]:
        m = models[name]
        eq = HSMM.from_hmm(m)
        ev = rng.dirichlet(np.ones(K), size=50)
        ev[[5, 6, 20]] = 1.0                          # unscored epochs
        tb = rng.integers(0, len(cfg.TIME_LABELS), 50)
        bins = rng.integers(0, len(cfg.TIME_LABELS), (1, 60))
        a = forecast(m, filter_states(m, ev, tb)[-1], bins, hs)
        c = forecast(eq, filter_states(eq, ev, tb)[-1], bins, hs)
        ok &= check(f"geometric HSMM == {name} (soft history, gaps, time bins)",
                    all(np.allclose(a[k], c[k], atol=1e-10) for k in hs))

    rec = data.load()[0]
    st = rec.stages.astype(int)
    t0 = rec.onset + 200
    tb = time_bins_epochs(len(st), rec.onset)
    alt = st.copy()
    alt[t0 + 1:] = rng.integers(0, K, len(st) - t0 - 1)
    for name, m in models.items():
        if name == "persistence":
            continue
        s1 = filter_states(m, evidence_from_labels(st), tb)[t0]
        s2 = filter_states(m, evidence_from_labels(alt), tb)[t0]
        bins = forecast_bins([(t0 - rec.onset) / 2], 60, m.time_dependent)
        f1, f2 = forecast(m, s1, bins, hs), forecast(m, s2, bins, hs)
        ok &= check(f"{name}: no leakage from labels after the origin",
                    all(np.array_equal(f1[k], f2[k]) for k in hs))
        ok &= check(f"{name}: forecasts sum to 1",
                    all(np.allclose(f1[k].sum(1), 1) and (f1[k] >= 0).all() for k in hs))

    states = filter_states(hsmm, evidence_from_labels(st), tb)
    good = True
    for t in range(rec.onset + 1, rec.onset + 400):
        run = 1
        while st[t - run] == st[t] and t - run >= 0:
            run += 1
        if st[t - run] < 0:
            continue                                  # run start not observed
        d = np.argmax(states[t][0, st[t]]) + 1
        good &= d == min(run, cfg.MAX_DUR) and np.isclose(states[t][0, st[t]].max(), 1)
    ok &= check("HSMM filter recovers elapsed duration from labels", good)

    p = predict_future([0.05, 0.05, 0.8, 0.05, 0.05], [2] * 30, 120, [0.5, 5, 30],
                       models=models)
    ok &= check("predict_future shape and normalisation",
                p.shape == (3, K) and np.allclose(p.sum(1), 1))
    p0 = predict_future(np.eye(K)[2], None, None, [5], model="hmm", models=models)
    ok &= check("predict_future without history (HMM) == e_N2 A^10",
                np.allclose(p0[0], np.linalg.matrix_power(hmm.A[0], 10)[2]))
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
