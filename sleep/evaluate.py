"""Evaluate the forecasts on held-out subjects.

    python -m sleep.evaluate               # test subjects of models/sleep/models.joblib
    python -m sleep.evaluate --folds 5     # 5-fold cross-validation grouped by subject

Origins: every scored epoch from sleep onset to the end of the trimmed recording. Targets:
the stage h minutes later, if scored and inside the recording. Each model's state at the
origin comes from a causal filter over the labels up to and including the origin (the
HSMM's elapsed duration included), so nothing after the origin is used. Time since onset
is known at the origin because onset is, by definition, at or before it.

Metrics per horizon: multiclass log loss (nats), Brier score (sum over 5 classes, 0-2),
accuracy of the most probable stage, top-label expected calibration error (10 bins), and a
class-wise reliability curve. 95% intervals bootstrap the test subjects (paired for the
difference from the stationary HMM).
"""

import argparse
import json
import os

import joblib
import numpy as np

from . import config as cfg
from . import data
from .models import K, evidence_from_labels, filter_states, forecast, forecast_bins, \
    from_dict, time_bins_epochs
from .train import MODEL_NAMES, fit_all

EPS = 1e-12


def stacked_states(model, rec):
    """Filtered state after every epoch of a recording, stacked on the batch axis."""
    st = rec.stages.astype(int)
    tb = time_bins_epochs(len(st), rec.onset)
    return np.concatenate(filter_states(model, evidence_from_labels(st), tb))


def recording_forecasts(model, rec, ks, name=None):
    """{k: (probs (n, K), targets (n,))} for every valid origin of one recording."""
    st = rec.stages.astype(int)
    T = len(st)
    origins = np.flatnonzero((np.arange(T) >= rec.onset) & (st >= 0))
    if name == "persistence":
        fc = {k: model.forecast(st[origins], k) for k in ks}
    else:
        state = stacked_states(model, rec)[origins]
        bins = forecast_bins((origins - rec.onset) / cfg.EPOCHS_PER_MIN, max(ks),
                             model.time_dependent)
        fc = forecast(model, state, bins, ks)
    out = {}
    for k in ks:
        tgt = origins + k
        ok = tgt < T
        ok[ok] = st[tgt[ok]] >= 0
        out[k] = (fc[k][ok], st[tgt[ok]])
    return out


def collect(models, recs, ks):
    """{name: {k: (P, y, subject)}} pooled over recordings."""
    res = {}
    for name, m in models.items():
        parts = {k: ([], [], []) for k in ks}
        for r in recs:
            for k, (p, y) in recording_forecasts(m, r, ks, name).items():
                parts[k][0].append(p)
                parts[k][1].append(y)
                parts[k][2].append(np.full(len(y), r.subject))
        res[name] = {k: tuple(np.concatenate(a) for a in parts[k]) for k in ks}
    return res


def reliability(P, y, n_bins=10):
    """Class-wise reliability: every (prob, outcome) pair pooled over the 5 classes."""
    p = P.ravel()
    o = (np.arange(K)[None] == y[:, None]).ravel()
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    mp = np.bincount(idx, p, n_bins) / np.maximum(cnt, 1)
    fr = np.bincount(idx, o, n_bins) / np.maximum(cnt, 1)
    return dict(mean_pred=mp.tolist(), frac=fr.tolist(), count=cnt.tolist())


def ece(P, y, n_bins=10):
    conf = P.max(1)
    hit = P.argmax(1) == y
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    gap = np.abs(np.bincount(idx, conf, n_bins) - np.bincount(idx, hit, n_bins))
    return float(gap.sum() / len(y))


def per_origin_losses(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))


def subject_bootstrap(loss, subj, ref_loss=None, n=2000, seed=0):
    """95% CI of the mean loss (or of loss - ref_loss), resampling subjects."""
    subjects, inv = np.unique(subj, return_inverse=True)
    diff = loss if ref_loss is None else loss - ref_loss
    s_sum = np.bincount(inv, diff)
    s_n = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(subjects), (n, len(subjects)))
    means = s_sum[draw].sum(1) / s_n[draw].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metrics(res, ks):
    out = {}
    for name in res:
        out[name] = {}
        for k in ks:
            P, y, subj = res[name][k]
            ll = per_origin_losses(P, y)
            Y = np.eye(K)[y]
            ref = per_origin_losses(*res["hmm"][k][:2])
            out[name][k] = dict(
                n=int(len(y)),
                log_loss=float(ll.mean()),
                log_loss_ci=subject_bootstrap(ll, subj),
                log_loss_vs_hmm=float((ll - ref).mean()),
                log_loss_vs_hmm_ci=subject_bootstrap(ll, subj, ref),
                brier=float(((P - Y) ** 2).sum(1).mean()),
                accuracy=float((P.argmax(1) == y).mean()),
                ece=ece(P, y),
                reliability=reliability(P, y),
            )
    return out


def print_table(m, ks):
    print(f"{'model':<12}{'horizon':>8}{'log loss':>10}{'95% CI':>16}{'vs HMM':>9}"
          f"{'95% CI':>18}{'Brier':>8}{'acc':>7}{'ECE':>7}")
    for k in ks:
        for name in m:
            r = m[name][k]
            ci = "[{:.3f},{:.3f}]".format(*r["log_loss_ci"])
            dci = "[{:+.3f},{:+.3f}]".format(*r["log_loss_vs_hmm_ci"])
            print(f"{name:<12}{k / cfg.EPOCHS_PER_MIN:>6.0f} m{r['log_loss']:>10.3f}{ci:>16}"
                  f"{r['log_loss_vs_hmm']:>+9.3f}{dci:>18}{r['brier']:>8.3f}"
                  f"{r['accuracy']:>7.3f}{r['ece']:>7.3f}")
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default=os.path.join(cfg.MODELS_DIR, "models.joblib"))
    ap.add_argument("--epochs", default=cfg.EPOCHS_CSV)
    ap.add_argument("--folds", type=int, default=0,
                    help="k-fold CV grouped by subject instead of the saved split")
    ap.add_argument("--horizons", type=float, nargs="+", default=cfg.HORIZONS_MIN)
    ap.add_argument("--out", default=None, help="metrics JSON (default: next to the models)")
    args = ap.parse_args(argv)

    ks = [int(round(h * cfg.EPOCHS_PER_MIN)) for h in args.horizons]
    recs = data.load(args.epochs)
    if args.folds:
        subjects = np.array(sorted({r.subject for r in recs}))
        rng = np.random.default_rng(cfg.SEED)
        folds = np.array_split(rng.permutation(subjects), args.folds)
        res = None
        for i, test_s in enumerate(folds):
            test_s = set(test_s.tolist())
            models = fit_all([r for r in recs if r.subject not in test_s])
            # fit_all's persistence covers 30 min; refit if longer horizons were asked for
            if max(ks) > max(models["persistence"].p_same):
                from .models import Persistence
                models["persistence"] = Persistence.fit(
                    [r for r in recs if r.subject not in test_s], max(ks))
            r_i = collect(models, [r for r in recs if r.subject in test_s], ks)
            res = r_i if res is None else {
                n: {k: tuple(np.concatenate([a, b]) for a, b in zip(res[n][k], r_i[n][k]))
                    for k in ks} for n in res}
            print(f"fold {i + 1}/{args.folds}: {len(test_s)} test subjects")
        desc = f"{args.folds}-fold CV over {len(subjects)} subjects"
    else:
        blob = joblib.load(args.models)
        models = {n: from_dict(blob["models"][n]) for n in MODEL_NAMES}
        test_s = set(blob["test_subjects"])
        test = [r for r in recs if r.subject in test_s]
        res = collect(models, test, ks)
        desc = f"{len(test_s)} held-out subjects, {len(test)} nights"
    print(f"evaluation: {desc}")
    m = metrics(res, ks)
    print_table(m, ks)

    out = args.out or os.path.join(os.path.dirname(args.models),
                                   "metrics_cv.json" if args.folds else "metrics.json")
    with open(out, "w") as f:
        json.dump({"description": desc, "horizons_min": [k / cfg.EPOCHS_PER_MIN for k in ks],
                   "models": {n: {str(k): v for k, v in m[n].items()} for n in m}}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
