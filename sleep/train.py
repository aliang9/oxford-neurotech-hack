"""Fit the sleep-dynamics models on a subject-level split, and on all subjects.

    python -m sleep.train

Writes
  models/sleep/models.joblib      fitted on the training subjects only (for evaluation and
                                  the dashboard), with the subject split
  models/sleep/models_all.joblib  fitted on every subject (for deployment, predict.py)
Both hold plain dicts of numpy arrays (see models.from_dict), not pickled classes.
"""

import argparse
import os

import joblib
import numpy as np

from . import config as cfg
from . import data
from .models import HMM, HSMM, Persistence, count

MODEL_NAMES = ["persistence", "hmm", "hmm_time", "hsmm", "hsmm_time"]


def split_subjects(recs, test_fraction=cfg.TEST_FRACTION, seed=cfg.SEED):
    """(train subjects, test subjects): both nights of a subject stay on one side."""
    subjects = np.array(sorted({r.subject for r in recs}))
    rng = np.random.default_rng(seed)
    test = set(rng.choice(subjects, int(round(test_fraction * len(subjects))), replace=False))
    return sorted(set(subjects) - test), sorted(test)


def fit_all(recs, max_k=max(cfg.HORIZONS_MIN) * cfg.EPOCHS_PER_MIN):
    stats = count(recs)
    return {
        "persistence": Persistence.fit(recs, max_k),
        "hmm": HMM.fit(stats, time_dependent=False),
        "hmm_time": HMM.fit(stats, time_dependent=True),
        "hsmm": HSMM.fit(stats, time_dependent=False),
        "hsmm_time": HSMM.fit(stats, time_dependent=True),
    }


def save(models, path, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"models": {k: m.to_dict() for k, m in models.items()}, **meta}, path)
    print(f"wrote {path}")


def describe(models):
    A = models["hmm"].A[0]
    print("stationary HMM transition matrix (rows: from, cols: to)")
    print("       " + "".join(f"{s:>7}" for s in cfg.STAGES))
    for s, row in zip(cfg.STAGES, A):
        print(f"  {s:>4} " + "".join(f"{p:7.3f}" for p in row))
    pmf = models["hsmm"].duration_pmf()
    d = np.arange(1, cfg.MAX_DUR + 1) / cfg.EPOCHS_PER_MIN
    print("HSMM stage durations (stationary): median / mean, minutes")
    for s, f in zip(cfg.STAGES, pmf):
        med = d[np.searchsorted(np.cumsum(f), 0.5)]
        print(f"  {s:>4}  {med:5.1f} / {np.dot(f, d):5.1f}"
              f"   (geometric HMM mean {0.5 / (1 - A[cfg.STAGES.index(s)].max()):5.1f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", default=cfg.EPOCHS_CSV)
    ap.add_argument("--out-dir", default=cfg.MODELS_DIR)
    args = ap.parse_args(argv)

    recs = data.load(args.epochs)
    train_s, test_s = split_subjects(recs)
    train = [r for r in recs if r.subject in train_s]
    print(f"{len(train_s)} training subjects ({len(train)} nights), "
          f"{len(test_s)} test subjects ({len(recs) - len(train)} nights)")
    models = fit_all(train)
    describe(models)
    save(models, os.path.join(args.out_dir, "models.joblib"),
         train_subjects=train_s, test_subjects=test_s)
    save(fit_all(recs), os.path.join(args.out_dir, "models_all.joblib"),
         train_subjects=sorted({r.subject for r in recs}), test_subjects=[])


if __name__ == "__main__":
    main()
