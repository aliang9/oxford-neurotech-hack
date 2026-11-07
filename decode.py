"""Offline task decoding from acquire.py --record files.

    HDF5 (raw EEG + markers) -> zero-phase notch + bandpass -> epochs per task
    -> reject epochs over a peak-to-peak threshold
    -> log band power (theta/alpha/beta/gamma x 8 channels)
    -> shrinkage LDA, stratified k-fold CV + permutation test

The gpype part (HDF5Reader -> filters -> Epochs -> Collector, run in batch)
follows examples s4e10 and s2e17. Features and classifier are plain
numpy/scipy/scikit-learn.
"""

import argparse

import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gpype as gp

from tasks import TASKS

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="recordings from acquire.py --record")
    ap.add_argument("--tmin", type=float, default=0.5, help="epoch start after cue, s (default: 0.5)")
    ap.add_argument("--tmax", type=float, default=5.0, help="epoch end after cue, s (default: 5)")
    ap.add_argument("--mains", type=float, default=50.0)
    ap.add_argument("--band", type=float, nargs=2, default=(1.0, 40.0), metavar=("LO", "HI"))
    ap.add_argument("--reject", type=float, default=150.0,
                    help="drop epochs whose peak-to-peak exceeds this on any used channel, "
                         "uV (0 = keep all)")
    ap.add_argument("--bad-factor", type=float, default=2.0,
                    help="exclude channels whose median epoch p-p is this many times the "
                         "median across channels (0 = keep all channels)")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="CH",
                    help="channels to leave out, e.g. --exclude Fz C4")
    ap.add_argument("--tasks", nargs="+", default=None, metavar="TASK",
                    help="only decode these tasks, e.g. --tasks math rest_open (default: all)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=200)
    return ap.parse_args(argv)


def read_events(path):
    with gp.Pipeline() as p:
        p.connect(gp.HDF5Reader(file_name=path, mode="batch"), gp.Collector())
        r = p.run()
    return r.rate, sorted({e.label for e in r.events})


def cut(path, condition, args):
    """(time, channel, trial) epochs of one condition, filtered offline."""
    with gp.Pipeline() as p:
        nodes = [gp.HDF5Reader(file_name=path, mode="batch"),
                 gp.ChannelSelector(roles=gp.Constants.ChannelRoles.SIGNAL),
                 gp.Bandstop(f_lo=args.mains - 2, f_hi=args.mains + 2, phase="zero"),
                 gp.Bandpass(f_lo=args.band[0], f_hi=args.band[1], phase="zero"),
                 gp.Epochs(condition=condition, tmin=args.tmin, tmax=args.tmax),
                 gp.Collector()]
        for a, b in zip(nodes, nodes[1:]):
            p.connect(a, b)
        return p.run()


def band_power(epochs, fs):
    """(trial, channel*band) log band power from (time, channel, trial) epochs."""
    f, pxx = welch(epochs, fs=fs, nperseg=min(int(fs), epochs.shape[0]), axis=0)
    feats = [np.log(pxx[(f >= lo) & (f < hi)].mean(axis=0)) for lo, hi in BANDS.values()]
    return np.concatenate(feats, axis=0).T  # (trial, band*channel)


def load_epochs(args):
    """All task epochs from all files: list of (task name, (time, channel) array)."""
    epochs, labels = [], None
    for path in args.files:
        fs, conditions = read_events(path)
        for cond in conditions:
            code = int(float(cond))
            if code not in TASKS or (args.tasks and TASKS[code][0] not in args.tasks):
                continue
            r = cut(path, cond, args)
            labels = labels or (list(r.labels) if r.labels else
                                [f"ch{i + 1}" for i in range(r.data.shape[1])])
            name = TASKS[int(float(cond))][0]
            epochs += [(name, r.data[:, :, k]) for k in range(r.data.shape[2])]
            print(f"{path}: {name:12s} {r.data.shape[2]:3d} epochs")
    if not epochs:
        raise SystemExit("No task markers found in the recordings.")
    return fs, labels, epochs


def pick_channels(labels, epochs, args):
    """Indices of the channels to use: drop --exclude and channels far noisier than the rest."""
    pp = np.median([np.ptp(e, axis=0) for _, e in epochs], axis=0)  # median p-p per channel
    ref = np.median(pp)
    print("\nMedian epoch peak-to-peak per channel (uV):")
    print("  " + "  ".join(f"{lab:>5s}" for lab in labels))
    print("  " + "  ".join(f"{v:5.0f}" for v in pp))
    bad = {lab for lab, v in zip(labels, pp) if args.bad_factor and v > args.bad_factor * ref}
    bad |= set(args.exclude)
    if bad:
        print(f"Excluding {sorted(bad)}: noisy (> {args.bad_factor:g} x the median {ref:.0f} uV) "
              "or --exclude. Usually poor electrode contact; re-wet/re-seat them next session.")
    use = [i for i, lab in enumerate(labels) if lab not in bad]
    if not use:
        raise SystemExit("All channels excluded.")
    return use


def main(argv=None):
    args = parse_args(argv)
    fs, labels, epochs = load_epochs(args)
    use = pick_channels(labels, epochs, args)
    labels = [labels[i] for i in use]

    X, y, rejected = [], [], {}
    for name, e in epochs:
        e = e[:, use]
        if args.reject and np.ptp(e, axis=0).max() >= args.reject:
            rejected[name] = rejected.get(name, 0) + 1
            continue
        X.append(band_power(e[:, :, None], fs)[0])
        y.append(name)
    if rejected:
        print(f"Rejected epochs (> {args.reject:g} uV p-p on {', '.join(labels)}): "
              + ", ".join(f"{n} {k}" for n, k in sorted(rejected.items())))
    if not X:
        raise SystemExit("Every epoch was rejected. Check the channel table above, then raise "
                         "--reject or --exclude the noisy channels.")
    X, y = np.vstack(X), np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n{len(y)} epochs, classes: " + ", ".join(f"{c} ({n})" for c, n in zip(classes, counts)))
    if len(classes) < 2 or counts.min() < args.folds:
        raise SystemExit(f"Need at least 2 classes with >= {args.folds} clean epochs each.")

    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    cv = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    score, _, p = permutation_test_score(clf, X, y, cv=cv, n_permutations=args.permutations,
                                         random_state=0)
    chance = counts.max() / counts.sum()
    print(f"\nCV accuracy {score:.2f}  (majority-class chance {chance:.2f}, "
          f"permutation p = {p:.3f}, {args.permutations} permutations)")

    pred = cross_val_predict(clf, X, y, cv=cv)
    cm = confusion_matrix(y, pred, labels=classes)
    w = max(len(c) for c in classes)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(" " * (w + 2) + "  ".join(f"{c[:6]:>6s}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:>{w}s}  " + "  ".join(f"{v:6d}" for v in row))

    names = [f"{lab}:{band}" for band in BANDS for lab in labels]
    F, _ = f_classif(X, y)
    top = np.argsort(F)[::-1][:8]
    print("\nMost discriminative features (ANOVA F):  " + ", ".join(f"{names[i]} ({F[i]:.1f})" for i in top))
    print("If these are mostly Fz or gamma, suspect eye/muscle artifacts rather than brain activity.")


if __name__ == "__main__":
    main()
