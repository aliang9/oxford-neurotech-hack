"""Train and evaluate the dance-move classifier on recorded blocks.

    python -m dance.train data/dance/p01_b*.csv --out models/p01.joblib

1. Cut every block into 1.5 s windows (every 0.25 s) that lie fully inside
   one trial, skipping the first 0.5 s after GO.
2. Leave-one-block-out evaluation: train on all blocks but one, test on it,
   repeat. Overlapping windows never end up on both sides of a split.
   A majority-class baseline is scored on the same folds.
3. Fit on all blocks (except --holdout) and save the model.
"""

import argparse
import os
import re

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict

from . import config as cfg
from .features import extract, feature_names, windows
from .stream import load_block, sample_labels


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="block CSVs from dance.record")
    ap.add_argument("--out", default="models/dance.joblib")
    ap.add_argument("--holdout", type=int, nargs="+", default=[], metavar="BLOCK",
                    help="leave these blocks out entirely, e.g. to replay them with dance.live")
    ap.add_argument("--moves", nargs="+", default=None, help="only use these moves")
    return ap.parse_args(argv)


def make_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced", n_jobs=-1, random_state=0)


def block_number(path):
    match = re.search(r"_b(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


def load_windows(paths, moves=None):
    """Features X, labels y, block of each window, trial of each window."""
    X, y, blocks, trials = [], [], [], []
    for k, path in enumerate(paths):
        block = block_number(path) or k + 1
        data, _ = load_block(path)
        codes, trial = sample_labels(data[:, cfg.MARKER_CH], cfg.ONSET_SKIP_S)
        n = 0
        for start, win in windows(data):
            end = start + cfg.WIN - 1
            if not codes[start] or trial[start] != trial[end]:
                continue  # not fully inside one trial
            name = cfg.MOVES[codes[start]][0]
            if moves and name not in moves:
                continue
            X.append(extract(win))
            y.append(name)
            blocks.append(block)
            trials.append(f"{block}-{trial[start]}")
            n += 1
        print(f"{path}: block {block}, {trial.max()} trials, {n} windows")
    if not X:
        raise SystemExit("No labelled windows found.")
    return np.array(X), np.array(y), np.array(blocks), np.array(trials)


def print_confusion(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=classes)
    w = max(len(c) for c in classes)
    print("\nConfusion matrix (rows = true, cols = predicted, in windows):")
    print(" " * (w + 2) + " ".join(f"{c[:6]:>6s}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:>{w}s}  " + " ".join(f"{v:6d}" for v in row))


def evaluate(X, y, blocks, trials):
    if len(set(blocks)) > 1:
        cv, groups = LeaveOneGroupOut(), blocks
        print(f"\nLeave-one-block-out over {len(set(blocks))} blocks")
    else:
        cv, groups = GroupKFold(5), trials
        print("\nOnly one block: 5-fold CV grouped by trial. Optimistic; record more blocks.")

    classes = sorted(set(y))
    for name, clf in [("majority baseline", DummyClassifier(strategy="most_frequent")),
                      ("random forest", make_model())]:
        pred = cross_val_predict(clf, X, y, groups=groups, cv=cv)
        print(f"  {name:18s} accuracy {accuracy_score(y, pred):.2f}   "
              f"macro-F1 {f1_score(y, pred, average='macro'):.2f}")
    print(f"  (chance with {len(classes)} balanced classes: {1 / len(classes):.2f})")

    print_confusion(y, pred, classes)
    print("\n" + classification_report(y, pred, labels=classes, zero_division=0))


def main(argv=None):
    args = parse_args(argv)
    paths = [p for p in args.files if block_number(p) not in args.holdout]
    if args.holdout:
        print(f"Holding out blocks {args.holdout}: not used for evaluation or training.")
    X, y, blocks, trials = load_windows(paths, args.moves)
    names, counts = np.unique(y, return_counts=True)
    print(f"\n{len(y)} windows: " + ", ".join(f"{n} {c}" for n, c in zip(names, counts)))

    evaluate(X, y, blocks, trials)

    model = make_model().fit(X, y)
    top = np.argsort(model.feature_importances_)[::-1][:10]
    print("Top features: " + ", ".join(feature_names()[i] for i in top))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"model": model, "classes": list(model.classes_),
                 "blocks": sorted(set(blocks.tolist()))}, args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
