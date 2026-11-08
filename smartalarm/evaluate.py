"""Compare fixed, reactive and Bayesian alarms on held-out nights.

    python -m smartalarm.evaluate              # main table + sensitivity analysis
    python -m smartalarm.evaluate --quick      # main table only

Each held-out night (test subjects of models/sleep/models.joblib) gets four deadlines,
6, 6.5, 7 and 7.5 h after its sleep onset, each with a window of --window minutes before
it. With --windows asleep, deadlines run every 30 min from 4 to 7.5 h and only windows in
which the person was asleep throughout are kept, so nobody can be woken from wake. Observations come from the simulated EEG stager (default) or the true labels
(--input labels). All policies run through the same Session code and are scored with the
same cost: stage cost of the TRUE stage at the alarm + sleep value x minutes before the
deadline. These are proxies, not validated measures of post-awakening performance.
"""

import argparse
import json
import os

import joblib
import numpy as np

from sleep import data
from sleep.models import from_dict
from . import config as cfg
from .session import Session
from .sources import ReplaySource, stager_prior

POLICIES = ["fixed", "reactive", "bayes"]
DEADLINES_H = [6.0, 6.5, 7.0, 7.5]
# --windows asleep: deadlines every 30 min from 4 to 7.5 h after onset, keeping only windows
# in which the person is asleep (no scored W) from the earliest to the latest time
ASLEEP_DEADLINES_H = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]


def windows(nights, window_min, kind="all"):
    """(night index, earliest, latest) for every evaluation window.

    'all': deadlines 6-7.5 h after onset. 'asleep': only windows where the person was
    asleep throughout, so nobody can be woken from wake; chosen from the true stages, the
    same windows for every policy, and never shown to the model."""
    out = []
    for i, r in enumerate(nights):
        for h in (ASLEEP_DEADLINES_H if kind == "asleep" else DEADLINES_H):
            latest = r.onset + int(h * 120)
            if latest >= len(r.stages):
                continue
            earliest = latest - int(window_min * 2)
            if kind == "asleep" and np.any(r.stages[earliest:latest + 1] == 0):
                continue
            out.append((i, earliest, latest))
    return out


def run(model, nights, prior, window_min, stage_cost, sleep_value, mode, kind="all"):
    """List of per-alarm records for every policy."""
    costs = np.array([stage_cost[s] for s in cfg.STAGES])
    rows = []
    for i, earliest, latest in windows(nights, window_min, kind):
        r = nights[i]
        h = (latest - r.onset) / 120
        if True:
            for pol in POLICIES:
                src = ReplaySource(r, mode, prior, seed=1000 + i)   # same noise for all
                s = Session(src, model, earliest, latest, pol, stage_cost, sleep_value,
                            log=False)
                s.advance(10 ** 6)
                t = s.fired["epoch"]
                truth = src.truth
                k = t
                while truth[k] < 0 and k > 0:
                    k -= 1
                stage = int(truth[k])
                mins = (latest - t) / 2
                post = s.inf.posterior()
                lost = np.sum(truth[t:latest] > 0) / 2           # true sleep given up
                rows.append(dict(policy=pol, subject=r.subject, rec=r.rec_id, deadline_h=h,
                                 stage=cfg.STAGES[stage], minutes_early=mins,
                                 sleep_lost=float(lost),
                                 realised=float(costs[stage] + sleep_value * mins),
                                 expected=float(post @ costs + sleep_value * mins)))
    return rows


def summarise(rows):
    out = {}
    fixed = {(r["rec"], r["deadline_h"]): r["realised"] for r in rows if r["policy"] == "fixed"}
    for pol in POLICIES:
        rs = [r for r in rows if r["policy"] == pol]
        real = np.array([r["realised"] for r in rs])
        diff = np.array([r["realised"] - fixed[(r["rec"], r["deadline_h"])] for r in rs])
        subj = np.array([r["subject"] for r in rs])
        out[pol] = dict(
            n=len(rs),
            stage_pct={s: 100 * np.mean([r["stage"] == s for r in rs]) for s in cfg.STAGES},
            realised=real.mean(), realised_se=real.std(ddof=1) / np.sqrt(len(real)),
            expected=np.mean([r["expected"] for r in rs]),
            vs_fixed=diff.mean(), vs_fixed_ci=bootstrap(diff, subj),
            minutes_early=np.mean([r["minutes_early"] for r in rs]),
            sleep_lost=np.mean([r["sleep_lost"] for r in rs]),
        )
    return out


def bootstrap(x, groups, n=2000, seed=0):
    g, inv = np.unique(groups, return_inverse=True)
    sums, cnt = np.bincount(inv, x), np.bincount(inv)
    draw = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    m = sums[draw].sum(1) / cnt[draw].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def print_main(summary):
    print(f"{'policy':<10}{'cost':>7}{'±SE':>6}{'vs fixed [95% CI]':>24}{'expected':>10}"
          f"{'min early':>11}{'sleep lost':>12}   wake stage %  " + " ".join(
              f"{s:>4}" for s in cfg.STAGES))
    for pol, m in summary.items():
        ci = "[{:+.2f}, {:+.2f}]".format(*m["vs_fixed_ci"])
        print(f"{pol:<10}{m['realised']:>7.2f}{m['realised_se']:>6.2f}"
              f"{m['vs_fixed']:>+9.2f} {ci:>14}{m['expected']:>10.2f}"
              f"{m['minutes_early']:>11.1f}{m['sleep_lost']:>12.1f}"
              f"{'':>17}" + " ".join(f"{m['stage_pct'][s]:>4.0f}" for s in cfg.STAGES))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=float, default=cfg.DEFAULT_WINDOW_MIN)
    ap.add_argument("--input", choices=["stager", "labels"], default="stager")
    ap.add_argument("--quick", action="store_true", help="skip the sensitivity analysis")
    ap.add_argument("--windows", choices=["all", "asleep"], default="all",
                    help="'asleep': only windows where the person was asleep throughout")
    ap.add_argument("--out", default="models/sleep/alarm_evaluation.json")
    args = ap.parse_args(argv)

    blob = joblib.load(cfg.MODEL_FILE)
    model = from_dict(blob["models"][cfg.MODEL_NAME])
    recs = data.load()
    test_s = set(blob["test_subjects"])
    nights = [r for r in recs if r.subject in test_s]
    prior = stager_prior([r for r in recs if r.subject not in test_s])
    base_cost = dict(cfg.STAGE_COST)

    rows = run(model, nights, prior, args.window, base_cost, cfg.SLEEP_VALUE_PER_MIN,
               args.input, args.windows)
    main_sum = summarise(rows)
    print(f"{len(windows(nights, args.window, args.windows))} windows ({args.windows}) on "
          f"{len(nights)} held-out nights, "
          f"{args.window:g} min window, input: {args.input}; costs {base_cost}, "
          f"sleep value {cfg.SLEEP_VALUE_PER_MIN}/min (lower cost is better)\n")
    print_main(main_sum)
    result = dict(settings=dict(window=args.window, input=args.input, stage_cost=base_cost,
                                sleep_value=cfg.SLEEP_VALUE_PER_MIN, nights=len(nights)),
                  main=main_sum, sensitivity=[])

    if not args.quick:
        variants = ([("N3 cost", dict(base_cost, N3=v), cfg.SLEEP_VALUE_PER_MIN, args.window)
                     for v in (5.0, 20.0)] +
                    [("REM cost", dict(base_cost, REM=v), cfg.SLEEP_VALUE_PER_MIN, args.window)
                     for v in (1.0, 8.0)] +
                    [("sleep value", base_cost, v, args.window) for v in (0.03, 0.3)] +
                    [("window", base_cost, cfg.SLEEP_VALUE_PER_MIN, w) for w in (15, 60)])
        print("\nSensitivity (realised cost, % woken from N3, minutes early)")
        print(f"{'setting':<22}" + "".join(f"{p:>26}" for p in POLICIES))
        for label, sc, sv, w in variants:
            sm = summarise(run(model, nights, prior, w, sc, sv, args.input, args.windows))
            val = {"N3 cost": sc["N3"], "REM cost": sc["REM"], "sleep value": sv,
                   "window": w}[label]
            print(f"{label + ' = ' + format(val, 'g'):<22}" + "".join(
                f"{sm[p]['realised']:>10.2f}{sm[p]['stage_pct']['N3']:>7.0f}%"
                f"{sm[p]['minutes_early']:>8.1f}" for p in POLICIES))
            result["sensitivity"].append(dict(setting=label, value=val, summary=sm))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
