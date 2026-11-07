"""Get every electrode recording properly before an experiment.

Run it while acquire.py is streaming. Two stages:

  1. Contact: a live per-channel table refreshed every second, with a hint
     for each failing channel. Fix the electrodes until every channel is
     green for --hold seconds in a row. The participant sits still, eyes
     open, jaw relaxed.
  2. Alpha check: 15 s eyes open, then 15 s eyes closed. Closing the eyes
     should at least multiply 8-13 Hz power at Pz/PO7/POz/PO8 by 1.3
     (session p01_s1 gave 3-5x). This shows the posterior electrodes pick
     up brain signal, not just low noise.

The result is written to a JSON report. Exit code 0 = passed.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import quality as q

ALPHA_PASS = 1.3


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream-name", default="core8_eeg")
    ap.add_argument("--hold", type=float, default=10.0,
                    help="seconds all channels must stay good in a row (default: 10)")
    ap.add_argument("--ignore", nargs="+", default=[], metavar="CH",
                    help="channels to leave out of the pass criterion, e.g. a dead electrode")
    ap.add_argument("--alpha-seconds", type=float, default=15.0)
    ap.add_argument("--no-alpha", action="store_true", help="skip the eyes-closed alpha check")
    ap.add_argument("--out", default=None,
                    help="report JSON (default: data/calibration_<time>.json)")
    return ap.parse_args(argv)


def contact_stage(mon, args):
    streak_start, rows = None, []
    try:
        while True:
            rows = mon.assess(ignore=args.ignore)
            now = time.monotonic()
            streak_start = (streak_start or now) if q.ok(rows) else None
            held = now - streak_start if streak_start else 0.0
            sys.stdout.write("\033[H\033[J")  # clear the terminal
            print("STAGE 1: electrode contact. Sit still, eyes open, jaw relaxed. Ctrl-C to give up.\n")
            print(q.format_table(rows))
            bar = "#" * int(20 * min(held / args.hold, 1))
            print(f"\nAll good for {held:4.1f} / {args.hold:g} s  [{bar:<20s}]")
            if held >= args.hold:
                return True, rows
            time.sleep(1.0)
    except KeyboardInterrupt:
        return False, rows


def countdown(text, seconds):
    print(f"\a\n{text}")
    for left in range(int(seconds), 0, -1):
        print(f"  {left:2d} s", end="\r", flush=True)
        time.sleep(1)


def alpha_stage(mon, args, use):
    input("\nSTAGE 2: alpha check. Press Enter, then follow the beeps.")
    settle = 2.0  # skip the moment of opening/closing the eyes
    countdown("Eyes OPEN, look at a point ahead.", settle + args.alpha_seconds)
    open_ = mon.latest(args.alpha_seconds)
    countdown("Eyes CLOSED, relax (beep when done).", settle + args.alpha_seconds)
    closed = mon.latest(args.alpha_seconds)
    print("\aOpen your eyes.\n")
    ratio = q.band_power(closed, mon.fs, 8, 13) / q.band_power(open_, mon.fs, 8, 13)
    per_ch = {lab: float(r) for lab, r in zip(mon.labels, ratio)}
    post = [per_ch[lab] for lab in q.POSTERIOR if lab in per_ch and lab in use]
    score = float(np.median(post)) if post else float("nan")
    print("Alpha, eyes closed / open:  " + "  ".join(f"{k} {v:.1f}x" for k, v in per_ch.items()))
    passed = bool(post) and score >= ALPHA_PASS
    print(f"Posterior median {score:.1f}x (need >= {ALPHA_PASS:g}x): {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("Check that the eyes really were closed, and Pz/PO7/POz/PO8 contact.")
    return passed, {"ratio_per_channel": per_ch, "posterior_median": score}


def main(argv=None):
    args = parse_args(argv)
    mon = q.QualityMonitor(args.stream_name, buffer=max(20.0, args.alpha_seconds + 5))
    use = [lab for lab in mon.labels if lab not in args.ignore]
    print("Collecting the first seconds of signal...")

    contact_ok, rows = contact_stage(mon, args)
    print("\nContact: " + ("PASS" if contact_ok else "NOT PASSED"))
    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stream": args.stream_name,
              "ignored": args.ignore, "contact_passed": contact_ok, "channels": rows}

    alpha_ok = True
    if contact_ok and not args.no_alpha:
        alpha_ok, report["alpha"] = alpha_stage(mon, args, use)
    report["passed"] = contact_ok and alpha_ok
    mon.close()

    out = args.out or time.strftime("data/calibration_%Y%m%d_%H%M%S.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nCalibration {'PASSED' if report['passed'] else 'FAILED'}; report written to {out}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
