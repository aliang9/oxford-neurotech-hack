"""Record one block of cued dance moves to CSV.

    python -m dance.record --subject p01 --block 1

Needs acquire.py running (it publishes core8_raw and receives the markers).

Each trial:  rest (+) -> "Next: MOVE" preview -> GO (beep, move code sent)
             -> dance for --duration s -> STOP (beep, 99 sent).

The markers go to acquire.py over UDP, which puts them in the marker channel
of the same LSL stream as the EEG, so labels and samples are aligned without
any clock bookkeeping. A background thread writes that stream to
data/dance/<subject>_b<block>.csv:  time, Fz..PO8, marker.

Keys: Space = start, Esc = abort.
"""

import argparse
import csv
import os
import random
import socket
import sys
import threading
import time
import types

from PySide6 import QtCore, QtGui, QtWidgets

from . import config as cfg
from .stream import LSLSource, load_block, sample_labels

NAMES = {name: code for code, (name, _) in cfg.MOVES.items()}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5, help="repetitions per move (default: 5)")
    ap.add_argument("--moves", nargs="+", default=list(NAMES), choices=list(NAMES))
    ap.add_argument("--duration", type=float, default=4.0, help="seconds of dancing (default: 4)")
    ap.add_argument("--preview", type=float, default=1.0, help="'Next: ...' time (default: 1)")
    ap.add_argument("--rest", type=float, nargs=2, default=(2.0, 3.0), metavar=("MIN", "MAX"))
    ap.add_argument("--out-dir", default="data/dance")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--port", type=int, default=cfg.MARKER_PORT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--skip-check", action="store_true", help="skip the signal-quality check")
    ap.add_argument("--ignore", nargs="+", default=[], metavar="CH",
                    help="channels the signal check should ignore")
    return ap.parse_args(argv)


def make_schedule(codes, reps, rng):
    """Balanced shuffled list, no move three times in a row."""
    schedule = list(codes) * reps
    while True:
        rng.shuffle(schedule)
        if all(not (a == b == c) for a, b, c in zip(schedule, schedule[1:], schedule[2:])):
            return schedule


class Recorder(threading.Thread):
    """Writes every sample of the raw LSL stream to CSV until stop()."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.source = LSLSource(cfg.RAW_STREAM)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time", *cfg.CHANNELS, "marker"])
        self.n = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            samples, times = self.source.pull(timeout=0.2)
            for t, row in zip(times, samples):
                self.writer.writerow([f"{t:.4f}", *(f"{v:.3f}" for v in row)])
            self.n += len(times)
        self.file.close()

    def stop(self):
        self._stop.set()
        self.join()


class Cues(QtWidgets.QLabel):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rng = random.Random(args.seed)
        self.schedule = make_schedule([NAMES[m] for m in args.moves], args.reps, self.rng)
        self.i = 0
        self.started = self.done = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setWordWrap(True)
        self.setFont(QtGui.QFont("Helvetica", 56, QtGui.QFont.Bold))
        n = len(self.schedule)
        mins = n * (args.preview + args.duration + sum(args.rest) / 2) / 60
        self.show_text(f"Block {args.block}: {n} moves, about {mins:.0f} min.\n\n"
                       "Press Space to start.", "white")

    def show_text(self, text, color):
        self.setStyleSheet(f"background: black; color: {color};")
        self.setText(text)
        self.repaint()  # draw now, so the marker goes out with the cue

    def send(self, code):
        self.sock.sendto(str(code).encode(), ("127.0.0.1", self.args.port))

    def after(self, seconds, fn):
        t = QtCore.QTimer(self, singleShot=True, timerType=QtCore.Qt.PreciseTimer)
        t.timeout.connect(lambda: None if self.done else fn())
        t.timeout.connect(t.deleteLater)
        t.start(int(1000 * seconds))

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.finish("Aborted.")
        elif e.key() == QtCore.Qt.Key_Space and not self.started:
            self.started = True
            self.rest()

    def rest(self):
        if self.i >= len(self.schedule):
            return self.finish("Done, thank you!")
        self.show_text("+", "#888")
        self.after(self.rng.uniform(*self.args.rest), self.preview)

    def preview(self):
        text = cfg.MOVES[self.schedule[self.i]][1]
        self.show_text(f"Next ({self.i + 1}/{len(self.schedule)}):\n\n{text}", "#888")
        self.after(self.args.preview, self.go)

    def go(self):
        code = self.schedule[self.i]
        self.show_text(f"GO!\n\n{cfg.MOVES[code][1]}", "#3f3")
        self.send(code)
        QtWidgets.QApplication.beep()
        self.after(self.args.duration, self.stop)

    def stop(self):
        self.send(cfg.TRIAL_END)
        QtWidgets.QApplication.beep()
        self.show_text("STOP", "#f44")
        self.i += 1
        self.after(0.5, self.rest)

    def finish(self, text):
        if self.done:
            return
        self.done = True  # a trial cut short by Esc has no end marker, so it is dropped
        self.show_text(text, "white")
        QtCore.QTimer.singleShot(1500, QtWidgets.QApplication.quit)


def signal_check(args):
    """The same pre-start check as tasks.py, on the filtered stream, sitting still."""
    import tasks
    check_args = types.SimpleNamespace(stream_name=cfg.QUALITY_STREAM, ignore=args.ignore)
    tasks.signal_check(check_args).close()


def summary(path):
    data, times = load_block(path)
    if len(times) < 2:
        return print(f"Only {len(times)} samples in {path}: nothing was recorded.")
    rate = (len(times) - 1) / (times[-1] - times[0])
    codes, trials = sample_labels(data[:, cfg.MARKER_CH])
    starts = [codes[trials == t][0] for t in range(1, trials.max() + 1)]
    counts = {name: starts.count(code) for code, (name, _) in cfg.MOVES.items()}
    print(f"\n{path}: {len(times)} samples, {times[-1] - times[0]:.0f} s, {rate:.1f} samples/s "
          f"(expect ~{cfg.FS})")
    print(f"{len(starts)} complete trials: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    if not starts:
        print("No trials found: are markers reaching acquire.py (same --port)?")


def main(argv=None):
    args = parse_args(argv)
    path = os.path.join(args.out_dir, f"{args.subject}_b{args.block}.csv")
    if os.path.exists(path) and not args.overwrite:
        sys.exit(f"{path} exists. Use another --block, or --overwrite.")
    if not args.skip_check:
        signal_check(args)

    os.makedirs(args.out_dir, exist_ok=True)
    recorder = Recorder(path)
    recorder.start()
    time.sleep(2.0)
    if recorder.n == 0:
        recorder.stop()
        sys.exit(f"No samples on '{cfg.RAW_STREAM}' after 2 s. Is acquire.py streaming?")

    app = QtWidgets.QApplication(sys.argv[:1])
    cues = Cues(args)
    cues.resize(1000, 700)
    cues.show() if args.windowed else cues.showFullScreen()
    app.exec()

    time.sleep(1.0)  # catch the last samples and marker
    recorder.stop()
    summary(path)


if __name__ == "__main__":
    main()
