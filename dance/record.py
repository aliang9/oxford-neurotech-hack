"""Record one block of cued dance moves to CSV.

    python -m dance.record --subject p01 --block 3                        # sequences (default)
    python -m dance.record --subject p01 --block 3 --protocol discrete    # one move per trial

Needs acquire.py running (it publishes core8_raw and receives the markers).

--protocol sequence (default): moves are chained the way they are danced
live. Each sequence is --seq-len moves back to back, each lasting a random
--segment MIN..MAX seconds, with no break in between:

    break (+) -> "Next: MOVE" -> GO MOVE (beep, code) -> ... "then: MOVE2"
              -> MOVE2 (beep, code) -> ... -> STOP (beep, 99) -> break ...

The last --preview seconds of each move show what comes next, so the
dancer flows into it. Every move appears --reps times per block, never
twice in a row, so the block has transitions between all kinds of moves.

--protocol discrete: the original trials, each a single move from rest:
rest (+) -> "Next: MOVE" -> GO (beep, code) -> --duration s -> STOP (beep, 99).

In both, the dancer keeps repeating the move until the cue changes.

The window shows the cues on the right and, on the left, the electrodes as
acquire.py's scope does: the last 10 s of the 1-40 Hz filtered core8_eeg
stream, +-50 uV per channel (--scale), with a line at every marker sent.
--no-scope shows the cues only.

The markers go to acquire.py over UDP, which puts them in the marker channel
of the same LSL stream as the EEG, so labels and samples are aligned without
any clock bookkeeping. A background thread writes that stream to
data/dance/<subject>_b<block>.csv:  time, Fz..PO8, marker.

Keys: Space = start, Esc = abort.
"""

import argparse
import csv
import html
import os
import random
import socket
import sys
import threading
import time
import types

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from . import config as cfg
from .stream import LSLSource, load_block, segments

NAMES = {name: code for code, (name, _) in cfg.MOVES.items()}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--protocol", choices=["sequence", "discrete"], default="sequence")
    ap.add_argument("--reps", type=int, default=5, help="repetitions per move (default: 5)")
    ap.add_argument("--moves", nargs="+", default=list(NAMES), choices=list(NAMES))
    ap.add_argument("--preview", type=float, default=1.0,
                    help="seconds the next move is shown before it starts (default: 1)")
    # sequence protocol
    ap.add_argument("--seq-len", type=int, default=None,
                    help="moves per sequence (default: the number of moves)")
    ap.add_argument("--segment", type=float, nargs=2, default=(3.0, 6.0), metavar=("MIN", "MAX"),
                    help="seconds per move in a sequence, drawn uniformly (default: 3 6)")
    ap.add_argument("--break", dest="break_s", type=float, default=10.0,
                    help="seconds of rest between sequences (default: 10)")
    # discrete protocol
    ap.add_argument("--duration", type=float, default=4.0,
                    help="discrete: seconds of dancing per trial (default: 4)")
    ap.add_argument("--rest", type=float, nargs=2, default=(2.0, 3.0), metavar=("MIN", "MAX"),
                    help="discrete: seconds of rest between trials (default: 2 3)")

    ap.add_argument("--out-dir", default="data/dance")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--port", type=int, default=cfg.MARKER_PORT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--no-scope", action="store_true", help="hide the electrode signals")
    ap.add_argument("--scale", type=float, default=50.0,
                    help="scope: uV each channel may swing before it is clipped (default: 50)")
    ap.add_argument("--skip-check", action="store_true", help="skip the signal-quality check")
    ap.add_argument("--ignore", nargs="+", default=[], metavar="CH",
                    help="channels the signal check should ignore")
    return ap.parse_args(argv)


def no_repeat_order(codes, reps, rng, max_run):
    """Every code reps times, shuffled, never more than max_run in a row."""
    while True:
        order = list(codes) * reps
        rng.shuffle(order)
        if all(len(set(order[i:i + max_run + 1])) > 1 for i in range(len(order) - max_run)):
            return order


def make_schedule(args, rng):
    """List of runs; a run is [(code, seconds), ...] danced without a break."""
    codes = [NAMES[m] for m in args.moves]
    if args.protocol == "discrete":
        return [[(c, args.duration)] for c in no_repeat_order(codes, args.reps, rng, 2)]
    if len(codes) < 2:
        sys.exit("A sequence needs at least 2 moves.")
    order = greedy_no_adjacent(codes, args.reps, rng)
    n = args.seq_len or len(codes)
    return [[(c, rng.uniform(*args.segment)) for c in order[i:i + n]]
            for i in range(0, len(order), n)]


def greedy_no_adjacent(codes, reps, rng):
    """Every code reps times, no code twice in a row (a "switch" to itself is no switch)."""
    while True:
        left = {c: reps for c in codes}
        order = []
        while any(left.values()):
            options = [c for c, n in left.items() if n and (not order or c != order[-1])]
            if not options:
                break  # only the previous move is left: start over
            c = rng.choices(options, weights=[left[o] for o in options])[0]
            order.append(c)
            left[c] -= 1
        else:
            return order


def cue_html(code, color, prefix="", then=None):
    """Big move name, optionally with the next move in small grey text below."""
    text = html.escape(cfg.MOVES[code][1]).replace("\n", "<br>")
    out = f"<div style='color:{color}'>{prefix}{text}</div>"
    if then is not None:
        nxt = html.escape(cfg.MOVES[then][1].split("\n")[0])
        out += f"<div style='color:#888; font-size:32pt'><br>then: {nxt}</div>"
    return out


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
        self.schedule = make_schedule(args, self.rng)
        self.run_i = self.seg_i = 0
        self.started = self.done = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.RichText)
        self.setFont(QtGui.QFont("Helvetica", 48, QtGui.QFont.Bold))
        n = sum(len(run) for run in self.schedule)
        mins = (sum(s for run in self.schedule for _, s in run)
                + len(self.schedule) * (self.pause_mean() + args.preview + 0.5)) / 60
        k = len(self.schedule)
        what = (f"{k} sequence{'s' * (k != 1)} of up to {max(map(len, self.schedule))} moves"
                if args.protocol == "sequence" else f"{n} moves")
        self.show_text(f"Block {args.block}: {what}, about {mins:.0f} min.<br><br>"
                       "<span style='font-size:32pt'>Keep repeating each move until the cue "
                       "changes" + (", then go straight into the next one."
                                    if args.protocol == "sequence" else ".")
                       + "</span><br><br>Press Space to start.", "white")

    def pause_mean(self):
        return self.args.break_s if self.args.protocol == "sequence" else sum(self.args.rest) / 2

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

    def start(self):
        if not self.started:
            self.started = True
            self.pause()

    def abort(self):
        self.finish("Aborted.")

    @property
    def run(self):
        return self.schedule[self.run_i]

    def pause(self):
        """Rest before the next run (a trial in discrete, a sequence in sequence)."""
        if self.run_i >= len(self.schedule):
            return self.finish("Done, thank you!")
        if self.args.protocol == "sequence":
            self.show_text(f"Break<br><br><span style='font-size:32pt'>sequence "
                           f"{self.run_i + 1}/{len(self.schedule)} next</span>", "#888")
            self.after(self.args.break_s, self.preview)
        else:
            self.show_text("+", "#888")
            self.after(self.rng.uniform(*self.args.rest), self.preview)

    def preview(self):
        if self.args.protocol == "sequence":
            where = f"Sequence {self.run_i + 1}/{len(self.schedule)}, first:"
        else:
            where = f"Next ({self.run_i + 1}/{len(self.schedule)}):"
        self.show_text(cue_html(self.run[0][0], "#888", f"{where}<br><br>"), "#888")
        self.after(self.args.preview, self.go)

    def go(self):
        code, seconds = self.run[self.seg_i]
        self.show_text(cue_html(code, "#3f3", "GO!<br><br>" if self.seg_i == 0 else ""), "#3f3")
        self.send(code)
        QtWidgets.QApplication.beep()
        if self.seg_i + 1 < len(self.run):
            nxt = self.run[self.seg_i + 1][0]
            lead = min(self.args.preview, seconds / 2)
            self.after(seconds - lead, lambda: self.show_text(cue_html(code, "#3f3", then=nxt),
                                                              "#3f3"))
            self.after(seconds, self.next_segment)
        else:
            self.after(seconds, self.stop)

    def next_segment(self):
        self.seg_i += 1
        self.go()

    def stop(self):
        self.send(cfg.TRIAL_END)
        QtWidgets.QApplication.beep()
        self.show_text("STOP", "#f44")
        self.run_i, self.seg_i = self.run_i + 1, 0
        self.after(0.5, self.pause)

    def finish(self, text):
        if self.done:
            return
        self.done = True  # a run cut short by Esc has no end marker, so its last move is dropped
        self.show_text(text, "white")
        QtCore.QTimer.singleShot(1500, QtWidgets.QApplication.quit)


class Scope(pg.PlotWidget):
    """The last few seconds of every electrode, one lane each, like acquire.py's scope."""

    def __init__(self, source, scale, seconds=10.0):
        super().__init__()
        self.source, self.scale, self.seconds = source, scale, seconds
        self.buf = np.zeros((0, cfg.N_EEG))
        self.total = 0          # samples received so far
        self.last_marker = 0
        self.lines = []         # (sample index, InfiniteLine) per marker on screen

        self.setBackground("k")
        self.setFocusPolicy(QtCore.Qt.NoFocus)  # Space and Esc belong to the cues
        self.setMenuEnabled(False)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.setTitle(f"Electrodes, 1-40 Hz, \u00b1{scale:g} \u00b5V", color="#aaa")
        self.getAxis("left").setTicks([[(-2 * i * scale, ch) for i, ch in enumerate(cfg.CHANNELS)]])
        self.setYRange(-(2 * cfg.N_EEG - 1) * scale, scale, padding=0.02)
        self.setXRange(-seconds, 0, padding=0)
        self.setLabel("bottom", "seconds")
        self.curves = [self.plot(pen=pg.mkPen("#ddd", width=1)) for _ in cfg.CHANNELS]

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(40)

    def refresh(self):
        samples, _ = self.source.pull()
        if len(samples):
            self.add_markers(np.rint(samples[:, cfg.MARKER_CH]).astype(int))
            self.buf = np.vstack([self.buf, samples[:, :cfg.N_EEG]])[-int(self.seconds * cfg.FS):]
            self.total += len(samples)
        if len(self.buf) < 2:
            return
        x = self.buf - self.buf.mean(axis=0)
        t = (np.arange(len(x)) - len(x)) / cfg.FS
        for i, curve in enumerate(self.curves):
            curve.setData(t, np.clip(x[:, i], -self.scale, self.scale) - 2 * i * self.scale)
        for line in [ln for k, ln in self.lines if (k - self.total) / cfg.FS < -self.seconds]:
            self.removeItem(line)
        self.lines = [(k, ln) for k, ln in self.lines if ln.scene() is not None]
        for k, line in self.lines:
            line.setValue((k - self.total) / cfg.FS)

    def add_markers(self, m):
        """A vertical line at the onset of each marker pulse, labelled with the move."""
        previous = np.r_[self.last_marker, m[:-1]]
        self.last_marker = m[-1]
        for i in np.flatnonzero((m != 0) & (m != previous)):
            move = m[i] in cfg.MOVES
            label = cfg.MOVES[m[i]][0] if move else "stop" if m[i] == cfg.TRIAL_END else str(m[i])
            line = pg.InfiniteLine(angle=90, label=label,
                                   pen=pg.mkPen("#3f3" if move else "#f44", style=QtCore.Qt.DashLine),
                                   labelOpts={"position": 0.98, "color": "#ccc"})
            self.addItem(line)
            self.lines.append((self.total + i, line))


class Window(QtWidgets.QWidget):
    """Electrodes on the left, cues on the right."""

    def __init__(self, cues, scope=None):
        super().__init__()
        self.setWindowTitle("Dance recording")
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("black"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if scope is not None:
            layout.addWidget(scope, 1)
        layout.addWidget(cues, 1)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=cues.start)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=cues.abort)


def signal_check(args):
    """The same pre-start check as tasks.py, on the filtered stream, sitting still."""
    import tasks
    check_args = types.SimpleNamespace(stream_name=cfg.QUALITY_STREAM, ignore=args.ignore)
    tasks.signal_check(check_args).close()


def summary(path, expected=None):
    data, times = load_block(path)
    if len(times) < 2:
        return print(f"Only {len(times)} samples in {path}: nothing was recorded.")
    rate = (len(times) - 1) / (times[-1] - times[0])
    segs = segments(data[:, cfg.MARKER_CH])
    starts = [code for _, _, code in segs]
    counts = {name: starts.count(code) for code, (name, _) in cfg.MOVES.items()}
    # back-to-back segments are transitions; a gap means a break or a discrete trial
    pairs = {(a[2], b[2]) for a, b in zip(segs, segs[1:]) if a[1] == b[0]}
    print(f"\n{path}: {len(times)} samples, {times[-1] - times[0]:.0f} s, {rate:.1f} samples/s "
          f"(expect ~{cfg.FS})")
    print(f"{len(starts)} complete segments" + (f" of {expected} cued" if expected else "") + ": "
          + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    if pairs:
        n = len(cfg.MOVES)
        print(f"{len(pairs)} of {n * (n - 1)} possible move-to-move transitions covered")
    if not starts:
        print("No segments found: are markers reaching acquire.py (same --port)?")
    elif expected and len(starts) < expected:
        print("Fewer segments than cued: some markers were lost, or the block was aborted.")


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
    scope = None
    if not args.no_scope:
        try:
            scope = Scope(LSLSource(cfg.QUALITY_STREAM, timeout=3.0), args.scale)
        except RuntimeError as e:
            print(f"{e} Showing the cues without the electrode scope.")
    window = Window(cues, scope)
    window.resize(1400, 800)
    window.show() if args.windowed else window.showFullScreen()
    app.exec()

    time.sleep(1.0)  # catch the last samples and marker
    recorder.stop()
    summary(path, expected=sum(len(run) for run in cues.schedule))


if __name__ == "__main__":
    main()
